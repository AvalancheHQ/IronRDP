use wide::i16x8;

/// Max RFX sub-band width. The 8-wide tiling and the `[_; MAX_SUBBAND_WIDTH + 1]` scratch in the
/// inverse passes rely on `subband_width` being one of {8, 16, 32}.
const MAX_SUBBAND_WIDTH: usize = 32;

/// Loads 8 contiguous `i16` from `s[off..]` into a vector. The caller guarantees `off + 8 <= s.len()`.
#[inline]
fn vld(s: &[i16], off: usize) -> i16x8 {
    i16x8::from_slice_unaligned(&s[off..][..8])
}

/// Stores a vector into `s[off..]`.
#[inline]
fn vst(s: &mut [i16], off: usize, v: i16x8) {
    s[off..][..8].copy_from_slice(v.as_array_ref());
}

/// Ceil average `(a + b + 1) >> 1`, overflow-free (SWAR; arithmetic shift).
#[inline]
fn ceil_avg(a: i16x8, b: i16x8) -> i16x8 {
    (a | b) - ((a ^ b) >> 1)
}

/// Floor average `(a + b) >> 1`, overflow-free (SWAR; arithmetic shift).
#[inline]
fn floor_avg(a: i16x8, b: i16x8) -> i16x8 {
    (a & b) + ((a ^ b) >> 1)
}

/// Halved difference `(a - b) >> 1`, overflow-free (SWAR; arithmetic shift).
///
/// `a - b` needs 17 bits, so it cannot be computed in `i16` lanes first. Splitting the shift over
/// both operands keeps every intermediate in range: `floor((a - b) / 2) == (a >> 1) - (b >> 1) - c`
/// where the borrow `c` is `1` when `a` is even and `b` is odd.
#[inline]
fn halved_diff(a: i16x8, b: i16x8) -> i16x8 {
    let one = i16x8::splat(1);
    (a >> 1) - (b >> 1) - (!a & b & one)
}

pub fn encode(buffer: &mut [i16], temp_buffer: &mut [i16]) {
    encode_block::<32>(&mut *buffer, temp_buffer);
    encode_block::<16>(&mut buffer[3072..], temp_buffer);
    encode_block::<8>(&mut buffer[3840..], temp_buffer);
}

fn encode_block<const SUBBAND_WIDTH: usize>(buffer: &mut [i16], temp_buffer: &mut [i16]) {
    dwt_vertical::<SUBBAND_WIDTH>(buffer, temp_buffer);
    dwt_horizontal::<SUBBAND_WIDTH>(buffer, temp_buffer);
}

// DWT in vertical direction, results in 2 sub-bands in L, H order in tmp buffer dwt.
//
// Vectorized over 8 contiguous columns per step (portable `wide`). Bit-exact with the former
// scalar code: the `i32` intermediates of the original never leave the `i16` range once the
// averages and the halved difference are computed with the overflow-free SWAR helpers, and the
// final `L` accumulation is a wrapping `i16` add (identical to i32-add-then-truncate).
fn dwt_vertical<const SUBBAND_WIDTH: usize>(buffer: &[i16], dwt: &mut [i16]) {
    const {
        assert!(
            SUBBAND_WIDTH == 8 || SUBBAND_WIDTH == 16 || SUBBAND_WIDTH == 32,
            "subband width must be one of 8, 16, or 32"
        )
    };
    let sw = SUBBAND_WIDTH;
    let total_width = sw * 2;

    let mut cb = 0;
    while cb < total_width {
        // `h` of the previous row, needed by the `L` accumulation.
        let mut h_prev = i16x8::ZERO;

        for n in 0..sw {
            let src = 2 * n * total_width + cb;

            let top = vld(buffer, src);
            let mid = vld(buffer, src + total_width);
            // The last row has no row below it and mirrors on itself.
            let bottom = if n < sw - 1 {
                vld(buffer, src + 2 * total_width)
            } else {
                top
            };

            let h = halved_diff(mid, floor_avg(top, bottom));
            vst(dwt, (sw + n) * total_width + cb, h);

            let l = top + if n == 0 { h } else { floor_avg(h_prev, h) };
            vst(dwt, n * total_width + cb, l);

            h_prev = h;
        }

        cb += 8;
    }
}

// DWT in horizontal direction, results in 4 sub-bands in HL(0), LH(1), HH(2),
// LL(3) order, stored in original buffer.
// The lower part L generates LL(3) and HL(0).
// The higher part H generates LH(1) and HH(2).
fn dwt_horizontal<const SUBBAND_WIDTH: usize>(buffer: &mut [i16], dwt: &[i16]) {
    let total_width = SUBBAND_WIDTH * 2;
    let squared_subband_width = SUBBAND_WIDTH.pow(2);

    let (hl, rest) = buffer.split_at_mut(squared_subband_width);
    let (lh, rest) = rest.split_at_mut(squared_subband_width);
    let (hh, ll) = rest.split_at_mut(squared_subband_width);
    let (l_src, h_src) = dwt.split_at(squared_subband_width * 2);

    for r in 0..SUBBAND_WIDTH {
        let src_row = r * total_width;
        let dst_row = r * SUBBAND_WIDTH;

        // L generates LL and HL.
        forward_horizontal_band::<SUBBAND_WIDTH>(
            &l_src[src_row..][..total_width],
            &mut hl[dst_row..][..SUBBAND_WIDTH],
            &mut ll[dst_row..][..SUBBAND_WIDTH],
        );
        // H generates LH and HH.
        forward_horizontal_band::<SUBBAND_WIDTH>(
            &h_src[src_row..][..total_width],
            &mut hh[dst_row..][..SUBBAND_WIDTH],
            &mut lh[dst_row..][..SUBBAND_WIDTH],
        );
    }
}

// One band of the forward horizontal pass, vectorized along `n` (`wide`): `src` is the
// interleaved row (len `2*sw`), `high`/`low` the two reconstructed sub-band rows (len `sw`).
// Bit-exact with the former scalar code (SWAR averages + wrapping `i16` arithmetic). `sw` is a
// multiple of 8 and <= MAX_SUBBAND_WIDTH.
fn forward_horizontal_band<const SUBBAND_WIDTH: usize>(src: &[i16], high: &mut [i16], low: &mut [i16]) {
    const {
        assert!(
            SUBBAND_WIDTH == 8 || SUBBAND_WIDTH == 16 || SUBBAND_WIDTH == 32,
            "subband width must be one of 8, 16, or 32"
        )
    };
    let sw = SUBBAND_WIDTH;

    // Split the interleaved row into its even and odd samples so both passes below can load 8
    // neighbours at a time. `even` gets one extra slot mirroring the last sample, which lets the
    // H pass read the right neighbour `even[n + 1]` in bounds (the last coefficient has none).
    let mut even = [0i16; MAX_SUBBAND_WIDTH + 1];
    let mut odd = [0i16; MAX_SUBBAND_WIDTH];
    for n in 0..sw {
        even[n] = src[2 * n];
        odd[n] = src[2 * n + 1];
    }
    even[sw] = even[sw - 1];

    // H: high[n] = (odd[n] - floor_avg(even[n], even[n + 1])) >> 1.
    let mut n = 0;
    while n < sw {
        vst(
            high,
            n,
            halved_diff(vld(&odd, n), floor_avg(vld(&even, n), vld(&even, n + 1))),
        );
        n += 8;
    }

    // Right-shifted copy so `high_pad[n] == high[n - 1]` (and `high[0]` for n == 0): lets the L
    // pass load the left neighbour contiguously instead of shuffling.
    let mut high_pad = [0i16; MAX_SUBBAND_WIDTH];
    high_pad[0] = high[0];
    high_pad[1..sw].copy_from_slice(&high[0..sw - 1]);

    // L: low[n] = even[n] + floor_avg(high[n - 1], high[n]).
    let mut n = 0;
    while n < sw {
        vst(low, n, vld(&even, n) + floor_avg(vld(&high_pad, n), vld(high, n)));
        n += 8;
    }
}

pub fn decode(buffer: &mut [i16], temp_buffer: &mut [i16]) {
    decode_block::<8>(&mut buffer[3840..], temp_buffer);
    decode_block::<16>(&mut buffer[3072..], temp_buffer);
    decode_block::<32>(&mut *buffer, temp_buffer);
}

fn decode_block<const SUBBAND_WIDTH: usize>(buffer: &mut [i16], temp_buffer: &mut [i16]) {
    inverse_horizontal::<SUBBAND_WIDTH>(buffer, temp_buffer);
    inverse_vertical::<SUBBAND_WIDTH>(buffer, temp_buffer);
}

// Inverse DWT horizontal pass (portable `wide`). The 4 sub-bands are stored HL(0), LH(1), HH(2),
// LL(3); the L band reconstructs from LL+HL, the H band from LH+HH. Each row is reconstructed by
// `horizontal_band`.
fn inverse_horizontal<const SUBBAND_WIDTH: usize>(buffer: &[i16], temp_buffer: &mut [i16]) {
    let sw = SUBBAND_WIDTH;
    let tw = sw * 2;
    let ssw = sw * sw;
    let (hl, rest) = buffer.split_at(ssw);
    let (lh, rest) = rest.split_at(ssw);
    let (hh, ll) = rest.split_at(ssw);
    let (l_dst, h_dst) = temp_buffer.split_at_mut(ssw * 2);

    for r in 0..sw {
        let row = r * sw;
        horizontal_band::<SUBBAND_WIDTH>(&ll[row..][..sw], &hl[row..][..sw], &mut l_dst[r * tw..][..tw]);
        horizontal_band::<SUBBAND_WIDTH>(&lh[row..][..sw], &hh[row..][..sw], &mut h_dst[r * tw..][..tw]);
    }
}

// One band of the inverse horizontal pass, vectorized along `n` (`wide`): `low`/`high` are the two
// source subband rows (len `sw`), `dst` the reconstructed row (len `2*sw`, even/odd interleaved).
// Bit-exact with the former scalar code (SWAR averages + wrapping `i16` arithmetic). `sw` is a
// multiple of 8 and ≤ 32.
fn horizontal_band<const SUBBAND_WIDTH: usize>(low: &[i16], high: &[i16], dst: &mut [i16]) {
    const {
        assert!(
            SUBBAND_WIDTH == 8 || SUBBAND_WIDTH == 16 || SUBBAND_WIDTH == 32,
            "subband width must be one of 8, 16, or 32"
        )
    };
    let sw = SUBBAND_WIDTH;

    // Left-shifted copy so `high_pad[n] == high[n-1]` (and `high[0]` for n == 0): lets the even
    // pass load the left neighbour contiguously instead of shuffling.
    let mut high_pad = [0i16; MAX_SUBBAND_WIDTH + 1];
    high_pad[0] = high[0];
    high_pad[1..sw].copy_from_slice(&high[0..sw - 1]);

    // `ev`/`od` padded so the odd pass can read `ev[n+1]` at n = sw-1 in bounds.
    let mut ev = [0i16; MAX_SUBBAND_WIDTH + 1];
    let mut od = [0i16; MAX_SUBBAND_WIDTH + 1];

    // EVEN: ev[n] = low[n] - ceil_avg(high[n-1], high[n]).
    let mut n = 0;
    while n < sw {
        vst(&mut ev, n, vld(low, n) - ceil_avg(vld(&high_pad, n), vld(high, n)));
        n += 8;
    }

    // ODD: od[n] = (high[n] << 1) + floor_avg(ev[n], ev[n+1]).
    let mut n = 0;
    while n < sw {
        vst(
            &mut od,
            n,
            (vld(high, n) << 1) + floor_avg(vld(&ev, n), vld(&ev, n + 1)),
        );
        n += 8;
    }
    // n = sw-1 has no right neighbour.
    od[sw - 1] = i32_to_i16_possible_truncation((i32::from(high[sw - 1]) << 1) + i32::from(ev[sw - 1]));

    // INTERLEAVE: dst[2n] = ev[n], dst[2n+1] = od[n].
    for n in 0..sw {
        dst[2 * n] = ev[n];
        dst[2 * n + 1] = od[n];
    }
}

// Inverse DWT vertical pass, vectorized over 8 contiguous columns per step (portable `wide`).
// Bit-exact with the former scalar code: `(2*x+1)>>1 == x` and `(x+x)>>1 == x` simplify the
// first/last rows, the averages use the overflow-free SWAR `ceil_avg`/`floor_avg`, and every other
// op is wrapping `i16` arithmetic (identical to i32-intermediate-then-truncate).
// Precondition: `subband_width` is a multiple of 8 and <= MAX_SUBBAND_WIDTH (for the 8-wide tiling).
fn inverse_vertical<const SUBBAND_WIDTH: usize>(buffer: &mut [i16], temp_buffer: &[i16]) {
    const {
        assert!(
            SUBBAND_WIDTH == 8 || SUBBAND_WIDTH == 16 || SUBBAND_WIDTH == 32,
            "subband width must be one of 8, 16, or 32"
        )
    };
    let sw = SUBBAND_WIDTH;
    let tw = sw * 2;

    let mut cb = 0;
    while cb < tw {
        // Row 0: L0 - ((H0*2 + 1) >> 1) == L0 - H0.
        vst(buffer, cb, vld(temp_buffer, cb) - vld(temp_buffer, cb + sw * tw));

        for k in 1..sw {
            let l = vld(temp_buffer, cb + k * tw);
            let h = vld(temp_buffer, cb + (sw + k) * tw);
            let lh = vld(temp_buffer, cb + (sw - 1 + k) * tw);

            let even = l - ceil_avg(lh, h);
            vst(buffer, cb + k * 2 * tw, even);

            let d0 = vld(buffer, cb + (k - 1) * 2 * tw);
            vst(buffer, cb + (2 * k - 1) * tw, (lh << 1) + floor_avg(d0, even));
        }

        // Final odd row: (lhN << 1) + ((d0 + d0) >> 1) == (lhN << 1) + d0.
        let lhn = vld(temp_buffer, cb + (2 * sw - 1) * tw);
        let dl = vld(buffer, cb + (2 * sw - 2) * tw);
        vst(buffer, cb + (2 * sw - 1) * tw, (lhn << 1) + dl);

        cb += 8;
    }
}

#[expect(clippy::as_conversions)]
#[expect(clippy::cast_possible_truncation)]
fn i32_to_i16_possible_truncation(value: i32) -> i16 {
    value as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Straightforward scalar transcription of the RFX forward DWT, used as the reference the
    /// vectorized implementation must match bit for bit.
    mod reference {
        use super::i32_to_i16_possible_truncation;

        pub(super) fn encode(buffer: &mut [i16], temp_buffer: &mut [i16]) {
            encode_block::<32>(&mut *buffer, temp_buffer);
            encode_block::<16>(&mut buffer[3072..], temp_buffer);
            encode_block::<8>(&mut buffer[3840..], temp_buffer);
        }

        fn encode_block<const SUBBAND_WIDTH: usize>(buffer: &mut [i16], temp_buffer: &mut [i16]) {
            dwt_vertical::<SUBBAND_WIDTH>(buffer, temp_buffer);
            dwt_horizontal::<SUBBAND_WIDTH>(buffer, temp_buffer);
        }

        fn dwt_vertical<const SUBBAND_WIDTH: usize>(buffer: &[i16], dwt: &mut [i16]) {
            let total_width = SUBBAND_WIDTH * 2;

            for x in 0..total_width {
                for n in 0..SUBBAND_WIDTH {
                    let y = n * 2;
                    let l_index = n * total_width + x;
                    let h_index = l_index + SUBBAND_WIDTH * total_width;
                    let src_index = y * total_width + x;

                    dwt[h_index] = i32_to_i16_possible_truncation(
                        (i32::from(buffer[src_index + total_width])
                            - ((i32::from(buffer[src_index])
                                + i32::from(
                                    buffer[src_index + if n < SUBBAND_WIDTH - 1 { 2 * total_width } else { 0 }],
                                ))
                                >> 1))
                            >> 1,
                    );
                    dwt[l_index] = i32_to_i16_possible_truncation(
                        i32::from(buffer[src_index])
                            + if n == 0 {
                                i32::from(dwt[h_index])
                            } else {
                                (i32::from(dwt[h_index - total_width]) + i32::from(dwt[h_index])) >> 1
                            },
                    );
                }
            }
        }

        fn dwt_horizontal<const SUBBAND_WIDTH: usize>(buffer: &mut [i16], dwt: &[i16]) {
            let total_width = SUBBAND_WIDTH * 2;
            let squared_subband_width = SUBBAND_WIDTH.pow(2);

            let (hl, rest) = buffer.split_at_mut(squared_subband_width);
            let (lh, rest) = rest.split_at_mut(squared_subband_width);
            let (hh, ll) = rest.split_at_mut(squared_subband_width);
            let (l_src, h_src) = dwt.split_at(squared_subband_width * 2);

            for row in 0..SUBBAND_WIDTH {
                let hl = &mut hl[row * SUBBAND_WIDTH..];
                let lh = &mut lh[row * SUBBAND_WIDTH..];
                let hh = &mut hh[row * SUBBAND_WIDTH..];
                let ll = &mut ll[row * SUBBAND_WIDTH..];
                let l_src = &l_src[row * total_width..];
                let h_src = &h_src[row * total_width..];

                for n in 0..SUBBAND_WIDTH {
                    let x = n * 2;

                    hl[n] = i32_to_i16_possible_truncation(
                        (i32::from(l_src[x + 1])
                            - ((i32::from(l_src[x])
                                + i32::from(l_src[if n < SUBBAND_WIDTH - 1 { x + 2 } else { x }]))
                                >> 1))
                            >> 1,
                    );
                    ll[n] = i32_to_i16_possible_truncation(
                        i32::from(l_src[x])
                            + if n == 0 {
                                i32::from(hl[n])
                            } else {
                                (i32::from(hl[n - 1]) + i32::from(hl[n])) >> 1
                            },
                    );
                }

                for n in 0..SUBBAND_WIDTH {
                    let x = n * 2;

                    hh[n] = i32_to_i16_possible_truncation(
                        (i32::from(h_src[x + 1])
                            - ((i32::from(h_src[x])
                                + i32::from(h_src[if n < SUBBAND_WIDTH - 1 { x + 2 } else { x }]))
                                >> 1))
                            >> 1,
                    );
                    lh[n] = i32_to_i16_possible_truncation(
                        i32::from(h_src[x])
                            + if n == 0 {
                                i32::from(hh[n])
                            } else {
                                (i32::from(hh[n - 1]) + i32::from(hh[n])) >> 1
                            },
                    );
                }
            }
        }
    }

    /// Deterministic pseudo-random tile covering the whole `i16` range (including the extremes,
    /// where the intermediate arithmetic wraps).
    fn sample_tile(seed: u64) -> Vec<i16> {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..4096)
            .map(|i| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                match i % 16 {
                    0 => i16::MIN,
                    1 => i16::MAX,
                    _ => {
                        let bits = u16::try_from((state >> 33) & 0xFFFF).expect("16 bits always fit in a u16");
                        i16::from_ne_bytes(bits.to_ne_bytes())
                    }
                }
            })
            .collect()
    }

    #[test]
    fn encode_matches_reference() {
        for seed in 0..8 {
            let input = sample_tile(seed);

            let mut expected = input.clone();
            reference::encode(&mut expected, &mut vec![0i16; 4096]);

            let mut actual = input.clone();
            encode(&mut actual, &mut vec![0i16; 4096]);

            assert_eq!(actual, expected, "mismatch for seed {seed}");
        }
    }
}
