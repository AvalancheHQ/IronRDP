#![expect(clippy::missing_panics_doc, reason = "panics in benches are allowed")]

use core::hint::black_box;
use core::num::{NonZeroU16, NonZeroUsize};

use criterion::{Criterion, criterion_group, criterion_main};
use ironrdp_core::{decode, encode_vec};
use ironrdp_graphics::color_conversion::{YCbCrBuffer, to_64x64_ycbcr_tile, ycbcr_to_rgba};
use ironrdp_graphics::diff::find_different_rects_sub;
use ironrdp_pdu::codecs::rfx;
use ironrdp_pdu::input::MousePdu;
use ironrdp_pdu::input::fast_path::{FastPathInput, FastPathInputEvent, KeyboardFlags};
use ironrdp_pdu::input::mouse::PointerFlags;
use ironrdp_server::BitmapUpdate;
use ironrdp_server::bench::encoder::rfx::{rfx_enc, rfx_enc_tile};

pub fn rfx_enc_tile_bench(c: &mut Criterion) {
    const WIDTH: NonZeroU16 = NonZeroU16::new(64).expect("value is guaranteed to be non-zero");
    const HEIGHT: NonZeroU16 = NonZeroU16::new(64).expect("value is guaranteed to be non-zero");
    const STRIDE: NonZeroUsize = NonZeroUsize::new(64 * 4).expect("value is guaranteed to be non-zero");

    let quant = rfx::Quant::default();
    let algo = rfx::EntropyAlgorithm::Rlgr3;

    let bitmap = BitmapUpdate {
        x: 0,
        y: 0,
        width: WIDTH,
        height: HEIGHT,
        format: ironrdp_server::PixelFormat::ARgb32,
        data: vec![0; 64 * 64 * 4].into(),
        stride: STRIDE,
    };
    c.bench_function("rfx_enc_tile", |b| b.iter(|| rfx_enc_tile(&bitmap, &quant, algo, 0, 0)));
}

pub fn rfx_enc_bench(c: &mut Criterion) {
    const WIDTH: NonZeroU16 = NonZeroU16::new(2048).expect("value is guaranteed to be non-zero");
    const HEIGHT: NonZeroU16 = NonZeroU16::new(2048).expect("value is guaranteed to be non-zero");
    // FIXME/QUESTION: It looks like we have a bug here, don't we? The stride value should be 2048 * 4.
    const STRIDE: NonZeroUsize = NonZeroUsize::new(64 * 4).expect("value is guaranteed to be non-zero");

    let quant = rfx::Quant::default();
    let algo = rfx::EntropyAlgorithm::Rlgr3;

    let bitmap = BitmapUpdate {
        x: 0,
        y: 0,
        width: WIDTH,
        height: HEIGHT,
        format: ironrdp_server::PixelFormat::ARgb32,
        data: vec![0; 2048 * 2048 * 4].into(),
        stride: STRIDE,
    };
    c.bench_function("rfx_enc", |b| b.iter(|| rfx_enc(&bitmap, &quant, algo)));
}

pub fn to_ycbcr_bench(c: &mut Criterion) {
    const WIDTH: usize = 64;
    const HEIGHT: usize = 64;

    let input = vec![0; WIDTH * HEIGHT * 4];
    let stride = WIDTH * 4;
    let mut y = [0i16; WIDTH * HEIGHT];
    let mut cb = [0i16; WIDTH * HEIGHT];
    let mut cr = [0i16; WIDTH * HEIGHT];
    let format = ironrdp_graphics::image_processing::PixelFormat::ARgb32;

    c.bench_function("to_ycbcr", |b| {
        b.iter(|| {
            to_64x64_ycbcr_tile(
                &input,
                WIDTH.try_into().expect("can't panic"),
                HEIGHT.try_into().expect("can't panic"),
                stride.try_into().expect("can't panic"),
                format,
                &mut y,
                &mut cb,
                &mut cr,
            )
        })
    });
}

pub fn ycbcr_to_rgba_bench(c: &mut Criterion) {
    const WIDTH: usize = 64;
    const HEIGHT: usize = 64;

    let y = vec![0i16; WIDTH * HEIGHT];
    let cb = vec![0i16; WIDTH * HEIGHT];
    let cr = vec![0i16; WIDTH * HEIGHT];
    let mut output = vec![0u8; WIDTH * HEIGHT * 4];

    c.bench_function("ycbcr_to_rgba", |b| {
        b.iter(|| {
            let input = YCbCrBuffer {
                y: &y,
                cb: &cb,
                cr: &cr,
            };

            ycbcr_to_rgba(input, &mut output).expect("color conversion should succeed")
        })
    });
}

/// Builds a 1080p ARGB framebuffer with horizontal color bands.
fn framebuffer_1080p() -> Vec<u8> {
    const WIDTH: usize = 1920;
    const HEIGHT: usize = 1080;

    let mut data = vec![0u8; WIDTH * HEIGHT * 4];

    for (idx, chunk) in data.chunks_exact_mut(4).enumerate() {
        let row = idx / WIDTH;
        let value = u8::try_from(row % 256).expect("modulo 256 always fits in a u8");
        chunk.copy_from_slice(&[value, value / 2, 255 - value, 255]);
    }

    data
}

/// Frame differencing is run by the server for every captured frame in order to
/// find out which regions must be re-encoded and sent to the client.
pub fn bitmap_diff_bench(c: &mut Criterion) {
    const WIDTH: usize = 1920;
    const HEIGHT: usize = 1080;
    const STRIDE: usize = WIDTH * 4;

    let previous = framebuffer_1080p();

    // Typical case: only a small part of the screen changed (e.g., a blinking cursor
    // and a small window repaint).
    let mut small_change = previous.clone();
    for row in 400..464 {
        let start = row * STRIDE + 800 * 4;
        small_change[start..start + 64 * 4].fill(0x7f);
    }

    // Worst case: the whole screen changed (e.g., a full screen video is playing).
    let full_change = vec![0x2au8; WIDTH * HEIGHT * 4];

    let mut group = c.benchmark_group("bitmap_diff");

    for (label, current) in [("small_change", &small_change), ("full_change", &full_change)] {
        group.bench_function(label, |b| {
            b.iter(|| {
                black_box(find_different_rects_sub::<4>(
                    &previous, STRIDE, WIDTH, HEIGHT, current, STRIDE, WIDTH, HEIGHT, 0, 0,
                ))
            })
        });
    }

    group.finish();
}

/// A batch of fast-path input events, as sent by a client dragging the mouse
/// while typing.
fn fastpath_input_batch() -> FastPathInput {
    let mut events = Vec::with_capacity(64);

    for i in 0..64u16 {
        if i % 4 == 0 {
            events.push(FastPathInputEvent::KeyboardEvent(
                KeyboardFlags::empty(),
                u8::try_from(i % 256).expect("modulo 256 always fits in a u8"),
            ));
        } else {
            events.push(FastPathInputEvent::MouseEvent(MousePdu {
                flags: PointerFlags::MOVE,
                number_of_wheel_rotation_units: 0,
                x_position: i * 8,
                y_position: i * 4,
            }));
        }
    }

    FastPathInput::new(events).expect("event count is within bounds")
}

pub fn fastpath_input_bench(c: &mut Criterion) {
    let input = fastpath_input_batch();
    let encoded = encode_vec(&input).expect("fast-path input encoding should succeed");

    let mut group = c.benchmark_group("fastpath_input");

    group.bench_function("encode", |b| {
        b.iter(|| black_box(encode_vec(black_box(&input)).expect("fast-path input encoding should succeed")))
    });

    group.bench_function("decode", |b| {
        b.iter(|| {
            black_box(decode::<FastPathInput>(black_box(encoded.as_slice())).expect("fast-path input should decode"))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    rfx_enc_tile_bench,
    rfx_enc_bench,
    to_ycbcr_bench,
    ycbcr_to_rgba_bench,
    bitmap_diff_bench,
    fastpath_input_bench
);
criterion_main!(benches);
