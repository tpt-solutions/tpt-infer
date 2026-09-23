//! End-to-end: decode → preprocess → tiny "infer" via `tpt-infer-ops`.
//!
//! The e2e path synthesizes a solid-color PNG in memory, preprocesses it to
//! an NCHW tensor, then runs a global-average-pool conv on the
//! `NaiveBackend` and checks the channel means.
#![cfg(feature = "std")]

use std::io::Cursor;

use tpt_infer_ops::{Backend, Conv2dOptions, NaiveBackend};
use tpt_infer_vision::{preprocess_image, preprocess_source, PreprocessConfig};

fn solid_png(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
    let img = image::RgbImage::from_pixel(width, height, image::Rgb(rgb));
    let mut png = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
        .unwrap();
    png
}

#[test]
fn preprocess_then_mean_pool_end_to_end() {
    // Solid RGB(255, 128, 0) — every pixel of the NCHW tensor must be
    // [1.0, 128/255, 0.0] after identity normalization, so the global
    // average pool returns the same per channel.
    let png = solid_png(32, 32, [255, 128, 0]);
    let mut cfg = PreprocessConfig::imagenet();
    cfg.width = 64;
    cfg.height = 64;
    cfg.mean = [0.0; 3];
    cfg.std = [1.0; 3];

    let tensor = preprocess_image(&png, &cfg).unwrap();
    assert_eq!(tensor.shape(), [1, 3, 64, 64]);

    // Global average pool as a 1×1-output conv: 3×3×64×64 OIHW weight with
    // 1/(64*64) on the spatial diagonal and 0 cross-channel → [1,3,1,1].
    let spatial = 64 * 64;
    let mut weight = vec![0.0f32; 3 * 3 * spatial];
    for c in 0..3 {
        let base = (c * 3 + c) * spatial;
        for v in &mut weight[base..base + spatial] {
            *v = 1.0 / spatial as f32;
        }
    }
    let mut out = vec![0.0f32; 3];
    NaiveBackend::new()
        .conv2d(
            tensor.as_slice(),
            [1, 3, 64, 64],
            &weight,
            [3, 3, 64, 64],
            &mut out,
            Conv2dOptions::new(),
        )
        .unwrap();

    let expected = [1.0f32, 128.0 / 255.0, 0.0f32];
    for (i, (got, exp)) in out.iter().zip(expected).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "channel {i}: got {got}, expected {exp}"
        );
    }
}

#[test]
fn preprocess_source_image_rgb_image_e2e() {
    // Same pipeline through the ImageSource boundary (tpt-kinetix path).
    let img = image::RgbImage::from_pixel(16, 16, image::Rgb([10u8, 20, 30]));
    let mut cfg = PreprocessConfig::imagenet();
    cfg.width = 4;
    cfg.height = 4;
    cfg.mean = [0.0; 3];
    cfg.std = [1.0; 3];

    let tensor = preprocess_source(&img, &cfg).unwrap();
    assert_eq!(tensor.shape(), [1, 3, 4, 4]);
    let plane = 16;
    let s = tensor.as_slice();
    assert!((s[0] - 10.0 / 255.0).abs() < 1e-5, "R = {}", s[0]);
    assert!((s[plane] - 20.0 / 255.0).abs() < 1e-5, "G = {}", s[plane]);
    assert!((s[2 * plane] - 30.0 / 255.0).abs() < 1e-5, "B = {}", s[2 * plane]);
}
