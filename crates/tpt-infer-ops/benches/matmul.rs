//! Criterion benchmarks for the selected CPU backend.
//!
//! Shapes: dense matmul `[1,784] x [784,10]` (MLP head) and conv2d
//! `[1,3,224,224]` input with `[8,3,3,3]` OIHW weights, stride 1, pad 1
//! (output `[1,8,224,224]`). Both run on whatever backend
//! [`select_backend`] picks for this machine.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use tpt_infer_ops::backend::{Backend, Conv2dOptions};
use tpt_infer_ops::dispatch::select_backend;

fn selected() -> impl Backend {
    let backend = select_backend();
    eprintln!("tpt-infer-ops bench backend: {}", backend.name());
    backend
}

fn bench_matmul(c: &mut Criterion) {
    let backend = selected();
    let a = vec![0.5f32; 784];
    let b = vec![0.25f32; 784 * 10];
    let mut out = vec![0.0f32; 10];
    c.bench_function("matmul [1,784]x[784,10]", |ben| {
        ben.iter(|| {
            backend
                .matmul(
                    black_box(a.as_slice()),
                    [1, 784],
                    black_box(b.as_slice()),
                    [784, 10],
                    black_box(out.as_mut_slice()),
                )
                .unwrap();
        })
    });
}

fn bench_conv2d(c: &mut Criterion) {
    let backend = selected();
    let input = vec![0.05f32; 3 * 224 * 224];
    let weight = vec![0.1f32; 8 * 3 * 3 * 3];
    let options = Conv2dOptions::with_stride_padding(1, 1, 1, 1);
    let mut out = vec![0.0f32; 8 * 224 * 224];
    c.bench_function("conv2d [1,3,224,224]x[8,3,3,3]", |ben| {
        ben.iter(|| {
            backend
                .conv2d(
                    black_box(input.as_slice()),
                    [1, 3, 224, 224],
                    black_box(weight.as_slice()),
                    [8, 3, 3, 3],
                    black_box(out.as_mut_slice()),
                    options,
                )
                .unwrap();
        })
    });
}

criterion_group!(benches, bench_matmul, bench_conv2d);
criterion_main!(benches);
