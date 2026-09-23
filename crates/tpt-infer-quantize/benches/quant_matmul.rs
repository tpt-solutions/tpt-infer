//! Criterion benchmark: INT8 quantized matmul vs f32 matmul throughput.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use tpt_infer_ops::{select_backend, Backend};
use tpt_infer_quantize::{quantized_matmul_i8, quantize_i8_symmetric};

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_1x784_x_784x10");
    let (m, k, n) = (1usize, 784usize, 10usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.001).cos()).collect();

    let backend = select_backend();
    group.bench_function(BenchmarkId::new("f32", backend.name()), |ben| {
        let mut out = vec![0.0f32; m * n];
        ben.iter(|| {
            backend.matmul(&a, [m, k], &b, [k, n], &mut out).unwrap();
            out[0]
        });
    });

    let (aq, as_) = quantize_i8_symmetric(&a);
    let (bq, bs) = quantize_i8_symmetric(&b);
    group.bench_function("i8_quantized", |ben| {
        let mut out = vec![0.0f32; m * n];
        ben.iter(|| {
            quantized_matmul_i8(&aq, [m, k], as_, &bq, [k, n], &[bs], &mut out).unwrap();
            out[0]
        });
    });
    group.finish();
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
