//! WebGPU backend (feature `webgpu`) — real `wgpu` compute-shader dispatch.
//!
//! [`WebGpuBackend::new`] performs the (async, fallible) adapter/device
//! acquisition via `pollster::block_on` and returns `None` when no GPU
//! adapter is available (e.g. a headless CI machine), so callers can skip
//! the backend gracefully instead of panicking. Once constructed, every
//! [`Backend`] method uploads its operands into GPU storage buffers,
//! dispatches a WGSL compute shader, and reads the result back
//! synchronously (via `Device::poll(Maintain::Wait)`), so the trait's
//! synchronous signature is preserved.
//!
//! `matmul`, `elementwise_add`, `relu`, `sigmoid`, `gelu`, and `softmax` are
//! real GPU kernels. [`Backend::conv2d`] still returns
//! [`OpError::Unsupported`]: a correct, reasonably efficient GPU conv2d
//! needs an im2col (or direct tiled) shader, which is a substantially
//! bigger undertaking than the other kernels and was left out to keep the
//! implemented kernels correct and well-tested rather than rushed.

use crate::backend::{
    validate_binary, validate_matmul, validate_softmax, validate_unary, Backend, Conv2dOptions,
    OpError,
};

const UNARY_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_buf: array<f32>;

@compute @workgroup_size(64)
fn relu_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&output_buf)) {
        return;
    }
    output_buf[i] = max(input_buf[i], 0.0);
}

@compute @workgroup_size(64)
fn sigmoid_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&output_buf)) {
        return;
    }
    output_buf[i] = 1.0 / (1.0 + exp(-input_buf[i]));
}

@compute @workgroup_size(64)
fn gelu_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&output_buf)) {
        return;
    }
    let x = input_buf[i];
    let inner = 0.7978846 * (x + 0.044715 * x * x * x);
    output_buf[i] = 0.5 * x * (1.0 + tanh(inner));
}
"#;

const BINARY_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a_buf: array<f32>;
@group(0) @binding(1) var<storage, read> b_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<f32>;

@compute @workgroup_size(64)
fn add_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&out_buf)) {
        return;
    }
    out_buf[i] = a_buf[i] + b_buf[i];
}
"#;

const MATMUL_SHADER: &str = r#"
struct Dims {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> a_buf: array<f32>;
@group(0) @binding(1) var<storage, read> b_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8)
fn matmul_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= dims.m || col >= dims.n) {
        return;
    }
    var acc: f32 = 0.0;
    for (var kk: u32 = 0u; kk < dims.k; kk = kk + 1u) {
        acc = acc + a_buf[row * dims.k + kk] * b_buf[kk * dims.n + col];
    }
    out_buf[row * dims.n + col] = acc;
}
"#;

const SOFTMAX_SHADER: &str = r#"
struct Dims {
    row_len: u32,
    num_rows: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> input_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_buf: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn softmax_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= dims.num_rows) {
        return;
    }
    let start = row * dims.row_len;
    var max_v: f32 = input_buf[start];
    for (var i: u32 = 1u; i < dims.row_len; i = i + 1u) {
        max_v = max(max_v, input_buf[start + i]);
    }
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < dims.row_len; i = i + 1u) {
        let e = exp(input_buf[start + i] - max_v);
        output_buf[start + i] = e;
        sum = sum + e;
    }
    for (var i: u32 = 0u; i < dims.row_len; i = i + 1u) {
        output_buf[start + i] = output_buf[start + i] / sum;
    }
}
"#;

/// Real WebGPU compute backend.
///
/// See the [module documentation](self) for which operators are GPU-backed.
pub struct WebGpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    unary_bgl: wgpu::BindGroupLayout,
    relu_pipeline: wgpu::ComputePipeline,
    sigmoid_pipeline: wgpu::ComputePipeline,
    gelu_pipeline: wgpu::ComputePipeline,
    binary_bgl: wgpu::BindGroupLayout,
    add_pipeline: wgpu::ComputePipeline,
    matmul_bgl: wgpu::BindGroupLayout,
    matmul_pipeline: wgpu::ComputePipeline,
    softmax_bgl: wgpu::BindGroupLayout,
    softmax_pipeline: wgpu::ComputePipeline,
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn make_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    bgl: &wgpu::BindGroupLayout,
    entry_point: &str,
    label: &str,
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

impl WebGpuBackend {
    /// Acquires a GPU adapter/device and compiles all compute pipelines.
    ///
    /// Returns `None` when no adapter is available (e.g. a headless CI
    /// sandbox with no GPU) or device creation fails, rather than
    /// panicking. This blocks on the (otherwise async) adapter/device
    /// request via `pollster`.
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("tpt-infer-ops webgpu device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .ok()?;

        let unary_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("unary"),
            source: wgpu::ShaderSource::Wgsl(UNARY_SHADER.into()),
        });
        let unary_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("unary_bgl"),
            entries: &[storage_entry(0, true), storage_entry(1, false)],
        });
        let relu_pipeline = make_pipeline(&device, &unary_module, &unary_bgl, "relu_main", "relu");
        let sigmoid_pipeline = make_pipeline(
            &device,
            &unary_module,
            &unary_bgl,
            "sigmoid_main",
            "sigmoid",
        );
        let gelu_pipeline = make_pipeline(&device, &unary_module, &unary_bgl, "gelu_main", "gelu");

        let binary_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("binary"),
            source: wgpu::ShaderSource::Wgsl(BINARY_SHADER.into()),
        });
        let binary_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("binary_bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        });
        let add_pipeline = make_pipeline(&device, &binary_module, &binary_bgl, "add_main", "add");

        let matmul_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul"),
            source: wgpu::ShaderSource::Wgsl(MATMUL_SHADER.into()),
        });
        let matmul_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matmul_bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                uniform_entry(3),
            ],
        });
        let matmul_pipeline = make_pipeline(
            &device,
            &matmul_module,
            &matmul_bgl,
            "matmul_main",
            "matmul",
        );

        let softmax_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("softmax"),
            source: wgpu::ShaderSource::Wgsl(SOFTMAX_SHADER.into()),
        });
        let softmax_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("softmax_bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                uniform_entry(2),
            ],
        });
        let softmax_pipeline = make_pipeline(
            &device,
            &softmax_module,
            &softmax_bgl,
            "softmax_main",
            "softmax",
        );

        Some(Self {
            device,
            queue,
            unary_bgl,
            relu_pipeline,
            sigmoid_pipeline,
            gelu_pipeline,
            binary_bgl,
            add_pipeline,
            matmul_bgl,
            matmul_pipeline,
            softmax_bgl,
            softmax_pipeline,
        })
    }

    fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for &v in data {
            bytes.extend_from_slice(&v.to_ne_bytes());
        }
        bytes
    }

    fn u32_to_bytes(data: &[u32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for &v in data {
            bytes.extend_from_slice(&v.to_ne_bytes());
        }
        bytes
    }

    fn upload(&self, data: &[f32], usage: wgpu::BufferUsages, label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &Self::f32_to_bytes(data),
                usage,
            })
    }

    fn upload_uniform(&self, data: &[u32], label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &Self::u32_to_bytes(data),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    fn empty_output(&self, len: usize, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len * 4).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Reads `len` f32s back from a GPU storage buffer via a staging buffer.
    fn read_back(&self, src: &wgpu::Buffer, len: usize) -> Vec<f32> {
        let size = (len * 4).max(4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_staging"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_encoder"),
            });
        encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("map_async callback dropped without a response")
            .expect("failed to map GPU staging buffer for readback");

        let data = slice.get_mapped_range();
        let mut result = vec![0.0f32; len];
        for (i, chunk) in data.chunks_exact(4).take(len).enumerate() {
            result[i] = f32::from_ne_bytes(chunk.try_into().expect("4-byte chunk"));
        }
        drop(data);
        staging.unmap();
        result
    }

    fn dispatch_unary(
        &self,
        pipeline: &wgpu::ComputePipeline,
        a: &[f32],
        out: &mut [f32],
    ) -> Result<(), OpError> {
        validate_unary(a, out)?;
        if a.is_empty() {
            return Ok(());
        }
        let in_buf = self.upload(
            a,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "unary_in",
        );
        let out_buf = self.empty_output(out.len(), "unary_out");
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("unary_bg"),
            layout: &self.unary_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = out.len().div_ceil(64) as u32;
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        let result = self.read_back(&out_buf, out.len());
        out.copy_from_slice(&result);
        Ok(())
    }
}

impl Backend for WebGpuBackend {
    fn name(&self) -> &'static str {
        "webgpu"
    }

    fn matmul(
        &self,
        a: &[f32],
        a_shape: [usize; 2],
        b: &[f32],
        b_shape: [usize; 2],
        out: &mut [f32],
    ) -> Result<(), OpError> {
        let (m, k, n) = validate_matmul(a, a_shape, b, b_shape, out)?;
        if m == 0 || n == 0 {
            return Ok(());
        }
        let a_buf = self.upload(
            a,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "matmul_a",
        );
        let b_buf = self.upload(
            b,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "matmul_b",
        );
        let out_buf = self.empty_output(out.len(), "matmul_out");
        let dims_buf = self.upload_uniform(&[m as u32, k as u32, n as u32, 0u32], "matmul_dims");

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bg"),
            layout: &self.matmul_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.matmul_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let gx = (n as u32).div_ceil(8);
            let gy = (m as u32).div_ceil(8);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        let result = self.read_back(&out_buf, out.len());
        out.copy_from_slice(&result);
        Ok(())
    }

    /// Not yet implemented: a correct GPU conv2d needs an im2col (or
    /// direct/tiled) shader, which is out of scope for this pass. Falls
    /// back to reporting [`OpError::Unsupported`] like the previous stub.
    fn conv2d(
        &self,
        _input: &[f32],
        _in_shape: [usize; 4],
        _weight: &[f32],
        _w_shape: [usize; 4],
        _out: &mut [f32],
        _options: Conv2dOptions,
    ) -> Result<(), OpError> {
        Err(OpError::Unsupported)
    }

    fn elementwise_add(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), OpError> {
        validate_binary(a, b, out)?;
        if a.is_empty() {
            return Ok(());
        }
        let a_buf = self.upload(
            a,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "add_a",
        );
        let b_buf = self.upload(
            b,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "add_b",
        );
        let out_buf = self.empty_output(out.len(), "add_out");
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("add_bg"),
            layout: &self.binary_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.add_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = out.len().div_ceil(64) as u32;
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        let result = self.read_back(&out_buf, out.len());
        out.copy_from_slice(&result);
        Ok(())
    }

    fn relu(&self, a: &[f32], out: &mut [f32]) -> Result<(), OpError> {
        self.dispatch_unary(&self.relu_pipeline, a, out)
    }

    fn softmax(&self, a: &[f32], out: &mut [f32], row_len: usize) -> Result<(), OpError> {
        validate_softmax(a, out, row_len)?;
        if a.is_empty() {
            return Ok(());
        }
        let num_rows = a.len() / row_len;
        let in_buf = self.upload(
            a,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "softmax_in",
        );
        let out_buf = self.empty_output(out.len(), "softmax_out");
        let dims_buf = self.upload_uniform(
            &[row_len as u32, num_rows as u32, 0u32, 0u32],
            "softmax_dims",
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax_bg"),
            layout: &self.softmax_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.softmax_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (num_rows as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        let result = self.read_back(&out_buf, out.len());
        out.copy_from_slice(&result);
        Ok(())
    }

    fn sigmoid(&self, a: &[f32], out: &mut [f32]) -> Result<(), OpError> {
        self.dispatch_unary(&self.sigmoid_pipeline, a, out)
    }

    fn gelu(&self, a: &[f32], out: &mut [f32]) -> Result<(), OpError> {
        self.dispatch_unary(&self.gelu_pipeline, a, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::naive::NaiveBackend;

    /// Tries to construct a real backend; prints and returns `None` when no
    /// GPU adapter is available so tests skip cleanly in headless/CI
    /// environments instead of failing.
    fn try_backend() -> Option<WebGpuBackend> {
        match WebGpuBackend::new() {
            Some(b) => Some(b),
            None => {
                println!("skipping webgpu test: no GPU adapter available");
                None
            }
        }
    }

    fn assert_close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            let tol = 1e-3 * (1.0 + w.abs());
            assert!(g.is_finite(), "index {i}: {g} not finite");
            assert!(
                (g - w).abs() <= tol,
                "index {i}: got {g}, want {w} (tol {tol})"
            );
        }
    }

    fn lcg(len: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn name_is_webgpu() {
        let Some(backend) = try_backend() else {
            return;
        };
        assert_eq!(backend.name(), "webgpu");
    }

    #[test]
    fn matmul_matches_naive() {
        let Some(backend) = try_backend() else {
            return;
        };
        let a = lcg(3 * 4, 1);
        let b = lcg(4 * 5, 2);
        let mut want = vec![0.0f32; 15];
        NaiveBackend::new()
            .matmul(&a, [3, 4], &b, [4, 5], &mut want)
            .unwrap();
        let mut got = vec![0.0f32; 15];
        backend.matmul(&a, [3, 4], &b, [4, 5], &mut got).unwrap();
        assert_close(&got, &want);
    }

    #[test]
    fn elementwise_add_matches_naive() {
        let Some(backend) = try_backend() else {
            return;
        };
        let a = lcg(64, 11);
        let b = lcg(64, 12);
        let mut want = vec![0.0f32; 64];
        NaiveBackend::new()
            .elementwise_add(&a, &b, &mut want)
            .unwrap();
        let mut got = vec![0.0f32; 64];
        backend.elementwise_add(&a, &b, &mut got).unwrap();
        assert_close(&got, &want);
    }

    #[test]
    fn relu_matches_naive() {
        let Some(backend) = try_backend() else {
            return;
        };
        let a = lcg(100, 13);
        let mut want = vec![0.0f32; 100];
        NaiveBackend::new().relu(&a, &mut want).unwrap();
        let mut got = vec![0.0f32; 100];
        backend.relu(&a, &mut got).unwrap();
        assert_close(&got, &want);
    }

    #[test]
    fn sigmoid_matches_naive() {
        let Some(backend) = try_backend() else {
            return;
        };
        let a = lcg(50, 15);
        let mut want = vec![0.0f32; 50];
        NaiveBackend::new().sigmoid(&a, &mut want).unwrap();
        let mut got = vec![0.0f32; 50];
        backend.sigmoid(&a, &mut got).unwrap();
        assert_close(&got, &want);
    }

    #[test]
    fn gelu_matches_naive() {
        let Some(backend) = try_backend() else {
            return;
        };
        let a = lcg(50, 16);
        let mut want = vec![0.0f32; 50];
        NaiveBackend::new().gelu(&a, &mut want).unwrap();
        let mut got = vec![0.0f32; 50];
        backend.gelu(&a, &mut got).unwrap();
        assert_close(&got, &want);
    }

    #[test]
    fn softmax_matches_naive() {
        let Some(backend) = try_backend() else {
            return;
        };
        let a = lcg(2 * 7, 14);
        let mut want = vec![0.0f32; 14];
        NaiveBackend::new().softmax(&a, &mut want, 7).unwrap();
        let mut got = vec![0.0f32; 14];
        backend.softmax(&a, &mut got, 7).unwrap();
        assert_close(&got, &want);
    }

    #[test]
    fn conv2d_reports_unsupported() {
        let Some(backend) = try_backend() else {
            return;
        };
        let a = [1.0f32; 9];
        let w = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        assert_eq!(
            backend
                .conv2d(
                    &a,
                    [1, 1, 3, 3],
                    &w,
                    [1, 1, 2, 2],
                    &mut out,
                    Conv2dOptions::new()
                )
                .unwrap_err(),
            OpError::Unsupported
        );
    }
}
