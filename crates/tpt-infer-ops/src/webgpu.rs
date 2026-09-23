//! WebGPU backend (feature `webgpu`) — currently a stub.
//!
//! The backend type, dispatch plumbing, and the `wgpu` dependency are staged
//! behind the `webgpu` feature, but GPU compute kernels are not implemented
//! yet: every operation returns [`OpError::Unsupported`]. Real kernels will
//! need an async adapter/device setup path, which does not fit the
//! synchronous [`Backend`] trait as-is.

use crate::backend::{Backend, Conv2dOptions, OpError};

/// WebGPU backend stub.
///
/// All operations return [`OpError::Unsupported`]; see the [module
/// documentation](self) for details.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WebGpuBackend;

impl WebGpuBackend {
    /// Creates the stub backend.
    pub const fn new() -> Self {
        Self
    }
}

impl Backend for WebGpuBackend {
    fn name(&self) -> &'static str {
        "webgpu"
    }

    fn matmul(
        &self,
        _a: &[f32],
        _a_shape: [usize; 2],
        _b: &[f32],
        _b_shape: [usize; 2],
        _out: &mut [f32],
    ) -> Result<(), OpError> {
        Err(OpError::Unsupported)
    }

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

    fn elementwise_add(&self, _a: &[f32], _b: &[f32], _out: &mut [f32]) -> Result<(), OpError> {
        Err(OpError::Unsupported)
    }

    fn relu(&self, _a: &[f32], _out: &mut [f32]) -> Result<(), OpError> {
        Err(OpError::Unsupported)
    }

    fn softmax(&self, _a: &[f32], _out: &mut [f32], _row_len: usize) -> Result<(), OpError> {
        Err(OpError::Unsupported)
    }

    fn sigmoid(&self, _a: &[f32], _out: &mut [f32]) -> Result<(), OpError> {
        Err(OpError::Unsupported)
    }

    fn gelu(&self, _a: &[f32], _out: &mut [f32]) -> Result<(), OpError> {
        Err(OpError::Unsupported)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_reports_unsupported() {
        let backend = WebGpuBackend::new();
        assert_eq!(backend.name(), "webgpu");
        let a = [1.0f32];
        let mut out = [0.0f32; 1];
        assert_eq!(
            backend.relu(&a, &mut out).unwrap_err(),
            OpError::Unsupported
        );
        assert_eq!(
            backend
                .matmul(&a, [1, 1], &a, [1, 1], &mut out)
                .unwrap_err(),
            OpError::Unsupported
        );
    }
}
