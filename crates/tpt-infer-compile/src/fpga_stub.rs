//! FPGA / photonic mesh instruction emitter — **seed for future
//! `tpt-crucible` integration, not a real implementation.**
//!
//! `tpt-crucible` (a sibling project, not part of this workspace) is
//! expected to target reconfigurable and photonic-mesh accelerators. This
//! module exists only to give that integration a stable extension point in
//! `tpt-infer-compile`'s public API ahead of time, so the eventual real
//! backend doesn't require restructuring [`crate::codegen`] or
//! [`crate::model`]. It performs no real instruction selection, scheduling,
//! or mesh placement — every method here is a placeholder.
//!
//! # Status
//!
//! Deliberately unimplemented beyond returning [`FpgaError::Unimplemented`].
//! Do not use this for anything beyond compiling against the trait shape.

use tpt_infer_graph::ComputationGraph;

/// Error returned by every [`FpgaEmitter`] method (there is currently no
/// success path — see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FpgaError {
    /// This backend is a structural placeholder; no target is implemented yet.
    Unimplemented,
}

impl std::fmt::Display for FpgaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unimplemented => {
                f.write_str("FPGA/photonic mesh emission is not implemented (tpt-crucible seed)")
            }
        }
    }
}

impl std::error::Error for FpgaError {}

/// A placeholder instruction in an eventual FPGA/photonic-mesh instruction
/// stream. Carries only the source node id it would have been derived from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshInstruction {
    /// The [`ComputationGraph`] node id this instruction would represent.
    pub source_node: usize,
}

/// Seed trait for a future ahead-of-time emitter targeting FPGA or photonic
/// compute meshes (`tpt-crucible`), mirroring the role [`crate::codegen`]
/// plays for the scalar-Rust backend.
///
/// Every implementation currently expected to return
/// [`FpgaError::Unimplemented`]; the trait exists so downstream crates can
/// depend on the shape of this API before the real backend lands.
pub trait FpgaEmitter {
    /// Emit a mesh instruction stream for `graph`.
    ///
    /// # Errors
    /// Always returns [`FpgaError::Unimplemented`] until a real backend is
    /// implemented.
    fn emit(&self, graph: &ComputationGraph) -> Result<Vec<MeshInstruction>, FpgaError>;
}

/// The only [`FpgaEmitter`] implementation available today: rejects every
/// graph with [`FpgaError::Unimplemented`].
#[derive(Debug, Default, Clone, Copy)]
pub struct UnimplementedFpgaEmitter;

impl FpgaEmitter for UnimplementedFpgaEmitter {
    fn emit(&self, _graph: &ComputationGraph) -> Result<Vec<MeshInstruction>, FpgaError> {
        Err(FpgaError::Unimplemented)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_always_reports_unimplemented() {
        let g = ComputationGraph::new();
        let err = UnimplementedFpgaEmitter.emit(&g).unwrap_err();
        assert_eq!(err, FpgaError::Unimplemented);
    }
}
