# Contributing

This project does not accept pull requests. Development is done by the
maintainer(s) directly.

If you find a bug, have a feature request, or hit a compatibility issue with
a real ONNX model, please **open a GitHub issue** — that's the only
contribution channel:

- Bug reports: include the crate/version, a minimal reproduction, and (if
  relevant) the `.onnx` model's operator set / a way to reproduce the
  structure without sharing proprietary weights.
- Compatibility gaps: if `tpt-infer-cli inspect your-model.onnx` reports an
  unsupported operator or a load/shape-inference failure, include that
  output — it's the fastest way to triage.
- Feature requests: describe the use case, not just the API you'd want.

For **security vulnerabilities**, do not open a public issue — see
[`SECURITY.md`](SECURITY.md) instead.

## Notes for anyone reading the source

- Any `usize` arithmetic derived from parsed/untrusted model or image data
  (declared tensor dims, kernel/pad/stride attributes, buffer sizes) must
  use checked arithmetic (`checked_add`/`checked_sub`/`checked_mul`/
  `try_fold`), never a plain operator or `.iter().product()` — see
  `crates/tpt-infer-onnx/src/shapes.rs` and `crates/tpt-infer-graph/src/graph.rs`
  for the pattern. This is a real trust boundary, not defensive-programming
  theater; see `SECURITY.md`.
- `crates/tpt-infer-compile/src/fpga_stub.rs` is an intentional, permanent
  placeholder seeding a future sibling project (`tpt-crucible`). It is not a
  missing feature to complete.
- The workspace's `rust-version` is `1.75` — avoid APIs stabilized after
  that (e.g. `Option::is_none_or`, stabilized in 1.82).
