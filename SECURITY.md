# Security Policy

## Trust boundary

`tpt-infer` is designed to load and execute machine-learning models and
images that may come from **untrusted sources**. Two inputs in particular
cross a real trust boundary and are treated as adversarial:

- **ONNX model files** (`tpt-infer-onnx`) — an arbitrary, potentially
  crafted or malformed `.onnx` file is untrusted input. The parser validates
  declared tensor dimensions, initializer sizes, and convolution/pooling
  geometry with checked (overflow-safe) arithmetic before they influence any
  memory allocation, and rejects models that exceed configurable size/opset
  limits (see [`load_from_bytes_with_limits`]).
- **Image bytes** (`tpt-infer-vision`) — arbitrary encoded image bytes
  passed to the preprocessing pipeline are untrusted input, decoded via the
  `image` crate.

Once a model has been loaded into a `ComputationGraph` and its shapes have
passed validation, the SIMD backend kernels (`tpt-infer-ops`) and the
zero-alloc runtime (`tpt-infer-runtime`) assume those shapes are internally
consistent — they are not re-validated against the original untrusted bytes
at every kernel call. If you construct a `ComputationGraph` directly (rather
than via `tpt-infer-onnx::load`), you are responsible for ensuring its node
dimensions and initializer data are consistent.

[`load_from_bytes_with_limits`]: https://docs.rs/tpt-infer-onnx

## Reporting a vulnerability

If you believe you've found a security issue — a way for a malformed model
or image to cause memory unsafety, a panic-based denial of service, or
incorrect output that could be security-relevant — please report it
privately rather than opening a public issue.

Email: **phillip@icb.co.nz**

Please include:
- A description of the issue and its impact.
- A minimal reproduction (a small `.onnx`/image file or the Rust code that
  constructs the problematic input) if possible.
- The affected crate(s) and version(s).

We'll acknowledge reports within a few days and aim to ship a fix or
mitigation before any public disclosure. Please give us a reasonable window
to respond before disclosing publicly.

## Supported versions

This project is pre-1.0 (`0.x`). Security fixes land on the latest published
release; we don't maintain older `0.x` lines separately.

## Dependency scanning

CI runs [`cargo-deny`](https://github.com/EmbarkStudios/cargo-deny) on every
push/PR (see `deny.toml`), checking for known advisories (RUSTSEC), license
compliance, and untrusted dependency sources. `prost`, `image`, and `wgpu`
are the dependencies with the most direct exposure to untrusted input and
are prioritized when triaging advisories.
