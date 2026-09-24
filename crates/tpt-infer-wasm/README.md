# tpt-infer-wasm

**WASM/JS bindings for `tpt-infer`: load and run ONNX models in the browser or Node.js.**

Part of the [tpt-infer](../../README.md) workspace — the Edge AI & Local Inference Runtime.

A thin [`wasm-bindgen`] wrapper exposing `tpt_infer_onnx::load_from_bytes` +
`tpt_infer_runtime::execute` as a `TptInferModel` JS class — no server round-trip, model
inference runs entirely client-side.

## Building

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version <matching the wasm-bindgen dependency version>
cargo build -p tpt-infer-wasm --target wasm32-unknown-unknown --release
wasm-bindgen --target web --out-dir pkg \
  target/wasm32-unknown-unknown/release/tpt_infer_wasm.wasm
```

(Or use [`wasm-pack`](https://rustwasm.github.io/wasm-pack/) — `wasm-pack build
crates/tpt-infer-wasm --target web` — for a packaged, npm-publishable output; this crate
doesn't require it, but `wasm-pack` handles the `wasm-bindgen-cli` version-matching step
above for you.)

**The `wasm-bindgen-cli` version must exactly match the `wasm-bindgen` crate version this
crate resolves to** (see `Cargo.lock`) — a mismatch fails with a schema-version error at
the `wasm-bindgen` CLI step, not at `cargo build`.

## Usage (Node.js)

```js
const fs = require("fs");
const { TptInferModel } = require("./pkg/tpt_infer_wasm.js");

const bytes = new Uint8Array(fs.readFileSync("model.onnx"));
const model = new TptInferModel(bytes);
console.log(`${model.nodeCount()} nodes, ${model.outputCount()} output(s)`);

const output = model.run(new Float32Array(inputLength).fill(0));
console.log(output);
```

## Usage (browser, `--target web`)

```html
<script type="module">
  import init, { TptInferModel } from "./pkg/tpt_infer_wasm.js";

  await init();
  const bytes = new Uint8Array(await (await fetch("model.onnx")).arrayBuffer());
  const model = new TptInferModel(bytes);
  const output = model.run(new Float32Array(inputLength).fill(0));
</script>
```

## What it doesn't do (yet)

- No AOT compilation exposed (only interpreted execution via `execute`) — `aot_compile`
  produces Rust source to be compiled by `rustc`, which has no meaning inside a WASM
  sandbox.
- No image decoding — pass a flattened `Float32Array` (preprocess with
  `tpt-infer-vision` natively, or any JS image-decoding path, before calling `run`).

## Relationship to the rest of the workspace

Depends on `tpt-infer-onnx`, `tpt-infer-graph`, `tpt-infer-ops`, `tpt-infer-runtime`,
and `tpt-infer-core` directly (not through the `tpt-infer` facade, to keep the WASM
binary's dependency surface minimal and explicit). `tpt-infer-ops`'s
`dispatch::select_backend` picks the WASM SIMD backend or the naive scalar backend
depending on what the target `wasm32` binary was compiled with — the same
auto-selection native builds get.

[`wasm-bindgen`]: https://rustwasm.github.io/wasm-bindgen/
