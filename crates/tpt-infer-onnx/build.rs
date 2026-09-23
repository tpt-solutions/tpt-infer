//! Compile `proto/onnx.proto3` into Rust structs via `prost-build`.
//!
//! Generated code lands in `$OUT_DIR/onnx.rs` and is included from `src/lib.rs`.

use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo manifest dir"));
    let proto_dir = manifest_dir.join("proto");
    let proto_file = proto_dir.join("onnx.proto3");

    println!("cargo:rerun-if-changed={}", proto_file.display());

    let mut config = prost_build::Config::new();
    if let Err(err) = config.compile_protos(&[&proto_file], &[&proto_dir]) {
        panic!("tpt-infer-onnx: failed to compile onnx.proto3: {err}");
    }
}
