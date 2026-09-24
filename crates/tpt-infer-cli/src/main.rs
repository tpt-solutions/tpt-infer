//! `tpt-infer` command-line tool: inspect, run, and AOT-compile ONNX models
//! without writing any Rust.
//!
//! ```text
//! tpt-infer inspect model.onnx
//! tpt-infer run model.onnx [--input raw_f32.bin]
//! tpt-infer compile model.onnx -o generated.rs
//! ```

use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use tpt_infer::prelude::*;

#[derive(Parser)]
#[command(
    name = "tpt-infer",
    version,
    about = "Inspect, run, and AOT-compile ONNX models with the tpt-infer runtime"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Load an ONNX model and print a structural summary: node/operator
    /// counts, input/output shapes, and whether the AOT compiler can
    /// handle every operator in the graph.
    Inspect {
        /// Path to the .onnx file.
        model: PathBuf,
    },
    /// Execute an ONNX model with the interpreted runtime and print the
    /// output tensor and its argmax.
    Run {
        /// Path to the .onnx file.
        model: PathBuf,
        /// Raw little-endian f32 input, flattened row-major to match the
        /// model's declared input shape. If omitted, a deterministic
        /// pseudo-random input of the right size is used instead, which is
        /// enough to smoke-test that a model runs end to end.
        #[arg(long)]
        input: Option<PathBuf>,
    },
    /// Ahead-of-time compile an ONNX model to a standalone Rust source file
    /// (a `pub fn execute(input: &[f32]) -> Vec<f32>`).
    Compile {
        /// Path to the .onnx file.
        model: PathBuf,
        /// Where to write the generated Rust source.
        #[arg(short, long)]
        output: PathBuf,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Inspect { model } => inspect(&model),
        Command::Run { model, input } => run(&model, input.as_deref()),
        Command::Compile { model, output } => compile(&model, &output),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn inspect(model: &Path) -> Result<(), Box<dyn Error>> {
    let g = load(model)?;

    println!("nodes:        {}", g.nodes().len());
    println!("initializers: {}", g.initializers().len());
    println!("inputs:       {}", g.inputs().len());
    println!("outputs:      {}", g.outputs().len());

    let mut op_counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for n in g.nodes() {
        *op_counts.entry(n.operator.name()).or_default() += 1;
    }
    println!("\noperator histogram:");
    for (op, count) in &op_counts {
        println!("  {op:<16} {count}");
    }

    println!("\noutputs:");
    for &id in g.outputs() {
        let n = &g.nodes()[id];
        println!(
            "  node {id} ({}): shape {:?}",
            n.name.as_deref().unwrap_or("<unnamed>"),
            n.dims()
        );
    }

    print!("\nAOT compile: ");
    match aot_compile(&g) {
        Ok(_) => println!("supported — every operator in this graph has codegen"),
        Err(e) => println!("not supported yet — {e}"),
    }

    Ok(())
}

/// The single non-initializer graph input the runtime expects data for.
fn find_runtime_input(g: &ComputationGraph) -> Result<&Node, Box<dyn Error>> {
    let initializer_names: HashSet<&str> =
        g.initializers().iter().map(|i| i.name.as_str()).collect();
    g.inputs()
        .iter()
        .map(|&id| &g.nodes()[id])
        .find(|n| match n.name.as_deref() {
            Some(name) => !initializer_names.contains(name),
            None => true,
        })
        .ok_or_else(|| {
            "model has no runtime input (every graph input is a weight initializer)".into()
        })
}

fn run(model: &Path, input_path: Option<&Path>) -> Result<(), Box<dyn Error>> {
    let g = load(model)?;
    let input_node = find_runtime_input(&g)?;
    let numel: usize = input_node.dims().iter().product();

    let input: Vec<f32> = match input_path {
        Some(path) => {
            let raw = std::fs::read(path)?;
            if raw.len() != numel * 4 {
                return Err(format!(
                    "input file '{}' is {} bytes; expected {} ({numel} f32 elements for shape {:?})",
                    path.display(),
                    raw.len(),
                    numel * 4,
                    input_node.dims()
                )
                .into());
            }
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        None => {
            println!(
                "no --input given; using deterministic pseudo-random input of shape {:?}",
                input_node.dims()
            );
            (0..numel)
                .map(|i| ((i as f32 * 12.989_8).sin() * 43_758.547).fract())
                .collect()
        }
    };

    let mut mem = vec![0u8; required_arena_bytes(&g) + 4096];
    let mut arena = BumpArena::new(&mut mem);
    let backend = select_backend();
    let output = execute(&g, &input, &mut arena, &backend)?;
    let out = output.as_slice();

    println!("output shape: {:?}", output.dims());
    let preview: Vec<String> = out.iter().take(10).map(|v| format!("{v:.6}")).collect();
    let ellipsis = if out.len() > preview.len() {
        ", ..."
    } else {
        ""
    };
    println!(
        "output[..{}]: [{}{ellipsis}]",
        preview.len(),
        preview.join(", ")
    );
    println!("argmax: {}", argmax(out));

    Ok(())
}

fn compile(model: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    let g = load(model)?;
    let compiled = aot_compile(&g)?;
    std::fs::write(output, compiled.source())?;
    println!(
        "wrote {} bytes of generated Rust to {}",
        compiled.source().len(),
        output.display()
    );
    Ok(())
}
