//! `compile_ptx` — JIT-compile a tiny SAXPY kernel from CUDA-C to PTX.
//!
//! On a host **with** the NVRTC runtime this prints the NVRTC version and the
//! head of the generated PTX. On a host **without** it, the example prints a
//! friendly message and exits successfully (no error exit code) — demonstrating
//! the crate's graceful-degradation contract.
//!
//! Run with:
//! ```sh
//! cargo run --example compile_ptx -p oxicuda-nvrtc
//! ```

use oxicuda_nvrtc::{Program, is_available, version};

const SAXPY: &str = r#"
extern "C" __global__ void saxpy(float a, float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
"#;

fn main() {
    println!("── OxiCUDA NVRTC — compile_ptx ─────────────────────────────────");

    if !is_available() {
        println!(
            "NVRTC runtime not found on this host — nothing to compile.\n\
             (Install the CUDA toolkit's `libnvrtc` to see real PTX output.)\n\
             This is the expected, non-fatal outcome on a CUDA-less machine."
        );
        return;
    }

    match version() {
        Ok(v) => println!("NVRTC version: {}.{}", v.major, v.minor),
        Err(e) => println!("Could not query NVRTC version: {e}"),
    }

    let mut program = match Program::new(SAXPY, "saxpy.cu") {
        Ok(p) => p,
        Err(e) => {
            println!("Failed to create NVRTC program: {e}");
            return;
        }
    };

    // Compile with default options; NVRTC targets a default virtual arch.
    if let Err(e) = program.compile(&[]) {
        println!("Compilation failed: {e}");
        return;
    }

    // Surface any warnings the compiler emitted.
    match program.log() {
        Ok(log) if !log.trim().is_empty() => println!("Compiler log:\n{log}"),
        Ok(_) => {}
        Err(e) => println!("Could not read compiler log: {e}"),
    }

    match program.ptx() {
        Ok(ptx) => {
            let text = ptx.as_str();
            println!("Generated {} bytes of PTX. Head:", text.len());
            for line in text.lines().take(12) {
                println!("  {line}");
            }
        }
        Err(e) => println!("Could not read PTX: {e}"),
    }
}
