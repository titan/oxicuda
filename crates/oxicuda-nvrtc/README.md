# oxicuda-nvrtc

Dynamic, safe Rust bindings for NVIDIA's NVRTC — the CUDA-C runtime JIT
compiler (CUDA-C source → PTX).

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-nvrtc` is a pure Rust wrapper around NVRTC (`nvrtc.h`). Like its
sibling [`oxicuda-driver`](https://crates.io/crates/oxicuda-driver), it loads
the NVRTC shared library entirely at **runtime** via
[`libloading`](https://crates.io/crates/libloading). There is **no `#[link]`
attribute, no `build.rs`, and no `-lnvrtc`** — the crate compiles on any
standard Rust toolchain with no CUDA Toolkit, headers, or link stubs present.

The NVRTC library is discovered the first time you call into the crate. A
process-wide `OnceLock` caches the resolved function table for the lifetime of
the process, so subsequent calls are essentially free.

## Graceful degradation

On a host **without** NVRTC nothing panics. `is_available()` returns `false`,
and every fallible entry point returns a typed `NvrtcError::Unavailable` that
names the library files that were tried. NVRTC therefore behaves as an optional
accelerator: probe once, then fall back to a CPU path.

Individual features degrade independently, too. Optional NVRTC entry points that
only exist in newer versions — CUBIN retrieval, C++ name expressions, and the
supported-architecture query — return `NvrtcError::NotSupported` when the
underlying symbol is absent, rather than failing the whole library load.

## Modules

| Module     | Description                                                       |
|------------|-------------------------------------------------------------------|
| `error`    | `NvrtcError` — the single error type for every fallible operation |
| `loader`   | Runtime library loading, the API table, and top-level queries     |
| `program`  | `Program` (RAII compilation unit), `Header`, `compile_to_ptx`     |
| `ptx`      | `Ptx` — owned, NUL-terminated PTX output                          |

## Quick Start

```rust,no_run
use oxicuda_nvrtc::{compile_to_ptx, is_available};

if is_available() {
    let src = r#"
        extern "C" __global__ void saxpy(float a, float* x, float* y, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) y[i] = a * x[i] + y[i];
        }
    "#;
    let ptx = compile_to_ptx(src, "saxpy.cu", &["--gpu-architecture=compute_75"])?;

    // `ptx.as_str()` feeds `oxicuda_driver::Module::from_ptx(&str)` directly.
    println!("{}", ptx.as_str());
}
# Ok::<(), oxicuda_nvrtc::NvrtcError>(())
```

For finer control — in-memory headers, C++ name expressions, CUBIN output, or
inspecting the compiler log — drive a `Program` directly:

```rust,no_run
use oxicuda_nvrtc::Program;

# fn run() -> Result<(), oxicuda_nvrtc::NvrtcError> {
let mut program = Program::new("/* CUDA-C */", "kernel.cu")?;
program.compile(&["--gpu-architecture=compute_86"])?;
let ptx = program.ptx()?;
let log = program.log()?;
# let _ = (ptx, log);
# Ok(())
# }
```

## Runtime Library Resolution

| Platform | Library names searched (in order)                                  |
|----------|--------------------------------------------------------------------|
| Linux    | `libnvrtc.so`, `libnvrtc.so.13`, `libnvrtc.so.12`, `libnvrtc.so.11` |
| Windows  | `nvrtc64_130_0.dll` … `nvrtc64_101_0.dll`                           |
| other    | *(unavailable — `is_available()` returns `false`)*                  |

## Platform Support

| Platform | Status                                             |
|----------|----------------------------------------------------|
| Linux    | Full support (CUDA-installed `libnvrtc`)           |
| Windows  | Full support (CUDA-installed `nvrtc64_*.dll`)      |
| macOS    | Compile only (`is_available()` returns `false`)    |

## License

Apache-2.0 — (C) 2026 COOLJAPAN OU (Team KitaSan)
