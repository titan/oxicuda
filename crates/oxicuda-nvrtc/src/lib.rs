//! # OxiCUDA NVRTC
//!
//! **Dynamic, zero-SDK-dependency Rust bindings for NVIDIA's NVRTC — the CUDA-C
//! runtime JIT compiler (CUDA-C source → PTX).**
//!
//! `oxicuda-nvrtc` loads the NVRTC shared library (`libnvrtc.so` on Linux,
//! `nvrtc64_*.dll` on Windows) entirely at **runtime** via
//! [`libloading`](https://crates.io/crates/libloading). Just like its sibling
//! [`oxicuda-driver`](https://crates.io/crates/oxicuda-driver), there is **no
//! `#[link]` attribute, no `build.rs`, and no `-lnvrtc`** — the crate compiles
//! on any standard Rust toolchain with no CUDA Toolkit, headers, or link stubs
//! present. The real NVRTC library is discovered the first time you call into
//! the crate.
//!
//! ## Graceful degradation
//!
//! On a host **without** NVRTC (no CUDA install, wrong platform, etc.) nothing
//! panics. [`is_available`] returns `false`, and every fallible entry point
//! returns a typed [`NvrtcError::Unavailable`] carrying the library names that
//! were tried. This makes NVRTC an *optional accelerator*: callers can probe
//! once with [`is_available`] and fall back to a CPU path.
//!
//! Individual NVRTC features also degrade independently. Optional entry points
//! that only exist in newer NVRTC versions — CUBIN retrieval
//! ([`Program::cubin`]), C++ name expressions
//! ([`Program::add_name_expression`] / [`Program::lowered_name`]), and the
//! supported-architecture query ([`supported_archs`]) — return
//! [`NvrtcError::NotSupported`] when the underlying symbol is absent, rather
//! than failing the whole library load.
//!
//! ## Runtime library resolution
//!
//! | Platform | Library names tried (in order)                                     |
//! |----------|--------------------------------------------------------------------|
//! | Linux    | `libnvrtc.so`, `libnvrtc.so.13`, `libnvrtc.so.12`, `libnvrtc.so.11` |
//! | Windows  | `nvrtc64_130_0.dll` … `nvrtc64_101_0.dll`                           |
//! | other    | *(unavailable — [`is_available`] returns `false`)*                  |
//!
//! The resolved function table is cached process-wide, so the (relatively
//! expensive) `dlopen` + symbol resolution happens at most once.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use oxicuda_nvrtc::{compile_to_ptx, is_available};
//!
//! if is_available() {
//!     let src = r#"
//!         extern "C" __global__ void saxpy(float a, float* x, float* y, int n) {
//!             int i = blockIdx.x * blockDim.x + threadIdx.x;
//!             if (i < n) y[i] = a * x[i] + y[i];
//!         }
//!     "#;
//!     let ptx = compile_to_ptx(src, "saxpy.cu", &["--gpu-architecture=compute_75"])?;
//!     // `ptx.as_str()` feeds `oxicuda_driver::Module::from_ptx(&str)` directly.
//!     println!("{}", ptx.as_str());
//! }
//! # Ok::<(), oxicuda_nvrtc::NvrtcError>(())
//! ```
//!
//! For finer control — headers, name expressions, CUBIN output, or inspecting
//! the compiler log — drive a [`Program`] directly.
//!
//! (C) 2026 COOLJAPAN OU (Team KitaSan)

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]

pub mod error;
pub mod loader;
pub mod program;
pub mod ptx;

// ---------------------------------------------------------------------------
// Public re-exports — the frozen crate surface
// ---------------------------------------------------------------------------

pub use error::NvrtcError;
pub use loader::{NvrtcVersion, is_available, supported_archs, version};
pub use program::{Header, Program, compile_to_ptx};
pub use ptx::Ptx;

// ---------------------------------------------------------------------------
// Compile-time feature flags
// ---------------------------------------------------------------------------

/// Compile-time feature availability.
pub mod features {
    /// Whether this crate was built (always `true`; the NVRTC runtime itself is
    /// resolved dynamically — use [`crate::is_available`] for that).
    pub const HAS_NVRTC_LOADER: bool = true;
}
