//! Error types for the OxiCUDA NVRTC runtime loader.
//!
//! Every fallible operation in this crate returns [`NvrtcError`]. The variants
//! distinguish four fundamentally different failure classes so that callers can
//! branch on *why* an operation did not succeed:
//!
//! * **Environment** — the NVRTC runtime is missing ([`NvrtcError::Unavailable`])
//!   or a *required* entry point is absent from an otherwise-loadable library
//!   ([`NvrtcError::SymbolMissing`]).
//! * **Graceful degradation** — an *optional* entry point (CUBIN retrieval,
//!   name expressions, supported-arch queries) is not present in this NVRTC
//!   version ([`NvrtcError::NotSupported`]). This is never a load failure.
//! * **Runtime** — a driver call returned a non-success status
//!   ([`NvrtcError::Api`]) or CUDA-C compilation failed
//!   ([`NvrtcError::Compilation`], carrying the eagerly-fetched compiler log).
//! * **Input** — an interior NUL byte ([`NvrtcError::NulInInput`]) or invalid
//!   UTF-8 in NVRTC output ([`NvrtcError::InvalidUtf8`]).

/// Errors produced while loading or driving the NVRTC runtime library.
///
/// `Clone` is derived so that the process-wide loader cache (which stores the
/// first load attempt) can hand out owned copies of a load failure to every
/// caller without re-probing the filesystem.
#[derive(Debug, Clone, thiserror::Error)]
pub enum NvrtcError {
    /// The NVRTC shared library could not be `dlopen`'d.
    ///
    /// This is the failure reported on a host without an NVIDIA CUDA
    /// installation. `candidates` lists every library file name that was
    /// attempted, in order, and `last_error` carries the OS-level message from
    /// the final attempt.
    #[error("NVRTC runtime library could not be loaded (tried: {candidates:?}): {last_error}")]
    Unavailable {
        /// Library file names that were attempted, in search order.
        candidates: Vec<String>,
        /// OS-level error description from the last failed attempt.
        last_error: String,
    },

    /// The NVRTC library loaded, but a **required** symbol was absent.
    ///
    /// Unlike [`NotSupported`](Self::NotSupported), this indicates a broken or
    /// unexpectedly-old library that is missing a symbol the crate depends on
    /// unconditionally.
    #[error("required NVRTC symbol '{symbol}' is missing: {reason}")]
    SymbolMissing {
        /// Name of the missing symbol (e.g. `"nvrtcCreateProgram"`).
        symbol: &'static str,
        /// OS-level error description from the symbol lookup.
        reason: String,
    },

    /// An **optional** NVRTC entry point is not present in the loaded runtime.
    ///
    /// Returned at call time (never at load time) when a feature such as CUBIN
    /// retrieval, name expressions, or the supported-architecture query is
    /// requested against an NVRTC version that predates it.
    #[error("NVRTC feature unavailable: symbol '{symbol}' is not present in this NVRTC runtime")]
    NotSupported {
        /// Name of the optional symbol that would be required (e.g.
        /// `"nvrtcGetCUBIN"`).
        symbol: &'static str,
    },

    /// A raw NVRTC entry point returned a non-success status code.
    ///
    /// `msg` is resolved via `nvrtcGetErrorString` when that optional symbol is
    /// available, falling back to the numeric code otherwise.
    #[error("NVRTC call '{call}' failed (code {code}): {msg}")]
    Api {
        /// Name of the NVRTC C entry point that failed.
        call: &'static str,
        /// Raw `nvrtcResult` status code.
        code: i32,
        /// Human-readable description of the status code.
        msg: String,
    },

    /// CUDA-C compilation failed.
    ///
    /// The full compiler diagnostic `log` is fetched eagerly (before the error
    /// is returned) so that callers see exactly what `nvrtc` reported.
    #[error("NVRTC compilation failed (code {code}): {msg}\n{log}")]
    Compilation {
        /// Raw `nvrtcResult` status code from `nvrtcCompileProgram`.
        code: i32,
        /// Human-readable description of the status code.
        msg: String,
        /// The complete compiler diagnostic log.
        log: String,
    },

    /// An input string contained an interior NUL byte and cannot be passed to
    /// the C ABI.
    ///
    /// This is detected in Rust *before* any FFI call is issued.
    #[error("interior NUL byte in {what} at byte {position}")]
    NulInInput {
        /// Which input contained the NUL (e.g. `"source"`, `"kernel name"`).
        what: &'static str,
        /// Byte offset of the first interior NUL.
        position: usize,
    },

    /// NVRTC output (PTX text, compiler log, or a lowered name) was not valid
    /// UTF-8.
    #[error("{what} returned by NVRTC is not valid UTF-8: {reason}")]
    InvalidUtf8 {
        /// Which output failed to decode (e.g. `"PTX"`, `"program log"`).
        what: &'static str,
        /// Description of the UTF-8 decoding error.
        reason: String,
    },
}
