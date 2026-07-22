//! Dynamic NVRTC library loader.
//!
//! This module locates and loads the NVRTC shared library (`libnvrtc.so` on
//! Linux, `nvrtc64_*.dll` on Windows) **at runtime** via [`libloading`], so
//! that no CUDA SDK — not even its headers or link stubs — is required at build
//! time. There is no `#[link]` attribute, no `build.rs`, and no `-lnvrtc`.
//!
//! # Platform support
//!
//! | Platform | Library names tried (in order)                                                            |
//! |----------|-------------------------------------------------------------------------------------------|
//! | Linux    | `libnvrtc.so`, `libnvrtc.so.13`, `libnvrtc.so.12`, `libnvrtc.so.11`                        |
//! | Windows  | `nvrtc64_130_0.dll` … `nvrtc64_101_0.dll`                                                  |
//! | other    | *(no candidates — NVRTC is unavailable, degrading gracefully)*                             |
//!
//! # Caching
//!
//! The resolved `NvrtcApi` function table is stored in a process-wide
//! [`OnceLock`]. The (relatively expensive) `dlopen` + symbol resolution runs
//! at most once; every subsequent access is a single atomic load. The embedded
//! [`Library`] handle keeps the shared object mapped for the lifetime of the
//! process.
//!
//! Application code does not interact with the `NvrtcApi` table directly. It calls the
//! high-level [`crate::Program`] API, or the top-level [`is_available`],
//! [`version`], and [`supported_archs`] helpers defined here.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::OnceLock;

use libloading::Library;

use crate::error::NvrtcError;

// ---------------------------------------------------------------------------
// Raw NVRTC C ABI
// ---------------------------------------------------------------------------

/// NVRTC status code (`nvrtcResult`). `0` is `NVRTC_SUCCESS`.
pub(crate) type NvrtcResult = i32;

/// The `NVRTC_SUCCESS` status code.
pub(crate) const NVRTC_SUCCESS: NvrtcResult = 0;

/// Opaque NVRTC program handle (`nvrtcProgram`).
pub(crate) type NvrtcProgram = *mut c_void;

// Required entry points ------------------------------------------------------

/// `nvrtcVersion(int *major, int *minor) -> nvrtcResult`
type FnVersion = unsafe extern "C" fn(*mut c_int, *mut c_int) -> NvrtcResult;

/// `nvrtcCreateProgram(prog*, src, name, numHeaders, headers**, includeNames**)`
type FnCreateProgram = unsafe extern "C" fn(
    *mut NvrtcProgram,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> NvrtcResult;

/// `nvrtcDestroyProgram(prog*) -> nvrtcResult`
type FnDestroyProgram = unsafe extern "C" fn(*mut NvrtcProgram) -> NvrtcResult;

/// `nvrtcCompileProgram(prog, numOptions, options**) -> nvrtcResult`
type FnCompileProgram =
    unsafe extern "C" fn(NvrtcProgram, c_int, *const *const c_char) -> NvrtcResult;

/// `nvrtcGetPTXSize` / `nvrtcGetProgramLogSize` / `nvrtcGetCUBINSize`
/// `(prog, size_t *out) -> nvrtcResult`
type FnGetSize = unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult;

/// `nvrtcGetPTX` / `nvrtcGetProgramLog` / `nvrtcGetCUBIN`
/// `(prog, char *out) -> nvrtcResult`
type FnGetText = unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult;

// Optional entry points ------------------------------------------------------

/// `nvrtcGetErrorString(nvrtcResult) -> const char*`
type FnGetErrorString = unsafe extern "C" fn(NvrtcResult) -> *const c_char;

/// `nvrtcAddNameExpression(prog, const char *name_expression) -> nvrtcResult`
type FnAddNameExpression = unsafe extern "C" fn(NvrtcProgram, *const c_char) -> NvrtcResult;

/// `nvrtcGetLoweredName(prog, const char *name_expression, const char **lowered) -> nvrtcResult`
type FnGetLoweredName =
    unsafe extern "C" fn(NvrtcProgram, *const c_char, *mut *const c_char) -> NvrtcResult;

/// `nvrtcGetNumSupportedArchs(int *numArchs) -> nvrtcResult`
type FnGetNumSupportedArchs = unsafe extern "C" fn(*mut c_int) -> NvrtcResult;

/// `nvrtcGetSupportedArchs(int *supportedArchs) -> nvrtcResult`
type FnGetSupportedArchs = unsafe extern "C" fn(*mut c_int) -> NvrtcResult;

// ---------------------------------------------------------------------------
// NvrtcApi — resolved function table
// ---------------------------------------------------------------------------

/// Resolved NVRTC C ABI entry points plus the resident library handle.
///
/// Required entry points are plain function pointers (their absence fails the
/// load). Optional entry points are `Option<fn>` — their absence degrades
/// gracefully to [`NvrtcError::NotSupported`] at the call site rather than
/// failing the load.
pub(crate) struct NvrtcApi {
    /// Keeping the loaded library inside the struct guarantees the mapping
    /// outlives every resolved function pointer.
    _lib: Library,

    // -- Required -----------------------------------------------------------
    pub(crate) version: FnVersion,
    pub(crate) create_program: FnCreateProgram,
    pub(crate) destroy_program: FnDestroyProgram,
    pub(crate) compile_program: FnCompileProgram,
    pub(crate) get_ptx_size: FnGetSize,
    pub(crate) get_ptx: FnGetText,
    pub(crate) get_program_log_size: FnGetSize,
    pub(crate) get_program_log: FnGetText,

    // -- Optional -----------------------------------------------------------
    pub(crate) get_error_string: Option<FnGetErrorString>,
    pub(crate) get_cubin_size: Option<FnGetSize>,
    pub(crate) get_cubin: Option<FnGetText>,
    pub(crate) add_name_expression: Option<FnAddNameExpression>,
    pub(crate) get_lowered_name: Option<FnGetLoweredName>,
    pub(crate) get_num_supported_archs: Option<FnGetNumSupportedArchs>,
    pub(crate) get_supported_archs: Option<FnGetSupportedArchs>,
}

// SAFETY: `NvrtcApi` holds raw C function pointers (which are `Copy`, `Send`,
// and `Sync`) plus a `libloading::Library` (itself `Send + Sync`). The NVRTC
// entry points are re-entrant and have no thread affinity; keeping `_lib`
// resident means the pointers never dangle. This mirrors the resident,
// thread-shared function table used by the sibling `oxicuda-driver` loader and
// the in-repo scirs2 NVRTC bridge this crate generalises.
unsafe impl Send for NvrtcApi {}
unsafe impl Sync for NvrtcApi {}

impl NvrtcApi {
    /// Translate a raw `nvrtcResult` code into a human-readable string.
    ///
    /// Uses `nvrtcGetErrorString` when that optional symbol is present,
    /// otherwise falls back to the numeric code.
    pub(crate) fn error_string(&self, code: NvrtcResult) -> String {
        if let Some(f) = self.get_error_string {
            // SAFETY: `f` is the resolved `nvrtcGetErrorString`; it accepts any
            // status code and returns a pointer to a static NUL-terminated C
            // string (or null).
            let ptr = unsafe { f(code) };
            if !ptr.is_null() {
                // SAFETY: `ptr` is non-null and points to a NUL-terminated C
                // string owned by the NVRTC library (static storage).
                let text = unsafe { CStr::from_ptr(ptr) };
                return text.to_string_lossy().into_owned();
            }
        }
        format!("NVRTC error code {code}")
    }
}

// ---------------------------------------------------------------------------
// Library resolution
// ---------------------------------------------------------------------------

/// Library file names to try, in order, for the current platform.
pub(crate) fn library_candidates() -> &'static [&'static str] {
    #[cfg(target_os = "linux")]
    {
        &[
            "libnvrtc.so",
            "libnvrtc.so.13",
            "libnvrtc.so.12",
            "libnvrtc.so.11",
        ]
    }
    #[cfg(target_os = "windows")]
    {
        &[
            "nvrtc64_130_0.dll",
            "nvrtc64_120_0.dll",
            "nvrtc64_112_0.dll",
            "nvrtc64_111_0.dll",
            "nvrtc64_110_0.dll",
            "nvrtc64_101_0.dll",
        ]
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        &[]
    }
}

/// Try each candidate library name in order, returning the first that loads.
///
/// # Errors
///
/// Returns [`NvrtcError::Unavailable`] if **all** candidates fail to load
/// (including the case where the platform defines no candidates), capturing the
/// names tried and the last OS-level error message.
fn open_library(names: &[&str]) -> Result<Library, NvrtcError> {
    let mut last_error = String::new();
    for name in names {
        // SAFETY: opening a vendor-provided shared library runs its
        // initialisation routines; the NVRTC runtime is designed for this. A
        // missing or broken library yields `Err`, which we record and skip.
        match unsafe { Library::new(*name) } {
            Ok(lib) => {
                tracing::debug!("loaded NVRTC library: {name}");
                return Ok(lib);
            }
            Err(e) => {
                tracing::debug!("failed to load NVRTC library {name}: {e}");
                last_error = e.to_string();
            }
        }
    }
    Err(NvrtcError::Unavailable {
        candidates: names.iter().map(|s| (*s).to_string()).collect(),
        last_error: if last_error.is_empty() {
            "no NVRTC library candidates are defined for this platform".to_string()
        } else {
            last_error
        },
    })
}

/// Load the NVRTC runtime and resolve every entry point.
///
/// Required symbols failing to resolve produces [`NvrtcError::SymbolMissing`];
/// optional symbols simply resolve to `None`. Never panics.
fn load() -> Result<NvrtcApi, NvrtcError> {
    let lib = open_library(library_candidates())?;

    // Resolve a REQUIRED symbol, returning `SymbolMissing` on absence.
    //
    // SAFETY (applies to every `required!` / `optional!` expansion below): each
    // symbol name matches its documented NVRTC C ABI export and is bound to a
    // function-pointer type whose signature matches that ABI. `Symbol::deref`
    // copies out a `'static` function pointer, so `lib` is free to move into the
    // returned struct afterwards without dangling.
    macro_rules! required {
        ($name:literal, $ty:ty) => {{
            let sym: libloading::Symbol<$ty> = unsafe { lib.get(concat!($name, "\0").as_bytes()) }
                .map_err(|e| NvrtcError::SymbolMissing {
                    symbol: $name,
                    reason: e.to_string(),
                })?;
            *sym
        }};
    }

    // Resolve an OPTIONAL symbol, returning `None` (never an error) on absence.
    macro_rules! optional {
        ($name:literal, $ty:ty) => {{
            match unsafe { lib.get::<$ty>(concat!($name, "\0").as_bytes()) } {
                Ok(sym) => Some(*sym),
                Err(_) => {
                    tracing::debug!(concat!("optional NVRTC symbol not found: ", $name));
                    None
                }
            }
        }};
    }

    let version = required!("nvrtcVersion", FnVersion);
    let create_program = required!("nvrtcCreateProgram", FnCreateProgram);
    let destroy_program = required!("nvrtcDestroyProgram", FnDestroyProgram);
    let compile_program = required!("nvrtcCompileProgram", FnCompileProgram);
    let get_ptx_size = required!("nvrtcGetPTXSize", FnGetSize);
    let get_ptx = required!("nvrtcGetPTX", FnGetText);
    let get_program_log_size = required!("nvrtcGetProgramLogSize", FnGetSize);
    let get_program_log = required!("nvrtcGetProgramLog", FnGetText);

    let get_error_string = optional!("nvrtcGetErrorString", FnGetErrorString);
    let get_cubin_size = optional!("nvrtcGetCUBINSize", FnGetSize);
    let get_cubin = optional!("nvrtcGetCUBIN", FnGetText);
    let add_name_expression = optional!("nvrtcAddNameExpression", FnAddNameExpression);
    let get_lowered_name = optional!("nvrtcGetLoweredName", FnGetLoweredName);
    let get_num_supported_archs = optional!("nvrtcGetNumSupportedArchs", FnGetNumSupportedArchs);
    let get_supported_archs = optional!("nvrtcGetSupportedArchs", FnGetSupportedArchs);

    Ok(NvrtcApi {
        version,
        create_program,
        destroy_program,
        compile_program,
        get_ptx_size,
        get_ptx,
        get_program_log_size,
        get_program_log,
        get_error_string,
        get_cubin_size,
        get_cubin,
        add_name_expression,
        get_lowered_name,
        get_num_supported_archs,
        get_supported_archs,
        _lib: lib,
    })
}

/// Process-wide cached NVRTC load result.
///
/// The success/failure of the first load attempt is remembered verbatim so
/// that a genuine [`NvrtcError::SymbolMissing`] is not collapsed into the
/// coarser [`NvrtcError::Unavailable`] on subsequent calls.
static NVRTC_API: OnceLock<Result<NvrtcApi, NvrtcError>> = OnceLock::new();

/// Return the resolved NVRTC API table, loading it once.
///
/// # Errors
///
/// Returns a clone of the cached load error — [`NvrtcError::Unavailable`] when
/// no library could be opened, or [`NvrtcError::SymbolMissing`] when a required
/// symbol was absent from an otherwise-loadable library.
pub(crate) fn api() -> Result<&'static NvrtcApi, NvrtcError> {
    match NVRTC_API.get_or_init(load) {
        Ok(api) => Ok(api),
        Err(e) => {
            // Emit the underlying diagnostic once so the reason is not lost
            // behind the many call sites that hit the cached error.
            static LOGGED: std::sync::Once = std::sync::Once::new();
            LOGGED.call_once(|| {
                tracing::warn!(error = %e, "NVRTC runtime load failed");
            });
            Err(e.clone())
        }
    }
}

/// Convert a raw `nvrtcResult` into a [`Result`], enriching the message via
/// `nvrtcGetErrorString` when available.
///
/// # Errors
///
/// Returns [`NvrtcError::Api`] for any non-success status code.
pub(crate) fn check(
    api: &NvrtcApi,
    call: &'static str,
    code: NvrtcResult,
) -> Result<(), NvrtcError> {
    if code == NVRTC_SUCCESS {
        Ok(())
    } else {
        Err(NvrtcError::Api {
            call,
            code,
            msg: api.error_string(code),
        })
    }
}

// ---------------------------------------------------------------------------
// Version query struct + top-level helpers
// ---------------------------------------------------------------------------

/// NVRTC runtime version, as reported by `nvrtcVersion`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NvrtcVersion {
    /// Major version component (e.g. `12` for NVRTC 12.4).
    pub major: i32,
    /// Minor version component (e.g. `4` for NVRTC 12.4).
    pub minor: i32,
}

/// Returns `true` if the NVRTC runtime library is present and every required
/// symbol resolved.
///
/// This never panics and caches its result: the first call performs the
/// `dlopen` probe, and all subsequent calls are a cheap atomic load returning
/// the same answer.
#[must_use]
pub fn is_available() -> bool {
    NVRTC_API.get_or_init(load).is_ok()
}

/// Query the NVRTC runtime version.
///
/// # Errors
///
/// Returns [`NvrtcError::Unavailable`] (or [`NvrtcError::SymbolMissing`]) when
/// the runtime cannot be loaded, or [`NvrtcError::Api`] if `nvrtcVersion`
/// reports a failure.
pub fn version() -> Result<NvrtcVersion, NvrtcError> {
    let api = api()?;
    let mut major: c_int = 0;
    let mut minor: c_int = 0;
    // SAFETY: `major`/`minor` are valid out-pointers for the duration of the
    // call; `api.version` is the resolved `nvrtcVersion`.
    let rc = unsafe { (api.version)(&mut major, &mut minor) };
    check(api, "nvrtcVersion", rc)?;
    Ok(NvrtcVersion { major, minor })
}

/// Query the list of GPU architectures (`sm_XX` as integers, e.g. `86`)
/// supported by this NVRTC runtime.
///
/// # Errors
///
/// Returns [`NvrtcError::NotSupported`] when the optional
/// `nvrtcGetNumSupportedArchs` / `nvrtcGetSupportedArchs` symbols are absent
/// (they were added in CUDA 11.2), [`NvrtcError::Unavailable`] when the runtime
/// cannot be loaded, or [`NvrtcError::Api`] on a driver-reported failure.
pub fn supported_archs() -> Result<Vec<i32>, NvrtcError> {
    let api = api()?;
    let get_num = api
        .get_num_supported_archs
        .ok_or(NvrtcError::NotSupported {
            symbol: "nvrtcGetNumSupportedArchs",
        })?;
    let get_archs = api.get_supported_archs.ok_or(NvrtcError::NotSupported {
        symbol: "nvrtcGetSupportedArchs",
    })?;

    let mut num: c_int = 0;
    // SAFETY: `num` is a valid out-pointer; `get_num` is the resolved
    // `nvrtcGetNumSupportedArchs`.
    let rc = unsafe { get_num(&mut num) };
    check(api, "nvrtcGetNumSupportedArchs", rc)?;
    if num <= 0 {
        return Ok(Vec::new());
    }

    let mut archs = vec![0i32; num as usize];
    // SAFETY: `archs` has room for exactly `num` `int`s, matching the count
    // just returned by `get_num`; `get_archs` writes that many entries.
    let rc = unsafe { get_archs(archs.as_mut_ptr()) };
    check(api, "nvrtcGetSupportedArchs", rc)?;
    Ok(archs)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_list_is_platform_specific() {
        let names = library_candidates();
        #[cfg(target_os = "linux")]
        {
            assert_eq!(names.first(), Some(&"libnvrtc.so"));
            assert!(names.contains(&"libnvrtc.so.12"));
            assert!(names.contains(&"libnvrtc.so.11"));
        }
        #[cfg(target_os = "windows")]
        {
            assert!(names.contains(&"nvrtc64_120_0.dll"));
        }
        // Every candidate must be a plausible library file name.
        for n in names {
            assert!(!n.is_empty());
        }
    }

    #[test]
    fn is_available_is_consistent_across_calls() {
        let a = is_available();
        let b = is_available();
        assert_eq!(a, b, "is_available() must be stable across calls");
    }

    #[test]
    fn open_library_reports_candidates_when_absent() {
        // A name that cannot exist as a real library.
        let err = open_library(&["definitely-not-a-real-nvrtc-xyz.so"])
            .expect_err("loading a bogus library must fail");
        match err {
            NvrtcError::Unavailable { candidates, .. } => {
                assert_eq!(
                    candidates,
                    vec!["definitely-not-a-real-nvrtc-xyz.so".to_string()]
                );
            }
            other => panic!("expected Unavailable, got {other:?}"),
        }
    }

    #[test]
    fn open_library_empty_candidates_is_unavailable() {
        let err = open_library(&[]).expect_err("no candidates must fail");
        assert!(matches!(err, NvrtcError::Unavailable { .. }));
    }
}
