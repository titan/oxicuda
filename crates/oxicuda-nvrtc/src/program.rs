//! The [`Program`] type — an RAII handle to an NVRTC compilation unit.
//!
//! A [`Program`] wraps a live `nvrtcProgram`. It is created from CUDA-C source
//! ([`Program::new`] / [`Program::with_headers`]), compiled ([`Program::compile`]),
//! and then queried for its [`Ptx`] output, CUBIN, compiler log, or lowered
//! name-expression mangling. The underlying `nvrtcProgram` is destroyed exactly
//! once, on [`Drop`], which never panics.

use std::ffi::{CStr, CString, c_char};
use std::ptr;

use crate::error::NvrtcError;
use crate::loader::{self, NVRTC_SUCCESS, NvrtcApi, NvrtcProgram, check};
use crate::ptx::Ptx;

/// A named header made available to the NVRTC compiler via `#include`.
///
/// `name` is the include path a kernel would `#include "..."`, and `contents`
/// is the header source text that should be substituted for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Header<'a> {
    /// The include name, e.g. `"helpers.cuh"`.
    pub name: &'a str,
    /// The full header source text.
    pub contents: &'a str,
}

/// An RAII handle to a live NVRTC program (`nvrtcProgram`).
///
/// The program is destroyed on [`Drop`]; if creation succeeded but a later step
/// fails, the program is still destroyed. `Program` is [`Send`] but not [`Sync`]
/// (see the `unsafe impl Send` rationale in the source).
pub struct Program {
    /// Raw `nvrtcProgram` handle. Null only transiently during [`Drop`].
    prog: NvrtcProgram,
    /// The resident, process-wide NVRTC function table.
    api: &'static NvrtcApi,
}

// SAFETY: NVRTC programs have no thread affinity — a `nvrtcProgram` created on
// one thread may be compiled and read from another, and distinct programs may
// be driven from distinct threads concurrently. The only non-`Send` field is
// the raw `prog` pointer, which is a plain library-side identifier that we own
// uniquely (there is no interior sharing). We therefore assert `Send`.
//
// We deliberately do NOT implement `Sync`: `&Program` would permit two threads
// to call `nvrtcAddNameExpression` / `nvrtcCompileProgram` on the same handle
// concurrently, which NVRTC does not guarantee is safe. `Sync` is left off so
// the borrow checker forbids that aliasing.
unsafe impl Send for Program {}

impl Program {
    /// Create a program from CUDA-C `source` with the given `name`.
    ///
    /// `name` is a display name used in compiler diagnostics (conventionally
    /// something like `"my_kernel.cu"`); it does not need to correspond to a
    /// real file.
    ///
    /// # Errors
    ///
    /// * [`NvrtcError::NulInInput`] if `source` or `name` contains an interior
    ///   NUL byte.
    /// * [`NvrtcError::Unavailable`] / [`NvrtcError::SymbolMissing`] if the
    ///   NVRTC runtime cannot be loaded.
    /// * [`NvrtcError::Api`] if `nvrtcCreateProgram` reports a failure.
    pub fn new(source: &str, name: &str) -> Result<Self, NvrtcError> {
        Self::with_headers(source, name, &[])
    }

    /// Create a program from CUDA-C `source`, supplying in-memory `headers`
    /// that the source may `#include`.
    ///
    /// # Errors
    ///
    /// * [`NvrtcError::NulInInput`] if `source`, `name`, or any header name /
    ///   contents contains an interior NUL byte.
    /// * [`NvrtcError::Unavailable`] / [`NvrtcError::SymbolMissing`] if the
    ///   NVRTC runtime cannot be loaded.
    /// * [`NvrtcError::Api`] if `nvrtcCreateProgram` reports a failure.
    pub fn with_headers(
        source: &str,
        name: &str,
        headers: &[Header<'_>],
    ) -> Result<Self, NvrtcError> {
        let api = loader::api()?;

        let c_source = to_cstring(source, "source")?;
        let c_name = to_cstring(name, "kernel name")?;

        // Build the two parallel arrays NVRTC expects: header *contents* and
        // include *names*. We keep both the owning `Vec<CString>`s and the
        // `Vec<*const c_char>` pointer arrays alive across the FFI call.
        let mut header_contents: Vec<CString> = Vec::with_capacity(headers.len());
        let mut include_names: Vec<CString> = Vec::with_capacity(headers.len());
        for h in headers {
            header_contents.push(to_cstring(h.contents, "header contents")?);
            include_names.push(to_cstring(h.name, "header name")?);
        }
        let content_ptrs: Vec<*const c_char> = header_contents.iter().map(|c| c.as_ptr()).collect();
        let name_ptrs: Vec<*const c_char> = include_names.iter().map(|c| c.as_ptr()).collect();

        let num_headers = headers.len() as std::ffi::c_int;
        let (headers_ptr, includes_ptr) = if headers.is_empty() {
            (ptr::null(), ptr::null())
        } else {
            (content_ptrs.as_ptr(), name_ptrs.as_ptr())
        };

        let mut prog: NvrtcProgram = ptr::null_mut();
        // SAFETY: `c_source`/`c_name` are valid NUL-terminated strings that
        // outlive the call. When `headers` is non-empty, `content_ptrs` and
        // `name_ptrs` each hold `num_headers` valid pointers into
        // `header_contents` / `include_names`, all of which outlive the call;
        // when empty, both pointers are null and `num_headers` is 0.
        let rc = unsafe {
            (api.create_program)(
                &mut prog,
                c_source.as_ptr(),
                c_name.as_ptr(),
                num_headers,
                headers_ptr,
                includes_ptr,
            )
        };
        check(api, "nvrtcCreateProgram", rc)?;
        Ok(Self { prog, api })
    }

    /// Register a C++ name expression for later mangled-name lookup.
    ///
    /// Must be called *before* [`compile`](Self::compile). After compilation,
    /// pass the same expression to [`lowered_name`](Self::lowered_name) to
    /// obtain the mangled symbol name.
    ///
    /// # Errors
    ///
    /// * [`NvrtcError::NotSupported`] if `nvrtcAddNameExpression` is absent from
    ///   this NVRTC runtime.
    /// * [`NvrtcError::NulInInput`] if `expr` contains an interior NUL byte.
    /// * [`NvrtcError::Api`] on a driver-reported failure.
    pub fn add_name_expression(&mut self, expr: &str) -> Result<(), NvrtcError> {
        let api = self.api;
        let f = api.add_name_expression.ok_or(NvrtcError::NotSupported {
            symbol: "nvrtcAddNameExpression",
        })?;
        let c_expr = to_cstring(expr, "name expression")?;
        // SAFETY: `self.prog` is a valid program handle; `c_expr` is a valid
        // NUL-terminated string that outlives the call.
        let rc = unsafe { f(self.prog, c_expr.as_ptr()) };
        check(api, "nvrtcAddNameExpression", rc)
    }

    /// Compile the program with the given `options` (e.g. `["--gpu-architecture=compute_86"]`).
    ///
    /// # Errors
    ///
    /// * [`NvrtcError::NulInInput`] if any option contains an interior NUL byte.
    /// * [`NvrtcError::Compilation`] if compilation fails; the compiler log is
    ///   fetched eagerly and attached.
    pub fn compile(&mut self, options: &[&str]) -> Result<(), NvrtcError> {
        let api = self.api;

        // Keep the owning `CString`s and the pointer array alive across the call.
        let mut c_options: Vec<CString> = Vec::with_capacity(options.len());
        for opt in options {
            c_options.push(to_cstring(opt, "compile option")?);
        }
        let option_ptrs: Vec<*const c_char> = c_options.iter().map(|c| c.as_ptr()).collect();

        let num_options = options.len() as std::ffi::c_int;
        let options_ptr = if options.is_empty() {
            ptr::null()
        } else {
            option_ptrs.as_ptr()
        };

        // SAFETY: `self.prog` is a valid program handle; when `options` is
        // non-empty, `option_ptrs` holds `num_options` valid pointers into
        // `c_options`, both of which outlive the call; when empty, the pointer
        // is null and `num_options` is 0.
        let rc = unsafe { (api.compile_program)(self.prog, num_options, options_ptr) };
        if rc != NVRTC_SUCCESS {
            // Fetch the compiler log eagerly so the failure is self-describing.
            let log = self.log_lossy();
            let msg = api.error_string(rc);
            return Err(NvrtcError::Compilation { code: rc, msg, log });
        }
        Ok(())
    }

    /// Return the compiler log (warnings and informational messages).
    ///
    /// # Errors
    ///
    /// * [`NvrtcError::Api`] on a driver-reported failure.
    /// * [`NvrtcError::InvalidUtf8`] if the log is not valid UTF-8.
    pub fn log(&self) -> Result<String, NvrtcError> {
        let api = self.api;
        let mut size: usize = 0;
        // SAFETY: `self.prog` is valid; `size` is a valid out-pointer.
        let rc = unsafe { (api.get_program_log_size)(self.prog, &mut size) };
        check(api, "nvrtcGetProgramLogSize", rc)?;
        // The size includes the trailing NUL, so a size of 0 or 1 is an empty log.
        if size <= 1 {
            return Ok(String::new());
        }
        let mut buf = vec![0u8; size];
        // SAFETY: `buf` has room for exactly `size` bytes including the trailing
        // NUL, matching the size just queried.
        let rc = unsafe { (api.get_program_log)(self.prog, buf.as_mut_ptr().cast::<c_char>()) };
        check(api, "nvrtcGetProgramLog", rc)?;
        strip_trailing_nul(&mut buf);
        String::from_utf8(buf).map_err(|e| NvrtcError::InvalidUtf8 {
            what: "program log",
            reason: e.utf8_error().to_string(),
        })
    }

    /// Return the compiled PTX.
    ///
    /// # Errors
    ///
    /// * [`NvrtcError::Api`] on a driver-reported failure.
    /// * [`NvrtcError::InvalidUtf8`] if the PTX is not valid UTF-8.
    pub fn ptx(&self) -> Result<Ptx, NvrtcError> {
        let api = self.api;
        let mut size: usize = 0;
        // SAFETY: `self.prog` is valid; `size` is a valid out-pointer.
        let rc = unsafe { (api.get_ptx_size)(self.prog, &mut size) };
        check(api, "nvrtcGetPTXSize", rc)?;
        if size == 0 {
            return Err(NvrtcError::Api {
                call: "nvrtcGetPTXSize",
                code: NVRTC_SUCCESS,
                msg: "NVRTC reported a zero-length PTX size".to_string(),
            });
        }
        // The reported size includes the trailing NUL; allocate exactly that.
        let mut buf = vec![0u8; size];
        // SAFETY: `buf` has room for exactly `size` bytes including the trailing
        // NUL, matching the size just queried.
        let rc = unsafe { (api.get_ptx)(self.prog, buf.as_mut_ptr().cast::<c_char>()) };
        check(api, "nvrtcGetPTX", rc)?;
        Ptx::from_nul_terminated(buf)
    }

    /// Return the compiled CUBIN (device machine code) as raw bytes.
    ///
    /// # Errors
    ///
    /// * [`NvrtcError::NotSupported`] if `nvrtcGetCUBINSize` / `nvrtcGetCUBIN`
    ///   are absent from this NVRTC runtime.
    /// * [`NvrtcError::Api`] on a driver-reported failure.
    pub fn cubin(&self) -> Result<Vec<u8>, NvrtcError> {
        let api = self.api;
        let get_size = api.get_cubin_size.ok_or(NvrtcError::NotSupported {
            symbol: "nvrtcGetCUBINSize",
        })?;
        let get_cubin = api.get_cubin.ok_or(NvrtcError::NotSupported {
            symbol: "nvrtcGetCUBIN",
        })?;

        let mut size: usize = 0;
        // SAFETY: `self.prog` is valid; `size` is a valid out-pointer.
        let rc = unsafe { get_size(self.prog, &mut size) };
        check(api, "nvrtcGetCUBINSize", rc)?;
        let mut buf = vec![0u8; size];
        if size > 0 {
            // SAFETY: `buf` has room for exactly `size` bytes matching the size
            // just queried. CUBIN is binary and carries no NUL terminator.
            let rc = unsafe { get_cubin(self.prog, buf.as_mut_ptr().cast::<c_char>()) };
            check(api, "nvrtcGetCUBIN", rc)?;
        }
        Ok(buf)
    }

    /// Return the mangled (lowered) symbol name for a previously-registered
    /// name expression.
    ///
    /// The `expr` must exactly match one passed to
    /// [`add_name_expression`](Self::add_name_expression) *before* compilation.
    ///
    /// # Errors
    ///
    /// * [`NvrtcError::NotSupported`] if `nvrtcGetLoweredName` is absent from
    ///   this NVRTC runtime.
    /// * [`NvrtcError::NulInInput`] if `expr` contains an interior NUL byte.
    /// * [`NvrtcError::Api`] on a driver-reported failure.
    /// * [`NvrtcError::InvalidUtf8`] if the lowered name is not valid UTF-8.
    pub fn lowered_name(&self, expr: &str) -> Result<String, NvrtcError> {
        let api = self.api;
        let f = api.get_lowered_name.ok_or(NvrtcError::NotSupported {
            symbol: "nvrtcGetLoweredName",
        })?;
        let c_expr = to_cstring(expr, "name expression")?;
        let mut out: *const c_char = ptr::null();
        // SAFETY: `self.prog` is valid; `c_expr` outlives the call; `out` is a
        // valid out-pointer that receives an NVRTC-owned string pointer.
        let rc = unsafe { f(self.prog, c_expr.as_ptr(), &mut out) };
        check(api, "nvrtcGetLoweredName", rc)?;
        if out.is_null() {
            return Err(NvrtcError::Api {
                call: "nvrtcGetLoweredName",
                code: NVRTC_SUCCESS,
                msg: "NVRTC returned a null lowered-name pointer".to_string(),
            });
        }
        // SAFETY: `out` is non-null and points to a NUL-terminated C string
        // owned by NVRTC and valid until the program is destroyed. We copy it
        // out immediately.
        let text = unsafe { CStr::from_ptr(out) };
        text.to_str()
            .map(str::to_owned)
            .map_err(|e| NvrtcError::InvalidUtf8 {
                what: "lowered name",
                reason: e.to_string(),
            })
    }

    /// Fetch the compiler log, decoding lossily and returning an empty string on
    /// any driver error. Used to attach diagnostics to a [`NvrtcError::Compilation`].
    fn log_lossy(&self) -> String {
        let api = self.api;
        let mut size: usize = 0;
        // SAFETY: `self.prog` is valid; `size` is a valid out-pointer.
        let rc = unsafe { (api.get_program_log_size)(self.prog, &mut size) };
        if rc != NVRTC_SUCCESS || size <= 1 {
            return String::new();
        }
        let mut buf = vec![0u8; size];
        // SAFETY: `buf` has room for exactly `size` bytes including the trailing
        // NUL, matching the size just queried.
        let rc = unsafe { (api.get_program_log)(self.prog, buf.as_mut_ptr().cast::<c_char>()) };
        if rc != NVRTC_SUCCESS {
            return String::new();
        }
        strip_trailing_nul(&mut buf);
        String::from_utf8_lossy(&buf).into_owned()
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        if self.prog.is_null() {
            return;
        }
        // SAFETY: `self.prog` is a valid handle created by `nvrtcCreateProgram`
        // and destroyed exactly once here. `nvrtcDestroyProgram` takes a pointer
        // to the handle and nulls it.
        let rc = unsafe { (self.api.destroy_program)(&mut self.prog) };
        if rc != NVRTC_SUCCESS {
            tracing::warn!(nvrtc_error = rc, "nvrtcDestroyProgram failed during drop");
        }
    }
}

/// Compile CUDA-C `source` (named `name`) to PTX in one call.
///
/// A convenience wrapper over [`Program::new`] + [`Program::compile`] +
/// [`Program::ptx`].
///
/// # Errors
///
/// Propagates any error from the underlying steps — most notably
/// [`NvrtcError::Unavailable`] when NVRTC is absent and
/// [`NvrtcError::Compilation`] when the source does not compile.
pub fn compile_to_ptx(source: &str, name: &str, options: &[&str]) -> Result<Ptx, NvrtcError> {
    let mut program = Program::new(source, name)?;
    program.compile(options)?;
    program.ptx()
}

/// Build a [`CString`] from `s`, mapping an interior NUL to
/// [`NvrtcError::NulInInput`].
fn to_cstring(s: &str, what: &'static str) -> Result<CString, NvrtcError> {
    CString::new(s).map_err(|e| NvrtcError::NulInInput {
        what,
        position: e.nul_position(),
    })
}

/// Remove any trailing NUL bytes from `buf` in place.
fn strip_trailing_nul(buf: &mut Vec<u8>) {
    while buf.last() == Some(&0) {
        buf.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_cstring_detects_interior_nul() {
        let err = to_cstring("ab\0cd", "source").expect_err("interior NUL must fail");
        match err {
            NvrtcError::NulInInput { what, position } => {
                assert_eq!(what, "source");
                assert_eq!(position, 2);
            }
            other => panic!("expected NulInInput, got {other:?}"),
        }
    }

    #[test]
    fn to_cstring_accepts_clean_input() {
        let c = to_cstring("clean", "source").expect("clean input");
        assert_eq!(c.as_bytes(), b"clean");
    }

    #[test]
    fn strip_trailing_nul_removes_all_terminators() {
        let mut buf = b"log text\0\0\0".to_vec();
        strip_trailing_nul(&mut buf);
        assert_eq!(buf, b"log text");
    }

    #[test]
    fn strip_trailing_nul_on_empty_is_noop() {
        let mut buf: Vec<u8> = Vec::new();
        strip_trailing_nul(&mut buf);
        assert!(buf.is_empty());
    }

    #[test]
    fn header_is_copy_and_holds_fields() {
        let h = Header {
            name: "helpers.cuh",
            contents: "#define PI 3.14",
        };
        let h2 = h; // Copy
        assert_eq!(h.name, h2.name);
        assert_eq!(h2.contents, "#define PI 3.14");
    }
}
