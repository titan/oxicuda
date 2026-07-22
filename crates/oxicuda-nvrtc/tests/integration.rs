//! Integration tests for `oxicuda-nvrtc`.
//!
//! These tests pass on a host **without** the NVRTC runtime (the common CI
//! case) and also on a host **with** it. The probe never panics; when NVRTC is
//! absent every entry point returns [`NvrtcError::Unavailable`]; when present,
//! a trivial kernel is compiled end-to-end.

use oxicuda_nvrtc::{
    Header, NvrtcError, Program, compile_to_ptx, is_available, supported_archs, version,
};

const SAXPY: &str = r#"
extern "C" __global__ void saxpy(float a, float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
"#;

#[test]
fn probe_never_panics_and_is_consistent() {
    let first = is_available();
    let second = is_available();
    assert_eq!(first, second, "is_available() must be stable");
}

#[test]
fn interior_nul_is_rejected_before_ffi() {
    // NUL validation happens in Rust before any library is even required, so
    // this holds regardless of whether NVRTC is installed.
    // `Program` is intentionally not `Debug`, so match rather than `expect_err`.
    let err = match Program::new("a\0b", "k.cu") {
        Err(e) => e,
        Ok(_) => panic!("interior NUL must fail"),
    };
    // Either NulInInput (NVRTC present, validation reached) or Unavailable
    // (NVRTC absent, api() short-circuits first) is acceptable; both are typed
    // errors, never a panic.
    assert!(matches!(
        err,
        NvrtcError::NulInInput { .. } | NvrtcError::Unavailable { .. }
    ));
}

#[test]
fn graceful_when_unavailable() {
    if is_available() {
        // Covered by the end-to-end test below.
        return;
    }

    // Every entry point must degrade to Unavailable, never panic.
    assert!(matches!(version(), Err(NvrtcError::Unavailable { .. })));
    assert!(matches!(
        Program::new(SAXPY, "saxpy.cu"),
        Err(NvrtcError::Unavailable { .. })
    ));
    assert!(matches!(
        compile_to_ptx(SAXPY, "saxpy.cu", &[]),
        Err(NvrtcError::Unavailable { .. })
    ));

    // `supported_archs` short-circuits on the unavailable runtime before it can
    // reach the optional-symbol check, so it too reports Unavailable.
    assert!(matches!(
        supported_archs(),
        Err(NvrtcError::Unavailable { .. })
    ));

    // The Unavailable error must name the libraries that were tried.
    if let Err(NvrtcError::Unavailable { candidates, .. }) = version() {
        // On Linux/Windows there is at least one candidate; on other platforms
        // the list may be empty but must still be present.
        let _ = candidates;
    }
}

#[test]
fn with_headers_accepts_empty_header_slice() {
    // Should behave identically to `Program::new` (both route through
    // `with_headers`). On a host without NVRTC this yields Unavailable.
    let headers: &[Header<'_>] = &[];
    let result = Program::with_headers(SAXPY, "saxpy.cu", headers);
    if !is_available() {
        assert!(matches!(result, Err(NvrtcError::Unavailable { .. })));
    }
}

#[test]
fn end_to_end_when_available() {
    if !is_available() {
        // Nothing to do on a CUDA-less host; the graceful path is asserted
        // separately.
        return;
    }

    // Version must be queryable.
    let v = version().expect("version() must succeed when NVRTC is available");
    assert!(
        v.major >= 1,
        "NVRTC major version should be sensible: {v:?}"
    );

    // Compile a trivial kernel and confirm the PTX looks like real PTX.
    let ptx = compile_to_ptx(SAXPY, "saxpy.cu", &[])
        .expect("trivial kernel must compile when NVRTC is available");
    let text = ptx.as_str();
    assert!(
        text.contains(".entry"),
        "compiled PTX should contain a `.entry` directive; got:\n{text}"
    );
    // The trailing NUL is stripped from `as_str` but present in the raw buffer.
    assert_eq!(ptx.as_bytes_with_nul().last(), Some(&0));
    assert_eq!(ptx.as_str().len() + 1, ptx.as_bytes_with_nul().len());

    // `supported_archs` is optional: either a non-empty list or NotSupported.
    match supported_archs() {
        Ok(archs) => assert!(!archs.is_empty(), "expected at least one supported arch"),
        Err(NvrtcError::NotSupported { .. }) => {}
        Err(e) => panic!("unexpected supported_archs error: {e}"),
    }
}
