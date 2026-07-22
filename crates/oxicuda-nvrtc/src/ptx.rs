//! The [`Ptx`] type — owned, NUL-terminated NVRTC PTX output.
//!
//! NVRTC returns PTX as a C string whose reported size *includes* the trailing
//! NUL terminator. [`Ptx`] owns that buffer and exposes it two ways:
//!
//! * [`Ptx::as_str`] — the PTX text **without** the trailing NUL, ready to hand
//!   straight to `oxicuda_driver::Module::from_ptx(&str)`.
//! * [`Ptx::as_bytes_with_nul`] / [`Ptx::into_bytes_with_nul`] — the raw buffer
//!   **including** the trailing NUL, for callers that want to feed a C API
//!   directly.
//!
//! UTF-8 is validated once, at construction (inside [`crate::Program::ptx`]), so
//! [`Ptx::as_str`] is infallible.

use crate::error::NvrtcError;

/// Owned PTX output from an NVRTC compilation.
///
/// The internal buffer always ends in exactly one NUL byte, and the text
/// portion preceding it is guaranteed to be valid UTF-8 (checked when the
/// `Ptx` is constructed).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ptx {
    /// PTX text followed by exactly one trailing NUL byte.
    bytes: Vec<u8>,
    /// Length of the UTF-8 text, i.e. `bytes.len() - 1` (index of the NUL).
    text_len: usize,
}

impl Ptx {
    /// Construct a [`Ptx`] from a raw NVRTC output buffer.
    ///
    /// The buffer is the exact `Vec<u8>` written by `nvrtcGetPTX` (its length is
    /// the `nvrtcGetPTXSize` value, which includes the trailing NUL). The text
    /// portion — everything up to the first NUL — is validated as UTF-8, and the
    /// buffer is normalised to end in exactly one NUL byte.
    ///
    /// This is `pub(crate)` because a well-formed `Ptx` can only originate from
    /// an NVRTC compilation; it is exercised directly by the crate's unit tests.
    ///
    /// # Errors
    ///
    /// Returns [`NvrtcError::InvalidUtf8`] if the text portion is not valid
    /// UTF-8.
    pub(crate) fn from_nul_terminated(mut bytes: Vec<u8>) -> Result<Self, NvrtcError> {
        // The text is everything up to the first NUL (NVRTC appends exactly one,
        // but we defensively scan rather than assume a fixed position).
        let text_len = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());

        // Validate UTF-8 of the text portion up front so `as_str` is infallible.
        if let Err(e) = std::str::from_utf8(&bytes[..text_len]) {
            return Err(NvrtcError::InvalidUtf8 {
                what: "PTX",
                reason: e.to_string(),
            });
        }

        // Normalise storage: text bytes followed by exactly one NUL. This keeps
        // `as_bytes_with_nul` correct even if NVRTC ever returned extra padding
        // or omitted the terminator.
        bytes.truncate(text_len);
        bytes.push(0);
        Ok(Self { bytes, text_len })
    }

    /// The PTX text, **without** the trailing NUL.
    ///
    /// This feeds `oxicuda_driver::Module::from_ptx(&str)` directly.
    #[must_use]
    pub fn as_str(&self) -> &str {
        // SAFETY: `bytes[..text_len]` was validated as UTF-8 at construction and
        // the buffer is never mutated afterwards.
        unsafe { std::str::from_utf8_unchecked(&self.bytes[..self.text_len]) }
    }

    /// The raw PTX buffer, **including** the trailing NUL byte.
    #[must_use]
    pub fn as_bytes_with_nul(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume this [`Ptx`], returning the owned buffer **including** the
    /// trailing NUL byte.
    #[must_use]
    pub fn into_bytes_with_nul(self) -> Vec<u8> {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_trailing_nul_for_as_str() {
        let ptx = Ptx::from_nul_terminated(b"hello world\0".to_vec()).expect("valid utf-8");
        assert_eq!(ptx.as_str(), "hello world");
        assert_eq!(ptx.as_bytes_with_nul(), b"hello world\0");
        assert_eq!(ptx.as_str().len() + 1, ptx.as_bytes_with_nul().len());
    }

    #[test]
    fn adds_terminator_when_input_has_none() {
        // Input without a trailing NUL is still normalised to be NUL-terminated.
        let ptx = Ptx::from_nul_terminated(b"abc".to_vec()).expect("valid utf-8");
        assert_eq!(ptx.as_str(), "abc");
        assert_eq!(ptx.as_bytes_with_nul(), b"abc\0");
    }

    #[test]
    fn empty_ptx_round_trips() {
        let ptx = Ptx::from_nul_terminated(b"\0".to_vec()).expect("valid utf-8");
        assert_eq!(ptx.as_str(), "");
        assert_eq!(ptx.as_bytes_with_nul(), b"\0");
        assert_eq!(ptx.into_bytes_with_nul(), b"\0");
    }

    #[test]
    fn into_bytes_with_nul_preserves_terminator() {
        let ptx = Ptx::from_nul_terminated(b".entry k\0".to_vec()).expect("valid utf-8");
        let bytes = ptx.into_bytes_with_nul();
        assert_eq!(bytes.last(), Some(&0));
        assert_eq!(&bytes[..bytes.len() - 1], b".entry k");
    }

    #[test]
    fn invalid_utf8_is_rejected() {
        // 0xFF is never valid UTF-8.
        let err = Ptx::from_nul_terminated(vec![0xFF, 0x00]).expect_err("invalid utf-8 must fail");
        assert!(matches!(err, NvrtcError::InvalidUtf8 { what: "PTX", .. }));
    }

    #[test]
    fn text_before_interior_nul_is_used() {
        // Everything up to the first NUL is the text; trailing bytes are dropped.
        let ptx = Ptx::from_nul_terminated(b"good\0junk".to_vec()).expect("valid utf-8");
        assert_eq!(ptx.as_str(), "good");
        assert_eq!(ptx.as_bytes_with_nul(), b"good\0");
    }
}
