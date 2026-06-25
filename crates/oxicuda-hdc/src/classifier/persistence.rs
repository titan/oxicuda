//! HD-classifier export / import (model persistence).
//!
//! A trained HD classifier is fully described by its binary class prototypes
//! (one [`Vec<i8>`] in `{-1, +1}` per class) together with two pieces of
//! metadata: the number of classes and the hypervector dimension. This module
//! provides a compact, hand-rolled, deterministic serialisation of that model
//! to a byte buffer and to a human-readable text form, with no `serde` and no
//! external crate involved — only the standard library.
//!
//! # Binary layout (produced by [`HdModel::to_bytes`])
//!
//! All multi-byte integers are little-endian.
//!
//! ```text
//! offset  size            field
//! ------  --------------  -------------------------------------------------
//! 0       4               magic = b"HDC1" (0x48 0x44 0x43 0x31)
//! 4       4               n_classes : u32 little-endian
//! 8       4               dim       : u32 little-endian
//! 12      n_classes * S   payload   (S = ceil(dim / 8) bytes per class)
//! ```
//!
//! The header is therefore always exactly `12` bytes and the total length is
//! `12 + n_classes * ceil(dim / 8)`.
//!
//! ## Bit order
//!
//! Each prototype is packed bit-by-bit, **MSB-first within every byte**: the
//! prototype element at index `i` lives in byte `i / 8` at bit position
//! `7 - (i % 8)`. A value of `+1` sets the bit to `1`; a value of `-1` clears
//! it to `0`. When `dim` is not a multiple of `8` the final byte of a class is
//! zero-padded in its least-significant (last-written) bits; those padding bits
//! are ignored on read, so any `dim` round-trips exactly.
//!
//! # Text layout (produced by [`HdModel::to_string_repr`])
//!
//! ```text
//! HDC1 <n_classes> <dim>
//! <+/- chars, length dim>      // class 0
//! <+/- chars, length dim>      // class 1
//! ...                          // one line per class
//! ```
//!
//! The first line is the magic token `HDC1` followed by the decimal
//! `n_classes` and `dim`, space-separated. Each subsequent line encodes one
//! prototype as exactly `dim` characters, `'+'` for `+1` and `'-'` for `-1`.
//!
//! # Error handling
//!
//! No new error variants are introduced. Malformed or truncated input is
//! reported with the existing [`HdcError`] variants:
//!
//! * [`HdcError::EmptyInput`] — empty buffer / empty string / no prototypes.
//! * [`HdcError::DimensionMismatch`] — wrong magic, a header that does not fit,
//!   a buffer / line whose length disagrees with the header, or a wrong number
//!   of text lines. `expected` and `got` carry the two lengths that disagree.
//! * [`HdcError::InvalidBinaryValue`] — a packed/text value that is not `±1`
//!   (in practice only reachable from the text parser via a stray character,
//!   which is mapped to a non-`±1` sentinel).
//! * [`HdcError::ZeroDimension`] — a header declaring `dim == 0`.
//! * [`HdcError::ClassNotFound`] — out-of-range class index in [`HdModel::prototype`].

use crate::error::{HdcError, HdcResult};
use crate::vector::binary::validate_binary;

/// 4-byte magic identifying the v1 HD-model format.
const MAGIC: [u8; 4] = *b"HDC1";

/// Magic token used at the start of the text representation.
const TEXT_MAGIC: &str = "HDC1";

/// Fixed binary header length: 4 (magic) + 4 (n_classes) + 4 (dim).
const HEADER_LEN: usize = 12;

/// Number of packed bytes needed to store `dim` bits (`ceil(dim / 8)`).
#[inline]
fn packed_bytes_per_class(dim: usize) -> usize {
    dim.div_ceil(8)
}

/// A serialisable trained HD classifier model.
///
/// Holds the binary class prototypes and the metadata required to reconstruct
/// them. Every prototype has length [`HdModel::dim`] and contains only `±1`
/// values; these invariants are established by [`HdModel::new`] and preserved
/// by every constructor in this module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HdModel {
    /// Number of class prototypes (`== prototypes.len()`).
    n_classes: usize,
    /// Hypervector dimension shared by every prototype (`> 0`).
    dim: usize,
    /// One binary prototype (`±1`, length `dim`) per class.
    prototypes: Vec<Vec<i8>>,
}

impl HdModel {
    /// Build a model from per-class binary prototypes, validating all invariants.
    ///
    /// # Errors
    ///
    /// * [`HdcError::EmptyInput`] if `prototypes` is empty.
    /// * [`HdcError::ZeroDimension`] if the (common) prototype length is `0`.
    /// * [`HdcError::DimensionMismatch`] if the prototypes are not all the same
    ///   length (`expected` = first row length, `got` = offending row length).
    /// * [`HdcError::InvalidBinaryValue`] if any value is not in `{-1, +1}`.
    pub fn new(prototypes: Vec<Vec<i8>>) -> HdcResult<Self> {
        if prototypes.is_empty() {
            return Err(HdcError::EmptyInput);
        }
        // The dimension is taken from the first prototype.
        let dim = prototypes[0].len();
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        for row in &prototypes {
            if row.len() != dim {
                return Err(HdcError::DimensionMismatch {
                    expected: dim,
                    got: row.len(),
                });
            }
            validate_binary(row)?;
        }
        let n_classes = prototypes.len();
        Ok(Self {
            n_classes,
            dim,
            prototypes,
        })
    }

    /// Number of class prototypes stored in this model.
    #[inline]
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Hypervector dimension shared by every prototype.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Borrow the prototype for `class`.
    ///
    /// # Errors
    ///
    /// [`HdcError::ClassNotFound`] if `class >= n_classes`.
    pub fn prototype(&self, class: usize) -> HdcResult<&[i8]> {
        self.prototypes
            .get(class)
            .map(|p| p.as_slice())
            .ok_or(HdcError::ClassNotFound(class))
    }

    /// Borrow all prototypes (one slice per class, in class order).
    #[inline]
    pub fn prototypes(&self) -> &[Vec<i8>] {
        &self.prototypes
    }

    /// Exact length, in bytes, of the buffer produced by [`HdModel::to_bytes`].
    #[inline]
    pub fn byte_len(&self) -> usize {
        HEADER_LEN + self.n_classes * packed_bytes_per_class(self.dim)
    }

    /// Serialise the model to the compact bit-packed binary format.
    ///
    /// See the module-level documentation for the precise byte layout (magic,
    /// little-endian `u32` header, MSB-first bit packing). The returned buffer
    /// has length [`HdModel::byte_len`] and round-trips exactly through
    /// [`HdModel::from_bytes`].
    pub fn to_bytes(&self) -> Vec<u8> {
        let stride = packed_bytes_per_class(self.dim);
        let mut buf = Vec::with_capacity(self.byte_len());

        buf.extend_from_slice(&MAGIC);
        // `as u32` is safe in practice: HD dims/class counts are far below 2^32.
        buf.extend_from_slice(&(self.n_classes as u32).to_le_bytes());
        buf.extend_from_slice(&(self.dim as u32).to_le_bytes());

        for proto in &self.prototypes {
            // Start each class with a zero-filled, fully-allocated chunk so the
            // padding bits in the final byte are deterministically `0`.
            let start = buf.len();
            buf.resize(start + stride, 0u8);
            for (i, &value) in proto.iter().enumerate() {
                if value == 1 {
                    let byte_index = start + (i / 8);
                    let bit = 7 - (i % 8);
                    buf[byte_index] |= 1u8 << bit;
                }
                // value == -1 leaves the (already-zero) bit clear.
            }
        }
        buf
    }

    /// Deserialise a model from the bit-packed binary format.
    ///
    /// # Errors
    ///
    /// * [`HdcError::EmptyInput`] if `buf` is empty.
    /// * [`HdcError::DimensionMismatch`] if the magic is wrong, the header is
    ///   truncated, or the total length disagrees with `12 + n_classes *
    ///   ceil(dim / 8)`. For a length disagreement `expected` is the required
    ///   length and `got` is the actual length of `buf`.
    /// * [`HdcError::ZeroDimension`] if the header declares `dim == 0`.
    /// * [`HdcError::InvalidBinaryValue`] propagated from reconstruction (not
    ///   reachable from a well-formed buffer, since unpacking only emits `±1`).
    pub fn from_bytes(buf: &[u8]) -> HdcResult<Self> {
        if buf.is_empty() {
            return Err(HdcError::EmptyInput);
        }
        // Header must be fully present before we trust any field.
        let header = buf.get(..HEADER_LEN).ok_or(HdcError::DimensionMismatch {
            expected: HEADER_LEN,
            got: buf.len(),
        })?;

        // Magic check. `expected`/`got` carry the two magic words as `u32`s so
        // the error still distinguishes "wrong magic" from "wrong length".
        let magic = &header[0..4];
        if magic != MAGIC {
            return Err(HdcError::DimensionMismatch {
                expected: u32::from_le_bytes(MAGIC) as usize,
                got: u32::from_le_bytes([magic[0], magic[1], magic[2], magic[3]]) as usize,
            });
        }

        let n_classes = read_u32_le(&header[4..8]) as usize;
        let dim = read_u32_le(&header[8..12]) as usize;
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }

        let stride = packed_bytes_per_class(dim);
        // `checked_mul` / `checked_add` guard against an attacker-supplied
        // header that would overflow `usize` when computing the expected size.
        let expected_len = n_classes
            .checked_mul(stride)
            .and_then(|payload| payload.checked_add(HEADER_LEN))
            .ok_or(HdcError::DimensionMismatch {
                expected: usize::MAX,
                got: buf.len(),
            })?;
        if buf.len() != expected_len {
            return Err(HdcError::DimensionMismatch {
                expected: expected_len,
                got: buf.len(),
            });
        }

        let payload = &buf[HEADER_LEN..];
        let mut prototypes = Vec::with_capacity(n_classes);
        for chunk in payload.chunks(stride) {
            let mut proto = Vec::with_capacity(dim);
            for i in 0..dim {
                let byte = chunk[i / 8];
                let bit = 7 - (i % 8);
                let set = (byte >> bit) & 1 == 1;
                proto.push(if set { 1i8 } else { -1i8 });
            }
            prototypes.push(proto);
        }

        // Re-run full validation so the returned value satisfies every invariant.
        Self::new(prototypes)
    }

    /// Serialise the model to the human-readable text format.
    ///
    /// The first line is `HDC1 <n_classes> <dim>`; each following line encodes
    /// one prototype as `dim` characters of `'+'` (for `+1`) and `'-'` (for
    /// `-1`). The result round-trips exactly through [`HdModel::from_string_repr`].
    pub fn to_string_repr(&self) -> String {
        // Header line + one `dim`-char line per class, plus newlines.
        let mut out = String::with_capacity(32 + self.n_classes * (self.dim + 1));
        out.push_str(TEXT_MAGIC);
        out.push(' ');
        out.push_str(&self.n_classes.to_string());
        out.push(' ');
        out.push_str(&self.dim.to_string());
        for proto in &self.prototypes {
            out.push('\n');
            for &value in proto {
                out.push(if value == 1 { '+' } else { '-' });
            }
        }
        out
    }

    /// Deserialise a model from the human-readable text format.
    ///
    /// Trailing whitespace / a final newline is tolerated. Each prototype line
    /// must consist solely of `'+'`/`'-'` characters and have length `dim`.
    ///
    /// # Errors
    ///
    /// * [`HdcError::EmptyInput`] if the input is empty or has no header line.
    /// * [`HdcError::DimensionMismatch`] if the header line is malformed, the
    ///   number of prototype lines is not `n_classes`, or a prototype line has
    ///   the wrong length. `expected`/`got` carry the disagreeing counts.
    /// * [`HdcError::ZeroDimension`] if the header declares `dim == 0`.
    /// * [`HdcError::InvalidBinaryValue`] if a prototype line contains a
    ///   character other than `'+'` or `'-'`.
    pub fn from_string_repr(s: &str) -> HdcResult<Self> {
        // Trim only a trailing newline/whitespace block; interior structure is
        // delimited by '\n'. An all-whitespace / empty input is rejected.
        let trimmed = s.trim_end_matches(['\n', '\r', ' ', '\t']);
        if trimmed.is_empty() {
            return Err(HdcError::EmptyInput);
        }

        let mut lines = trimmed.split('\n');
        let header = lines.next().ok_or(HdcError::EmptyInput)?;

        // Parse "HDC1 <n_classes> <dim>".
        let mut fields = header.split_whitespace();
        let tag = fields.next().ok_or(HdcError::EmptyInput)?;
        if tag != TEXT_MAGIC {
            // Wrong magic token: report via DimensionMismatch on token length.
            return Err(HdcError::DimensionMismatch {
                expected: TEXT_MAGIC.len(),
                got: tag.len(),
            });
        }
        let n_classes = parse_usize_field(fields.next())?;
        let dim = parse_usize_field(fields.next())?;
        // Reject any extra tokens on the header line.
        if fields.next().is_some() {
            return Err(HdcError::DimensionMismatch {
                expected: 3,
                got: 4,
            });
        }
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }

        let mut prototypes = Vec::with_capacity(n_classes);
        for line in lines {
            // Reject trailing carriage returns that survived a CRLF split.
            let line = line.strip_suffix('\r').unwrap_or(line);
            if line.chars().count() != dim {
                return Err(HdcError::DimensionMismatch {
                    expected: dim,
                    got: line.chars().count(),
                });
            }
            let mut proto = Vec::with_capacity(dim);
            for ch in line.chars() {
                let value = match ch {
                    '+' => 1i8,
                    '-' => -1i8,
                    // Any other character is an invalid binary symbol. Map it to
                    // a non-±1 sentinel so the error type is InvalidBinaryValue.
                    _ => return Err(HdcError::InvalidBinaryValue(0)),
                };
                proto.push(value);
            }
            prototypes.push(proto);
        }

        if prototypes.len() != n_classes {
            return Err(HdcError::DimensionMismatch {
                expected: n_classes,
                got: prototypes.len(),
            });
        }

        Self::new(prototypes)
    }
}

/// Read a little-endian `u32` from a 4-byte slice, bounds-checked.
///
/// Returns `0` if `slice` is shorter than 4 bytes; callers in this module only
/// invoke it on slices already proven to be exactly 4 bytes long.
#[inline]
fn read_u32_le(slice: &[u8]) -> u32 {
    match slice.get(..4).and_then(|s| <[u8; 4]>::try_from(s).ok()) {
        Some(bytes) => u32::from_le_bytes(bytes),
        None => 0,
    }
}

/// Parse a decimal `usize` from an optional whitespace-delimited field.
///
/// # Errors
///
/// [`HdcError::DimensionMismatch`] if the field is missing or not a valid
/// non-negative integer (`expected = 0` flags a parse failure as the "wanted a
/// number" sentinel; `got` carries the field's character length, or `0` when
/// the field was absent entirely).
#[inline]
fn parse_usize_field(field: Option<&str>) -> HdcResult<usize> {
    match field {
        Some(text) => text
            .parse::<usize>()
            .map_err(|_| HdcError::DimensionMismatch {
                expected: 0,
                got: text.len(),
            }),
        None => Err(HdcError::DimensionMismatch {
            expected: 0,
            got: 0,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::vector::binary::random_binary;

    /// Build a deterministic model with `n_classes` random ±1 prototypes of `dim`.
    fn make_model(n_classes: usize, dim: usize, seed: u64) -> HdModel {
        let mut rng = LcgRng::new(seed);
        let mut prototypes = Vec::with_capacity(n_classes);
        for _ in 0..n_classes {
            prototypes.push(random_binary(dim, &mut rng).expect("random_binary"));
        }
        HdModel::new(prototypes).expect("model construction")
    }

    #[test]
    fn new_rejects_empty() {
        let err = HdModel::new(Vec::new()).unwrap_err();
        assert!(matches!(err, HdcError::EmptyInput));
    }

    #[test]
    fn new_rejects_ragged_rows() {
        let protos = vec![vec![1i8, -1, 1], vec![1i8, -1]];
        let err = HdModel::new(protos).unwrap_err();
        match err {
            HdcError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn new_rejects_bad_value() {
        let protos = vec![vec![1i8, 0, -1]];
        let err = HdModel::new(protos).unwrap_err();
        match err {
            HdcError::InvalidBinaryValue(v) => assert_eq!(v, 0),
            other => panic!("expected InvalidBinaryValue, got {other:?}"),
        }
    }

    #[test]
    fn new_rejects_zero_dim() {
        // A single empty prototype => dim 0.
        let err = HdModel::new(vec![Vec::new()]).unwrap_err();
        assert!(matches!(err, HdcError::ZeroDimension));
    }

    #[test]
    fn new_accepts_valid() {
        let model = make_model(4, 64, 7);
        assert_eq!(model.n_classes(), 4);
        assert_eq!(model.dim(), 64);
    }

    #[test]
    fn byte_round_trip_various_dims() {
        // Includes dims that are / are not multiples of 8, and edge dims 1 & 13.
        for &(n_classes, dim) in &[
            (1usize, 1usize),
            (3, 8),
            (5, 13),
            (2, 17),
            (4, 256),
            (7, 1000),
            (10, 999),
        ] {
            let model = make_model(n_classes, dim, 1000 + dim as u64);
            let bytes = model.to_bytes();
            let restored = HdModel::from_bytes(&bytes).expect("from_bytes");
            assert_eq!(
                model, restored,
                "round-trip failed for ({n_classes}, {dim})"
            );
        }
    }

    #[test]
    fn byte_len_matches_header_plus_payload() {
        let model = make_model(6, 13, 99);
        let bytes = model.to_bytes();
        // dim=13 => ceil(13/8) = 2 bytes/class.
        let expected = HEADER_LEN + 6 * 2;
        assert_eq!(bytes.len(), expected);
        assert_eq!(bytes.len(), model.byte_len());
    }

    #[test]
    fn header_fields_are_little_endian() {
        let model = make_model(2, 13, 5);
        let bytes = model.to_bytes();
        assert_eq!(&bytes[0..4], b"HDC1");
        assert_eq!(
            u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            2
        );
        assert_eq!(
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            13
        );
    }

    #[test]
    fn bit_packing_is_smaller_than_one_byte_per_dim() {
        // For dim > 8 the packed payload must be strictly smaller than `dim`
        // bytes per class (otherwise packing did nothing).
        let dim = 1000usize;
        let model = make_model(3, dim, 2);
        let stride = packed_bytes_per_class(dim);
        assert!(stride < dim, "stride {stride} not < dim {dim}");
        assert_eq!(stride, 125); // ceil(1000/8)
        // Whole-buffer payload check too.
        let payload = model.to_bytes().len() - HEADER_LEN;
        assert!(payload < 3 * dim);
    }

    #[test]
    fn from_bytes_rejects_empty() {
        let err = HdModel::from_bytes(&[]).unwrap_err();
        assert!(matches!(err, HdcError::EmptyInput));
    }

    #[test]
    fn from_bytes_rejects_bad_magic() {
        let model = make_model(2, 16, 3);
        let mut bytes = model.to_bytes();
        bytes[0] = b'X';
        let err = HdModel::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, HdcError::DimensionMismatch { .. }));
    }

    #[test]
    fn from_bytes_rejects_short_header() {
        let err = HdModel::from_bytes(&[0x48, 0x44, 0x43]).unwrap_err();
        match err {
            HdcError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, HEADER_LEN);
                assert_eq!(got, 3);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn from_bytes_rejects_truncated_payload() {
        let model = make_model(4, 32, 11);
        let mut bytes = model.to_bytes();
        let full = bytes.len();
        bytes.truncate(full - 1);
        let err = HdModel::from_bytes(&bytes).unwrap_err();
        match err {
            HdcError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, full);
                assert_eq!(got, full - 1);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn from_bytes_rejects_extra_payload() {
        let model = make_model(2, 24, 13);
        let mut bytes = model.to_bytes();
        bytes.push(0u8);
        let err = HdModel::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, HdcError::DimensionMismatch { .. }));
    }

    #[test]
    fn string_round_trip_various_dims() {
        for &(n_classes, dim) in &[(1usize, 1usize), (3, 13), (4, 100), (2, 1000)] {
            let model = make_model(n_classes, dim, 4242 + dim as u64);
            let text = model.to_string_repr();
            let restored = HdModel::from_string_repr(&text).expect("from_string_repr");
            assert_eq!(
                model, restored,
                "text round-trip failed for ({n_classes}, {dim})"
            );
        }
    }

    #[test]
    fn string_round_trip_tolerates_trailing_newline() {
        let model = make_model(3, 17, 21);
        let mut text = model.to_string_repr();
        text.push('\n');
        let restored = HdModel::from_string_repr(&text).expect("from_string_repr");
        assert_eq!(model, restored);
    }

    #[test]
    fn string_repr_shape_is_correct() {
        let model = make_model(3, 5, 1);
        let text = model.to_string_repr();
        let lines: Vec<&str> = text.split('\n').collect();
        assert_eq!(lines.len(), 4); // header + 3 classes
        assert_eq!(lines[0], "HDC1 3 5");
        for line in &lines[1..] {
            assert_eq!(line.len(), 5);
            assert!(line.chars().all(|c| c == '+' || c == '-'));
        }
    }

    #[test]
    fn from_string_repr_rejects_empty() {
        assert!(matches!(
            HdModel::from_string_repr("").unwrap_err(),
            HdcError::EmptyInput
        ));
        assert!(matches!(
            HdModel::from_string_repr("   \n  \n").unwrap_err(),
            HdcError::EmptyInput
        ));
    }

    #[test]
    fn from_string_repr_rejects_bad_magic() {
        let err = HdModel::from_string_repr("NOPE 1 3\n+++").unwrap_err();
        assert!(matches!(err, HdcError::DimensionMismatch { .. }));
    }

    #[test]
    fn from_string_repr_rejects_wrong_line_count() {
        // Header claims 3 classes but only 2 lines follow.
        let err = HdModel::from_string_repr("HDC1 3 4\n++--\n--++").unwrap_err();
        match err {
            HdcError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn from_string_repr_rejects_wrong_line_length() {
        // dim=4 but the (only) prototype line has length 3.
        let err = HdModel::from_string_repr("HDC1 1 4\n+++").unwrap_err();
        match err {
            HdcError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, 4);
                assert_eq!(got, 3);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn from_string_repr_rejects_bad_char() {
        let err = HdModel::from_string_repr("HDC1 1 4\n++x-").unwrap_err();
        assert!(matches!(err, HdcError::InvalidBinaryValue(_)));
    }

    #[test]
    fn from_string_repr_rejects_non_numeric_header() {
        let err = HdModel::from_string_repr("HDC1 two 4\n++--").unwrap_err();
        assert!(matches!(err, HdcError::DimensionMismatch { .. }));
    }

    #[test]
    fn prototype_returns_slices_and_class_not_found() {
        let model = make_model(3, 16, 8);
        for class in 0..3 {
            let proto = model.prototype(class).expect("prototype");
            assert_eq!(proto.len(), 16);
            validate_binary(proto).expect("valid");
        }
        let err = model.prototype(3).unwrap_err();
        assert!(matches!(err, HdcError::ClassNotFound(3)));
    }

    #[test]
    fn known_vector_bit_packing_msb_first() {
        // dim=10, single class. Pattern: +,-,+,-,+,-,+,-, +,- => bits 1010_1010 10
        // Byte 0 MSB-first = 0b1010_1010 = 0xAA; byte 1 = 0b1000_0000 = 0x80
        // (the two used bits are +,- => 1,0 in the two most-significant positions).
        let proto = vec![1i8, -1, 1, -1, 1, -1, 1, -1, 1, -1];
        let model = HdModel::new(vec![proto]).expect("model");
        let bytes = model.to_bytes();
        assert_eq!(bytes.len(), HEADER_LEN + 2);
        assert_eq!(bytes[HEADER_LEN], 0xAA);
        assert_eq!(bytes[HEADER_LEN + 1], 0x80);
        // And it round-trips.
        let restored = HdModel::from_bytes(&bytes).expect("from_bytes");
        assert_eq!(model, restored);
    }

    #[test]
    fn padding_bits_are_zero_and_ignored() {
        // dim=1, value +1 => byte 0b1000_0000 = 0x80, low 7 bits are padding 0.
        let model = HdModel::new(vec![vec![1i8]]).expect("model");
        let bytes = model.to_bytes();
        assert_eq!(bytes.len(), HEADER_LEN + 1);
        assert_eq!(bytes[HEADER_LEN], 0x80);
        // Flipping a padding bit must NOT change the decoded model (length is
        // still valid, and only the top bit is read for dim=1).
        let mut tampered = bytes.clone();
        tampered[HEADER_LEN] |= 0b0000_0001;
        let restored = HdModel::from_bytes(&tampered).expect("from_bytes");
        assert_eq!(model, restored);
    }
}
