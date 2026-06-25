//! Foundation-model adapter / weight-loading interface.
//!
//! Foundation forecasters such as Moirai and Chronos are intended to be loaded
//! from *pretrained* checkpoints rather than trained from scratch. This module
//! provides a pure-Rust, dependency-free checkpoint interface so a forecaster's
//! learnable tensors can be exported to and re-imported from a compact,
//! self-describing byte buffer.
//!
//! # Format ([`WeightStore`] binary layout)
//!
//! ```text
//!   magic   : 4 bytes  = b"OXTS"
//!   version : u32 LE   = 1
//!   n_tensors: u32 LE
//!   repeat n_tensors times:
//!     name_len : u32 LE
//!     name     : name_len UTF-8 bytes
//!     numel    : u32 LE
//!     data     : numel × f32 LE
//!   checksum : u32 LE  (FNV-1a over everything preceding)
//! ```
//!
//! The format is deterministic and endian-explicit (little-endian) so a buffer
//! written on one machine round-trips bit-exactly on another. Foundation models
//! implement the [`FoundationAdapter`] trait to expose their tensors by name.

use crate::error::{TsError, TsResult};
use std::collections::BTreeMap;

/// Magic bytes prefixing every [`WeightStore`] buffer.
pub const STORE_MAGIC: [u8; 4] = *b"OXTS";

/// Current [`WeightStore`] serialisation version.
pub const STORE_VERSION: u32 = 1;

/// A named collection of float tensors representing a model checkpoint.
///
/// Tensors are stored flat (`Vec<f32>`) keyed by name. A [`BTreeMap`] keeps the
/// key order deterministic, so serialisation is reproducible regardless of
/// insertion order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WeightStore {
    tensors: BTreeMap<String, Vec<f32>>,
}

impl WeightStore {
    /// Create an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tensors: BTreeMap::new(),
        }
    }

    /// Number of tensors held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the store holds no tensors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Insert (or overwrite) a named tensor.
    pub fn insert(&mut self, name: impl Into<String>, data: Vec<f32>) {
        self.tensors.insert(name.into(), data);
    }

    /// Borrow a tensor by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.tensors.get(name).map(Vec::as_slice)
    }

    /// Fetch a required tensor, erroring if absent.
    ///
    /// # Errors
    ///
    /// - [`TsError::Internal`] when `name` is not present.
    pub fn require(&self, name: &str) -> TsResult<&[f32]> {
        self.tensors
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| TsError::Internal(format!("missing tensor '{name}'")))
    }

    /// Fetch a required tensor and verify its length, erroring otherwise.
    ///
    /// # Errors
    ///
    /// - [`TsError::Internal`] when `name` is absent.
    /// - [`TsError::WeightShapeMismatch`] when the length differs from `expected`.
    pub fn require_len(&self, name: &str, expected: usize) -> TsResult<&[f32]> {
        let t = self.require(name)?;
        if t.len() != expected {
            return Err(TsError::WeightShapeMismatch {
                msg: format!("tensor '{name}' len {} != expected {expected}", t.len()),
            });
        }
        Ok(t)
    }

    /// Iterate over `(name, data)` pairs in deterministic key order.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Vec<f32>)> {
        self.tensors.iter()
    }

    /// Serialise the store to a byte buffer (see module docs for the layout).
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&STORE_MAGIC);
        out.extend_from_slice(&STORE_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u32).to_le_bytes());
        for (name, data) in &self.tensors {
            let name_bytes = name.as_bytes();
            out.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(name_bytes);
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            for &v in data {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        let checksum = fnv1a(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        out
    }

    /// Deserialise a store from a byte buffer produced by [`Self::to_bytes`].
    ///
    /// # Errors
    ///
    /// - [`TsError::Internal`] on bad magic, version, truncation, or checksum
    ///   mismatch.
    pub fn from_bytes(buf: &[u8]) -> TsResult<Self> {
        let mut cur = Cursor::new(buf);

        let magic = cur.take(4)?;
        if magic != STORE_MAGIC {
            return Err(TsError::Internal("bad WeightStore magic".to_string()));
        }
        let version = cur.read_u32()?;
        if version != STORE_VERSION {
            return Err(TsError::Internal(format!(
                "unsupported WeightStore version {version}"
            )));
        }
        let n_tensors = cur.read_u32()? as usize;

        let mut tensors = BTreeMap::new();
        for _ in 0..n_tensors {
            let name_len = cur.read_u32()? as usize;
            let name_bytes = cur.take(name_len)?;
            let name = String::from_utf8(name_bytes.to_vec())
                .map_err(|_| TsError::Internal("non-UTF8 tensor name".to_string()))?;
            let numel = cur.read_u32()? as usize;
            let mut data = Vec::with_capacity(numel);
            for _ in 0..numel {
                data.push(cur.read_f32()?);
            }
            tensors.insert(name, data);
        }

        // Verify checksum (computed over the buffer up to the checksum field).
        let consumed = cur.pos();
        let stored_checksum = cur.read_u32()?;
        let computed = fnv1a(&buf[..consumed]);
        if stored_checksum != computed {
            return Err(TsError::Internal(format!(
                "WeightStore checksum mismatch: stored {stored_checksum:#x} computed {computed:#x}"
            )));
        }

        Ok(Self { tensors })
    }
}

/// Trait implemented by foundation forecasters that support checkpoint
/// export / import via a [`WeightStore`].
pub trait FoundationAdapter: Sized {
    /// Export all learnable tensors into a fresh [`WeightStore`].
    fn export_weights(&self) -> WeightStore;

    /// Overwrite this model's learnable tensors from `store`.
    ///
    /// The architecture (layer count, dimensions) is fixed by the model's
    /// existing configuration; only the tensor *values* are replaced. A tensor
    /// whose length does not match the model's expectation is an error.
    ///
    /// # Errors
    ///
    /// Returns an error if a required tensor is missing or has the wrong length.
    fn import_weights(&mut self, store: &WeightStore) -> TsResult<()>;

    /// Convenience: serialise the model's weights to a byte buffer.
    #[must_use]
    fn to_checkpoint(&self) -> Vec<u8> {
        self.export_weights().to_bytes()
    }

    /// Convenience: load weights from a serialised checkpoint buffer.
    ///
    /// # Errors
    ///
    /// Propagates [`WeightStore::from_bytes`] and [`Self::import_weights`] errors.
    fn load_checkpoint(&mut self, buf: &[u8]) -> TsResult<()> {
        let store = WeightStore::from_bytes(buf)?;
        self.import_weights(&store)
    }
}

// ─── FNV-1a checksum ────────────────────────────────────────────────────────

/// 32-bit FNV-1a hash for buffer integrity checking.
fn fnv1a(data: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for &b in data {
        hash ^= b as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

// ─── Minimal little-endian cursor ───────────────────────────────────────────

/// A tiny bounds-checked little-endian reader over a byte slice.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn take(&mut self, n: usize) -> TsResult<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(TsError::Internal(
                "WeightStore buffer truncated".to_string(),
            ));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn read_u32(&mut self) -> TsResult<u32> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn read_f32(&mut self) -> TsResult<f32> {
        let s = self.take(4)?;
        Ok(f32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_roundtrip_bit_exact() {
        let mut s = WeightStore::new();
        s.insert("layer0.q_w", vec![1.0, -2.5, 3.25, 4.0]);
        s.insert("layer0.k_w", vec![0.0; 8]);
        s.insert("head_b", vec![std::f32::consts::PI, -1.0]);

        let bytes = s.to_bytes();
        let restored = WeightStore::from_bytes(&bytes).expect("decode");
        assert_eq!(s, restored);
        assert_eq!(restored.require("head_b").expect("head_b").len(), 2);
    }

    #[test]
    fn store_deterministic_regardless_of_insert_order() {
        let mut a = WeightStore::new();
        a.insert("b", vec![1.0]);
        a.insert("a", vec![2.0]);
        a.insert("c", vec![3.0]);

        let mut b = WeightStore::new();
        b.insert("c", vec![3.0]);
        b.insert("a", vec![2.0]);
        b.insert("b", vec![1.0]);

        assert_eq!(a.to_bytes(), b.to_bytes());
    }

    #[test]
    fn store_magic_and_version_present() {
        let s = WeightStore::new();
        let bytes = s.to_bytes();
        assert_eq!(&bytes[0..4], &STORE_MAGIC);
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(version, STORE_VERSION);
    }

    #[test]
    fn store_detects_corruption() {
        let mut s = WeightStore::new();
        s.insert("w", vec![1.0, 2.0, 3.0]);
        let mut bytes = s.to_bytes();
        // Flip a byte in the tensor data (not the checksum trailer).
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        assert!(matches!(
            WeightStore::from_bytes(&bytes).unwrap_err(),
            TsError::Internal(_)
        ));
    }

    #[test]
    fn store_rejects_bad_magic() {
        let mut bytes = WeightStore::new().to_bytes();
        bytes[0] = b'X';
        assert!(matches!(
            WeightStore::from_bytes(&bytes).unwrap_err(),
            TsError::Internal(_)
        ));
    }

    #[test]
    fn store_rejects_truncation() {
        let mut s = WeightStore::new();
        s.insert("w", vec![1.0, 2.0]);
        let bytes = s.to_bytes();
        let truncated = &bytes[..bytes.len() - 6];
        assert!(WeightStore::from_bytes(truncated).is_err());
    }

    #[test]
    fn store_require_len_checks_shape() {
        let mut s = WeightStore::new();
        s.insert("w", vec![1.0, 2.0, 3.0]);
        assert!(s.require_len("w", 3).is_ok());
        assert!(matches!(
            s.require_len("w", 4).unwrap_err(),
            TsError::WeightShapeMismatch { .. }
        ));
    }

    #[test]
    fn store_require_missing_errors() {
        let s = WeightStore::new();
        assert!(matches!(
            s.require("nope").unwrap_err(),
            TsError::Internal(_)
        ));
    }

    #[test]
    fn store_empty_roundtrip() {
        let s = WeightStore::new();
        assert!(s.is_empty());
        let bytes = s.to_bytes();
        let restored = WeightStore::from_bytes(&bytes).expect("decode");
        assert!(restored.is_empty());
        assert_eq!(restored.len(), 0);
    }

    #[test]
    fn fnv1a_known_vector() {
        // FNV-1a("") = 0x811c9dc5; FNV-1a("a") = 0xe40c292c.
        assert_eq!(fnv1a(b""), 0x811c_9dc5);
        assert_eq!(fnv1a(b"a"), 0xe40c_292c);
    }

    #[test]
    fn store_preserves_special_floats() {
        let mut s = WeightStore::new();
        s.insert("x", vec![0.0, -0.0, 1e-30, 1e30]);
        let restored = WeightStore::from_bytes(&s.to_bytes()).expect("decode");
        let x = restored.require("x").expect("x");
        assert_eq!(x[0].to_bits(), 0.0_f32.to_bits());
        assert_eq!(x[1].to_bits(), (-0.0_f32).to_bits());
        assert_eq!(x[2], 1e-30);
        assert_eq!(x[3], 1e30);
    }
}
