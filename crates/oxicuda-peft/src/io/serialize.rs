//! Pure-Rust adapter serialization — save / load PEFT adapter weights as a
//! self-describing little-endian byte stream.
//!
//! No `serde`, no `bincode`, no `zip`: the container is hand-rolled so the crate keeps its
//! "pure Rust, two dependencies" footprint. The format is a flat collection of *named tensors*
//! (`name → Vec<f32>`), which is exactly what every adapter in this crate decomposes into
//! (LoRA `A`/`B`, DoRA magnitude/direction, VeRA scaling vectors, prompt embeddings, …).
//!
//! # Wire format
//!
//! ```text
//!   offset  bytes  field
//!   0       4      magic            = b"OXPA"  (OxiCUDA-Peft-Adapter)
//!   4       4      format_version   = u32 LE   (current = 1)
//!   8       4      tensor_count     = u32 LE
//!   ...            tensor_count × tensor records
//!   tail    8      checksum         = u64 LE   (FNV-1a over every byte before it)
//! ```
//!
//! Each tensor record is:
//!
//! ```text
//!   4   name_len  : u32 LE
//!   N   name      : UTF-8 bytes (N = name_len)
//!   4   elem_len  : u32 LE        (number of f32 values)
//!   4·M data      : f32 LE  (M = elem_len, each value as IEEE-754 little-endian bits)
//! ```
//!
//! Loading validates the magic, the version, every length against the remaining buffer, and
//! the trailing checksum, returning a [`crate::error::PeftError`] instead of panicking on any malformed input.

use crate::error::{PeftError, PeftResult};
use std::collections::BTreeMap;
use std::path::Path;

/// Magic bytes prefixing every adapter container: `OXPA` = OxiCUDA-Peft-Adapter.
pub const MAGIC: [u8; 4] = *b"OXPA";

/// Highest container format version this build can read and the version it writes.
pub const FORMAT_VERSION: u32 = 1;

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Compute the FNV-1a 64-bit hash of a byte slice (used as the container checksum).
#[must_use]
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// A named collection of `f32` tensors representing a single adapter's trainable state.
///
/// Insertion order is irrelevant; tensors are stored in a [`BTreeMap`] so serialization is
/// deterministic (lexicographic by name) regardless of the order they were added.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AdapterPayload {
    tensors: BTreeMap<String, Vec<f32>>,
}

impl AdapterPayload {
    /// Construct an empty payload.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tensors: BTreeMap::new(),
        }
    }

    /// Insert (or overwrite) a named tensor, returning `self` for chaining.
    #[must_use]
    pub fn with_tensor(mut self, name: impl Into<String>, data: Vec<f32>) -> Self {
        self.tensors.insert(name.into(), data);
        self
    }

    /// Insert (or overwrite) a named tensor in place.
    pub fn insert(&mut self, name: impl Into<String>, data: Vec<f32>) {
        self.tensors.insert(name.into(), data);
    }

    /// Borrow a tensor by name, if present.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.tensors.get(name).map(Vec::as_slice)
    }

    /// Number of tensors stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the payload holds no tensors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Iterate over `(name, tensor)` pairs in deterministic (lexicographic) order.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Vec<f32>)> {
        self.tensors.iter()
    }

    /// Total number of `f32` scalars across every tensor.
    #[must_use]
    pub fn total_scalars(&self) -> usize {
        self.tensors.values().map(Vec::len).sum()
    }

    /// Serialize the payload to a self-describing byte vector (see module docs for the layout).
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        // Header (4) + version (4) + count (4) + body, finally + checksum (8).
        let mut out = Vec::with_capacity(16 + self.total_scalars() * 4);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        // `tensor_count` is bounded by the map size; saturate rather than truncate silently.
        let count = u32::try_from(self.tensors.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&count.to_le_bytes());
        for (name, data) in &self.tensors {
            let name_len = u32::try_from(name.len()).unwrap_or(u32::MAX);
            out.extend_from_slice(&name_len.to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            let elem_len = u32::try_from(data.len()).unwrap_or(u32::MAX);
            out.extend_from_slice(&elem_len.to_le_bytes());
            for &v in data {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        let checksum = fnv1a(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        out
    }

    /// Reconstruct a payload from bytes produced by [`Self::to_bytes`].
    ///
    /// # Errors
    ///
    /// - [`PeftError::CorruptData`] when the buffer is too short, the magic is wrong, a length
    ///   field overruns the buffer, a tensor name is not valid UTF-8, or the checksum mismatches.
    /// - [`PeftError::UnsupportedVersion`] when the stored format version exceeds
    ///   [`FORMAT_VERSION`].
    pub fn from_bytes(bytes: &[u8]) -> PeftResult<Self> {
        // The smallest legal container is magic + version + count + checksum = 20 bytes.
        if bytes.len() < 20 {
            return Err(PeftError::CorruptData {
                msg: format!("buffer too short: {} bytes", bytes.len()),
            });
        }
        let mut cur = Cursor::new(bytes);
        let magic = cur.read_array4()?;
        if magic != MAGIC {
            return Err(PeftError::CorruptData {
                msg: "bad magic header".to_string(),
            });
        }
        let version = cur.read_u32()?;
        if version > FORMAT_VERSION {
            return Err(PeftError::UnsupportedVersion {
                found: version,
                supported: FORMAT_VERSION,
            });
        }
        let count = cur.read_u32()? as usize;
        let mut tensors = BTreeMap::new();
        for _ in 0..count {
            let name_len = cur.read_u32()? as usize;
            let name = cur.read_utf8(name_len)?;
            let elem_len = cur.read_u32()? as usize;
            let mut data = Vec::with_capacity(elem_len);
            for _ in 0..elem_len {
                data.push(cur.read_f32()?);
            }
            tensors.insert(name, data);
        }
        // Everything before the trailing 8-byte checksum must hash to that checksum.
        let body_end = cur.pos;
        if body_end + 8 > bytes.len() {
            return Err(PeftError::CorruptData {
                msg: "missing trailing checksum".to_string(),
            });
        }
        let stored = u64::from_le_bytes([
            bytes[body_end],
            bytes[body_end + 1],
            bytes[body_end + 2],
            bytes[body_end + 3],
            bytes[body_end + 4],
            bytes[body_end + 5],
            bytes[body_end + 6],
            bytes[body_end + 7],
        ]);
        let computed = fnv1a(&bytes[..body_end]);
        if stored != computed {
            return Err(PeftError::CorruptData {
                msg: format!("checksum mismatch: stored {stored:#x} computed {computed:#x}"),
            });
        }
        Ok(Self { tensors })
    }

    /// Serialize and write the payload to `path`, overwriting any existing file.
    ///
    /// # Errors
    ///
    /// [`PeftError::Io`] when the file cannot be written.
    pub fn save_to_file(&self, path: &Path) -> PeftResult<()> {
        std::fs::write(path, self.to_bytes()).map_err(|e| PeftError::Io { msg: e.to_string() })
    }

    /// Read and deserialize a payload from `path`.
    ///
    /// # Errors
    ///
    /// - [`PeftError::Io`] when the file cannot be read.
    /// - Any error from [`Self::from_bytes`] when the contents are malformed.
    pub fn load_from_file(path: &Path) -> PeftResult<Self> {
        let bytes = std::fs::read(path).map_err(|e| PeftError::Io { msg: e.to_string() })?;
        Self::from_bytes(&bytes)
    }
}

/// A minimal forward-only cursor that bounds-checks every read against the backing slice.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Borrow the next `n` bytes, advancing the cursor, or fail if they run past the end.
    fn take(&mut self, n: usize) -> PeftResult<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| PeftError::CorruptData {
                msg: "length overflow".to_string(),
            })?;
        if end > self.buf.len() {
            return Err(PeftError::CorruptData {
                msg: format!(
                    "read of {n} bytes at offset {} overruns buffer of {}",
                    self.pos,
                    self.buf.len()
                ),
            });
        }
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_array4(&mut self) -> PeftResult<[u8; 4]> {
        let s = self.take(4)?;
        Ok([s[0], s[1], s[2], s[3]])
    }

    fn read_u32(&mut self) -> PeftResult<u32> {
        Ok(u32::from_le_bytes(self.read_array4()?))
    }

    fn read_f32(&mut self) -> PeftResult<f32> {
        Ok(f32::from_le_bytes(self.read_array4()?))
    }

    fn read_utf8(&mut self, n: usize) -> PeftResult<String> {
        let bytes = self.take(n)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| PeftError::CorruptData {
            msg: format!("invalid UTF-8 in tensor name: {e}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> AdapterPayload {
        AdapterPayload::new()
            .with_tensor("lora.A", vec![0.1, -0.2, 0.3, 0.4])
            .with_tensor("lora.B", vec![0.0, 0.0, 0.0])
            .with_tensor("scale", vec![2.0])
    }

    #[test]
    fn roundtrip_bytes_is_bit_exact() {
        let p = sample();
        let bytes = p.to_bytes();
        let back = AdapterPayload::from_bytes(&bytes).expect("valid container must decode");
        assert_eq!(p, back);
        // Spot-check exact values survived the f32 LE encoding.
        assert_eq!(
            back.get("lora.A"),
            Some([0.1_f32, -0.2, 0.3, 0.4].as_slice())
        );
        assert_eq!(back.get("scale"), Some([2.0_f32].as_slice()));
    }

    #[test]
    fn header_is_well_formed() {
        let bytes = sample().to_bytes();
        assert_eq!(&bytes[0..4], &MAGIC);
        assert_eq!(
            u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            FORMAT_VERSION
        );
        assert_eq!(
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            3,
            "three tensors were stored"
        );
    }

    #[test]
    fn serialization_is_deterministic_regardless_of_insertion_order() {
        let mut a = AdapterPayload::new();
        a.insert("z", vec![1.0]);
        a.insert("a", vec![2.0]);
        let mut b = AdapterPayload::new();
        b.insert("a", vec![2.0]);
        b.insert("z", vec![1.0]);
        assert_eq!(a.to_bytes(), b.to_bytes());
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut bytes = sample().to_bytes();
        bytes[0] = b'X';
        let err = AdapterPayload::from_bytes(&bytes).expect_err("bad magic must fail");
        assert!(matches!(err, PeftError::CorruptData { .. }));
    }

    #[test]
    fn flipped_payload_byte_fails_checksum() {
        let mut bytes = sample().to_bytes();
        // Corrupt a value byte well inside the payload (after the 12-byte header).
        let idx = 20;
        bytes[idx] ^= 0xFF;
        let err = AdapterPayload::from_bytes(&bytes).expect_err("checksum must catch corruption");
        assert!(matches!(err, PeftError::CorruptData { .. }));
    }

    #[test]
    fn truncated_buffer_is_rejected() {
        let bytes = sample().to_bytes();
        let err = AdapterPayload::from_bytes(&bytes[..bytes.len() - 4])
            .expect_err("truncation must fail");
        assert!(matches!(err, PeftError::CorruptData { .. }));
    }

    #[test]
    fn future_version_is_rejected() {
        let mut bytes = sample().to_bytes();
        // Bump the stored version above what we support, then repair the checksum so the
        // version gate (not the checksum) is what trips.
        let future = (FORMAT_VERSION + 1).to_le_bytes();
        bytes[4..8].copy_from_slice(&future);
        let body_end = bytes.len() - 8;
        let new_sum = fnv1a(&bytes[..body_end]);
        bytes[body_end..].copy_from_slice(&new_sum.to_le_bytes());
        let err = AdapterPayload::from_bytes(&bytes).expect_err("future version must fail");
        assert!(matches!(
            err,
            PeftError::UnsupportedVersion {
                found,
                supported
            } if found == FORMAT_VERSION + 1 && supported == FORMAT_VERSION
        ));
    }

    #[test]
    fn empty_payload_roundtrips() {
        let p = AdapterPayload::new();
        let back = AdapterPayload::from_bytes(&p.to_bytes()).expect("empty must roundtrip");
        assert!(back.is_empty());
        assert_eq!(back.len(), 0);
        assert_eq!(back.total_scalars(), 0);
    }

    #[test]
    fn file_roundtrip_via_temp_dir() {
        let p = sample();
        let mut path = std::env::temp_dir();
        path.push(format!(
            "oxicuda_peft_serialize_{}.oxpa",
            std::process::id()
        ));
        p.save_to_file(&path).expect("save must succeed");
        let back = AdapterPayload::load_from_file(&path).expect("load must succeed");
        assert_eq!(p, back);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_missing_file_is_io_error() {
        let mut path = std::env::temp_dir();
        path.push("oxicuda_peft_definitely_missing_file.oxpa");
        let _ = std::fs::remove_file(&path);
        let err = AdapterPayload::load_from_file(&path).expect_err("missing file must error");
        assert!(matches!(err, PeftError::Io { .. }));
    }
}
