//! Pure-Rust binary serialisation / deserialisation for sketch persistence.
//!
//! Provides a compact, self-describing little-endian byte format for the core
//! sketches so they can be written to disk, sent over the wire, or merged
//! across processes without any external dependency. The format is *not* the
//! Apache DataSketches wire format (see [`crate::topk::fi_serde`] for a
//! DataSketches-FI-compatible byte layout of the frequent-items sketch); it is
//! a stable internal format for OxiCUDA.
//!
//! ## Frame layout
//!
//! Every serialised blob begins with a fixed header:
//!
//! ```text
//!   bytes 0..4   magic   = b"OXSK"
//!   byte  4      version = 1
//!   byte  5      kind    (SketchKind discriminant)
//!   bytes 6..8   reserved (zero)
//! ```
//!
//! followed by a per-kind payload of fixed-width little-endian integers. Each
//! supported sketch implements [`SketchSerialize`] to read/write its own payload.
//! Round-tripping is exact: `from_bytes(to_bytes(s)) == s` cell-for-cell.

use crate::cardinality::hll::HyperLogLog;
use crate::error::{SketchError, SketchResult};
use crate::frequency::count_min::CountMinSketch;
use crate::hash::twouniv::TwoUniversal;
use crate::membership::bloom::BloomFilter;

/// Magic prefix identifying an OxiCUDA sketch blob.
pub const MAGIC: [u8; 4] = *b"OXSK";
/// Current serialisation format version.
pub const VERSION: u8 = 1;
/// Header length in bytes.
pub const HEADER_LEN: usize = 8;

/// Discriminant byte tagging the sketch kind inside a serialised frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SketchKind {
    /// [`HyperLogLog`].
    HyperLogLog = 1,
    /// [`CountMinSketch`].
    CountMin = 2,
    /// [`BloomFilter`].
    Bloom = 3,
}

impl SketchKind {
    fn from_u8(b: u8) -> SketchResult<Self> {
        match b {
            1 => Ok(SketchKind::HyperLogLog),
            2 => Ok(SketchKind::CountMin),
            3 => Ok(SketchKind::Bloom),
            other => Err(SketchError::InvalidParameter {
                name: "kind".to_string(),
                reason: format!("unknown sketch kind discriminant {other}"),
            }),
        }
    }
}

/// Trait for sketches that support exact binary round-tripping.
pub trait SketchSerialize: Sized {
    /// The kind tag written into the frame header.
    const KIND: SketchKind;

    /// Append this sketch's payload (everything after the 8-byte header) to `out`.
    fn write_payload(&self, out: &mut Vec<u8>);

    /// Parse a payload (the bytes *after* the header) back into a sketch.
    fn read_payload(payload: &[u8]) -> SketchResult<Self>;

    /// Serialise to a self-describing byte vector (header + payload).
    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN + 16);
        out.extend_from_slice(&MAGIC);
        out.push(VERSION);
        out.push(Self::KIND as u8);
        out.push(0);
        out.push(0);
        self.write_payload(&mut out);
        out
    }

    /// Deserialise from a byte slice, validating the header.
    fn from_bytes(bytes: &[u8]) -> SketchResult<Self> {
        if bytes.len() < HEADER_LEN {
            return Err(SketchError::InvalidParameter {
                name: "bytes".to_string(),
                reason: format!("truncated header: need {HEADER_LEN}, got {}", bytes.len()),
            });
        }
        if bytes[0..4] != MAGIC {
            return Err(SketchError::InvalidParameter {
                name: "magic".to_string(),
                reason: "bad magic prefix (not an OxiCUDA sketch blob)".to_string(),
            });
        }
        if bytes[4] != VERSION {
            return Err(SketchError::InvalidParameter {
                name: "version".to_string(),
                reason: format!("unsupported version {} (expected {VERSION})", bytes[4]),
            });
        }
        let kind = SketchKind::from_u8(bytes[5])?;
        if kind != Self::KIND {
            return Err(SketchError::InvalidParameter {
                name: "kind".to_string(),
                reason: format!("frame kind {:?} != expected {:?}", kind, Self::KIND),
            });
        }
        Self::read_payload(&bytes[HEADER_LEN..])
    }
}

/// Cursor-style little-endian reader with bounds checks (no panics).
struct ByteReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn need(&self, n: usize) -> SketchResult<()> {
        if self.pos + n > self.buf.len() {
            return Err(SketchError::InvalidParameter {
                name: "payload".to_string(),
                reason: format!(
                    "truncated payload: need {n} bytes at offset {}, have {}",
                    self.pos,
                    self.buf.len()
                ),
            });
        }
        Ok(())
    }

    fn u8(&mut self) -> SketchResult<u8> {
        self.need(1)?;
        let v = self.buf[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn u32(&mut self) -> SketchResult<u32> {
        self.need(4)?;
        let mut b = [0u8; 4];
        b.copy_from_slice(&self.buf[self.pos..self.pos + 4]);
        self.pos += 4;
        Ok(u32::from_le_bytes(b))
    }

    fn u64(&mut self) -> SketchResult<u64> {
        self.need(8)?;
        let mut b = [0u8; 8];
        b.copy_from_slice(&self.buf[self.pos..self.pos + 8]);
        self.pos += 8;
        Ok(u64::from_le_bytes(b))
    }
}

fn write_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

impl SketchSerialize for HyperLogLog {
    const KIND: SketchKind = SketchKind::HyperLogLog;

    fn write_payload(&self, out: &mut Vec<u8>) {
        write_u32(out, self.p);
        write_u64(out, self.seed);
        write_u64(out, self.registers.len() as u64);
        out.extend_from_slice(&self.registers);
    }

    fn read_payload(payload: &[u8]) -> SketchResult<Self> {
        let mut r = ByteReader::new(payload);
        let p = r.u32()?;
        let seed = r.u64()?;
        let n = r.u64()? as usize;
        let mut hll = HyperLogLog::new(p, seed)?;
        if hll.registers.len() != n {
            return Err(SketchError::ShapeMismatch {
                expected: vec![hll.registers.len()],
                got: vec![n],
            });
        }
        for reg in hll.registers.iter_mut() {
            *reg = r.u8()?;
        }
        Ok(hll)
    }
}

impl SketchSerialize for CountMinSketch {
    const KIND: SketchKind = SketchKind::CountMin;

    fn write_payload(&self, out: &mut Vec<u8>) {
        write_u64(out, self.d as u64);
        write_u64(out, self.w as u64);
        // Hash coefficients (a, b) per row, so the deserialised sketch hashes
        // keys identically.
        for h in &self.hashes {
            write_u64(out, h.a);
            write_u64(out, h.b);
        }
        for &c in &self.table {
            write_u64(out, c);
        }
    }

    fn read_payload(payload: &[u8]) -> SketchResult<Self> {
        let mut r = ByteReader::new(payload);
        let d = r.u64()? as usize;
        let w = r.u64()? as usize;
        if d == 0 || w == 0 {
            return Err(SketchError::InvalidParameter {
                name: "(d,w)".to_string(),
                reason: "deserialised dimensions must be positive".to_string(),
            });
        }
        let mut hashes = Vec::with_capacity(d);
        for _ in 0..d {
            let a = r.u64()?;
            let b = r.u64()?;
            hashes.push(TwoUniversal::with_coeffs(a, b, w as u64));
        }
        let mut table = vec![0u64; d * w];
        for cell in table.iter_mut() {
            *cell = r.u64()?;
        }
        Ok(CountMinSketch {
            d,
            w,
            table,
            hashes,
        })
    }
}

impl SketchSerialize for BloomFilter {
    const KIND: SketchKind = SketchKind::Bloom;

    fn write_payload(&self, out: &mut Vec<u8>) {
        write_u64(out, self.m as u64);
        write_u64(out, self.k as u64);
        write_u64(out, self.seed_base);
        write_u64(out, self.bits.len() as u64);
        for &word in &self.bits {
            write_u64(out, word);
        }
    }

    fn read_payload(payload: &[u8]) -> SketchResult<Self> {
        let mut r = ByteReader::new(payload);
        let m = r.u64()? as usize;
        let k = r.u64()? as usize;
        let seed_base = r.u64()?;
        let n_words = r.u64()? as usize;
        let mut bf = BloomFilter::new(m, k, seed_base)?;
        if bf.bits.len() != n_words {
            return Err(SketchError::ShapeMismatch {
                expected: vec![bf.bits.len()],
                got: vec![n_words],
            });
        }
        for word in bf.bits.iter_mut() {
            *word = r.u64()?;
        }
        Ok(bf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn hll_roundtrip_exact() {
        let mut h = HyperLogLog::new(12, 999).expect("ok");
        for i in 0..5000u64 {
            h.add_u64(i);
        }
        let bytes = h.to_bytes();
        assert_eq!(&bytes[0..4], &MAGIC);
        assert_eq!(bytes[5], SketchKind::HyperLogLog as u8);
        let back = HyperLogLog::from_bytes(&bytes).expect("decode ok");
        assert_eq!(back.p, h.p);
        assert_eq!(back.seed, h.seed);
        assert_eq!(back.registers, h.registers);
        assert!((back.estimate() - h.estimate()).abs() < 1e-9);
    }

    #[test]
    fn count_min_roundtrip_preserves_queries() {
        let mut rng = LcgRng::new(7);
        let mut cm = CountMinSketch::new(5, 512, &mut rng).expect("ok");
        for i in 0..2000u64 {
            cm.update(i % 100, 1);
        }
        let bytes = cm.to_bytes();
        let back = CountMinSketch::from_bytes(&bytes).expect("decode ok");
        assert_eq!(back.d, cm.d);
        assert_eq!(back.w, cm.w);
        assert_eq!(back.table, cm.table);
        // Identical hash coefficients ⇒ identical query answers.
        for key in 0..100u64 {
            assert_eq!(back.query(key), cm.query(key), "query mismatch for {key}");
        }
    }

    #[test]
    fn bloom_roundtrip_membership() {
        let mut bf = BloomFilter::new(4096, 5, 31).expect("ok");
        for i in 0..500u64 {
            bf.insert(i);
        }
        let bytes = bf.to_bytes();
        let back = BloomFilter::from_bytes(&bytes).expect("decode ok");
        assert_eq!(back.bits, bf.bits);
        for i in 0..500u64 {
            assert!(back.contains(i), "lost membership for {i} after round-trip");
        }
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = HyperLogLog::new(4, 0).expect("ok").to_bytes();
        bytes[0] = b'X';
        assert!(HyperLogLog::from_bytes(&bytes).is_err());
    }

    #[test]
    fn rejects_wrong_kind() {
        let bloom_bytes = BloomFilter::new(64, 3, 0).expect("ok").to_bytes();
        // Decoding bloom bytes as an HLL must fail on the kind check.
        assert!(HyperLogLog::from_bytes(&bloom_bytes).is_err());
    }

    #[test]
    fn rejects_truncated() {
        let bytes = HyperLogLog::new(8, 0).expect("ok").to_bytes();
        assert!(HyperLogLog::from_bytes(&bytes[..6]).is_err());
        assert!(HyperLogLog::from_bytes(&bytes[..HEADER_LEN + 2]).is_err());
    }

    #[test]
    fn rejects_bad_version() {
        let mut bytes = HyperLogLog::new(8, 0).expect("ok").to_bytes();
        bytes[4] = 200;
        assert!(HyperLogLog::from_bytes(&bytes).is_err());
    }
}
