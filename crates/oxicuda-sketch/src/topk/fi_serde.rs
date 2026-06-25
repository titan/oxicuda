//! Apache DataSketches **Frequent-Items** byte-serialisation compatibility.
//!
//! Implements the on-wire layout of the DataSketches `FrequentLongsSketch`
//! (a.k.a. `frequencies` / FI sketch) so OxiCUDA's heavy-hitter state can be
//! persisted and exchanged in the canonical DataSketches binary frame. This is
//! a *format* (not a re-implementation of their hash map); the in-memory engine
//! remains OxiCUDA's [`crate::topk::misra_gries::MisraGries`] / a
//! `(key, count)` map plus a global `offset`.
//!
//! ## Frame layout (little-endian, exactly as DataSketches)
//!
//! The preamble is a sequence of 8-byte "long" slots; the first one packs the
//! header bytes:
//!
//! ```text
//!   byte 0  : preLongs       (1 if empty, else 4)
//!   byte 1  : serVer         = 1
//!   byte 2  : familyId       = 10  (FREQUENCY)
//!   byte 3  : lgMaxMapSize   (log2 of the maximum internal map size)
//!   byte 4  : lgCurMapSize   (log2 of the current internal map size)
//!   byte 5  : flags          (bit 0 = BIG_ENDIAN, bit 2 = EMPTY)
//!   bytes 6,7: unused (zero)
//! ```
//!
//! When the sketch is non-empty the preamble continues:
//!
//! ```text
//!   long 1 : [ activeItems : u32 | unused : u32 ]
//!   long 2 : streamLength   (u64) — total weight seen
//!   long 3 : offset         (u64) — accumulated lower-bound subtracted
//! ```
//!
//! followed by `activeItems` count longs, then `activeItems` key longs. The
//! count-then-key ordering matches the DataSketches reference serialiser.
//!
//! Estimated frequency of an item is `count + offset`; the guaranteed lower
//! bound is `count`, the upper bound `count + offset`.

use crate::error::{SketchError, SketchResult};
use crate::topk::misra_gries::MisraGries;

/// DataSketches serialisation version for the FI family.
pub const SER_VER: u8 = 1;
/// DataSketches family id for `FREQUENCY`.
pub const FAMILY_FREQUENCY: u8 = 10;
/// `EMPTY` flag bit (bit 2).
pub const FLAG_EMPTY: u8 = 1 << 2;

/// A frequent-items sketch with a DataSketches-compatible byte format.
///
/// Wraps a Misra-Gries map and the global `offset` (the accumulated weight that
/// has been decremented away), which together reproduce the DataSketches FI
/// estimate semantics (`est = count + offset`).
#[derive(Debug, Clone)]
pub struct FrequentItemsSerde {
    /// Underlying Misra-Gries heavy-hitter engine.
    pub mg: MisraGries,
    /// log2 of the maximum internal map size (DataSketches `lgMaxMapSize`).
    pub lg_max_map_size: u8,
    /// Accumulated lower bound subtracted from all counters (DataSketches `offset`).
    pub offset: u64,
}

impl FrequentItemsSerde {
    /// Wrap a [`MisraGries`] sketch with a given `lg_max_map_size` and starting
    /// `offset` (use `0` for a freshly-built sketch).
    ///
    /// `lg_max_map_size` must be at least `ceil(log2(k))` so the active items fit.
    pub fn new(mg: MisraGries, lg_max_map_size: u8, offset: u64) -> SketchResult<Self> {
        let needed = (mg.k as f64).log2().ceil() as u8;
        if lg_max_map_size < needed {
            return Err(SketchError::InvalidParameter {
                name: "lg_max_map_size".to_string(),
                reason: format!("must be >= ceil(log2(k)) = {needed}"),
            });
        }
        Ok(Self {
            mg,
            lg_max_map_size,
            offset,
        })
    }

    /// Number of active (non-empty) items.
    #[must_use]
    pub fn active_items(&self) -> usize {
        self.mg.candidates().len()
    }

    /// Whether the sketch has seen no items.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mg.n == 0 && self.mg.candidates().is_empty()
    }

    /// log2 of the current map size: smallest power of two ≥ active items,
    /// floored at 3 (DataSketches' minimum `lgCurMapSize`).
    fn lg_cur_map_size(&self) -> u8 {
        let active = self.active_items().max(1) as u64;
        let mut lg = 3u8;
        while (1u64 << lg) < active && lg < self.lg_max_map_size {
            lg += 1;
        }
        lg
    }

    /// Serialise into the DataSketches FI binary frame.
    #[must_use]
    pub fn to_datasketches_bytes(&self) -> Vec<u8> {
        let active = self.active_items();
        let empty = self.is_empty();
        let pre_longs: u8 = if empty { 1 } else { 4 };
        let mut out: Vec<u8> = Vec::new();

        // Preamble long 0 (header bytes).
        out.push(pre_longs);
        out.push(SER_VER);
        out.push(FAMILY_FREQUENCY);
        out.push(self.lg_max_map_size);
        out.push(self.lg_cur_map_size());
        out.push(if empty { FLAG_EMPTY } else { 0 });
        out.push(0);
        out.push(0);

        if empty {
            return out;
        }

        // Long 1: activeItems (u32) | unused (u32).
        out.extend_from_slice(&(active as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        // Long 2: streamLength.
        out.extend_from_slice(&self.mg.n.to_le_bytes());
        // Long 3: offset.
        out.extend_from_slice(&self.offset.to_le_bytes());

        // Counts (longs), then keys (longs).
        for &(_, c) in self.mg.candidates() {
            out.extend_from_slice(&c.to_le_bytes());
        }
        for &(key, _) in self.mg.candidates() {
            out.extend_from_slice(&key.to_le_bytes());
        }
        out
    }

    /// Deserialise from a DataSketches FI binary frame.
    pub fn from_datasketches_bytes(bytes: &[u8]) -> SketchResult<Self> {
        if bytes.len() < 8 {
            return Err(SketchError::InvalidParameter {
                name: "bytes".to_string(),
                reason: "truncated preamble".to_string(),
            });
        }
        let pre_longs = bytes[0];
        let ser_ver = bytes[1];
        let family = bytes[2];
        let lg_max_map_size = bytes[3];
        let flags = bytes[5];
        if ser_ver != SER_VER {
            return Err(SketchError::InvalidParameter {
                name: "serVer".to_string(),
                reason: format!("unsupported FI serVer {ser_ver}"),
            });
        }
        if family != FAMILY_FREQUENCY {
            return Err(SketchError::InvalidParameter {
                name: "familyId".to_string(),
                reason: format!("not a FREQUENCY family frame (got {family})"),
            });
        }
        let is_empty = (flags & FLAG_EMPTY) != 0;
        if is_empty || pre_longs == 1 {
            // Empty sketch: reconstruct a default MG with capacity 2^lgMaxMapSize.
            let k = (1usize << lg_max_map_size).max(2);
            let mg = MisraGries::new(k)?;
            return FrequentItemsSerde::new(mg, lg_max_map_size, 0);
        }
        if bytes.len() < 32 {
            return Err(SketchError::InvalidParameter {
                name: "bytes".to_string(),
                reason: "truncated non-empty preamble".to_string(),
            });
        }
        let active = read_u32(bytes, 8)? as usize;
        let stream_length = read_u64(bytes, 16)?;
        let offset = read_u64(bytes, 24)?;

        let counts_start = 32usize;
        let keys_start = counts_start + active * 8;
        let need = keys_start + active * 8;
        if bytes.len() < need {
            return Err(SketchError::InvalidParameter {
                name: "bytes".to_string(),
                reason: format!("truncated body: need {need}, have {}", bytes.len()),
            });
        }
        // Rebuild a Misra-Gries with capacity large enough for the active items.
        let k = (1usize << lg_max_map_size).max(active + 1).max(2);
        let mut mg = MisraGries::new(k)?;
        let mut slots = Vec::with_capacity(active);
        for i in 0..active {
            let c = read_u64(bytes, counts_start + i * 8)?;
            let key = read_u64(bytes, keys_start + i * 8)?;
            slots.push((key, c));
        }
        mg.slots = slots;
        mg.n = stream_length;
        FrequentItemsSerde::new(mg, lg_max_map_size, offset)
    }

    /// Estimated frequency of `key`: `count + offset` if present, else `offset`
    /// is *not* added (an unseen key has estimate `0` lower / `offset` upper).
    /// Returns the DataSketches point estimate `count + offset` for active keys.
    #[must_use]
    pub fn estimate(&self, key: u64) -> u64 {
        let count = self
            .mg
            .candidates()
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, c)| *c)
            .unwrap_or(0);
        if count == 0 { 0 } else { count + self.offset }
    }
}

fn read_u32(buf: &[u8], off: usize) -> SketchResult<u32> {
    if off + 4 > buf.len() {
        return Err(SketchError::InvalidParameter {
            name: "u32".to_string(),
            reason: format!("read out of bounds at {off}"),
        });
    }
    let mut b = [0u8; 4];
    b.copy_from_slice(&buf[off..off + 4]);
    Ok(u32::from_le_bytes(b))
}

fn read_u64(buf: &[u8], off: usize) -> SketchResult<u64> {
    if off + 8 > buf.len() {
        return Err(SketchError::InvalidParameter {
            name: "u64".to_string(),
            reason: format!("read out of bounds at {off}"),
        });
    }
    let mut b = [0u8; 8];
    b.copy_from_slice(&buf[off..off + 8]);
    Ok(u64::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_fi() -> FrequentItemsSerde {
        // k = 16 ⇒ lgMaxMapSize >= 4.
        let mut mg = MisraGries::new(16).expect("ok");
        for _ in 0..200 {
            mg.add(7);
        }
        for _ in 0..120 {
            mg.add(9);
        }
        for i in 0..50u64 {
            mg.add(i + 100);
        }
        FrequentItemsSerde::new(mg, 4, 3).expect("ok")
    }

    #[test]
    fn fi_new_validates_lg_size() {
        let mg = MisraGries::new(16).expect("ok");
        // ceil(log2(16)) = 4, so 3 is too small.
        assert!(FrequentItemsSerde::new(mg.clone(), 3, 0).is_err());
        assert!(FrequentItemsSerde::new(mg, 4, 0).is_ok());
    }

    #[test]
    fn fi_header_bytes_correct() {
        let fi = build_fi();
        let bytes = fi.to_datasketches_bytes();
        assert_eq!(bytes[0], 4, "non-empty preLongs must be 4");
        assert_eq!(bytes[1], SER_VER);
        assert_eq!(bytes[2], FAMILY_FREQUENCY);
        assert_eq!(bytes[3], 4, "lgMaxMapSize");
        assert_eq!(bytes[5] & FLAG_EMPTY, 0, "non-empty must not set EMPTY");
    }

    #[test]
    fn fi_empty_frame() {
        let mg = MisraGries::new(16).expect("ok");
        let fi = FrequentItemsSerde::new(mg, 4, 0).expect("ok");
        assert!(fi.is_empty());
        let bytes = fi.to_datasketches_bytes();
        assert_eq!(bytes.len(), 8, "empty frame is a single preamble long");
        assert_eq!(bytes[0], 1, "empty preLongs must be 1");
        assert_eq!(bytes[5] & FLAG_EMPTY, FLAG_EMPTY);
        let back = FrequentItemsSerde::from_datasketches_bytes(&bytes).expect("ok");
        assert!(back.is_empty());
    }

    #[test]
    fn fi_roundtrip_preserves_active_items() {
        let fi = build_fi();
        let bytes = fi.to_datasketches_bytes();
        let back = FrequentItemsSerde::from_datasketches_bytes(&bytes).expect("ok");
        assert_eq!(back.mg.n, fi.mg.n, "streamLength preserved");
        assert_eq!(back.offset, fi.offset, "offset preserved");
        // Active item multiset must match exactly.
        let mut orig: Vec<(u64, u64)> = fi.mg.candidates().to_vec();
        let mut got: Vec<(u64, u64)> = back.mg.candidates().to_vec();
        orig.sort_unstable();
        got.sort_unstable();
        assert_eq!(orig, got, "active (key,count) set must survive round-trip");
    }

    #[test]
    fn fi_heavy_hitters_survive_roundtrip() {
        let fi = build_fi();
        let bytes = fi.to_datasketches_bytes();
        let back = FrequentItemsSerde::from_datasketches_bytes(&bytes).expect("ok");
        // 7 was the heaviest; it must remain and estimate count+offset.
        assert!(back.estimate(7) >= 200, "heavy item 7 estimate too low");
        assert_eq!(back.estimate(7), fi.estimate(7));
        // Unseen key estimates 0.
        assert_eq!(back.estimate(999_999), 0);
    }

    #[test]
    fn fi_rejects_bad_family() {
        let fi = build_fi();
        let mut bytes = fi.to_datasketches_bytes();
        bytes[2] = 3; // not FREQUENCY
        assert!(FrequentItemsSerde::from_datasketches_bytes(&bytes).is_err());
    }

    #[test]
    fn fi_rejects_bad_server() {
        let fi = build_fi();
        let mut bytes = fi.to_datasketches_bytes();
        bytes[1] = 99;
        assert!(FrequentItemsSerde::from_datasketches_bytes(&bytes).is_err());
    }

    #[test]
    fn fi_rejects_truncated_body() {
        let fi = build_fi();
        let bytes = fi.to_datasketches_bytes();
        let truncated = &bytes[..bytes.len() - 8];
        assert!(FrequentItemsSerde::from_datasketches_bytes(truncated).is_err());
    }

    #[test]
    fn fi_lg_cur_map_size_monotone() {
        // More active items ⇒ lgCurMapSize does not shrink.
        let small = {
            let mut mg = MisraGries::new(32).expect("ok");
            mg.add(1);
            FrequentItemsSerde::new(mg, 5, 0).expect("ok")
        };
        let large = {
            let mut mg = MisraGries::new(32).expect("ok");
            for i in 0..20u64 {
                mg.add(i);
            }
            FrequentItemsSerde::new(mg, 5, 0).expect("ok")
        };
        assert!(large.lg_cur_map_size() >= small.lg_cur_map_size());
    }
}
