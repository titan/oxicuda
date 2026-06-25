//! Self-describing checkpoint format for [`LoraModel`].
//!
//! Serialises a complete [`LoraModel`] (its [`LoraConfig`] plus every named
//! [`LoraLinear`] adapter, including the A / B factor matrices, rank and
//! scaling) to a flat little-endian byte buffer, and reconstructs it exactly.
//!
//! The crate depends only on `thiserror`, so rather than pulling in `serde`
//! this is a hand-rolled little-endian layout. It is deliberately simple,
//! versioned, and length-prefixed so it is forward-checkable.
//!
//! # Byte layout (all integers little-endian)
//!
//! ```text
//! magic            : 8 bytes  = b"OXLORA01"
//! ── config ──
//! rank             : u32
//! alpha            : f32
//! dropout          : f32
//! n_target_modules : u32
//!   repeated n_target_modules times:
//!     name_len     : u32
//!     name_bytes   : name_len bytes (UTF-8)
//! ── adapters ──
//! n_adapters       : u32      (adapters emitted in sorted name order)
//!   repeated n_adapters times:
//!     name_len     : u32
//!     name_bytes   : name_len bytes (UTF-8)
//!     in_features  : u32
//!     out_features : u32
//!     rank         : u32
//!     scaling      : f32      (α/r, stored verbatim for exact round-trip)
//!     a_len        : u32      (= rank * in_features)
//!     a_bytes      : a_len  f32 values
//!     b_len        : u32      (= out_features * rank)
//!     b_bytes      : b_len  f32 values
//! ```
//!
//! Adapters are written in **sorted key order** so the byte output is
//! deterministic regardless of `HashMap` iteration order.

use crate::error::{GenError, GenResult};
use crate::lora::adapter::{LoraConfig, LoraLinear, LoraModel};

/// Magic header identifying an `oxicuda-gen` LoRA checkpoint, version 01.
pub const LORA_CKPT_MAGIC: &[u8; 8] = b"OXLORA01";

// ─── Low-level little-endian writers ──────────────────────────────────────────

fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_str(buf: &mut Vec<u8>, s: &str) {
    push_u32(buf, s.len() as u32);
    buf.extend_from_slice(s.as_bytes());
}

fn push_f32_slice(buf: &mut Vec<u8>, xs: &[f32]) {
    push_u32(buf, xs.len() as u32);
    for &x in xs {
        push_f32(buf, x);
    }
}

// ─── Low-level little-endian readers (bounds-checked) ─────────────────────────

/// Cursor over a byte slice that reads little-endian primitives with bounds
/// checks, returning `GenError::Internal` on truncation.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn take(&mut self, n: usize) -> GenResult<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| GenError::Internal("checkpoint length overflow".to_string()))?;
        if end > self.data.len() {
            return Err(GenError::Internal(format!(
                "checkpoint truncated: need {n} bytes at offset {}, have {}",
                self.pos,
                self.data.len() - self.pos
            )));
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_u32(&mut self) -> GenResult<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_f32(&mut self) -> GenResult<f32> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_usize(&mut self) -> GenResult<usize> {
        Ok(self.read_u32()? as usize)
    }

    fn read_str(&mut self) -> GenResult<String> {
        let len = self.read_usize()?;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| GenError::Internal(format!("invalid UTF-8 in checkpoint name: {e}")))
    }

    fn read_f32_vec(&mut self) -> GenResult<Vec<f32>> {
        let len = self.read_usize()?;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(self.read_f32()?);
        }
        Ok(out)
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Serialise a [`LoraModel`] to a self-describing little-endian byte buffer.
///
/// Adapters are emitted in sorted name order, so the output is deterministic.
pub fn save(model: &LoraModel) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(LORA_CKPT_MAGIC);

    // ── config ──
    let cfg = model.config();
    push_u32(&mut buf, cfg.rank as u32);
    push_f32(&mut buf, cfg.alpha);
    push_f32(&mut buf, cfg.dropout);
    push_u32(&mut buf, cfg.target_modules.len() as u32);
    for m in &cfg.target_modules {
        push_str(&mut buf, m);
    }

    // ── adapters (sorted for determinism) ──
    let mut names: Vec<&String> = model.adapters().keys().collect();
    names.sort();
    push_u32(&mut buf, names.len() as u32);
    for name in names {
        // `name` came from the model's own key set, so the adapter is present.
        let adapter = match model.get_adapter(name) {
            Some(a) => a,
            None => continue,
        };
        push_str(&mut buf, name);
        push_u32(&mut buf, adapter.in_features() as u32);
        push_u32(&mut buf, adapter.out_features() as u32);
        push_u32(&mut buf, adapter.rank() as u32);
        push_f32(&mut buf, adapter.scaling());
        push_f32_slice(&mut buf, adapter.matrix_a());
        push_f32_slice(&mut buf, adapter.matrix_b());
    }

    buf
}

/// Reconstruct a [`LoraModel`] from a checkpoint produced by [`save`].
///
/// # Errors
/// - `Internal` if the magic header is wrong, the buffer is truncated, a name
///   is not valid UTF-8, or trailing bytes remain after the declared content.
/// - `DimensionMismatch` (via [`LoraLinear::from_parts`]) if a stored matrix
///   length is inconsistent with its declared shape.
/// - `InvalidLoraRank` / `InvalidLoraAlpha` (via [`LoraConfig::with_options`])
///   if the stored config is invalid.
pub fn load(bytes: &[u8]) -> GenResult<LoraModel> {
    let mut r = Reader::new(bytes);
    let magic = r.take(8)?;
    if magic != LORA_CKPT_MAGIC {
        return Err(GenError::Internal(format!(
            "bad LoRA checkpoint magic: expected {:?}, got {:?}",
            LORA_CKPT_MAGIC, magic
        )));
    }

    // ── config ──
    let rank = r.read_usize()?;
    let alpha = r.read_f32()?;
    let dropout = r.read_f32()?;
    let n_modules = r.read_usize()?;
    let mut target_modules = Vec::with_capacity(n_modules);
    for _ in 0..n_modules {
        target_modules.push(r.read_str()?);
    }
    let config = LoraConfig::with_options(rank, alpha, dropout, target_modules)?;
    let mut model = LoraModel::new(config);

    // ── adapters ──
    let n_adapters = r.read_usize()?;
    for _ in 0..n_adapters {
        let name = r.read_str()?;
        let in_features = r.read_usize()?;
        let out_features = r.read_usize()?;
        let a_rank = r.read_usize()?;
        let scaling = r.read_f32()?;
        let matrix_a = r.read_f32_vec()?;
        let matrix_b = r.read_f32_vec()?;
        let adapter = LoraLinear::from_parts(
            in_features,
            out_features,
            a_rank,
            scaling,
            matrix_a,
            matrix_b,
        )?;
        model.add_adapter(name, adapter);
    }

    if r.pos != bytes.len() {
        return Err(GenError::Internal(format!(
            "trailing bytes after LoRA checkpoint: consumed {}, total {}",
            r.pos,
            bytes.len()
        )));
    }

    Ok(model)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn build_model() -> LoraModel {
        let config = LoraConfig::with_options(
            4,
            8.0,
            0.05,
            vec!["q_proj".to_string(), "v_proj".to_string()],
        )
        .expect("config should construct");
        let mut model = LoraModel::new(config.clone());
        let mut rng = LcgRng::new(123);
        let a1 = LoraLinear::new(16, 32, &config, &mut rng).expect("adapter 1");
        let mut a2 = LoraLinear::new(8, 8, &config, &mut rng).expect("adapter 2");
        // Make B non-zero so the round-trip covers a populated B factor too.
        for (i, v) in a2.matrix_b_mut().iter_mut().enumerate() {
            *v = (i as f32) * 0.013 - 0.1;
        }
        model.add_adapter("attn.q_proj", a1);
        model.add_adapter("attn.v_proj", a2);
        model
    }

    #[test]
    fn save_load_roundtrip_identity() {
        let model = build_model();
        let bytes = save(&model);
        let restored = load(&bytes).expect("load should succeed");

        // Config preserved exactly.
        assert_eq!(restored.config().rank, model.config().rank);
        assert_eq!(
            restored.config().alpha.to_bits(),
            model.config().alpha.to_bits(),
            "alpha must round-trip bit-for-bit"
        );
        assert_eq!(
            restored.config().dropout.to_bits(),
            model.config().dropout.to_bits()
        );
        assert_eq!(
            restored.config().target_modules,
            model.config().target_modules
        );

        // Adapter set preserved.
        assert_eq!(restored.adapter_count(), model.adapter_count());

        for (name, orig) in model.adapters() {
            let got = restored
                .get_adapter(name)
                .unwrap_or_else(|| panic!("adapter {name} missing after load"));
            assert_eq!(got.in_features(), orig.in_features());
            assert_eq!(got.out_features(), orig.out_features());
            assert_eq!(got.rank(), orig.rank(), "rank must round-trip exactly");
            assert_eq!(
                got.scaling().to_bits(),
                orig.scaling().to_bits(),
                "scaling (α/r) must round-trip bit-for-bit"
            );
            // A and B factor matrices must be bit-for-bit identical.
            assert_eq!(got.matrix_a().len(), orig.matrix_a().len());
            for (a, b) in got.matrix_a().iter().zip(orig.matrix_a()) {
                assert_eq!(a.to_bits(), b.to_bits(), "A factor mismatch");
            }
            assert_eq!(got.matrix_b().len(), orig.matrix_b().len());
            for (a, b) in got.matrix_b().iter().zip(orig.matrix_b()) {
                assert_eq!(a.to_bits(), b.to_bits(), "B factor mismatch");
            }
        }
    }

    #[test]
    fn save_is_deterministic() {
        let model = build_model();
        let a = save(&model);
        let b = save(&model);
        assert_eq!(a, b, "serialisation must be byte-deterministic");
    }

    #[test]
    fn roundtrip_preserves_forward_output() {
        // A stronger functional check: the restored model's adapter must produce
        // exactly the same forward output as the original.
        let model = build_model();
        let bytes = save(&model);
        let restored = load(&bytes).expect("load");
        let orig = model.get_adapter("attn.v_proj").expect("orig adapter");
        let got = restored
            .get_adapter("attn.v_proj")
            .expect("restored adapter");
        let x = vec![0.5_f32, -0.25, 0.75, 1.0, -1.0, 0.1, 0.2, 0.3]; // batch=1, in=8
        let base = vec![0.0_f32; 8];
        let out_orig = orig.forward(&x, &base, 1).expect("orig forward");
        let out_got = got.forward(&x, &base, 1).expect("restored forward");
        for (a, b) in out_orig.iter().zip(&out_got) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "forward output must match exactly"
            );
        }
    }

    #[test]
    fn empty_model_roundtrip() {
        let config = LoraConfig::new(2, 2.0).expect("config");
        let model = LoraModel::new(config);
        let bytes = save(&model);
        let restored = load(&bytes).expect("load empty");
        assert_eq!(restored.adapter_count(), 0);
        assert_eq!(restored.config().rank, 2);
    }

    #[test]
    fn load_rejects_bad_magic() {
        let mut bytes = save(&build_model());
        bytes[0] = b'X';
        assert!(matches!(load(&bytes), Err(GenError::Internal(_))));
    }

    #[test]
    fn load_rejects_truncated() {
        let bytes = save(&build_model());
        let truncated = &bytes[..bytes.len() / 2];
        assert!(matches!(load(truncated), Err(GenError::Internal(_))));
    }

    #[test]
    fn load_rejects_trailing_bytes() {
        let mut bytes = save(&build_model());
        bytes.push(0xAB);
        assert!(matches!(load(&bytes), Err(GenError::Internal(_))));
    }

    #[test]
    fn magic_header_present() {
        let bytes = save(&build_model());
        assert_eq!(&bytes[..8], LORA_CKPT_MAGIC);
    }
}
