//! LoRA hub / registry conventions for shared adapters.
//!
//! A lightweight, in-process catalogue that maps a unique adapter *name* to its
//! [`AdapterCard`] (metadata following a fixed convention) and its [`AdapterPayload`]
//! (the serialized tensors). It models the directory-of-adapters layout that tools such as
//! the HuggingFace PEFT hub use — each adapter is identified by `base_model` + `task` + `name`,
//! tagged with its method, rank and α — but keeps everything pure-Rust and dependency-free.
//!
//! The registry supports:
//! - registration with duplicate-name rejection ([`AdapterRegistry::register`]),
//! - upsert ([`AdapterRegistry::insert_or_replace`]),
//! - lookup / removal / listing,
//! - filtered queries by base model or by task (the sharing convention: many task adapters
//!   target one base model),
//! - whole-registry serialization to a single byte stream so a hub can be saved / shipped.
//!
//! Every adapter name is required to be a non-empty, slug-safe identifier
//! (`[A-Za-z0-9._-]+`) so names round-trip cleanly through filesystem paths and the binary
//! container.

use crate::error::{PeftError, PeftResult};
use crate::io::serialize::{AdapterPayload, FORMAT_VERSION, MAGIC};
use std::collections::BTreeMap;

/// Family of PEFT method an adapter was produced with.
///
/// Stored as a stable `u8` tag in the serialized registry so the catalogue is
/// forward-compatible with new methods (unknown tags decode to [`PeftMethod::Other`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeftMethod {
    /// Vanilla LoRA (`B·A` low-rank delta).
    Lora,
    /// Quantized LoRA (NF4 base + LoRA).
    QLora,
    /// Weight-decomposed LoRA.
    Dora,
    /// Adaptive-rank LoRA.
    AdaLora,
    /// IA³ element-wise scaling.
    Ia3,
    /// Prefix / prompt / P-tuning soft-prompt method.
    Prompt,
    /// Bottleneck adapter (Houlsby / Pfeiffer / parallel).
    Adapter,
    /// Any other / experimental method.
    Other,
}

impl PeftMethod {
    /// Stable serialization tag.
    #[must_use]
    pub fn tag(self) -> u8 {
        match self {
            PeftMethod::Lora => 0,
            PeftMethod::QLora => 1,
            PeftMethod::Dora => 2,
            PeftMethod::AdaLora => 3,
            PeftMethod::Ia3 => 4,
            PeftMethod::Prompt => 5,
            PeftMethod::Adapter => 6,
            PeftMethod::Other => 255,
        }
    }

    /// Decode a serialization tag (unknown tags map to [`PeftMethod::Other`]).
    #[must_use]
    pub fn from_tag(tag: u8) -> Self {
        match tag {
            0 => PeftMethod::Lora,
            1 => PeftMethod::QLora,
            2 => PeftMethod::Dora,
            3 => PeftMethod::AdaLora,
            4 => PeftMethod::Ia3,
            5 => PeftMethod::Prompt,
            6 => PeftMethod::Adapter,
            _ => PeftMethod::Other,
        }
    }
}

/// Metadata describing a single registered adapter — the "model card" of the hub convention.
#[derive(Debug, Clone, PartialEq)]
pub struct AdapterCard {
    /// Identifier of the frozen base model the adapter targets (e.g. `"llama-7b"`).
    pub base_model: String,
    /// Downstream task the adapter was trained for (e.g. `"sst2"`).
    pub task: String,
    /// PEFT method family.
    pub method: PeftMethod,
    /// Low-rank dimension (0 for methods without a rank, such as BitFit / prompt-tuning).
    pub rank: usize,
    /// Scaling factor α (or any method-specific scalar; `0.0` if not applicable).
    pub alpha: f32,
    /// Number of trainable scalar parameters introduced by the adapter.
    pub trainable_params: usize,
}

impl AdapterCard {
    /// Build a card for a low-rank adapter, deriving `trainable_params` from the payload.
    #[must_use]
    pub fn new(
        base_model: impl Into<String>,
        task: impl Into<String>,
        method: PeftMethod,
        rank: usize,
        alpha: f32,
        trainable_params: usize,
    ) -> Self {
        Self {
            base_model: base_model.into(),
            task: task.into(),
            method,
            rank,
            alpha,
            trainable_params,
        }
    }
}

/// One registry entry: its card plus its serialized tensors.
#[derive(Debug, Clone, PartialEq)]
pub struct AdapterEntry {
    /// Descriptive metadata.
    pub card: AdapterCard,
    /// The adapter's trainable tensors.
    pub payload: AdapterPayload,
}

/// Magic bytes prefixing a serialized registry: `OXPH` = OxiCUDA-Peft-Hub.
pub const REGISTRY_MAGIC: [u8; 4] = *b"OXPH";

/// In-memory catalogue of named adapters sharing a common base-model convention.
#[derive(Debug, Clone, Default)]
pub struct AdapterRegistry {
    entries: BTreeMap<String, AdapterEntry>,
}

impl AdapterRegistry {
    /// Construct an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Validate that `name` is a non-empty slug (`[A-Za-z0-9._-]+`).
    fn validate_name(name: &str) -> PeftResult<()> {
        if name.is_empty() {
            return Err(PeftError::CorruptData {
                msg: "adapter name must be non-empty".to_string(),
            });
        }
        if !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-')
        {
            return Err(PeftError::CorruptData {
                msg: format!("adapter name '{name}' has non-slug characters"),
            });
        }
        Ok(())
    }

    /// Register a new adapter, rejecting a name that is already present.
    ///
    /// # Errors
    ///
    /// - [`PeftError::CorruptData`] when `name` is not a valid slug.
    /// - [`PeftError::DuplicateAdapter`] when `name` already exists.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        card: AdapterCard,
        payload: AdapterPayload,
    ) -> PeftResult<()> {
        let name = name.into();
        Self::validate_name(&name)?;
        if self.entries.contains_key(&name) {
            return Err(PeftError::DuplicateAdapter { name });
        }
        self.entries.insert(name, AdapterEntry { card, payload });
        Ok(())
    }

    /// Insert an adapter, replacing any existing entry with the same name.
    ///
    /// # Errors
    ///
    /// [`PeftError::CorruptData`] when `name` is not a valid slug.
    pub fn insert_or_replace(
        &mut self,
        name: impl Into<String>,
        card: AdapterCard,
        payload: AdapterPayload,
    ) -> PeftResult<()> {
        let name = name.into();
        Self::validate_name(&name)?;
        self.entries.insert(name, AdapterEntry { card, payload });
        Ok(())
    }

    /// Look up an adapter by name.
    ///
    /// # Errors
    ///
    /// [`PeftError::AdapterNotFound`] when no entry matches `name`.
    pub fn get(&self, name: &str) -> PeftResult<&AdapterEntry> {
        self.entries
            .get(name)
            .ok_or_else(|| PeftError::AdapterNotFound {
                name: name.to_string(),
            })
    }

    /// Remove and return an adapter by name.
    ///
    /// # Errors
    ///
    /// [`PeftError::AdapterNotFound`] when no entry matches `name`.
    pub fn remove(&mut self, name: &str) -> PeftResult<AdapterEntry> {
        self.entries
            .remove(name)
            .ok_or_else(|| PeftError::AdapterNotFound {
                name: name.to_string(),
            })
    }

    /// Whether an adapter with `name` is registered.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Number of registered adapters.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// All adapter names in deterministic (lexicographic) order.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    /// Iterate over `(name, entry)` pairs in deterministic order.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &AdapterEntry)> {
        self.entries.iter()
    }

    /// Names of every adapter targeting a given base model (the sharing convention).
    #[must_use]
    pub fn names_for_base_model(&self, base_model: &str) -> Vec<String> {
        self.entries
            .iter()
            .filter(|(_, e)| e.card.base_model == base_model)
            .map(|(n, _)| n.clone())
            .collect()
    }

    /// Names of every adapter trained for a given task.
    #[must_use]
    pub fn names_for_task(&self, task: &str) -> Vec<String> {
        self.entries
            .iter()
            .filter(|(_, e)| e.card.task == task)
            .map(|(n, _)| n.clone())
            .collect()
    }

    /// Serialize the entire registry to a single self-describing byte stream.
    ///
    /// Layout: `REGISTRY_MAGIC` + `FORMAT_VERSION` (u32) + `entry_count` (u32), then per entry
    /// the name, card fields, and the [`AdapterPayload::to_bytes`] block length-prefixed, with a
    /// trailing FNV-1a checksum identical in spirit to the single-adapter container.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&REGISTRY_MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        let count = u32::try_from(self.entries.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&count.to_le_bytes());
        for (name, entry) in &self.entries {
            write_str(&mut out, name);
            let card = &entry.card;
            write_str(&mut out, &card.base_model);
            write_str(&mut out, &card.task);
            out.push(card.method.tag());
            out.extend_from_slice(&(u32::try_from(card.rank).unwrap_or(u32::MAX)).to_le_bytes());
            out.extend_from_slice(&card.alpha.to_le_bytes());
            out.extend_from_slice(
                &(u64::try_from(card.trainable_params).unwrap_or(u64::MAX)).to_le_bytes(),
            );
            let blob = entry.payload.to_bytes();
            out.extend_from_slice(&(u32::try_from(blob.len()).unwrap_or(u32::MAX)).to_le_bytes());
            out.extend_from_slice(&blob);
        }
        let checksum = fnv1a(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        out
    }

    /// Reconstruct a registry from bytes produced by [`Self::to_bytes`].
    ///
    /// # Errors
    ///
    /// - [`PeftError::CorruptData`] for a bad magic, overruns, invalid UTF-8 / payload, or a
    ///   checksum mismatch.
    /// - [`PeftError::UnsupportedVersion`] when the stored version exceeds [`FORMAT_VERSION`].
    pub fn from_bytes(bytes: &[u8]) -> PeftResult<Self> {
        if bytes.len() < 20 {
            return Err(PeftError::CorruptData {
                msg: format!("registry buffer too short: {} bytes", bytes.len()),
            });
        }
        // Verify the trailing checksum first so later reads operate on a validated stream.
        let body_end = bytes.len() - 8;
        let stored = read_u64(&bytes[body_end..])?;
        let computed = fnv1a(&bytes[..body_end]);
        if stored != computed {
            return Err(PeftError::CorruptData {
                msg: "registry checksum mismatch".to_string(),
            });
        }
        let mut pos = 0usize;
        let magic = read_array4(bytes, &mut pos)?;
        if magic != REGISTRY_MAGIC {
            return Err(PeftError::CorruptData {
                msg: "bad registry magic".to_string(),
            });
        }
        let version = read_u32(bytes, &mut pos)?;
        if version > FORMAT_VERSION {
            return Err(PeftError::UnsupportedVersion {
                found: version,
                supported: FORMAT_VERSION,
            });
        }
        let count = read_u32(bytes, &mut pos)? as usize;
        let mut entries = BTreeMap::new();
        for _ in 0..count {
            let name = read_str(bytes, &mut pos)?;
            let base_model = read_str(bytes, &mut pos)?;
            let task = read_str(bytes, &mut pos)?;
            let method = PeftMethod::from_tag(read_u8(bytes, &mut pos)?);
            let rank = read_u32(bytes, &mut pos)? as usize;
            let alpha = read_f32(bytes, &mut pos)?;
            let trainable_params = read_u64_at(bytes, &mut pos)? as usize;
            let blob_len = read_u32(bytes, &mut pos)? as usize;
            let blob = read_slice(bytes, &mut pos, blob_len)?;
            let payload = AdapterPayload::from_bytes(blob)?;
            entries.insert(
                name,
                AdapterEntry {
                    card: AdapterCard {
                        base_model,
                        task,
                        method,
                        rank,
                        alpha,
                        trainable_params,
                    },
                    payload,
                },
            );
        }
        Ok(Self { entries })
    }
}

/// Length-prefix and append a UTF-8 string (`u32` length + bytes).
fn write_str(out: &mut Vec<u8>, s: &str) {
    let len = u32::try_from(s.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// FNV-1a 64-bit over a byte slice (kept private; the [`MAGIC`] re-export pins the shared format).
fn fnv1a(bytes: &[u8]) -> u64 {
    // Touch the shared constant so a format-version bump there is felt at compile time here.
    debug_assert_eq!(MAGIC.len(), 4);
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn need<'a>(bytes: &'a [u8], pos: &mut usize, n: usize) -> PeftResult<&'a [u8]> {
    let end = pos.checked_add(n).ok_or_else(|| PeftError::CorruptData {
        msg: "registry length overflow".to_string(),
    })?;
    if end > bytes.len() {
        return Err(PeftError::CorruptData {
            msg: "registry read overruns buffer".to_string(),
        });
    }
    let s = &bytes[*pos..end];
    *pos = end;
    Ok(s)
}

fn read_array4(bytes: &[u8], pos: &mut usize) -> PeftResult<[u8; 4]> {
    let s = need(bytes, pos, 4)?;
    Ok([s[0], s[1], s[2], s[3]])
}

fn read_u8(bytes: &[u8], pos: &mut usize) -> PeftResult<u8> {
    Ok(need(bytes, pos, 1)?[0])
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> PeftResult<u32> {
    Ok(u32::from_le_bytes(read_array4(bytes, pos)?))
}

fn read_f32(bytes: &[u8], pos: &mut usize) -> PeftResult<f32> {
    Ok(f32::from_le_bytes(read_array4(bytes, pos)?))
}

fn read_u64_at(bytes: &[u8], pos: &mut usize) -> PeftResult<u64> {
    let s = need(bytes, pos, 8)?;
    Ok(u64::from_le_bytes([
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
    ]))
}

fn read_u64(slice: &[u8]) -> PeftResult<u64> {
    if slice.len() < 8 {
        return Err(PeftError::CorruptData {
            msg: "missing registry checksum".to_string(),
        });
    }
    Ok(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

fn read_str(bytes: &[u8], pos: &mut usize) -> PeftResult<String> {
    let len = read_u32(bytes, pos)? as usize;
    let s = need(bytes, pos, len)?;
    String::from_utf8(s.to_vec()).map_err(|e| PeftError::CorruptData {
        msg: format!("invalid UTF-8 in registry string: {e}"),
    })
}

fn read_slice<'a>(bytes: &'a [u8], pos: &mut usize, n: usize) -> PeftResult<&'a [u8]> {
    need(bytes, pos, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(base: &str, task: &str) -> AdapterCard {
        AdapterCard::new(base, task, PeftMethod::Lora, 8, 16.0, 4096)
    }

    fn payload() -> AdapterPayload {
        AdapterPayload::new()
            .with_tensor("A", vec![0.1, 0.2, 0.3])
            .with_tensor("B", vec![0.0, 0.0])
    }

    #[test]
    fn register_then_get() {
        let mut reg = AdapterRegistry::new();
        reg.register("sst2-lora", card("bert", "sst2"), payload())
            .expect("first registration succeeds");
        let entry = reg.get("sst2-lora").expect("entry is present");
        assert_eq!(entry.card.rank, 8);
        assert_eq!(entry.payload.get("A"), Some([0.1_f32, 0.2, 0.3].as_slice()));
    }

    #[test]
    fn duplicate_registration_is_rejected() {
        let mut reg = AdapterRegistry::new();
        reg.register("a", card("bert", "sst2"), payload())
            .expect("first ok");
        let err = reg
            .register("a", card("bert", "qqp"), payload())
            .expect_err("duplicate must fail");
        assert!(matches!(err, PeftError::DuplicateAdapter { .. }));
        // insert_or_replace overrides instead.
        reg.insert_or_replace("a", card("bert", "qqp"), payload())
            .expect("upsert ok");
        assert_eq!(reg.get("a").expect("present").card.task, "qqp");
    }

    #[test]
    fn invalid_name_rejected() {
        let mut reg = AdapterRegistry::new();
        assert!(matches!(
            reg.register("bad name", card("bert", "sst2"), payload()),
            Err(PeftError::CorruptData { .. })
        ));
        assert!(matches!(
            reg.register("", card("bert", "sst2"), payload()),
            Err(PeftError::CorruptData { .. })
        ));
    }

    #[test]
    fn missing_lookup_and_remove_error() {
        let mut reg = AdapterRegistry::new();
        assert!(matches!(
            reg.get("nope"),
            Err(PeftError::AdapterNotFound { .. })
        ));
        assert!(matches!(
            reg.remove("nope"),
            Err(PeftError::AdapterNotFound { .. })
        ));
    }

    #[test]
    fn filter_by_base_model_and_task() {
        let mut reg = AdapterRegistry::new();
        reg.register("l1", card("llama", "sst2"), payload())
            .expect("ok");
        reg.register("l2", card("llama", "qqp"), payload())
            .expect("ok");
        reg.register("b1", card("bert", "sst2"), payload())
            .expect("ok");
        assert_eq!(reg.names_for_base_model("llama"), vec!["l1", "l2"]);
        assert_eq!(reg.names_for_task("sst2"), vec!["b1", "l1"]);
        assert_eq!(reg.len(), 3);
    }

    #[test]
    fn remove_shrinks_registry() {
        let mut reg = AdapterRegistry::new();
        reg.register("x", card("bert", "sst2"), payload())
            .expect("ok");
        assert!(reg.contains("x"));
        let removed = reg.remove("x").expect("remove ok");
        assert_eq!(removed.card.base_model, "bert");
        assert!(!reg.contains("x"));
        assert!(reg.is_empty());
    }

    #[test]
    fn registry_bytes_roundtrip_preserves_everything() {
        let mut reg = AdapterRegistry::new();
        reg.register(
            "adapter.one",
            AdapterCard::new("llama-7b", "sst2", PeftMethod::Dora, 16, 32.0, 8192),
            payload(),
        )
        .expect("ok");
        reg.register(
            "adapter-two",
            AdapterCard::new("llama-7b", "qqp", PeftMethod::Ia3, 0, 0.0, 512),
            AdapterPayload::new().with_tensor("scale", vec![1.0, 1.0, 1.0]),
        )
        .expect("ok");

        let bytes = reg.to_bytes();
        assert_eq!(&bytes[0..4], &REGISTRY_MAGIC);
        let back = AdapterRegistry::from_bytes(&bytes).expect("roundtrip decodes");
        assert_eq!(back.len(), 2);
        let one = back.get("adapter.one").expect("present");
        assert_eq!(one.card.method, PeftMethod::Dora);
        assert_eq!(one.card.alpha, 32.0);
        assert_eq!(one.card.trainable_params, 8192);
        assert_eq!(one.payload, payload());
        let two = back.get("adapter-two").expect("present");
        assert_eq!(two.card.method, PeftMethod::Ia3);
        assert_eq!(two.card.rank, 0);
        assert_eq!(
            two.payload.get("scale"),
            Some([1.0_f32, 1.0, 1.0].as_slice())
        );
    }

    #[test]
    fn corrupt_registry_bytes_rejected() {
        let mut reg = AdapterRegistry::new();
        reg.register("a", card("bert", "sst2"), payload())
            .expect("ok");
        let mut bytes = reg.to_bytes();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        assert!(matches!(
            AdapterRegistry::from_bytes(&bytes),
            Err(PeftError::CorruptData { .. })
        ));
    }

    #[test]
    fn method_tag_roundtrip() {
        for m in [
            PeftMethod::Lora,
            PeftMethod::QLora,
            PeftMethod::Dora,
            PeftMethod::AdaLora,
            PeftMethod::Ia3,
            PeftMethod::Prompt,
            PeftMethod::Adapter,
            PeftMethod::Other,
        ] {
            assert_eq!(PeftMethod::from_tag(m.tag()), m);
        }
        // Unknown tags fall back to Other.
        assert_eq!(PeftMethod::from_tag(123), PeftMethod::Other);
    }

    #[test]
    fn empty_registry_roundtrips() {
        let reg = AdapterRegistry::new();
        let back = AdapterRegistry::from_bytes(&reg.to_bytes()).expect("empty roundtrip");
        assert!(back.is_empty());
    }
}
