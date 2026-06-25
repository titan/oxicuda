//! Adapter input/output: pure-Rust serialization and a shared-adapter registry/hub.
//!
//! - [`serialize`] defines the self-describing `OXPA` byte container ([`AdapterPayload`]) for a
//!   single adapter's named tensors, plus temp-file save/load — no `serde` / `bincode` / `zip`.
//! - [`registry`] defines the [`AdapterRegistry`] hub convention: many task adapters keyed by
//!   `base_model` + `task` + `name`, each carrying an [`AdapterCard`], with whole-hub
//!   serialization.

/// Adapter weight registry / hub conventions for shared adapters.
pub mod registry;
/// Self-describing binary serialization of adapter tensor payloads.
pub mod serialize;

pub use registry::{AdapterCard, AdapterEntry, AdapterRegistry, PeftMethod, REGISTRY_MAGIC};
pub use serialize::{AdapterPayload, FORMAT_VERSION, MAGIC};
