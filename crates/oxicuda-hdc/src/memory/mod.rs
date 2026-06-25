//! Memory modules: item memory (symbol→HV store), associative/Hopfield memory, SPA cleanup
//! memory over real unit vectors, and VSA Resonator Networks.

pub mod assoc_memory;
pub mod cleanup_refine;
pub mod hetero_associative;
pub mod item_memory;
pub mod resonator;
pub mod resonator_attention;
pub mod spa_cleanup;

pub use cleanup_refine::{CleanupConfig, CleanupMode, CleanupResult, cleanup};
pub use hetero_associative::{HeteroAssocConfig, HeteroAssociativeMemory};
pub use resonator::{ResonatorConfig, ResonatorNetwork, ResonatorResult};
pub use resonator_attention::{
    AttentionResonator, AttentionResonatorConfig, AttentionResonatorResult, softmax_stable,
};
pub use spa_cleanup::SpaCleanup;
