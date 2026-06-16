//! Memory modules: item memory (symbol→HV store), associative/Hopfield memory, SPA cleanup
//! memory over real unit vectors, and VSA Resonator Networks.

pub mod assoc_memory;
pub mod hetero_associative;
pub mod item_memory;
pub mod resonator;
pub mod spa_cleanup;

pub use hetero_associative::{HeteroAssocConfig, HeteroAssociativeMemory};
pub use resonator::{ResonatorConfig, ResonatorNetwork, ResonatorResult};
pub use spa_cleanup::SpaCleanup;
