/// APrompt: learnable prompt key/value pairs injected into multi-head attention.
pub mod aprompt;
/// ATTEMPT: attentional mixture of soft prompts for multi-task prompt transfer.
pub mod attempt;
/// P-Tuning v2: per-layer deep prefix tuning.
pub mod p_tuning_v2;
/// Prefix-Tuning: prepend virtual key/value tokens to attention.
pub mod prefix_tuning;
/// Prompt Pool / L2P: pool of (key, prompt) pairs with top-N cosine selection.
pub mod prompt_pool;
/// Prompt-Tuning: prepend soft prompt embeddings to the input sequence.
pub mod prompt_tuning;
/// SPoT: Soft Prompt Transfer — cross-task prompt initialization via cosine retrieval.
pub mod spot;

pub use aprompt::{APrompt, APromptConfig};
pub use attempt::{AttemptConfig, AttemptRouter};
pub use prompt_pool::{PromptPool, PromptPoolConfig};
pub use spot::{SoftPromptLibrary, SourceTask, SpotConfig};
