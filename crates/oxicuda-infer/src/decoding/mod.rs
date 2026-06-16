//! Decoding strategies for autoregressive language model generation.
//!
//! # Modules
//!
//! | Module | Algorithm |
//! |--------|-----------|
//! | [`mod@beam_search`] | Functional beam search with length normalisation |
//! | [`prompt_lookup`] | N-gram prompt-lookup drafting + no-repeat-ngram blocking |

pub mod beam_search;
pub mod prompt_lookup;
pub use beam_search::{BeamCandidate, BeamConfig, beam_search};
pub use prompt_lookup::{PromptLookupDecoder, no_repeat_ngram_banned};
