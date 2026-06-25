//! BPE tokenizer and vocabulary management.
//!
//! # Usage
//!
//! ```ignore
//! use oxicuda_lm::tokenizer::{BpeBuilder, BpeTokenizer};
//!
//! let tokenizer = BpeBuilder::new()
//!     .add_merge(b"a", b"b")   // "ab" → id 256
//!     .add_special("<eos>", 0)
//!     .build()?;
//!
//! let ids = tokenizer.encode("ab")?;
//! assert_eq!(ids, vec![256]);
//! let text = tokenizer.decode(&ids)?;
//! assert_eq!(text, "ab");
//! ```

pub mod bpe;
pub mod unigram;
pub mod vocab;
pub mod wordpiece;

pub use bpe::{BpeBuilder, BpeTokenizer};
pub use unigram::UnigramTokenizer;
pub use vocab::Vocab;
pub use wordpiece::{WordPieceTokenizer, basic_pretokenize};
