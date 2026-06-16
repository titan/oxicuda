//! Tag-scheme utilities for sequence labelling: BIO / BIOES conversion and
//! validation, entity-span extraction, and span-based (entity-level) F1
//! (`seqeval` semantics).

pub mod bioes;
pub mod span_f1;

pub use bioes::{
    Span, Tag, bio_to_bioes, bioes_to_bio, extract_spans, parse_tags, validate_bio, validate_bioes,
};
pub use span_f1::{PrfScore, SpanF1Report, span_f1, span_f1_from_spans};
