//! BIO / BIOES tag-scheme conversion, validation, and span extraction.
//!
//! Sequence-labelling tasks (NER, chunking) encode multi-token spans with a
//! *tagging scheme*.  Two common schemes are:
//!
//! * **BIO** (a.k.a. IOB2): `B-X` begins a span of type `X`, `I-X` continues it,
//!   `O` is outside any span.
//! * **BIOES** (a.k.a. BILOU): adds `E-X` (end of a multi-token span) and `S-X`
//!   (single-token span), which gives the model an explicit "end" signal and
//!   often improves boundary accuracy.
//!
//! A tag is represented as a [`Tag`] enum.  Tags are parsed from / formatted to
//! the canonical `"<prefix>-<type>"` string (or `"O"`).  This module provides:
//!
//! * [`Tag::parse`] / [`Tag::to_tag_string`] — string ⇆ enum.
//! * [`bio_to_bioes`] / [`bioes_to_bio`] — scheme conversion.
//! * [`validate_bio`] / [`validate_bioes`] — well-formedness checks.
//! * [`extract_spans`] — robust span extraction (seqeval semantics).

use crate::error::{SeqError, SeqResult};

// ─── Tag ─────────────────────────────────────────────────────────────────────

/// A single tagging-scheme tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tag {
    /// Outside any span.
    Outside,
    /// Begin a span of the given type.
    Begin(String),
    /// Inside (continue) a span of the given type.
    Inside(String),
    /// End of a multi-token span of the given type (BIOES only).
    End(String),
    /// Single-token span of the given type (BIOES only).
    Single(String),
}

impl Tag {
    /// Parse a tag string such as `"B-PER"`, `"I-LOC"`, `"S-ORG"`, or `"O"`.
    ///
    /// The case-sensitive single-letter prefix selects the variant; the
    /// remainder after the `-` is the entity type.
    ///
    /// # Errors
    ///
    /// [`SeqError::InvalidObservation`] for an unrecognised prefix or a missing
    /// `-type` body on a non-`O` tag.
    pub fn parse(s: &str) -> SeqResult<Tag> {
        if s == "O" {
            return Ok(Tag::Outside);
        }
        let mut parts = s.splitn(2, '-');
        let prefix = parts.next().unwrap_or("");
        let body = parts.next();
        let ty = match body {
            Some(b) if !b.is_empty() => b.to_string(),
            _ => {
                return Err(SeqError::InvalidObservation(format!(
                    "tag '{s}' is missing an entity type"
                )));
            }
        };
        match prefix {
            "B" => Ok(Tag::Begin(ty)),
            "I" => Ok(Tag::Inside(ty)),
            "E" => Ok(Tag::End(ty)),
            "S" => Ok(Tag::Single(ty)),
            other => Err(SeqError::InvalidObservation(format!(
                "unknown tag prefix '{other}' in '{s}'"
            ))),
        }
    }

    /// Format this tag back into its canonical string.
    #[must_use]
    pub fn to_tag_string(&self) -> String {
        match self {
            Tag::Outside => "O".to_string(),
            Tag::Begin(t) => format!("B-{t}"),
            Tag::Inside(t) => format!("I-{t}"),
            Tag::End(t) => format!("E-{t}"),
            Tag::Single(t) => format!("S-{t}"),
        }
    }

    /// The entity type (`None` for [`Tag::Outside`]).
    #[must_use]
    pub fn entity_type(&self) -> Option<&str> {
        match self {
            Tag::Outside => None,
            Tag::Begin(t) | Tag::Inside(t) | Tag::End(t) | Tag::Single(t) => Some(t.as_str()),
        }
    }
}

/// Parse a whole slice of tag strings into [`Tag`] values.
///
/// # Errors
///
/// Propagates the first [`Tag::parse`] failure.
pub fn parse_tags(tags: &[&str]) -> SeqResult<Vec<Tag>> {
    tags.iter().map(|t| Tag::parse(t)).collect()
}

// ─── Span ────────────────────────────────────────────────────────────────────

/// An entity span: `(entity_type, start, end)` with `end` **inclusive**.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Span {
    /// Entity type label.
    pub entity_type: String,
    /// Inclusive start index.
    pub start: usize,
    /// Inclusive end index.
    pub end: usize,
}

/// Extract entity spans from a tag sequence using seqeval-style semantics.
///
/// A new span opens whenever:
/// * the tag is `B-X` or `S-X`, **or**
/// * the tag is `I-X`/`E-X` but the previous position did not continue an open
///   span of the same type `X` (robust handling of malformed input).
///
/// A span closes at an `E-X`/`S-X`, when the type changes, or at `O`.  This
/// matches the behaviour of `seqeval.metrics.sequence_labeling.get_entities`.
#[must_use]
pub fn extract_spans(tags: &[Tag]) -> Vec<Span> {
    let mut spans = Vec::new();
    let mut open: Option<(String, usize)> = None; // (type, start)

    for (i, tag) in tags.iter().enumerate() {
        // Decide whether the currently-open span must close *before* this tag.
        let close_before = match (&open, tag) {
            (Some(_), Tag::Begin(_)) | (Some(_), Tag::Single(_)) | (Some(_), Tag::Outside) => true,
            (Some((ot, _)), Tag::Inside(t)) | (Some((ot, _)), Tag::End(t)) => ot != t,
            _ => false,
        };
        if close_before {
            if let Some((ot, os)) = open.take() {
                spans.push(Span {
                    entity_type: ot,
                    start: os,
                    end: i - 1,
                });
            }
        }

        match tag {
            Tag::Outside => {}
            Tag::Begin(t) | Tag::Single(t) => {
                // Open a fresh span starting here.
                open = Some((t.clone(), i));
            }
            Tag::Inside(t) | Tag::End(t) => {
                if open.is_none() {
                    // Dangling I/E: treat as the beginning of a new span.
                    open = Some((t.clone(), i));
                }
            }
        }

        // S/E close immediately *after* being counted.
        if matches!(tag, Tag::Single(_) | Tag::End(_)) {
            if let Some((ot, os)) = open.take() {
                spans.push(Span {
                    entity_type: ot,
                    start: os,
                    end: i,
                });
            }
        }
    }
    // Flush a trailing open span.
    if let Some((ot, os)) = open.take() {
        spans.push(Span {
            entity_type: ot,
            start: os,
            end: tags.len() - 1,
        });
    }
    spans
}

// ─── Conversion ──────────────────────────────────────────────────────────────

/// Convert a BIO tag sequence to BIOES.
///
/// Single-token spans become `S-X`; the final token of a multi-token span
/// becomes `E-X`.  The input need not be perfectly well-formed: spans are first
/// extracted with [`extract_spans`] (which repairs dangling `I-`), then
/// re-emitted in BIOES.
///
/// # Errors
///
/// [`SeqError::EmptyInput`] if `tags` is empty.
pub fn bio_to_bioes(tags: &[Tag]) -> SeqResult<Vec<Tag>> {
    if tags.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let spans = extract_spans(tags);
    let mut out = vec![Tag::Outside; tags.len()];
    for sp in spans {
        if sp.start == sp.end {
            out[sp.start] = Tag::Single(sp.entity_type);
        } else {
            out[sp.start] = Tag::Begin(sp.entity_type.clone());
            for idx in sp.start + 1..sp.end {
                out[idx] = Tag::Inside(sp.entity_type.clone());
            }
            out[sp.end] = Tag::End(sp.entity_type);
        }
    }
    Ok(out)
}

/// Convert a BIOES tag sequence back to BIO.
///
/// `S-X` becomes `B-X`; `E-X` becomes `I-X`.  Spans are extracted first to
/// normalise any malformed input.
///
/// # Errors
///
/// [`SeqError::EmptyInput`] if `tags` is empty.
pub fn bioes_to_bio(tags: &[Tag]) -> SeqResult<Vec<Tag>> {
    if tags.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let spans = extract_spans(tags);
    let mut out = vec![Tag::Outside; tags.len()];
    for sp in spans {
        out[sp.start] = Tag::Begin(sp.entity_type.clone());
        for idx in sp.start + 1..=sp.end {
            out[idx] = Tag::Inside(sp.entity_type.clone());
        }
    }
    Ok(out)
}

// ─── Validation ──────────────────────────────────────────────────────────────

/// Validate that a tag sequence is well-formed **BIO** (IOB2): every `I-X` must
/// be preceded by a `B-X` or `I-X` of the *same* type, and no `E`/`S` tags are
/// allowed.
///
/// # Errors
///
/// [`SeqError::GraphInvariantViolated`] describing the first violation.
pub fn validate_bio(tags: &[Tag]) -> SeqResult<()> {
    let mut prev: Option<&Tag> = None;
    for (i, tag) in tags.iter().enumerate() {
        match tag {
            Tag::End(_) | Tag::Single(_) => {
                return Err(SeqError::GraphInvariantViolated(format!(
                    "BIO sequence contains BIOES tag '{}' at {i}",
                    tag.to_tag_string()
                )));
            }
            Tag::Inside(t) => {
                let ok = matches!(prev, Some(Tag::Begin(p)) | Some(Tag::Inside(p)) if p == t);
                if !ok {
                    return Err(SeqError::GraphInvariantViolated(format!(
                        "I-{t} at {i} not preceded by B-{t}/I-{t}"
                    )));
                }
            }
            _ => {}
        }
        prev = Some(tag);
    }
    Ok(())
}

/// Validate that a tag sequence is well-formed **BIOES**.
///
/// Rules enforced:
/// * `I-X` and `E-X` must continue an *open* span of type `X` (i.e. the previous
///   tag is `B-X` or `I-X`).
/// * `B-X` must be *closed* by a subsequent `E-X` (via `I-X`*) — i.e. a span
///   that starts with `B` may not be followed by `O`/`B`/`S` or a type change
///   before an `E`.
///
/// # Errors
///
/// [`SeqError::GraphInvariantViolated`] describing the first violation.
pub fn validate_bioes(tags: &[Tag]) -> SeqResult<()> {
    let n = tags.len();
    for i in 0..n {
        match &tags[i] {
            Tag::Inside(t) | Tag::End(t) => {
                let prev_ok =
                    i > 0 && matches!(&tags[i - 1], Tag::Begin(p) | Tag::Inside(p) if p == t);
                if !prev_ok {
                    return Err(SeqError::GraphInvariantViolated(format!(
                        "{} at {i} does not continue an open span of type {t}",
                        tags[i].to_tag_string()
                    )));
                }
            }
            Tag::Begin(t) => {
                // Must be followed by I-t or E-t (a span cannot end with B).
                let next_ok =
                    i + 1 < n && matches!(&tags[i + 1], Tag::Inside(p) | Tag::End(p) if p == t);
                if !next_ok {
                    return Err(SeqError::GraphInvariantViolated(format!(
                        "B-{t} at {i} is not continued by I-{t}/E-{t}"
                    )));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tags(strs: &[&str]) -> Vec<Tag> {
        parse_tags(strs).expect("parse")
    }

    #[test]
    fn parse_outside_and_typed() {
        assert_eq!(Tag::parse("O").expect("o"), Tag::Outside);
        assert_eq!(Tag::parse("B-PER").expect("b"), Tag::Begin("PER".into()));
        assert_eq!(Tag::parse("I-LOC").expect("i"), Tag::Inside("LOC".into()));
        assert_eq!(Tag::parse("E-ORG").expect("e"), Tag::End("ORG".into()));
        assert_eq!(Tag::parse("S-MISC").expect("s"), Tag::Single("MISC".into()));
    }

    #[test]
    fn parse_handles_hyphenated_types() {
        // Entity type may itself contain a hyphen.
        assert_eq!(
            Tag::parse("B-GPE-CITY").expect("ok"),
            Tag::Begin("GPE-CITY".into())
        );
    }

    #[test]
    fn parse_rejects_bad_prefix_and_empty_type() {
        assert!(Tag::parse("X-PER").is_err());
        assert!(Tag::parse("B-").is_err());
        assert!(Tag::parse("B").is_err());
    }

    #[test]
    fn to_tag_string_roundtrip() {
        for s in ["O", "B-PER", "I-PER", "E-PER", "S-LOC"] {
            assert_eq!(Tag::parse(s).expect("p").to_tag_string(), s);
        }
    }

    #[test]
    fn extract_single_and_multi_spans() {
        // "B-PER I-PER O S-LOC" → PER[0..1], LOC[3..3]
        let t = tags(&["B-PER", "I-PER", "O", "S-LOC"]);
        let spans = extract_spans(&t);
        assert_eq!(spans.len(), 2);
        assert_eq!(
            spans[0],
            Span {
                entity_type: "PER".into(),
                start: 0,
                end: 1
            }
        );
        assert_eq!(
            spans[1],
            Span {
                entity_type: "LOC".into(),
                start: 3,
                end: 3
            }
        );
    }

    #[test]
    fn extract_adjacent_same_type_spans() {
        // "B-PER B-PER" → two separate PER spans.
        let t = tags(&["B-PER", "B-PER"]);
        let spans = extract_spans(&t);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].start, 0);
        assert_eq!(spans[0].end, 0);
        assert_eq!(spans[1].start, 1);
        assert_eq!(spans[1].end, 1);
    }

    #[test]
    fn extract_dangling_inside_starts_new_span() {
        // Malformed: "I-PER I-PER" with no B → still one PER span [0..1].
        let t = tags(&["I-PER", "I-PER"]);
        let spans = extract_spans(&t);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].start, 0);
        assert_eq!(spans[0].end, 1);
    }

    #[test]
    fn extract_type_change_splits() {
        // "B-PER I-LOC" → PER[0] then LOC[1].
        let t = tags(&["B-PER", "I-LOC"]);
        let spans = extract_spans(&t);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].entity_type, "PER");
        assert_eq!(spans[0].end, 0);
        assert_eq!(spans[1].entity_type, "LOC");
        assert_eq!(spans[1].start, 1);
    }

    #[test]
    fn bio_to_bioes_basic() {
        // B-PER I-PER O B-LOC  →  B-PER E-PER O S-LOC
        let bio = tags(&["B-PER", "I-PER", "O", "B-LOC"]);
        let bioes = bio_to_bioes(&bio).expect("conv");
        let got: Vec<String> = bioes.iter().map(Tag::to_tag_string).collect();
        assert_eq!(got, vec!["B-PER", "E-PER", "O", "S-LOC"]);
    }

    #[test]
    fn bioes_to_bio_basic() {
        // B-PER E-PER O S-LOC → B-PER I-PER O B-LOC
        let bioes = tags(&["B-PER", "E-PER", "O", "S-LOC"]);
        let bio = bioes_to_bio(&bioes).expect("conv");
        let got: Vec<String> = bio.iter().map(Tag::to_tag_string).collect();
        assert_eq!(got, vec!["B-PER", "I-PER", "O", "B-LOC"]);
    }

    #[test]
    fn conversion_roundtrip_preserves_spans() {
        let bio = tags(&["O", "B-PER", "I-PER", "I-PER", "O", "B-LOC", "O"]);
        let bioes = bio_to_bioes(&bio).expect("a");
        let back = bioes_to_bio(&bioes).expect("b");
        assert_eq!(extract_spans(&bio), extract_spans(&back));
    }

    #[test]
    fn convert_rejects_empty() {
        assert!(matches!(bio_to_bioes(&[]), Err(SeqError::EmptyInput)));
        assert!(matches!(bioes_to_bio(&[]), Err(SeqError::EmptyInput)));
    }

    #[test]
    fn validate_bio_accepts_well_formed() {
        let t = tags(&["O", "B-PER", "I-PER", "O", "B-LOC"]);
        assert!(validate_bio(&t).is_ok());
    }

    #[test]
    fn validate_bio_rejects_dangling_inside() {
        let t = tags(&["O", "I-PER"]);
        assert!(matches!(
            validate_bio(&t),
            Err(SeqError::GraphInvariantViolated(_))
        ));
    }

    #[test]
    fn validate_bio_rejects_type_mismatch_continuation() {
        let t = tags(&["B-PER", "I-LOC"]);
        assert!(validate_bio(&t).is_err());
    }

    #[test]
    fn validate_bio_rejects_bioes_tags() {
        let t = tags(&["S-PER"]);
        assert!(validate_bio(&t).is_err());
    }

    #[test]
    fn validate_bioes_accepts_well_formed() {
        let t = tags(&["B-PER", "I-PER", "E-PER", "O", "S-LOC"]);
        assert!(validate_bioes(&t).is_ok());
    }

    #[test]
    fn validate_bioes_rejects_begin_without_end() {
        // B-PER followed by O is illegal in BIOES.
        let t = tags(&["B-PER", "O"]);
        assert!(matches!(
            validate_bioes(&t),
            Err(SeqError::GraphInvariantViolated(_))
        ));
    }

    #[test]
    fn validate_bioes_rejects_dangling_end() {
        let t = tags(&["E-PER"]);
        assert!(validate_bioes(&t).is_err());
    }
}
