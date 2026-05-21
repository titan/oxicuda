//! Character n-gram F-score (chrF and chrF++) for translation evaluation.
//!
//! References:
//! * Popović, M. (2015). *chrF: character n-gram F-score for automatic MT
//!   evaluation*. WMT 2015.
//!   <https://aclanthology.org/W15-3049/>.
//! * Popović, M. (2017). *chrF++: words helping character n-grams*.
//!   WMT 2017.
//!   <https://aclanthology.org/W17-4770/>.
//!
//! # ChrF formula
//!
//! For each character n-gram order `n ∈ 1..=max_char_n`:
//!
//! ```text
//! precision_n = |hyp_ngrams(n) ∩ ref_ngrams(n)| / |hyp_ngrams(n)|
//! recall_n    = |hyp_ngrams(n) ∩ ref_ngrams(n)| / |ref_ngrams(n)|
//! F_{β,n}     = (1+β²) · precision_n · recall_n
//!             / (β² · precision_n + recall_n + ε)
//! chrF        = mean_{n=1}^{max_char_n}(F_{β,n})
//! ```
//!
//! Intersection counts are computed as `Σ min(count_hyp, count_ref)` over
//! shared n-grams, allowing for **multiplicity** (the same n-gram occurring
//! several times in both strings contributes multiple matched copies).
//!
//! # ChrF++ extension
//!
//! ChrF++ (Popović 2017) augments the character n-gram scores with word
//! unigrams and bigrams, returning the mean over all `max_char_n + 2`
//! F-scores.  This adds a lexical-alignment signal that can distinguish
//! adequacy from fluency.
//!
//! # Unicode safety
//!
//! All character-level operations work on Rust [`char`] values, making them
//! correct for multi-byte UTF-8 characters (e.g., CJK, Arabic, accented
//! Latin) without any additional library dependencies.

use std::collections::HashMap;

use crate::error::{SeqError, SeqResult};

// -------------------------------------------------------------------------
// Low-level n-gram utilities
// -------------------------------------------------------------------------

/// Count character n-grams in `text`, returning a `HashMap<String, usize>`.
///
/// Each key is the string formed by joining `n` consecutive Unicode scalar
/// values; each value is the number of times that n-gram appears.
///
/// Returns an empty map if `text` has fewer than `n` characters or `n == 0`.
pub fn char_ngram_counts(text: &str, n: usize) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    if n == 0 {
        return counts;
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < n {
        return counts;
    }
    for i in 0..=chars.len() - n {
        let ngram: String = chars[i..i + n].iter().collect();
        *counts.entry(ngram).or_insert(0) += 1;
    }
    counts
}

/// Count word n-grams in `text`, returning a `HashMap<String, usize>`.
///
/// Words are defined as whitespace-delimited tokens and are lowercased before
/// any n-gram extraction.  Keys are word n-grams joined by a single space.
///
/// Returns an empty map if `text` has fewer than `n` whitespace-delimited
/// words, or if `n == 0`.
pub fn word_ngram_counts(text: &str, n: usize) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    if n == 0 {
        return counts;
    }
    let words: Vec<String> = text.split_whitespace().map(|w| w.to_lowercase()).collect();
    if words.len() < n {
        return counts;
    }
    for i in 0..=words.len() - n {
        let ngram = words[i..i + n].join(" ");
        *counts.entry(ngram).or_insert(0) += 1;
    }
    counts
}

/// Compute the intersection count of two n-gram frequency maps.
///
/// `|A ∩ B| = Σ_{g ∈ A} min(A[g], B[g])`
///
/// N-grams that appear only in `a` contribute 0.
pub fn ngram_intersection(a: &HashMap<String, usize>, b: &HashMap<String, usize>) -> usize {
    a.iter()
        .filter_map(|(ngram, &cnt_a)| b.get(ngram).map(|&cnt_b| cnt_a.min(cnt_b)))
        .sum()
}

// -------------------------------------------------------------------------
// F-score
// -------------------------------------------------------------------------

/// Compute the F_{β} score from precision and recall.
///
/// ```text
/// F_β = (1 + β²) · P · R / (β² · P + R + ε)
/// ```
///
/// Returns `0.0` if both `precision` and `recall` are below `1e-15`
/// (avoiding division by zero).
pub fn f_beta(precision: f64, recall: f64, beta: f64) -> f64 {
    const EPS: f64 = 1e-15;
    if precision + recall < EPS {
        return 0.0;
    }
    let beta2 = beta * beta;
    (1.0 + beta2) * precision * recall / (beta2 * precision + recall + EPS)
}

// -------------------------------------------------------------------------
// Single-sentence ChrF / ChrF++
// -------------------------------------------------------------------------

/// Compute the ChrF score for a single hypothesis–reference pair.
///
/// # Parameters
///
/// * `hypothesis` — the generated (system output) string.
/// * `reference` — the ground-truth string.
/// * `max_char_n` — highest character n-gram order to include (default 6).
/// * `beta` — F-score beta; β = 2 is the canonical ChrF choice (recall-heavy).
///
/// # Returns
///
/// The mean F_{β} over character n-gram orders 1..=`max_char_n`,
/// a value in `[0.0, 1.0]`.
pub fn chrf_score(hypothesis: &str, reference: &str, max_char_n: usize, beta: f64) -> f64 {
    if max_char_n == 0 {
        return 0.0;
    }
    let mut total_f = 0.0_f64;
    let mut n_orders = 0usize;

    for n in 1..=max_char_n {
        let hyp_counts = char_ngram_counts(hypothesis, n);
        let ref_counts = char_ngram_counts(reference, n);

        let hyp_total: usize = hyp_counts.values().sum();
        let ref_total: usize = ref_counts.values().sum();

        // If both sides have no n-grams of order n, skip this order to avoid
        // skewing the mean with spurious 0.0 entries.
        if hyp_total == 0 && ref_total == 0 {
            continue;
        }

        let matches = ngram_intersection(&hyp_counts, &ref_counts);
        let prec = matches as f64 / (hyp_total as f64 + 1e-15);
        let rec = matches as f64 / (ref_total as f64 + 1e-15);
        total_f += f_beta(prec, rec, beta);
        n_orders += 1;
    }

    if n_orders == 0 {
        return 0.0;
    }
    total_f / n_orders as f64
}

/// Compute the ChrF++ score for a single hypothesis–reference pair.
///
/// ChrF++ extends ChrF by additionally computing word unigrams and bigrams
/// (i.e., word 1-grams and 2-grams), then returning the mean of all
/// `max_char_n + 2` F-scores (character orders 1..=`max_char_n` plus word
/// unigrams and bigrams).
///
/// # Parameters
///
/// * `hypothesis`, `reference`, `max_char_n`, `beta` — same as [`chrf_score`].
///
/// # Returns
///
/// The ChrF++ score in `[0.0, 1.0]`.
pub fn chrf_plus_plus(hypothesis: &str, reference: &str, max_char_n: usize, beta: f64) -> f64 {
    // Gather char F-scores.
    let mut f_scores: Vec<f64> = Vec::new();

    for n in 1..=max_char_n {
        let hyp_counts = char_ngram_counts(hypothesis, n);
        let ref_counts = char_ngram_counts(reference, n);
        let hyp_total: usize = hyp_counts.values().sum();
        let ref_total: usize = ref_counts.values().sum();
        if hyp_total == 0 && ref_total == 0 {
            continue;
        }
        let matches = ngram_intersection(&hyp_counts, &ref_counts);
        let prec = matches as f64 / (hyp_total as f64 + 1e-15);
        let rec = matches as f64 / (ref_total as f64 + 1e-15);
        f_scores.push(f_beta(prec, rec, beta));
    }

    // Word unigrams and bigrams.
    for n in [1usize, 2] {
        let hyp_counts = word_ngram_counts(hypothesis, n);
        let ref_counts = word_ngram_counts(reference, n);
        let hyp_total: usize = hyp_counts.values().sum();
        let ref_total: usize = ref_counts.values().sum();
        // Include the word order even if totals are zero (contributes 0.0).
        let matches = ngram_intersection(&hyp_counts, &ref_counts);
        let prec = matches as f64 / (hyp_total as f64 + 1e-15);
        let rec = matches as f64 / (ref_total as f64 + 1e-15);
        f_scores.push(f_beta(prec, rec, beta));
    }

    if f_scores.is_empty() {
        return 0.0;
    }
    f_scores.iter().sum::<f64>() / f_scores.len() as f64
}

// -------------------------------------------------------------------------
// Corpus-level variants
// -------------------------------------------------------------------------

/// Corpus-level ChrF: unweighted average of per-sentence [`chrf_score`].
///
/// # Errors
///
/// * [`SeqError::EmptyInput`] if `hypotheses` is empty.
/// * [`SeqError::LengthMismatch`] if `hypotheses.len() != references.len()`.
pub fn corpus_chrf(
    hypotheses: &[&str],
    references: &[&str],
    max_char_n: usize,
    beta: f64,
) -> SeqResult<f64> {
    if hypotheses.is_empty() || references.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    if hypotheses.len() != references.len() {
        return Err(SeqError::LengthMismatch {
            a: hypotheses.len(),
            b: references.len(),
        });
    }
    let total: f64 = hypotheses
        .iter()
        .zip(references.iter())
        .map(|(&h, &r)| chrf_score(h, r, max_char_n, beta))
        .sum();
    Ok(total / hypotheses.len() as f64)
}

/// Corpus-level ChrF++: unweighted average of per-sentence
/// [`chrf_plus_plus`].
///
/// # Errors
///
/// * [`SeqError::EmptyInput`] if `hypotheses` is empty.
/// * [`SeqError::LengthMismatch`] if `hypotheses.len() != references.len()`.
pub fn corpus_chrf_plus_plus(
    hypotheses: &[&str],
    references: &[&str],
    max_char_n: usize,
    beta: f64,
) -> SeqResult<f64> {
    if hypotheses.is_empty() || references.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    if hypotheses.len() != references.len() {
        return Err(SeqError::LengthMismatch {
            a: hypotheses.len(),
            b: references.len(),
        });
    }
    let total: f64 = hypotheses
        .iter()
        .zip(references.iter())
        .map(|(&h, &r)| chrf_plus_plus(h, r, max_char_n, beta))
        .sum();
    Ok(total / hypotheses.len() as f64)
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // char_ngram_counts
    // -----------------------------------------------------------------------

    #[test]
    fn char_ngrams_bigrams_abc() {
        let counts = char_ngram_counts("abc", 2);
        assert_eq!(counts.get("ab").copied().unwrap_or(0), 1);
        assert_eq!(counts.get("bc").copied().unwrap_or(0), 1);
        assert_eq!(counts.len(), 2);
    }

    #[test]
    fn char_ngrams_n_greater_than_len_returns_empty() {
        let counts = char_ngram_counts("ab", 5);
        assert!(counts.is_empty());
    }

    #[test]
    fn char_ngrams_handles_zero_n() {
        let counts = char_ngram_counts("hello", 0);
        assert!(counts.is_empty());
    }

    #[test]
    fn char_ngrams_repeated_chars() {
        // "aaa" bigrams: "aa" appears twice.
        let counts = char_ngram_counts("aaa", 2);
        assert_eq!(counts.get("aa").copied().unwrap_or(0), 2);
        assert_eq!(counts.len(), 1);
    }

    #[test]
    fn char_ngrams_unicode_cjk() {
        // Two CJK characters: "你好"  chars = ['你','好'], unigrams = 2.
        let counts = char_ngram_counts("你好", 1);
        assert_eq!(counts.len(), 2);
        assert_eq!(counts.get("你").copied().unwrap_or(0), 1);
        assert_eq!(counts.get("好").copied().unwrap_or(0), 1);
    }

    // -----------------------------------------------------------------------
    // word_ngram_counts
    // -----------------------------------------------------------------------

    #[test]
    fn word_ngrams_unigrams_hello_world() {
        let counts = word_ngram_counts("hello world", 1);
        assert_eq!(counts.get("hello").copied().unwrap_or(0), 1);
        assert_eq!(counts.get("world").copied().unwrap_or(0), 1);
        assert_eq!(counts.len(), 2);
    }

    #[test]
    fn word_ngrams_bigrams_with_repetition() {
        // "A B A" → lowercased ["a","b","a"] → bigrams "a b", "b a".
        let counts = word_ngram_counts("A B A", 2);
        assert_eq!(counts.get("a b").copied().unwrap_or(0), 1);
        assert_eq!(counts.get("b a").copied().unwrap_or(0), 1);
        assert_eq!(counts.len(), 2);
    }

    #[test]
    fn word_ngrams_n_greater_than_word_count_returns_empty() {
        let counts = word_ngram_counts("hello", 3);
        assert!(counts.is_empty());
    }

    #[test]
    fn word_ngrams_empty_string_returns_empty() {
        let counts = word_ngram_counts("", 1);
        assert!(counts.is_empty());
    }

    #[test]
    fn word_ngrams_lowercases_tokens() {
        let counts = word_ngram_counts("Hello WORLD", 1);
        assert!(counts.contains_key("hello"));
        assert!(counts.contains_key("world"));
        assert!(!counts.contains_key("Hello"));
    }

    // -----------------------------------------------------------------------
    // ngram_intersection
    // -----------------------------------------------------------------------

    #[test]
    fn ngram_intersection_identical_maps() {
        let mut a = HashMap::new();
        a.insert("ab".to_string(), 2usize);
        a.insert("bc".to_string(), 3usize);
        let b = a.clone();
        assert_eq!(ngram_intersection(&a, &b), 5);
    }

    #[test]
    fn ngram_intersection_disjoint_maps() {
        let mut a = HashMap::new();
        a.insert("ab".to_string(), 2usize);
        let mut b = HashMap::new();
        b.insert("cd".to_string(), 3usize);
        assert_eq!(ngram_intersection(&a, &b), 0);
    }

    #[test]
    fn ngram_intersection_partial_overlap() {
        let mut a = HashMap::new();
        a.insert("ab".to_string(), 3usize);
        a.insert("cd".to_string(), 2usize);
        let mut b = HashMap::new();
        b.insert("ab".to_string(), 2usize); // min(3,2) = 2
        b.insert("ef".to_string(), 5usize);
        assert_eq!(ngram_intersection(&a, &b), 2);
    }

    // -----------------------------------------------------------------------
    // f_beta
    // -----------------------------------------------------------------------

    #[test]
    fn f_beta_both_zero_returns_zero() {
        assert_eq!(f_beta(0.0, 0.0, 1.0), 0.0);
    }

    #[test]
    fn f_beta_perfect_precision_and_recall() {
        let f = f_beta(1.0, 1.0, 1.0);
        assert!((f - 1.0).abs() < 1e-10, "got {f}");
    }

    #[test]
    fn f_beta_harmonic_mean_beta_one() {
        // P = 0.5, R = 1.0, beta = 1: F1 = 2*0.5*1.0 / (0.5+1.0) = 2/3.
        let f = f_beta(0.5, 1.0, 1.0);
        let expected = 2.0 * 0.5 / (0.5 + 1.0);
        assert!((f - expected).abs() < 1e-9, "got {f}");
    }

    #[test]
    fn f_beta_recall_heavy_beta_two() {
        // beta=2 weights recall twice as much as precision.
        // P=0.5, R=1.0: F2 = 5*0.5*1.0 / (4*0.5 + 1.0) = 2.5/3.0.
        let f = f_beta(0.5, 1.0, 2.0);
        let expected = 5.0 * 0.5 * 1.0 / (4.0 * 0.5 + 1.0);
        assert!((f - expected).abs() < 1e-9, "got {f}");
    }

    // -----------------------------------------------------------------------
    // chrf_score
    // -----------------------------------------------------------------------

    #[test]
    fn chrf_score_identical_strings_is_one() {
        let s = "the cat sat on the mat";
        let score = chrf_score(s, s, 6, 2.0);
        assert!((score - 1.0).abs() < 1e-9, "got {score}");
    }

    #[test]
    fn chrf_score_empty_hypothesis_is_zero() {
        let score = chrf_score("", "some reference text", 6, 2.0);
        assert!(score < 1e-10, "got {score}");
    }

    #[test]
    fn chrf_score_completely_different_strings_is_low() {
        // No shared characters at all.
        let score = chrf_score("aaaa", "bbbb", 6, 2.0);
        assert!(score < 1e-10, "got {score}");
    }

    #[test]
    fn chrf_score_partial_overlap_between_zero_and_one() {
        let hyp = "the cat";
        let reference = "the dog";
        let score = chrf_score(hyp, reference, 6, 2.0);
        assert!(score > 0.0 && score < 1.0, "got {score}");
    }

    #[test]
    fn chrf_score_max_char_n_zero_returns_zero() {
        let score = chrf_score("hello", "hello", 0, 2.0);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn chrf_score_symmetric_for_equal_length_identical() {
        // P == R when hyp == ref, so F is the same regardless of beta.
        let s = "same";
        let score_b1 = chrf_score(s, s, 4, 1.0);
        let score_b2 = chrf_score(s, s, 4, 2.0);
        assert!((score_b1 - 1.0).abs() < 1e-9);
        assert!((score_b2 - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // chrf_plus_plus
    // -----------------------------------------------------------------------

    #[test]
    fn chrf_plus_plus_identical_strings_is_one() {
        let s = "the quick brown fox";
        let score = chrf_plus_plus(s, s, 6, 2.0);
        assert!((score - 1.0).abs() < 1e-9, "got {score}");
    }

    #[test]
    fn chrf_plus_plus_different_strings_differs_from_chrf() {
        // ChrF++ may differ from ChrF when word n-grams add signal.
        // We just verify the score is in [0, 1].
        let hyp = "i like cats";
        let reference = "she loves dogs";
        let score = chrf_plus_plus(hyp, reference, 6, 2.0);
        assert!(
            (0.0..=1.0 + 1e-10).contains(&score),
            "score out of range: {score}"
        );
    }

    #[test]
    fn chrf_plus_plus_identical_has_same_score_as_chrf_is_one() {
        // Both metrics should give 1.0 on identical strings.
        let s = "hello world";
        let chrf = chrf_score(s, s, 6, 2.0);
        let chrfpp = chrf_plus_plus(s, s, 6, 2.0);
        assert!((chrf - 1.0).abs() < 1e-9);
        assert!((chrfpp - 1.0).abs() < 1e-9);
    }

    #[test]
    fn chrf_plus_plus_empty_hypothesis_is_low() {
        let score = chrf_plus_plus("", "some reference text here", 6, 2.0);
        assert!(score < 0.5, "got {score}");
    }

    // -----------------------------------------------------------------------
    // corpus_chrf
    // -----------------------------------------------------------------------

    #[test]
    fn corpus_chrf_all_perfect_is_one() {
        let hyps = vec!["hello world", "foo bar baz"];
        let refs = vec!["hello world", "foo bar baz"];
        let score = corpus_chrf(&hyps, &refs, 6, 2.0).expect("ok");
        assert!((score - 1.0).abs() < 1e-9, "got {score}");
    }

    #[test]
    fn corpus_chrf_empty_error() {
        let err = corpus_chrf(&[], &[], 6, 2.0).unwrap_err();
        assert!(matches!(err, SeqError::EmptyInput));
    }

    #[test]
    fn corpus_chrf_length_mismatch_error() {
        let hyps = vec!["hello"];
        let refs = vec!["hello", "world"];
        let err = corpus_chrf(&hyps, &refs, 6, 2.0).unwrap_err();
        assert!(matches!(err, SeqError::LengthMismatch { .. }));
    }

    #[test]
    fn corpus_chrf_single_sentence_matches_sentence_level() {
        let hyp = "the cat sat";
        let reference = "the dog sat";
        let sentence = chrf_score(hyp, reference, 6, 2.0);
        let corpus = corpus_chrf(&[hyp], &[reference], 6, 2.0).expect("ok");
        assert!((sentence - corpus).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // corpus_chrf_plus_plus
    // -----------------------------------------------------------------------

    #[test]
    fn corpus_chrf_plus_plus_all_perfect_is_one() {
        let hyps = vec!["the quick brown fox", "jumps over the lazy dog"];
        let refs = vec!["the quick brown fox", "jumps over the lazy dog"];
        let score = corpus_chrf_plus_plus(&hyps, &refs, 6, 2.0).expect("ok");
        assert!((score - 1.0).abs() < 1e-9, "got {score}");
    }

    #[test]
    fn corpus_chrf_plus_plus_empty_error() {
        let err = corpus_chrf_plus_plus(&[], &[], 6, 2.0).unwrap_err();
        assert!(matches!(err, SeqError::EmptyInput));
    }

    #[test]
    fn corpus_chrf_plus_plus_length_mismatch_error() {
        let hyps = vec!["a", "b", "c"];
        let refs = vec!["a", "b"];
        let err = corpus_chrf_plus_plus(&hyps, &refs, 6, 2.0).unwrap_err();
        assert!(matches!(err, SeqError::LengthMismatch { .. }));
    }

    #[test]
    fn corpus_chrf_plus_plus_single_sentence_matches_sentence_level() {
        let hyp = "translation output here";
        let reference = "the reference sentence here";
        let sentence = chrf_plus_plus(hyp, reference, 6, 2.0);
        let corpus = corpus_chrf_plus_plus(&[hyp], &[reference], 6, 2.0).expect("ok");
        assert!((sentence - corpus).abs() < 1e-12);
    }

    #[test]
    fn chrf_score_in_range_for_partial_match() {
        let hyp = "the cat sat on the mat";
        let reference = "the cat sat on the floor";
        let score = chrf_score(hyp, reference, 6, 2.0);
        assert!(
            score > 0.5 && score < 1.0,
            "expected partial match score, got {score}"
        );
    }
}
