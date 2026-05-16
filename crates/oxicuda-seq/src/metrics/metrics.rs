//! Sequence-model evaluation metrics.

use crate::error::{SeqError, SeqResult};

/// Token-level accuracy across a list of (prediction, reference) pairs.
pub fn token_accuracy(predictions: &[Vec<usize>], references: &[Vec<usize>]) -> SeqResult<f64> {
    if predictions.len() != references.len() {
        return Err(SeqError::LengthMismatch {
            a: predictions.len(),
            b: references.len(),
        });
    }
    let mut correct = 0usize;
    let mut total = 0usize;
    for (p, r) in predictions.iter().zip(references.iter()) {
        if p.len() != r.len() {
            return Err(SeqError::LengthMismatch {
                a: p.len(),
                b: r.len(),
            });
        }
        for (a, b) in p.iter().zip(r.iter()) {
            if a == b {
                correct += 1;
            }
            total += 1;
        }
    }
    if total == 0 {
        return Err(SeqError::EmptyInput);
    }
    Ok(correct as f64 / total as f64)
}

/// Sequence-level (exact-match) accuracy.
pub fn sequence_accuracy(predictions: &[Vec<usize>], references: &[Vec<usize>]) -> SeqResult<f64> {
    if predictions.len() != references.len() {
        return Err(SeqError::LengthMismatch {
            a: predictions.len(),
            b: references.len(),
        });
    }
    if predictions.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let mut correct = 0usize;
    for (p, r) in predictions.iter().zip(references.iter()) {
        if p == r {
            correct += 1;
        }
    }
    Ok(correct as f64 / predictions.len() as f64)
}

/// Levenshtein edit distance between two slices.
pub fn edit_distance<T: Eq>(a: &[T], b: &[T]) -> usize {
    let m = a.len();
    let n = b.len();
    let cols = n + 1;
    let mut dp = vec![0usize; (m + 1) * cols];
    for i in 0..=m {
        dp[i * cols] = i;
    }
    for j in 0..=n {
        dp[j] = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            let del = dp[(i - 1) * cols + j] + 1;
            let ins = dp[i * cols + (j - 1)] + 1;
            let sub = dp[(i - 1) * cols + (j - 1)] + cost;
            dp[i * cols + j] = del.min(ins).min(sub);
        }
    }
    dp[m * cols + n]
}

/// BLEU-n score with add-1 smoothing.  Reference: BLEU paper (Papineni 2002).
pub fn bleu_n(prediction: &[usize], reference: &[usize], n: usize) -> SeqResult<f64> {
    if n == 0 {
        return Err(SeqError::InvalidConfiguration("n must be > 0".to_string()));
    }
    if prediction.is_empty() || reference.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let mut log_prec = 0.0;
    for order in 1..=n {
        let (matches, total) = count_ngrams(prediction, reference, order);
        let m = matches as f64 + 1.0;
        let t = total as f64 + 1.0;
        log_prec += (m / t).ln();
    }
    log_prec /= n as f64;
    let pred_len = prediction.len() as f64;
    let ref_len = reference.len() as f64;
    let bp = if pred_len >= ref_len {
        1.0
    } else {
        (1.0 - ref_len / pred_len).exp()
    };
    Ok(bp * log_prec.exp())
}

fn count_ngrams(prediction: &[usize], reference: &[usize], n: usize) -> (usize, usize) {
    let mut matches = 0usize;
    let total = if prediction.len() >= n {
        prediction.len() - n + 1
    } else {
        0
    };
    for i in 0..total {
        let pred_ngram = &prediction[i..i + n];
        let mut found = false;
        if reference.len() >= n {
            for j in 0..=reference.len() - n {
                if &reference[j..j + n] == pred_ngram {
                    found = true;
                    break;
                }
            }
        }
        if found {
            matches += 1;
        }
    }
    (matches, total)
}

/// Perplexity given a sequence of conditional probabilities of the true token
/// at each step.  `probs[t]` ∈ (0, 1].
pub fn perplexity(probs: &[f64]) -> SeqResult<f64> {
    if probs.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let mut log_sum = 0.0;
    for &p in probs {
        if !(0.0..=1.0 + 1e-12).contains(&p) {
            return Err(SeqError::ProbabilityOutOfRange(p));
        }
        let pp = p.max(1e-300);
        log_sum += pp.ln();
    }
    let avg = log_sum / probs.len() as f64;
    Ok((-avg).exp())
}

/// Log loss / cross-entropy over a list of (true_label, predicted_probs) pairs.
pub fn log_loss(true_labels: &[usize], pred_probs: &[Vec<f64>]) -> SeqResult<f64> {
    if true_labels.len() != pred_probs.len() {
        return Err(SeqError::LengthMismatch {
            a: true_labels.len(),
            b: pred_probs.len(),
        });
    }
    if true_labels.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let mut total = 0.0;
    for (y, p) in true_labels.iter().zip(pred_probs.iter()) {
        if *y >= p.len() {
            return Err(SeqError::IndexOutOfBounds {
                index: *y,
                len: p.len(),
            });
        }
        let pp = p[*y].max(1e-300);
        total -= pp.ln();
    }
    Ok(total / true_labels.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levenshtein_kitten_sitting() {
        let a: Vec<u8> = b"kitten".to_vec();
        let b: Vec<u8> = b"sitting".to_vec();
        assert_eq!(edit_distance(&a, &b), 3);
    }

    #[test]
    fn token_accuracy_basic() {
        let p = vec![vec![0, 1, 2]];
        let r = vec![vec![0, 1, 2]];
        let acc = token_accuracy(&p, &r).expect("ok");
        assert!((acc - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bleu_identical_one() {
        let a = vec![1, 2, 3, 4];
        let b = a.clone();
        let s = bleu_n(&a, &b, 1).expect("ok");
        assert!((s - 1.0).abs() < 1e-9, "bleu={s}");
    }

    #[test]
    fn perplexity_basic() {
        let p = vec![1.0, 1.0, 1.0];
        let pp = perplexity(&p).expect("ok");
        assert!((pp - 1.0).abs() < 1e-12);
    }

    #[test]
    fn log_loss_zero_for_correct() {
        let y = vec![0, 1];
        let p = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let l = log_loss(&y, &p).expect("ok");
        assert!(l < 1e-9);
    }
}
