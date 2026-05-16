//! HDC capacity and performance metrics.

use crate::distance::hamming::hamming_frac;
use crate::error::{HdcError, HdcResult};

/// Theoretical capacity of associative memory (Hopfield-like).
///
/// Classic Hopfield (1982) capacity: C ≈ 0.138 * D patterns can be stored and
/// reliably retrieved in an N-neuron Hopfield network with random binary patterns.
/// Amit, Gutfreund & Sompolinsky (1985) give the exact phase-transition point.
/// The tighter Hopfield bound is C ≈ D / (2 * ln(D)) for perfect retrieval; the
/// practical capacity with some errors is ~0.138 * D.
pub fn hopfield_capacity(dim: usize) -> usize {
    if dim < 2 {
        return 0;
    }
    // Use the Amit et al. practical capacity: ≈ 0.138 * D
    let cap = 0.138 * dim as f64;
    cap as usize
}

/// Empirical accuracy: correct / total classifications.
pub fn classification_accuracy(predictions: &[usize], labels: &[usize]) -> HdcResult<f64> {
    if predictions.len() != labels.len() {
        return Err(HdcError::DimensionMismatch {
            expected: labels.len(),
            got: predictions.len(),
        });
    }
    if predictions.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let correct = predictions
        .iter()
        .zip(labels.iter())
        .filter(|&(p, l)| p == l)
        .count();
    Ok(correct as f64 / predictions.len() as f64)
}

/// Expected Hamming distance between two independent random binary HVs ≈ 0.5.
pub fn expected_random_hamming() -> f64 {
    0.5
}

/// Signal-to-noise ratio for bundling K HVs of dimension D.
///
/// SNR = sqrt(D) / sqrt(K): K orthogonal patterns bundled, target signal is recovered.
pub fn bundle_snr(dim: usize, k: usize) -> HdcResult<f64> {
    if k == 0 {
        return Err(HdcError::EmptyInput);
    }
    if dim == 0 {
        return Err(HdcError::ZeroDimension);
    }
    Ok((dim as f64).sqrt() / (k as f64).sqrt())
}

/// Dimensionality required for P(collision) ≤ p_collision among n_items random HVs.
///
/// Based on Birthday paradox analog for binary HDC:
/// D ≥ 2 * ln(n_items^2 / (1 - p_collision))
pub fn required_dimension(n_items: usize, p_collision: f64) -> HdcResult<usize> {
    if n_items == 0 {
        return Err(HdcError::EmptyInput);
    }
    if p_collision <= 0.0 || p_collision >= 1.0 {
        return Err(HdcError::InvalidProbability(p_collision));
    }
    let n_sq = (n_items as f64).powi(2);
    let denom = 1.0 - p_collision;
    if denom <= 0.0 {
        return Err(HdcError::InvalidProbability(p_collision));
    }
    let dim = 2.0 * (n_sq / denom).ln();
    Ok(dim.ceil() as usize)
}

/// Orthogonality metric: average pairwise Hamming distance among a set of HVs.
/// An ideal set of random binary HVs should have ≈ 0.5 average pairwise Hamming distance.
pub fn average_pairwise_hamming(hvs: &[Vec<i8>]) -> HdcResult<f64> {
    if hvs.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    if hvs.len() == 1 {
        return Ok(0.0);
    }
    let n = hvs.len();
    let mut total = 0f64;
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            total += hamming_frac(&hvs[i], &hvs[j])?;
            count += 1;
        }
    }
    if count == 0 {
        return Err(HdcError::DivisionByZero);
    }
    Ok(total / count as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hopfield_capacity_d10000() {
        let cap = hopfield_capacity(10_000);
        assert!(
            cap >= 1000,
            "capacity for D=10000 should be >= 1000, got {cap}"
        );
    }

    #[test]
    fn classification_accuracy_all_correct() {
        let preds = vec![0, 1, 2, 0, 1];
        let labels = vec![0, 1, 2, 0, 1];
        let acc = classification_accuracy(&preds, &labels).expect("acc");
        assert!((acc - 1.0).abs() < 1e-9);
    }

    #[test]
    fn bundle_snr_k1_equals_sqrt_d() {
        let snr = bundle_snr(100, 1).expect("snr");
        assert!((snr - 10.0).abs() < 1e-9, "snr={snr}");
    }
}
