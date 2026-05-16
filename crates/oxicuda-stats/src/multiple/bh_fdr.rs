//! Benjamini-Hochberg FDR adjustment.

use crate::error::{StatsError, StatsResult};

/// BH-FDR adjusted p-values.
pub fn bh_fdr(p_values: &[f64]) -> StatsResult<Vec<f64>> {
    if p_values.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    let m = p_values.len();
    for &p in p_values {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::ProbabilityOutOfRange { value: p });
        }
    }
    let mut indexed: Vec<(usize, f64)> = p_values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut adjusted = vec![0.0; m];
    let mut running_min: f64 = 1.0;
    // Walk from largest p down: q = p * m / rank, monotonized
    for (rank, &(idx, p)) in indexed.iter().enumerate().rev() {
        let factor = m as f64 / (rank as f64 + 1.0);
        let adj = (factor * p).min(1.0);
        running_min = running_min.min(adj);
        adjusted[idx] = running_min;
    }
    Ok(adjusted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bh_fdr_basic() {
        let p = [0.01, 0.04, 0.06, 0.5];
        let adj = bh_fdr(&p).expect("ok");
        // Standard R: 0.04, 0.08, 0.08, 0.5
        assert!((adj[0] - 0.04).abs() < 1e-12);
        assert!((adj[1] - 0.08).abs() < 1e-12);
        assert!((adj[2] - 0.08).abs() < 1e-12);
        assert!((adj[3] - 0.5).abs() < 1e-12);
    }
}
