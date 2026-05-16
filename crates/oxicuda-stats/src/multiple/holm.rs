//! Holm-Bonferroni step-down adjustment.

use crate::error::{StatsError, StatsResult};

/// Holm-Bonferroni step-down adjustment.
///
/// Returns adjusted p-values aligned with the input ordering.
pub fn holm(p_values: &[f64]) -> StatsResult<Vec<f64>> {
    if p_values.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    let m = p_values.len();
    // Validate
    for &p in p_values {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::ProbabilityOutOfRange { value: p });
        }
    }
    let mut indexed: Vec<(usize, f64)> = p_values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut adjusted = vec![0.0; m];
    let mut running_max: f64 = 0.0;
    for (rank, &(idx, p)) in indexed.iter().enumerate() {
        let factor = (m - rank) as f64;
        let adj = (factor * p).min(1.0);
        running_max = running_max.max(adj);
        adjusted[idx] = running_max;
    }
    Ok(adjusted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holm_basic() {
        let p = [0.01, 0.04, 0.06];
        let adj = holm(&p).expect("ok");
        // First: 0.01*3 = 0.03; next: max(0.03, 0.04*2)=0.08; last: max(0.08, 0.06*1)=0.08
        assert!((adj[0] - 0.03).abs() < 1e-12);
        assert!((adj[1] - 0.08).abs() < 1e-12);
        assert!((adj[2] - 0.08).abs() < 1e-12);
    }
}
