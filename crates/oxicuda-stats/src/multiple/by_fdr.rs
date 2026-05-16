//! Benjamini-Yekutieli FDR adjustment (handles arbitrary dependence).

use crate::error::{StatsError, StatsResult};

/// BY-FDR adjusted p-values.
pub fn by_fdr(p_values: &[f64]) -> StatsResult<Vec<f64>> {
    if p_values.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    let m = p_values.len();
    for &p in p_values {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::ProbabilityOutOfRange { value: p });
        }
    }
    // c_m = sum_{k=1..m} 1/k
    let c_m: f64 = (1..=m).map(|k| 1.0 / k as f64).sum();
    let mut indexed: Vec<(usize, f64)> = p_values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut adjusted = vec![0.0; m];
    let mut running_min: f64 = 1.0;
    for (rank, &(idx, p)) in indexed.iter().enumerate().rev() {
        let factor = m as f64 * c_m / (rank as f64 + 1.0);
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
    fn by_more_conservative_than_bh() {
        let p = [0.01, 0.04, 0.06];
        let bh = crate::multiple::bh_fdr::bh_fdr(&p).expect("ok");
        let by = by_fdr(&p).expect("ok");
        // BY should be uniformly >= BH (more conservative)
        for (a, b) in bh.iter().zip(&by) {
            assert!(b >= a, "BY={b} should be >= BH={a}");
        }
    }
}
