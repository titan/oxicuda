//! Bonferroni multiple-comparison adjustment.

use crate::error::{StatsError, StatsResult};

/// Multiply each p-value by `m` (number of tests), clipped to `[0, 1]`.
pub fn bonferroni(p_values: &[f64]) -> StatsResult<Vec<f64>> {
    if p_values.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    let m = p_values.len() as f64;
    let mut out = Vec::with_capacity(p_values.len());
    for &p in p_values {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::ProbabilityOutOfRange { value: p });
        }
        out.push((p * m).min(1.0));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bonferroni_basic() {
        let p = [0.01, 0.04, 0.6];
        let adj = bonferroni(&p).expect("ok");
        assert!((adj[0] - 0.03).abs() < 1e-12);
        assert!((adj[1] - 0.12).abs() < 1e-12);
        assert!((adj[2] - 1.0).abs() < 1e-12);
    }
}
