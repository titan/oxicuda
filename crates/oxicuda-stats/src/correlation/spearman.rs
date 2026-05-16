//! Spearman rank correlation (Pearson on ranks).

use crate::correlation::pearson::{PearsonResult, pearson_r};
use crate::error::StatsResult;
use crate::nonparametric::rank_with_ties;

/// Spearman rank-correlation result.
pub type SpearmanResult = PearsonResult;

/// Spearman rho via Pearson r applied to mid-ranks.
pub fn spearman_rho(x: &[f64], y: &[f64]) -> StatsResult<SpearmanResult> {
    let rx = rank_with_ties(x);
    let ry = rank_with_ties(y);
    pearson_r(&rx, &ry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spearman_monotone_positive() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 4.0, 9.0, 16.0, 25.0];
        let r = spearman_rho(&x, &y).expect("ok");
        assert!((r.r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn spearman_monotone_negative() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [25.0, 16.0, 9.0, 4.0, 1.0];
        let r = spearman_rho(&x, &y).expect("ok");
        assert!((r.r + 1.0).abs() < 1e-9);
    }
}
