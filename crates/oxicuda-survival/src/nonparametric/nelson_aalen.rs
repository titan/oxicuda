//! Nelson-Aalen cumulative hazard estimator.

use crate::data::{Dataset, RiskSet};
use crate::error::SurvivalResult;

/// Nelson-Aalen estimator output.
///
/// - `times[i]`         — unique event time t_i
/// - `cum_hazard[i]`    — Ĥ(t_i) = Σ_{k<=i} d_k / n_k
/// - `variance[i]`      — Var(Ĥ(t_i)) = Σ_{k<=i} d_k / n_k²
#[derive(Debug, Clone)]
pub struct NelsonAalen {
    pub times: Vec<f64>,
    pub cum_hazard: Vec<f64>,
    pub variance: Vec<f64>,
}

impl NelsonAalen {
    /// Standard error of Ĥ.
    #[must_use]
    pub fn standard_error(&self) -> Vec<f64> {
        self.variance.iter().map(|v| v.max(0.0).sqrt()).collect()
    }

    /// Survival estimate derived from cumulative hazard: Ŝ(t) = exp(-Ĥ(t)).
    #[must_use]
    pub fn survival(&self) -> Vec<f64> {
        self.cum_hazard.iter().map(|h| (-h).exp()).collect()
    }
}

/// Estimate cumulative hazard via Nelson-Aalen.
pub fn nelson_aalen_estimate(data: &Dataset) -> SurvivalResult<NelsonAalen> {
    let rs = RiskSet::from_dataset(data)?;
    let mut h = Vec::with_capacity(rs.len());
    let mut var = Vec::with_capacity(rs.len());
    let mut h_cur = 0.0_f64;
    let mut v_cur = 0.0_f64;
    for (_t, d, n) in rs.iter() {
        if n > 0.0 && d > 0.0 {
            h_cur += d / n;
            v_cur += d / (n * n);
        }
        h.push(h_cur);
        var.push(v_cur);
    }
    Ok(NelsonAalen {
        times: rs.times,
        cum_hazard: h,
        variance: var,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn na_no_censoring() {
        let d = Dataset::from_arrays(&[1.0, 2.0, 3.0, 4.0], &[true, true, true, true]).expect("ok");
        let na = nelson_aalen_estimate(&d).expect("ok");
        // H(1)=1/4=0.25; H(2)=0.25+1/3; H(3)=+1/2; H(4)=+1
        assert!((na.cum_hazard[0] - 0.25).abs() < 1.0e-12);
        assert!((na.cum_hazard[1] - (0.25 + 1.0 / 3.0)).abs() < 1.0e-12);
        assert!((na.cum_hazard[2] - (0.25 + 1.0 / 3.0 + 0.5)).abs() < 1.0e-12);
        assert!((na.cum_hazard[3] - (0.25 + 1.0 / 3.0 + 0.5 + 1.0)).abs() < 1.0e-12);
    }

    #[test]
    fn na_variance_formula() {
        let d = Dataset::from_arrays(&[1.0, 2.0, 3.0, 4.0], &[true, true, true, true]).expect("ok");
        let na = nelson_aalen_estimate(&d).expect("ok");
        assert!((na.variance[0] - 1.0 / 16.0).abs() < 1.0e-12);
    }

    #[test]
    fn na_survival_from_hazard() {
        let d = Dataset::from_arrays(&[1.0, 2.0], &[true, true]).expect("ok");
        let na = nelson_aalen_estimate(&d).expect("ok");
        let s = na.survival();
        assert!((s[0] - (-0.5_f64).exp()).abs() < 1.0e-12);
    }
}
