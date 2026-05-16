//! ANOVA effect-size statistics: eta^2, partial eta^2, omega^2.

use crate::error::{StatsError, StatsResult};

/// Eta-squared: SS_between / SS_total.
pub fn eta_squared(ss_between: f64, ss_total: f64) -> StatsResult<f64> {
    if ss_total <= 0.0 {
        return Err(StatsError::NumericalInstability("ss_total <= 0".into()));
    }
    if ss_between < 0.0 {
        return Err(StatsError::InvalidParameter {
            name: "ss_between".into(),
            reason: "must be >= 0".into(),
        });
    }
    Ok((ss_between / ss_total).clamp(0.0, 1.0))
}

/// Partial eta-squared: SS_effect / (SS_effect + SS_error).
pub fn partial_eta_squared(ss_effect: f64, ss_error: f64) -> StatsResult<f64> {
    if ss_effect + ss_error <= 0.0 {
        return Err(StatsError::NumericalInstability("zero denominator".into()));
    }
    Ok((ss_effect / (ss_effect + ss_error)).clamp(0.0, 1.0))
}

/// Omega-squared (less biased): `(SSB - (k-1) * MSE) / (SST + MSE)`.
pub fn omega_squared(ss_between: f64, ss_total: f64, k: usize, mse: f64) -> StatsResult<f64> {
    if ss_total <= 0.0 {
        return Err(StatsError::NumericalInstability("ss_total <= 0".into()));
    }
    if k < 2 {
        return Err(StatsError::InvalidParameter {
            name: "k".into(),
            reason: "must be >= 2".into(),
        });
    }
    let num = ss_between - (k - 1) as f64 * mse;
    let den = ss_total + mse;
    Ok((num / den).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eta_squared_basic() {
        let v = eta_squared(50.0, 100.0).expect("ok");
        assert!((v - 0.5).abs() < 1e-12);
    }

    #[test]
    fn partial_eta_squared_basic() {
        let v = partial_eta_squared(20.0, 80.0).expect("ok");
        assert!((v - 0.2).abs() < 1e-12);
    }

    #[test]
    fn omega_squared_basic() {
        let v = omega_squared(50.0, 100.0, 3, 5.0).expect("ok");
        // (50 - 2*5) / (100 + 5) = 40 / 105
        assert!((v - 40.0 / 105.0).abs() < 1e-10);
    }
}
