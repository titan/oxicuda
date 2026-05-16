//! McNemar's test for matched-pair categorical data.

use crate::distributions::chi_squared::ChiSquared;
use crate::error::{StatsError, StatsResult};

/// Result of a McNemar test.
#[derive(Debug, Clone, Copy)]
pub struct McnemarResult {
    pub chi_squared: f64,
    pub p_value: f64,
}

/// McNemar's test on the off-diagonal counts of a 2x2 paired contingency table.
///
/// With continuity correction: `chi^2 = (|b - c| - 1)^2 / (b + c)` ~ chi^2(1).
pub fn mcnemar(b: usize, c: usize, continuity_correction: bool) -> StatsResult<McnemarResult> {
    if b + c == 0 {
        return Err(StatsError::NumericalInstability("b + c = 0".into()));
    }
    let bf = b as f64;
    let cf = c as f64;
    let diff = (bf - cf).abs();
    let num = if continuity_correction {
        (diff - 1.0).max(0.0).powi(2)
    } else {
        diff * diff
    };
    let chi2 = num / (bf + cf);
    let p = 1.0 - ChiSquared::new(1.0)?.cdf(chi2)?;
    Ok(McnemarResult {
        chi_squared: chi2,
        p_value: p,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcnemar_small_diff_high_p() {
        let r = mcnemar(10, 11, true).expect("ok");
        assert!(r.p_value > 0.5);
    }

    #[test]
    fn mcnemar_big_diff_low_p() {
        let r = mcnemar(50, 5, true).expect("ok");
        assert!(r.p_value < 0.01);
    }
}
