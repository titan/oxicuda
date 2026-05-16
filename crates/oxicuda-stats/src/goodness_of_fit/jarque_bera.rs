//! Jarque-Bera test for normality.

use crate::descriptive::summary::{kurtosis, mean, skewness, std_dev};
use crate::distributions::chi_squared::ChiSquared;
use crate::error::{StatsError, StatsResult};

/// Result of a Jarque-Bera test.
#[derive(Debug, Clone, Copy)]
pub struct JarqueBeraResult {
    pub jb_statistic: f64,
    pub p_value: f64,
}

/// Jarque-Bera normality test using the moment-based statistic.
///
/// JB = n/6 * (S^2 + 0.25 * (K - 3)^2) where S is skewness and K is (uncorrected) kurtosis.
pub fn jarque_bera(x: &[f64]) -> StatsResult<JarqueBeraResult> {
    if x.len() < 4 {
        return Err(StatsError::InsufficientSampleSize {
            got: x.len(),
            need: 4,
        });
    }
    let n = x.len() as f64;
    let m = mean(x)?;
    let s = std_dev(x)?;
    if s <= 0.0 {
        return Err(StatsError::NumericalInstability("zero std dev".into()));
    }
    // Use population moments to match Jarque-Bera definition: S = m3/s^3, K = m4/s^4
    let s3: f64 = x.iter().map(|v| ((v - m) / s).powi(3)).sum::<f64>() / n;
    let s4: f64 = x.iter().map(|v| ((v - m) / s).powi(4)).sum::<f64>() / n;
    let jb = n / 6.0 * (s3 * s3 + 0.25 * (s4 - 3.0).powi(2));
    let _ = skewness(x)?; // touch dependency
    let _ = kurtosis(x)?;
    let p = 1.0 - ChiSquared::new(2.0)?.cdf(jb.max(0.0))?;
    Ok(JarqueBeraResult {
        jb_statistic: jb,
        p_value: p,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn jb_normal_high_p() {
        let mut rng = LcgRng::new(7);
        let x: Vec<f64> = (0..200).map(|_| rng.next_normal()).collect();
        let r = jarque_bera(&x).expect("ok");
        assert!(r.p_value > 0.05);
    }

    #[test]
    fn jb_skewed_low_p() {
        let mut rng = LcgRng::new(31);
        let x: Vec<f64> = (0..500).map(|_| rng.next_f64().powi(3)).collect();
        let r = jarque_bera(&x).expect("ok");
        assert!(r.p_value < 0.01);
    }
}
