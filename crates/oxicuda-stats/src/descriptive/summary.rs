//! Sample mean, variance, standard deviation, skewness, kurtosis.

use crate::error::{StatsError, StatsResult};

/// Sample mean.
pub fn mean(x: &[f64]) -> StatsResult<f64> {
    if x.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    let s: f64 = x.iter().sum();
    Ok(s / x.len() as f64)
}

/// Population variance (divisor n).
pub fn variance(x: &[f64]) -> StatsResult<f64> {
    if x.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    let m = mean(x)?;
    let s: f64 = x.iter().map(|v| (v - m) * (v - m)).sum();
    Ok(s / x.len() as f64)
}

/// Sample variance (divisor n-1, Bessel's correction).
pub fn sample_var(x: &[f64]) -> StatsResult<f64> {
    if x.len() < 2 {
        return Err(StatsError::InsufficientSampleSize {
            got: x.len(),
            need: 2,
        });
    }
    let m = mean(x)?;
    let s: f64 = x.iter().map(|v| (v - m) * (v - m)).sum();
    Ok(s / (x.len() - 1) as f64)
}

/// Population standard deviation.
pub fn std_dev(x: &[f64]) -> StatsResult<f64> {
    Ok(variance(x)?.sqrt())
}

/// Sample standard deviation.
pub fn sample_std(x: &[f64]) -> StatsResult<f64> {
    Ok(sample_var(x)?.sqrt())
}

/// Sample skewness (Fisher-Pearson, adjusted for bias).
pub fn skewness(x: &[f64]) -> StatsResult<f64> {
    if x.len() < 3 {
        return Err(StatsError::InsufficientSampleSize {
            got: x.len(),
            need: 3,
        });
    }
    let m = mean(x)?;
    let n = x.len() as f64;
    let s2: f64 = x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n;
    let s3: f64 = x.iter().map(|v| (v - m).powi(3)).sum::<f64>() / n;
    let sd = s2.sqrt();
    if sd < 1e-300 {
        return Ok(0.0);
    }
    let g1 = s3 / sd.powi(3);
    // Adjusted (Fisher's): G1 = g1 * sqrt(n(n-1)) / (n-2)
    let adj = (n * (n - 1.0)).sqrt() / (n - 2.0);
    Ok(adj * g1)
}

/// Sample excess kurtosis (Fisher's definition; normal distribution has 0).
pub fn kurtosis(x: &[f64]) -> StatsResult<f64> {
    if x.len() < 4 {
        return Err(StatsError::InsufficientSampleSize {
            got: x.len(),
            need: 4,
        });
    }
    let m = mean(x)?;
    let n = x.len() as f64;
    let s2: f64 = x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n;
    let s4: f64 = x.iter().map(|v| (v - m).powi(4)).sum::<f64>() / n;
    if s2 < 1e-300 {
        return Ok(0.0);
    }
    let g2 = s4 / (s2 * s2) - 3.0;
    // Unbiased adjustment
    let adj = ((n + 1.0) * g2 + 6.0) * (n - 1.0) / ((n - 2.0) * (n - 3.0));
    Ok(adj)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_simple() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((mean(&x).expect("ok") - 3.0).abs() < 1e-12);
    }

    #[test]
    fn sample_var_matches_known() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        // variance of 1..5 with n-1: sum((xi - 3)^2) = 10, /(5-1) = 2.5
        assert!((sample_var(&x).expect("ok") - 2.5).abs() < 1e-12);
    }

    #[test]
    fn std_dev_simple() {
        let x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        // Known population stddev = 2
        assert!((std_dev(&x).expect("ok") - 2.0).abs() < 1e-12);
    }

    #[test]
    fn empty_input_errs() {
        let x: [f64; 0] = [];
        assert!(mean(&x).is_err());
        assert!(variance(&x).is_err());
    }

    #[test]
    fn skewness_zero_for_symmetric() {
        let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let s = skewness(&x).expect("ok");
        assert!(s.abs() < 1e-12);
    }

    #[test]
    fn kurtosis_finite() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let k = kurtosis(&x).expect("ok");
        assert!(k.is_finite());
    }
}
