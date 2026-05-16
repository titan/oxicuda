//! Standardized effect-size measures.

use crate::descriptive::summary::{mean, sample_std, sample_var};
use crate::error::{StatsError, StatsResult};

/// Cohen's d for two independent samples (using pooled standard deviation).
pub fn cohen_d(x1: &[f64], x2: &[f64]) -> StatsResult<f64> {
    if x1.len() < 2 || x2.len() < 2 {
        return Err(StatsError::InsufficientSampleSize {
            got: x1.len().min(x2.len()),
            need: 2,
        });
    }
    let m1 = mean(x1)?;
    let m2 = mean(x2)?;
    let s1 = sample_var(x1)?;
    let s2 = sample_var(x2)?;
    let n1 = x1.len() as f64;
    let n2 = x2.len() as f64;
    let sp = (((n1 - 1.0) * s1 + (n2 - 1.0) * s2) / (n1 + n2 - 2.0)).sqrt();
    if sp <= 0.0 {
        return Err(StatsError::NumericalInstability("zero pooled SD".into()));
    }
    Ok((m1 - m2) / sp)
}

/// Hedges' g (bias-corrected Cohen's d).
pub fn hedges_g(x1: &[f64], x2: &[f64]) -> StatsResult<f64> {
    let d = cohen_d(x1, x2)?;
    let n = (x1.len() + x2.len()) as f64;
    let correction = 1.0 - 3.0 / (4.0 * n - 9.0);
    Ok(d * correction)
}

/// Glass's delta (uses control-group SD as the standardizer).
pub fn glass_delta(treatment: &[f64], control: &[f64]) -> StatsResult<f64> {
    let mt = mean(treatment)?;
    let mc = mean(control)?;
    let sc = sample_std(control)?;
    if sc <= 0.0 {
        return Err(StatsError::NumericalInstability("zero control SD".into()));
    }
    Ok((mt - mc) / sc)
}

/// Cohen's f for ANOVA: sqrt(eta^2 / (1 - eta^2)).
pub fn cohen_f(eta_squared: f64) -> StatsResult<f64> {
    if !(0.0..1.0).contains(&eta_squared) {
        return Err(StatsError::ProbabilityOutOfRange { value: eta_squared });
    }
    Ok((eta_squared / (1.0 - eta_squared)).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cohen_d_basic() {
        let x1 = [1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = [3.0, 4.0, 5.0, 6.0, 7.0];
        let d = cohen_d(&x1, &x2).expect("ok");
        // Means differ by 2, pooled SD = sqrt(2.5) ~ 1.58 => d ~ -1.26
        assert!(d < 0.0);
        assert!(d.is_finite());
    }

    #[test]
    fn hedges_g_smaller_in_magnitude() {
        let x1 = [1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = [3.0, 4.0, 5.0, 6.0, 7.0];
        let d = cohen_d(&x1, &x2).expect("ok");
        let g = hedges_g(&x1, &x2).expect("ok");
        assert!(g.abs() < d.abs());
    }

    #[test]
    fn cohen_f_basic() {
        let f = cohen_f(0.1).expect("ok");
        assert!((f - (0.1f64 / 0.9).sqrt()).abs() < 1e-12);
    }
}
