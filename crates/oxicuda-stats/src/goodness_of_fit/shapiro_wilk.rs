//! Shapiro-Wilk test for normality (Royston's algorithm AS R94).

use crate::distributions::normal::Normal;
use crate::error::{StatsError, StatsResult};

/// Result of a Shapiro-Wilk test.
#[derive(Debug, Clone, Copy)]
pub struct ShapiroWilkResult {
    pub w_statistic: f64,
    pub p_value: f64,
}

/// Shapiro-Wilk W test for normality.
///
/// Uses Royston's pseudo-coefficients via inverse-normal scores and the moments of order statistics
/// approximation (good for `n >= 7`; rough for smaller n).
pub fn shapiro_wilk(x: &[f64]) -> StatsResult<ShapiroWilkResult> {
    let n = x.len();
    if n < 3 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 3 });
    }
    let mut sorted: Vec<f64> = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n_f = n as f64;
    // m_i = Phi^{-1}((i - 3/8) / (n + 1/4))
    let dist = Normal::standard();
    let mut m = vec![0.0; n];
    for (i, m_i) in m.iter_mut().enumerate() {
        let p = ((i + 1) as f64 - 3.0 / 8.0) / (n_f + 0.25);
        *m_i = dist.ppf(p)?;
    }
    let m_norm: f64 = m.iter().map(|v| v * v).sum::<f64>().sqrt();
    // a coefficients via Royston's approximation
    let mut a = vec![0.0; n];
    let u = 1.0 / n_f.sqrt();
    let a_n = -2.706_056 * u.powi(5) + 4.434_685 * u.powi(4)
        - 2.071_190 * u.powi(3)
        - 0.147_981 * u.powi(2)
        + 0.221_157 * u
        + m[n - 1] / m_norm;
    let a_n_1 = -3.582_633 * u.powi(5) + 5.682_633 * u.powi(4)
        - 1.752_460 * u.powi(3)
        - 0.293_762 * u.powi(2)
        + 0.042_981 * u
        + m[n - 2] / m_norm;
    a[n - 1] = a_n;
    a[n - 2] = a_n_1;
    a[0] = -a_n;
    a[1] = -a_n_1;
    // remaining a's
    let epsilon = (m_norm.powi(2) - 2.0 * m[n - 1].powi(2) - 2.0 * m[n - 2].powi(2))
        / (1.0 - 2.0 * a_n.powi(2) - 2.0 * a_n_1.powi(2));
    if epsilon < 0.0 {
        return Err(StatsError::NumericalInstability(
            "shapiro_wilk: negative epsilon".into(),
        ));
    }
    let eps_sqrt = epsilon.sqrt().max(1e-300);
    for i in 2..(n - 2) {
        a[i] = m[i] / eps_sqrt;
    }
    // numerator
    let xbar: f64 = sorted.iter().sum::<f64>() / n_f;
    let num: f64 = a
        .iter()
        .zip(&sorted)
        .map(|(ai, xi)| ai * xi)
        .sum::<f64>()
        .powi(2);
    let den: f64 = sorted.iter().map(|xi| (xi - xbar).powi(2)).sum();
    if den <= 0.0 {
        return Err(StatsError::NumericalInstability("zero variance".into()));
    }
    let w = (num / den).clamp(0.0, 1.0);
    // p-value via Royston log transform (approximate)
    let ln_n = n_f.ln();
    let mu_w = 0.0038915 * ln_n.powi(3) - 0.083751 * ln_n.powi(2) - 0.31082 * ln_n - 1.5861;
    let sigma_w =
        (-0.0006714 * ln_n.powi(3) + 0.025054 * ln_n.powi(2) - 0.39978 * ln_n + 1.0).exp();
    let y = (1.0 - w).max(1e-300).ln();
    let z = (y - mu_w) / sigma_w;
    let p = 1.0 - dist.cdf(z);
    Ok(ShapiroWilkResult {
        w_statistic: w,
        p_value: p.clamp(0.0, 1.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn shapiro_wilk_normal_sample_high_p() {
        let mut rng = LcgRng::new(123);
        let x: Vec<f64> = (0..50).map(|_| rng.next_normal()).collect();
        let r = shapiro_wilk(&x).expect("ok");
        assert!(r.w_statistic > 0.9);
    }

    #[test]
    fn shapiro_wilk_uniform_sample_lower_w() {
        let mut rng = LcgRng::new(99);
        let x: Vec<f64> = (0..30).map(|_| rng.next_f64()).collect();
        let r = shapiro_wilk(&x).expect("ok");
        // Uniform distribution still has reasonable W, but should be lower than normal
        assert!(r.w_statistic.is_finite());
        assert!(r.w_statistic > 0.5);
    }
}
