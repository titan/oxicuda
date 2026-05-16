//! Quantile / percentile computation with linear interpolation.

use crate::error::{StatsError, StatsResult};

/// Quantile via linear interpolation (type-7 / numpy default).
pub fn quantile(x: &[f64], q: f64) -> StatsResult<f64> {
    if x.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    if !(0.0..=1.0).contains(&q) {
        return Err(StatsError::ProbabilityOutOfRange { value: q });
    }
    let mut sorted: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return Err(StatsError::NonFiniteValue(0));
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let h = q * (n as f64 - 1.0);
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = h - lo as f64;
    Ok(sorted[lo] + frac * (sorted[hi] - sorted[lo]))
}

/// Percentile (0 to 100).
pub fn percentile(x: &[f64], p: f64) -> StatsResult<f64> {
    quantile(x, p / 100.0)
}

/// Inclusive quantile (R's type-6 / Excel inclusive).
pub fn quantile_inclusive(x: &[f64], q: f64) -> StatsResult<f64> {
    if x.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    if !(0.0..=1.0).contains(&q) {
        return Err(StatsError::ProbabilityOutOfRange { value: q });
    }
    let mut sorted: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 {
        return Ok(sorted[0]);
    }
    let h = q * (n as f64 + 1.0) - 1.0;
    let h = h.max(0.0).min(n as f64 - 1.0);
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = h - lo as f64;
    Ok(sorted[lo] + frac * (sorted[hi] - sorted[lo]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_of_sorted() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let m = quantile(&x, 0.5).expect("ok");
        assert!((m - 3.0).abs() < 1e-12);
    }

    #[test]
    fn quartiles() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        // q1, q2, q3 (linear interp)
        assert!((quantile(&x, 0.25).expect("ok") - 2.0).abs() < 1e-12);
        assert!((quantile(&x, 0.75).expect("ok") - 4.0).abs() < 1e-12);
    }

    #[test]
    fn percentile_simple() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&x, 50.0).expect("ok") - 3.0).abs() < 1e-12);
    }

    #[test]
    fn endpoints() {
        let x = [1.0, 4.0, 9.0];
        assert!((quantile(&x, 0.0).expect("ok") - 1.0).abs() < 1e-12);
        assert!((quantile(&x, 1.0).expect("ok") - 9.0).abs() < 1e-12);
    }
}
