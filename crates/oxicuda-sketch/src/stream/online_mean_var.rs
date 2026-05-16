//! Welford's online algorithm for mean & variance (Welford 1962).
//!
//! Numerically stable single-pass algorithm.

/// Welford online mean / variance accumulator.
#[derive(Debug, Clone, Default)]
pub struct WelfordOnline {
    pub n: u64,
    pub mean: f64,
    pub m2: f64,
}

impl WelfordOnline {
    /// Empty accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Add a single observation.
    pub fn add(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// Sample variance (n - 1 denominator). Returns 0 if `n < 2`.
    #[must_use]
    pub fn sample_variance(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            self.m2 / (self.n - 1) as f64
        }
    }

    /// Population variance (n denominator).
    #[must_use]
    pub fn population_variance(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.m2 / self.n as f64
        }
    }

    /// Sample standard deviation.
    #[must_use]
    pub fn sample_stddev(&self) -> f64 {
        self.sample_variance().sqrt()
    }

    /// Merge another Welford accumulator into this one (Chan, Golub, LeVeque 1979).
    pub fn merge(&mut self, other: &WelfordOnline) {
        if other.n == 0 {
            return;
        }
        let n_a = self.n as f64;
        let n_b = other.n as f64;
        let n = n_a + n_b;
        let delta = other.mean - self.mean;
        self.mean += delta * n_b / n;
        self.m2 += other.m2 + delta * delta * n_a * n_b / n;
        self.n += other.n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welford_empty() {
        let w = WelfordOnline::new();
        assert_eq!(w.n, 0);
        assert_eq!(w.sample_variance(), 0.0);
    }

    #[test]
    fn welford_constant_input_var_zero() {
        let mut w = WelfordOnline::new();
        for _ in 0..100 {
            w.add(5.0);
        }
        assert!((w.mean - 5.0).abs() < 1e-12);
        assert!(w.sample_variance() < 1e-12);
    }

    #[test]
    fn welford_simple_known_var() {
        let mut w = WelfordOnline::new();
        for &x in &[1.0, 2.0, 3.0, 4.0, 5.0] {
            w.add(x);
        }
        // Mean = 3, sample variance = 2.5.
        assert!((w.mean - 3.0).abs() < 1e-12);
        assert!((w.sample_variance() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn welford_merge_match_direct() {
        let mut w1 = WelfordOnline::new();
        let mut w2 = WelfordOnline::new();
        let mut full = WelfordOnline::new();
        for &x in &[1.0, 2.0, 3.0] {
            w1.add(x);
            full.add(x);
        }
        for &x in &[4.0, 5.0, 6.0] {
            w2.add(x);
            full.add(x);
        }
        w1.merge(&w2);
        assert!((w1.mean - full.mean).abs() < 1e-12);
        assert!((w1.sample_variance() - full.sample_variance()).abs() < 1e-12);
    }
}
