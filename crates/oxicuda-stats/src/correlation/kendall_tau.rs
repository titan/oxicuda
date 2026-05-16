//! Kendall's tau (concordant - discordant) / total pairs.

use crate::distributions::normal::Normal;
use crate::error::{StatsError, StatsResult};

/// Kendall's tau-b result with tie adjustment.
#[derive(Debug, Clone, Copy)]
pub struct KendallResult {
    pub tau: f64,
    pub z: f64,
    pub p_value_two_sided: f64,
}

/// Kendall's tau-b via O(n^2) concordance counting.
pub fn kendall_tau(x: &[f64], y: &[f64]) -> StatsResult<KendallResult> {
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatch {
            a: x.len(),
            b: y.len(),
        });
    }
    let n = x.len();
    if n < 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    let mut concordant: i64 = 0;
    let mut discordant: i64 = 0;
    let mut tied_x = 0i64;
    let mut tied_y = 0i64;
    let mut tied_both = 0i64;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = (x[j] - x[i]).signum();
            let dy = (y[j] - y[i]).signum();
            if dx == 0.0 && dy == 0.0 {
                tied_both += 1;
            } else if dx == 0.0 {
                tied_x += 1;
            } else if dy == 0.0 {
                tied_y += 1;
            } else if dx * dy > 0.0 {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }
    let total_pairs = (n * (n - 1) / 2) as f64;
    let n1 = (total_pairs - tied_x as f64 - tied_both as f64).max(0.0);
    let n2 = (total_pairs - tied_y as f64 - tied_both as f64).max(0.0);
    let denom = (n1 * n2).sqrt().max(1.0);
    let tau = (concordant - discordant) as f64 / denom;
    let nf = n as f64;
    let var = (2.0 * (2.0 * nf + 5.0)) / (9.0 * nf * (nf - 1.0));
    let z = tau / var.sqrt();
    let dist = Normal::standard();
    let p = 2.0 * (1.0 - dist.cdf(z.abs()));
    Ok(KendallResult {
        tau: tau.clamp(-1.0, 1.0),
        z,
        p_value_two_sided: p.clamp(0.0, 1.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kendall_perfect_concordance() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r = kendall_tau(&x, &y).expect("ok");
        assert!((r.tau - 1.0).abs() < 1e-12);
    }

    #[test]
    fn kendall_perfect_discordance() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0, 4.0, 3.0, 2.0, 1.0];
        let r = kendall_tau(&x, &y).expect("ok");
        assert!((r.tau + 1.0).abs() < 1e-12);
    }
}
