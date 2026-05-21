use crate::error::{CausalError, CausalResult};

/// Local Average Treatment Effect (Imbens & Angrist, 1994) result.
///
/// All probabilities live on [0, 1] and `p_complier + p_always_taker +
/// p_never_taker ≈ 1` under the standard monotonicity assumption.
#[derive(Clone, Debug)]
pub struct LateResult {
    pub estimate: f64,
    pub std_err: f64,
    pub compliance_rate: f64,
    pub p_always_taker: f64,
    pub p_never_taker: f64,
    pub p_complier: f64,
    pub monotonicity_holds: bool,
}

/// Wald-form LATE estimator for a binary instrument `Z` and binary treatment
/// `D` on a continuous outcome `Y`.
pub struct LateEstimator;

impl Default for LateEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl LateEstimator {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    pub fn fit(&self, z: &[u8], d: &[u8], y: &[f64]) -> CausalResult<LateResult> {
        let n = z.len();
        if n == 0 || d.is_empty() || y.is_empty() {
            return Err(CausalError::EmptyInput);
        }
        if n < 2 {
            return Err(CausalError::EmptyInput);
        }
        if d.len() != n || y.len() != n {
            return Err(CausalError::DimensionMismatch {
                expected: n,
                got: d.len().min(y.len()),
            });
        }

        validate_binary(z, "z")?;
        validate_binary(d, "d")?;
        for (idx, &v) in y.iter().enumerate() {
            if !v.is_finite() {
                return Err(CausalError::Internal {
                    msg: format!("y[{idx}] is not finite"),
                });
            }
        }

        let strata = collect_strata(z, d, y)?;
        let StratumStats {
            n_z1,
            n_z0,
            mean_y_z1,
            mean_y_z0,
            mean_d_z1,
            mean_d_z0,
            var_y_z1,
            var_y_z0,
            var_d_z1,
            var_d_z0,
            cov_yd_z1,
            cov_yd_z0,
        } = strata;

        let denom = mean_d_z1 - mean_d_z0;
        let monotonicity_holds = denom > 0.0;

        if denom.abs() < 1e-12 {
            return Err(CausalError::Internal {
                msg: "zero compliance: E[D|Z=1] == E[D|Z=0]".to_string(),
            });
        }

        let estimate = (mean_y_z1 - mean_y_z0) / denom;
        let compliance_rate = denom.abs().clamp(0.0, 1.0);

        let p_always_taker = mean_d_z0.clamp(0.0, 1.0);
        let p_never_taker = (1.0 - mean_d_z1).clamp(0.0, 1.0);
        let p_complier = if monotonicity_holds {
            (1.0 - p_always_taker - p_never_taker).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Delta method for σ²(LATE):
        //   Var(LATE) ≈ Var(num)/denom² − 2·LATE·Cov(num, denom)/denom²
        //               + LATE²·Var(denom)/denom²
        // where num = Ȳ₁ − Ȳ₀, denom = D̄₁ − D̄₀, all four means independent
        // across strata. Var(Ȳ) = σ²_Y/n, Cov(Ȳ, D̄) = σ_YD/n.
        let var_num = safe_div(var_y_z1, n_z1 as f64) + safe_div(var_y_z0, n_z0 as f64);
        let var_den = safe_div(var_d_z1, n_z1 as f64) + safe_div(var_d_z0, n_z0 as f64);
        let cov_num_den = safe_div(cov_yd_z1, n_z1 as f64) + safe_div(cov_yd_z0, n_z0 as f64);
        let var_late = (var_num - 2.0 * estimate * cov_num_den + estimate * estimate * var_den)
            / (denom * denom);
        let std_err = if var_late.is_finite() && var_late > 0.0 {
            var_late.sqrt()
        } else {
            0.0
        };

        Ok(LateResult {
            estimate,
            std_err,
            compliance_rate,
            p_always_taker,
            p_never_taker,
            p_complier,
            monotonicity_holds,
        })
    }
}

fn validate_binary(v: &[u8], name: &str) -> CausalResult<()> {
    for (idx, &b) in v.iter().enumerate() {
        if b > 1 {
            return Err(CausalError::Internal {
                msg: format!("{name}[{idx}] = {b}, expected 0 or 1"),
            });
        }
    }
    Ok(())
}

struct StratumStats {
    n_z1: usize,
    n_z0: usize,
    mean_y_z1: f64,
    mean_y_z0: f64,
    mean_d_z1: f64,
    mean_d_z0: f64,
    var_y_z1: f64,
    var_y_z0: f64,
    var_d_z1: f64,
    var_d_z0: f64,
    cov_yd_z1: f64,
    cov_yd_z0: f64,
}

fn collect_strata(z: &[u8], d: &[u8], y: &[f64]) -> CausalResult<StratumStats> {
    let mut n_z1 = 0_usize;
    let mut n_z0 = 0_usize;
    let mut sum_y_z1 = 0.0_f64;
    let mut sum_y_z0 = 0.0_f64;
    let mut sum_d_z1 = 0.0_f64;
    let mut sum_d_z0 = 0.0_f64;

    for i in 0..z.len() {
        if z[i] == 1 {
            n_z1 += 1;
            sum_y_z1 += y[i];
            sum_d_z1 += d[i] as f64;
        } else {
            n_z0 += 1;
            sum_y_z0 += y[i];
            sum_d_z0 += d[i] as f64;
        }
    }

    if n_z1 == 0 || n_z0 == 0 {
        return Err(CausalError::Internal {
            msg: "instrument strata cannot be empty (need at least one Z=0 and one Z=1)"
                .to_string(),
        });
    }

    let mean_y_z1 = sum_y_z1 / n_z1 as f64;
    let mean_y_z0 = sum_y_z0 / n_z0 as f64;
    let mean_d_z1 = sum_d_z1 / n_z1 as f64;
    let mean_d_z0 = sum_d_z0 / n_z0 as f64;

    let mut var_y_z1 = 0.0_f64;
    let mut var_y_z0 = 0.0_f64;
    let mut var_d_z1 = 0.0_f64;
    let mut var_d_z0 = 0.0_f64;
    let mut cov_yd_z1 = 0.0_f64;
    let mut cov_yd_z0 = 0.0_f64;

    for i in 0..z.len() {
        let dy_i = d[i] as f64;
        if z[i] == 1 {
            let dy = y[i] - mean_y_z1;
            let dd = dy_i - mean_d_z1;
            var_y_z1 += dy * dy;
            var_d_z1 += dd * dd;
            cov_yd_z1 += dy * dd;
        } else {
            let dy = y[i] - mean_y_z0;
            let dd = dy_i - mean_d_z0;
            var_y_z0 += dy * dy;
            var_d_z0 += dd * dd;
            cov_yd_z0 += dy * dd;
        }
    }
    // Sample variance (Bessel correction) when stratum size > 1, else 0.
    var_y_z1 = if n_z1 > 1 {
        var_y_z1 / (n_z1 as f64 - 1.0)
    } else {
        0.0
    };
    var_y_z0 = if n_z0 > 1 {
        var_y_z0 / (n_z0 as f64 - 1.0)
    } else {
        0.0
    };
    var_d_z1 = if n_z1 > 1 {
        var_d_z1 / (n_z1 as f64 - 1.0)
    } else {
        0.0
    };
    var_d_z0 = if n_z0 > 1 {
        var_d_z0 / (n_z0 as f64 - 1.0)
    } else {
        0.0
    };
    cov_yd_z1 = if n_z1 > 1 {
        cov_yd_z1 / (n_z1 as f64 - 1.0)
    } else {
        0.0
    };
    cov_yd_z0 = if n_z0 > 1 {
        cov_yd_z0 / (n_z0 as f64 - 1.0)
    } else {
        0.0
    };

    Ok(StratumStats {
        n_z1,
        n_z0,
        mean_y_z1,
        mean_y_z0,
        mean_d_z1,
        mean_d_z0,
        var_y_z1,
        var_y_z0,
        var_d_z1,
        var_d_z0,
        cov_yd_z1,
        cov_yd_z0,
    })
}

fn safe_div(num: f64, den: f64) -> f64 {
    if den.abs() < 1e-18 { 0.0 } else { num / den }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn simulate_late(rng: &mut LcgRng, n: usize, true_late: f64) -> (Vec<u8>, Vec<u8>, Vec<f64>) {
        // 60% compliers, 20% always-takers, 20% never-takers.
        let mut z = vec![0u8; n];
        let mut d = vec![0u8; n];
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            let u = rng.next_f32() as f64;
            let is_complier = u < 0.6;
            let is_always = (0.6..0.8).contains(&u);
            let is_never = u >= 0.8;
            z[i] = if rng.next_f32() < 0.5 { 1 } else { 0 };
            d[i] = if is_always {
                1
            } else if is_never {
                0
            } else if is_complier {
                z[i]
            } else {
                0
            };
            let noise = (rng.next_normal() as f64) * 0.5;
            // Baseline outcome 1.0 + 0.0 effect for non-compliers + true_late for compliers.
            let base = 1.0;
            let treatment_effect = if is_complier && d[i] == 1 {
                true_late
            } else if is_always && d[i] == 1 {
                0.7
            } else {
                0.0
            };
            y[i] = base + treatment_effect + noise;
        }
        (z, d, y)
    }

    #[test]
    fn test_known_dgp_recovery() {
        let mut rng = LcgRng::new(12345);
        let (z, d, y) = simulate_late(&mut rng, 5000, 2.0);
        let res = LateEstimator::new().fit(&z, &d, &y).unwrap();
        assert!(
            (res.estimate - 2.0).abs() < 0.15,
            "expected LATE near 2.0, got {}",
            res.estimate
        );
        assert!(res.monotonicity_holds);
    }

    #[test]
    fn test_perfect_compliance_equals_ate() {
        let n = 1000_usize;
        let mut rng = LcgRng::new(7);
        let mut z = vec![0u8; n];
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            z[i] = if rng.next_f32() < 0.5 { 1 } else { 0 };
            let noise = (rng.next_normal() as f64) * 0.2;
            y[i] = 0.5 + 1.5 * (z[i] as f64) + noise;
        }
        let d = z.clone();
        let res = LateEstimator::new().fit(&z, &d, &y).unwrap();
        assert!((res.estimate - 1.5).abs() < 0.1, "got {}", res.estimate);
        // 100% compliers => compliance rate = 1.
        assert!((res.compliance_rate - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_zero_compliance_returns_err() {
        let n = 200_usize;
        let z: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
        // d does not respond to z at all (constant 0).
        let d = vec![0u8; n];
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let res = LateEstimator::new().fit(&z, &d, &y);
        assert!(res.is_err());
    }

    #[test]
    fn test_mismatched_lengths_returns_err() {
        let z = vec![0u8, 1, 0, 1];
        let d = vec![0u8, 1, 1];
        let y = vec![0.0_f64, 1.0, 0.5, 0.7];
        let res = LateEstimator::new().fit(&z, &d, &y);
        assert!(res.is_err());
    }

    #[test]
    fn test_empty_inputs_return_err() {
        let res = LateEstimator::new().fit(&[], &[], &[]);
        assert!(res.is_err());
    }

    #[test]
    fn test_non_binary_z_returns_err() {
        let z = vec![0u8, 1, 2, 0];
        let d = vec![0u8, 1, 1, 0];
        let y = vec![0.0_f64, 1.0, 0.5, 0.7];
        let res = LateEstimator::new().fit(&z, &d, &y);
        assert!(res.is_err());
    }

    #[test]
    fn test_non_binary_d_returns_err() {
        let z = vec![0u8, 1, 0, 1];
        let d = vec![0u8, 1, 5, 0];
        let y = vec![0.0_f64, 1.0, 0.5, 0.7];
        let res = LateEstimator::new().fit(&z, &d, &y);
        assert!(res.is_err());
    }

    #[test]
    fn test_monotonicity_violation_flagged() {
        // Construct deterministic data where E[D|Z=1] < E[D|Z=0].
        let z = vec![0u8, 0, 0, 0, 1, 1, 1, 1];
        let d = vec![1u8, 1, 1, 0, 0, 0, 0, 1];
        let y = vec![1.0_f64, 1.2, 0.9, 0.5, 0.4, 0.3, 0.5, 0.8];
        let res = LateEstimator::new().fit(&z, &d, &y);
        // Either it errors (denom too small or zero) or returns with
        // monotonicity_holds = false. Both are acceptable signalling.
        if let Ok(r) = res {
            assert!(!r.monotonicity_holds);
        }
    }

    #[test]
    fn test_compliance_rate_in_unit_interval() {
        let mut rng = LcgRng::new(2025);
        let (z, d, y) = simulate_late(&mut rng, 1000, 1.0);
        let res = LateEstimator::new().fit(&z, &d, &y).unwrap();
        assert!((0.0..=1.0).contains(&res.compliance_rate));
    }

    #[test]
    fn test_population_shares_sum_to_one() {
        let mut rng = LcgRng::new(8_888);
        let (z, d, y) = simulate_late(&mut rng, 2000, 1.0);
        let res = LateEstimator::new().fit(&z, &d, &y).unwrap();
        let total = res.p_always_taker + res.p_never_taker + res.p_complier;
        assert!(
            (total - 1.0).abs() < 0.05,
            "p_always + p_never + p_complier = {total}"
        );
    }

    #[test]
    fn test_standard_error_positive_under_monotonicity() {
        let mut rng = LcgRng::new(321);
        let (z, d, y) = simulate_late(&mut rng, 1000, 1.0);
        let res = LateEstimator::new().fit(&z, &d, &y).unwrap();
        assert!(res.monotonicity_holds);
        assert!(res.std_err > 0.0);
    }

    #[test]
    fn test_doubling_n_halves_variance_approx() {
        let mut rng = LcgRng::new(55);
        let (z1, d1, y1) = simulate_late(&mut rng, 800, 1.0);
        let res_small = LateEstimator::new().fit(&z1, &d1, &y1).unwrap();
        let mut rng2 = LcgRng::new(55);
        let (z2, d2, y2) = simulate_late(&mut rng2, 1600, 1.0);
        let res_large = LateEstimator::new().fit(&z2, &d2, &y2).unwrap();
        // Var halves => SE shrinks by sqrt(2). Allow generous tolerance.
        let ratio = res_small.std_err / res_large.std_err;
        assert!(
            (1.2..2.1).contains(&ratio),
            "se ratio {ratio} not in expected range"
        );
    }

    #[test]
    fn test_single_sample_returns_err() {
        let z = vec![1u8];
        let d = vec![1u8];
        let y = vec![1.0_f64];
        let res = LateEstimator::new().fit(&z, &d, &y);
        assert!(res.is_err());
    }

    #[test]
    fn test_all_z_one_returns_err() {
        let z = vec![1u8, 1, 1, 1, 1, 1];
        let d = vec![1u8, 0, 1, 1, 0, 1];
        let y = vec![1.0_f64, 0.5, 0.8, 1.2, 0.3, 0.9];
        let res = LateEstimator::new().fit(&z, &d, &y);
        assert!(res.is_err());
    }

    #[test]
    fn test_ci_coverage_montecarlo() {
        let true_late = 1.5_f64;
        let trials = 100_usize;
        let mut covered = 0_usize;
        for trial in 0..trials {
            let mut rng = LcgRng::new(1_000 + trial as u64);
            let (z, d, y) = simulate_late(&mut rng, 1000, true_late);
            let res = LateEstimator::new().fit(&z, &d, &y).unwrap();
            let lo = res.estimate - 1.96 * res.std_err;
            let hi = res.estimate + 1.96 * res.std_err;
            if lo <= true_late && true_late <= hi {
                covered += 1;
            }
        }
        let rate = covered as f64 / trials as f64;
        assert!(rate >= 0.85, "coverage {rate} below 0.85");
    }

    #[test]
    fn test_output_deterministic_given_same_inputs() {
        let z = vec![0u8, 1, 0, 1, 1, 0, 1, 0, 0, 1];
        let d = vec![0u8, 1, 0, 1, 0, 0, 1, 0, 1, 0];
        let y = vec![0.1_f64, 1.2, 0.0, 1.4, 0.3, -0.1, 1.1, 0.05, 0.6, 0.2];
        let a = LateEstimator::new().fit(&z, &d, &y).unwrap();
        let b = LateEstimator::new().fit(&z, &d, &y).unwrap();
        assert_eq!(a.estimate, b.estimate);
        assert_eq!(a.std_err, b.std_err);
        assert_eq!(a.compliance_rate, b.compliance_rate);
    }
}
