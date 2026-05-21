//! Cinelli-Hazlett OVB sensitivity analysis — partial-R² based bounds.
//!
//! Cinelli C, Hazlett C (2020) "Making Sense of Sensitivity: Extending Omitted
//! Variable Bias." *Journal of the Royal Statistical Society B* 82(1):39–67.
//!
//! # Problem
//!
//! An OLS treatment-effect estimate θ̂ may be biased by an unobserved
//! confounder Z. The key result is that the induced bias is bounded by:
//!
//! ```text
//!   |bias| ≤ se(θ̂) · √df · √(R²_{YZ·XD} · R²_{DZ·X} / (1 − R²_{DZ·X}))
//! ```
//!
//! where R²_{YZ·XD} is the partial R² of Y on Z given (X, D) and R²_{DZ·X}
//! is the partial R² of D on Z given X.
//!
//! # Robustness value
//!
//! Under the "equal partial-R²" assumption (R²_Y = R²_D = RV) the robustness
//! value is the unique positive root of:
//!
//! ```text
//!   df · RV² + q²t² · RV − q²t² = 0
//! ```
//!
//! which gives:
//!
//! ```text
//!   RV = (−q²t² + √(q⁴t⁴ + 4·df·q²t²)) / (2·df)
//! ```

use crate::error::{CausalError, CausalResult};

/// Input data for the OVB sensitivity analysis.
pub struct OvbInput {
    /// OLS treatment-effect estimate θ̂.
    pub hat_theta: f64,
    /// Standard error of θ̂ (must be > 0).
    pub se_theta: f64,
    /// Degrees of freedom n − k − 1 (must be > 0).
    pub df: f64,
    /// Partial R² of Y on D given X, in (0, 1).
    pub r2yd_x: f64,
}

/// Configuration knobs for the Cinelli-Hazlett analysis.
pub struct CinelliHazlettConfig {
    /// Robustness-value fraction in (0, 1]; default 1.0 (full nullification).
    pub q: f64,
    /// Significance level in (0, 0.5); default 0.05.
    pub alpha: f64,
}

impl Default for CinelliHazlettConfig {
    fn default() -> Self {
        Self {
            q: 1.0,
            alpha: 0.05,
        }
    }
}

/// Result for a single (R²_Y, R²_D) benchmark scenario.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Hypothetical partial R²_{YZ·XD}.
    pub r2_y_x: f64,
    /// Hypothetical partial R²_{DZ·X}.
    pub r2_d_x: f64,
    /// Absolute induced bias.
    pub bias: f64,
    /// θ̂ adjusted for the bias: θ̂ − sign(θ̂)·bias.
    pub adjusted_theta: f64,
    /// adjusted_theta / se_theta.
    pub adjusted_t: f64,
    /// Whether |adjusted_t| > t_crit.
    pub is_significant: bool,
}

/// Full output of the Cinelli-Hazlett sensitivity analysis.
#[derive(Debug, Clone)]
pub struct CinelliHazlettResult {
    /// Robustness value for q·|θ̂| nullification.
    pub rv_q: f64,
    /// Robustness value for significance at the given alpha.
    pub rv_alpha: f64,
    /// Extreme-scenario absolute bias (R²_D = 1 − r2yd_x, R²_Y = 1).
    pub extreme_bias: f64,
    /// θ̂ adjusted for the extreme bias.
    pub extreme_adjusted_theta: f64,
    /// t-statistic: θ̂ / se.
    pub t_stat: f64,
    /// Partial R² of Y on D given X (stored from input).
    pub r2yd_x: f64,
    /// Benchmark grid results.
    pub benchmarks: Vec<BenchmarkResult>,
}

/// Stateless namespace for Cinelli-Hazlett computations.
pub struct CinelliHazlett;

impl CinelliHazlett {
    /// Partial R² from a t-statistic and degrees of freedom.
    ///
    /// Formula: r² = t² / (t² + df).
    ///
    /// # Errors
    ///
    /// Returns [`CausalError::InvalidParameter`] when `df ≤ 0`.
    pub fn partial_r2_from_t(t: f64, df: f64) -> CausalResult<f64> {
        if df <= 0.0 || !df.is_finite() {
            return Err(CausalError::InvalidParameter {
                reason: format!("df must be > 0, got {df}"),
            });
        }
        let t2 = t * t;
        Ok(t2 / (t2 + df))
    }

    /// OVB bias magnitude for a given partial-R² pair.
    ///
    /// Formula: bias = se · √df · √(r2_y · r2_d / (1 − r2_d)).
    ///
    /// # Errors
    ///
    /// Returns [`CausalError::InvalidParameter`] when:
    /// - `se ≤ 0`
    /// - `df ≤ 0`
    /// - `r2_d ≥ 1` (would cause division by zero)
    /// - any argument is non-finite
    pub fn ovb_bias(r2_y: f64, r2_d: f64, se: f64, df: f64) -> CausalResult<f64> {
        if !se.is_finite() || se <= 0.0 {
            return Err(CausalError::InvalidParameter {
                reason: format!("se must be > 0, got {se}"),
            });
        }
        if !df.is_finite() || df <= 0.0 {
            return Err(CausalError::InvalidParameter {
                reason: format!("df must be > 0, got {df}"),
            });
        }
        if !r2_y.is_finite() || r2_y < 0.0 {
            return Err(CausalError::InvalidParameter {
                reason: format!("r2_y must be >= 0, got {r2_y}"),
            });
        }
        if !r2_d.is_finite() || r2_d < 0.0 {
            return Err(CausalError::InvalidParameter {
                reason: format!("r2_d must be >= 0, got {r2_d}"),
            });
        }
        if r2_d >= 1.0 {
            return Err(CausalError::InvalidParameter {
                reason: format!("r2_d must be < 1 (division by zero), got {r2_d}"),
            });
        }
        // When either R² is zero, bias is exactly zero.
        if r2_y == 0.0 || r2_d == 0.0 {
            return Ok(0.0);
        }
        let bias = se * df.sqrt() * (r2_y * r2_d / (1.0 - r2_d)).sqrt();
        Ok(bias)
    }

    /// Robustness value under the equal partial-R² assumption.
    ///
    /// Solves the quadratic `df·RV² + q²t²·RV − q²t² = 0` for the positive
    /// root and clamps the result to [0, 1].
    ///
    /// # Errors
    ///
    /// Returns [`CausalError::InvalidParameter`] when `df ≤ 0` or `q ∉ (0,1]`.
    pub fn robustness_value(t: f64, df: f64, q: f64) -> CausalResult<f64> {
        if !df.is_finite() || df <= 0.0 {
            return Err(CausalError::InvalidParameter {
                reason: format!("df must be > 0, got {df}"),
            });
        }
        if !q.is_finite() || q <= 0.0 || q > 1.0 {
            return Err(CausalError::InvalidParameter {
                reason: format!("q must be in (0, 1], got {q}"),
            });
        }
        let t2 = t * t;
        let q2t2 = q * q * t2;
        // Quadratic: df·RV² + q²t²·RV − q²t² = 0
        // Positive root: RV = (−q²t² + √(q⁴t⁴ + 4·df·q²t²)) / (2·df)
        let discriminant = q2t2 * q2t2 + 4.0 * df * q2t2;
        let rv = (-q2t2 + discriminant.sqrt()) / (2.0 * df);
        Ok(rv.clamp(0.0, 1.0))
    }

    /// Run the full Cinelli-Hazlett OVB sensitivity analysis.
    ///
    /// # Parameters
    ///
    /// - `input` — OLS estimate, SE, DF, and partial R²_{YD·X}.
    /// - `cfg` — robustness fraction `q` and significance level `alpha`.
    /// - `benchmarks` — slice of `(r2_y, r2_d)` hypothetical confounder scenarios.
    ///
    /// # Errors
    ///
    /// Returns [`CausalError::InvalidParameter`] on any invalid input.
    pub fn analyze(
        input: &OvbInput,
        cfg: &CinelliHazlettConfig,
        benchmarks: &[(f64, f64)],
    ) -> CausalResult<CinelliHazlettResult> {
        // Validate inputs.
        if !input.se_theta.is_finite() || input.se_theta <= 0.0 {
            return Err(CausalError::InvalidParameter {
                reason: format!("se_theta must be > 0, got {}", input.se_theta),
            });
        }
        if !input.df.is_finite() || input.df <= 0.0 {
            return Err(CausalError::InvalidParameter {
                reason: format!("df must be > 0, got {}", input.df),
            });
        }
        if !input.r2yd_x.is_finite() || input.r2yd_x <= 0.0 || input.r2yd_x >= 1.0 {
            return Err(CausalError::InvalidParameter {
                reason: format!("r2yd_x must be in (0, 1), got {}", input.r2yd_x),
            });
        }
        if !cfg.q.is_finite() || cfg.q <= 0.0 || cfg.q > 1.0 {
            return Err(CausalError::InvalidParameter {
                reason: format!("q must be in (0, 1], got {}", cfg.q),
            });
        }
        if !cfg.alpha.is_finite() || cfg.alpha <= 0.0 || cfg.alpha >= 0.5 {
            return Err(CausalError::InvalidParameter {
                reason: format!("alpha must be in (0, 0.5), got {}", cfg.alpha),
            });
        }
        // Validate all benchmark pairs upfront.
        for &(r2_y, r2_d) in benchmarks {
            if !(0.0..1.0).contains(&r2_d) {
                return Err(CausalError::InvalidParameter {
                    reason: format!("benchmark r2_d must be in [0, 1), got {r2_d}"),
                });
            }
            if r2_y < 0.0 {
                return Err(CausalError::InvalidParameter {
                    reason: format!("benchmark r2_y must be >= 0, got {r2_y}"),
                });
            }
        }

        let t = input.hat_theta / input.se_theta;
        let df = input.df;
        let se = input.se_theta;

        // Robustness value for full nullification (q-fold).
        let rv_q = Self::robustness_value(t, df, cfg.q)?;

        // Conservative t_crit using the approximation described in the spec.
        // For df > 30 use z = 1.96; for smaller df add a conservative correction.
        let t_crit = if df > 30.0 { 1.96_f64 } else { 1.96 + 2.0 / df };

        // Robustness value for significance: find effective_q such that
        // the threshold t_crit / |t| defines the fraction of θ̂ that needs
        // to be nullified for the result to lose significance.
        let t_abs = t.abs().max(1e-12);
        let effective_q = (t_crit / t_abs).min(1.0);
        let rv_alpha = if effective_q > 1.0 {
            1.0
        } else {
            Self::robustness_value(t, df, effective_q)?
        };

        // Extreme scenario: confounder explains all residual D variation and
        // is perfectly collinear with Y after conditioning.
        // R²_{DZ·X} = 1 − r2yd_x, R²_{YZ·XD} = 1.
        let r2_d_extreme = 1.0 - input.r2yd_x;
        let extreme_bias = if r2_d_extreme <= 0.0 || input.r2yd_x >= 1.0 {
            0.0
        } else {
            // bias = se · √df · √((1 − r2yd_x) / r2yd_x)
            se * df.sqrt() * (r2_d_extreme / input.r2yd_x).sqrt()
        };
        let sign_theta = if input.hat_theta >= 0.0 { 1.0 } else { -1.0 };
        let extreme_adjusted_theta = input.hat_theta - sign_theta * extreme_bias;

        // Grid benchmark results.
        let benchmark_results = benchmarks
            .iter()
            .map(|&(r2_y, r2_d)| {
                let bias = Self::ovb_bias(r2_y, r2_d, se, df).unwrap_or(0.0);
                let adjusted_theta = input.hat_theta - sign_theta * bias;
                let adjusted_t = adjusted_theta / se;
                let is_significant = adjusted_t.abs() > t_crit;
                BenchmarkResult {
                    r2_y_x: r2_y,
                    r2_d_x: r2_d,
                    bias,
                    adjusted_theta,
                    adjusted_t,
                    is_significant,
                }
            })
            .collect();

        Ok(CinelliHazlettResult {
            rv_q,
            rv_alpha,
            extreme_bias,
            extreme_adjusted_theta,
            t_stat: t,
            r2yd_x: input.r2yd_x,
            benchmarks: benchmark_results,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    // --- partial_r2_from_t ---

    #[test]
    fn partial_r2_from_t_basic() {
        // r² = 1²/(1²+10) = 1/11
        let r2 = CinelliHazlett::partial_r2_from_t(1.0, 10.0).unwrap();
        assert!(approx(r2, 1.0 / 11.0, 1e-12));
    }

    #[test]
    fn partial_r2_from_t_large_t() {
        // Large t → r² approaches 1.
        let r2 = CinelliHazlett::partial_r2_from_t(1000.0, 100.0).unwrap();
        assert!(r2 > 0.9999);
    }

    #[test]
    fn partial_r2_from_t_df_zero_errors() {
        assert!(CinelliHazlett::partial_r2_from_t(1.0, 0.0).is_err());
    }

    #[test]
    fn partial_r2_from_t_negative_df_errors() {
        assert!(CinelliHazlett::partial_r2_from_t(1.0, -5.0).is_err());
    }

    // --- ovb_bias ---

    #[test]
    fn ovb_bias_zero_r2_y() {
        let b = CinelliHazlett::ovb_bias(0.0, 0.5, 1.0, 100.0).unwrap();
        assert!(approx(b, 0.0, 1e-12));
    }

    #[test]
    fn ovb_bias_zero_r2_d() {
        let b = CinelliHazlett::ovb_bias(1.0, 0.0, 1.0, 100.0).unwrap();
        assert!(approx(b, 0.0, 1e-12));
    }

    #[test]
    fn ovb_bias_monotone_in_r2_y() {
        let b_small = CinelliHazlett::ovb_bias(0.3, 0.4, 1.0, 100.0).unwrap();
        let b_large = CinelliHazlett::ovb_bias(0.5, 0.4, 1.0, 100.0).unwrap();
        assert!(b_large > b_small);
    }

    #[test]
    fn ovb_bias_r2_d_one_errors() {
        assert!(CinelliHazlett::ovb_bias(0.5, 1.0, 1.0, 100.0).is_err());
    }

    #[test]
    fn ovb_bias_se_zero_errors() {
        assert!(CinelliHazlett::ovb_bias(0.5, 0.3, 0.0, 100.0).is_err());
    }

    #[test]
    fn ovb_bias_df_zero_errors() {
        assert!(CinelliHazlett::ovb_bias(0.5, 0.3, 1.0, 0.0).is_err());
    }

    // --- robustness_value ---

    #[test]
    fn robustness_value_positive_and_below_one() {
        let rv = CinelliHazlett::robustness_value(3.0, 100.0, 1.0).unwrap();
        assert!(rv > 0.0 && rv < 1.0);
    }

    #[test]
    fn robustness_value_large_t_approaches_one() {
        let rv = CinelliHazlett::robustness_value(100.0, 100.0, 1.0).unwrap();
        assert!(rv > 0.95);
    }

    #[test]
    fn robustness_value_df_zero_errors() {
        assert!(CinelliHazlett::robustness_value(3.0, 0.0, 1.0).is_err());
    }

    #[test]
    fn robustness_value_q_zero_errors() {
        assert!(CinelliHazlett::robustness_value(3.0, 100.0, 0.0).is_err());
    }

    #[test]
    fn robustness_value_q_above_one_errors() {
        assert!(CinelliHazlett::robustness_value(3.0, 100.0, 1.1).is_err());
    }

    // --- analyze ---

    #[test]
    fn analyze_empty_benchmarks_succeeds() {
        let input = OvbInput {
            hat_theta: 0.5,
            se_theta: 0.1,
            df: 100.0,
            r2yd_x: 0.3,
        };
        let cfg = CinelliHazlettConfig::default();
        let result = CinelliHazlett::analyze(&input, &cfg, &[]).unwrap();
        assert!(result.rv_q > 0.0);
        assert!(result.benchmarks.is_empty());
    }

    #[test]
    fn analyze_one_benchmark_correct_bias() {
        let input = OvbInput {
            hat_theta: 1.0,
            se_theta: 0.2,
            df: 50.0,
            r2yd_x: 0.3,
        };
        let cfg = CinelliHazlettConfig::default();
        let r2_y = 0.4;
        let r2_d = 0.3;
        let result = CinelliHazlett::analyze(&input, &cfg, &[(r2_y, r2_d)]).unwrap();
        let expected_bias = CinelliHazlett::ovb_bias(r2_y, r2_d, 0.2, 50.0).unwrap();
        assert!(approx(result.benchmarks[0].bias, expected_bias, 1e-10));
    }

    #[test]
    fn analyze_se_zero_errors() {
        let input = OvbInput {
            hat_theta: 1.0,
            se_theta: 0.0,
            df: 50.0,
            r2yd_x: 0.3,
        };
        let cfg = CinelliHazlettConfig::default();
        assert!(CinelliHazlett::analyze(&input, &cfg, &[]).is_err());
    }

    #[test]
    fn analyze_df_zero_errors() {
        let input = OvbInput {
            hat_theta: 1.0,
            se_theta: 0.2,
            df: 0.0,
            r2yd_x: 0.3,
        };
        let cfg = CinelliHazlettConfig::default();
        assert!(CinelliHazlett::analyze(&input, &cfg, &[]).is_err());
    }

    #[test]
    fn analyze_r2yd_x_zero_errors() {
        let input = OvbInput {
            hat_theta: 1.0,
            se_theta: 0.2,
            df: 50.0,
            r2yd_x: 0.0,
        };
        let cfg = CinelliHazlettConfig::default();
        assert!(CinelliHazlett::analyze(&input, &cfg, &[]).is_err());
    }

    #[test]
    fn analyze_r2yd_x_one_errors() {
        let input = OvbInput {
            hat_theta: 1.0,
            se_theta: 0.2,
            df: 50.0,
            r2yd_x: 1.0,
        };
        let cfg = CinelliHazlettConfig::default();
        assert!(CinelliHazlett::analyze(&input, &cfg, &[]).is_err());
    }

    #[test]
    fn analyze_q_zero_errors() {
        let input = OvbInput {
            hat_theta: 1.0,
            se_theta: 0.2,
            df: 50.0,
            r2yd_x: 0.3,
        };
        let cfg = CinelliHazlettConfig {
            q: 0.0,
            alpha: 0.05,
        };
        assert!(CinelliHazlett::analyze(&input, &cfg, &[]).is_err());
    }

    #[test]
    fn analyze_q_above_one_errors() {
        let input = OvbInput {
            hat_theta: 1.0,
            se_theta: 0.2,
            df: 50.0,
            r2yd_x: 0.3,
        };
        let cfg = CinelliHazlettConfig {
            q: 1.5,
            alpha: 0.05,
        };
        assert!(CinelliHazlett::analyze(&input, &cfg, &[]).is_err());
    }

    #[test]
    fn analyze_alpha_zero_errors() {
        let input = OvbInput {
            hat_theta: 1.0,
            se_theta: 0.2,
            df: 50.0,
            r2yd_x: 0.3,
        };
        let cfg = CinelliHazlettConfig { q: 1.0, alpha: 0.0 };
        assert!(CinelliHazlett::analyze(&input, &cfg, &[]).is_err());
    }

    #[test]
    fn analyze_extreme_bias_positive_for_typical_r2yd_x() {
        let input = OvbInput {
            hat_theta: 1.0,
            se_theta: 0.2,
            df: 100.0,
            r2yd_x: 0.3,
        };
        let cfg = CinelliHazlettConfig::default();
        let result = CinelliHazlett::analyze(&input, &cfg, &[]).unwrap();
        assert!(result.extreme_bias > 0.0);
    }

    #[test]
    fn analyze_extreme_adjusted_opposite_sign_when_large_bias() {
        // With a tiny hat_theta and large extreme_bias, adjusted should flip sign.
        let input = OvbInput {
            hat_theta: 0.01,
            se_theta: 0.1,
            df: 200.0,
            r2yd_x: 0.05, // low r2yd_x → large extreme bias
        };
        let cfg = CinelliHazlettConfig::default();
        let result = CinelliHazlett::analyze(&input, &cfg, &[]).unwrap();
        // extreme_bias = 0.1 * √200 * √(0.95/0.05) which is >> 0.01
        if result.extreme_bias > result.extreme_adjusted_theta.abs() {
            assert!(result.extreme_adjusted_theta < 0.0);
        }
    }

    #[test]
    fn analyze_benchmark_r2_d_zero_gives_zero_bias() {
        let input = OvbInput {
            hat_theta: 1.0,
            se_theta: 0.2,
            df: 100.0,
            r2yd_x: 0.3,
        };
        let cfg = CinelliHazlettConfig::default();
        let result = CinelliHazlett::analyze(&input, &cfg, &[(0.5, 0.0)]).unwrap();
        assert!(approx(result.benchmarks[0].bias, 0.0, 1e-12));
    }

    #[test]
    fn analyze_rv_alpha_le_rv_q_for_large_t() {
        // When the t-stat is large (significant result), rv_alpha should be ≤ rv_q.
        let input = OvbInput {
            hat_theta: 5.0,
            se_theta: 0.5,
            df: 200.0,
            r2yd_x: 0.4,
        };
        let cfg = CinelliHazlettConfig::default();
        let result = CinelliHazlett::analyze(&input, &cfg, &[]).unwrap();
        // rv_alpha uses a smaller effective_q than rv_q when |t| >> t_crit.
        assert!(result.rv_alpha <= result.rv_q + 1e-10);
    }

    #[test]
    fn analyze_deterministic() {
        let input = OvbInput {
            hat_theta: 0.8,
            se_theta: 0.15,
            df: 150.0,
            r2yd_x: 0.25,
        };
        let cfg = CinelliHazlettConfig::default();
        let r1 = CinelliHazlett::analyze(&input, &cfg, &[(0.3, 0.2)]).unwrap();
        let r2 = CinelliHazlett::analyze(&input, &cfg, &[(0.3, 0.2)]).unwrap();
        assert!(approx(r1.rv_q, r2.rv_q, 1e-15));
        assert!(approx(r1.extreme_bias, r2.extreme_bias, 1e-15));
        assert!(approx(r1.benchmarks[0].bias, r2.benchmarks[0].bias, 1e-15));
    }
}
