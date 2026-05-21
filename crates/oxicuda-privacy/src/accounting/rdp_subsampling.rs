//! Tight Rényi DP amplification via Poisson subsampling.
//!
//! Reference: Wang, Balle, Kasiviswanathan (2019), "Subsampled Rényi Differential Privacy
//! and Analytical Moments Accountant", AISTATS 2019.
//!
//! # Core Formula (Theorem 1)
//! For mechanism M satisfying (α, ε_M(α))-RDP, and Poisson subsampling at rate q:
//!
//! `ε_sub(α) = (1/(α-1)) · log( Σ_{k=0}^{α} C(α,k) · q^k · (1-q)^{α-k} · T_k )`
//!
//! where T_0 = T_1 = 1 and T_k = exp((k-1) · ε_M(k)) for k ≥ 2.
//!
//! # RDP → (ε, δ)-DP Conversion
//! Uses the optimal conversion from Balle, Barthe, Gaboardi (2020):
//! for composed n-step RDP accountant, convert each order and take the minimum ε.

use crate::error::{PrivacyError, PrivacyResult};

// ─── RdpMechanism ─────────────────────────────────────────────────────────────

/// A mechanism described by its Rényi Differential Privacy (RDP) profile.
///
/// Supports Gaussian, Laplace, and custom mechanisms specified by explicit RDP values.
#[derive(Debug, Clone)]
pub enum RdpMechanism {
    /// Gaussian mechanism with L2 sensitivity and noise standard deviation.
    Gaussian {
        /// L2 sensitivity Δ > 0.
        sensitivity: f64,
        /// Noise standard deviation σ > 0.
        sigma: f64,
    },
    /// Laplace mechanism with L1 sensitivity and scale parameter.
    Laplace {
        /// L1 sensitivity Δ > 0.
        sensitivity: f64,
        /// Noise scale b > 0 (standard deviation = b√2).
        scale: f64,
    },
    /// Custom mechanism with pre-specified RDP values per order.
    ///
    /// `rdp_values[i]` = ε(α = i + 2) for i = 0, 1, 2, ...
    Custom {
        /// RDP values indexed by α - 2.
        rdp_values: Vec<f64>,
    },
}

impl RdpMechanism {
    /// Compute ε_M(alpha) for integer `alpha` ≥ 2.
    ///
    /// # Errors
    /// - `InvalidParameter` if `alpha < 2`.
    /// - `InvalidParameter` if `alpha - 2` is out of range for `Custom` mechanism.
    /// - `NonPositiveSensitivity` if sensitivity ≤ 0 (Gaussian/Laplace).
    /// - `InvalidParameter` if sigma/scale ≤ 0 (Gaussian/Laplace).
    pub fn rdp_at(&self, alpha: usize) -> PrivacyResult<f64> {
        if alpha < 2 {
            return Err(PrivacyError::InvalidParameter(format!(
                "alpha must be ≥ 2 for RDP, got {alpha}"
            )));
        }
        match self {
            Self::Gaussian { sensitivity, sigma } => {
                if *sensitivity <= 0.0 {
                    return Err(PrivacyError::NonPositiveSensitivity(*sensitivity));
                }
                if *sigma <= 0.0 {
                    return Err(PrivacyError::InvalidParameter(format!(
                        "sigma must be positive, got {sigma}"
                    )));
                }
                Ok(RdpSubsampling::gaussian_rdp(
                    alpha as f64,
                    *sensitivity,
                    *sigma,
                ))
            }
            Self::Laplace { sensitivity, scale } => {
                if *sensitivity <= 0.0 {
                    return Err(PrivacyError::NonPositiveSensitivity(*sensitivity));
                }
                if *scale <= 0.0 {
                    return Err(PrivacyError::InvalidParameter(format!(
                        "scale must be positive, got {scale}"
                    )));
                }
                // Safe upper bound: ε_L(α) = α · Δ / b (convexity bound).
                Ok((alpha as f64) * sensitivity / scale)
            }
            Self::Custom { rdp_values } => {
                let idx = alpha - 2;
                if idx >= rdp_values.len() {
                    return Err(PrivacyError::InvalidParameter(format!(
                        "alpha={alpha} out of range: Custom has {} values (α up to {})",
                        rdp_values.len(),
                        rdp_values.len() + 1
                    )));
                }
                Ok(rdp_values[idx])
            }
        }
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for Poisson subsampling RDP amplification.
#[derive(Debug, Clone)]
pub struct RdpSubsamplingConfig {
    /// Poisson subsampling rate q ∈ (0, 1].
    pub sampling_rate: f64,
    /// Maximum Rényi order α (inclusive), must be ≥ 2.
    pub max_order: usize,
}

// ─── Result ───────────────────────────────────────────────────────────────────

/// Result of Poisson subsampling RDP amplification.
#[derive(Debug, Clone)]
pub struct RdpSubsamplingResult {
    /// The evaluated Rényi orders [2, 3, ..., max_order].
    pub orders: Vec<usize>,
    /// Amplified ε_sub(α) for each corresponding order.
    pub epsilon_rdp: Vec<f64>,
}

impl RdpSubsamplingResult {
    /// Convert n_steps compositions to (ε, δ)-DP via optimal RDP → (ε, δ) conversion.
    ///
    /// Uses the Balle-Barthe-Gaboardi 2020 formula for each α:
    /// `ε(α) = rdp_composed + (log((α-1)/α) - (log(δ) + log(α))) / (α-1)`
    ///
    /// Returns the minimum over all orders α.
    ///
    /// # Errors
    /// - `InvalidParameter` if `n_steps == 0`.
    /// - `InvalidDelta` if `delta ∉ (0, 1)`.
    /// - `InvalidParameter` if all orders produce NaN or infinite ε.
    pub fn to_epsilon_delta(&self, n_steps: usize, delta: f64) -> PrivacyResult<f64> {
        if n_steps == 0 {
            return Err(PrivacyError::InvalidParameter("n_steps must be ≥ 1".into()));
        }
        if !(delta > 0.0 && delta < 1.0) {
            return Err(PrivacyError::InvalidDelta(delta));
        }

        let mut best_epsilon = f64::INFINITY;

        for (i, &alpha) in self.orders.iter().enumerate() {
            let rdp_composed = (n_steps as f64) * self.epsilon_rdp[i];
            let a = alpha as f64;
            // Optimal RDP → (ε, δ) conversion (Balle et al. 2020):
            // ε(α) = rdp_n + (ln((α-1)/α) - (ln(δ) + ln(α))) / (α-1)
            let eps_candidate =
                rdp_composed + (((a - 1.0) / a).ln() - (delta.ln() + a.ln())) / (a - 1.0);

            if eps_candidate.is_finite() && eps_candidate < best_epsilon {
                best_epsilon = eps_candidate;
            }
        }

        if best_epsilon.is_infinite() {
            return Err(PrivacyError::InvalidParameter(
                "RDP → (ε,δ) conversion produced no finite ε for any order".into(),
            ));
        }

        Ok(best_epsilon.max(0.0))
    }
}

// ─── RdpSubsampling ───────────────────────────────────────────────────────────

/// Poisson subsampling RDP amplification (Wang-Balle-Kasiviswanathan 2019).
pub struct RdpSubsampling;

impl RdpSubsampling {
    /// Compute the subsampled RDP curve for a mechanism under Poisson subsampling.
    ///
    /// For each α ∈ {2, …, max_order}:
    ///
    /// `ε_sub(α) = (1/(α-1)) · ln( Σ_{k=0}^{α} C(α,k) · q^k · (1-q)^{α-k} · T_k )`
    ///
    /// where T_0 = T_1 = 1 and T_k = exp((k-1) · ε_M(k)) for k ≥ 2.
    ///
    /// # Errors
    /// - `InvalidParameter` if `sampling_rate ∉ (0, 1]` or `max_order < 2`.
    /// - Propagates errors from `mechanism.rdp_at`.
    /// - `InvalidParameter` if the binomial sum produces NaN (numerical overflow).
    pub fn amplify(
        mechanism: &RdpMechanism,
        cfg: &RdpSubsamplingConfig,
    ) -> PrivacyResult<RdpSubsamplingResult> {
        let q = cfg.sampling_rate;
        if q <= 0.0 || q > 1.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "sampling_rate must be in (0, 1], got {q}"
            )));
        }
        if cfg.max_order < 2 {
            return Err(PrivacyError::InvalidParameter(format!(
                "max_order must be ≥ 2, got {}",
                cfg.max_order
            )));
        }

        let one_minus_q = 1.0 - q;
        let mut orders = Vec::with_capacity(cfg.max_order - 1);
        let mut epsilon_rdp = Vec::with_capacity(cfg.max_order - 1);

        for alpha in 2..=cfg.max_order {
            // Accumulate the binomial sum S = Σ_{k=0}^{α} C(α,k)·q^k·(1-q)^{α-k}·T_k
            let mut sum = 0.0_f64;

            for k in 0..=alpha {
                let binom = Self::binomial_coeff(alpha, k);
                let q_pow = q.powi(k as i32);
                let omq_pow = one_minus_q.powi((alpha - k) as i32);
                // T_k: T_0 = T_1 = 1, T_k = exp((k-1) * ε_M(k)) for k ≥ 2.
                let t_k = if k < 2 {
                    1.0
                } else {
                    let rdp = mechanism.rdp_at(k)?;
                    ((k - 1) as f64 * rdp).exp()
                };
                sum += binom * q_pow * omq_pow * t_k;
            }

            // Guard against numerical anomalies.
            let eps_sub = if sum.is_nan() {
                return Err(PrivacyError::InvalidParameter(format!(
                    "NaN encountered in binomial sum at alpha={alpha}"
                )));
            } else if sum <= 0.0 {
                0.0
            } else {
                (sum.ln() / (alpha - 1) as f64).max(0.0)
            };

            orders.push(alpha);
            epsilon_rdp.push(eps_sub);
        }

        Ok(RdpSubsamplingResult {
            orders,
            epsilon_rdp,
        })
    }

    /// Gaussian mechanism RDP: ε_Gauss(α) = α · Δ² / (2σ²).
    ///
    /// This is the exact closed-form RDP for the Gaussian mechanism
    /// (Mironov 2017, Proposition 3).
    #[must_use]
    pub fn gaussian_rdp(alpha: f64, sensitivity: f64, sigma: f64) -> f64 {
        alpha * sensitivity * sensitivity / (2.0 * sigma * sigma)
    }

    /// Binomial coefficient C(n, k) computed via iterative integer product.
    ///
    /// Uses the recurrence `C(n, k) = Π_{i=0}^{k-1} (n-i)/(i+1)` with symmetry
    /// `C(n, k) = C(n, n-k)` to minimize multiplications. Exact for n, k ≤ ~60
    /// before f64 precision degrades; accurate to within rounding for larger n.
    ///
    /// Returns 0.0 if k > n, and 1.0 for k == 0 or k == n.
    #[must_use]
    pub fn binomial_coeff(n: usize, k: usize) -> f64 {
        if k > n {
            return 0.0;
        }
        if k == 0 || k == n {
            return 1.0;
        }
        // Exploit symmetry C(n, k) = C(n, n-k).
        let k = k.min(n - k);
        let mut result = 1.0_f64;
        for i in 0..k {
            result *= (n - i) as f64 / (i + 1) as f64;
        }
        result
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── RdpMechanism::rdp_at ─────────────────────────────────────────────────

    #[test]
    fn rdp_at_gaussian() {
        // Gaussian: ε(α) = α·Δ²/(2σ²). With Δ=σ=1, ε(2) = 2/2 = 1.
        let m = RdpMechanism::Gaussian {
            sensitivity: 1.0,
            sigma: 1.0,
        };
        let eps = m.rdp_at(2).expect("ok");
        let expected = RdpSubsampling::gaussian_rdp(2.0, 1.0, 1.0);
        assert!(
            (eps - expected).abs() < 1e-12,
            "rdp_at(2) = {eps}, expected {expected}"
        );
    }

    #[test]
    fn rdp_at_laplace() {
        // Laplace bound: ε(α) = α·Δ/b = 2·1.0/1.0 = 2.
        let m = RdpMechanism::Laplace {
            sensitivity: 1.0,
            scale: 1.0,
        };
        let eps = m.rdp_at(2).expect("ok");
        assert!((eps - 2.0).abs() < 1e-12, "got {eps}");
    }

    #[test]
    fn rdp_at_alpha_lt_2_err() {
        let m = RdpMechanism::Gaussian {
            sensitivity: 1.0,
            sigma: 1.0,
        };
        assert!(m.rdp_at(0).is_err());
        assert!(m.rdp_at(1).is_err());
    }

    // ── RdpSubsampling static methods ─────────────────────────────────────────

    #[test]
    fn gaussian_rdp_formula() {
        // ε_Gauss(2, 1, 1) = 2·1·1/(2·1·1) = 1.0.
        let eps = RdpSubsampling::gaussian_rdp(2.0, 1.0, 1.0);
        assert!((eps - 1.0).abs() < 1e-12, "expected 1.0, got {eps}");
    }

    #[test]
    fn binomial_c52() {
        let c = RdpSubsampling::binomial_coeff(5, 2);
        assert!((c - 10.0).abs() < 1e-10, "C(5,2) = {c}, expected 10");
    }

    #[test]
    fn binomial_c100() {
        assert!((RdpSubsampling::binomial_coeff(10, 0) - 1.0).abs() < 1e-12);
        assert!((RdpSubsampling::binomial_coeff(10, 10) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn binomial_symmetry() {
        let c1 = RdpSubsampling::binomial_coeff(8, 3);
        let c2 = RdpSubsampling::binomial_coeff(8, 5);
        assert!((c1 - c2).abs() < 1e-10, "C(8,3)={c1}, C(8,5)={c2}");
    }

    #[test]
    fn binomial_k_gt_n_returns_zero() {
        assert!((RdpSubsampling::binomial_coeff(3, 5) - 0.0).abs() < 1e-12);
    }

    // ── Custom mechanism ──────────────────────────────────────────────────────

    #[test]
    fn custom_round_trip() {
        // rdp_values[0]=0.5 → α=2, rdp_values[1]=0.3 → α=3, rdp_values[2]=0.2 → α=4.
        let m = RdpMechanism::Custom {
            rdp_values: vec![0.5, 0.3, 0.2],
        };
        assert!((m.rdp_at(2).expect("ok") - 0.5).abs() < 1e-12);
        assert!((m.rdp_at(4).expect("ok") - 0.2).abs() < 1e-12);
    }

    #[test]
    fn custom_out_of_range() {
        let m = RdpMechanism::Custom {
            rdp_values: vec![0.5],
        };
        // Only α=2 defined; α=4 → index 2 out of range.
        assert!(m.rdp_at(4).is_err());
    }

    // ── amplify correctness ───────────────────────────────────────────────────

    #[test]
    fn q_1_no_amplification() {
        // q=1: only k=α term survives in sum → S = T_α = exp((α-1)·ε_M(α)).
        // So ε_sub = ln(T_α)/(α-1) = ε_M(α). Verify for α=2.
        let m = RdpMechanism::Gaussian {
            sensitivity: 1.0,
            sigma: 1.0,
        };
        let cfg = RdpSubsamplingConfig {
            sampling_rate: 1.0,
            max_order: 2,
        };
        let result = RdpSubsampling::amplify(&m, &cfg).expect("ok");
        let eps_sub = result.epsilon_rdp[0]; // α=2
        let eps_m2 = m.rdp_at(2).expect("ok");
        assert!(
            (eps_sub - eps_m2).abs() < 1e-8,
            "q=1 should give ε_sub(2)=ε_M(2)={eps_m2:.6}, got {eps_sub:.6}"
        );
    }

    #[test]
    fn q_small_amplifies() {
        // Small q should give ε_sub << ε_M for Gaussian mechanism.
        let m = RdpMechanism::Gaussian {
            sensitivity: 1.0,
            sigma: 1.0,
        };
        let eps_m2 = m.rdp_at(2).expect("ok");
        let cfg = RdpSubsamplingConfig {
            sampling_rate: 0.01,
            max_order: 2,
        };
        let result = RdpSubsampling::amplify(&m, &cfg).expect("ok");
        let eps_sub = result.epsilon_rdp[0];
        assert!(
            eps_sub < eps_m2,
            "q=0.01 should amplify: ε_sub={eps_sub:.6} < ε_M(2)={eps_m2:.6}"
        );
    }

    #[test]
    fn q_small_lt_q_large() {
        // Smaller subsampling rate → smaller ε_sub at same order.
        let m = RdpMechanism::Gaussian {
            sensitivity: 1.0,
            sigma: 1.0,
        };
        let cfg_small = RdpSubsamplingConfig {
            sampling_rate: 0.01,
            max_order: 2,
        };
        let cfg_large = RdpSubsamplingConfig {
            sampling_rate: 0.1,
            max_order: 2,
        };
        let eps_small = RdpSubsampling::amplify(&m, &cfg_small)
            .expect("ok")
            .epsilon_rdp[0];
        let eps_large = RdpSubsampling::amplify(&m, &cfg_large)
            .expect("ok")
            .epsilon_rdp[0];
        assert!(
            eps_small < eps_large,
            "q=0.01 should give smaller ε than q=0.1: {eps_small:.8} vs {eps_large:.8}"
        );
    }

    #[test]
    fn epsilon_rdp_length() {
        // max_order=5 → orders = [2,3,4,5], length = 4.
        let m = RdpMechanism::Gaussian {
            sensitivity: 1.0,
            sigma: 1.0,
        };
        let cfg = RdpSubsamplingConfig {
            sampling_rate: 0.1,
            max_order: 5,
        };
        let result = RdpSubsampling::amplify(&m, &cfg).expect("ok");
        assert_eq!(result.orders.len(), 4, "orders: {:?}", result.orders);
        assert_eq!(result.epsilon_rdp.len(), 4);
        assert_eq!(result.orders, vec![2, 3, 4, 5]);
    }

    // ── to_epsilon_delta ─────────────────────────────────────────────────────

    #[test]
    fn to_epsilon_delta_finite() {
        // Gaussian, q=0.01, max_order=10, 100 steps, δ=1e-5 → finite positive ε.
        let m = RdpMechanism::Gaussian {
            sensitivity: 1.0,
            sigma: 1.0,
        };
        let cfg = RdpSubsamplingConfig {
            sampling_rate: 0.01,
            max_order: 10,
        };
        let result = RdpSubsampling::amplify(&m, &cfg).expect("ok");
        let eps = result.to_epsilon_delta(100, 1e-5).expect("ok");
        assert!(
            eps > 0.0 && eps.is_finite(),
            "expected finite positive ε, got {eps}"
        );
    }

    #[test]
    fn to_epsilon_delta_err_n_steps_zero() {
        let m = RdpMechanism::Gaussian {
            sensitivity: 1.0,
            sigma: 1.0,
        };
        let cfg = RdpSubsamplingConfig {
            sampling_rate: 0.1,
            max_order: 5,
        };
        let result = RdpSubsampling::amplify(&m, &cfg).expect("ok");
        assert!(result.to_epsilon_delta(0, 1e-5).is_err());
    }

    #[test]
    fn to_epsilon_delta_err_delta() {
        let m = RdpMechanism::Gaussian {
            sensitivity: 1.0,
            sigma: 1.0,
        };
        let cfg = RdpSubsamplingConfig {
            sampling_rate: 0.1,
            max_order: 5,
        };
        let result = RdpSubsampling::amplify(&m, &cfg).expect("ok");
        assert!(result.to_epsilon_delta(10, 0.0).is_err());
        assert!(result.to_epsilon_delta(10, 1.0).is_err());
    }

    // ── Error paths for amplify ───────────────────────────────────────────────

    #[test]
    fn err_q_le_0() {
        let m = RdpMechanism::Gaussian {
            sensitivity: 1.0,
            sigma: 1.0,
        };
        let cfg = RdpSubsamplingConfig {
            sampling_rate: 0.0,
            max_order: 4,
        };
        assert!(RdpSubsampling::amplify(&m, &cfg).is_err());
    }

    #[test]
    fn err_q_gt_1() {
        let m = RdpMechanism::Gaussian {
            sensitivity: 1.0,
            sigma: 1.0,
        };
        let cfg = RdpSubsamplingConfig {
            sampling_rate: 1.5,
            max_order: 4,
        };
        assert!(RdpSubsampling::amplify(&m, &cfg).is_err());
    }

    #[test]
    fn err_max_order_lt_2() {
        let m = RdpMechanism::Gaussian {
            sensitivity: 1.0,
            sigma: 1.0,
        };
        let cfg = RdpSubsamplingConfig {
            sampling_rate: 0.1,
            max_order: 1,
        };
        assert!(RdpSubsampling::amplify(&m, &cfg).is_err());
    }

    // ── Monotonicity ──────────────────────────────────────────────────────────

    #[test]
    fn monotone_alpha_gaussian() {
        // For Gaussian mechanism with q=0.5, ε_sub should increase with α.
        // (Larger Rényi order → larger RDP → larger subsampled bound.)
        let m = RdpMechanism::Gaussian {
            sensitivity: 1.0,
            sigma: 1.0,
        };
        let cfg = RdpSubsamplingConfig {
            sampling_rate: 0.5,
            max_order: 8,
        };
        let result = RdpSubsampling::amplify(&m, &cfg).expect("ok");
        for w in result.epsilon_rdp.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-10,
                "non-monotone: ε[i+1]={:.6} < ε[i]={:.6}",
                w[1],
                w[0]
            );
        }
    }

    // ── Laplace mechanism ─────────────────────────────────────────────────────

    #[test]
    fn laplace_amplify_runs() {
        // Laplace mechanism should work with amplification.
        let m = RdpMechanism::Laplace {
            sensitivity: 1.0,
            scale: 1.0,
        };
        let cfg = RdpSubsamplingConfig {
            sampling_rate: 0.05,
            max_order: 6,
        };
        let result = RdpSubsampling::amplify(&m, &cfg).expect("ok");
        assert_eq!(result.orders.len(), 5);
        for &eps in &result.epsilon_rdp {
            assert!(eps >= 0.0 && eps.is_finite(), "invalid eps: {eps}");
        }
    }
}
