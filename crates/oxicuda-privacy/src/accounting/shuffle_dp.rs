//! Shuffle-DP privacy amplification via the Feldman-McMillan-Talwar 2022 bound.
//!
//! Reference: Feldman, McMillan, Talwar (2022), "Hiding Among the Clones:
//! A Simple and Nearly Optimal Analysis of Privacy Amplification by Shuffling",
//! STOC 2022 / FOCS 2021.
//!
//! # Main Result (Theorem 1)
//! If each of n users applies an ε₀-LDP mechanism M and the outputs are shuffled,
//! the shuffled protocol satisfies (ε_central, δ)-DP where:
//!
//! `ε_central ≤ log(1 + (e^{ε₀} − 1) · A)`
//!
//! and the amplification coefficient A = 8 · √(e^{ε₀} · ln(4/δ) / n).
//!
//! The result always yields ε_central ≤ ε₀ (no worse than local DP).
//!
//! # Composition
//! Basic adaptive composition is used for multiple rounds: total ε = t · ε_central,
//! total δ = t · δ (union bound).

use crate::error::{PrivacyError, PrivacyResult};

// ─── ShuffleConfig ────────────────────────────────────────────────────────────

/// Configuration for Shuffle-DP amplification.
#[derive(Debug, Clone)]
pub struct ShuffleConfig {
    /// Number of users contributing to the shuffle protocol, must be ≥ 2.
    pub n_users: usize,
    /// Target failure probability δ ∈ (0, 1).
    pub delta: f64,
}

// ─── ShuffleResult ────────────────────────────────────────────────────────────

/// Result of Shuffle-DP privacy amplification.
#[derive(Debug, Clone)]
pub struct ShuffleResult {
    /// Central DP guarantee ε_central ≤ ε₀ (local DP parameter).
    pub epsilon_central: f64,
    /// Amplification factor ε₀ / ε_central (∞ if ε_central ≈ 0).
    pub amplification_factor: f64,
}

// ─── ShuffleDp ───────────────────────────────────────────────────────────────

/// Shuffle-DP privacy amplification (Feldman-McMillan-Talwar 2022).
pub struct ShuffleDp;

impl ShuffleDp {
    /// Compute the FMT amplification coefficient A = 8 · √(e^{ε₀} · ln(4/δ) / n).
    ///
    /// A larger A means more privacy cost relative to the shuffle size.
    ///
    /// # Errors
    /// - `InvalidParameter` if `epsilon_local < 0` or `n == 0`.
    /// - `InvalidDelta` if `delta ∉ (0, 1)`.
    pub fn fmt_coefficient(epsilon_local: f64, n: usize, delta: f64) -> PrivacyResult<f64> {
        if epsilon_local < 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "epsilon_local must be ≥ 0, got {epsilon_local}"
            )));
        }
        if !(delta > 0.0 && delta < 1.0) {
            return Err(PrivacyError::InvalidDelta(delta));
        }
        if n == 0 {
            return Err(PrivacyError::InvalidParameter("n must be ≥ 1".into()));
        }
        let a = 8.0 * (epsilon_local.exp() * (4.0 / delta).ln() / n as f64).sqrt();
        Ok(a)
    }

    /// Compute central DP guarantee for a single shuffle round.
    ///
    /// Applies the FMT 2022 Theorem 1 bound:
    /// `ε_central = log(1 + (e^{ε₀} − 1) · A)` clamped to [0, ε₀].
    ///
    /// # Errors
    /// - `InvalidParameter` if `epsilon_local < 0`.
    /// - `InvalidParameter` if `cfg.n_users < 2`.
    /// - `InvalidDelta` if `cfg.delta ∉ (0, 1)`.
    pub fn amplify(epsilon_local: f64, cfg: &ShuffleConfig) -> PrivacyResult<ShuffleResult> {
        Self::validate_inputs(epsilon_local, cfg)?;

        // Special case: 0-LDP is already 0-DP centrally.
        if epsilon_local == 0.0 {
            return Ok(ShuffleResult {
                epsilon_central: 0.0,
                amplification_factor: f64::INFINITY,
            });
        }

        let a = Self::fmt_coefficient(epsilon_local, cfg.n_users, cfg.delta)?;
        let factor = (epsilon_local.exp() - 1.0) * a;
        let eps_central = (1.0 + factor).ln();

        // Clamp: cannot be worse than local DP and must be non-negative.
        let eps_central = eps_central.min(epsilon_local).max(0.0);

        let amplification_factor = if eps_central > 1e-12 {
            epsilon_local / eps_central
        } else {
            f64::INFINITY
        };

        Ok(ShuffleResult {
            epsilon_central: eps_central,
            amplification_factor,
        })
    }

    /// Compute central DP guarantee when each user sends k messages per shuffle round.
    ///
    /// Treats k messages as effective local privacy ε_eff = k · ε₀ and computes
    /// the total central guarantee for the combined k-message per-user protocol.
    ///
    /// # Errors
    /// - `InvalidParameter` if `k == 0`.
    /// - Propagates `amplify` errors.
    pub fn amplify_multi(
        epsilon_local: f64,
        k: usize,
        cfg: &ShuffleConfig,
    ) -> PrivacyResult<ShuffleResult> {
        if k == 0 {
            return Err(PrivacyError::InvalidParameter(
                "k (messages per user) must be ≥ 1".into(),
            ));
        }
        let eps_eff = k as f64 * epsilon_local;
        let single = Self::amplify(eps_eff, cfg)?;
        let amplification_factor = if single.epsilon_central > 1e-12 {
            eps_eff / single.epsilon_central
        } else {
            f64::INFINITY
        };
        Ok(ShuffleResult {
            epsilon_central: single.epsilon_central,
            amplification_factor,
        })
    }

    /// Compose t independent shuffle rounds via basic composition.
    ///
    /// Total guarantees: ε_total = t · ε_central, δ_total = t · δ (union bound).
    ///
    /// # Errors
    /// - `InvalidParameter` if `t_rounds == 0`.
    /// - Propagates `amplify` errors.
    pub fn compose_rounds(
        epsilon_local: f64,
        cfg: &ShuffleConfig,
        t_rounds: usize,
    ) -> PrivacyResult<(f64, f64)> {
        if t_rounds == 0 {
            return Err(PrivacyError::InvalidParameter(
                "t_rounds must be ≥ 1".into(),
            ));
        }
        let single = Self::amplify(epsilon_local, cfg)?;
        let eps_total = t_rounds as f64 * single.epsilon_central;
        let delta_total = t_rounds as f64 * cfg.delta;
        Ok((eps_total, delta_total))
    }

    /// Find the minimum number of users required to achieve ε_target central DP.
    ///
    /// Solves the FMT bound for n:
    ///   n ≥ 64 · e^{ε₀} · ln(4/δ) · ((e^{ε₀}−1) / (e^{ε_target}−1))²
    ///
    /// Returns 0 if ε_target ≥ ε₀ (no amplification needed) or ε₀ = 0.
    ///
    /// # Errors
    /// - `InvalidParameter` if `epsilon_local < 0` or `epsilon_target ≤ 0`.
    /// - `InvalidDelta` if `delta ∉ (0, 1)`.
    pub fn min_users(epsilon_local: f64, epsilon_target: f64, delta: f64) -> PrivacyResult<usize> {
        if epsilon_local < 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "epsilon_local must be ≥ 0, got {epsilon_local}"
            )));
        }
        if epsilon_target <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "epsilon_target must be > 0, got {epsilon_target}"
            )));
        }
        if !(delta > 0.0 && delta < 1.0) {
            return Err(PrivacyError::InvalidDelta(delta));
        }

        // No amplification needed cases.
        if epsilon_target >= epsilon_local || epsilon_local == 0.0 {
            return Ok(0);
        }

        // Derive from: (e^{ε₀}−1)·8·√(e^{ε₀}·ln(4/δ)/n) ≤ e^{ε_target}−1
        // Solving for n: n ≥ 64·e^{ε₀}·ln(4/δ)·((e^{ε₀}−1)/(e^{ε_target}−1))²
        let exp_local = epsilon_local.exp();
        let exp_target = epsilon_target.exp();
        let ratio = (exp_local - 1.0) / (exp_target - 1.0);
        let n_f64 = 64.0 * exp_local * (4.0 / delta).ln() * ratio * ratio;
        let n = (n_f64.ceil() as usize).max(2);
        Ok(n)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn validate_inputs(epsilon_local: f64, cfg: &ShuffleConfig) -> PrivacyResult<()> {
        if epsilon_local < 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "epsilon_local must be ≥ 0, got {epsilon_local}"
            )));
        }
        if !(cfg.delta > 0.0 && cfg.delta < 1.0) {
            return Err(PrivacyError::InvalidDelta(cfg.delta));
        }
        if cfg.n_users < 2 {
            return Err(PrivacyError::InvalidParameter(format!(
                "n_users must be ≥ 2, got {}",
                cfg.n_users
            )));
        }
        Ok(())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cfg(n: usize, delta: f64) -> ShuffleConfig {
        ShuffleConfig { n_users: n, delta }
    }

    // ── Core amplification ────────────────────────────────────────────────────

    #[test]
    fn large_n_amplifies() {
        // Large shuffle → significant amplification: ε_central < ε₀.
        let cfg = make_cfg(1_000_000, 1e-6);
        let result = ShuffleDp::amplify(1.0, &cfg).expect("ok");
        assert!(
            result.epsilon_central < 1.0,
            "ε_central={:.6} should be < ε₀=1.0",
            result.epsilon_central
        );
    }

    #[test]
    fn small_n_near_local() {
        // Very small shuffle; ε_central is clamped to ε₀.
        let cfg = make_cfg(2, 1e-6);
        let result = ShuffleDp::amplify(1.0, &cfg).expect("ok");
        assert!(
            result.epsilon_central <= 1.0,
            "ε_central={:.6} must be ≤ ε₀=1.0",
            result.epsilon_central
        );
    }

    #[test]
    fn epsilon_zero_local() {
        // ε₀ = 0 → already 0-DP, ε_central = 0.
        let cfg = make_cfg(1000, 1e-5);
        let result = ShuffleDp::amplify(0.0, &cfg).expect("ok");
        assert!(
            result.epsilon_central.abs() < 1e-12,
            "ε_central should be 0, got {}",
            result.epsilon_central
        );
    }

    #[test]
    fn epsilon_central_le_local() {
        // ε_central ≤ ε₀ for various parameters.
        for &eps0 in &[0.1f64, 0.5, 1.0, 2.0, 5.0] {
            for &n in &[10usize, 1000, 100_000] {
                let cfg = make_cfg(n, 1e-6);
                let result = ShuffleDp::amplify(eps0, &cfg).expect("ok");
                assert!(
                    result.epsilon_central <= eps0 + 1e-10,
                    "ε_central={:.6} > ε₀={eps0} for n={n}",
                    result.epsilon_central
                );
            }
        }
    }

    // ── FMT coefficient ───────────────────────────────────────────────────────

    #[test]
    fn fmt_coefficient_monotone_n() {
        // Larger n → smaller coefficient (more amplification).
        let a_small = ShuffleDp::fmt_coefficient(1.0, 100, 1e-5).expect("ok");
        let a_large = ShuffleDp::fmt_coefficient(1.0, 10_000, 1e-5).expect("ok");
        assert!(
            a_small > a_large,
            "A(n=100)={a_small:.6} should be > A(n=10000)={a_large:.6}"
        );
    }

    // ── Amplification factor ──────────────────────────────────────────────────

    #[test]
    fn amplification_factor_ge_1() {
        // Amplification factor ≥ 1 for any valid inputs (ε_local ≥ ε_central).
        for &n in &[100usize, 10_000, 1_000_000] {
            let cfg = make_cfg(n, 1e-6);
            let result = ShuffleDp::amplify(1.0, &cfg).expect("ok");
            assert!(
                result.amplification_factor >= 1.0 - 1e-10,
                "amplification_factor={:.4} should be ≥ 1 for n={n}",
                result.amplification_factor
            );
        }
    }

    // ── compose_rounds ────────────────────────────────────────────────────────

    #[test]
    fn compose_rounds_1_eq_amplify() {
        let cfg = make_cfg(10_000, 1e-6);
        let (eps_t1, _) = ShuffleDp::compose_rounds(1.0, &cfg, 1).expect("ok");
        let single = ShuffleDp::amplify(1.0, &cfg).expect("ok").epsilon_central;
        assert!(
            (eps_t1 - single).abs() < 1e-12,
            "t=1 compose ε={eps_t1} vs amplify ε={single}"
        );
    }

    #[test]
    fn compose_rounds_t2() {
        let cfg = make_cfg(10_000, 1e-6);
        let (eps_t1, _) = ShuffleDp::compose_rounds(1.0, &cfg, 1).expect("ok");
        let (eps_t2, _) = ShuffleDp::compose_rounds(1.0, &cfg, 2).expect("ok");
        assert!(
            (eps_t2 - 2.0 * eps_t1).abs() < 1e-12,
            "t=2 should be 2×t=1: {eps_t2:.8} vs {:.8}",
            2.0 * eps_t1
        );
    }

    // ── amplify_multi ─────────────────────────────────────────────────────────

    #[test]
    fn amplify_multi_k1_eq_amplify() {
        let cfg = make_cfg(10_000, 1e-6);
        let single = ShuffleDp::amplify(1.0, &cfg).expect("ok").epsilon_central;
        let multi = ShuffleDp::amplify_multi(1.0, 1, &cfg)
            .expect("ok")
            .epsilon_central;
        assert!(
            (single - multi).abs() < 1e-12,
            "k=1 multi={multi:.8} should equal single={single:.8}"
        );
    }

    #[test]
    fn amplify_multi_k2_larger_than_k1() {
        // More messages → higher effective ε → less amplification → larger ε_central.
        let cfg = make_cfg(10_000, 1e-6);
        let eps_k1 = ShuffleDp::amplify_multi(1.0, 1, &cfg)
            .expect("ok")
            .epsilon_central;
        let eps_k2 = ShuffleDp::amplify_multi(1.0, 2, &cfg)
            .expect("ok")
            .epsilon_central;
        assert!(
            eps_k2 >= eps_k1,
            "k=2 ε_central={eps_k2:.6} should be ≥ k=1 ε_central={eps_k1:.6}"
        );
    }

    // ── min_users ─────────────────────────────────────────────────────────────

    #[test]
    fn min_users_returns_valid_n() {
        // The returned n should satisfy amplify(ε₀, n, δ).ε_central ≤ ε_target.
        let eps0 = 2.0;
        let eps_target = 0.5;
        let delta = 1e-6;
        let n = ShuffleDp::min_users(eps0, eps_target, delta).expect("ok");
        assert!(n >= 2, "n={n} must be ≥ 2");
        let cfg = make_cfg(n, delta);
        let result = ShuffleDp::amplify(eps0, &cfg).expect("ok");
        assert!(
            result.epsilon_central <= eps_target + 1e-9,
            "ε_central={:.6} should be ≤ ε_target={eps_target}",
            result.epsilon_central
        );
    }

    #[test]
    fn min_users_target_ge_local() {
        // ε_target ≥ ε₀ → no shuffle needed, return 0.
        let n = ShuffleDp::min_users(1.0, 2.0, 1e-5).expect("ok");
        assert_eq!(n, 0, "ε_target ≥ ε_local should need 0 users");
    }

    // ── Error paths ───────────────────────────────────────────────────────────

    #[test]
    fn err_epsilon_negative() {
        let cfg = make_cfg(1000, 1e-5);
        assert!(ShuffleDp::amplify(-0.1, &cfg).is_err());
    }

    #[test]
    fn err_n_lt_2() {
        let cfg = ShuffleConfig {
            n_users: 1,
            delta: 1e-5,
        };
        assert!(ShuffleDp::amplify(1.0, &cfg).is_err());
    }

    #[test]
    fn err_delta_zero() {
        let cfg = ShuffleConfig {
            n_users: 1000,
            delta: 0.0,
        };
        assert!(ShuffleDp::amplify(1.0, &cfg).is_err());
    }

    #[test]
    fn err_delta_one() {
        let cfg = ShuffleConfig {
            n_users: 1000,
            delta: 1.0,
        };
        assert!(ShuffleDp::amplify(1.0, &cfg).is_err());
    }

    #[test]
    fn err_t_rounds_0() {
        let cfg = make_cfg(1000, 1e-5);
        assert!(ShuffleDp::compose_rounds(1.0, &cfg, 0).is_err());
    }

    #[test]
    fn err_k_0() {
        let cfg = make_cfg(1000, 1e-5);
        assert!(ShuffleDp::amplify_multi(1.0, 0, &cfg).is_err());
    }
}
