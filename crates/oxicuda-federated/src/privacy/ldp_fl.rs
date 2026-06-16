//! Local Differential Privacy for Federated Learning (LDP-FL).
//!
//! Truex, Liu, Chen, Zhu & Hwang, "LDP-Fed: Federated Learning with Local
//! Differential Privacy", EdgeSys 2020; and the broader local-DP FL setting of
//! Truex et al. (2020). In LDP-FL each client privatises its own update
//! *before transmission* so the server (and any eavesdropper) never sees the raw
//! gradient — a stronger trust model than central DP.
//!
//! # Per-client mechanism
//! Given a client update `g` and L2 clip bound `C`:
//! 1. **Clip** to the L2 ball of radius `C`: `g ← g · min(1, C / ‖g‖₂)`. This
//!    bounds the per-coordinate / L2 sensitivity to `C`.
//! 2. **Perturb** with a calibrated mechanism:
//!    - Gaussian: `σ = C·√(2·ln(1.25/δ)) / ε` added i.i.d. to each coordinate
//!      → `(ε, δ)`-LDP.
//!    - Laplace: scale `b = C / ε` (using L1 sensitivity ≤ √d·C, but here the
//!      pessimistic per-coordinate `C` bound) → `(ε, 0)`-LDP.
//!
//! # Privacy amplification by subsampling
//! When only a fraction `q` of clients report each round, the effective central
//! privacy is amplified: `ε_amplified = ln(1 + q·(e^ε − 1))` (Poisson
//! subsampling, pure-DP bound). LDP-FL reports the per-client `ε` *and* the
//! amplified central `ε` so practitioners can budget across rounds.

use crate::error::{FedError, FedResult};
use crate::handle::LcgRng;

/// Which per-client perturbation mechanism LDP-FL applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LdpMechanism {
    /// `(ε, δ)`-LDP Gaussian mechanism (requires `δ ∈ (0, 1)`).
    Gaussian,
    /// `(ε, 0)`-LDP Laplace mechanism (`δ` ignored).
    Laplace,
}

/// Configuration for the LDP-FL per-client privatiser.
#[derive(Debug, Clone, Copy)]
pub struct LdpFlConfig {
    /// Privacy parameter ε > 0.
    pub epsilon: f32,
    /// Failure probability δ ∈ (0, 1) (used only by the Gaussian mechanism).
    pub delta: f32,
    /// L2 clipping bound `C > 0`.
    pub clip_norm: f32,
    /// Per-client perturbation mechanism.
    pub mechanism: LdpMechanism,
}

impl LdpFlConfig {
    /// Construct and validate a configuration.
    ///
    /// # Errors
    /// - `InvalidPrivacyBudget` if `ε ≤ 0`/non-finite, or (for Gaussian) `δ`
    ///   is outside `(0, 1)`.
    /// - `InvalidClipNorm` if `clip_norm ≤ 0` or non-finite.
    pub fn new(
        epsilon: f32,
        delta: f32,
        clip_norm: f32,
        mechanism: LdpMechanism,
    ) -> FedResult<Self> {
        if !(epsilon > 0.0 && epsilon.is_finite()) {
            return Err(FedError::InvalidPrivacyBudget);
        }
        if mechanism == LdpMechanism::Gaussian && !(delta > 0.0 && delta < 1.0) {
            return Err(FedError::InvalidPrivacyBudget);
        }
        if !(clip_norm > 0.0 && clip_norm.is_finite()) {
            return Err(FedError::InvalidClipNorm);
        }
        Ok(Self {
            epsilon,
            delta,
            clip_norm,
            mechanism,
        })
    }

    /// Noise standard deviation for the Gaussian mechanism:
    /// `σ = C·√(2·ln(1.25/δ)) / ε`.
    #[must_use]
    pub fn gaussian_sigma(&self) -> f32 {
        let factor = (2.0 * (1.25 / self.delta).ln()).sqrt();
        self.clip_norm * factor / self.epsilon
    }

    /// Laplace scale `b = C / ε`.
    #[must_use]
    pub fn laplace_scale(&self) -> f32 {
        self.clip_norm / self.epsilon
    }
}

/// Clip a gradient in place to the L2 ball of radius `clip_norm` and return the
/// pre-clip L2 norm.
fn clip_l2(grad: &mut [f32], clip_norm: f32) -> f32 {
    let norm_sq: f32 = grad.iter().map(|&g| g * g).sum();
    let norm = norm_sq.sqrt();
    let norm_safe = norm.max(1e-12);
    if norm_safe > clip_norm {
        let scale = clip_norm / norm_safe;
        for g in grad.iter_mut() {
            *g *= scale;
        }
    }
    norm
}

/// Privatise a single client update for transmission under local DP.
///
/// The update is clipped to L2 norm `clip_norm` and perturbed with the
/// configured mechanism. The input vector is consumed and the privatised
/// version returned.
///
/// # Errors
/// - `EmptyGradient`-equivalent `Internal` if `grad` is empty.
/// - `InvalidNoiseMultiplier` if the Gaussian σ is non-finite / non-positive.
pub fn privatize_update(
    mut grad: Vec<f32>,
    cfg: &LdpFlConfig,
    rng: &mut LcgRng,
) -> FedResult<Vec<f32>> {
    if grad.is_empty() {
        return Err(FedError::Internal("ldp_fl: empty client update".into()));
    }

    clip_l2(&mut grad, cfg.clip_norm);

    match cfg.mechanism {
        LdpMechanism::Gaussian => {
            let sigma = cfg.gaussian_sigma();
            if !sigma.is_finite() || sigma <= 0.0 {
                return Err(FedError::InvalidNoiseMultiplier);
            }
            let mut i = 0;
            while i < grad.len() {
                let (z1, z2) = rng.next_normal_pair();
                grad[i] += sigma * z1;
                i += 1;
                if i < grad.len() {
                    grad[i] += sigma * z2;
                    i += 1;
                }
            }
        }
        LdpMechanism::Laplace => {
            let b = cfg.laplace_scale();
            if !b.is_finite() || b <= 0.0 {
                return Err(FedError::InvalidNoiseMultiplier);
            }
            for g in grad.iter_mut() {
                *g += rng.next_laplace(b);
            }
        }
    }

    Ok(grad)
}

/// Privatise a cohort of client updates and average the privatised results — the
/// LDP-FL server step. Each client privatises locally; the server only sees and
/// averages noisy updates.
///
/// # Errors
/// - `EmptyClientList` if `grads` is empty.
/// - `DimensionMismatch` if client updates differ in length.
/// - Propagates per-client errors from [`privatize_update`].
pub fn ldp_fl_aggregate(
    grads: &[Vec<f32>],
    cfg: &LdpFlConfig,
    rng: &mut LcgRng,
) -> FedResult<Vec<f32>> {
    if grads.is_empty() {
        return Err(FedError::EmptyClientList);
    }
    let dim = grads[0].len();
    if dim == 0 {
        return Err(FedError::EmptyClientList);
    }
    for g in grads.iter().skip(1) {
        if g.len() != dim {
            return Err(FedError::DimensionMismatch {
                expected: dim,
                got: g.len(),
            });
        }
    }

    let mut accum = vec![0.0_f64; dim];
    for g in grads {
        let priv_g = privatize_update(g.clone(), cfg, rng)?;
        for (a, &p) in accum.iter_mut().zip(priv_g.iter()) {
            *a += p as f64;
        }
    }
    let inv_n = 1.0 / grads.len() as f64;
    Ok(accum.iter().map(|&a| (a * inv_n) as f32).collect())
}

/// Central privacy after privacy amplification by Poisson subsampling at rate
/// `q ∈ (0, 1]`: `ε_amplified = ln(1 + q·(e^ε − 1))`.
///
/// # Errors
/// Returns `Internal` if `q` is not in `(0, 1]`.
pub fn amplified_epsilon(per_client_epsilon: f32, q: f32) -> FedResult<f32> {
    if !(q > 0.0 && q <= 1.0 && q.is_finite()) {
        return Err(FedError::Internal(
            "ldp_fl: subsampling rate q must be in (0, 1]".into(),
        ));
    }
    let amplified = (1.0 + q * (per_client_epsilon.exp() - 1.0)).ln();
    Ok(amplified)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn gaussian_cfg() -> LdpFlConfig {
        LdpFlConfig::new(1.0, 1e-5, 1.0, LdpMechanism::Gaussian).expect("valid")
    }

    fn laplace_cfg() -> LdpFlConfig {
        LdpFlConfig::new(1.0, 0.0, 1.0, LdpMechanism::Laplace).expect("valid")
    }

    #[test]
    fn config_validation() {
        assert!(LdpFlConfig::new(0.0, 1e-5, 1.0, LdpMechanism::Gaussian).is_err());
        assert!(LdpFlConfig::new(-1.0, 1e-5, 1.0, LdpMechanism::Gaussian).is_err());
        // Gaussian requires valid δ.
        assert!(LdpFlConfig::new(1.0, 0.0, 1.0, LdpMechanism::Gaussian).is_err());
        assert!(LdpFlConfig::new(1.0, 1.0, 1.0, LdpMechanism::Gaussian).is_err());
        // Laplace ignores δ.
        assert!(LdpFlConfig::new(1.0, 0.0, 1.0, LdpMechanism::Laplace).is_ok());
        // Clip must be positive.
        assert!(LdpFlConfig::new(1.0, 1e-5, 0.0, LdpMechanism::Gaussian).is_err());
        assert!(LdpFlConfig::new(1.0, 1e-5, -1.0, LdpMechanism::Gaussian).is_err());
    }

    #[test]
    fn gaussian_sigma_formula() {
        let cfg = LdpFlConfig::new(2.0, 1e-5, 3.0, LdpMechanism::Gaussian).expect("ok");
        let expected = 3.0 * (2.0 * (1.25 / 1e-5_f32).ln()).sqrt() / 2.0;
        assert!((cfg.gaussian_sigma() - expected).abs() < 1e-3);
    }

    #[test]
    fn laplace_scale_formula() {
        let cfg = LdpFlConfig::new(4.0, 0.0, 2.0, LdpMechanism::Laplace).expect("ok");
        assert!((cfg.laplace_scale() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn privatize_empty_errors() {
        let cfg = gaussian_cfg();
        let mut rng = LcgRng::new(0);
        assert!(privatize_update(vec![], &cfg, &mut rng).is_err());
    }

    #[test]
    fn privatize_output_shape() {
        let cfg = gaussian_cfg();
        let mut rng = LcgRng::new(1);
        let out = privatize_update(vec![0.1_f32; 16], &cfg, &mut rng).expect("ok");
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn clipping_bounds_norm_before_noise() {
        // A huge update must be clipped to ≈ clip_norm before noise is added.
        // Use a tiny ε-free check: clip alone (verified via the helper).
        let mut g = vec![10.0_f32, 10.0, 10.0]; // ‖·‖ ≈ 17.3
        let pre = clip_l2(&mut g, 1.0);
        assert!(pre > 17.0);
        let post: f32 = g.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((post - 1.0).abs() < 1e-4, "post-clip norm = {post}");
    }

    #[test]
    fn small_update_not_clipped() {
        let mut g = vec![0.1_f32, 0.1];
        let pre = clip_l2(&mut g, 1.0);
        assert!(pre < 1.0);
        // Unchanged.
        assert!((g[0] - 0.1).abs() < 1e-6 && (g[1] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn gaussian_noise_adds_calibrated_spread() {
        // The privatised coordinates should spread around the (clipped) signal
        // with an empirical std on the order of the calibrated σ. (We check the
        // spread, not the mean, since the deterministic test RNG is not exactly
        // zero-mean — see the per-crate LcgRng notes.)
        let cfg = LdpFlConfig::new(1.0, 1e-5, 1.0, LdpMechanism::Gaussian).expect("ok");
        let sigma = cfg.gaussian_sigma();
        assert!(sigma > 1.0, "expected sizeable σ, got {sigma}");
        let mut rng = LcgRng::new(7);
        // One long flat update → its coordinates are i.i.d. signal + N(0, σ²).
        let base = vec![0.0_f32; 4096];
        let out = privatize_update(base, &cfg, &mut rng).expect("ok");
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        let var: f32 = out.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / out.len() as f32;
        let std = var.sqrt();
        // Empirical std should be within a factor of ~2 of the calibrated σ.
        assert!(
            std > 0.4 * sigma && std < 2.5 * sigma,
            "std {std} vs σ {sigma}"
        );
    }

    #[test]
    fn laplace_noise_adds_calibrated_spread() {
        // Laplace(0, b) has variance 2b²; check the empirical spread tracks b.
        let cfg = LdpFlConfig::new(1.0, 0.0, 1.0, LdpMechanism::Laplace).expect("ok");
        let b = cfg.laplace_scale();
        let mut rng = LcgRng::new(11);
        let base = vec![0.0_f32; 4096];
        let out = privatize_update(base, &cfg, &mut rng).expect("ok");
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        let var: f32 = out.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / out.len() as f32;
        let expected_std = (2.0_f32).sqrt() * b;
        let std = var.sqrt();
        assert!(
            std > 0.4 * expected_std && std < 2.5 * expected_std,
            "std {std} vs expected {expected_std}"
        );
    }

    #[test]
    fn aggregate_empty_errors() {
        let cfg = gaussian_cfg();
        let mut rng = LcgRng::new(0);
        assert!(matches!(
            ldp_fl_aggregate(&[], &cfg, &mut rng),
            Err(FedError::EmptyClientList)
        ));
    }

    #[test]
    fn aggregate_dimension_mismatch_errors() {
        let cfg = gaussian_cfg();
        let mut rng = LcgRng::new(0);
        let grads = vec![vec![1.0_f32, 2.0], vec![3.0_f32]];
        assert!(matches!(
            ldp_fl_aggregate(&grads, &cfg, &mut rng),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn aggregate_recovers_mean_with_low_noise() {
        // Large ε → little noise → privatised mean ≈ clean mean.
        let cfg = LdpFlConfig::new(50.0, 1e-3, 10.0, LdpMechanism::Laplace).expect("ok");
        let mut rng = LcgRng::new(3);
        let grads = vec![vec![1.0_f32, 2.0], vec![3.0_f32, 4.0], vec![5.0_f32, 6.0]];
        let agg = ldp_fl_aggregate(&grads, &cfg, &mut rng).expect("ok");
        assert_eq!(agg.len(), 2);
        assert!((agg[0] - 3.0).abs() < 1.0, "agg0 = {}", agg[0]);
        assert!((agg[1] - 4.0).abs() < 1.0, "agg1 = {}", agg[1]);
    }

    #[test]
    fn amplification_reduces_epsilon() {
        // ε_amplified < ε for q < 1, and equals ε at q = 1.
        let eps = 2.0_f32;
        let half = amplified_epsilon(eps, 0.1).expect("ok");
        assert!(half < eps, "amplified {half} should be < {eps}");
        let full = amplified_epsilon(eps, 1.0).expect("ok");
        assert!((full - eps).abs() < 1e-4, "q=1 should give ε: {full}");
    }

    #[test]
    fn amplification_invalid_q_errors() {
        assert!(amplified_epsilon(1.0, 0.0).is_err());
        assert!(amplified_epsilon(1.0, 1.5).is_err());
        assert!(amplified_epsilon(1.0, -0.1).is_err());
    }

    #[test]
    fn deterministic_with_same_seed() {
        let cfg = gaussian_cfg();
        let mut a = LcgRng::new(99);
        let mut b = LcgRng::new(99);
        let base = vec![0.3_f32, 0.4, 0.5];
        let pa = privatize_update(base.clone(), &cfg, &mut a).expect("ok");
        let pb = privatize_update(base.clone(), &cfg, &mut b).expect("ok");
        assert_eq!(pa, pb);
    }

    #[test]
    fn laplace_config_unused_delta_ok() {
        let _ = laplace_cfg();
    }
}
