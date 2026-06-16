//! MOON: Model-Contrastive Federated Learning.
//!
//! Li, He & Song, "Model-Contrastive Federated Learning", CVPR 2021.
//!
//! MOON addresses client drift on non-IID data by adding a *model-level*
//! contrastive term to each client's local objective. Let `z` be the
//! representation (penultimate-layer feature) produced by the **current** local
//! model, `z_glob` the representation of the received **global** model, and
//! `z_prev` the representation of the client's **previous** local model — all on
//! the same input. MOON pulls `z` toward `z_glob` (positive pair) and pushes it
//! away from `z_prev` (negative pair) with an InfoNCE / NT-Xent loss:
//!
//! `ℓ_con = −log [ exp(sim(z, z_glob)/τ) / ( exp(sim(z, z_glob)/τ) + exp(sim(z, z_prev)/τ) ) ]`
//!
//! where `sim(a, b) = aᵀb / (‖a‖·‖b‖)` is cosine similarity and `τ` a
//! temperature. The total client loss is `ℓ = ℓ_sup + μ·ℓ_con`.
//!
//! This module implements the contrastive loss and its gradient with respect to
//! the current representation `z` (the quantity back-propagated into the
//! encoder), which is the self-contained, framework-agnostic core of MOON.

use crate::error::{FedError, FedResult};

/// Configuration for the MOON model-contrastive term.
#[derive(Debug, Clone, Copy)]
pub struct MoonConfig {
    /// Weight `μ ≥ 0` of the contrastive term in the total client loss.
    pub mu: f32,
    /// Temperature `τ > 0` of the contrastive softmax.
    pub temperature: f32,
}

impl MoonConfig {
    /// Construct and validate a configuration.
    ///
    /// # Errors
    /// Returns `InvalidProximalMu` if `mu` is negative / non-finite, or
    /// `Internal` if `temperature ≤ 0` / non-finite.
    pub fn new(mu: f32, temperature: f32) -> FedResult<Self> {
        if !(mu >= 0.0 && mu.is_finite()) {
            return Err(FedError::InvalidProximalMu);
        }
        if !(temperature > 0.0 && temperature.is_finite()) {
            return Err(FedError::Internal(
                "moon: temperature must be finite and > 0".into(),
            ));
        }
        Ok(Self { mu, temperature })
    }
}

impl Default for MoonConfig {
    fn default() -> Self {
        Self {
            mu: 1.0,
            temperature: 0.5,
        }
    }
}

const EPS: f32 = 1e-12;

/// L2 norm of a vector (in f32, with a small floor to avoid division by zero).
fn norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt().max(EPS)
}

/// Dot product of two equal-length vectors.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Cosine similarity `aᵀb / (‖a‖‖b‖)`.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b) / (norm(a) * norm(b))
}

/// Validate that the three representations share the same non-zero length.
fn validate(z: &[f32], z_glob: &[f32], z_prev: &[f32]) -> FedResult<()> {
    if z.is_empty() {
        return Err(FedError::EmptyClientList);
    }
    if z_glob.len() != z.len() {
        return Err(FedError::DimensionMismatch {
            expected: z.len(),
            got: z_glob.len(),
        });
    }
    if z_prev.len() != z.len() {
        return Err(FedError::DimensionMismatch {
            expected: z.len(),
            got: z_prev.len(),
        });
    }
    Ok(())
}

/// Compute the MOON model-contrastive loss `ℓ_con` for one sample.
///
/// `z` is the current local representation, `z_glob` the global-model
/// representation (positive), `z_prev` the previous-local-model representation
/// (negative). Returns a non-negative scalar (before scaling by `μ`).
///
/// # Errors
/// - `EmptyClientList` if `z` is empty.
/// - `DimensionMismatch` if the three vectors differ in length.
pub fn moon_contrastive_loss(
    z: &[f32],
    z_glob: &[f32],
    z_prev: &[f32],
    cfg: &MoonConfig,
) -> FedResult<f32> {
    validate(z, z_glob, z_prev)?;
    let pos = cosine(z, z_glob) / cfg.temperature;
    let neg = cosine(z, z_prev) / cfg.temperature;

    // ℓ = −log( e^pos / (e^pos + e^neg) ) = log(1 + e^{neg − pos}), computed in a
    // numerically-stable form via softplus.
    let diff = neg - pos;
    let loss = softplus(diff);
    Ok(loss)
}

/// Numerically-stable `log(1 + e^x)`.
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        x.max(0.0) + (-x.abs()).exp().ln_1p()
    }
}

/// Total MOON client loss `ℓ_sup + μ·ℓ_con` for one sample.
///
/// # Errors
/// Propagates errors from [`moon_contrastive_loss`].
pub fn moon_total_loss(
    supervised_loss: f32,
    z: &[f32],
    z_glob: &[f32],
    z_prev: &[f32],
    cfg: &MoonConfig,
) -> FedResult<f32> {
    let con = moon_contrastive_loss(z, z_glob, z_prev, cfg)?;
    Ok(supervised_loss + cfg.mu * con)
}

/// Gradient of `μ·ℓ_con` with respect to the current representation `z`.
///
/// Returns `∂(μ·ℓ_con)/∂z`, the contribution back-propagated into the encoder.
/// Using `σ = sigmoid(neg − pos)`:
///
/// `∂ℓ_con/∂z = (σ/τ) · ( ∂sim(z,z_prev)/∂z − ∂sim(z,z_glob)/∂z )`,
///
/// and for cosine similarity
/// `∂sim(z,u)/∂z = u/(‖z‖‖u‖) − (zᵀu)·z / (‖z‖³‖u‖)`.
///
/// # Errors
/// - `EmptyClientList` if `z` is empty.
/// - `DimensionMismatch` if the three vectors differ in length.
pub fn moon_contrastive_grad(
    z: &[f32],
    z_glob: &[f32],
    z_prev: &[f32],
    cfg: &MoonConfig,
) -> FedResult<Vec<f32>> {
    validate(z, z_glob, z_prev)?;
    let nz = norm(z);
    let pos = cosine(z, z_glob) / cfg.temperature;
    let neg = cosine(z, z_prev) / cfg.temperature;
    // σ = ∂ softplus(neg − pos) / ∂(neg − pos) = sigmoid(neg − pos).
    let sigma = sigmoid(neg - pos);

    // Per-coordinate cosine gradient ∂sim(z,u)/∂z.
    let grad_sim = |u: &[f32], out: &mut [f32]| {
        let nu = norm(u);
        let zu = dot(z, u);
        let inv = 1.0 / (nz * nu);
        let coef = zu / (nz * nz * nz * nu);
        for (o, (&zj, &uj)) in out.iter_mut().zip(z.iter().zip(u.iter())) {
            *o = uj * inv - zj * coef;
        }
    };

    let mut g_glob = vec![0.0_f32; z.len()];
    let mut g_prev = vec![0.0_f32; z.len()];
    grad_sim(z_glob, &mut g_glob);
    grad_sim(z_prev, &mut g_prev);

    // ∂(μ ℓ_con)/∂z = μ·(σ/τ)·(grad_prev − grad_glob).
    let scale = cfg.mu * sigma / cfg.temperature;
    let grad = g_prev
        .iter()
        .zip(g_glob.iter())
        .map(|(&gp, &gg)| scale * (gp - gg))
        .collect();
    Ok(grad)
}

/// Numerically-stable logistic sigmoid.
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> MoonConfig {
        MoonConfig::new(1.0, 0.5).expect("valid config")
    }

    #[test]
    fn config_validation() {
        assert!(MoonConfig::new(-1.0, 0.5).is_err());
        assert!(MoonConfig::new(1.0, 0.0).is_err());
        assert!(MoonConfig::new(1.0, -0.5).is_err());
        assert!(MoonConfig::new(0.0, 0.5).is_ok());
    }

    #[test]
    fn loss_empty_errors() {
        let c = cfg();
        assert!(moon_contrastive_loss(&[], &[], &[], &c).is_err());
    }

    #[test]
    fn loss_dimension_mismatch() {
        let c = cfg();
        assert!(matches!(
            moon_contrastive_loss(&[1.0, 2.0], &[1.0], &[1.0, 2.0], &c),
            Err(FedError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            moon_contrastive_loss(&[1.0, 2.0], &[1.0, 2.0], &[1.0], &c),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn loss_is_non_negative() {
        let c = cfg();
        let z = vec![0.3_f32, -0.5, 0.8];
        let zg = vec![0.1_f32, 0.2, 0.9];
        let zp = vec![-0.4_f32, 0.7, 0.1];
        let l = moon_contrastive_loss(&z, &zg, &zp, &c).expect("ok");
        assert!(l >= 0.0, "loss = {l}");
        assert!(l.is_finite());
    }

    #[test]
    fn loss_small_when_aligned_with_global() {
        // z identical to global, opposite to prev → positive sim 1, negative −1.
        let c = cfg();
        let z = vec![1.0_f32, 0.0];
        let zg = vec![1.0_f32, 0.0];
        let zp = vec![-1.0_f32, 0.0];
        let aligned = moon_contrastive_loss(&z, &zg, &zp, &c).expect("ok");

        // z aligned with prev instead → much larger loss.
        let zg2 = vec![-1.0_f32, 0.0];
        let zp2 = vec![1.0_f32, 0.0];
        let misaligned = moon_contrastive_loss(&z, &zg2, &zp2, &c).expect("ok");
        assert!(
            aligned < misaligned,
            "aligned {aligned} should be < misaligned {misaligned}"
        );
    }

    #[test]
    fn total_loss_adds_scaled_contrastive() {
        let c = MoonConfig::new(2.0, 0.5).expect("ok");
        let z = vec![0.2_f32, 0.9];
        let zg = vec![0.3_f32, 0.8];
        let zp = vec![0.9_f32, -0.1];
        let con = moon_contrastive_loss(&z, &zg, &zp, &c).expect("ok");
        let total = moon_total_loss(1.5, &z, &zg, &zp, &c).expect("ok");
        assert!((total - (1.5 + 2.0 * con)).abs() < 1e-5, "total = {total}");
    }

    #[test]
    fn mu_zero_gives_zero_contrastive_contribution() {
        let c = MoonConfig::new(0.0, 0.5).expect("ok");
        let z = vec![0.2_f32, 0.9];
        let zg = vec![0.3_f32, 0.8];
        let zp = vec![0.9_f32, -0.1];
        let total = moon_total_loss(3.0, &z, &zg, &zp, &c).expect("ok");
        assert!((total - 3.0).abs() < 1e-6, "total = {total}");
        let grad = moon_contrastive_grad(&z, &zg, &zp, &c).expect("ok");
        for &g in &grad {
            assert!(g.abs() < 1e-6, "grad should vanish for μ=0: {g}");
        }
    }

    #[test]
    fn grad_shape_matches_input() {
        let c = cfg();
        let z = vec![0.1_f32, 0.2, 0.3, 0.4];
        let zg = vec![0.5_f32, 0.1, 0.2, 0.0];
        let zp = vec![0.0_f32, 0.3, 0.1, 0.6];
        let grad = moon_contrastive_grad(&z, &zg, &zp, &c).expect("ok");
        assert_eq!(grad.len(), z.len());
        for &g in &grad {
            assert!(g.is_finite());
        }
    }

    #[test]
    fn grad_matches_finite_difference() {
        // Numerically validate the analytic gradient against central differences.
        let c = MoonConfig::new(1.0, 0.5).expect("ok");
        let z = vec![0.3_f32, -0.6, 0.2];
        let zg = vec![0.1_f32, 0.4, 0.7];
        let zp = vec![-0.5_f32, 0.2, 0.3];
        let grad = moon_contrastive_grad(&z, &zg, &zp, &c).expect("ok");

        let h = 1e-3_f32;
        for j in 0..z.len() {
            let mut zp_plus = z.clone();
            let mut zp_minus = z.clone();
            zp_plus[j] += h;
            zp_minus[j] -= h;
            let lp = c.mu * moon_contrastive_loss(&zp_plus, &zg, &zp, &c).expect("ok");
            let lm = c.mu * moon_contrastive_loss(&zp_minus, &zg, &zp, &c).expect("ok");
            let fd = (lp - lm) / (2.0 * h);
            assert!(
                (grad[j] - fd).abs() < 5e-2,
                "coord {j}: analytic {} vs fd {}",
                grad[j],
                fd
            );
        }
    }

    #[test]
    fn grad_empty_and_mismatch_error() {
        let c = cfg();
        assert!(moon_contrastive_grad(&[], &[], &[], &c).is_err());
        assert!(matches!(
            moon_contrastive_grad(&[1.0, 2.0], &[1.0, 2.0], &[1.0], &c),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn deterministic_repeatable() {
        let c = cfg();
        let z = vec![0.4_f32, 0.5, 0.6];
        let zg = vec![0.1_f32, 0.0, 0.9];
        let zp = vec![0.7_f32, 0.2, 0.1];
        let a = moon_contrastive_loss(&z, &zg, &zp, &c).expect("ok");
        let b = moon_contrastive_loss(&z, &zg, &zp, &c).expect("ok");
        assert_eq!(a, b);
        let ga = moon_contrastive_grad(&z, &zg, &zp, &c).expect("ok");
        let gb = moon_contrastive_grad(&z, &zg, &zp, &c).expect("ok");
        assert_eq!(ga, gb);
    }

    #[test]
    fn temperature_affects_loss_magnitude() {
        let z = vec![1.0_f32, 0.0];
        let zg = vec![0.8_f32, 0.6];
        let zp = vec![0.6_f32, -0.8];
        let hot = MoonConfig::new(1.0, 2.0).expect("ok");
        let cold = MoonConfig::new(1.0, 0.1).expect("ok");
        let l_hot = moon_contrastive_loss(&z, &zg, &zp, &hot).expect("ok");
        let l_cold = moon_contrastive_loss(&z, &zg, &zp, &cold).expect("ok");
        // Different temperatures produce different (finite) losses.
        assert!(l_hot.is_finite() && l_cold.is_finite());
        assert!((l_hot - l_cold).abs() > 1e-4, "hot={l_hot}, cold={l_cold}");
    }
}
