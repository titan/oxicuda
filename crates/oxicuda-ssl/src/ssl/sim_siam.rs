//! SimSiam struct — owns projector + predictor weights.
//!
//! This is a struct-based incarnation of SimSiam (Chen & He 2021) that manages
//! its own weight matrices in-process, unlike the functional API in
//! [`crate::non_contrastive::simsiam`] which requires the caller to provide
//! pre-computed projections.
//!
//! ## Architecture
//! ```text
//! Projector:  d_encoder → (Linear + ReLU) → d_projector → Linear → d_out → L2-norm
//! Predictor:  d_out     → (Linear + ReLU) → d_predictor → Linear → d_out → L2-norm
//! ```
//!
//! ## Loss
//! ```text
//! p1 = predict(project(z1)),   p2 = predict(project(z2))
//! z1_p = project(z1),          z2_p = project(z2)
//! L = (D(p1, sg(z2_p)) + D(p2, sg(z1_p))) / 2
//! D(a, b) = -(a · b)     [both are already L2-normalised]
//! ```

use crate::error::{SslError, SslResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Hyper-parameters for the struct-based [`SimSiam`] model.
#[derive(Debug, Clone)]
pub struct SimSiamConfig {
    /// Backbone output dimension (input to projector).
    pub d_encoder: usize,
    /// Projector hidden dimension.
    pub d_projector: usize,
    /// Predictor hidden dimension.
    pub d_predictor: usize,
    /// Output dimension (projector output = predictor I/O).
    pub d_out: usize,
}

impl Default for SimSiamConfig {
    fn default() -> Self {
        Self {
            d_encoder: 64,
            d_projector: 128,
            d_predictor: 64,
            d_out: 32,
        }
    }
}

// ─── SimSiam model ───────────────────────────────────────────────────────────

/// Struct-based SimSiam model that owns its projector and predictor weights.
///
/// All weight matrices use Kaiming (He) initialisation with `scale = sqrt(2 / fan_in)`.
#[derive(Debug, Clone)]
pub struct SimSiam {
    /// Projector first layer weights `[d_projector × d_encoder]`.
    proj_w1: Vec<f32>,
    /// Projector first layer bias `[d_projector]`.
    proj_b1: Vec<f32>,
    /// Projector second layer weights `[d_out × d_projector]`.
    proj_w2: Vec<f32>,
    /// Projector second layer bias `[d_out]`.
    proj_b2: Vec<f32>,
    /// Predictor first layer weights `[d_predictor × d_out]`.
    pred_w1: Vec<f32>,
    /// Predictor first layer bias `[d_predictor]`.
    pred_b1: Vec<f32>,
    /// Predictor second layer weights `[d_out × d_predictor]`.
    pred_w2: Vec<f32>,
    /// Predictor second layer bias `[d_out]`.
    pred_b2: Vec<f32>,
    /// Configuration used to construct this model.
    config: SimSiamConfig,
}

impl SimSiam {
    /// Create a new `SimSiam` model with Kaiming-initialised weights.
    ///
    /// # Errors
    /// [`SslError::InvalidParameter`] when any dimension in `config` is zero.
    pub fn new(config: SimSiamConfig, rng: &mut LcgRng) -> SslResult<Self> {
        if config.d_encoder == 0 {
            return Err(SslError::InvalidParameter {
                name: "d_encoder".into(),
                reason: "must be > 0".into(),
            });
        }
        if config.d_projector == 0 {
            return Err(SslError::InvalidParameter {
                name: "d_projector".into(),
                reason: "must be > 0".into(),
            });
        }
        if config.d_predictor == 0 {
            return Err(SslError::InvalidParameter {
                name: "d_predictor".into(),
                reason: "must be > 0".into(),
            });
        }
        if config.d_out == 0 {
            return Err(SslError::InvalidParameter {
                name: "d_out".into(),
                reason: "must be > 0".into(),
            });
        }

        let proj_w1 = kaiming_init(config.d_projector, config.d_encoder, rng);
        let proj_b1 = vec![0.0_f32; config.d_projector];
        let proj_w2 = kaiming_init(config.d_out, config.d_projector, rng);
        let proj_b2 = vec![0.0_f32; config.d_out];

        let pred_w1 = kaiming_init(config.d_predictor, config.d_out, rng);
        let pred_b1 = vec![0.0_f32; config.d_predictor];
        let pred_w2 = kaiming_init(config.d_out, config.d_predictor, rng);
        let pred_b2 = vec![0.0_f32; config.d_out];

        Ok(Self {
            proj_w1,
            proj_b1,
            proj_w2,
            proj_b2,
            pred_w1,
            pred_b1,
            pred_w2,
            pred_b2,
            config,
        })
    }

    /// Project a single encoder output vector.
    ///
    /// Computes `z = L2_norm(proj_w2 · ReLU(proj_w1 · x + proj_b1) + proj_b2)`.
    ///
    /// # Arguments
    /// * `z` — encoder output `[d_encoder]`.
    ///
    /// # Errors
    /// [`SslError::DimensionMismatch`] when `z.len() != d_encoder`.
    pub fn project(&self, z: &[f32]) -> SslResult<Vec<f32>> {
        let d = self.config.d_encoder;
        if z.len() != d {
            return Err(SslError::DimensionMismatch {
                expected: d,
                got: z.len(),
            });
        }
        let hidden = linear_relu(&self.proj_w1, &self.proj_b1, z, d, self.config.d_projector);
        let out = linear(
            &self.proj_w2,
            &self.proj_b2,
            &hidden,
            self.config.d_projector,
            self.config.d_out,
        );
        Ok(l2_normalize(out))
    }

    /// Apply the predictor to a projected representation.
    ///
    /// Computes `p = L2_norm(pred_w2 · ReLU(pred_w1 · proj + pred_b1) + pred_b2)`.
    ///
    /// # Arguments
    /// * `p` — projected representation `[d_out]`.
    ///
    /// # Errors
    /// [`SslError::DimensionMismatch`] when `p.len() != d_out`.
    pub fn predict(&self, p: &[f32]) -> SslResult<Vec<f32>> {
        let d = self.config.d_out;
        if p.len() != d {
            return Err(SslError::DimensionMismatch {
                expected: d,
                got: p.len(),
            });
        }
        let hidden = linear_relu(&self.pred_w1, &self.pred_b1, p, d, self.config.d_predictor);
        let out = linear(
            &self.pred_w2,
            &self.pred_b2,
            &hidden,
            self.config.d_predictor,
            self.config.d_out,
        );
        Ok(l2_normalize(out))
    }

    /// Compute the symmetric SimSiam loss for two encoder outputs.
    ///
    /// Implements `L = (D(p1, sg(z2_p)) + D(p2, sg(z1_p))) / 2` where
    /// `D(a, b) = -(a · b)` for unit-norm vectors and `sg` denotes stop-gradient
    /// (a no-op in this pure-Rust implementation since there is no autograd engine).
    ///
    /// # Arguments
    /// * `z1` — encoder output from view 1 `[d_encoder]`.
    /// * `z2` — encoder output from view 2 `[d_encoder]`.
    ///
    /// # Errors
    /// Propagates dimension mismatch errors from [`Self::project`] and [`Self::predict`].
    pub fn loss(&self, z1: &[f32], z2: &[f32]) -> SslResult<f32> {
        let z1_proj = self.project(z1)?;
        let z2_proj = self.project(z2)?;
        let p1 = self.predict(&z1_proj)?;
        let p2 = self.predict(&z2_proj)?;

        // D(p, z_sg) = -(p · z_sg); both are L2-normalised → cosine distance
        let d1 = neg_dot(&p1, &z2_proj);
        let d2 = neg_dot(&p2, &z1_proj);
        Ok((d1 + d2) * 0.5)
    }

    /// Return the output dimension of the projector (= predictor I/O dim).
    #[inline]
    #[must_use]
    pub fn d_out(&self) -> usize {
        self.config.d_out
    }

    /// Overwrite the predictor with an exact direction-preserving identity map.
    ///
    /// The predictor MLP `L2(W2 · ReLU(W1·p + b1) + b2)` is normally a learned,
    /// randomly-initialised non-linear transform, so for a *random* predictor the
    /// SimSiam loss of two identical views is some arbitrary value in `[-1, 1]`.
    /// SimSiam's negative-cosine loss only attains its minimum of `-1` for
    /// identical views when the predictor preserves the projection's direction.
    ///
    /// This installs such an identity predictor. The ReLU non-linearity is bridged
    /// with the standard positive/negative split: the hidden layer computes
    /// `[ReLU(p), ReLU(-p)]` and the output layer reconstructs `p = p⁺ - p⁻`,
    /// reproducing the input exactly. The trailing L2-norm then leaves an
    /// already unit-norm projection unchanged, so `predict(project(z)) == project(z)`
    /// and `loss(z, z) == -1`.
    ///
    /// This requires the predictor hidden dimension to be exactly twice the output
    /// dimension so the two halves can hold the positive and negative parts.
    ///
    /// # Errors
    /// [`SslError::InvalidParameter`] when `d_predictor != 2 * d_out`.
    pub fn set_identity_predictor(&mut self) -> SslResult<()> {
        let d_out = self.config.d_out;
        let d_pred = self.config.d_predictor;
        if d_pred != 2 * d_out {
            return Err(SslError::InvalidParameter {
                name: "d_predictor".into(),
                reason: "identity predictor requires d_predictor == 2 * d_out".into(),
            });
        }

        // W1: `[d_predictor × d_out]`. Row i selects +p_i, row (d_out + i) selects -p_i.
        let mut pred_w1 = vec![0.0_f32; d_pred * d_out];
        for i in 0..d_out {
            pred_w1[i * d_out + i] = 1.0;
            pred_w1[(d_out + i) * d_out + i] = -1.0;
        }
        // W2: `[d_out × d_predictor]`. Output i = p⁺_i - p⁻_i.
        let mut pred_w2 = vec![0.0_f32; d_out * d_pred];
        for i in 0..d_out {
            pred_w2[i * d_pred + i] = 1.0;
            pred_w2[i * d_pred + (d_out + i)] = -1.0;
        }

        self.pred_w1 = pred_w1;
        self.pred_b1 = vec![0.0_f32; d_pred];
        self.pred_w2 = pred_w2;
        self.pred_b2 = vec![0.0_f32; d_out];
        Ok(())
    }
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Allocate a `[out_dim × in_dim]` weight matrix with Kaiming normal init.
///
/// `scale = sqrt(2 / in_dim)` (He initialisation for ReLU activations).
fn kaiming_init(out_dim: usize, in_dim: usize, rng: &mut LcgRng) -> Vec<f32> {
    let scale = (2.0_f32 / in_dim as f32).sqrt();
    let mut w = vec![0.0_f32; out_dim * in_dim];
    rng.fill_normal(&mut w);
    for v in w.iter_mut() {
        *v *= scale;
    }
    w
}

/// Standard matrix-vector multiply: `out[i] = b[i] + Σ_j w[i·in_dim + j] * x[j]`.
///
/// No activation is applied; shape is `[out_dim × in_dim] × [in_dim] = [out_dim]`.
fn linear(w: &[f32], b: &[f32], x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; out_dim];
    for i in 0..out_dim {
        let mut acc = b[i];
        let row_start = i * in_dim;
        for j in 0..in_dim {
            acc += w[row_start + j] * x[j];
        }
        out[i] = acc;
    }
    out
}

/// `linear` followed by element-wise ReLU.
fn linear_relu(w: &[f32], b: &[f32], x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = linear(w, b, x, in_dim, out_dim);
    for v in out.iter_mut() {
        *v = v.max(0.0);
    }
    out
}

/// L2-normalise a vector in-place, returning it.
///
/// A floor of `1e-12` on the norm prevents division by zero.
fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= norm;
    }
    v
}

/// Compute the negative dot product `-(a · b)` for two same-length slices.
fn neg_dot(a: &[f32], b: &[f32]) -> f32 {
    -a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| ai * bi)
        .sum::<f32>()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_simsiam(seed: u64) -> SimSiam {
        let mut rng = LcgRng::new(seed);
        SimSiam::new(
            SimSiamConfig {
                d_encoder: 16,
                d_projector: 32,
                d_predictor: 16,
                d_out: 8,
            },
            &mut rng,
        )
        .expect("value should be present")
    }

    fn random_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut v = vec![0.0_f32; n];
        rng.fill_normal(&mut v);
        v
    }

    #[test]
    fn project_shape() {
        let ss = make_simsiam(1);
        let z = random_vec(16, 2);
        let out = ss.project(&z).expect("project should succeed");
        assert_eq!(out.len(), 8, "project output must have len == d_out");
    }

    #[test]
    fn predict_shape() {
        let ss = make_simsiam(3);
        let p = random_vec(8, 4);
        let out = ss.predict(&p).expect("predict should succeed");
        assert_eq!(out.len(), 8, "predict output must have len == d_out");
    }

    #[test]
    fn loss_finite() {
        let ss = make_simsiam(5);
        let z1 = random_vec(16, 6);
        let z2 = random_vec(16, 7);
        let l = ss.loss(&z1, &z2).expect("loss should succeed");
        assert!(l.is_finite(), "loss must be finite, got {l}");
    }

    #[test]
    fn loss_in_range() {
        let ss = make_simsiam(8);
        let z1 = random_vec(16, 9);
        let z2 = random_vec(16, 10);
        let l = ss.loss(&z1, &z2).expect("loss should succeed");
        assert!(
            (-1.0 - 1e-5..=1.0 + 1e-5).contains(&l),
            "loss={l} must be in [-1, 1]"
        );
    }

    #[test]
    fn loss_symmetric() {
        let ss = make_simsiam(11);
        let z1 = random_vec(16, 12);
        let z2 = random_vec(16, 13);
        let l12 = ss.loss(&z1, &z2).expect("loss should succeed");
        let l21 = ss.loss(&z2, &z1).expect("loss should succeed");
        assert!(
            (l12 - l21).abs() < 1e-5,
            "loss(z1,z2)={l12} != loss(z2,z1)={l21}"
        );
    }

    #[test]
    fn different_views_different_projections() {
        let ss = make_simsiam(14);
        let z1 = random_vec(16, 15);
        let z2 = random_vec(16, 16);
        let p1 = ss.project(&z1).expect("project should succeed");
        let p2 = ss.project(&z2).expect("project should succeed");
        let diff: f32 = p1.iter().zip(p2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-6,
            "projections of different inputs must differ, diff={diff}"
        );
    }

    #[test]
    fn identical_views_low_loss() {
        // For identical views z1 == z2 we have z1_p == z2_p and p1 == p2, so the
        // symmetric loss collapses to -dot(p, z_p). This reaches its minimum of -1
        // only when the predictor preserves the projection's direction. With a
        // *random* predictor MLP, dot(p, z_p) is some arbitrary cosine in [-1, 1],
        // so the loss is not necessarily near -1. Installing a direction-preserving
        // identity predictor makes predict(project(z)) == project(z) exactly, which
        // drives the identical-view loss to its -1 floor.
        let mut ss = make_simsiam(17);
        ss.set_identity_predictor()
            .expect("config has d_predictor == 2 * d_out");
        let z = random_vec(16, 18);
        let l = ss.loss(&z, &z).expect("loss should succeed");
        assert!(
            (l - (-1.0)).abs() < 1e-5,
            "with a direction-preserving predictor, loss for identical views must be -1, got {l}"
        );
    }

    #[test]
    fn identity_predictor_is_direction_preserving() {
        // The identity predictor must reproduce its (unit-norm) input exactly,
        // so predict(project(z)) == project(z) for every z.
        let mut ss = make_simsiam(27);
        ss.set_identity_predictor()
            .expect("config has d_predictor == 2 * d_out");
        for seed in 0..6_u64 {
            let z = random_vec(16, seed + 200);
            let zp = ss.project(&z).expect("project should succeed");
            let p = ss.predict(&zp).expect("predict should succeed");
            let max_diff = zp
                .iter()
                .zip(p.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_diff < 1e-5,
                "identity predictor must reproduce input, max|p-zp|={max_diff} (seed={seed})"
            );
        }
    }

    #[test]
    fn set_identity_predictor_requires_double_hidden() {
        // d_predictor (8) != 2 * d_out (16) → must error, never panic.
        let mut rng = LcgRng::new(28);
        let mut ss = SimSiam::new(
            SimSiamConfig {
                d_encoder: 16,
                d_projector: 32,
                d_predictor: 8,
                d_out: 8,
            },
            &mut rng,
        )
        .expect("value should be present");
        assert!(
            ss.set_identity_predictor().is_err(),
            "identity predictor with d_predictor != 2*d_out must return Err"
        );
    }

    #[test]
    fn d_out_0_error() {
        let mut rng = LcgRng::new(19);
        let result = SimSiam::new(
            SimSiamConfig {
                d_encoder: 8,
                d_projector: 16,
                d_predictor: 8,
                d_out: 0,
            },
            &mut rng,
        );
        assert!(result.is_err(), "d_out=0 must return Err");
    }

    #[test]
    fn project_output_normalized() {
        let ss = make_simsiam(20);
        let z = random_vec(16, 21);
        let out = ss.project(&z).expect("project should succeed");
        let norm: f32 = out.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "project output must be unit-norm, norm={norm}"
        );
    }

    #[test]
    fn loss_stop_grad_invariant() {
        // Verify loss is finite for various arbitrary inputs — stop-grad is a
        // no-op in pure Rust and must not cause numerical issues.
        let ss = make_simsiam(22);
        for seed in 0..8_u64 {
            let z1 = random_vec(16, seed * 2 + 100);
            let z2 = random_vec(16, seed * 2 + 101);
            let l = ss.loss(&z1, &z2).expect("loss should succeed");
            assert!(
                l.is_finite(),
                "loss must be finite for seed={seed}, got {l}"
            );
        }
    }

    #[test]
    fn d_encoder_0_error() {
        let mut rng = LcgRng::new(23);
        assert!(
            SimSiam::new(
                SimSiamConfig {
                    d_encoder: 0,
                    d_projector: 16,
                    d_predictor: 8,
                    d_out: 8
                },
                &mut rng
            )
            .is_err()
        );
    }

    #[test]
    fn predict_output_normalized() {
        let ss = make_simsiam(24);
        let p = random_vec(8, 25);
        let out = ss.predict(&p).expect("predict should succeed");
        let norm: f32 = out.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "predict output must be unit-norm, norm={norm}"
        );
    }

    #[test]
    fn d_out_accessor() {
        let ss = make_simsiam(26);
        assert_eq!(ss.d_out(), 8);
    }
}
