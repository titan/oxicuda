//! Deep & Cross Network (DCN, Wang 2017) and DCN-V2 (Wang 2021).
//!
//! DCN augments a deep MLP tower with an explicit feature-crossing tower. Each
//! cross layer applies, for an input column vector `x_l` and the original input
//! `x_0`:
//!
//! - **DCN (vector form, Wang 2017)**:
//!   `x_{l+1} = x_0 ⊙ (x_l · w_l) + b_l + x_l`
//!   where `w_l ∈ ℝ^d` and the scalar `x_l · w_l` modulates `x_0`.
//! - **DCN-V2 (matrix form, Wang 2021)**:
//!   `x_{l+1} = x_0 ⊙ (W_l x_l + b_l) + x_l`
//!   where `W_l ∈ ℝ^{d×d}` is a full cross matrix giving richer interactions.
//!
//! The cross tower's final representation is concatenated with the deep MLP tower
//! (parallel structure) and projected to a logit, then passed through a sigmoid.

use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

/// Cross-layer parametrisation: classic vector form or DCN-V2 matrix form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossKind {
    /// Wang 2017 vector cross: weight per cross layer is a length-`d` vector.
    Vector,
    /// Wang 2021 matrix cross: weight per cross layer is a `d × d` matrix.
    MatrixV2,
}

/// Configuration for a Deep & Cross Network.
#[derive(Debug, Clone)]
pub struct DcnConfig {
    /// Dimensionality of the (dense) input feature vector `d`.
    pub input_dim: usize,
    /// Number of stacked cross layers.
    pub n_cross_layers: usize,
    /// Hidden sizes of the parallel deep MLP tower (may be empty).
    pub deep_dims: Vec<usize>,
    /// Whether to use the vector or matrix-V2 cross formulation.
    pub kind: CrossKind,
}

/// Deep & Cross Network model (parallel cross + deep towers → sigmoid logit).
pub struct Dcn {
    config: DcnConfig,
    /// Per cross layer weight. For [`CrossKind::Vector`] each is length `d`;
    /// for [`CrossKind::MatrixV2`] each is length `d * d` (row-major).
    cross_w: Vec<Vec<f32>>,
    /// Per cross layer bias, length `d`.
    cross_b: Vec<Vec<f32>>,
    /// Deep MLP layers as `(weight [out × in], bias [out])`.
    deep_layers: Vec<(Vec<f32>, Vec<f32>)>,
    /// Final logit projection over `concat(cross_out, deep_out)`: length `cross_dim + deep_dim`.
    head_w: Vec<f32>,
    /// Final logit bias.
    head_b: f32,
    /// Output dimension of the deep tower (0 when `deep_dims` is empty).
    deep_out_dim: usize,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Dcn {
    /// Build a DCN with Gaussian-initialised parameters.
    ///
    /// # Errors
    ///
    /// Returns [`RecsysError::InvalidConfig`] if `input_dim` is zero, and
    /// [`RecsysError::InvalidEmbeddingDim`] if any deep hidden size is zero.
    pub fn new(config: DcnConfig, rng: &mut LcgRng) -> RecsysResult<Self> {
        let d = config.input_dim;
        if d == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "input_dim must be > 0".to_string(),
            });
        }
        for &h in &config.deep_dims {
            if h == 0 {
                return Err(RecsysError::InvalidEmbeddingDim { d: 0 });
            }
        }

        // Cross-tower parameters.
        let scale = (1.0 / d as f32).sqrt();
        let mut cross_w = Vec::with_capacity(config.n_cross_layers);
        let mut cross_b = Vec::with_capacity(config.n_cross_layers);
        for _ in 0..config.n_cross_layers {
            let w_len = match config.kind {
                CrossKind::Vector => d,
                CrossKind::MatrixV2 => d * d,
            };
            cross_w.push((0..w_len).map(|_| rng.next_normal() * scale).collect());
            cross_b.push(vec![0.0_f32; d]);
        }

        // Deep tower.
        let mut deep_layers = Vec::new();
        let mut in_dim = d;
        for &out_dim in &config.deep_dims {
            let sc = (2.0 / in_dim as f32).sqrt();
            let w: Vec<f32> = (0..out_dim * in_dim)
                .map(|_| rng.next_normal() * sc)
                .collect();
            let b = vec![0.0_f32; out_dim];
            deep_layers.push((w, b));
            in_dim = out_dim;
        }
        let deep_out_dim = if config.deep_dims.is_empty() {
            0
        } else {
            in_dim
        };

        // Head over concat(cross_out [d], deep_out [deep_out_dim]).
        let head_in = d + deep_out_dim;
        let sc = (1.0 / head_in as f32).sqrt();
        let head_w: Vec<f32> = (0..head_in).map(|_| rng.next_normal() * sc).collect();

        Ok(Self {
            config,
            cross_w,
            cross_b,
            deep_layers,
            head_w,
            head_b: 0.0,
            deep_out_dim,
        })
    }

    /// Run the cross tower only, returning the final `[d]` cross representation.
    ///
    /// # Errors
    ///
    /// Returns [`RecsysError::DimensionMismatch`] when `x.len() != input_dim`.
    pub fn cross_forward(&self, x: &[f32]) -> RecsysResult<Vec<f32>> {
        let d = self.config.input_dim;
        if x.len() != d {
            return Err(RecsysError::DimensionMismatch {
                expected: d,
                got: x.len(),
            });
        }
        let x0 = x;
        let mut x_l = x.to_vec();
        for (w, b) in self.cross_w.iter().zip(self.cross_b.iter()) {
            let mut next = vec![0.0_f32; d];
            match self.config.kind {
                CrossKind::Vector => {
                    // scalar s = x_l · w
                    let s: f32 = x_l.iter().zip(w.iter()).map(|(&a, &c)| a * c).sum();
                    for i in 0..d {
                        next[i] = x0[i] * s + b[i] + x_l[i];
                    }
                }
                CrossKind::MatrixV2 => {
                    // proj = W x_l + b   (W is [d × d] row-major)
                    for i in 0..d {
                        let mut acc = b[i];
                        let row = &w[i * d..(i + 1) * d];
                        for (j, &rj) in row.iter().enumerate() {
                            acc += rj * x_l[j];
                        }
                        // Hadamard with x0, plus residual.
                        next[i] = x0[i] * acc + x_l[i];
                    }
                }
            }
            x_l = next;
        }
        Ok(x_l)
    }

    /// Run the deep MLP tower only, returning the final `[deep_out_dim]` vector
    /// (empty when no deep layers are configured).
    ///
    /// # Errors
    ///
    /// Returns [`RecsysError::DimensionMismatch`] when `x.len() != input_dim`.
    pub fn deep_forward(&self, x: &[f32]) -> RecsysResult<Vec<f32>> {
        let d = self.config.input_dim;
        if x.len() != d {
            return Err(RecsysError::DimensionMismatch {
                expected: d,
                got: x.len(),
            });
        }
        // No deep tower configured → empty representation (consistent with
        // `deep_out_dim() == 0` and the parallel-head concatenation).
        if self.deep_layers.is_empty() {
            return Ok(Vec::new());
        }
        let mut cur = x.to_vec();
        let mut cur_dim = d;
        for (w, b) in &self.deep_layers {
            let out_dim = b.len();
            let mut out = vec![0.0_f32; out_dim];
            for o in 0..out_dim {
                let mut acc = b[o];
                let row = &w[o * cur_dim..(o + 1) * cur_dim];
                for (j, &rj) in row.iter().enumerate() {
                    acc += rj * cur[j];
                }
                // ReLU on every deep layer.
                out[o] = acc.max(0.0);
            }
            cur = out;
            cur_dim = out_dim;
        }
        Ok(cur)
    }

    /// Full forward pass: returns the click-through probability in `[0, 1]`.
    ///
    /// # Errors
    ///
    /// Propagates dimension errors from [`Self::cross_forward`] / [`Self::deep_forward`].
    pub fn forward(&self, x: &[f32]) -> RecsysResult<f32> {
        let cross_out = self.cross_forward(x)?;
        let deep_out = self.deep_forward(x)?;

        debug_assert_eq!(deep_out.len(), self.deep_out_dim);
        let mut logit = self.head_b;
        for (i, &v) in cross_out.iter().enumerate() {
            logit += self.head_w[i] * v;
        }
        let off = cross_out.len();
        for (i, &v) in deep_out.iter().enumerate() {
            logit += self.head_w[off + i] * v;
        }
        Ok(sigmoid(logit))
    }

    /// Number of stacked cross layers.
    pub fn n_cross_layers(&self) -> usize {
        self.config.n_cross_layers
    }

    /// Output dimension of the deep tower.
    pub fn deep_out_dim(&self) -> usize {
        self.deep_out_dim
    }

    /// Cross-layer formulation in use.
    pub fn kind(&self) -> CrossKind {
        self.config.kind
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(kind: CrossKind, n_cross: usize, deep: Vec<usize>) -> DcnConfig {
        DcnConfig {
            input_dim: 6,
            n_cross_layers: n_cross,
            deep_dims: deep,
            kind,
        }
    }

    #[test]
    fn vector_forward_in_unit_interval() {
        let mut rng = LcgRng::new(1);
        let model =
            Dcn::new(cfg(CrossKind::Vector, 2, vec![8, 4]), &mut rng).expect("model must build");
        let x: Vec<f32> = (0..6).map(|_| rng.next_f32()).collect();
        let p = model.forward(&x).expect("forward must succeed");
        assert!((0.0..=1.0).contains(&p), "prob {p} not in [0,1]");
    }

    #[test]
    fn matrix_v2_forward_in_unit_interval() {
        let mut rng = LcgRng::new(2);
        let model =
            Dcn::new(cfg(CrossKind::MatrixV2, 3, vec![8, 4]), &mut rng).expect("model must build");
        let x: Vec<f32> = (0..6).map(|_| rng.next_f32()).collect();
        let p = model.forward(&x).expect("forward must succeed");
        assert!((0.0..=1.0).contains(&p), "prob {p} not in [0,1]");
    }

    #[test]
    fn cross_output_has_input_dim() {
        let mut rng = LcgRng::new(3);
        let model =
            Dcn::new(cfg(CrossKind::Vector, 4, vec![]), &mut rng).expect("model must build");
        let x = vec![0.5_f32; 6];
        let out = model.cross_forward(&x).expect("cross must succeed");
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn zero_cross_layers_is_identity() {
        // With no cross layers the cross tower returns x unchanged.
        let mut rng = LcgRng::new(4);
        let model =
            Dcn::new(cfg(CrossKind::Vector, 0, vec![]), &mut rng).expect("model must build");
        let x = vec![0.1_f32, -0.2, 0.3, 0.4, -0.5, 0.6];
        let out = model.cross_forward(&x).expect("cross must succeed");
        for (a, b) in out.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-6, "expected identity, got {a} vs {b}");
        }
    }

    #[test]
    fn deep_output_dim_matches_last_hidden() {
        let mut rng = LcgRng::new(5);
        let model = Dcn::new(cfg(CrossKind::MatrixV2, 1, vec![10, 7, 3]), &mut rng)
            .expect("model must build");
        assert_eq!(model.deep_out_dim(), 3);
        let out = model
            .deep_forward(&[0.2_f32; 6])
            .expect("deep must succeed");
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn empty_deep_returns_empty() {
        let mut rng = LcgRng::new(6);
        let model =
            Dcn::new(cfg(CrossKind::Vector, 2, vec![]), &mut rng).expect("model must build");
        assert_eq!(model.deep_out_dim(), 0);
        let out = model
            .deep_forward(&[0.3_f32; 6])
            .expect("deep must succeed");
        assert!(out.is_empty());
    }

    #[test]
    fn deep_relu_nonnegative() {
        let mut rng = LcgRng::new(7);
        let model =
            Dcn::new(cfg(CrossKind::Vector, 1, vec![12, 8]), &mut rng).expect("model must build");
        let x: Vec<f32> = (0..6)
            .map(|i| if i % 2 == 0 { -1.0 } else { 1.0 })
            .collect();
        let out = model.deep_forward(&x).expect("deep must succeed");
        assert!(out.iter().all(|&v| v >= 0.0), "ReLU output must be >= 0");
    }

    #[test]
    fn forward_finite_extreme_inputs() {
        let mut rng = LcgRng::new(8);
        let model =
            Dcn::new(cfg(CrossKind::MatrixV2, 2, vec![8]), &mut rng).expect("model must build");
        let x = vec![1e3_f32; 6];
        let p = model.forward(&x).expect("forward must succeed");
        assert!(p.is_finite(), "prob must be finite for extreme input");
        assert!((0.0..=1.0).contains(&p));
    }

    #[test]
    fn dimension_mismatch_cross_errors() {
        let mut rng = LcgRng::new(9);
        let model =
            Dcn::new(cfg(CrossKind::Vector, 1, vec![4]), &mut rng).expect("model must build");
        let err = model.cross_forward(&[1.0, 2.0, 3.0]);
        assert!(matches!(err, Err(RecsysError::DimensionMismatch { .. })));
    }

    #[test]
    fn dimension_mismatch_deep_errors() {
        let mut rng = LcgRng::new(10);
        let model =
            Dcn::new(cfg(CrossKind::MatrixV2, 1, vec![4]), &mut rng).expect("model must build");
        let err = model.deep_forward(&[1.0, 2.0]);
        assert!(matches!(err, Err(RecsysError::DimensionMismatch { .. })));
    }

    #[test]
    fn zero_input_dim_rejected() {
        let mut rng = LcgRng::new(11);
        let err = Dcn::new(
            DcnConfig {
                input_dim: 0,
                n_cross_layers: 1,
                deep_dims: vec![4],
                kind: CrossKind::Vector,
            },
            &mut rng,
        );
        assert!(matches!(err, Err(RecsysError::InvalidConfig { .. })));
    }

    #[test]
    fn zero_deep_hidden_rejected() {
        let mut rng = LcgRng::new(12);
        let err = Dcn::new(
            DcnConfig {
                input_dim: 6,
                n_cross_layers: 1,
                deep_dims: vec![0],
                kind: CrossKind::Vector,
            },
            &mut rng,
        );
        assert!(matches!(err, Err(RecsysError::InvalidEmbeddingDim { .. })));
    }

    #[test]
    fn n_cross_layers_reported() {
        let mut rng = LcgRng::new(13);
        let model =
            Dcn::new(cfg(CrossKind::MatrixV2, 5, vec![4]), &mut rng).expect("model must build");
        assert_eq!(model.n_cross_layers(), 5);
        assert_eq!(model.kind(), CrossKind::MatrixV2);
    }

    #[test]
    fn vector_single_layer_matches_formula() {
        // Build a model then overwrite parameters to a known state and verify the
        // closed-form DCN vector cross: x1 = x0 * (x0·w) + b + x0.
        let mut rng = LcgRng::new(14);
        let mut model =
            Dcn::new(cfg(CrossKind::Vector, 1, vec![]), &mut rng).expect("model must build");
        model.cross_w[0] = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        model.cross_b[0] = vec![0.5; 6];
        let x0 = vec![2.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0];
        // s = x0·w = 2.0; x1[i] = x0[i]*2 + 0.5 + x0[i] = 3*x0[i] + 0.5
        let out = model.cross_forward(&x0).expect("cross must succeed");
        assert!((out[0] - (3.0 * 2.0 + 0.5)).abs() < 1e-5, "got {}", out[0]);
        assert!((out[1] - (3.0 * 1.0 + 0.5)).abs() < 1e-5, "got {}", out[1]);
    }

    #[test]
    fn matrix_identity_weight_doubles_residual() {
        // With W = I and b = 0: x1[i] = x0[i]*(x_l[i]) + x_l[i]. For l=0, x_l=x0,
        // so x1[i] = x0[i]^2 + x0[i].
        let mut rng = LcgRng::new(15);
        let mut model =
            Dcn::new(cfg(CrossKind::MatrixV2, 1, vec![]), &mut rng).expect("model must build");
        let d = 6;
        let mut w = vec![0.0_f32; d * d];
        for i in 0..d {
            w[i * d + i] = 1.0;
        }
        model.cross_w[0] = w;
        model.cross_b[0] = vec![0.0; d];
        let x0 = vec![0.5_f32, 1.0, 2.0, 0.0, -1.0, 3.0];
        let out = model.cross_forward(&x0).expect("cross must succeed");
        for i in 0..d {
            let expected = x0[i] * x0[i] + x0[i];
            assert!(
                (out[i] - expected).abs() < 1e-5,
                "i={i}: got {} want {expected}",
                out[i]
            );
        }
    }
}
