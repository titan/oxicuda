//! OML — Online-aware Meta-Learning (Javed & White, 2019:
//! "Meta-Learning Representations for Continual Learning").
//!
//! OML factorises the network into two cooperating parts:
//!
//! * a **Representation-Learning Network (RLN)** — a slow, meta-learned encoder
//!   that maps an input into a fixed-width representation `z`.  The RLN is held
//!   *frozen* during the inner (online) loop and is updated only in the outer
//!   (meta) loop so that the representations it produces remain robust to the
//!   online updates of the head;
//! * a **Prediction-Learning Network (PLN)** — a fast, inner-updated linear head
//!   that sits on top of `z` and is the only thing that moves while a stream of
//!   samples arrives one (or a few) at a time.
//!
//! The meta-objective is *catastrophic-forgetting aware*: after a short online
//! trajectory has been replayed through the PLN, the loss is evaluated on a
//! "remember" set (the current sample plus earlier ones) and the gradient of
//! that loss is used to nudge the **RLN** (and the PLN *initialisation*) so that
//! online adaptation hurts the remembered samples as little as possible.
//!
//! # Architecture (dense CPU reference)
//!
//! ```text
//!   x ──▶ Linear(in→hidden) ──ReLU──▶ Linear(hidden→repr) ──ReLU──▶ z   (RLN, slow)
//!   z ──▶ Linear(repr→classes) ──▶ logits                                (PLN, fast)
//! ```
//!
//! The inner loop performs online SGD on the PLN only.  The outer loop applies a
//! first-order (FOMAML-style) meta-update: the remember-set gradient evaluated
//! at the *adapted* parameters is back-propagated through the (frozen) RLN and
//! applied both to the RLN weights and to the PLN initialisation.  All forward
//! and backward passes are analytic — no finite differences — keeping the dense
//! reference exact and fast.

use crate::error::{MetaError, MetaResult};
use crate::gradient::inner_loop::cross_entropy_loss;
use crate::handle::LcgRng;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Hyper-parameters for an [`Oml`] learner.
#[derive(Debug, Clone)]
pub struct OmlConfig {
    /// Input feature dimensionality.
    pub in_dim: usize,
    /// RLN hidden width (first encoder layer).
    pub hidden_dim: usize,
    /// Representation width produced by the RLN (PLN input width).
    pub repr_dim: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Inner-loop (online) learning rate for the PLN head.
    pub inner_lr: f32,
    /// Outer-loop (meta) learning rate for the RLN and PLN initialisation.
    pub meta_lr: f32,
    /// Number of online passes over the trajectory in the inner loop
    /// (each pass performs one SGD step per trajectory sample).
    pub inner_passes: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Dense linear layer
// ─────────────────────────────────────────────────────────────────────────────

/// A dense affine layer `y = W x + b` with row-major weights `[out × in]`.
///
/// Used internally for both RLN layers and exposed as the adapted-PLN return
/// type of [`Oml::inner_adapt`].
#[derive(Debug, Clone)]
pub struct OmlLinear {
    w: Vec<f32>,
    b: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl OmlLinear {
    fn new(in_dim: usize, out_dim: usize, rng: &mut LcgRng) -> Self {
        let limit = (6.0_f32 / (in_dim + out_dim) as f32).sqrt();
        let mut w = vec![0.0_f32; out_dim * in_dim];
        for v in w.iter_mut() {
            *v = (rng.next_f32() * 2.0 - 1.0) * limit;
        }
        Self {
            w,
            b: vec![0.0_f32; out_dim],
            in_dim,
            out_dim,
        }
    }

    /// Forward pass `W x + b`. Assumes `x.len() == in_dim` (caller-validated).
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut out = self.b.clone();
        for (o, out_o) in out.iter_mut().enumerate() {
            let row = &self.w[o * self.in_dim..(o + 1) * self.in_dim];
            *out_o += row
                .iter()
                .zip(x.iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum::<f32>();
        }
        out
    }

    /// Read-only view of the weight matrix (`[out × in]` row-major).
    pub fn weights(&self) -> &[f32] {
        &self.w
    }

    /// Read-only view of the bias vector (length `out`).
    pub fn bias(&self) -> &[f32] {
        &self.b
    }

    /// Number of input features.
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// Number of output features.
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn relu_inplace(v: &mut [f32]) {
    for x in v.iter_mut() {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
}

/// Numerically-stable softmax.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|&z| (z - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for e in exps.iter_mut() {
            *e /= sum;
        }
    }
    exps
}

// ─────────────────────────────────────────────────────────────────────────────
// OML learner
// ─────────────────────────────────────────────────────────────────────────────

/// An Online-aware Meta-Learning learner with a two-layer RLN encoder and a
/// linear PLN head.
pub struct Oml {
    /// RLN first layer: `in_dim → hidden_dim`.
    rln1: OmlLinear,
    /// RLN second layer: `hidden_dim → repr_dim`.
    rln2: OmlLinear,
    /// Meta-learned PLN initialisation: `repr_dim → n_classes`.
    pln_init: OmlLinear,
    config: OmlConfig,
}

impl Oml {
    /// Construct an OML learner with Xavier-initialised weights and zero biases.
    ///
    /// # Errors
    /// Returns an error if any dimension is zero, `n_classes < 2`, a learning
    /// rate is non-positive/non-finite, or `inner_passes == 0`.
    pub fn new(config: OmlConfig, rng: &mut LcgRng) -> MetaResult<Self> {
        if config.in_dim == 0 {
            return Err(MetaError::InvalidFeatDim { dim: config.in_dim });
        }
        if config.hidden_dim == 0 || config.repr_dim == 0 {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "hidden_dim and repr_dim must be > 0".into(),
            });
        }
        if config.n_classes < 2 {
            return Err(MetaError::InvalidNWay {
                n_way: config.n_classes,
            });
        }
        if config.inner_lr <= 0.0 || !config.inner_lr.is_finite() {
            return Err(MetaError::InvalidLr {
                lr: config.inner_lr,
            });
        }
        if config.meta_lr <= 0.0 || !config.meta_lr.is_finite() {
            return Err(MetaError::InvalidLr { lr: config.meta_lr });
        }
        if config.inner_passes == 0 {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "inner_passes must be > 0".into(),
            });
        }

        let rln1 = OmlLinear::new(config.in_dim, config.hidden_dim, rng);
        let rln2 = OmlLinear::new(config.hidden_dim, config.repr_dim, rng);
        let pln_init = OmlLinear::new(config.repr_dim, config.n_classes, rng);
        Ok(Self {
            rln1,
            rln2,
            pln_init,
            config,
        })
    }

    /// Read-only access to the configuration.
    pub fn config(&self) -> &OmlConfig {
        &self.config
    }

    /// The current (meta-learned) PLN initialisation.
    pub fn pln_init(&self) -> &OmlLinear {
        &self.pln_init
    }

    /// Forward the RLN encoder, returning `(hidden_relu, representation)`.
    fn rln_forward(&self, x: &[f32]) -> MetaResult<(Vec<f32>, Vec<f32>)> {
        if x.len() != self.config.in_dim {
            return Err(MetaError::DimensionMismatch {
                expected: self.config.in_dim,
                got: x.len(),
            });
        }
        let mut h1 = self.rln1.forward(x);
        relu_inplace(&mut h1);
        let mut z = self.rln2.forward(&h1);
        relu_inplace(&mut z);
        Ok((h1, z))
    }

    /// Encode a single input into its `repr_dim`-dimensional representation `z`.
    pub fn representation(&self, x: &[f32]) -> MetaResult<Vec<f32>> {
        let (_, z) = self.rln_forward(x)?;
        Ok(z)
    }

    /// Class logits for a single input under a given PLN head.
    ///
    /// # Errors
    /// `DimensionMismatch` if `x.len() != in_dim` or the head's input width does
    /// not match `repr_dim`.
    pub fn forward_logits(&self, pln: &OmlLinear, x: &[f32]) -> MetaResult<Vec<f32>> {
        if pln.in_dim != self.config.repr_dim {
            return Err(MetaError::DimensionMismatch {
                expected: self.config.repr_dim,
                got: pln.in_dim,
            });
        }
        let (_, z) = self.rln_forward(x)?;
        Ok(pln.forward(&z))
    }

    /// Inner loop: starting from the PLN initialisation, perform online SGD on a
    /// streaming trajectory (one gradient step per sample, repeated for
    /// `inner_passes` passes) with the RLN held frozen.
    ///
    /// `traj_x` is row-major `[n × in_dim]`; `traj_y` are the labels.
    ///
    /// # Errors
    /// `EmptySupport` for an empty trajectory, `DimensionMismatch` on a bad
    /// shape, or `InvalidNWay` if a label is out of range.
    pub fn inner_adapt(&self, traj_x: &[f32], traj_y: &[u32]) -> MetaResult<OmlLinear> {
        let n = traj_y.len();
        if n == 0 {
            return Err(MetaError::EmptySupport);
        }
        if traj_x.len() != n * self.config.in_dim {
            return Err(MetaError::DimensionMismatch {
                expected: n * self.config.in_dim,
                got: traj_x.len(),
            });
        }
        let repr = self.config.repr_dim;
        let mut pln = self.pln_init.clone();
        let lr = self.config.inner_lr;

        for _ in 0..self.config.inner_passes {
            for s in 0..n {
                let x = &traj_x[s * self.config.in_dim..(s + 1) * self.config.in_dim];
                let y = traj_y[s] as usize;
                if y >= self.config.n_classes {
                    return Err(MetaError::InvalidNWay {
                        n_way: self.config.n_classes,
                    });
                }
                let (_, z) = self.rln_forward(x)?;
                let logits = pln.forward(&z);
                let p = softmax(&logits);
                // dL/dlogits = softmax − onehot.
                let mut dlogits = p;
                dlogits[y] -= 1.0;
                // Online SGD step on the linear head.
                for (c, &dl) in dlogits.iter().enumerate() {
                    pln.b[c] -= lr * dl;
                    let row = &mut pln.w[c * repr..(c + 1) * repr];
                    for (wv, &zj) in row.iter_mut().zip(z.iter()) {
                        *wv -= lr * dl * zj;
                    }
                }
            }
        }
        Ok(pln)
    }

    /// Mean cross-entropy of a labelled set under a fixed PLN head.
    fn loss_on_set(&self, pln: &OmlLinear, xs: &[f32], ys: &[u32]) -> MetaResult<f32> {
        let n = ys.len();
        if n == 0 {
            return Err(MetaError::EmptySupport);
        }
        if xs.len() != n * self.config.in_dim {
            return Err(MetaError::DimensionMismatch {
                expected: n * self.config.in_dim,
                got: xs.len(),
            });
        }
        let mut total = 0.0_f32;
        for s in 0..n {
            let x = &xs[s * self.config.in_dim..(s + 1) * self.config.in_dim];
            let (_, z) = self.rln_forward(x)?;
            let logits = pln.forward(&z);
            total += cross_entropy_loss(&logits, &ys[s..s + 1], self.config.n_classes)?;
        }
        Ok(total / n as f32)
    }

    /// Forgetting-aware meta-loss: adapt the PLN online on `traj`, then evaluate
    /// the mean cross-entropy on the remember set under the adapted head.
    ///
    /// # Errors
    /// Propagates [`Self::inner_adapt`] / `Self::loss_on_set` errors.
    pub fn meta_loss(
        &self,
        traj_x: &[f32],
        traj_y: &[u32],
        remember_x: &[f32],
        remember_y: &[u32],
    ) -> MetaResult<f32> {
        let pln = self.inner_adapt(traj_x, traj_y)?;
        self.loss_on_set(&pln, remember_x, remember_y)
    }

    /// One meta-training step.
    ///
    /// 1. Adapt the PLN online on `traj` (RLN frozen).
    /// 2. Compute the remember-set loss and its gradient w.r.t. the RLN and the
    ///    adapted PLN (first-order / FOMAML approximation).
    /// 3. Update the RLN weights and the PLN initialisation by `meta_lr`.
    ///
    /// Returns the remember-set meta-loss measured *before* this step's update.
    ///
    /// # Errors
    /// Propagates shape / label errors from adaptation and evaluation.
    pub fn meta_step(
        &mut self,
        traj_x: &[f32],
        traj_y: &[u32],
        remember_x: &[f32],
        remember_y: &[u32],
    ) -> MetaResult<f32> {
        let pln = self.inner_adapt(traj_x, traj_y)?;
        let n = remember_y.len();
        if n == 0 {
            return Err(MetaError::EmptySupport);
        }
        if remember_x.len() != n * self.config.in_dim {
            return Err(MetaError::DimensionMismatch {
                expected: n * self.config.in_dim,
                got: remember_x.len(),
            });
        }

        let in_dim = self.config.in_dim;
        let hidden = self.config.hidden_dim;
        let repr = self.config.repr_dim;
        let inv_n = 1.0_f32 / n as f32;

        let mut g_w1 = vec![0.0_f32; self.rln1.w.len()];
        let mut g_b1 = vec![0.0_f32; self.rln1.b.len()];
        let mut g_w2 = vec![0.0_f32; self.rln2.w.len()];
        let mut g_b2 = vec![0.0_f32; self.rln2.b.len()];
        let mut g_w3 = vec![0.0_f32; self.pln_init.w.len()];
        let mut g_b3 = vec![0.0_f32; self.pln_init.b.len()];
        let mut meta_loss = 0.0_f32;

        for s in 0..n {
            let x = &remember_x[s * in_dim..(s + 1) * in_dim];
            let y = remember_y[s] as usize;
            if y >= self.config.n_classes {
                return Err(MetaError::InvalidNWay {
                    n_way: self.config.n_classes,
                });
            }
            let (h1, z) = self.rln_forward(x)?;
            let logits = pln.forward(&z);
            let p = softmax(&logits);
            meta_loss += -(p[y].max(1e-12)).ln();

            // dL/dlogits averaged over the remember set.
            let mut dlogits = p;
            dlogits[y] -= 1.0;
            for d in dlogits.iter_mut() {
                *d *= inv_n;
            }

            // PLN' grads (applied to the PLN initialisation, FOMAML-style).
            for (c, &dl) in dlogits.iter().enumerate() {
                g_b3[c] += dl;
                let row = &mut g_w3[c * repr..(c + 1) * repr];
                for (gw, &zj) in row.iter_mut().zip(z.iter()) {
                    *gw += dl * zj;
                }
            }

            // Back-prop into the representation: dz = (W3)^T dlogits.
            let mut dz = vec![0.0_f32; repr];
            for (c, &dl) in dlogits.iter().enumerate() {
                let row = &pln.w[c * repr..(c + 1) * repr];
                for (dzj, &wj) in dz.iter_mut().zip(row.iter()) {
                    *dzj += wj * dl;
                }
            }
            // Through the RLN's second ReLU (z > 0): dz becomes da2.
            for (dzj, &zj) in dz.iter_mut().zip(z.iter()) {
                if zj <= 0.0 {
                    *dzj = 0.0;
                }
            }
            // RLN2 grads: dW2 = da2 ⊗ h1, db2 = da2.
            for (o, &d2) in dz.iter().enumerate() {
                g_b2[o] += d2;
                let row = &mut g_w2[o * hidden..(o + 1) * hidden];
                for (gw, &hi) in row.iter_mut().zip(h1.iter()) {
                    *gw += d2 * hi;
                }
            }
            // Back-prop into the hidden layer: dh1 = (W2)^T da2.
            let mut dh1 = vec![0.0_f32; hidden];
            for (o, &d2) in dz.iter().enumerate() {
                let row = &self.rln2.w[o * hidden..(o + 1) * hidden];
                for (dh, &wi) in dh1.iter_mut().zip(row.iter()) {
                    *dh += wi * d2;
                }
            }
            // Through the RLN's first ReLU (h1 > 0): dh1 becomes da1.
            for (dh, &hi) in dh1.iter_mut().zip(h1.iter()) {
                if hi <= 0.0 {
                    *dh = 0.0;
                }
            }
            // RLN1 grads: dW1 = da1 ⊗ x, db1 = da1.
            for (o, &d1) in dh1.iter().enumerate() {
                g_b1[o] += d1;
                let row = &mut g_w1[o * in_dim..(o + 1) * in_dim];
                for (gw, &xi) in row.iter_mut().zip(x.iter()) {
                    *gw += d1 * xi;
                }
            }
        }
        meta_loss *= inv_n;

        // Apply the meta-update.
        let mlr = self.config.meta_lr;
        for (wv, &g) in self.rln1.w.iter_mut().zip(g_w1.iter()) {
            *wv -= mlr * g;
        }
        for (bv, &g) in self.rln1.b.iter_mut().zip(g_b1.iter()) {
            *bv -= mlr * g;
        }
        for (wv, &g) in self.rln2.w.iter_mut().zip(g_w2.iter()) {
            *wv -= mlr * g;
        }
        for (bv, &g) in self.rln2.b.iter_mut().zip(g_b2.iter()) {
            *bv -= mlr * g;
        }
        for (wv, &g) in self.pln_init.w.iter_mut().zip(g_w3.iter()) {
            *wv -= mlr * g;
        }
        for (bv, &g) in self.pln_init.b.iter_mut().zip(g_b3.iter()) {
            *bv -= mlr * g;
        }

        Ok(meta_loss)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> OmlConfig {
        OmlConfig {
            in_dim: 4,
            hidden_dim: 8,
            repr_dim: 4,
            n_classes: 2,
            inner_lr: 0.2,
            meta_lr: 0.1,
            inner_passes: 3,
        }
    }

    /// A tiny, linearly-separable 2-class task in 4-D.
    /// Class 0 clusters near (+1,+1,0,0), class 1 near (0,0,+1,+1).
    fn synthetic_task() -> (Vec<f32>, Vec<u32>) {
        let x = vec![
            1.0, 1.0, 0.0, 0.0, // c0
            0.9, 1.1, 0.1, 0.0, // c0
            0.0, 0.0, 1.0, 1.0, // c1
            0.1, 0.0, 1.1, 0.9, // c1
        ];
        let y = vec![0_u32, 0, 1, 1];
        (x, y)
    }

    #[test]
    fn new_rejects_bad_dims() {
        let mut rng = LcgRng::new(1);
        let mut cfg = tiny_config();
        cfg.in_dim = 0;
        assert!(matches!(
            Oml::new(cfg, &mut rng),
            Err(MetaError::InvalidFeatDim { .. })
        ));
    }

    #[test]
    fn new_rejects_single_class() {
        let mut rng = LcgRng::new(1);
        let mut cfg = tiny_config();
        cfg.n_classes = 1;
        assert!(matches!(
            Oml::new(cfg, &mut rng),
            Err(MetaError::InvalidNWay { .. })
        ));
    }

    #[test]
    fn new_rejects_zero_passes() {
        let mut rng = LcgRng::new(1);
        let mut cfg = tiny_config();
        cfg.inner_passes = 0;
        assert!(matches!(
            Oml::new(cfg, &mut rng),
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
    }

    #[test]
    fn representation_shape() {
        let mut rng = LcgRng::new(7);
        let oml = Oml::new(tiny_config(), &mut rng).expect("value should be present");
        let z = oml
            .representation(&[0.1, 0.2, 0.3, 0.4])
            .expect("representation should succeed");
        assert_eq!(z.len(), 4);
        assert!(z.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    #[test]
    fn inner_adapt_shapes() {
        let mut rng = LcgRng::new(7);
        let oml = Oml::new(tiny_config(), &mut rng).expect("value should be present");
        let (x, y) = synthetic_task();
        let pln = oml.inner_adapt(&x, &y).expect("inner_adapt should succeed");
        assert_eq!(pln.in_dim(), 4);
        assert_eq!(pln.out_dim(), 2);
        assert_eq!(pln.weights().len(), 2 * 4);
        assert_eq!(pln.bias().len(), 2);
    }

    #[test]
    fn inner_adapt_empty_errors() {
        let mut rng = LcgRng::new(7);
        let oml = Oml::new(tiny_config(), &mut rng).expect("value should be present");
        assert!(matches!(
            oml.inner_adapt(&[], &[]),
            Err(MetaError::EmptySupport)
        ));
    }

    #[test]
    fn inner_adapt_reduces_last_sample_loss() {
        // The adapted PLN should fit the most-recently-seen trajectory sample
        // better than the un-adapted initialisation.
        let mut rng = LcgRng::new(11);
        let oml = Oml::new(tiny_config(), &mut rng).expect("value should be present");
        let (x, y) = synthetic_task();
        let n = y.len();
        let last = &x[(n - 1) * 4..n * 4];
        let last_y = &y[n - 1..n];

        let logits_init = oml
            .forward_logits(oml.pln_init(), last)
            .expect("value should be present");
        let loss_init =
            cross_entropy_loss(&logits_init, last_y, 2).expect("cross_entropy_loss should succeed");

        let adapted = oml.inner_adapt(&x, &y).expect("inner_adapt should succeed");
        let logits_adapt = oml
            .forward_logits(&adapted, last)
            .expect("forward_logits should succeed");
        let loss_adapt = cross_entropy_loss(&logits_adapt, last_y, 2)
            .expect("cross_entropy_loss should succeed");

        assert!(
            loss_adapt < loss_init,
            "online adaptation must reduce the last-sample loss: {loss_adapt} !< {loss_init}"
        );
    }

    #[test]
    fn meta_loss_is_finite() {
        let mut rng = LcgRng::new(3);
        let oml = Oml::new(tiny_config(), &mut rng).expect("value should be present");
        let (x, y) = synthetic_task();
        let loss = oml
            .meta_loss(&x, &y, &x, &y)
            .expect("meta_loss should succeed");
        assert!(loss.is_finite() && loss >= 0.0);
    }

    #[test]
    fn meta_step_decreases_meta_loss() {
        // A handful of meta-steps on the same task must not increase — and in
        // practice decreases — the forgetting-aware remember-set loss.
        let mut rng = LcgRng::new(2024);
        let mut oml = Oml::new(tiny_config(), &mut rng).expect("value should be present");
        let (x, y) = synthetic_task();
        let first = oml
            .meta_step(&x, &y, &x, &y)
            .expect("meta_step should succeed");
        let mut last = first;
        for _ in 0..6 {
            last = oml
                .meta_step(&x, &y, &x, &y)
                .expect("meta_step should succeed");
            assert!(last.is_finite());
        }
        assert!(
            last <= first + 1e-4,
            "meta-loss should hold or decrease: {last} vs {first}"
        );
        assert!(
            last < first,
            "expected a strict decrease: {last} vs {first}"
        );
    }

    #[test]
    fn meta_step_deterministic_under_seed() {
        let (x, y) = synthetic_task();
        let mut a = Oml::new(tiny_config(), &mut LcgRng::new(99)).expect("value should be present");
        let mut b = Oml::new(tiny_config(), &mut LcgRng::new(99)).expect("value should be present");
        let mut la = 0.0;
        let mut lb = 0.0;
        for _ in 0..5 {
            la = a
                .meta_step(&x, &y, &x, &y)
                .expect("meta_step should succeed");
            lb = b
                .meta_step(&x, &y, &x, &y)
                .expect("meta_step should succeed");
        }
        assert_eq!(la, lb);
        assert_eq!(a.pln_init().weights(), b.pln_init().weights());
    }

    #[test]
    fn online_updates_on_long_sequence_no_nan() {
        // Replay a long, repeated stream and confirm nothing diverges.
        let mut rng = LcgRng::new(55);
        let oml = Oml::new(tiny_config(), &mut rng).expect("value should be present");
        let (base_x, base_y) = synthetic_task();
        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..16 {
            x.extend_from_slice(&base_x);
            y.extend_from_slice(&base_y);
        }
        let adapted = oml.inner_adapt(&x, &y).expect("inner_adapt should succeed");
        assert!(adapted.weights().iter().all(|v| v.is_finite()));
        assert!(adapted.bias().iter().all(|v| v.is_finite()));
        let logits = oml
            .forward_logits(&adapted, &base_x[..4])
            .expect("forward_logits should succeed");
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_logits_shape() {
        let mut rng = LcgRng::new(7);
        let oml = Oml::new(tiny_config(), &mut rng).expect("value should be present");
        let logits = oml
            .forward_logits(oml.pln_init(), &[0.5, 0.5, 0.5, 0.5])
            .expect("value should be present");
        assert_eq!(logits.len(), 2);
        assert!(logits.iter().all(|v| v.is_finite()));
    }
}
