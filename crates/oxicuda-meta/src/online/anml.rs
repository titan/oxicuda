//! ANML — A Neuromodulated Meta-Learning algorithm (Beaulieu et al. 2020:
//! "Learning to Continually Learn", ECAI 2020).
//!
//! ANML extends the OML representation/prediction factorisation with a learned
//! **neuromodulatory network (NM)** that gates the prediction network's
//! representation *multiplicatively*.  The NM acts as a selective-plasticity
//! controller: it decides, per representation unit and conditioned on the
//! current input, how strongly that unit participates — which in turn governs
//! how much the online (inner-loop) update perturbs each weight.  This gating
//! is the defining difference from plain OML and is what gives ANML its
//! resistance to catastrophic forgetting under long online trajectories.
//!
//! # Architecture (dense CPU reference)
//!
//! ```text
//!   ── Prediction Network (PN) ──────────────────────────────────────────────
//!   x ──▶ Linear(in→repr) ──ReLU──▶ p                                  (PN encoder, slow)
//!
//!   ── Neuromodulatory Network (NM) ─────────────────────────────────────────
//!   x ──▶ Linear(in→repr) ──sigmoid──▶ g                              (NM gate, slow)
//!
//!   ── Gated representation + head ──────────────────────────────────────────
//!   z = p ⊙ g                                                          (element-wise gate)
//!   x ──▶ Linear(repr→classes) on z ──▶ logits                         (PN head, fast)
//! ```
//!
//! The inner (online) loop performs SGD on the **PN head only** while a stream
//! of samples arrives one at a time; the slow PN encoder and the NM are frozen.
//! The outer (meta) loop applies a first-order (FOMAML-style) update: the
//! remember-set loss evaluated at the *adapted* head, with the slow encoder and
//! gate held at their current values, is back-propagated to the NM weights and
//! to the PN-encoder weights and to the PN-head *initialisation*.  All forward
//! and backward passes are analytic — no finite differences — keeping the dense
//! reference exact and fast.

use crate::error::{MetaError, MetaResult};
use crate::gradient::inner_loop::cross_entropy_loss;
use crate::handle::LcgRng;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Hyper-parameters for an [`Anml`] learner.
#[derive(Debug, Clone)]
pub struct AnmlConfig {
    /// Input feature dimensionality.
    pub in_dim: usize,
    /// Representation width produced by both the PN encoder and the NM gate
    /// (they share the gated width).
    pub repr_dim: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Inner-loop (online) learning rate for the PN head.
    pub inner_lr: f32,
    /// Outer-loop (meta) learning rate for the NM, the PN encoder, and the PN
    /// head initialisation.
    pub meta_lr: f32,
    /// Number of online passes over the trajectory in the inner loop (each pass
    /// performs one SGD step per trajectory sample).
    pub inner_passes: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Dense linear layer
// ─────────────────────────────────────────────────────────────────────────────

/// A dense affine layer `y = W x + b` with row-major weights `[out × in]`.
///
/// Reused for the PN encoder, the NM gate, and (as the adapted return type of
/// [`Anml::inner_adapt`]) the PN head.
#[derive(Debug, Clone)]
pub struct AnmlLinear {
    w: Vec<f32>,
    b: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl AnmlLinear {
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
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Row-major softmax over `n_classes` logits.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&z| (z - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward cache (per example) — reused by inner and outer passes
// ─────────────────────────────────────────────────────────────────────────────

/// Cached intermediates of one ANML forward pass for one example.
struct AnmlForward {
    /// PN-encoder pre-activations (length `repr_dim`).
    pn_pre: Vec<f32>,
    /// PN-encoder post-ReLU activations `p` (length `repr_dim`).
    p: Vec<f32>,
    /// NM gate values `g = sigmoid(NM(x))` (length `repr_dim`).
    g: Vec<f32>,
    /// Gated representation `z = p ⊙ g` (length `repr_dim`) — head input.
    z: Vec<f32>,
}

// ─────────────────────────────────────────────────────────────────────────────
// ANML learner
// ─────────────────────────────────────────────────────────────────────────────

/// An ANML meta-learner owning the slow PN encoder, the slow NM gate network,
/// and the (meta-learned) PN head initialisation.
pub struct Anml {
    /// PN encoder `Linear(in → repr)`, frozen during the inner loop.
    pn_encoder: AnmlLinear,
    /// NM gate `Linear(in → repr)`, frozen during the inner loop.
    nm_gate: AnmlLinear,
    /// PN head `Linear(repr → classes)` — the meta-learned *initialisation*
    /// from which each inner trajectory adapts.
    pn_head: AnmlLinear,
    cfg: AnmlConfig,
}

impl Anml {
    /// Construct an ANML learner with Xavier-initialised PN encoder, NM gate and
    /// PN head.
    ///
    /// # Errors
    /// * [`MetaError::InvalidFeatDim`] if `in_dim == 0` or `repr_dim == 0`.
    /// * [`MetaError::InvalidNWay`] if `n_classes < 2`.
    /// * [`MetaError::InvalidLr`] if either learning rate is non-positive or not
    ///   finite.
    pub fn new(cfg: AnmlConfig, rng: &mut LcgRng) -> MetaResult<Self> {
        if cfg.in_dim == 0 {
            return Err(MetaError::InvalidFeatDim { dim: cfg.in_dim });
        }
        if cfg.repr_dim == 0 {
            return Err(MetaError::InvalidFeatDim { dim: cfg.repr_dim });
        }
        if cfg.n_classes < 2 {
            return Err(MetaError::InvalidNWay {
                n_way: cfg.n_classes,
            });
        }
        if cfg.inner_lr <= 0.0 || !cfg.inner_lr.is_finite() {
            return Err(MetaError::InvalidLr { lr: cfg.inner_lr });
        }
        if cfg.meta_lr <= 0.0 || !cfg.meta_lr.is_finite() {
            return Err(MetaError::InvalidLr { lr: cfg.meta_lr });
        }
        let pn_encoder = AnmlLinear::new(cfg.in_dim, cfg.repr_dim, rng);
        let nm_gate = AnmlLinear::new(cfg.in_dim, cfg.repr_dim, rng);
        let pn_head = AnmlLinear::new(cfg.repr_dim, cfg.n_classes, rng);
        Ok(Self {
            pn_encoder,
            nm_gate,
            pn_head,
            cfg,
        })
    }

    /// Read-only access to the configuration.
    pub fn config(&self) -> &AnmlConfig {
        &self.cfg
    }

    /// Read-only access to the slow PN encoder.
    pub fn pn_encoder(&self) -> &AnmlLinear {
        &self.pn_encoder
    }

    /// Read-only access to the slow NM gate network.
    pub fn nm_gate(&self) -> &AnmlLinear {
        &self.nm_gate
    }

    /// Read-only access to the meta-learned PN head initialisation.
    pub fn pn_head(&self) -> &AnmlLinear {
        &self.pn_head
    }

    /// Compute the gated representation `z = ReLU(PN(x)) ⊙ sigmoid(NM(x))` for a
    /// single input, caching every intermediate needed by the backward pass.
    fn forward_repr(&self, x: &[f32]) -> AnmlForward {
        let pn_pre = self.pn_encoder.forward(x);
        let p: Vec<f32> = pn_pre.iter().map(|&v| v.max(0.0)).collect();
        let nm_pre = self.nm_gate.forward(x);
        let g: Vec<f32> = nm_pre.iter().map(|&v| sigmoid(v)).collect();
        let z: Vec<f32> = p.iter().zip(g.iter()).map(|(&pi, &gi)| pi * gi).collect();
        AnmlForward { pn_pre, p, g, z }
    }

    /// Public gated-representation accessor for a single input.
    ///
    /// Returns `z = ReLU(PN(x)) ⊙ sigmoid(NM(x))` (length `repr_dim`).
    ///
    /// # Errors
    /// [`MetaError::DimensionMismatch`] if `x.len() != in_dim`.
    pub fn representation(&self, x: &[f32]) -> MetaResult<Vec<f32>> {
        if x.len() != self.cfg.in_dim {
            return Err(MetaError::DimensionMismatch {
                expected: self.cfg.in_dim,
                got: x.len(),
            });
        }
        Ok(self.forward_repr(x).z)
    }

    /// Logits `head · z` of a single example under an explicit (possibly
    /// inner-adapted) head.
    fn head_logits(head: &AnmlLinear, z: &[f32]) -> Vec<f32> {
        head.forward(z)
    }

    /// Predict the class of a single input under an explicit head.
    ///
    /// # Errors
    /// [`MetaError::DimensionMismatch`] if `x.len() != in_dim`.
    pub fn predict_with(&self, head: &AnmlLinear, x: &[f32]) -> MetaResult<u32> {
        if x.len() != self.cfg.in_dim {
            return Err(MetaError::DimensionMismatch {
                expected: self.cfg.in_dim,
                got: x.len(),
            });
        }
        let z = self.forward_repr(x).z;
        let logits = Self::head_logits(head, &z);
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        Ok(best as u32)
    }

    /// Validate a `(x_flat, y)` trajectory/remember dataset against the config.
    fn check_dataset(&self, x_flat: &[f32], y: &[u32]) -> MetaResult<usize> {
        let n = y.len();
        if n == 0 {
            return Err(MetaError::EmptySupport);
        }
        if x_flat.len() != n * self.cfg.in_dim {
            return Err(MetaError::DimensionMismatch {
                expected: n * self.cfg.in_dim,
                got: x_flat.len(),
            });
        }
        for &lbl in y {
            if lbl as usize >= self.cfg.n_classes {
                return Err(MetaError::Internal {
                    msg: format!("label {lbl} >= n_classes {}", self.cfg.n_classes),
                });
            }
        }
        Ok(n)
    }

    /// One online SGD step on the PN head for a single `(x, y)` sample.
    ///
    /// The slow PN encoder and NM gate are frozen, so the gated representation
    /// `z` is a constant and only the head weights/bias move along the
    /// softmax-cross-entropy gradient `(softmax − onehot) ⊗ z`.
    fn head_sgd_step(&self, head: &mut AnmlLinear, z: &[f32], label: u32) {
        let logits = Self::head_logits(head, z);
        let probs = softmax(&logits);
        let lr = self.cfg.inner_lr;
        for (c, &prob) in probs.iter().enumerate() {
            let dlogit = prob - if c == label as usize { 1.0 } else { 0.0 };
            let row = &mut head.w[c * head.in_dim..(c + 1) * head.in_dim];
            for (wij, &zi) in row.iter_mut().zip(z.iter()) {
                *wij -= lr * dlogit * zi;
            }
            head.b[c] -= lr * dlogit;
        }
    }

    /// Inner (online) adaptation: starting from the meta-learned PN head, replay
    /// the `(x_flat, y)` trajectory `inner_passes` times, taking one online SGD
    /// step per sample.  Returns the adapted head.
    ///
    /// # Errors
    /// Propagates dataset-validation errors from `check_dataset`.
    pub fn inner_adapt(&self, x_flat: &[f32], y: &[u32]) -> MetaResult<AnmlLinear> {
        let n = self.check_dataset(x_flat, y)?;
        // Precompute the (frozen) gated representations once.
        let reprs: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let x = &x_flat[i * self.cfg.in_dim..(i + 1) * self.cfg.in_dim];
                self.forward_repr(x).z
            })
            .collect();
        let mut head = self.pn_head.clone();
        for _ in 0..self.cfg.inner_passes {
            for (i, &lbl) in y.iter().enumerate() {
                self.head_sgd_step(&mut head, &reprs[i], lbl);
            }
        }
        Ok(head)
    }

    /// Mean cross-entropy of an explicit head over a `(x_flat, y)` dataset.
    ///
    /// # Errors
    /// Propagates dataset-validation errors and any cross-entropy error.
    pub fn dataset_loss(&self, head: &AnmlLinear, x_flat: &[f32], y: &[u32]) -> MetaResult<f32> {
        let n = self.check_dataset(x_flat, y)?;
        let mut logits = Vec::with_capacity(n * self.cfg.n_classes);
        for i in 0..n {
            let x = &x_flat[i * self.cfg.in_dim..(i + 1) * self.cfg.in_dim];
            let z = self.forward_repr(x).z;
            logits.extend_from_slice(&Self::head_logits(head, &z));
        }
        cross_entropy_loss(&logits, y, self.cfg.n_classes)
    }

    /// One ANML meta-update on a single task.
    ///
    /// 1. **Inner loop:** adapt the PN head on the `(traj_x, traj_y)` online
    ///    trajectory.
    /// 2. **Outer loop:** compute the remember-set loss `(remember_x,
    ///    remember_y)` under the *adapted* head, holding the slow encoder and the
    ///    gate at their current values, and back-propagate analytically to the
    ///    NM gate, the PN encoder and the PN head initialisation.  A first-order
    ///    (FOMAML) approximation treats the adapted head as a constant during
    ///    the outer back-prop (the head gradient is applied to the
    ///    *initialisation*).
    ///
    /// Returns the scalar remember-set loss measured *before* the meta-update.
    ///
    /// # Errors
    /// Propagates dataset-validation and cross-entropy errors.
    pub fn meta_update_task(
        &mut self,
        traj_x: &[f32],
        traj_y: &[u32],
        remember_x: &[f32],
        remember_y: &[u32],
    ) -> MetaResult<f32> {
        // Inner adaptation of the head on the online trajectory.
        let adapted_head = self.inner_adapt(traj_x, traj_y)?;
        let m = self.check_dataset(remember_x, remember_y)?;

        let in_dim = self.cfg.in_dim;
        let repr_dim = self.cfg.repr_dim;
        let n_classes = self.cfg.n_classes;

        // Accumulators for the meta-gradients of the slow parameters.
        let mut g_pn_enc_w = vec![0.0_f32; repr_dim * in_dim];
        let mut g_pn_enc_b = vec![0.0_f32; repr_dim];
        let mut g_nm_w = vec![0.0_f32; repr_dim * in_dim];
        let mut g_nm_b = vec![0.0_f32; repr_dim];
        let mut g_head_w = vec![0.0_f32; n_classes * repr_dim];
        let mut g_head_b = vec![0.0_f32; n_classes];

        let pre_loss = {
            let mut logits_all = Vec::with_capacity(m * n_classes);

            for i in 0..m {
                let x = &remember_x[i * in_dim..(i + 1) * in_dim];
                let fwd = self.forward_repr(x);
                let logits = Self::head_logits(&adapted_head, &fwd.z);
                logits_all.extend_from_slice(&logits);
                let probs = softmax(&logits);
                let label = remember_y[i] as usize;

                // dL/dlogit = (softmax − onehot) / m  (mean over remember set).
                let inv_m = 1.0 / m as f32;
                let mut dlogit = probs;
                dlogit[label] -= 1.0;
                for d in dlogit.iter_mut() {
                    *d *= inv_m;
                }

                // Head-initialisation gradient: (FOMAML) treat adapted head as a
                // function of the init; first-order term applies dlogit ⊗ z to the
                // init head directly.
                for c in 0..n_classes {
                    let hd = dlogit[c];
                    let row = &mut g_head_w[c * repr_dim..(c + 1) * repr_dim];
                    for (gw, &zi) in row.iter_mut().zip(fwd.z.iter()) {
                        *gw += hd * zi;
                    }
                    g_head_b[c] += hd;
                }

                // Back-prop into z through the *adapted* head weights.
                let mut dz = vec![0.0_f32; repr_dim];
                for (c, &hd) in dlogit.iter().enumerate() {
                    let row = &adapted_head.w[c * repr_dim..(c + 1) * repr_dim];
                    for (dzi, &wij) in dz.iter_mut().zip(row.iter()) {
                        *dzi += hd * wij;
                    }
                }

                // z = p ⊙ g  ⇒  dp = dz ⊙ g,  dg = dz ⊙ p.
                // p = ReLU(pn_pre)  ⇒  dpn_pre = dp · 1[pn_pre > 0].
                // g = sigmoid(nm_pre) ⇒ dnm_pre = dg · g · (1 − g).
                for k in 0..repr_dim {
                    let dp = dz[k] * fwd.g[k];
                    let dg = dz[k] * fwd.p[k];
                    let dpn_pre = if fwd.pn_pre[k] > 0.0 { dp } else { 0.0 };
                    let dnm_pre = dg * fwd.g[k] * (1.0 - fwd.g[k]);
                    // PN-encoder weight/bias gradients.
                    let pn_row = &mut g_pn_enc_w[k * in_dim..(k + 1) * in_dim];
                    for (gw, &xi) in pn_row.iter_mut().zip(x.iter()) {
                        *gw += dpn_pre * xi;
                    }
                    g_pn_enc_b[k] += dpn_pre;
                    // NM-gate weight/bias gradients.
                    let nm_row = &mut g_nm_w[k * in_dim..(k + 1) * in_dim];
                    for (gw, &xi) in nm_row.iter_mut().zip(x.iter()) {
                        *gw += dnm_pre * xi;
                    }
                    g_nm_b[k] += dnm_pre;
                }
            }
            cross_entropy_loss(&logits_all, remember_y, n_classes)?
        };

        // Apply the meta-update (gradient descent) to every slow parameter and
        // to the head initialisation.
        let lr = self.cfg.meta_lr;
        for (w, g) in self.pn_encoder.w.iter_mut().zip(g_pn_enc_w.iter()) {
            *w -= lr * g;
        }
        for (b, g) in self.pn_encoder.b.iter_mut().zip(g_pn_enc_b.iter()) {
            *b -= lr * g;
        }
        for (w, g) in self.nm_gate.w.iter_mut().zip(g_nm_w.iter()) {
            *w -= lr * g;
        }
        for (b, g) in self.nm_gate.b.iter_mut().zip(g_nm_b.iter()) {
            *b -= lr * g;
        }
        for (w, g) in self.pn_head.w.iter_mut().zip(g_head_w.iter()) {
            *w -= lr * g;
        }
        for (b, g) in self.pn_head.b.iter_mut().zip(g_head_b.iter()) {
            *b -= lr * g;
        }

        Ok(pre_loss)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> AnmlConfig {
        AnmlConfig {
            in_dim: 6,
            repr_dim: 8,
            n_classes: 3,
            inner_lr: 0.1,
            meta_lr: 0.05,
            inner_passes: 5,
        }
    }

    fn make() -> Anml {
        let mut rng = LcgRng::new(2026);
        Anml::new(cfg(), &mut rng).expect("valid ANML cfg")
    }

    /// Build a linearly separable n-class dataset: class c lives on the c-th
    /// coordinate axis with small jitter.
    fn make_dataset(
        n_per_class: usize,
        in_dim: usize,
        n_classes: usize,
        rng: &mut LcgRng,
    ) -> (Vec<f32>, Vec<u32>) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for c in 0..n_classes {
            for _ in 0..n_per_class {
                for j in 0..in_dim {
                    let base = if j == c { 1.0_f32 } else { 0.0_f32 };
                    x.push(base + (rng.next_f32() - 0.5) * 0.05);
                }
                y.push(c as u32);
            }
        }
        (x, y)
    }

    // ── construction validation ──────────────────────────────────────────────

    #[test]
    fn new_valid_succeeds() {
        let mut rng = LcgRng::new(1);
        assert!(Anml::new(cfg(), &mut rng).is_ok());
    }

    #[test]
    fn new_zero_in_dim_errs() {
        let mut c = cfg();
        c.in_dim = 0;
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            Anml::new(c, &mut rng),
            Err(MetaError::InvalidFeatDim { .. })
        ));
    }

    #[test]
    fn new_one_class_errs() {
        let mut c = cfg();
        c.n_classes = 1;
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            Anml::new(c, &mut rng),
            Err(MetaError::InvalidNWay { .. })
        ));
    }

    #[test]
    fn new_bad_lr_errs() {
        let mut c = cfg();
        c.inner_lr = 0.0;
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            Anml::new(c, &mut rng),
            Err(MetaError::InvalidLr { .. })
        ));
    }

    // ── forward / gate properties ────────────────────────────────────────────

    #[test]
    fn gate_in_unit_interval_and_representation_nonneg() {
        let net = make();
        let mut rng = LcgRng::new(7);
        let x: Vec<f32> = (0..net.config().in_dim)
            .map(|_| rng.next_f32() * 2.0 - 1.0)
            .collect();
        let fwd = net.forward_repr(&x);
        for &g in fwd.g.iter() {
            assert!((0.0..=1.0).contains(&g), "gate outside [0,1]: {g}");
        }
        // z = ReLU(p) ⊙ g, both factors ≥ 0, so z ≥ 0.
        for &z in fwd.z.iter() {
            assert!(z >= 0.0, "gated representation must be ≥ 0, got {z}");
        }
    }

    #[test]
    fn representation_wrong_dim_errs() {
        let net = make();
        assert!(matches!(
            net.representation(&[0.0; 2]),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    // ── inner adaptation reduces trajectory loss ─────────────────────────────

    #[test]
    fn inner_adapt_reduces_trajectory_loss() {
        let net = make();
        let mut rng = LcgRng::new(123);
        let (x, y) = make_dataset(4, net.config().in_dim, net.config().n_classes, &mut rng);
        let loss_before = net
            .dataset_loss(net.pn_head(), &x, &y)
            .expect("loss before");
        let adapted = net.inner_adapt(&x, &y).expect("inner adapt");
        let loss_after = net.dataset_loss(&adapted, &x, &y).expect("loss after");
        assert!(
            loss_after < loss_before,
            "online inner adaptation must reduce trajectory loss: {loss_before} -> {loss_after}"
        );
    }

    #[test]
    fn inner_adapt_separable_reaches_high_accuracy() {
        // Head-only online SGD over the (random, untrained) gated representation
        // is a linear classifier on `z`.  A separable trajectory is linearly
        // separable in `z`, but the few-step `inner_passes=5` default does not
        // give the head enough SGD steps to converge — 20 online passes drive it
        // to 100% on this seed (verified across lr ∈ {0.1,…,1.0}).  This mirrors
        // the real ANML inner loop, where the online phase replays the stream
        // many times before the remember-set is evaluated.
        let mut c = cfg();
        c.inner_passes = 20;
        let mut rng = LcgRng::new(2026);
        let net = Anml::new(c, &mut rng).expect("valid ANML cfg");
        let mut rng = LcgRng::new(321);
        let (x, y) = make_dataset(6, net.config().in_dim, net.config().n_classes, &mut rng);
        let adapted = net.inner_adapt(&x, &y).expect("inner adapt");
        let mut correct = 0usize;
        for (i, &lbl) in y.iter().enumerate() {
            let xi = &x[i * net.config().in_dim..(i + 1) * net.config().in_dim];
            let pred = net.predict_with(&adapted, xi).expect("predict");
            if pred == lbl {
                correct += 1;
            }
        }
        let acc = correct as f32 / y.len() as f32;
        assert!(
            acc >= 0.8,
            "adapted ANML head should classify a separable trajectory well, acc={acc}"
        );
    }

    #[test]
    fn inner_adapt_empty_errs() {
        let net = make();
        assert!(matches!(
            net.inner_adapt(&[], &[]),
            Err(MetaError::EmptySupport)
        ));
    }

    #[test]
    fn inner_adapt_bad_label_errs() {
        let net = make();
        let x = vec![0.0_f32; net.config().in_dim];
        let y = vec![99_u32];
        assert!(matches!(
            net.inner_adapt(&x, &y),
            Err(MetaError::Internal { .. })
        ));
    }

    // ── meta-update reduces remember-set loss ────────────────────────────────

    #[test]
    fn meta_update_reduces_remember_loss_over_iterations() {
        let mut net = make();
        let mut rng = LcgRng::new(2024);
        let in_dim = net.config().in_dim;
        let n_classes = net.config().n_classes;
        let (traj_x, traj_y) = make_dataset(3, in_dim, n_classes, &mut rng);
        let (rem_x, rem_y) = make_dataset(4, in_dim, n_classes, &mut rng);

        let first = net
            .meta_update_task(&traj_x, &traj_y, &rem_x, &rem_y)
            .expect("meta update 1");
        let mut last = first;
        for _ in 0..40 {
            last = net
                .meta_update_task(&traj_x, &traj_y, &rem_x, &rem_y)
                .expect("meta update");
        }
        assert!(
            last < first,
            "repeated ANML meta-updates must reduce the remember-set loss: {first} -> {last}"
        );
        assert!(last.is_finite());
    }

    #[test]
    fn meta_update_changes_nm_and_encoder() {
        let mut net = make();
        let mut rng = LcgRng::new(55);
        let in_dim = net.config().in_dim;
        let n_classes = net.config().n_classes;
        let (traj_x, traj_y) = make_dataset(3, in_dim, n_classes, &mut rng);
        let (rem_x, rem_y) = make_dataset(3, in_dim, n_classes, &mut rng);
        let nm_before = net.nm_gate().weights().to_vec();
        let enc_before = net.pn_encoder().weights().to_vec();
        net.meta_update_task(&traj_x, &traj_y, &rem_x, &rem_y)
            .expect("meta update");
        let nm_after = net.nm_gate().weights().to_vec();
        let enc_after = net.pn_encoder().weights().to_vec();
        assert_ne!(
            nm_before, nm_after,
            "NM gate must move under the meta-update"
        );
        assert_ne!(
            enc_before, enc_after,
            "PN encoder must move under the meta-update"
        );
    }

    #[test]
    fn meta_update_deterministic_with_seed() {
        let in_dim = 6;
        let n_classes = 3;
        let mut data_rng = LcgRng::new(9);
        let (traj_x, traj_y) = make_dataset(3, in_dim, n_classes, &mut data_rng);
        let (rem_x, rem_y) = make_dataset(3, in_dim, n_classes, &mut data_rng);

        let mut rng_a = LcgRng::new(2026);
        let mut net_a = Anml::new(cfg(), &mut rng_a).expect("a");
        let mut rng_b = LcgRng::new(2026);
        let mut net_b = Anml::new(cfg(), &mut rng_b).expect("b");
        let la = net_a
            .meta_update_task(&traj_x, &traj_y, &rem_x, &rem_y)
            .expect("a update");
        let lb = net_b
            .meta_update_task(&traj_x, &traj_y, &rem_x, &rem_y)
            .expect("b update");
        assert_eq!(la, lb);
        assert_eq!(net_a.nm_gate().weights(), net_b.nm_gate().weights());
    }
}
