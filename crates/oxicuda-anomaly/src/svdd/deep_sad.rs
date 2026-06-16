//! DeepSAD — Deep Semi-Supervised Anomaly Detection (Ruff et al. 2020).
//!
//! DeepSAD generalises DeepSVDD to the semi-supervised setting. A compact MLP
//! `φ(·; W)` maps inputs into a feature space; a fixed centre `c` defines a
//! hypersphere. Unlabeled points and labeled-normal points (`η = +1`) are
//! pulled toward `c`, while labeled anomalies (`η = −1`) are pushed away via an
//! **inverse-distance** term:
//!
//! ```text
//! L(W) = (1/N) Σ_i ℓ_i ,   ℓ_i = ‖φ(x_i) − c‖²            if η_i = +1
//!                          ℓ_i = η · 1 / (‖φ(x_i) − c‖² + ε)  if η_i = −1
//! ```
//!
//! Minimising `ℓ_i` for `η = +1` shrinks the distance (pull in); minimising the
//! inverse term for `η = −1` grows the distance (push out). The anomaly score
//! is the squared distance to the centre, `‖φ(x) − c‖²` — larger ⇒ more
//! anomalous.
//!
//! The encoder uses ReLU on hidden layers and a **linear, bias-free output
//! layer** (the standard DeepSVDD/DeepSAD convention to avoid hypersphere
//! collapse). Gradients are computed by explicit backpropagation and applied
//! with full-batch gradient descent.
//!
//! # Reference
//! Ruff, L., Vandermeulen, R. A., Görnitz, N., Binder, A., Müller, E.,
//! Müller, K.-R., & Kloft, M. (2020). *Deep Semi-Supervised Anomaly
//! Detection*. ICLR 2020.

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

/// Numerical floor inside the inverse-distance term.
const INV_EPS: f32 = 1e-3;
/// Per-sample gradient-norm clip (safety net against the inverse term exploding).
const GRAD_CLIP: f32 = 1.0e3;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Hyper-parameters for [`DeepSad`].
#[derive(Debug, Clone)]
pub struct DeepSadConfig {
    /// Layer dimensions `[input, h1, …, rep_dim]` (length `≥ 2`).
    pub dims: Vec<usize>,
    /// Gradient-descent learning rate (default 0.01).
    pub learning_rate: f32,
    /// Weight `η` applied to the labeled-anomaly (push-out) term (default 1.0).
    pub eta: f32,
    /// L2 weight decay applied to encoder weights (default 0.0).
    pub weight_decay: f32,
    /// RNG seed for weight initialisation (default 42).
    pub seed: u64,
}

impl DeepSadConfig {
    /// Configuration with the given layer spec and default optimiser settings.
    #[must_use]
    pub fn new(dims: &[usize]) -> Self {
        Self {
            dims: dims.to_vec(),
            learning_rate: 0.01,
            eta: 1.0,
            weight_decay: 0.0,
            seed: 42,
        }
    }
}

// ─── Encoder (MLP with explicit backprop) ─────────────────────────────────────

/// Xavier-uniform initialisation for a `fan_out × fan_in` weight matrix.
fn xavier_init(fan_in: usize, fan_out: usize, rng: &mut LcgRng) -> Vec<f32> {
    let limit = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
    (0..fan_in * fan_out)
        .map(|_| rng.next_f32() * 2.0 * limit - limit)
        .collect()
}

/// Compact MLP encoder with cached-forward / backward for DeepSAD training.
#[derive(Debug, Clone)]
struct SadEncoder {
    /// Layer dimensions including input and output.
    dims: Vec<usize>,
    /// Per-layer weights `[out * in]`, row-major.
    weights: Vec<Vec<f32>>,
    /// Per-layer biases `[out]`. The last (output) layer's bias stays zero.
    biases: Vec<Vec<f32>>,
}

impl SadEncoder {
    fn new(dims: &[usize], rng: &mut LcgRng) -> AnomalyResult<Self> {
        if dims.len() < 2 {
            return Err(AnomalyError::InvalidLayerDims {
                msg: "need at least [input_dim, rep_dim]".into(),
            });
        }
        for &d in dims {
            if d == 0 {
                return Err(AnomalyError::InvalidLayerDims {
                    msg: "zero dimension in layer spec".into(),
                });
            }
        }
        let n_layers = dims.len() - 1;
        let mut weights = Vec::with_capacity(n_layers);
        let mut biases = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            weights.push(xavier_init(dims[l], dims[l + 1], rng));
            biases.push(vec![0.0_f32; dims[l + 1]]);
        }
        Ok(Self {
            dims: dims.to_vec(),
            weights,
            biases,
        })
    }

    #[inline]
    fn n_layers(&self) -> usize {
        self.weights.len()
    }

    /// Plain forward pass (no caching) returning the representation `φ(x)`.
    fn forward(&self, x: &[f32]) -> AnomalyResult<Vec<f32>> {
        if x.len() != self.dims[0] {
            return Err(AnomalyError::DimensionMismatch {
                expected: self.dims[0],
                got: x.len(),
            });
        }
        let n_layers = self.n_layers();
        let mut activation = x.to_vec();
        for layer in 0..n_layers {
            let in_dim = self.dims[layer];
            let out_dim = self.dims[layer + 1];
            let w = &self.weights[layer];
            let b = &self.biases[layer];
            let mut out = vec![0.0_f32; out_dim];
            for o in 0..out_dim {
                let mut acc = b[o];
                for i in 0..in_dim {
                    acc += w[o * in_dim + i] * activation[i];
                }
                out[o] = if layer < n_layers - 1 {
                    acc.max(0.0)
                } else {
                    acc
                };
            }
            activation = out;
        }
        Ok(activation)
    }

    /// Forward pass caching per-layer inputs (`acts`) and pre-activations (`pre`).
    ///
    /// `acts[0] = x`, `acts[layer+1]` is the output of `layer`; `pre[layer]` is
    /// the pre-activation of `layer`. The representation is `acts[n_layers]`.
    fn forward_cache(&self, x: &[f32]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let n_layers = self.n_layers();
        let mut acts: Vec<Vec<f32>> = Vec::with_capacity(n_layers + 1);
        let mut pre: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
        acts.push(x.to_vec());
        for layer in 0..n_layers {
            let in_dim = self.dims[layer];
            let out_dim = self.dims[layer + 1];
            let w = &self.weights[layer];
            let b = &self.biases[layer];
            let a_prev = &acts[layer];
            let mut z = vec![0.0_f32; out_dim];
            let mut a = vec![0.0_f32; out_dim];
            for o in 0..out_dim {
                let mut acc = b[o];
                for i in 0..in_dim {
                    acc += w[o * in_dim + i] * a_prev[i];
                }
                z[o] = acc;
                a[o] = if layer < n_layers - 1 {
                    acc.max(0.0)
                } else {
                    acc
                };
            }
            pre.push(z);
            acts.push(a);
        }
        (acts, pre)
    }

    /// Backpropagate `grad_out = ∂ℓ/∂φ` into the gradient accumulators.
    fn backward(
        &self,
        acts: &[Vec<f32>],
        pre: &[Vec<f32>],
        grad_out: &[f32],
        grad_w: &mut [Vec<f32>],
        grad_b: &mut [Vec<f32>],
    ) {
        let n_layers = self.n_layers();
        // `delta` is ∂ℓ/∂z at the current layer (output layer is linear).
        let mut delta = grad_out.to_vec();
        for layer in (0..n_layers).rev() {
            let in_dim = self.dims[layer];
            let out_dim = self.dims[layer + 1];
            let a_prev = &acts[layer];

            {
                let gw = &mut grad_w[layer];
                let gb = &mut grad_b[layer];
                for o in 0..out_dim {
                    let go = delta[o];
                    let base = o * in_dim;
                    for i in 0..in_dim {
                        gw[base + i] += go * a_prev[i];
                    }
                    // The output layer keeps a fixed zero bias (collapse guard).
                    if layer < n_layers - 1 {
                        gb[o] += go;
                    }
                }
            }

            if layer > 0 {
                let w = &self.weights[layer];
                let pre_prev = &pre[layer - 1];
                let mut new_delta = vec![0.0_f32; in_dim];
                for i in 0..in_dim {
                    let mut s = 0.0_f32;
                    for o in 0..out_dim {
                        s += w[o * in_dim + i] * delta[o];
                    }
                    // ReLU derivative of the previous layer.
                    new_delta[i] = if pre_prev[i] > 0.0 { s } else { 0.0 };
                }
                delta = new_delta;
            }
        }
    }
}

// ─── DeepSad ──────────────────────────────────────────────────────────────────

/// DeepSAD semi-supervised anomaly detector.
#[derive(Debug, Clone)]
pub struct DeepSad {
    config: DeepSadConfig,
    encoder: SadEncoder,
    center: Option<Vec<f32>>,
    /// Input dimensionality (`dims[0]`).
    pub input_dim: usize,
    /// Representation dimensionality (`dims[last]`).
    pub rep_dim: usize,
}

impl DeepSad {
    /// Build a DeepSAD model from `config`, initialising encoder weights.
    ///
    /// # Errors
    /// [`AnomalyError::InvalidLayerDims`] if `dims` is malformed.
    pub fn new(config: DeepSadConfig) -> AnomalyResult<Self> {
        let mut rng = LcgRng::new(config.seed);
        let encoder = SadEncoder::new(&config.dims, &mut rng)?;
        let input_dim = config.dims[0];
        let rep_dim = config.dims[config.dims.len() - 1];
        Ok(Self {
            config,
            encoder,
            center: None,
            input_dim,
            rep_dim,
        })
    }

    /// Encoder feature representation `φ(x)`.
    ///
    /// # Errors
    /// [`AnomalyError::DimensionMismatch`] if `x.len() != input_dim`.
    pub fn forward(&self, x: &[f32]) -> AnomalyResult<Vec<f32>> {
        self.encoder.forward(x)
    }

    /// Validate batch shapes shared by `fit` / `sad_loss`.
    fn validate(&self, x: &[f32], n_samples: usize, labels: &[f32]) -> AnomalyResult<()> {
        if n_samples == 0 {
            return Err(AnomalyError::EmptyInput);
        }
        if x.len() != n_samples * self.input_dim {
            return Err(AnomalyError::DimensionMismatch {
                expected: n_samples * self.input_dim,
                got: x.len(),
            });
        }
        if labels.len() != n_samples {
            return Err(AnomalyError::DimensionMismatch {
                expected: n_samples,
                got: labels.len(),
            });
        }
        Ok(())
    }

    /// Initialise the fixed centre `c` as the mean representation of the
    /// `η = +1` (unlabeled / normal) samples; falls back to all samples if none
    /// are labeled normal. Near-zero components are nudged to `0.01`.
    fn init_center(&mut self, x: &[f32], labels: &[f32]) -> AnomalyResult<()> {
        let mut center = vec![0.0_f32; self.rep_dim];
        let mut count = 0_usize;
        for (s, &eta) in labels.iter().enumerate() {
            if eta >= 0.0 {
                let xi = &x[s * self.input_dim..(s + 1) * self.input_dim];
                let rep = self.encoder.forward(xi)?;
                for (ck, rk) in center.iter_mut().zip(rep.iter()) {
                    *ck += rk;
                }
                count += 1;
            }
        }
        if count == 0 {
            for (s, _) in labels.iter().enumerate() {
                let xi = &x[s * self.input_dim..(s + 1) * self.input_dim];
                let rep = self.encoder.forward(xi)?;
                for (ck, rk) in center.iter_mut().zip(rep.iter()) {
                    *ck += rk;
                }
            }
            count = labels.len();
        }
        let inv = 1.0 / count as f32;
        for ck in &mut center {
            *ck *= inv;
            if ck.abs() < 0.01 {
                *ck = 0.01;
            }
        }
        self.center = Some(center);
        Ok(())
    }

    /// Mean SAD loss `L(W)` over a batch (no parameter update).
    ///
    /// # Errors
    /// [`AnomalyError::NotFitted`] if the centre is uninitialised, plus the
    /// shape errors from `DeepSad::validate`.
    pub fn sad_loss(&self, x: &[f32], n_samples: usize, labels: &[f32]) -> AnomalyResult<f32> {
        self.validate(x, n_samples, labels)?;
        let c = self.center.as_ref().ok_or(AnomalyError::NotFitted)?;
        let mut total = 0.0_f32;
        for (s, &eta) in labels.iter().enumerate() {
            let xi = &x[s * self.input_dim..(s + 1) * self.input_dim];
            let rep = self.encoder.forward(xi)?;
            let dsq: f32 = rep
                .iter()
                .zip(c.iter())
                .map(|(r, ck)| (r - ck).powi(2))
                .sum();
            total += if eta >= 0.0 {
                dsq
            } else {
                self.config.eta / (dsq + INV_EPS)
            };
        }
        Ok(total / n_samples as f32)
    }

    /// One full-batch gradient-descent step; returns the loss **before** the
    /// update (so a sequence of calls yields a descending loss trace).
    fn train_step(&mut self, x: &[f32], n_samples: usize, labels: &[f32]) -> AnomalyResult<f32> {
        let c = self.center.clone().ok_or(AnomalyError::NotFitted)?;
        let rep_dim = self.rep_dim;
        let in_dim = self.input_dim;

        let mut grad_w: Vec<Vec<f32>> = self
            .encoder
            .weights
            .iter()
            .map(|w| vec![0.0_f32; w.len()])
            .collect();
        let mut grad_b: Vec<Vec<f32>> = self
            .encoder
            .biases
            .iter()
            .map(|b| vec![0.0_f32; b.len()])
            .collect();

        let mut total_loss = 0.0_f32;
        for (s, &eta) in labels.iter().enumerate() {
            let xi = &x[s * in_dim..(s + 1) * in_dim];
            let (acts, pre) = self.encoder.forward_cache(xi);
            let phi = &acts[self.encoder.n_layers()];

            let mut diff = vec![0.0_f32; rep_dim];
            let mut dsq = 0.0_f32;
            for ((dk, pk), ck) in diff.iter_mut().zip(phi.iter()).zip(c.iter()) {
                let d = pk - ck;
                *dk = d;
                dsq += d * d;
            }

            let mut grad_out = vec![0.0_f32; rep_dim];
            if eta >= 0.0 {
                // Pull in: ℓ = ‖φ − c‖², ∂ℓ/∂φ = 2 (φ − c).
                total_loss += dsq;
                for (gk, dk) in grad_out.iter_mut().zip(diff.iter()) {
                    *gk = 2.0 * dk;
                }
            } else {
                // Push out: ℓ = η / (‖φ − c‖² + ε), ∂ℓ/∂φ = −2η (φ − c)/(·)².
                let denom = dsq + INV_EPS;
                total_loss += self.config.eta / denom;
                let coef = -2.0 * self.config.eta / (denom * denom);
                for (gk, dk) in grad_out.iter_mut().zip(diff.iter()) {
                    *gk = coef * dk;
                }
            }

            clip_gradient(&mut grad_out, GRAD_CLIP);
            self.encoder
                .backward(&acts, &pre, &grad_out, &mut grad_w, &mut grad_b);
        }

        // Average gradients and apply the SGD update.
        let inv_n = 1.0 / n_samples as f32;
        let lr = self.config.learning_rate;
        let wd = self.config.weight_decay;
        let n_layers = self.encoder.n_layers();
        for layer in 0..n_layers {
            let w = &mut self.encoder.weights[layer];
            let gw = &grad_w[layer];
            for (wj, gj) in w.iter_mut().zip(gw.iter()) {
                *wj -= lr * (gj * inv_n + wd * *wj);
            }
            if layer < n_layers - 1 {
                let b = &mut self.encoder.biases[layer];
                let gb = &grad_b[layer];
                for (bj, gj) in b.iter_mut().zip(gb.iter()) {
                    *bj -= lr * gj * inv_n;
                }
            }
        }

        Ok(total_loss * inv_n)
    }

    /// Fit DeepSAD for `n_steps` gradient-descent steps.
    ///
    /// `x` is `[n_samples × input_dim]` row-major; `labels[i] ∈ {+1, −1}`
    /// (`+1` = unlabeled / normal, `−1` = labeled anomaly). The centre is
    /// initialised on the first call. Returns the loss trace of length
    /// `n_steps + 1` (loss before each step plus the final loss).
    ///
    /// # Errors
    /// Shape errors from `DeepSad::validate`.
    pub fn fit(
        &mut self,
        x: &[f32],
        n_samples: usize,
        labels: &[f32],
        n_steps: usize,
    ) -> AnomalyResult<Vec<f32>> {
        self.validate(x, n_samples, labels)?;
        if self.center.is_none() {
            self.init_center(x, labels)?;
        }
        let mut history = Vec::with_capacity(n_steps + 1);
        for _ in 0..n_steps {
            history.push(self.train_step(x, n_samples, labels)?);
        }
        history.push(self.sad_loss(x, n_samples, labels)?);
        Ok(history)
    }

    /// Anomaly score `‖φ(x) − c‖²` (squared distance to the centre).
    ///
    /// # Errors
    /// [`AnomalyError::NotFitted`] if the centre is uninitialised;
    /// [`AnomalyError::DimensionMismatch`] if `x.len() != input_dim`.
    pub fn score(&self, x: &[f32]) -> AnomalyResult<f32> {
        let c = self.center.as_ref().ok_or(AnomalyError::NotFitted)?;
        let rep = self.encoder.forward(x)?;
        Ok(rep
            .iter()
            .zip(c.iter())
            .map(|(r, ck)| (r - ck).powi(2))
            .sum())
    }

    /// Batch scoring; `x` is `[n × input_dim]`, returns `[n]`.
    ///
    /// # Errors
    /// [`AnomalyError::NotFitted`] / [`AnomalyError::DimensionMismatch`].
    pub fn score_batch(&self, x: &[f32], n: usize) -> AnomalyResult<Vec<f32>> {
        let c = self.center.as_ref().ok_or(AnomalyError::NotFitted)?;
        if x.len() != n * self.input_dim {
            return Err(AnomalyError::DimensionMismatch {
                expected: n * self.input_dim,
                got: x.len(),
            });
        }
        let mut scores = Vec::with_capacity(n);
        for i in 0..n {
            let xi = &x[i * self.input_dim..(i + 1) * self.input_dim];
            let rep = self.encoder.forward(xi)?;
            let s: f32 = rep
                .iter()
                .zip(c.iter())
                .map(|(r, ck)| (r - ck).powi(2))
                .sum();
            scores.push(s);
        }
        Ok(scores)
    }

    /// Whether the centre has been initialised (i.e. `fit` has run).
    #[inline]
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.center.is_some()
    }
}

/// Scale a gradient vector in place so its L2 norm does not exceed `max_norm`.
fn clip_gradient(g: &mut [f32], max_norm: f32) {
    let norm = g.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > max_norm && norm > 0.0 {
        let scale = max_norm / norm;
        for v in g.iter_mut() {
            *v *= scale;
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Toy semi-supervised batch: `n_normal` normals near `+1`, `n_anom`
    /// anomalies near `−1`, with matching `±1` labels.
    fn toy_batch(dim: usize, n_normal: usize, n_anom: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut rng = LcgRng::new(seed);
        let mut x = Vec::new();
        let mut labels = Vec::new();
        for _ in 0..n_normal {
            for _ in 0..dim {
                x.push(1.0 + rng.next_f32() * 0.1);
            }
            labels.push(1.0_f32);
        }
        for _ in 0..n_anom {
            for _ in 0..dim {
                x.push(-1.0 - rng.next_f32() * 0.1);
            }
            labels.push(-1.0_f32);
        }
        (x, labels)
    }

    // ── Test (a): loss decreases over a few training steps ────────────────────
    #[test]
    fn loss_decreases_during_training() {
        let cfg = DeepSadConfig {
            dims: vec![3, 6, 2],
            learning_rate: 0.02,
            eta: 1.0,
            weight_decay: 0.0,
            seed: 1,
        };
        let mut model = DeepSad::new(cfg).expect("new");
        let (x, labels) = toy_batch(3, 8, 2, 10);
        let history = model.fit(&x, labels.len(), &labels, 30).expect("fit");
        assert!(
            history.iter().all(|l| l.is_finite()),
            "loss must stay finite"
        );
        assert!(
            history
                .last()
                .expect("loss history should have a final entry")
                < history
                    .first()
                    .expect("loss history should have an initial entry"),
            "final loss {:?} should be below initial loss {:?}",
            history.last(),
            history.first()
        );
    }

    // ── Test (b): labeled anomalies end up farther from the centre ────────────
    #[test]
    fn anomalies_score_higher_after_training() {
        let cfg = DeepSadConfig {
            dims: vec![3, 8, 2],
            learning_rate: 0.03,
            eta: 1.0,
            weight_decay: 0.0,
            seed: 2,
        };
        let mut model = DeepSad::new(cfg).expect("new");
        let (x, labels) = toy_batch(3, 10, 4, 20);
        model.fit(&x, labels.len(), &labels, 60).expect("fit");

        let normal_pt = [1.0_f32, 1.0, 1.0];
        let anom_pt = [-1.0_f32, -1.0, -1.0];
        let s_normal = model.score(&normal_pt).expect("score normal");
        let s_anom = model.score(&anom_pt).expect("score anom");
        assert!(
            s_anom > s_normal,
            "anomaly score {s_anom} should exceed normal score {s_normal}"
        );
    }

    // ── Test (c): scores are finite ───────────────────────────────────────────
    #[test]
    fn scores_finite() {
        let cfg = DeepSadConfig::new(&[4, 8, 3]);
        let mut model = DeepSad::new(cfg).expect("new");
        let (x, labels) = toy_batch(4, 6, 3, 30);
        model.fit(&x, labels.len(), &labels, 15).expect("fit");
        for q in &[[0.5_f32, 0.5, 0.5, 0.5], [9.0, -9.0, 3.0, -3.0], [0.0; 4]] {
            let s = model.score(q).expect("score");
            assert!(s.is_finite() && s >= 0.0, "score={s}");
        }
    }

    // ── Test (d): the η sign controls the gradient direction ──────────────────
    #[test]
    fn eta_sign_controls_gradient_direction() {
        // Centre is fixed from a batch of normals; `p` is a *different* point so
        // its distance to the centre is non-zero (otherwise the gradient is 0).
        let (x_train, labels_train) = toy_batch(3, 6, 0, 7);
        let p = [0.6_f32, 0.4, 0.7];

        let make = || {
            let mut m = DeepSad::new(DeepSadConfig {
                dims: vec![3, 6, 2],
                learning_rate: 0.05,
                eta: 1.0,
                weight_decay: 0.0,
                seed: 5,
            })
            .expect("new");
            m.fit(&x_train, labels_train.len(), &labels_train, 0)
                .expect("init centre");
            m
        };

        // η = +1 → one step pulls φ(p) toward the centre (distance shrinks).
        let mut pull = make();
        let d_before_pull = pull.score(&p).expect("score");
        pull.train_step(&p, 1, &[1.0]).expect("pull step");
        let d_after_pull = pull.score(&p).expect("score");
        assert!(
            d_after_pull < d_before_pull,
            "η=+1 should shrink distance: {d_before_pull} → {d_after_pull}"
        );

        // η = −1 → one step pushes φ(p) away from the centre (distance grows).
        let mut push = make();
        let d_before_push = push.score(&p).expect("score");
        push.train_step(&p, 1, &[-1.0]).expect("push step");
        let d_after_push = push.score(&p).expect("score");
        assert!(
            d_after_push > d_before_push,
            "η=−1 should grow distance: {d_before_push} → {d_after_push}"
        );
    }

    // ── Test (e): empty / dimension-mismatch errors ───────────────────────────
    #[test]
    fn empty_and_dim_mismatch_errors() {
        let mut model = DeepSad::new(DeepSadConfig::new(&[3, 4, 2])).expect("new");

        // n = 0 → EmptyInput
        assert!(matches!(
            model.fit(&[], 0, &[], 1),
            Err(AnomalyError::EmptyInput)
        ));

        // x length inconsistent with n_samples * input_dim
        assert!(matches!(
            model.fit(&[1.0, 2.0, 3.0, 4.0], 2, &[1.0, 1.0], 1),
            Err(AnomalyError::DimensionMismatch { .. })
        ));

        // labels length mismatch
        assert!(matches!(
            model.fit(&[1.0, 2.0, 3.0], 1, &[1.0, -1.0], 1),
            Err(AnomalyError::DimensionMismatch { .. })
        ));

        // score before fit → NotFitted
        assert!(matches!(
            model.score(&[0.0, 0.0, 0.0]),
            Err(AnomalyError::NotFitted)
        ));

        // malformed encoder spec → InvalidLayerDims
        assert!(matches!(
            DeepSad::new(DeepSadConfig::new(&[4])),
            Err(AnomalyError::InvalidLayerDims { .. })
        ));
    }
}
