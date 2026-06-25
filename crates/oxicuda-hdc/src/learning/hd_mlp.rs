//! Hyperdimensional + neural-network hybrid: HD features as input to a small MLP head.
//!
//! This module realises the **HD-NN hybrid** family of models (Imani et al. 2019;
//! Hernández-Cano's `NeuralHD` / `HD2FPGA` lines), where a high-dimensional binary
//! hypervector is no longer consumed by a *linear* prototype/centroid classifier but is
//! instead treated as a fixed **feature vector** that feeds a compact, fully trainable
//! single-hidden-layer multilayer perceptron (MLP). The HD *encoder* (random projection,
//! n-gram binding, record encoding, …) lives elsewhere in the crate and produces the
//! crate-standard binary `Vec<i8>` in `{−1, +1}`; here that vector — optionally already
//! sub-sampled or projected down to a smaller `input_dim` — is cast element-wise to
//! `±1.0` `f32` and pushed through the network.
//!
//! Unlike the linear HD classifiers (a single dot-product against one prototype per
//! class), the hidden layer with a `tanh` non-linearity lets the head learn *non-linear*
//! decision boundaries over the HD feature space, which is the entire motivation of the
//! hybrid: keep HD's cheap, robust, high-dimensional encoding for the front-end and bolt
//! a tiny back-prop-trained head onto it for accuracy.
//!
//! # Network architecture
//!
//! ```text
//!   x  ∈ ℝ^{input_dim}      (HD hypervector cast ±1 → ±1.0)
//!   │
//!   │  W1 ∈ ℝ^{hidden_dim × input_dim},  b1 ∈ ℝ^{hidden_dim}
//!   ▼
//!   z1 = W1·x + b1
//!   h  = tanh(z1)           ∈ ℝ^{hidden_dim}
//!   │
//!   │  W2 ∈ ℝ^{n_classes × hidden_dim},  b2 ∈ ℝ^{n_classes}
//!   ▼
//!   logits = W2·h + b2      ∈ ℝ^{n_classes}
//!   p      = softmax(logits)
//! ```
//!
//! # Weight layout
//!
//! All parameters are stored as flat, **row-major** `Vec<f32>`:
//! - `w1[i * input_dim + j]`  — weight from input `j` to hidden unit `i`.
//! - `b1[i]`                  — bias of hidden unit `i`.
//! - `w2[k * hidden_dim + i]` — weight from hidden unit `i` to class logit `k`.
//! - `b2[k]`                  — bias of class logit `k`.
//!
//! # Training & gradient math
//!
//! Training minimises the multinomial **cross-entropy** loss with a numerically stable
//! softmax, by plain stochastic gradient descent (one parameter update per sample). For a
//! sample with one-hot target `t` (true class `c`) the gradients are derived by hand:
//!
//! ```text
//!   ∂L/∂logits = p − t                       (softmax+cross-entropy collapse)
//!   ∂L/∂W2     = (p − t) ⊗ hᵀ                 (outer product)
//!   ∂L/∂b2     =  p − t
//!   ∂L/∂h      = W2ᵀ · (p − t)
//!   ∂L/∂z1     = (∂L/∂h) ⊙ (1 − h²)           (tanh derivative: 1 − tanh² )
//!   ∂L/∂W1     = (∂L/∂z1) ⊗ xᵀ
//!   ∂L/∂b1     =  ∂L/∂z1
//! ```
//!
//! and the SGD step subtracts `learning_rate ·` each gradient. Weights are initialised
//! from a small Gaussian (`normal_pair_f32`, Box–Muller) scaled by `1/√fan_in`
//! (He/Xavier-style); biases start at zero. Everything is deterministic from
//! [`HdMlpConfig::seed`].

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;

/// Configuration for an [`HdMlp`] hybrid HD-feature → MLP classifier.
#[derive(Debug, Clone)]
pub struct HdMlpConfig {
    /// Dimensionality of the HD feature vector fed to the network (`> 0`).
    pub input_dim: usize,
    /// Width of the single hidden layer (`> 0`).
    pub hidden_dim: usize,
    /// Number of output classes (`≥ 2`).
    pub n_classes: usize,
    /// SGD learning rate (finite and `> 0`).
    pub learning_rate: f32,
    /// Number of training epochs over the data set (`≥ 1`).
    pub epochs: usize,
    /// Seed for deterministic weight initialisation and per-epoch shuffling.
    pub seed: u64,
}

impl Default for HdMlpConfig {
    fn default() -> Self {
        Self {
            input_dim: 256,
            hidden_dim: 32,
            n_classes: 2,
            learning_rate: 0.05,
            epochs: 50,
            seed: 0,
        }
    }
}

impl HdMlpConfig {
    /// Build and validate a configuration.
    ///
    /// # Errors
    ///
    /// - [`HdcError::ZeroDimension`] if `input_dim == 0` or `hidden_dim == 0`.
    /// - [`HdcError::EmptyInput`] if `n_classes < 2` or `epochs == 0`.
    /// - [`HdcError::InvalidProbability`] (reused) if `learning_rate` is non-finite or `≤ 0`.
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        n_classes: usize,
        learning_rate: f32,
        epochs: usize,
        seed: u64,
    ) -> HdcResult<Self> {
        if input_dim == 0 || hidden_dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        if n_classes < 2 {
            return Err(HdcError::EmptyInput);
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(HdcError::InvalidProbability(learning_rate as f64));
        }
        if epochs == 0 {
            return Err(HdcError::EmptyInput);
        }
        Ok(Self {
            input_dim,
            hidden_dim,
            n_classes,
            learning_rate,
            epochs,
            seed,
        })
    }
}

/// Single-hidden-layer MLP classifier operating on HD feature hypervectors.
///
/// See the [module documentation](self) for the architecture, the flat row-major weight
/// layout, and the hand-derived back-propagation gradients.
pub struct HdMlp {
    cfg: HdMlpConfig,
    /// First-layer weights, row-major `[hidden_dim × input_dim]`.
    w1: Vec<f32>,
    /// First-layer biases, length `hidden_dim`.
    b1: Vec<f32>,
    /// Second-layer weights, row-major `[n_classes × hidden_dim]`.
    w2: Vec<f32>,
    /// Second-layer biases, length `n_classes`.
    b2: Vec<f32>,
}

impl HdMlp {
    /// Create a new network with Gaussian-initialised weights and zero biases.
    ///
    /// Weights of each layer are drawn from `N(0, 1)` via the Box–Muller
    /// [`LcgRng::normal_pair_f32`] and scaled by `1/√fan_in` (the input width of that
    /// layer), giving a He/Xavier-style initialisation. The whole process is deterministic
    /// from `cfg.seed`.
    ///
    /// # Errors
    ///
    /// Propagates any validation error from the contained [`HdMlpConfig`]; a config built
    /// via [`HdMlpConfig::new`] is already valid, so this only fails if the fields were set
    /// directly to invalid values.
    pub fn new(cfg: HdMlpConfig) -> HdcResult<Self> {
        if cfg.input_dim == 0 || cfg.hidden_dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        if cfg.n_classes < 2 {
            return Err(HdcError::EmptyInput);
        }
        if !cfg.learning_rate.is_finite() || cfg.learning_rate <= 0.0 {
            return Err(HdcError::InvalidProbability(cfg.learning_rate as f64));
        }
        if cfg.epochs == 0 {
            return Err(HdcError::EmptyInput);
        }

        let mut rng = LcgRng::new(cfg.seed);

        let scale1 = 1.0f32 / (cfg.input_dim as f32).sqrt();
        let w1 = Self::init_layer(&mut rng, cfg.hidden_dim * cfg.input_dim, scale1);
        let b1 = vec![0.0f32; cfg.hidden_dim];

        let scale2 = 1.0f32 / (cfg.hidden_dim as f32).sqrt();
        let w2 = Self::init_layer(&mut rng, cfg.n_classes * cfg.hidden_dim, scale2);
        let b2 = vec![0.0f32; cfg.n_classes];

        Ok(Self {
            cfg,
            w1,
            b1,
            w2,
            b2,
        })
    }

    /// Draw `count` Gaussian weights scaled by `scale`, consuming the RNG two-at-a-time.
    fn init_layer(rng: &mut LcgRng, count: usize, scale: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(count);
        while out.len() < count {
            let (a, b) = rng.normal_pair_f32();
            out.push(a * scale);
            if out.len() < count {
                out.push(b * scale);
            }
        }
        out
    }

    /// HD feature dimension expected at the input layer.
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.cfg.input_dim
    }

    /// Width of the hidden layer.
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.cfg.hidden_dim
    }

    /// Number of output classes.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.cfg.n_classes
    }

    /// Validate a hypervector's length and cast it to `±1.0` `f32` features.
    fn features(&self, hv: &[i8]) -> HdcResult<Vec<f32>> {
        if hv.len() != self.cfg.input_dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.cfg.input_dim,
                got: hv.len(),
            });
        }
        // Treat any non-zero as its sign; the standard encoding is already {−1, +1}.
        Ok(hv
            .iter()
            .map(|&v| if v >= 0 { 1.0f32 } else { -1.0f32 })
            .collect())
    }

    /// Compute the hidden activations `h = tanh(W1·x + b1)` from pre-cast features.
    fn hidden(&self, x: &[f32]) -> Vec<f32> {
        self.w1
            .chunks_exact(self.cfg.input_dim)
            .zip(self.b1.iter())
            .map(|(row, &bias)| {
                let dot: f32 = row.iter().zip(x.iter()).map(|(&w, &xv)| w * xv).sum();
                (dot + bias).tanh()
            })
            .collect()
    }

    /// Compute raw logits `W2·h + b2` from hidden activations.
    fn logits(&self, h: &[f32]) -> Vec<f32> {
        self.w2
            .chunks_exact(self.cfg.hidden_dim)
            .zip(self.b2.iter())
            .map(|(row, &bias)| {
                let dot: f32 = row.iter().zip(h.iter()).map(|(&w, &hv)| w * hv).sum();
                dot + bias
            })
            .collect()
    }

    /// Numerically stable softmax (in place): subtract the max before exponentiating.
    fn softmax(z: &mut [f32]) {
        let mut max = f32::NEG_INFINITY;
        for &v in z.iter() {
            if v > max {
                max = v;
            }
        }
        if !max.is_finite() {
            max = 0.0;
        }
        let mut sum = 0.0f32;
        for v in z.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for v in z.iter_mut() {
                *v *= inv;
            }
        } else {
            // Degenerate fallback: uniform distribution.
            let u = 1.0 / z.len() as f32;
            for v in z.iter_mut() {
                *v = u;
            }
        }
    }

    /// Return the class-probability distribution `softmax(forward(hv))`.
    ///
    /// # Errors
    ///
    /// [`HdcError::DimensionMismatch`] if `hv.len() != input_dim`.
    pub fn predict_proba(&self, hv: &[i8]) -> HdcResult<Vec<f32>> {
        let x = self.features(hv)?;
        let h = self.hidden(&x);
        let mut z = self.logits(&h);
        Self::softmax(&mut z);
        Ok(z)
    }

    /// Return the predicted class (`argmax` of the probabilities).
    ///
    /// On ties the lowest class index wins.
    ///
    /// # Errors
    ///
    /// [`HdcError::DimensionMismatch`] if `hv.len() != input_dim`.
    pub fn predict(&self, hv: &[i8]) -> HdcResult<usize> {
        let probs = self.predict_proba(hv)?;
        let mut best = 0usize;
        let mut best_p = probs[0];
        for (k, &p) in probs.iter().enumerate().skip(1) {
            if p > best_p {
                best_p = p;
                best = k;
            }
        }
        Ok(best)
    }

    /// Train the network in place with SGD, returning the per-epoch mean cross-entropy loss.
    ///
    /// For every epoch the sample order is shuffled deterministically (Fisher–Yates with an
    /// [`LcgRng`] seeded from `cfg.seed + epoch`), then each sample drives a full
    /// forward+backward pass and an SGD parameter update following the gradient math in the
    /// [module documentation](self). The returned vector has one entry per epoch holding the
    /// mean cross-entropy `−log p[true]` over the training set *before* that epoch's updates
    /// would normally be measured — here it is the running mean of the loss seen during the
    /// epoch.
    ///
    /// # Errors
    ///
    /// - [`HdcError::DimensionMismatch`] if `hvs.len() != labels.len()` or any hypervector
    ///   length differs from `input_dim`.
    /// - [`HdcError::EmptyInput`] if `hvs` is empty.
    /// - [`HdcError::ClassNotFound`] if any label is `≥ n_classes`.
    pub fn fit(&mut self, hvs: &[Vec<i8>], labels: &[usize]) -> HdcResult<Vec<f32>> {
        if hvs.len() != labels.len() {
            return Err(HdcError::DimensionMismatch {
                expected: hvs.len(),
                got: labels.len(),
            });
        }
        if hvs.is_empty() {
            return Err(HdcError::EmptyInput);
        }
        for &y in labels {
            if y >= self.cfg.n_classes {
                return Err(HdcError::ClassNotFound(y));
            }
        }
        // Pre-cast all samples to ±1.0 features, validating lengths up front.
        let mut feats: Vec<Vec<f32>> = Vec::with_capacity(hvs.len());
        for hv in hvs {
            feats.push(self.features(hv)?);
        }

        let n = feats.len();
        let lr = self.cfg.learning_rate;
        let input_dim = self.cfg.input_dim;
        let hidden_dim = self.cfg.hidden_dim;

        let mut losses = Vec::with_capacity(self.cfg.epochs);
        // Reusable scratch buffers to avoid per-sample allocation.
        let mut order: Vec<usize> = (0..n).collect();

        for epoch in 0..self.cfg.epochs {
            // Deterministic Fisher–Yates shuffle for this epoch.
            let mut rng = LcgRng::new(self.cfg.seed.wrapping_add(epoch as u64).wrapping_add(1));
            for i in (1..n).rev() {
                let j = rng.next_usize(i + 1);
                order.swap(i, j);
            }

            let mut epoch_loss = 0.0f32;

            for &idx in order.iter() {
                let x = &feats[idx];
                let y = labels[idx];

                // ---- forward ----
                let h = self.hidden(x);
                let mut probs = self.logits(&h);
                Self::softmax(&mut probs);

                // cross-entropy loss for this sample
                let p_true = probs[y].max(1e-12);
                epoch_loss += -p_true.ln();

                // ---- backward ----
                // dlogits = p - onehot(y)
                let mut dlogits = probs; // reuse buffer; now holds p
                dlogits[y] -= 1.0;

                // dh = W2^T · dlogits   (length hidden_dim).  Accumulated while the read-only
                // view of W2 is still valid, before the in-place W2 update below.
                let mut dh = vec![0.0f32; hidden_dim];
                for (w2row, &g) in self.w2.chunks_exact(hidden_dim).zip(dlogits.iter()) {
                    if g == 0.0 {
                        continue;
                    }
                    for (acc, &w) in dh.iter_mut().zip(w2row.iter()) {
                        *acc += w * g;
                    }
                }

                // ---- update W2, b2 :  W2[k,i] -= lr * dlogits[k] * h[i] ----
                for ((w2row, b), &g) in self
                    .w2
                    .chunks_exact_mut(hidden_dim)
                    .zip(self.b2.iter_mut())
                    .zip(dlogits.iter())
                {
                    *b -= lr * g;
                    if g == 0.0 {
                        continue;
                    }
                    let lg = lr * g;
                    for (w, &hv) in w2row.iter_mut().zip(h.iter()) {
                        *w -= lg * hv;
                    }
                }

                // dz1 = dh ⊙ (1 - h^2)   (tanh derivative)
                // ---- update W1, b1 :  W1[i,j] -= lr * dz1[i] * x[j] ----
                for (((w1row, b), &dhi), &hi) in self
                    .w1
                    .chunks_exact_mut(input_dim)
                    .zip(self.b1.iter_mut())
                    .zip(dh.iter())
                    .zip(h.iter())
                {
                    let dz = dhi * (1.0 - hi * hi);
                    *b -= lr * dz;
                    if dz == 0.0 {
                        continue;
                    }
                    let ldz = lr * dz;
                    for (w, &xv) in w1row.iter_mut().zip(x.iter()) {
                        *w -= ldz * xv;
                    }
                }
            }

            losses.push(epoch_loss / n as f32);
        }

        Ok(losses)
    }

    /// Compute classification accuracy on a labelled set, in `[0, 1]`.
    ///
    /// # Errors
    ///
    /// - [`HdcError::DimensionMismatch`] if `hvs.len() != labels.len()` or any hypervector
    ///   length differs from `input_dim`.
    /// - [`HdcError::EmptyInput`] if `hvs` is empty.
    /// - [`HdcError::ClassNotFound`] if any label is `≥ n_classes`.
    pub fn accuracy(&self, hvs: &[Vec<i8>], labels: &[usize]) -> HdcResult<f32> {
        if hvs.len() != labels.len() {
            return Err(HdcError::DimensionMismatch {
                expected: hvs.len(),
                got: labels.len(),
            });
        }
        if hvs.is_empty() {
            return Err(HdcError::EmptyInput);
        }
        for &y in labels {
            if y >= self.cfg.n_classes {
                return Err(HdcError::ClassNotFound(y));
            }
        }
        let mut correct = 0usize;
        for (hv, &y) in hvs.iter().zip(labels.iter()) {
            if self.predict(hv)? == y {
                correct += 1;
            }
        }
        Ok(correct as f32 / hvs.len() as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Build a random ±1 prototype hypervector of length `dim`.
    fn prototype(rng: &mut LcgRng, dim: usize) -> Vec<i8> {
        let mut v = vec![0i8; dim];
        rng.fill_binary(&mut v);
        v
    }

    /// Copy `proto` and flip each bit independently with probability `flip_p`.
    fn noisy(rng: &mut LcgRng, proto: &[i8], flip_p: f32) -> Vec<i8> {
        proto
            .iter()
            .map(|&b| if rng.next_f32() < flip_p { -b } else { b })
            .collect()
    }

    /// Generate a 2-class, separable-in-HD data set from two prototypes with bit flips.
    fn make_dataset(
        seed: u64,
        dim: usize,
        per_class: usize,
        flip_p: f32,
    ) -> (Vec<Vec<i8>>, Vec<usize>) {
        let mut rng = LcgRng::new(seed);
        let p0 = prototype(&mut rng, dim);
        let p1 = prototype(&mut rng, dim);
        let mut hvs = Vec::with_capacity(2 * per_class);
        let mut labels = Vec::with_capacity(2 * per_class);
        for _ in 0..per_class {
            hvs.push(noisy(&mut rng, &p0, flip_p));
            labels.push(0usize);
            hvs.push(noisy(&mut rng, &p1, flip_p));
            labels.push(1usize);
        }
        (hvs, labels)
    }

    #[test]
    fn config_rejects_zero_dims() {
        assert!(matches!(
            HdMlpConfig::new(0, 8, 2, 0.05, 10, 1),
            Err(HdcError::ZeroDimension)
        ));
        assert!(matches!(
            HdMlpConfig::new(16, 0, 2, 0.05, 10, 1),
            Err(HdcError::ZeroDimension)
        ));
    }

    #[test]
    fn config_rejects_bad_classes_lr_epochs() {
        // n_classes < 2
        assert!(matches!(
            HdMlpConfig::new(16, 8, 1, 0.05, 10, 1),
            Err(HdcError::EmptyInput)
        ));
        // non-positive learning rate
        assert!(matches!(
            HdMlpConfig::new(16, 8, 2, 0.0, 10, 1),
            Err(HdcError::InvalidProbability(_))
        ));
        // non-finite learning rate
        assert!(matches!(
            HdMlpConfig::new(16, 8, 2, f32::NAN, 10, 1),
            Err(HdcError::InvalidProbability(_))
        ));
        // zero epochs
        assert!(matches!(
            HdMlpConfig::new(16, 8, 2, 0.05, 0, 1),
            Err(HdcError::EmptyInput)
        ));
    }

    #[test]
    fn predict_proba_is_a_distribution() {
        let cfg = HdMlpConfig::new(64, 16, 3, 0.05, 5, 7).expect("valid cfg");
        let net = HdMlp::new(cfg).expect("valid net");
        let mut rng = LcgRng::new(99);
        let hv = prototype(&mut rng, 64);
        let p = net.predict_proba(&hv).expect("forward ok");
        assert_eq!(p.len(), 3);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax must sum to 1, got {sum}");
        for &v in &p {
            assert!((0.0..=1.0).contains(&v), "prob out of range: {v}");
        }
    }

    #[test]
    fn untrained_net_runs_forward() {
        let cfg = HdMlpConfig::new(128, 8, 2, 0.05, 1, 3).expect("valid cfg");
        let net = HdMlp::new(cfg).expect("valid net");
        let mut rng = LcgRng::new(11);
        let hv = prototype(&mut rng, 128);
        let cls = net.predict(&hv).expect("predict ok");
        assert!(cls < 2);
        assert_eq!(net.input_dim(), 128);
        assert_eq!(net.hidden_dim(), 8);
        assert_eq!(net.n_classes(), 2);
    }

    #[test]
    fn training_decreases_loss_and_fits_separable_data() {
        let dim = 256;
        let (hvs, labels) = make_dataset(2024, dim, 18, 0.12);
        let cfg = HdMlpConfig::new(dim, 32, 2, 0.08, 90, 42).expect("valid cfg");
        let mut net = HdMlp::new(cfg).expect("valid net");
        let losses = net.fit(&hvs, &labels).expect("fit ok");

        assert_eq!(losses.len(), 90);
        let first = losses[0];
        let last = losses[losses.len() - 1];
        assert!(
            last < first,
            "training loss should decrease: first={first}, last={last}"
        );
        let acc = net.accuracy(&hvs, &labels).expect("accuracy ok");
        assert!(
            acc > 0.9,
            "training accuracy should exceed 0.9 on separable data, got {acc}"
        );
    }

    #[test]
    fn fit_rejects_length_mismatch() {
        let cfg = HdMlpConfig::new(32, 8, 2, 0.05, 3, 1).expect("valid cfg");
        let mut net = HdMlp::new(cfg).expect("valid net");
        let mut rng = LcgRng::new(5);
        let hvs = vec![prototype(&mut rng, 32), prototype(&mut rng, 32)];
        let labels = vec![0usize]; // shorter
        assert!(matches!(
            net.fit(&hvs, &labels),
            Err(HdcError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn fit_rejects_out_of_range_label() {
        let cfg = HdMlpConfig::new(32, 8, 2, 0.05, 3, 1).expect("valid cfg");
        let mut net = HdMlp::new(cfg).expect("valid net");
        let mut rng = LcgRng::new(6);
        let hvs = vec![prototype(&mut rng, 32), prototype(&mut rng, 32)];
        let labels = vec![0usize, 5usize]; // 5 >= n_classes
        assert!(matches!(
            net.fit(&hvs, &labels),
            Err(HdcError::ClassNotFound(5))
        ));
    }

    #[test]
    fn predict_rejects_dim_mismatch() {
        let cfg = HdMlpConfig::new(64, 8, 2, 0.05, 1, 1).expect("valid cfg");
        let net = HdMlp::new(cfg).expect("valid net");
        let wrong = vec![1i8; 10];
        assert!(matches!(
            net.predict(&wrong),
            Err(HdcError::DimensionMismatch {
                expected: 64,
                got: 10
            })
        ));
        assert!(matches!(
            net.predict_proba(&wrong),
            Err(HdcError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn training_is_deterministic() {
        let dim = 128;
        let (hvs, labels) = make_dataset(77, dim, 12, 0.1);

        let cfg_a = HdMlpConfig::new(dim, 16, 2, 0.07, 40, 123).expect("valid cfg");
        let mut net_a = HdMlp::new(cfg_a).expect("valid net");
        let losses_a = net_a.fit(&hvs, &labels).expect("fit a");

        let cfg_b = HdMlpConfig::new(dim, 16, 2, 0.07, 40, 123).expect("valid cfg");
        let mut net_b = HdMlp::new(cfg_b).expect("valid net");
        let losses_b = net_b.fit(&hvs, &labels).expect("fit b");

        assert_eq!(losses_a, losses_b, "identical seed+data → identical losses");
        for hv in &hvs {
            assert_eq!(
                net_a.predict(hv).expect("pred a"),
                net_b.predict(hv).expect("pred b"),
                "identical seed+data → identical predictions"
            );
        }
    }

    #[test]
    fn accuracy_is_in_unit_interval() {
        let dim = 96;
        let (hvs, labels) = make_dataset(321, dim, 8, 0.15);
        let cfg = HdMlpConfig::new(dim, 16, 2, 0.05, 10, 9).expect("valid cfg");
        let net = HdMlp::new(cfg).expect("valid net");
        let acc = net.accuracy(&hvs, &labels).expect("accuracy ok");
        assert!((0.0..=1.0).contains(&acc), "accuracy out of range: {acc}");
    }
}
