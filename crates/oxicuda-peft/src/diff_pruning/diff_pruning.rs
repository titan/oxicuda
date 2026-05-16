use crate::handle::LcgRng;

/// Configuration for the Diff-Pruning L0 regulariser.
///
/// Default values follow the original paper (β=2/3, γ=−0.1, ζ=1.1, λ_L0=0.01).
#[derive(Debug, Clone)]
pub struct DiffPrunerConfig {
    /// Temperature β for the Hard Concrete distribution.
    pub beta: f32,
    /// Stretch lower bound γ (negative, e.g. −0.1).
    pub gamma: f32,
    /// Stretch upper bound ζ (> 1, e.g. 1.1).
    pub zeta: f32,
    /// L0 regularisation coefficient λ.
    pub lambda_l0: f32,
}

impl Default for DiffPrunerConfig {
    fn default() -> Self {
        Self {
            beta: 2.0 / 3.0,
            gamma: -0.1,
            zeta: 1.1,
            lambda_l0: 0.01,
        }
    }
}

/// Diff-Pruning: learn a sparse binary mask over weight differences using Hard Concrete relaxation.
///
/// For each weight element i the effective update is `z_i * delta_i` where `z_i ∈ [0, 1]`
/// is drawn from the Hard Concrete distribution parameterised by `log_alpha_i`.
/// The base weight `base_w` is fixed; only `delta` and `log_alpha` are trained.
#[derive(Debug, Clone)]
pub struct DiffPruner {
    /// Frozen base weight vector.
    pub base_w: Vec<f32>,
    /// Log-α parameters controlling the per-element mask probability.
    pub log_alpha: Vec<f32>,
    /// Trainable weight difference vector.
    pub delta: Vec<f32>,
    /// Hard Concrete hyperparameters.
    pub config: DiffPrunerConfig,
}

impl DiffPruner {
    /// Construct a `DiffPruner` from a base weight vector.
    ///
    /// `log_alpha` is initialised from N(0, 0.01); `delta` is zero-initialised.
    #[must_use]
    pub fn new(w: &[f32], cfg: DiffPrunerConfig, rng: &mut LcgRng) -> Self {
        let n = w.len();
        let mut log_alpha = vec![0.0_f32; n];
        rng.fill_normal(&mut log_alpha);
        for v in log_alpha.iter_mut() {
            *v *= 0.01;
        }
        let delta = vec![0.0_f32; n];
        Self {
            base_w: w.to_vec(),
            log_alpha,
            delta,
            config: cfg,
        }
    }

    /// Sample a stochastic mask using the Hard Concrete distribution.
    ///
    /// For each element: `s = sigmoid((log_α - log(u/(1-u))) / β)`,
    /// then `s_bar = s * (ζ - γ) + γ`, then `z = clamp(s_bar, 0, 1)`.
    #[must_use]
    pub fn compute_mask(&self, rng: &mut LcgRng) -> Vec<f32> {
        let beta = self.config.beta;
        let gamma = self.config.gamma;
        let zeta = self.config.zeta;
        let stretch = zeta - gamma;
        self.log_alpha
            .iter()
            .map(|&log_a| {
                let u = (rng.next_f32() + 1e-12).min(1.0 - 1e-12);
                let log_odds = (u / (1.0 - u)).ln();
                let s = sigmoid_f32((log_a - log_odds) / beta);
                let s_bar = s * stretch + gamma;
                s_bar.clamp(0.0, 1.0)
            })
            .collect()
    }

    /// Compute the deterministic (inference-time) mask without randomness.
    ///
    /// Uses `sigmoid(log_α) * (ζ - γ) + γ` clamped to `[0, 1]`.
    #[must_use]
    pub fn compute_mask_deterministic(&self) -> Vec<f32> {
        let gamma = self.config.gamma;
        let zeta = self.config.zeta;
        let stretch = zeta - gamma;
        self.log_alpha
            .iter()
            .map(|&log_a| {
                let s = sigmoid_f32(log_a);
                let s_bar = s * stretch + gamma;
                s_bar.clamp(0.0, 1.0)
            })
            .collect()
    }

    /// Compute the masked weight: `base_w + compute_mask(rng) * delta`.
    #[must_use]
    pub fn forward(&self, rng: &mut LcgRng) -> Vec<f32> {
        let mask = self.compute_mask(rng);
        self.base_w
            .iter()
            .zip(self.delta.iter())
            .zip(mask.iter())
            .map(|((&w, &d), &z)| w + z * d)
            .collect()
    }

    /// Compute the expected L0 pseudo-norm (regularisation term).
    ///
    /// `L0 ≈ Σ_i sigmoid(log_α_i - β · log(-γ/ζ))`
    #[must_use]
    pub fn l0_regularizer(&self) -> f32 {
        let beta = self.config.beta;
        let gamma = self.config.gamma;
        let zeta = self.config.zeta;
        // log(-gamma/zeta): gamma is negative so -gamma is positive
        let log_ratio = (-gamma / zeta).ln();
        self.log_alpha
            .iter()
            .map(|&log_a| sigmoid_f32(log_a - beta * log_ratio))
            .sum()
    }
}

/// Numerically stable sigmoid function.
#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
