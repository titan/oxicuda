//! Differentially private synthetic-data generation — PATE-GAN and DP-GAN.
//!
//! References:
//! - Jordon, Yoon & van der Schaar (2019), "PATE-GAN: Generating Synthetic Data
//!   with Differential Privacy Guarantees", ICLR.
//! - Xie, Lin, Wang, Wang & Zhou (2018), "Differentially Private Generative
//!   Adversarial Network", arXiv:1802.06739 (DP-GAN).
//! - Abadi et al. (2016), "Deep Learning with Differential Privacy" (DP-SGD /
//!   the moments accountant reused by the DP-GAN discriminator).
//!
//! # PATE-GAN (Jordon 2019)
//! A *student* generator `G` is trained against a *student* discriminator `D`.
//! `D`'s training labels are **not** the true real/fake labels of the private
//! data — they come from the noisy-argmax aggregation of an ensemble of teacher
//! discriminators, each trained on a disjoint partition of the private data, run
//! through the crate's existing [`pate_aggregate`] mechanism. The generator and
//! student therefore only ever observe the private data through the
//! differentially-private aggregated votes, so the whole pipeline is DP by
//! post-processing. Each labelling query spends a fixed per-query `ε`; over the
//! query schedule the cumulative spend is composed with the crate's
//! [`BudgetMonitor`] (basic, i.e. sequential, composition of pure-ε queries).
//!
//! # DP-GAN (Xie 2018, with our DP-Adam)
//! A generator `G` is trained against a discriminator `D` that is optimised with
//! the crate's [`DpAdamState`] (per-sample gradient clipping + Gaussian noise).
//! Because `D` is the only network that touches the private data and it is
//! trained under DP-SGD/Adam, the generator — updated only from `D`'s signal —
//! is DP by post-processing. The discriminator's privacy is accounted with the
//! crate's Sampled-Gaussian moments accountant
//! ([`SampledGaussianMechanism`]), exactly as `dp_adam_harness` does, composing
//! one Sampled-Gaussian RDP term per discriminator step.
//!
//! # Networks
//! Both generators are genuine two-layer MLPs (`tanh` hidden layer, `tanh`
//! bounded output so synthetic rows always lie in `(-1, 1)` and are finite);
//! discriminators (student + teachers) are genuine two-layer MLP binary
//! classifiers. All forward passes and gradients are implemented by hand
//! (analytic back-propagation, including back-propagation of the discriminator
//! signal through to the generator's parameters); there is no stubbed loop.
//!
//! # What is *not* asserted
//! Reproducing the real data distribution faithfully requires a training scale
//! that cannot be CPU-verified here, so the tests assert only CPU-verifiable
//! properties: privacy-budget accounting consistency with the crate's own
//! accountants, structural correctness + finiteness + determinism of sampling,
//! and the sanity of the DP voting path. They do **not** assert distributional
//! fidelity of the synthetic data.

use crate::accounting::budget_monitor::BudgetMonitor;
use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;
use crate::mechanism::pate::{PateConfig, PateMechanism, pate_aggregate};
use crate::mechanism::sampled_gaussian::{SampledGaussianConfig, SampledGaussianMechanism};
use crate::optimizer::dp_adam::{DpAdamConfig, DpAdamState};

/// L1 sensitivity of a binary vote histogram: a single teacher switching its
/// vote moves one count down by 1 and another up by 1 (Papernot 2017).
const VOTE_L1_SENSITIVITY: f64 = 2.0;

// ─── Numerically-stable scalar helpers ───────────────────────────────────────

/// Numerically-stable logistic sigmoid `σ(x) = 1 / (1 + e^{-x})`.
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Draw a `latent`-dimensional standard-normal latent vector from `rng`.
fn sample_latent(rng: &mut LcgRng, latent: usize) -> Vec<f64> {
    (0..latent).map(|_| rng.normal_pair().0).collect()
}

/// Fill `slice` with i.i.d. `Uniform(-radius, radius)` weights from `rng`.
fn fill_uniform_sym(slice: &mut [f64], radius: f64, rng: &mut LcgRng) {
    for v in slice.iter_mut() {
        *v = (rng.next_f64() * 2.0 - 1.0) * radius;
    }
}

/// In-place stochastic-gradient-descent update `θ ← θ − lr · g`.
fn sgd_update(params: &mut [f64], grad: &[f64], lr: f64) {
    for (p, &g) in params.iter_mut().zip(grad.iter()) {
        *p -= lr * g;
    }
}

// ─── Discriminator MLP (data_dim → hidden → 1) ───────────────────────────────

/// Dimensions of a two-layer MLP binary classifier (`tanh` hidden, scalar
/// logit output). Parameters are stored flat as `[w1 | b1 | w2 | b2]`:
/// `w1` is `hidden×data` row-major, `b1` is `hidden`, `w2` is `hidden`, `b2` is
/// a single scalar.
#[derive(Debug, Clone, Copy)]
struct DiscDims {
    data: usize,
    hidden: usize,
}

impl DiscDims {
    fn n_params(&self) -> usize {
        self.hidden * self.data + 2 * self.hidden + 1
    }
    fn b1_off(&self) -> usize {
        self.hidden * self.data
    }
    fn w2_off(&self) -> usize {
        self.hidden * self.data + self.hidden
    }
    fn b2_off(&self) -> usize {
        self.hidden * self.data + 2 * self.hidden
    }

    /// Forward pass: returns `(logit, hidden_activations)`.
    fn forward(&self, params: &[f64], x: &[f64]) -> (f64, Vec<f64>) {
        let b1_off = self.b1_off();
        let w2_off = self.w2_off();
        let b2_off = self.b2_off();
        let mut a1 = vec![0.0_f64; self.hidden];
        for (hh, a) in a1.iter_mut().enumerate() {
            let base = hh * self.data;
            let mut pre = params[b1_off + hh];
            for (dd, &xv) in x.iter().enumerate() {
                pre += params[base + dd] * xv;
            }
            *a = pre.tanh();
        }
        let mut logit = params[b2_off];
        for (hh, &av) in a1.iter().enumerate() {
            logit += params[w2_off + hh] * av;
        }
        (logit, a1)
    }

    /// Accumulate `∂loss/∂params` into `grad` given `∂loss/∂logit = dlogit`.
    fn backward(&self, params: &[f64], x: &[f64], a1: &[f64], dlogit: f64, grad: &mut [f64]) {
        let b1_off = self.b1_off();
        let w2_off = self.w2_off();
        let b2_off = self.b2_off();
        grad[b2_off] += dlogit;
        for (hh, &av) in a1.iter().enumerate() {
            grad[w2_off + hh] += dlogit * av;
            let dpre = dlogit * params[w2_off + hh] * (1.0 - av * av);
            grad[b1_off + hh] += dpre;
            let base = hh * self.data;
            for (dd, &xv) in x.iter().enumerate() {
                grad[base + dd] += dpre * xv;
            }
        }
    }

    /// Accumulate `∂loss/∂x` into `dx` (back-propagation of the discriminator
    /// signal to its *input* — the path used to update the generator).
    fn input_grad(&self, params: &[f64], a1: &[f64], dlogit: f64, dx: &mut [f64]) {
        let w2_off = self.w2_off();
        for (hh, &av) in a1.iter().enumerate() {
            let dpre = dlogit * params[w2_off + hh] * (1.0 - av * av);
            let base = hh * self.data;
            for (dd, dxv) in dx.iter_mut().enumerate() {
                *dxv += params[base + dd] * dpre;
            }
        }
    }
}

/// Accumulate the binary-cross-entropy gradient of one sample (target `y`) into
/// `grad` for the discriminator with the given flat `params`.
fn accum_bce_grad(dims: &DiscDims, params: &[f64], x: &[f64], y: f64, grad: &mut [f64]) {
    let (logit, a1) = dims.forward(params, x);
    let dlogit = sigmoid(logit) - y; // ∂BCE/∂logit
    dims.backward(params, x, &a1, dlogit, grad);
}

// ─── Generator MLP (latent → hidden → data_dim, tanh-bounded output) ──────────

/// Dimensions of the two-layer generator MLP (`tanh` hidden, `tanh` bounded
/// output). Parameters are stored flat as `[w1 | b1 | w2 | b2]`: `w1` is
/// `hidden×latent` row-major, `b1` is `hidden`, `w2` is `data×hidden`
/// row-major, `b2` is `data`.
#[derive(Debug, Clone, Copy)]
struct GenDims {
    latent: usize,
    hidden: usize,
    data: usize,
}

impl GenDims {
    fn n_params(&self) -> usize {
        self.hidden * self.latent + self.hidden + self.data * self.hidden + self.data
    }
    fn b1_off(&self) -> usize {
        self.hidden * self.latent
    }
    fn w2_off(&self) -> usize {
        self.hidden * self.latent + self.hidden
    }
    fn b2_off(&self) -> usize {
        self.w2_off() + self.data * self.hidden
    }

    /// Forward pass: returns `(output_row, hidden_activations)`. The output is
    /// `tanh`-bounded, so every coordinate lies in `(-1, 1)` and is finite.
    fn forward(&self, params: &[f64], z: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let b1_off = self.b1_off();
        let w2_off = self.w2_off();
        let b2_off = self.b2_off();
        let mut a1 = vec![0.0_f64; self.hidden];
        for (hh, a) in a1.iter_mut().enumerate() {
            let base = hh * self.latent;
            let mut pre = params[b1_off + hh];
            for (ll, &zv) in z.iter().enumerate() {
                pre += params[base + ll] * zv;
            }
            *a = pre.tanh();
        }
        let mut out = vec![0.0_f64; self.data];
        for (dd, o) in out.iter_mut().enumerate() {
            let base = w2_off + dd * self.hidden;
            let mut pre = params[b2_off + dd];
            for (hh, &av) in a1.iter().enumerate() {
                pre += params[base + hh] * av;
            }
            *o = pre.tanh();
        }
        (out, a1)
    }

    /// Accumulate `∂loss/∂params` into `grad` given `∂loss/∂output = dout`.
    fn backward(
        &self,
        params: &[f64],
        z: &[f64],
        a1: &[f64],
        out: &[f64],
        dout: &[f64],
        grad: &mut [f64],
    ) {
        let b1_off = self.b1_off();
        let w2_off = self.w2_off();
        let b2_off = self.b2_off();
        let mut da1 = vec![0.0_f64; self.hidden];
        for (dd, (&ov, &dov)) in out.iter().zip(dout.iter()).enumerate() {
            let dpre = dov * (1.0 - ov * ov); // tanh' at the output
            grad[b2_off + dd] += dpre;
            let base = w2_off + dd * self.hidden;
            for (hh, &av) in a1.iter().enumerate() {
                grad[base + hh] += dpre * av;
                da1[hh] += params[base + hh] * dpre;
            }
        }
        for (hh, (&dav, &av)) in da1.iter().zip(a1.iter()).enumerate() {
            let dpre = dav * (1.0 - av * av);
            grad[b1_off + hh] += dpre;
            let base = hh * self.latent;
            for (ll, &zv) in z.iter().enumerate() {
                grad[base + ll] += dpre * zv;
            }
        }
    }
}

// ─── Trained generator ────────────────────────────────────────────────────────

/// A trained synthetic-data generator returned by [`pate_gan_train`] /
/// [`dp_gan_train`].
///
/// Sampling is deterministic: [`sample`](Self::sample) re-seeds a fresh
/// [`LcgRng`] from the generator's stored `sample_seed` on every call, so
/// repeated calls (and two generators trained with the same seed) produce
/// identical synthetic rows.
#[derive(Debug, Clone)]
pub struct SyntheticGenerator {
    dims: GenDims,
    params: Vec<f64>,
    sample_seed: u64,
}

impl SyntheticGenerator {
    /// Width (number of columns) of every synthetic row.
    #[must_use]
    pub fn data_dim(&self) -> usize {
        self.dims.data
    }

    /// Latent dimension consumed by the generator per row.
    #[must_use]
    pub fn latent_dim(&self) -> usize {
        self.dims.latent
    }

    /// Generate `n` synthetic rows (each of width [`data_dim`](Self::data_dim)).
    ///
    /// Deterministic under the generator's stored sampling seed: every call
    /// returns the same rows. All coordinates are finite (the output layer is
    /// `tanh`-bounded).
    #[must_use]
    pub fn sample(&self, n: usize) -> Vec<Vec<f64>> {
        self.sample_with_seed(n, self.sample_seed)
    }

    /// Generate `n` synthetic rows using an explicit sampling `seed`.
    #[must_use]
    pub fn sample_with_seed(&self, n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = LcgRng::new(seed);
        let mut rows = Vec::with_capacity(n);
        for _ in 0..n {
            let z = sample_latent(&mut rng, self.dims.latent);
            let (out, _) = self.dims.forward(&self.params, &z);
            rows.push(out);
        }
        rows
    }
}

/// Initialise discriminator parameters (Xavier-style `Uniform(-1/√fan_in)`
/// weights, zero biases) deterministically from `rng`.
fn init_disc(dims: &DiscDims, rng: &mut LcgRng) -> Vec<f64> {
    let mut p = vec![0.0_f64; dims.n_params()];
    let hd = dims.hidden * dims.data;
    let r1 = 1.0 / (dims.data as f64).sqrt();
    fill_uniform_sym(&mut p[0..hd], r1, rng);
    let r2 = 1.0 / (dims.hidden as f64).sqrt();
    let w2_off = dims.w2_off();
    fill_uniform_sym(&mut p[w2_off..w2_off + dims.hidden], r2, rng);
    p
}

/// Initialise generator parameters deterministically from `rng`.
fn init_gen(dims: &GenDims, rng: &mut LcgRng) -> Vec<f64> {
    let mut p = vec![0.0_f64; dims.n_params()];
    let hl = dims.hidden * dims.latent;
    let r1 = 1.0 / (dims.latent as f64).sqrt();
    fill_uniform_sym(&mut p[0..hl], r1, rng);
    let r2 = 1.0 / (dims.hidden as f64).sqrt();
    let w2_off = dims.w2_off();
    let w2_len = dims.data * dims.hidden;
    fill_uniform_sym(&mut p[w2_off..w2_off + w2_len], r2, rng);
    p
}

/// One generator SGD step: push the generator so the (fixed) discriminator
/// classifies its outputs as *real* (target 1, the non-saturating GAN loss).
/// The discriminator signal is back-propagated through `D` to the generator's
/// output and then through the generator to its parameters.
fn generator_step(
    gnet: &GenDims,
    gp: &mut [f64],
    disc: &DiscDims,
    dp: &[f64],
    latent_batch: &[Vec<f64>],
    lr: f64,
) {
    if latent_batch.is_empty() {
        return;
    }
    let mut grad = vec![0.0_f64; gnet.n_params()];
    let mut dx = vec![0.0_f64; gnet.data];
    for z in latent_batch {
        let (out, a1g) = gnet.forward(gp, z);
        let (logit, a1d) = disc.forward(dp, &out);
        let dlogit = sigmoid(logit) - 1.0; // generator wants D(out) → "real"
        for v in dx.iter_mut() {
            *v = 0.0;
        }
        disc.input_grad(dp, &a1d, dlogit, &mut dx);
        gnet.backward(gp, z, &a1g, &out, &dx, &mut grad);
    }
    let inv = 1.0 / latent_batch.len() as f64;
    for g in grad.iter_mut() {
        *g *= inv;
    }
    sgd_update(gp, &grad, lr);
}

// ─── PATE-GAN ─────────────────────────────────────────────────────────────────

/// Configuration of the PATE-GAN synthesiser.
#[derive(Debug, Clone)]
pub struct PateGanConfig {
    /// Width of each (real and synthetic) data row, `d ≥ 1`.
    pub data_dim: usize,
    /// Generator latent dimension, `≥ 1`.
    pub latent_dim: usize,
    /// Hidden-layer width for the generator and all discriminators, `≥ 1`.
    pub hidden_dim: usize,
    /// Number of teacher discriminators `T ≥ 2` (disjoint private partitions).
    pub n_teachers: usize,
    /// Number of training rounds, `≥ 1`.
    pub rounds: usize,
    /// Generated samples labelled (one PATE query each) per round, `≥ 1`.
    pub student_queries_per_round: usize,
    /// SGD steps per round for each teacher, `≥ 1`.
    pub teacher_steps: usize,
    /// SGD steps per round for the student discriminator, `≥ 1`.
    pub student_steps: usize,
    /// SGD steps per round for the generator, `≥ 1`.
    pub gen_steps: usize,
    /// Minibatch size for teacher/generator training, `≥ 1`.
    pub batch_size: usize,
    /// SGD learning rate `> 0`.
    pub learning_rate: f64,
    /// Total pure-ε privacy budget spent across *all* teacher-labelling queries
    /// (`> 0`). Per-query budget is this divided by [`queries`](Self::queries);
    /// the LNMax noise scale is `2 / ε_per_query`, so a tighter budget yields a
    /// larger noise scale.
    pub epsilon_budget: f64,
}

impl PateGanConfig {
    /// A small, fast configuration for the given row width (used by the tests).
    #[must_use]
    pub fn small(data_dim: usize) -> Self {
        Self {
            data_dim,
            latent_dim: 3,
            hidden_dim: 8,
            n_teachers: 5,
            rounds: 3,
            student_queries_per_round: 8,
            teacher_steps: 4,
            student_steps: 4,
            gen_steps: 4,
            batch_size: 12,
            learning_rate: 0.05,
            epsilon_budget: 6.0,
        }
    }

    /// Total number of PATE labelling queries over the whole run.
    #[must_use]
    pub fn queries(&self) -> usize {
        self.rounds * self.student_queries_per_round
    }

    /// Per-query pure-ε budget `ε_budget / queries`.
    #[must_use]
    pub fn epsilon_per_query(&self) -> f64 {
        self.epsilon_budget / self.queries() as f64
    }

    /// LNMax Laplace noise scale `2 / ε_per_query` (L1 vote sensitivity 2).
    /// Monotonically *decreasing* in `ε_per_query`, hence increasing as the
    /// privacy budget tightens.
    #[must_use]
    pub fn laplace_scale(&self) -> f64 {
        VOTE_L1_SENSITIVITY / self.epsilon_per_query()
    }

    /// Validate the configuration.
    ///
    /// # Errors
    /// `InvalidParameter` for any zero dimension/step/batch/round count, fewer
    /// than two teachers, a non-positive learning rate, or a non-positive /
    /// non-finite privacy budget.
    pub fn validate(&self) -> PrivacyResult<()> {
        if self.data_dim == 0 || self.latent_dim == 0 || self.hidden_dim == 0 {
            return Err(PrivacyError::InvalidParameter(
                "data_dim, latent_dim and hidden_dim must be ≥ 1".into(),
            ));
        }
        if self.n_teachers < 2 {
            return Err(PrivacyError::InvalidParameter(format!(
                "n_teachers must be ≥ 2, got {}",
                self.n_teachers
            )));
        }
        if self.rounds == 0
            || self.student_queries_per_round == 0
            || self.teacher_steps == 0
            || self.student_steps == 0
            || self.gen_steps == 0
            || self.batch_size == 0
        {
            return Err(PrivacyError::InvalidParameter(
                "rounds, per-round step counts and batch_size must be ≥ 1".into(),
            ));
        }
        if self.learning_rate <= 0.0 || !self.learning_rate.is_finite() {
            return Err(PrivacyError::InvalidParameter(format!(
                "learning_rate must be positive, got {}",
                self.learning_rate
            )));
        }
        if self.epsilon_budget <= 0.0 || !self.epsilon_budget.is_finite() {
            return Err(PrivacyError::NonPositiveEpsilon(self.epsilon_budget));
        }
        Ok(())
    }
}

/// Result of [`pate_gan_train`].
#[derive(Debug, Clone)]
pub struct PateGanReport {
    /// The trained synthetic-data generator.
    pub generator: SyntheticGenerator,
    /// Cumulative ε pulled from the crate's [`BudgetMonitor`] after composing
    /// the full schedule of teacher-labelling queries (basic composition).
    pub cumulative_epsilon: f64,
    /// Total number of PATE labelling queries composed.
    pub queries: usize,
    /// Per-query pure-ε budget actually spent.
    pub epsilon_per_query: f64,
    /// LNMax Laplace noise scale used for the teacher vote aggregation.
    pub laplace_scale: f64,
}

/// Contiguous `[lo, hi)` private-data partition assigned to teacher `t`.
fn teacher_partition(t: usize, n_rows: usize, n_teachers: usize) -> (usize, usize) {
    let chunk = n_rows / n_teachers;
    let lo = t * chunk;
    let hi = if t + 1 == n_teachers {
        n_rows
    } else {
        (t + 1) * chunk
    };
    (lo, hi)
}

/// DP label for a teacher-vote histogram via the crate's PATE noisy-argmax,
/// using LNMax (Laplace) noise calibrated to `epsilon_per_query` (scale
/// `2 / ε`, L1 vote sensitivity 2). This is the exact labelling primitive used
/// inside [`pate_gan_train`]; it is exposed so the DP voting path can be
/// exercised directly.
///
/// # Errors
/// - `InvalidParameter` if `epsilon_per_query ≤ 0` or non-finite.
/// - Propagates [`PateConfig::new`] / [`pate_aggregate`] validation errors.
pub fn dp_label_from_votes(
    votes: &[usize],
    num_classes: usize,
    epsilon_per_query: f64,
    rng: &mut LcgRng,
) -> PrivacyResult<usize> {
    if epsilon_per_query <= 0.0 || !epsilon_per_query.is_finite() {
        return Err(PrivacyError::NonPositiveEpsilon(epsilon_per_query));
    }
    let scale = VOTE_L1_SENSITIVITY / epsilon_per_query;
    let cfg = PateConfig::new(num_classes, PateMechanism::Laplace { scale })?;
    pate_aggregate(votes, &cfg, rng)
}

/// Train a PATE-GAN synthesiser on `real_data` (flat `[n_rows × data_dim]`).
///
/// The private rows are partitioned across `cfg.n_teachers` teacher
/// discriminators. Each round trains the teachers (on their private partition
/// vs. current fakes), labels `cfg.student_queries_per_round` generated samples
/// through the DP teacher-vote aggregation, trains the student discriminator on
/// those DP labels, and updates the generator against the student. The cumulative
/// privacy spend is composed with the crate's [`BudgetMonitor`].
///
/// `seed` makes the whole run reproducible.
///
/// # Errors
/// - Propagates [`PateGanConfig::validate`].
/// - `DimensionMismatch` if `real_data.len() != n_rows * cfg.data_dim`.
/// - `InvalidParameter` if `n_rows < cfg.n_teachers` (each teacher needs ≥ 1
///   private row).
/// - Propagates accountant / mechanism errors.
pub fn pate_gan_train(
    real_data: &[f64],
    n_rows: usize,
    cfg: &PateGanConfig,
    seed: u64,
) -> PrivacyResult<PateGanReport> {
    cfg.validate()?;
    let d = cfg.data_dim;
    if real_data.len() != n_rows * d {
        return Err(PrivacyError::DimensionMismatch {
            expected: n_rows * d,
            got: real_data.len(),
        });
    }
    if n_rows < cfg.n_teachers {
        return Err(PrivacyError::InvalidParameter(format!(
            "n_rows={n_rows} must be ≥ n_teachers={} so every teacher gets a partition",
            cfg.n_teachers
        )));
    }

    let gnet = GenDims {
        latent: cfg.latent_dim,
        hidden: cfg.hidden_dim,
        data: d,
    };
    let disc = DiscDims {
        data: d,
        hidden: cfg.hidden_dim,
    };

    // Independent RNG streams (different seed offsets) so weight init, latent
    // draws and PATE noise do not alias.
    let mut init_rng = LcgRng::new(seed ^ 0x5044_4145_4741_4E00);
    let mut latent_rng = LcgRng::new(seed ^ 0x4C41_5445_4E54_0001);
    let mut noise_rng = LcgRng::new(seed ^ 0x564F_5445_4E4F_5345);

    let mut gp = init_gen(&gnet, &mut init_rng);
    let mut student = init_disc(&disc, &mut init_rng);
    let mut teachers: Vec<Vec<f64>> = (0..cfg.n_teachers)
        .map(|_| init_disc(&disc, &mut init_rng))
        .collect();

    // Privacy: basic (sequential) composition of pure-ε LNMax queries.
    let queries = cfg.queries();
    let eps_q = cfg.epsilon_per_query();
    let scale = cfg.laplace_scale();
    let pate_cfg = PateConfig::new(2, PateMechanism::Laplace { scale })?;
    let mut monitor = BudgetMonitor::new(cfg.epsilon_budget + 1.0, 0.0)?;

    let lr = cfg.learning_rate;
    let mut teacher_grad = vec![0.0_f64; disc.n_params()];

    for _round in 0..cfg.rounds {
        // Current fake batch shared by teacher and student "fake" training.
        let fake_batch: Vec<Vec<f64>> = (0..cfg.batch_size)
            .map(|_| {
                let z = sample_latent(&mut latent_rng, cfg.latent_dim);
                gnet.forward(&gp, &z).0
            })
            .collect();

        // 1. Train each teacher on its private partition (real → 1) vs fakes (→ 0).
        for (t, tp) in teachers.iter_mut().enumerate() {
            let (lo, hi) = teacher_partition(t, n_rows, cfg.n_teachers);
            for _ in 0..cfg.teacher_steps {
                for g in teacher_grad.iter_mut() {
                    *g = 0.0;
                }
                let mut count = 0usize;
                for i in lo..hi {
                    let row = &real_data[i * d..i * d + d];
                    accum_bce_grad(&disc, tp, row, 1.0, &mut teacher_grad);
                    count += 1;
                }
                for f in &fake_batch {
                    accum_bce_grad(&disc, tp, f, 0.0, &mut teacher_grad);
                    count += 1;
                }
                if count > 0 {
                    let inv = 1.0 / count as f64;
                    for g in teacher_grad.iter_mut() {
                        *g *= inv;
                    }
                    sgd_update(tp, &teacher_grad, lr);
                }
            }
        }

        // 2. Label generated samples through the DP teacher-vote aggregation.
        let mut labelled: Vec<(Vec<f64>, f64)> = Vec::with_capacity(cfg.student_queries_per_round);
        for _ in 0..cfg.student_queries_per_round {
            let z = sample_latent(&mut latent_rng, cfg.latent_dim);
            let x = gnet.forward(&gp, &z).0;
            let votes: Vec<usize> = teachers
                .iter()
                .map(|tp| {
                    let (logit, _) = disc.forward(tp, &x);
                    usize::from(sigmoid(logit) > 0.5)
                })
                .collect();
            let label = pate_aggregate(&votes, &pate_cfg, &mut noise_rng)?;
            monitor.try_spend(eps_q, 0.0)?;
            labelled.push((x, label as f64));
        }

        // 3. Train the student discriminator on the DP labels.
        for _ in 0..cfg.student_steps {
            for g in teacher_grad.iter_mut() {
                *g = 0.0;
            }
            for (x, y) in &labelled {
                accum_bce_grad(&disc, &student, x, *y, &mut teacher_grad);
            }
            let inv = 1.0 / labelled.len() as f64;
            for g in teacher_grad.iter_mut() {
                *g *= inv;
            }
            sgd_update(&mut student, &teacher_grad, lr);
        }

        // 4. Update the generator against the student discriminator.
        for _ in 0..cfg.gen_steps {
            let latent_batch: Vec<Vec<f64>> = (0..cfg.batch_size)
                .map(|_| sample_latent(&mut latent_rng, cfg.latent_dim))
                .collect();
            generator_step(&gnet, &mut gp, &disc, &student, &latent_batch, lr);
        }
    }

    let cumulative_epsilon = monitor.spent().0;
    Ok(PateGanReport {
        generator: SyntheticGenerator {
            dims: gnet,
            params: gp,
            sample_seed: seed ^ 0x5359_4E54_4853_4D50,
        },
        cumulative_epsilon,
        queries,
        epsilon_per_query: eps_q,
        laplace_scale: scale,
    })
}

// ─── DP-GAN (DP-Adam discriminator) ───────────────────────────────────────────

/// Configuration of the DP-GAN synthesiser (DP-Adam discriminator).
#[derive(Debug, Clone)]
pub struct DpGanConfig {
    /// Width of each (real and synthetic) data row, `d ≥ 1`.
    pub data_dim: usize,
    /// Generator latent dimension, `≥ 1`.
    pub latent_dim: usize,
    /// Hidden-layer width for the generator and the discriminator, `≥ 1`.
    pub hidden_dim: usize,
    /// Number of training rounds, `≥ 1`.
    pub rounds: usize,
    /// DP-Adam discriminator steps per round, `≥ 1` (one moments-accountant
    /// composition each).
    pub disc_steps_per_round: usize,
    /// Generator SGD steps per round, `≥ 1`.
    pub gen_steps_per_round: usize,
    /// Real (and fake) minibatch size per discriminator step, `≥ 1`.
    pub batch_size: usize,
    /// Generator SGD learning rate `> 0`.
    pub gen_learning_rate: f64,
    /// DP-Adam configuration (σ, clip, lr, β₁, β₂, ε_adam) for the discriminator.
    pub adam: DpAdamConfig,
    /// Maximum integer Rényi order tracked by the moments accountant, `≥ 2`.
    pub max_order: usize,
}

impl DpGanConfig {
    /// A small, fast configuration for the given row width (used by the tests).
    #[must_use]
    pub fn small(data_dim: usize) -> Self {
        Self {
            data_dim,
            latent_dim: 3,
            hidden_dim: 8,
            rounds: 3,
            disc_steps_per_round: 4,
            gen_steps_per_round: 4,
            batch_size: 12,
            gen_learning_rate: 0.05,
            adam: DpAdamConfig {
                sigma: 1.0,
                grad_clip: 1.0,
                learning_rate: 0.05,
                beta1: 0.9,
                beta2: 0.999,
                epsilon_adam: 1e-8,
            },
            max_order: 32,
        }
    }

    /// Total number of DP-Adam discriminator steps over the whole run.
    #[must_use]
    pub fn steps(&self) -> usize {
        self.rounds * self.disc_steps_per_round
    }

    /// Validate the configuration (also round-trips the DP-Adam invariants).
    ///
    /// # Errors
    /// `InvalidParameter` for any zero dimension/step/batch/round count, a
    /// non-positive generator learning rate, or `max_order < 2`; propagates
    /// [`DpAdamConfig::new`].
    pub fn validate(&self) -> PrivacyResult<()> {
        if self.data_dim == 0 || self.latent_dim == 0 || self.hidden_dim == 0 {
            return Err(PrivacyError::InvalidParameter(
                "data_dim, latent_dim and hidden_dim must be ≥ 1".into(),
            ));
        }
        if self.rounds == 0
            || self.disc_steps_per_round == 0
            || self.gen_steps_per_round == 0
            || self.batch_size == 0
        {
            return Err(PrivacyError::InvalidParameter(
                "rounds, per-round step counts and batch_size must be ≥ 1".into(),
            ));
        }
        if self.gen_learning_rate <= 0.0 || !self.gen_learning_rate.is_finite() {
            return Err(PrivacyError::InvalidParameter(format!(
                "gen_learning_rate must be positive, got {}",
                self.gen_learning_rate
            )));
        }
        if self.max_order < 2 {
            return Err(PrivacyError::InvalidParameter(format!(
                "max_order must be ≥ 2, got {}",
                self.max_order
            )));
        }
        let a = &self.adam;
        DpAdamConfig::new(
            a.sigma,
            a.grad_clip,
            a.learning_rate,
            a.beta1,
            a.beta2,
            a.epsilon_adam,
        )?;
        Ok(())
    }
}

/// Result of [`dp_gan_train`].
#[derive(Debug, Clone)]
pub struct DpGanReport {
    /// The trained synthetic-data generator.
    pub generator: SyntheticGenerator,
    /// Cumulative ε at `delta` pulled from the crate's Sampled-Gaussian moments
    /// accountant after composing one RDP term per discriminator step.
    pub cumulative_epsilon: f64,
    /// Total number of DP-Adam discriminator steps composed.
    pub steps: usize,
    /// Poisson subsampling rate `q = batch_size / n_rows` used by the accountant.
    pub sampling_rate: f64,
    /// Gaussian noise multiplier σ of the DP-Adam discriminator.
    pub noise_multiplier: f64,
    /// Maximum Rényi order tracked by the accountant.
    pub max_order: usize,
    /// Target δ for the reported ε.
    pub delta: f64,
}

/// Train a DP-GAN synthesiser on `real_data` (flat `[n_rows × data_dim]`).
///
/// The discriminator is optimised with the crate's [`DpAdamState`] (per-sample
/// clipping + Gaussian noise) on per-sample gradients of the binary
/// real-vs-fake loss; the generator is updated from the discriminator's signal
/// (post-processing). Privacy is accounted with the crate's
/// [`SampledGaussianMechanism`] at subsampling rate `q = batch_size / n_rows`,
/// composing one RDP term per discriminator step, and reported at `delta`.
///
/// Per-sample clipping bounds the influence of any single real row to the clip
/// `C`; the fake-sample gradients are a deterministic function of the
/// (already-DP) generator and so carry no extra privacy cost — only the real
/// subsampling rate enters the accountant.
///
/// # Errors
/// - Propagates [`DpGanConfig::validate`].
/// - `InvalidDelta` if `delta ∉ (0, 1)`.
/// - `DimensionMismatch` if `real_data.len() != n_rows * cfg.data_dim`.
/// - `InvalidParameter` if `cfg.batch_size > n_rows`.
/// - Propagates optimiser / accountant errors.
pub fn dp_gan_train(
    real_data: &[f64],
    n_rows: usize,
    cfg: &DpGanConfig,
    seed: u64,
    delta: f64,
) -> PrivacyResult<DpGanReport> {
    cfg.validate()?;
    if !(delta > 0.0 && delta < 1.0) {
        return Err(PrivacyError::InvalidDelta(delta));
    }
    let d = cfg.data_dim;
    if real_data.len() != n_rows * d {
        return Err(PrivacyError::DimensionMismatch {
            expected: n_rows * d,
            got: real_data.len(),
        });
    }
    if cfg.batch_size > n_rows {
        return Err(PrivacyError::InvalidParameter(format!(
            "batch_size={} must be ≤ n_rows={n_rows}",
            cfg.batch_size
        )));
    }

    let gnet = GenDims {
        latent: cfg.latent_dim,
        hidden: cfg.hidden_dim,
        data: d,
    };
    let disc = DiscDims {
        data: d,
        hidden: cfg.hidden_dim,
    };
    let np = disc.n_params();

    let mut init_rng = LcgRng::new(seed ^ 0x4450_4741_4E00_0000);
    let mut latent_rng = LcgRng::new(seed ^ 0x4C41_5445_4E54_0002);
    let mut sample_rng = LcgRng::new(seed ^ 0x5355_4253_414D_504C);
    let mut noise_rng = LcgRng::new(seed ^ 0x4450_4144_414D_4E5A);

    let mut gp = init_gen(&gnet, &mut init_rng);
    let mut dstate = DpAdamState::new(np);
    let dinit = init_disc(&disc, &mut init_rng);
    dstate.params.copy_from_slice(&dinit);

    // Moments accountant: one Sampled-Gaussian RDP term per discriminator step.
    let q = cfg.batch_size as f64 / n_rows as f64;
    let acct_cfg = SampledGaussianConfig::new(q, cfg.adam.sigma, cfg.max_order)?;
    let mut accountant = SampledGaussianMechanism::new(&acct_cfg)?;

    let batch = cfg.batch_size;
    let combined = 2 * batch; // real (target 1) + fake (target 0) per step
    let mut batch_grads = vec![0.0_f64; combined * np];

    for _round in 0..cfg.rounds {
        for _ in 0..cfg.disc_steps_per_round {
            for v in batch_grads.iter_mut() {
                *v = 0.0;
            }
            // Real subsample (private, target 1).
            for s in 0..batch {
                let idx = (sample_rng.next_u64() as usize) % n_rows;
                let row = &real_data[idx * d..idx * d + d];
                let seg = &mut batch_grads[s * np..s * np + np];
                accum_bce_grad(&disc, &dstate.params, row, 1.0, seg);
            }
            // Fake batch (public post-processing of G, target 0).
            for s in 0..batch {
                let z = sample_latent(&mut latent_rng, cfg.latent_dim);
                let x = gnet.forward(&gp, &z).0;
                let off = (batch + s) * np;
                let seg = &mut batch_grads[off..off + np];
                accum_bce_grad(&disc, &dstate.params, &x, 0.0, seg);
            }
            // One DP-Adam step (per-sample clip + Gaussian noise + Adam update).
            dstate.step(&batch_grads, combined, &cfg.adam, &mut noise_rng)?;
            accountant.compose(1)?;
        }

        // Update the generator against the current DP discriminator.
        for _ in 0..cfg.gen_steps_per_round {
            let latent_batch: Vec<Vec<f64>> = (0..batch)
                .map(|_| sample_latent(&mut latent_rng, cfg.latent_dim))
                .collect();
            generator_step(
                &gnet,
                &mut gp,
                &disc,
                &dstate.params,
                &latent_batch,
                cfg.gen_learning_rate,
            );
        }
    }

    let cumulative_epsilon = accountant.get_epsilon(delta)?;
    Ok(DpGanReport {
        generator: SyntheticGenerator {
            dims: gnet,
            params: gp,
            sample_seed: seed ^ 0x4450_4753_594E_5448,
        },
        cumulative_epsilon,
        steps: cfg.steps(),
        sampling_rate: q,
        noise_multiplier: cfg.adam.sigma,
        max_order: cfg.max_order,
        delta,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic synthetic "private" dataset (in-process, no external
    /// download) — used only to drive the mechanisms, never asserted against.
    fn make_real(n_rows: usize, d: usize, seed: u64) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        (0..n_rows * d)
            .map(|_| rng.normal_pair().0.tanh())
            .collect()
    }

    // 1a. PATE-GAN privacy accounting is consistent with the crate's own
    //     BudgetMonitor for the same query schedule (pulled, not hardcoded).
    #[test]
    fn pate_gan_accounting_matches_budget_monitor() {
        let d = 4;
        let n_rows = 60;
        let real = make_real(n_rows, d, 1);
        let cfg = PateGanConfig::small(d);
        let report = pate_gan_train(&real, n_rows, &cfg, 7).expect("train");

        // Reconstruct the schedule independently through the same accountant.
        let q = cfg.queries();
        let eps_q = cfg.epsilon_per_query();
        let mut monitor = BudgetMonitor::new(cfg.epsilon_budget + 1.0, 0.0).expect("monitor");
        for _ in 0..q {
            monitor.try_spend(eps_q, 0.0).expect("spend");
        }
        assert_eq!(report.queries, q);
        assert!(
            (monitor.spent().0 - report.cumulative_epsilon).abs() < 1e-9,
            "cumulative ε {} must equal the accountant's sum {}",
            report.cumulative_epsilon,
            monitor.spent().0
        );
        assert!(report.cumulative_epsilon > 0.0 && report.cumulative_epsilon.is_finite());
    }

    // 1b. Tighter privacy budget ⇒ more teacher-vote noise (monotone), and the
    //     trained report reflects the smaller per-query ε.
    #[test]
    fn pate_gan_tighter_budget_more_noise() {
        let d = 4;
        let n_rows = 60;
        let real = make_real(n_rows, d, 2);
        let loose = PateGanConfig::small(d);
        let mut tight = loose.clone();
        tight.epsilon_budget = loose.epsilon_budget / 4.0;

        assert!(
            tight.laplace_scale() > loose.laplace_scale(),
            "tighter budget must raise the noise scale: {} > {}",
            tight.laplace_scale(),
            loose.laplace_scale()
        );

        let r_loose = pate_gan_train(&real, n_rows, &loose, 3).expect("loose");
        let r_tight = pate_gan_train(&real, n_rows, &tight, 3).expect("tight");
        assert!(r_tight.epsilon_per_query < r_loose.epsilon_per_query);
        assert!(r_tight.laplace_scale > r_loose.laplace_scale);
        assert!(r_tight.cumulative_epsilon < r_loose.cumulative_epsilon);
    }

    // 2. PATE-GAN sampling: exactly n rows of the right width, all finite,
    //    deterministic under the fixed seed.
    #[test]
    fn pate_gan_sample_structure_finite_deterministic() {
        let d = 5;
        let n_rows = 50;
        let real = make_real(n_rows, d, 4);
        let report = pate_gan_train(&real, n_rows, &PateGanConfig::small(d), 9).expect("train");

        let rows = report.generator.sample(13);
        assert_eq!(rows.len(), 13);
        for r in &rows {
            assert_eq!(r.len(), d);
            for &v in r {
                assert!(v.is_finite(), "synthetic value must be finite, got {v}");
            }
        }
        // Deterministic across calls.
        assert_eq!(rows, report.generator.sample(13));
        // Deterministic across identically-seeded trainings.
        let again = pate_gan_train(&real, n_rows, &PateGanConfig::small(d), 9).expect("train");
        assert_eq!(rows, again.generator.sample(13));
    }

    // 3. DP-GAN privacy accounting is consistent with the crate's own
    //    Sampled-Gaussian moments accountant for the same schedule.
    #[test]
    fn dp_gan_accounting_matches_moments_accountant() {
        let d = 4;
        let n_rows = 60;
        let real = make_real(n_rows, d, 5);
        let cfg = DpGanConfig::small(d);
        let delta = 1e-5;
        let report = dp_gan_train(&real, n_rows, &cfg, 11, delta).expect("train");

        let q = cfg.batch_size as f64 / n_rows as f64;
        let acct_cfg = SampledGaussianConfig::new(q, cfg.adam.sigma, cfg.max_order).expect("cfg");
        let mut acct = SampledGaussianMechanism::new(&acct_cfg).expect("acct");
        acct.compose(cfg.steps()).expect("compose");
        let eps = acct.get_epsilon(delta).expect("eps");

        assert_eq!(report.steps, cfg.steps());
        assert!(
            (eps - report.cumulative_epsilon).abs() < 1e-9,
            "cumulative ε {} must equal the moments accountant's {eps}",
            report.cumulative_epsilon
        );
        assert!(report.cumulative_epsilon > 0.0 && report.cumulative_epsilon.is_finite());
    }

    // 3b. DP-GAN sampling structure + finiteness + determinism.
    #[test]
    fn dp_gan_sample_structure_finite_deterministic() {
        let d = 6;
        let n_rows = 48;
        let real = make_real(n_rows, d, 6);
        let cfg = DpGanConfig::small(d);
        let a = dp_gan_train(&real, n_rows, &cfg, 21, 1e-5).expect("a");

        let rows = a.generator.sample(9);
        assert_eq!(rows.len(), 9);
        for r in &rows {
            assert_eq!(r.len(), d);
            for &v in r {
                assert!(v.is_finite());
            }
        }
        assert_eq!(rows, a.generator.sample(9));
        let b = dp_gan_train(&real, n_rows, &cfg, 21, 1e-5).expect("b");
        assert_eq!(rows, b.generator.sample(9));
    }

    // 3c. DP-GAN: a larger noise multiplier ⇒ smaller reported ε (monotone),
    //     mirroring the moments accountant.
    #[test]
    fn dp_gan_more_noise_smaller_epsilon() {
        let d = 4;
        let n_rows = 60;
        let real = make_real(n_rows, d, 7);
        let mut lo = DpGanConfig::small(d);
        lo.adam.sigma = 0.8;
        let mut hi = lo.clone();
        hi.adam.sigma = 4.0;
        let r_lo = dp_gan_train(&real, n_rows, &lo, 1, 1e-5).expect("lo");
        let r_hi = dp_gan_train(&real, n_rows, &hi, 1, 1e-5).expect("hi");
        assert!(
            r_hi.cumulative_epsilon < r_lo.cumulative_epsilon,
            "more noise → smaller ε: {} < {}",
            r_hi.cumulative_epsilon,
            r_lo.cumulative_epsilon
        );
    }

    // 4. Mechanism correctness: with a unanimous teacher ensemble, the DP label
    //    equals the teachers' label w.h.p. at low noise (high ε), degrading
    //    toward random as ε → 0.
    #[test]
    fn dp_voting_unanimous_correct_then_degrades() {
        let votes = vec![1usize; 7]; // unanimous class 1
        let trials = 400;

        // Low noise (high ε): near-certain agreement with the teachers.
        let mut rng_lo = LcgRng::new(123);
        let mut correct_lo = 0usize;
        for _ in 0..trials {
            if dp_label_from_votes(&votes, 2, 50.0, &mut rng_lo).expect("lo") == 1 {
                correct_lo += 1;
            }
        }
        // High noise (ε → 0): degrades toward a coin flip.
        let mut rng_hi = LcgRng::new(123);
        let mut correct_hi = 0usize;
        for _ in 0..trials {
            if dp_label_from_votes(&votes, 2, 0.001, &mut rng_hi).expect("hi") == 1 {
                correct_hi += 1;
            }
        }

        assert!(
            correct_lo as f64 > 0.95 * trials as f64,
            "low noise should agree with the teachers almost always, got {correct_lo}/{trials}"
        );
        assert!(
            correct_hi < correct_lo,
            "high noise must degrade accuracy: {correct_hi} < {correct_lo}"
        );
        // Clearly below near-certain — moving toward the random ~50% baseline.
        assert!(
            (correct_hi as f64) < 0.75 * trials as f64,
            "high noise should approach random, got {correct_hi}/{trials}"
        );
    }

    // 5. Input validation for both drivers and the labelling primitive.
    #[test]
    fn validation_errors() {
        let d = 4;
        let n_rows = 40;
        let real = make_real(n_rows, d, 8);

        // Dimension mismatch.
        let pcfg = PateGanConfig::small(d);
        assert!(matches!(
            pate_gan_train(&real[..10], 3, &pcfg, 1),
            Err(PrivacyError::DimensionMismatch { .. })
        ));
        // Too few rows for the teachers.
        let tiny = make_real(3, d, 8);
        let mut few = PateGanConfig::small(d);
        few.n_teachers = 5;
        assert!(pate_gan_train(&tiny, 3, &few, 1).is_err());
        // Bad budget.
        let mut bad_budget = PateGanConfig::small(d);
        bad_budget.epsilon_budget = 0.0;
        assert!(bad_budget.validate().is_err());

        // DP-GAN: bad delta and oversized batch.
        let dcfg = DpGanConfig::small(d);
        assert!(matches!(
            dp_gan_train(&real, n_rows, &dcfg, 1, 0.0),
            Err(PrivacyError::InvalidDelta(_))
        ));
        let mut big_batch = DpGanConfig::small(d);
        big_batch.batch_size = n_rows + 1;
        assert!(dp_gan_train(&real, n_rows, &big_batch, 1, 1e-5).is_err());

        // Labelling primitive rejects non-positive ε.
        let mut rng = LcgRng::new(1);
        assert!(dp_label_from_votes(&[1, 1, 0], 2, 0.0, &mut rng).is_err());
        assert!(dp_label_from_votes(&[1, 1, 0], 2, -1.0, &mut rng).is_err());
    }

    // 6. The two networks really learn: hand-checked gradient sanity — a single
    //    BCE SGD step moves a discriminator's logit on a fixed point toward its
    //    target, and a generator step changes the generator's parameters.
    #[test]
    fn networks_have_real_gradients() {
        let mut rng = LcgRng::new(77);
        let disc = DiscDims { data: 3, hidden: 6 };
        let mut dp = init_disc(&disc, &mut rng);
        let x = vec![0.5, -0.2, 0.7];
        let (logit0, _) = disc.forward(&dp, &x);
        // Several SGD steps toward target 1 should raise the logit.
        for _ in 0..20 {
            let mut grad = vec![0.0; disc.n_params()];
            accum_bce_grad(&disc, &dp, &x, 1.0, &mut grad);
            sgd_update(&mut dp, &grad, 0.3);
        }
        let (logit1, _) = disc.forward(&dp, &x);
        assert!(
            logit1 > logit0,
            "BCE-toward-1 must raise the logit: {logit1} > {logit0}"
        );

        // Generator step changes generator params (non-trivial back-prop).
        let gnet = GenDims {
            latent: 3,
            hidden: 6,
            data: 3,
        };
        let mut gp = init_gen(&gnet, &mut rng);
        let before = gp.clone();
        let latent_batch: Vec<Vec<f64>> = (0..8).map(|_| sample_latent(&mut rng, 3)).collect();
        generator_step(&gnet, &mut gp, &disc, &dp, &latent_batch, 0.1);
        assert_ne!(before, gp, "generator parameters must move after a step");
        for &v in &gp {
            assert!(v.is_finite());
        }
    }
}
