//! MoCo v3 — Chen et al. 2021 — "An Empirical Study of Training Self-Supervised
//! Vision Transformers".
//!
//! MoCo v3 retains the momentum encoder of MoCo v1/v2 but **removes the memory
//! queue**: all N keys come directly from the current batch (like SimCLR). It
//! also adds a **predictor head** on top of the online network (like BYOL) and
//! uses a **symmetric InfoNCE loss** computed between two augmented views.
//!
//! ## Architecture
//! ```text
//!   View 1 ──▶ encoder ──▶ projector ──▶ predictor ──▶ q1
//!   View 2 ──▶ [EMA encoder + projector, stop-grad] ──▶ k2
//!
//!   Loss = (InfoNCE(q1, sg(k2)) + InfoNCE(q2, sg(k1))) / 2
//! ```
//!
//! ## Within-batch InfoNCE
//! Given queries Q `[N, D]` and keys K `[N, D]` (both L2-normalised):
//! ```text
//!   L = -(1/N) Σ_i log(exp(Q_i·K_i/τ) / Σ_j exp(Q_i·K_j/τ))
//! ```
//! Diagonal entries are positives; off-diagonal within each row are negatives.
//! A numerically stable log-sum-exp is used throughout.

use crate::error::{SslError, SslResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Hyper-parameters for MoCo v3 training.
///
/// Temperature is higher than MoCo v1 (`0.07`) because MoCo v3 is tuned for
/// ViT training where a sharper distribution often hurts. Momentum of `0.99`
/// gives a slower EMA than v1/v2's `0.999`, consistent with Chen et al. 2021.
#[derive(Debug, Clone, PartialEq)]
pub struct MocoV3Config {
    /// Softmax temperature τ for the InfoNCE loss (default 0.2).
    pub temperature: f32,
    /// EMA momentum for the target encoder update (default 0.99).
    pub momentum: f32,
}

impl Default for MocoV3Config {
    fn default() -> Self {
        Self {
            temperature: 0.2,
            momentum: 0.99,
        }
    }
}

// ─── Within-batch InfoNCE loss ────────────────────────────────────────────────

/// Compute the within-batch InfoNCE loss and top-1 accuracy for one direction.
///
/// `q` and `k` are `[n, d]` row-major slices of query and key embeddings. Both
/// are L2-normalised internally. The `n×n` similarity matrix is built; the
/// diagonal is the positive pair. Off-diagonal entries within each row serve as
/// negatives. A numerically stable log-sum-exp is applied per row.
///
/// Returns `(loss, top1_accuracy)` where `top1_accuracy` is the fraction of
/// queries whose highest cosine similarity across all keys matches the
/// diagonal (correct positive).
///
/// # Errors
/// - [`SslError::EmptyInput`] if `n == 0` or `d == 0`.
/// - [`SslError::DimensionMismatch`] if slice lengths disagree with `n * d`.
/// - [`SslError::BatchTooSmall`] if `n < 2` (need at least one negative).
/// - [`SslError::InvalidTemperature`] if `temperature` is `<= 0` or non-finite.
pub fn moco_v3_loss(
    q: &[f32],
    k: &[f32],
    n: usize,
    d: usize,
    config: &MocoV3Config,
) -> SslResult<(f32, f32)> {
    validate_inputs(q, k, n, d, config.temperature)?;
    let q_norm = l2_normalize_rows(q, n, d);
    let k_norm = l2_normalize_rows(k, n, d);
    within_batch_info_nce(&q_norm, &k_norm, n, d, config.temperature)
}

/// Symmetric MoCo v3 loss: mean of both InfoNCE directions.
///
/// Computes `(InfoNCE(q1, sg(k2)) + InfoNCE(q2, sg(k1))) / 2` and returns the
/// mean `(loss, top1_accuracy)` across both directions.
///
/// # Arguments
/// * `q1` — online predictions from view 1, `[n, d]` row-major.
/// * `k2` — target projections from view 2 (stop-gradient), `[n, d]`.
/// * `q2` — online predictions from view 2, `[n, d]`.
/// * `k1` — target projections from view 1 (stop-gradient), `[n, d]`.
/// * `n`  — batch size (must be ≥ 2).
/// * `d`  — feature dimension (must be ≥ 1).
///
/// # Errors
/// - [`SslError::EmptyInput`] if `n == 0` or `d == 0`.
/// - [`SslError::DimensionMismatch`] for any shape mismatch.
/// - [`SslError::BatchTooSmall`] if `n < 2`.
/// - [`SslError::InvalidTemperature`] if temperature is invalid.
pub fn moco_v3_symmetric_loss(
    q1: &[f32],
    k2: &[f32],
    q2: &[f32],
    k1: &[f32],
    n: usize,
    d: usize,
    config: &MocoV3Config,
) -> SslResult<(f32, f32)> {
    // Validate all four inputs share the same shape.
    validate_inputs(q1, k2, n, d, config.temperature)?;
    if q2.len() != n * d {
        return Err(SslError::DimensionMismatch {
            expected: n * d,
            got: q2.len(),
        });
    }
    if k1.len() != n * d {
        return Err(SslError::DimensionMismatch {
            expected: n * d,
            got: k1.len(),
        });
    }

    let q1_n = l2_normalize_rows(q1, n, d);
    let k2_n = l2_normalize_rows(k2, n, d);
    let q2_n = l2_normalize_rows(q2, n, d);
    let k1_n = l2_normalize_rows(k1, n, d);

    let (loss_a, acc_a) = within_batch_info_nce(&q1_n, &k2_n, n, d, config.temperature)?;
    let (loss_b, acc_b) = within_batch_info_nce(&q2_n, &k1_n, n, d, config.temperature)?;

    Ok(((loss_a + loss_b) * 0.5, (acc_a + acc_b) * 0.5))
}

// ─── Parameter state ─────────────────────────────────────────────────────────

/// Lightweight parameter state for a MoCo v3 online+target encoder pair.
///
/// In a full training system the online and target encoder would be neural
/// network weight tensors; here we represent them as flat `Vec<f32>` buffers to
/// demonstrate the EMA momentum update and cosine schedule without coupling to
/// any specific model architecture.
#[derive(Debug, Clone)]
pub struct MocoV3State {
    /// Flattened online network parameters.
    pub online_params: Vec<f32>,
    /// Flattened target (EMA) network parameters. Initialised identically to
    /// `online_params` and updated via [`MocoV3State::momentum_update`].
    pub target_params: Vec<f32>,
    /// Global training step counter (incremented externally by the caller).
    pub step: usize,
}

impl MocoV3State {
    /// Allocate a new state with `n_params` parameters drawn uniformly from
    /// `[-0.01, 0.01]`. Online and target are initialised to the **same**
    /// random values (matching the practice of copying the online encoder at
    /// the start of training).
    ///
    /// # Panics
    /// Never panics (uses `LcgRng::next_f32` which is always in `[0, 1)`).
    #[must_use]
    pub fn new(n_params: usize, rng: &mut LcgRng) -> Self {
        let mut params = vec![0.0_f32; n_params];
        for v in params.iter_mut() {
            // Map [0, 1) → [-0.01, 0.01].
            *v = (rng.next_f32() - 0.5) * 0.02;
        }
        Self {
            online_params: params.clone(),
            target_params: params,
            step: 0,
        }
    }

    /// EMA update: `target ← momentum * target + (1 − momentum) * online`.
    ///
    /// This is the stop-gradient update applied after each optimiser step on
    /// the online network. Only `target_params` is modified; `online_params`
    /// is read-only here.
    pub fn momentum_update(&mut self, momentum: f32) {
        let one_minus = 1.0 - momentum;
        for (t, o) in self.target_params.iter_mut().zip(self.online_params.iter()) {
            *t = momentum * *t + one_minus * *o;
        }
    }

    /// Cosine annealing schedule for the EMA momentum coefficient.
    ///
    /// Starts at `base_momentum` (e.g., `0.99`) and increases to `1.0` as
    /// `step` → `max_steps`, following:
    /// ```text
    ///   m(t) = 1 − (1 − base_momentum) * (cos(π * t / T) + 1) / 2
    /// ```
    /// At `t = 0`: `m = base_momentum`. At `t = T`: `m ≈ 1.0`.
    #[must_use]
    pub fn cosine_momentum_schedule(step: usize, max_steps: usize, base_momentum: f32) -> f32 {
        if max_steps == 0 {
            return 1.0;
        }
        let t = step.min(max_steps) as f64;
        let cap = max_steps as f64;
        let decay = 1.0 - base_momentum as f64;
        let cosine_factor = (std::f64::consts::PI * t / cap).cos() + 1.0;
        let m = 1.0 - decay * cosine_factor * 0.5;
        m as f32
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Validate that `q` and `k` have the expected shape and temperature is valid.
#[inline]
fn validate_inputs(q: &[f32], k: &[f32], n: usize, d: usize, temperature: f32) -> SslResult<()> {
    if n == 0 || d == 0 {
        return Err(SslError::EmptyInput);
    }
    if n < 2 {
        return Err(SslError::BatchTooSmall);
    }
    if !(temperature.is_finite() && temperature > 0.0) {
        return Err(SslError::InvalidTemperature { temp: temperature });
    }
    if q.len() != n * d {
        return Err(SslError::DimensionMismatch {
            expected: n * d,
            got: q.len(),
        });
    }
    if k.len() != n * d {
        return Err(SslError::DimensionMismatch {
            expected: n * d,
            got: k.len(),
        });
    }
    Ok(())
}

/// Return a new buffer where each row of the `[n, d]` row-major matrix is
/// L2-normalised. Rows with near-zero norm are left as-is (effectively zero
/// vectors after normalisation).
fn l2_normalize_rows(z: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut out = z.to_vec();
    for i in 0..n {
        let row = &mut out[i * d..(i + 1) * d];
        let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        let inv = if norm > 1e-12 { 1.0 / norm } else { 1.0 };
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
    out
}

/// Core within-batch InfoNCE implementation.
///
/// Expects `q` and `k` to already be L2-normalised `[n, d]` matrices.
/// Builds the full `n × n` cosine similarity matrix, applies temperature
/// scaling, then computes the cross-entropy loss for each row treating the
/// diagonal as the positive. Uses f64 accumulators for stability.
///
/// Returns `(mean_loss, top1_accuracy)`.
fn within_batch_info_nce(
    q: &[f32],
    k: &[f32],
    n: usize,
    d: usize,
    temperature: f32,
) -> SslResult<(f32, f32)> {
    let inv_t = 1.0_f64 / temperature as f64;

    // Build the n×n similarity matrix in f64 for numerical precision.
    // sim[i * n + j] = Q_i · K_j / τ
    let mut sim = vec![0.0_f64; n * n];
    for i in 0..n {
        let q_row = &q[i * d..(i + 1) * d];
        for j in 0..n {
            let k_row = &k[j * d..(j + 1) * d];
            let dot: f64 = q_row
                .iter()
                .zip(k_row.iter())
                .map(|(&a, &b)| (a as f64) * (b as f64))
                .sum();
            sim[i * n + j] = dot * inv_t;
        }
    }

    let mut total_loss = 0.0_f64;
    let mut n_correct: usize = 0;

    for i in 0..n {
        let row = &sim[i * n..(i + 1) * n];

        // Numerically stable log-sum-exp: subtract row maximum before exp.
        let max_v = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let sum_exp: f64 = row.iter().map(|&s| (s - max_v).exp()).sum();
        let log_z = max_v + sum_exp.ln();

        // Positive logit is the diagonal entry row[i].
        let positive_logit = row[i];
        total_loss += -(positive_logit - log_z);

        // Top-1: the column with the highest logit should be i.
        let argmax = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        if argmax == i {
            n_correct += 1;
        }
    }

    let loss = (total_loss / n as f64) as f32;
    let top1 = n_correct as f32 / n as f32;

    if !loss.is_finite() {
        return Err(SslError::NanEncountered {
            location: "moco_v3_info_nce",
        });
    }

    Ok((loss, top1))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // ── Helper: build aligned one-hot query/key pair ──────────────────────────

    /// Construct `n` one-hot unit vectors of dimension `d` (cycling through dims).
    fn one_hot_batch(n: usize, d: usize) -> Vec<f32> {
        let mut z = vec![0.0_f32; n * d];
        for i in 0..n {
            z[i * d + i % d] = 1.0;
        }
        z
    }

    // ─── Loss correctness ────────────────────────────────────────────────────

    /// When Q and K are identical and each row is a distinct one-hot vector, the
    /// diagonal always has the maximum similarity → top-1 accuracy = 1.0.
    #[test]
    fn moco_v3_loss_aligned_batch_high_accuracy() {
        let n = 8;
        let d = 16;
        let q = one_hot_batch(n, d);
        let k = q.clone();
        let cfg = MocoV3Config::default();
        let (loss, acc) = moco_v3_loss(&q, &k, n, d, &cfg).unwrap();
        assert!(loss.is_finite(), "loss must be finite, got {loss}");
        assert!(
            (acc - 1.0).abs() < 1e-6,
            "top-1 accuracy must be 1.0 for identical aligned batch, got {acc}"
        );
    }

    /// Loss is finite for a random (non-degenerate) batch.
    #[test]
    fn moco_v3_loss_output_finite() {
        let n = 4;
        let d = 8;
        let mut rng = LcgRng::new(42);
        let mut q = vec![0.0_f32; n * d];
        let mut k = vec![0.0_f32; n * d];
        rng.fill_normal(&mut q);
        rng.fill_normal(&mut k);
        let cfg = MocoV3Config::default();
        let (loss, acc) = moco_v3_loss(&q, &k, n, d, &cfg).unwrap();
        assert!(loss.is_finite(), "loss = {loss}");
        assert!((0.0..=1.0).contains(&acc), "acc = {acc}");
    }

    /// Lower temperature makes the distribution sharper: the loss should be
    /// lower (more peaked) for aligned pairs at τ=0.07 than at τ=0.5.
    #[test]
    fn moco_v3_loss_temperature_effect() {
        let n = 4;
        let d = 8;
        let q = one_hot_batch(n, d);
        let k = q.clone();

        let cfg_low = MocoV3Config {
            temperature: 0.07,
            ..Default::default()
        };
        let cfg_high = MocoV3Config {
            temperature: 0.5,
            ..Default::default()
        };

        let (loss_low, _) = moco_v3_loss(&q, &k, n, d, &cfg_low).unwrap();
        let (loss_high, _) = moco_v3_loss(&q, &k, n, d, &cfg_high).unwrap();

        // With a sharper distribution, perfectly aligned pairs yield lower loss.
        assert!(
            loss_low < loss_high,
            "lower temperature should produce lower loss for aligned pairs, \
             got loss_low={loss_low} loss_high={loss_high}"
        );
    }

    // ─── Symmetric loss ───────────────────────────────────────────────────────

    /// Swapping (q1,k2,q2,k1) → (q2,k1,q1,k2) must produce the same result.
    #[test]
    fn moco_v3_symmetric_loss_symmetry() {
        let n = 4;
        let d = 8;
        let mut rng = LcgRng::new(7);
        let mut q1 = vec![0.0_f32; n * d];
        let mut k2 = vec![0.0_f32; n * d];
        let mut q2 = vec![0.0_f32; n * d];
        let mut k1 = vec![0.0_f32; n * d];
        rng.fill_normal(&mut q1);
        rng.fill_normal(&mut k2);
        rng.fill_normal(&mut q2);
        rng.fill_normal(&mut k1);

        let cfg = MocoV3Config::default();
        let (loss_ab, acc_ab) = moco_v3_symmetric_loss(&q1, &k2, &q2, &k1, n, d, &cfg).unwrap();
        let (loss_ba, acc_ba) = moco_v3_symmetric_loss(&q2, &k1, &q1, &k2, n, d, &cfg).unwrap();

        assert!(
            (loss_ab - loss_ba).abs() < 1e-5,
            "symmetric loss must be order-invariant: {loss_ab} vs {loss_ba}"
        );
        assert!(
            (acc_ab - acc_ba).abs() < 1e-5,
            "symmetric accuracy must be order-invariant: {acc_ab} vs {acc_ba}"
        );
    }

    /// Symmetric loss is finite for random inputs.
    #[test]
    fn moco_v3_symmetric_loss_finite() {
        let n = 6;
        let d = 12;
        let mut rng = LcgRng::new(99);
        let mut q1 = vec![0.0_f32; n * d];
        let mut k2 = vec![0.0_f32; n * d];
        let mut q2 = vec![0.0_f32; n * d];
        let mut k1 = vec![0.0_f32; n * d];
        rng.fill_normal(&mut q1);
        rng.fill_normal(&mut k2);
        rng.fill_normal(&mut q2);
        rng.fill_normal(&mut k1);
        let cfg = MocoV3Config::default();
        let (loss, acc) = moco_v3_symmetric_loss(&q1, &k2, &q2, &k1, n, d, &cfg).unwrap();
        assert!(loss.is_finite(), "loss = {loss}");
        assert!((0.0..=1.0).contains(&acc), "acc = {acc}");
    }

    // ─── MocoV3State ─────────────────────────────────────────────────────────

    /// After many EMA steps with a fixed (different) online parameter, the
    /// target must converge toward the online value.
    #[test]
    fn moco_v3_state_momentum_update_convergence() {
        let mut rng = LcgRng::new(0);
        let n = 16;
        let mut state = MocoV3State::new(n, &mut rng);

        // Drive online params to a constant value.
        for v in state.online_params.iter_mut() {
            *v = 1.0;
        }
        // Initial target is close to 0 (random in [-0.01, 0.01]).
        let initial_dist: f32 = state
            .target_params
            .iter()
            .zip(state.online_params.iter())
            .map(|(t, o)| (t - o).abs())
            .sum::<f32>();

        let momentum = 0.9_f32;
        for _ in 0..100 {
            state.momentum_update(momentum);
        }

        let final_dist: f32 = state
            .target_params
            .iter()
            .zip(state.online_params.iter())
            .map(|(t, o)| (t - o).abs())
            .sum::<f32>();

        assert!(
            final_dist < initial_dist,
            "target must converge toward online: initial_dist={initial_dist}, \
             final_dist={final_dist}"
        );
        // After 100 steps at momentum=0.9, distance should be tiny.
        assert!(
            final_dist < 1e-3,
            "after 100 EMA steps target must be nearly equal to online; \
             final_dist={final_dist}"
        );
    }

    /// Cosine schedule endpoints: at step=0 → base_momentum; at step=max_steps → 1.0.
    #[test]
    fn moco_v3_cosine_momentum_schedule_endpoints() {
        let base = 0.99_f32;
        let max_steps = 1000_usize;

        let m0 = MocoV3State::cosine_momentum_schedule(0, max_steps, base);
        let m_end = MocoV3State::cosine_momentum_schedule(max_steps, max_steps, base);

        assert!(
            (m0 - base).abs() < 1e-6,
            "at step=0, schedule must equal base_momentum; got {m0}"
        );
        assert!(
            (m_end - 1.0_f32).abs() < 1e-6,
            "at step=max_steps, schedule must equal 1.0; got {m_end}"
        );
    }

    /// Cosine momentum schedule is monotone non-decreasing.
    #[test]
    fn moco_v3_cosine_momentum_monotone_increasing() {
        let base = 0.99_f32;
        let max_steps = 200_usize;
        let mut prev = MocoV3State::cosine_momentum_schedule(0, max_steps, base);
        for step in 1..=max_steps {
            let curr = MocoV3State::cosine_momentum_schedule(step, max_steps, base);
            assert!(
                curr >= prev - 1e-7,
                "schedule must be non-decreasing: step={step}, prev={prev}, curr={curr}"
            );
            prev = curr;
        }
    }

    // ─── Error handling ───────────────────────────────────────────────────────

    /// n=0 or d=0 must return EmptyInput.
    #[test]
    fn empty_input_returns_error() {
        let cfg = MocoV3Config::default();
        assert!(moco_v3_loss(&[], &[], 0, 0, &cfg).is_err());
        assert!(moco_v3_symmetric_loss(&[], &[], &[], &[], 0, 0, &cfg).is_err());
    }

    /// Mismatched slice length must return DimensionMismatch.
    #[test]
    fn dimension_mismatch_returns_error() {
        let cfg = MocoV3Config::default();
        // q has 3 elements but n*d = 1*2 = 2 → mismatch.
        let q = vec![1.0_f32, 0.0, 0.5];
        let k = vec![1.0_f32, 0.0];
        let result = moco_v3_loss(&q, &k, 1, 2, &cfg);
        // n=1 fails BatchTooSmall before DimensionMismatch — use n=2 instead.
        let q2 = vec![1.0_f32; 5]; // 5 ≠ 2*2=4
        let k2 = vec![1.0_f32; 4];
        let err = moco_v3_loss(&q2, &k2, 2, 2, &cfg);
        assert!(
            matches!(err, Err(SslError::DimensionMismatch { .. })),
            "expected DimensionMismatch, got {err:?}"
        );
        // Suppress unused variable warning from earlier binding.
        let _ = result;
    }

    // ─── Additional robustness tests ─────────────────────────────────────────

    /// Batch size of exactly 2 (minimum valid) must succeed.
    #[test]
    fn moco_v3_loss_minimum_batch_size() {
        let n = 2;
        let d = 4;
        let q = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let k = q.clone();
        let cfg = MocoV3Config::default();
        let (loss, acc) = moco_v3_loss(&q, &k, n, d, &cfg).unwrap();
        assert!(loss.is_finite(), "loss = {loss}");
        assert!((acc - 1.0).abs() < 1e-6, "acc = {acc}");
    }

    /// State initialisation creates matching online/target params.
    #[test]
    fn moco_v3_state_new_initialises_equal_params() {
        let mut rng = LcgRng::new(123);
        let state = MocoV3State::new(64, &mut rng);
        assert_eq!(state.online_params, state.target_params);
        assert_eq!(state.step, 0);
        // All values should be in [-0.01, 0.01].
        for &v in &state.online_params {
            assert!((-0.01..=0.01).contains(&v), "param out of range: {v}");
        }
    }
}
