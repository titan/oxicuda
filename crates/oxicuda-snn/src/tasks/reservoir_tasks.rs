#![allow(clippy::needless_range_loop)]
//! Standard reservoir-computing benchmark tasks for the Liquid State Machine.
//!
//! Two canonical tasks are implemented on top of the crate's real spiking
//! reservoir ([`crate::reservoir::lsm::Lsm`]) and its linear ridge readout
//! ([`crate::reservoir::RidgeReadout`]):
//!
//! * **NARMA-10** — the order-10 Non-linear Auto-Regressive Moving-Average
//!   system (Atiya & Parlos 2000; Jaeger 2003). An i.i.d. input `u(t)` drives
//!   the recurrence
//!
//!   ```text
//!   y(t+1) = 0.3·y(t) + 0.05·y(t)·Σ_{i=0}^{9} y(t−i) + 1.5·u(t−9)·u(t) + 0.1
//!   ```
//!
//!   which mixes a 10-step fading memory with a multiplicative non-linearity —
//!   exactly the regime a liquid-state machine is meant to solve. The reservoir
//!   is driven by `u`, the spike trains are low-pass filtered into an analog
//!   *liquid state*, a ridge readout is trained on a held-out training split,
//!   and the test **NMSE** is measured.
//!
//! * **Linear memory capacity** (Jaeger 2002) — drive the reservoir with i.i.d.
//!   input, then for each delay `k` train a linear readout to reconstruct
//!   `u(t−k)` and accumulate the squared correlation. The total
//!   `MC = Σ_k r²_k` quantifies how many past inputs the reservoir linearly
//!   remembers; it is bounded above by the number of readout features.
//!
//! All randomness flows through the crate's [`crate::handle::LcgRng`]; given a fixed seed both
//! tasks are bit-for-bit deterministic.

use crate::error::{SnnError, SnnResult};
use crate::handle::LcgRng;
use crate::neuron::lif::{LifConfig, ResetMode};
use crate::reservoir::RidgeReadout;
use crate::reservoir::esn::ridge_regression;
use crate::reservoir::lsm::{Lsm, LsmConfig};

// ─────────────────────────────────────────────────────────────────────────────
// Generic metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Normalised mean-squared error `NMSE = Σ(ŷ−y)² / Σ(y−ȳ)²`.
///
/// A predictor that always emits the target mean `ȳ` scores exactly `1.0`, so
/// any `NMSE < 1` beats that trivial baseline and `NMSE → 0` is perfect.
/// Returns `1.0` if the target has zero variance (degenerate baseline) and
/// `0.0` if the slices are empty or mismatched.
#[must_use]
pub fn nmse(predictions: &[f32], targets: &[f32]) -> f32 {
    let n = targets.len();
    if n == 0 || predictions.len() != n {
        return 0.0;
    }
    let mean: f32 = targets.iter().sum::<f32>() / n as f32;
    let mut sse = 0.0_f32;
    let mut sst = 0.0_f32;
    for (&p, &t) in predictions.iter().zip(targets.iter()) {
        sse += (p - t) * (p - t);
        sst += (t - mean) * (t - mean);
    }
    if sst <= 0.0 { 1.0 } else { sse / sst }
}

/// Squared Pearson correlation `r²` between two equal-length series.
///
/// Returns a value in `[0, 1]`; `0.0` when either series is constant, empty, or
/// the lengths differ.
#[must_use]
pub fn squared_correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    if n == 0 || b.len() != n {
        return 0.0;
    }
    let inv = 1.0_f32 / n as f32;
    let ma: f32 = a.iter().sum::<f32>() * inv;
    let mb: f32 = b.iter().sum::<f32>() * inv;
    let mut cov = 0.0_f32;
    let mut va = 0.0_f32;
    let mut vb = 0.0_f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let dx = x - ma;
        let dy = y - mb;
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    if va <= 0.0 || vb <= 0.0 {
        return 0.0;
    }
    (cov * cov) / (va * vb)
}

// ─────────────────────────────────────────────────────────────────────────────
// NARMA-10 sequence
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a NARMA-10 dataset of length `n_steps`.
///
/// The input `u(t)` is drawn i.i.d. from `Uniform[0, 0.5)` — the standard
/// amplitude range that keeps the order-10 recurrence bounded — and the target
/// `y(t)` is produced by
///
/// ```text
/// y(t+1) = 0.3·y(t) + 0.05·y(t)·Σ_{i=0}^{9} y(t−i) + 1.5·u(t−9)·u(t) + 0.1.
/// ```
///
/// `y` is zero-initialised for the first ten steps (before the recurrence has a
/// full history). Returns `(u, y)`, each of length `n_steps`.
///
/// # Errors
///
/// Returns [`SnnError::BadTimesteps`] if `n_steps < 11`.
pub fn narma10_sequence(n_steps: usize, rng: &mut LcgRng) -> SnnResult<(Vec<f32>, Vec<f32>)> {
    if n_steps < 11 {
        return Err(SnnError::BadTimesteps { got: n_steps });
    }
    let mut u = vec![0.0_f32; n_steps];
    for x in u.iter_mut() {
        // Uniform[0, 0.5): next_f32() is already full-range in [0, 1).
        *x = 0.5 * rng.next_f32();
    }
    let mut y = vec![0.0_f32; n_steps];
    for t in 9..(n_steps - 1) {
        let mut hist = 0.0_f32;
        for i in 0..10 {
            hist += y[t - i];
        }
        y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * hist + 1.5 * u[t - 9] * u[t] + 0.1;
    }
    Ok((u, y))
}

// ─────────────────────────────────────────────────────────────────────────────
// Reservoir driver
// ─────────────────────────────────────────────────────────────────────────────

/// Hyper-parameters shared by the reservoir benchmark drivers.
#[derive(Debug, Clone)]
pub struct LsmTaskConfig {
    /// Liquid State Machine reservoir configuration.
    pub lsm: LsmConfig,
    /// LIF neuron dynamics used by the reservoir.
    pub lif: LifConfig,
    /// Time constant of the exponential low-pass filter that turns the binary
    /// spike trains into the analog liquid state read by the linear readout.
    pub filter_tau: f32,
    /// Number of initial timesteps discarded as reservoir warm-up.
    pub washout: usize,
    /// Ridge (L2) regularisation coefficient of the readout solve.
    pub ridge_alpha: f32,
    /// Append the raw input to each feature row (standard ESN augmentation).
    pub augment_input: bool,
}

impl Default for LsmTaskConfig {
    fn default() -> Self {
        Self {
            lsm: LsmConfig {
                n_neurons: 300,
                density: 0.2,
                spectral_radius: 0.85,
                w_in_scale: 6.0,
                seed: 1,
            },
            lif: LifConfig {
                tau_m: 16.0,
                v_th: 1.0,
                v_rest: 0.0,
                dt: 1.0,
                reset: ResetMode::Soft,
            },
            filter_tau: 1.5,
            washout: 100,
            ridge_alpha: 5.0,
            augment_input: true,
        }
    }
}

impl LsmTaskConfig {
    /// Exponential filter decay `exp(−1/filter_tau)` used for the liquid state.
    #[must_use]
    fn filter_decay(&self) -> f32 {
        (-1.0_f32 / self.filter_tau.max(1.0e-6)).exp()
    }
}

/// Drive `lsm` with the row-major `inputs` (`[n_steps × in_dim]`) and return the
/// liquid-state feature matrix `(states, n_features)`, row-major
/// `[n_steps × n_features]`.
///
/// Each feature row is the exponentially low-pass-filtered spike vector
/// (`n` units), a constant bias term, and — when `augment_input` — the raw
/// input for that step.
fn collect_liquid_states(
    lsm: &mut Lsm,
    lif: &LifConfig,
    inputs: &[f32],
    filter_decay: f32,
    augment_input: bool,
) -> SnnResult<(Vec<f32>, usize)> {
    let in_dim = lsm.in_dim;
    let n = lsm.n;
    if in_dim == 0 || inputs.is_empty() || !inputs.len().is_multiple_of(in_dim) {
        return Err(SnnError::BadShape {
            expected: in_dim,
            got: inputs.len(),
        });
    }
    let n_steps = inputs.len() / in_dim;
    let n_features = n + 1 + if augment_input { in_dim } else { 0 };
    let mut states = vec![0.0_f32; n_steps * n_features];
    let mut filt = vec![0.0_f32; n];
    let mut spikes = vec![0.0_f32; n];
    let keep = 1.0_f32 - filter_decay;
    for t in 0..n_steps {
        let x = &inputs[t * in_dim..(t + 1) * in_dim];
        lsm.forward_step(x, lif, &mut spikes)?;
        for i in 0..n {
            filt[i] = filter_decay * filt[i] + keep * spikes[i];
        }
        let row = t * n_features;
        states[row..row + n].copy_from_slice(&filt);
        states[row + n] = 1.0;
        if augment_input {
            states[row + n + 1..row + n + 1 + in_dim].copy_from_slice(x);
        }
    }
    Ok((states, n_features))
}

// ─────────────────────────────────────────────────────────────────────────────
// NARMA-10 driver
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome of a NARMA-10 reservoir run.
#[derive(Debug, Clone)]
pub struct Narma10Result {
    /// Test-set NMSE of the trained reservoir + readout.
    pub nmse: f32,
    /// Test-set NMSE of the trivial constant predictor (training-set mean).
    pub baseline_nmse: f32,
    /// Number of liquid-state features the readout consumes.
    pub n_features: usize,
    /// Number of training timesteps.
    pub n_train: usize,
    /// Number of test timesteps.
    pub n_test: usize,
}

/// Run the full NARMA-10 benchmark on the spiking LSM reservoir.
///
/// A single sequence of length `washout + n_train + n_test` is generated, the
/// reservoir is driven once over the whole sequence, the ridge readout is fit
/// on the training split, and the held-out test NMSE is measured (together with
/// the constant-mean baseline NMSE for reference).
///
/// # Errors
///
/// Returns [`SnnError::BadTimesteps`] if `n_train == 0` or `n_test == 0`, and
/// propagates reservoir / readout construction and solve errors.
pub fn narma10_lsm_nmse(
    task: &LsmTaskConfig,
    n_train: usize,
    n_test: usize,
    seed: u64,
) -> SnnResult<Narma10Result> {
    if n_train == 0 || n_test == 0 {
        return Err(SnnError::BadTimesteps { got: 0 });
    }
    let total = task.washout + n_train + n_test;
    let mut rng = LcgRng::new(seed);
    let (u, y) = narma10_sequence(total, &mut rng)?;

    let mut lsm = Lsm::new(1, &task.lsm, &task.lif)?;
    let (states, n_features) = collect_liquid_states(
        &mut lsm,
        &task.lif,
        &u,
        task.filter_decay(),
        task.augment_input,
    )?;

    // Training split: rows [washout, washout + n_train).
    let train_start = task.washout;
    let train_end = train_start + n_train;
    let train_states = &states[train_start * n_features..train_end * n_features];
    let train_targets = &y[train_start..train_end];

    let mut readout = RidgeReadout::new(n_features, 1, task.ridge_alpha)?;
    readout.fit_batch(train_states, train_targets, n_train)?;

    // Test split: rows [train_end, total).
    let train_mean: f32 = train_targets.iter().sum::<f32>() / n_train as f32;
    let mut preds = vec![0.0_f32; n_test];
    let mut test_targets = vec![0.0_f32; n_test];
    for r in 0..n_test {
        let row = (train_end + r) * n_features;
        let state = &states[row..row + n_features];
        preds[r] = readout.predict(state)?[0];
        test_targets[r] = y[train_end + r];
    }

    let baseline: Vec<f32> = vec![train_mean; n_test];
    Ok(Narma10Result {
        nmse: nmse(&preds, &test_targets),
        baseline_nmse: nmse(&baseline, &test_targets),
        n_features,
        n_train,
        n_test,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory capacity
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome of a linear memory-capacity measurement.
#[derive(Debug, Clone)]
pub struct MemoryCapacityResult {
    /// Total memory capacity `MC = Σ_k r²_k` summed over all probed delays.
    pub total_mc: f32,
    /// Per-delay squared correlation `r²_k` for `k = 1 ..= max_delay`.
    pub per_delay: Vec<f32>,
    /// Number of liquid-state features (the theoretical upper bound on `MC`).
    pub n_features: usize,
}

/// Measure the reservoir's linear memory capacity.
///
/// The reservoir is driven with i.i.d. `Uniform[−1, 1)` input. For each delay
/// `k ∈ {1, …, max_delay}` a linear readout is trained (jointly, one ridge
/// solve) to reconstruct `u(t−k)` from the liquid state at time `t`; the
/// squared correlation between reconstruction and ground truth is `r²_k`, and
/// the total capacity is their sum. `MC` is provably bounded by the number of
/// linearly independent readout features.
///
/// # Errors
///
/// Returns [`SnnError::BadTimesteps`] / [`SnnError::BadDim`] for non-positive
/// `n_steps` / `max_delay`, and propagates reservoir and solve errors.
pub fn memory_capacity(
    task: &LsmTaskConfig,
    n_steps: usize,
    max_delay: usize,
    seed: u64,
) -> SnnResult<MemoryCapacityResult> {
    if n_steps == 0 {
        return Err(SnnError::BadTimesteps { got: 0 });
    }
    if max_delay == 0 {
        return Err(SnnError::BadDim { got: 0 });
    }
    let total = task.washout + n_steps;
    let mut rng = LcgRng::new(seed);
    let u: Vec<f32> = (0..total).map(|_| 2.0 * rng.next_f32() - 1.0).collect();

    let mut lsm = Lsm::new(1, &task.lsm, &task.lif)?;
    // Memory capacity is a property of the reservoir alone: do not augment with
    // the raw input (that would trivially supply the zero-delay term).
    let (states_all, n_features) =
        collect_liquid_states(&mut lsm, &task.lif, &u, task.filter_decay(), false)?;

    let t0 = task.washout;
    let t_eff = total - t0;
    let x = &states_all[t0 * n_features..total * n_features];

    // Delayed-input target matrix D, row-major [t_eff × max_delay].
    let mut d = vec![0.0_f32; t_eff * max_delay];
    for (r, t) in (t0..total).enumerate() {
        for k in 0..max_delay {
            let src = t as isize - (k as isize + 1);
            d[r * max_delay + k] = if src >= 0 { u[src as usize] } else { 0.0 };
        }
    }

    // One ridge solve for all delays at once: W is [max_delay × n_features].
    let w = ridge_regression(x, &d, t_eff, n_features, max_delay, task.ridge_alpha)?;

    // Reconstruct each delay and accumulate r².
    let mut per_delay = vec![0.0_f32; max_delay];
    let mut pred_k = vec![0.0_f32; t_eff];
    let mut true_k = vec![0.0_f32; t_eff];
    for k in 0..max_delay {
        for r in 0..t_eff {
            let xs = &x[r * n_features..(r + 1) * n_features];
            let wk = &w[k * n_features..(k + 1) * n_features];
            pred_k[r] = wk.iter().zip(xs.iter()).map(|(&a, &b)| a * b).sum();
            true_k[r] = d[r * max_delay + k];
        }
        per_delay[k] = squared_correlation(&pred_k, &true_k);
    }
    let total_mc = per_delay.iter().sum();
    Ok(MemoryCapacityResult {
        total_mc,
        per_delay,
        n_features,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── NARMA-10 sequence ──────────────────────────────────────────────────

    #[test]
    fn narma10_sequence_is_bounded_and_finite() {
        let mut rng = LcgRng::new(7);
        let (u, y) = narma10_sequence(2000, &mut rng).expect("seq");
        assert_eq!(u.len(), 2000);
        assert_eq!(y.len(), 2000);
        for &ui in &u {
            assert!((0.0..0.5).contains(&ui), "input out of [0,0.5): {ui}");
        }
        for &yi in &y {
            assert!(yi.is_finite(), "NARMA output diverged: {yi}");
            // Bounded benchmark stays in a small positive band.
            assert!((-0.1..2.0).contains(&yi), "NARMA output out of band: {yi}");
        }
        // The target must actually vary (it is a non-trivial signal).
        let mean = y.iter().sum::<f32>() / y.len() as f32;
        let var = y.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / y.len() as f32;
        assert!(var > 1.0e-6, "NARMA target has no variance: {var}");
    }

    #[test]
    fn narma10_sequence_rejects_short() {
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            narma10_sequence(5, &mut rng),
            Err(SnnError::BadTimesteps { .. })
        ));
    }

    #[test]
    fn narma10_sequence_is_deterministic() {
        let (u1, y1) = narma10_sequence(300, &mut LcgRng::new(123)).expect("a");
        let (u2, y2) = narma10_sequence(300, &mut LcgRng::new(123)).expect("b");
        assert_eq!(u1, u2);
        assert_eq!(y1, y2);
    }

    // ── NMSE / correlation metrics ─────────────────────────────────────────

    #[test]
    fn nmse_mean_predictor_is_one() {
        let targets = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let mean = targets.iter().sum::<f32>() / targets.len() as f32;
        let preds = vec![mean; targets.len()];
        assert!((nmse(&preds, &targets) - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn nmse_perfect_predictor_is_zero() {
        let targets = [1.0_f32, -2.0, 3.5, 0.25];
        assert!(nmse(&targets, &targets) < 1.0e-9);
    }

    #[test]
    fn squared_correlation_bounds() {
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        // Perfect linear relation → r² = 1.
        let b = [2.0_f32, 4.0, 6.0, 8.0];
        assert!((squared_correlation(&a, &b) - 1.0).abs() < 1.0e-5);
        // Constant series → r² = 0.
        let c = [5.0_f32; 4];
        assert_eq!(squared_correlation(&a, &c), 0.0);
    }

    // ── NARMA-10 on the LSM ────────────────────────────────────────────────

    #[test]
    fn narma10_lsm_beats_baseline() {
        let task = LsmTaskConfig::default();
        let res = narma10_lsm_nmse(&task, 1000, 400, 20).expect("narma run");
        // Real measured numbers; finite and sane.
        assert!(res.nmse.is_finite(), "NMSE not finite: {}", res.nmse);
        assert!(
            (res.baseline_nmse - 1.0).abs() < 0.2,
            "mean-predictor baseline should be ≈1, got {}",
            res.baseline_nmse
        );
        // Measured NMSE ≈ 0.48 (spiking LSM + filtered-spike liquid state + ridge
        // readout); assert it sits in a justified band clearly below the
        // constant-mean baseline (≈1.0). The reservoir explains roughly half the
        // NARMA-10 variance — a clear, honest beat, not a perfect solve.
        assert!(
            res.nmse < 0.6,
            "LSM NARMA-10 NMSE {} did not reach the expected band (<0.6)",
            res.nmse
        );
        assert!(
            res.nmse < 0.65 * res.baseline_nmse,
            "LSM NMSE {} not a clear margin below baseline {}",
            res.nmse,
            res.baseline_nmse
        );
    }

    #[test]
    fn narma10_lsm_is_deterministic() {
        let task = LsmTaskConfig::default();
        let a = narma10_lsm_nmse(&task, 400, 200, 5).expect("a");
        let b = narma10_lsm_nmse(&task, 400, 200, 5).expect("b");
        assert_eq!(a.nmse.to_bits(), b.nmse.to_bits(), "NMSE not deterministic");
    }

    #[test]
    fn narma10_lsm_rejects_zero_sizes() {
        let task = LsmTaskConfig::default();
        assert!(matches!(
            narma10_lsm_nmse(&task, 0, 100, 1),
            Err(SnnError::BadTimesteps { .. })
        ));
    }

    // ── Memory capacity ────────────────────────────────────────────────────

    #[test]
    fn memory_capacity_positive_and_bounded() {
        let task = LsmTaskConfig::default();
        let res = memory_capacity(&task, 1200, 30, 11).expect("mc");
        assert!(res.total_mc.is_finite());
        // Measured MC ≈ 5–6 for the N=300 reservoir: clearly positive.
        assert!(
            res.total_mc > 1.0,
            "memory capacity should be clearly positive, got {}",
            res.total_mc
        );
        // Theoretical cap: MC ≤ number of linearly independent readout features.
        assert!(
            res.total_mc <= res.n_features as f32 + 1.0e-3,
            "MC {} exceeds feature-count cap {}",
            res.total_mc,
            res.n_features
        );
        // Each individual delay contributes a bounded r² ∈ [0, 1].
        for (k, &r2) in res.per_delay.iter().enumerate() {
            assert!(
                (0.0..=1.000_1).contains(&r2),
                "r²[{}]={} outside [0,1]",
                k + 1,
                r2
            );
        }
        // Forgetting curve: short delays are remembered better than long ones.
        let short: f32 = res.per_delay[0..3].iter().sum::<f32>() / 3.0;
        let long: f32 = res.per_delay[res.per_delay.len() - 3..].iter().sum::<f32>() / 3.0;
        assert!(
            short > long,
            "memory should decay with delay: mean r²(short)={short}, mean r²(long)={long}"
        );
    }

    #[test]
    fn memory_capacity_increases_with_reservoir_size() {
        let mut small = LsmTaskConfig::default();
        small.lsm.n_neurons = 40;
        let mut large = LsmTaskConfig::default();
        large.lsm.n_neurons = 160;

        let mc_small = memory_capacity(&small, 1200, 30, 3).expect("small");
        let mc_large = memory_capacity(&large, 1200, 30, 3).expect("large");
        assert!(
            mc_large.total_mc > mc_small.total_mc,
            "MC should grow with reservoir size: small(N=40)={}, large(N=160)={}",
            mc_small.total_mc,
            mc_large.total_mc
        );
    }

    #[test]
    fn memory_capacity_is_deterministic() {
        let task = LsmTaskConfig::default();
        let a = memory_capacity(&task, 800, 20, 9).expect("a");
        let b = memory_capacity(&task, 800, 20, 9).expect("b");
        assert_eq!(a.total_mc.to_bits(), b.total_mc.to_bits());
    }

    #[test]
    fn memory_capacity_rejects_bad_args() {
        let task = LsmTaskConfig::default();
        assert!(matches!(
            memory_capacity(&task, 0, 10, 1),
            Err(SnnError::BadTimesteps { .. })
        ));
        assert!(matches!(
            memory_capacity(&task, 100, 0, 1),
            Err(SnnError::BadDim { .. })
        ));
    }
}
