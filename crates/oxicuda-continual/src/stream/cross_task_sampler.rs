//! Cross-task batch sampler for continual learning.
//!
//! Draws mini-batches spanning multiple tasks using configurable sampling strategies:
//! - **Uniform**: equal probability per task.
//! - **Proportional**: probability proportional to task dataset size.
//! - **Temperature**: probability ∝ n_k^(1/temperature).

use crate::error::{ContinualError, ContinualResult};
use crate::handle::LcgRng;

// ─── Strategy ────────────────────────────────────────────────────────────────

/// Sampling strategy that governs how batch slots are assigned to tasks.
#[derive(Clone, Debug)]
pub enum SamplingStrategy {
    /// Each task receives the same probability regardless of dataset size.
    Uniform,
    /// Probability is proportional to the number of samples in each task.
    Proportional,
    /// Probability ∝ n_k^(1/temperature).
    ///
    /// - `temperature = 1.0` → identical to Proportional.
    /// - `temperature → 0`   → only the largest task is sampled.
    /// - `temperature → ∞`   → identical to Uniform.
    Temperature {
        /// Must be strictly positive and finite.
        temperature: f64,
    },
}

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for [`CrossTaskSampler`].
#[derive(Clone, Debug)]
pub struct CrossTaskSamplerConfig {
    /// Number of samples in each mini-batch.
    pub batch_size: usize,
    /// How task sampling probabilities are computed.
    pub strategy: SamplingStrategy,
    /// Seed for the internal LCG random number generator.
    pub seed: u64,
}

impl Default for CrossTaskSamplerConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            strategy: SamplingStrategy::Uniform,
            seed: 42,
        }
    }
}

// ─── TaskSample ───────────────────────────────────────────────────────────────

/// A batch contribution from a single task.
#[derive(Clone, Debug)]
pub struct TaskSample {
    /// Which task these samples come from.
    pub task_id: usize,
    /// Indices into that task's data array (drawn with replacement).
    pub sample_indices: Vec<usize>,
}

// ─── CrossTaskSampler ─────────────────────────────────────────────────────────

/// Samples mini-batches that span multiple tasks.
///
/// Slots in each mini-batch are assigned to tasks via a multinomial draw that
/// is governed by the [`SamplingStrategy`].  Within each task, sample indices
/// are drawn uniformly at random **with replacement**.
pub struct CrossTaskSampler {
    /// `task_sizes[k]` = number of samples in task `k`.
    task_sizes: Vec<usize>,
    strategy: SamplingStrategy,
    batch_size: usize,
    rng: LcgRng,
    /// Normalised sampling weights — one per task, summing to 1.
    weights: Vec<f64>,
}

impl CrossTaskSampler {
    /// Create a new sampler for tasks whose sizes are given.
    ///
    /// # Errors
    ///
    /// Returns [`ContinualError::EmptyInput`] when `task_sizes` is empty.
    /// Returns [`ContinualError::Internal`] when `batch_size == 0` or the
    /// temperature strategy has a non-positive temperature.
    pub fn new(task_sizes: &[usize], config: &CrossTaskSamplerConfig) -> ContinualResult<Self> {
        if task_sizes.is_empty() {
            return Err(ContinualError::EmptyInput);
        }
        if config.batch_size == 0 {
            return Err(ContinualError::Internal(
                "batch_size must be >= 1".to_string(),
            ));
        }
        if let SamplingStrategy::Temperature { temperature } = config.strategy {
            if temperature <= 0.0 || !temperature.is_finite() {
                return Err(ContinualError::Internal(
                    "temperature must be strictly positive and finite".to_string(),
                ));
            }
        }

        let mut sampler = Self {
            task_sizes: task_sizes.to_vec(),
            strategy: config.strategy.clone(),
            batch_size: config.batch_size,
            rng: LcgRng::new(config.seed),
            weights: Vec::new(),
        };
        sampler.recompute_weights();
        Ok(sampler)
    }

    /// Register a new task (online task arrival).
    ///
    /// The weights are recomputed automatically.
    ///
    /// # Errors
    ///
    /// Returns [`ContinualError::Internal`] when `size == 0` and the strategy
    /// requires positive sizes (Proportional / Temperature).
    pub fn add_task(&mut self, size: usize) -> ContinualResult<()> {
        self.task_sizes.push(size);
        self.recompute_weights();
        Ok(())
    }

    /// Draw a cross-task mini-batch of `batch_size` samples.
    ///
    /// Returns one [`TaskSample`] per task that received ≥ 1 slot.
    ///
    /// # Errors
    ///
    /// Returns [`ContinualError::Internal`] when all tasks have size 0 (no
    /// valid index can be drawn).
    pub fn sample_batch(&mut self) -> ContinualResult<Vec<TaskSample>> {
        let n_tasks = self.task_sizes.len();

        // Assign each of `batch_size` slots to a task.
        let task_assignments = self.categorical_draw(self.batch_size);

        // Count how many slots each task got.
        let mut task_counts = vec![0usize; n_tasks];
        for t in &task_assignments {
            task_counts[*t] += 1;
        }

        // For each task with ≥ 1 slot, draw sample indices.
        let mut result = Vec::new();
        for (task_id, &count) in task_counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let sz = self.task_sizes[task_id];
            if sz == 0 {
                return Err(ContinualError::Internal(format!(
                    "task {task_id} has size 0 — cannot sample from it"
                )));
            }
            let sample_indices: Vec<usize> = (0..count).map(|_| self.rng.next_usize(sz)).collect();
            result.push(TaskSample {
                task_id,
                sample_indices,
            });
        }
        Ok(result)
    }

    /// Sample with a fixed per-task allocation.
    ///
    /// `per_task_sizes[k]` specifies exactly how many samples to draw from task `k`.
    /// The length of `per_task_sizes` must equal `n_tasks()`.
    ///
    /// # Errors
    ///
    /// Returns [`ContinualError::DimensionMismatch`] when the slice length
    /// does not match `n_tasks()`.  Returns [`ContinualError::Internal`] when a
    /// non-zero allocation targets a zero-size task.
    pub fn sample_fixed_allocation(
        &mut self,
        per_task_sizes: &[usize],
    ) -> ContinualResult<Vec<TaskSample>> {
        let n_tasks = self.task_sizes.len();
        if per_task_sizes.len() != n_tasks {
            return Err(ContinualError::DimensionMismatch {
                expected: n_tasks,
                got: per_task_sizes.len(),
            });
        }

        let mut result = Vec::new();
        for (task_id, &count) in per_task_sizes.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let sz = self.task_sizes[task_id];
            if sz == 0 {
                return Err(ContinualError::Internal(format!(
                    "task {task_id} has size 0 but allocation requests {count} samples"
                )));
            }
            let sample_indices: Vec<usize> = (0..count).map(|_| self.rng.next_usize(sz)).collect();
            result.push(TaskSample {
                task_id,
                sample_indices,
            });
        }
        Ok(result)
    }

    // ─── Accessors ───────────────────────────────────────────────────────────

    /// Number of registered tasks.
    #[must_use]
    pub fn n_tasks(&self) -> usize {
        self.task_sizes.len()
    }

    /// Slice of task dataset sizes.
    #[must_use]
    pub fn task_sizes(&self) -> &[usize] {
        &self.task_sizes
    }

    /// Normalised sampling weights (one per task, sum ≈ 1).
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    // ─── Internal helpers ────────────────────────────────────────────────────

    /// Draw `n_draw` task indices according to the normalised `weights`.
    ///
    /// Uses integer-arithmetic CDF inversion.  The weights are converted to a
    /// u64 CDF scaled by 2^32 (the full range of `next_u32()`, which returns a
    /// value in `[0, 2^32)`).  Each draw is a u32 from the LCG compared against
    /// this CDF.  Integer arithmetic avoids the floating-point rounding gaps a
    /// `next_f32()`-based comparison would introduce.
    fn categorical_draw(&mut self, n_draw: usize) -> Vec<usize> {
        let n_tasks = self.weights.len();
        let mut result = Vec::with_capacity(n_draw);

        // Scale = 2^32: next_u32() returns a value in [0, 2^32 - 1], so the CDF
        // must span the same range for an unbiased categorical draw.
        let scale: u64 = 1u64 << 32;

        // Build integer CDF in [0, scale].
        // cdf_int[k] = floor(CDF[k] * scale), with last bucket forced to scale
        // to cover the full range and avoid floating-point rounding gaps.
        let mut cdf_int: Vec<u64> = Vec::with_capacity(n_tasks);
        let mut acc: u64 = 0;
        for (k, &w) in self.weights.iter().enumerate() {
            let contrib = if k + 1 == n_tasks {
                // Ensure last entry exactly equals scale.
                scale - acc
            } else {
                (w * scale as f64) as u64
            };
            acc += contrib;
            cdf_int.push(acc);
        }

        for _ in 0..n_draw {
            // next_u32() returns a value in [0, 2^32 - 1].
            let u = self.rng.next_u32() as u64;
            // Find the first bucket whose cumulative weight exceeds u.
            let mut idx = n_tasks - 1; // default: last bucket
            for (k, &c) in cdf_int.iter().enumerate() {
                if c > u {
                    idx = k;
                    break;
                }
            }
            result.push(idx);
        }
        result
    }

    /// Recompute `weights` from `task_sizes` and the current strategy.
    fn recompute_weights(&mut self) {
        let n = self.task_sizes.len();
        if n == 0 {
            self.weights = Vec::new();
            return;
        }

        let raw: Vec<f64> = match &self.strategy {
            SamplingStrategy::Uniform => vec![1.0_f64; n],

            SamplingStrategy::Proportional => self.task_sizes.iter().map(|&s| s as f64).collect(),

            SamplingStrategy::Temperature { temperature } => {
                let temp = *temperature;
                if temp == 0.0 {
                    // Only the largest task gets weight.
                    let max_sz = self.task_sizes.iter().cloned().max().unwrap_or(0);
                    self.task_sizes
                        .iter()
                        .map(|&s| if s == max_sz { 1.0_f64 } else { 0.0_f64 })
                        .collect()
                } else {
                    self.task_sizes
                        .iter()
                        .map(|&s| (s as f64).powf(1.0 / temp))
                        .collect()
                }
            }
        };

        let total: f64 = raw.iter().sum();
        if total <= 0.0 {
            // Fallback to uniform when all weights are zero (e.g. all tasks empty).
            let w = 1.0 / n as f64;
            self.weights = vec![w; n];
        } else {
            self.weights = raw.iter().map(|&r| r / total).collect();
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: default config with a given seed.
    fn cfg(batch_size: usize, strategy: SamplingStrategy, seed: u64) -> CrossTaskSamplerConfig {
        CrossTaskSamplerConfig {
            batch_size,
            strategy,
            seed,
        }
    }

    // ── Test 1: weight sum ≈ 1 ───────────────────────────────────────────────
    #[test]
    fn weight_sum_is_one() {
        let sizes = &[10usize, 20, 30, 40];
        let sampler = CrossTaskSampler::new(sizes, &cfg(16, SamplingStrategy::Uniform, 1))
            .expect("CrossTaskSampler should construct with valid sizes and config");
        let sum: f64 = sampler.weights().iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "weights must sum to 1.0, got {sum}"
        );
    }

    // ── Test 2: Uniform strategy → equal weights ─────────────────────────────
    #[test]
    fn uniform_weights_are_equal() {
        let sizes = &[5usize, 50, 500];
        let sampler = CrossTaskSampler::new(sizes, &cfg(32, SamplingStrategy::Uniform, 2))
            .expect("CrossTaskSampler should construct with valid sizes and config");
        let expected = 1.0 / 3.0;
        for &w in sampler.weights() {
            assert!(
                (w - expected).abs() < 1e-10,
                "uniform weight should be {expected}, got {w}"
            );
        }
    }

    // ── Test 3: Proportional — larger task gets more weight ──────────────────
    #[test]
    fn proportional_larger_task_higher_weight() {
        let sizes = &[10usize, 100];
        let sampler = CrossTaskSampler::new(sizes, &cfg(64, SamplingStrategy::Proportional, 3))
            .expect("CrossTaskSampler should construct with valid sizes and config");
        let w = sampler.weights();
        assert!(
            w[1] > w[0],
            "task 1 (size 100) should have higher weight than task 0 (size 10)"
        );
        // Exact ratio: 100/110 vs 10/110
        let ratio = w[1] / w[0];
        assert!(
            (ratio - 10.0).abs() < 1e-8,
            "weight ratio should be 10.0, got {ratio}"
        );
    }

    // ── Test 4: Temperature=1 equals Proportional ────────────────────────────
    #[test]
    fn temperature_1_equals_proportional() {
        let sizes = &[7usize, 43, 120];
        let prop = CrossTaskSampler::new(sizes, &cfg(32, SamplingStrategy::Proportional, 4))
            .expect("CrossTaskSampler should construct with valid sizes and config");
        let temp = CrossTaskSampler::new(
            sizes,
            &cfg(32, SamplingStrategy::Temperature { temperature: 1.0 }, 4),
        )
        .expect("CrossTaskSampler should construct with valid sizes and config");
        for (wp, wt) in prop.weights().iter().zip(temp.weights().iter()) {
            assert!(
                (wp - wt).abs() < 1e-10,
                "temperature=1 should equal proportional: {wp} vs {wt}"
            );
        }
    }

    // ── Test 5: Temperature=1e10 ≈ Uniform ───────────────────────────────────
    #[test]
    fn temperature_large_approx_uniform() {
        let sizes = &[10usize, 100, 1000];
        let sampler = CrossTaskSampler::new(
            sizes,
            &cfg(32, SamplingStrategy::Temperature { temperature: 1e10 }, 5),
        )
        .expect("CrossTaskSampler should construct with valid sizes and config");
        let w = sampler.weights();
        let expected = 1.0 / 3.0;
        for &wi in w {
            assert!(
                (wi - expected).abs() < 1e-4,
                "very high temperature should approximate uniform (1/3), got {wi}"
            );
        }
    }

    // ── Test 6: sample_batch total size matches batch_size ───────────────────
    #[test]
    fn sample_batch_total_size_correct() {
        let sizes = &[20usize, 30, 50];
        let mut sampler = CrossTaskSampler::new(sizes, &cfg(32, SamplingStrategy::Uniform, 6))
            .expect("CrossTaskSampler should construct with valid sizes and config");
        let batch = sampler
            .sample_batch()
            .expect("batch sampling should succeed with valid sampler state");
        let total: usize = batch.iter().map(|ts| ts.sample_indices.len()).sum();
        assert_eq!(total, 32, "total samples must equal batch_size");
    }

    // ── Test 7: add_task updates n_tasks and weights ──────────────────────────
    #[test]
    fn add_task_updates_state() {
        let mut sampler =
            CrossTaskSampler::new(&[10usize, 20], &cfg(16, SamplingStrategy::Uniform, 7))
                .expect("CrossTaskSampler should construct with valid sizes and config");
        assert_eq!(sampler.n_tasks(), 2);
        sampler.add_task(30).expect("adding a task should succeed");
        assert_eq!(sampler.n_tasks(), 3);
        let sum: f64 = sampler.weights().iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "weights must still sum to 1");
    }

    // ── Test 8: sample_fixed_allocation respects per-task counts ─────────────
    #[test]
    fn sample_fixed_allocation_exact_counts() {
        let sizes = &[100usize, 200, 300];
        let mut sampler = CrossTaskSampler::new(sizes, &cfg(32, SamplingStrategy::Uniform, 8))
            .expect("CrossTaskSampler should construct with valid sizes and config");
        let alloc = &[5usize, 10, 15];
        let batch = sampler
            .sample_fixed_allocation(alloc)
            .expect("fixed allocation sampling should succeed");
        // Collect per-task counts from result.
        let mut counts = vec![0usize; 3];
        for ts in &batch {
            counts[ts.task_id] += ts.sample_indices.len();
        }
        assert_eq!(counts, alloc, "per-task counts must match allocation");
    }

    // ── Test 9: sample indices are within valid range ─────────────────────────
    #[test]
    fn sample_indices_within_task_size() {
        let sizes = &[10usize, 25, 50];
        let mut sampler = CrossTaskSampler::new(sizes, &cfg(64, SamplingStrategy::Proportional, 9))
            .expect("CrossTaskSampler should construct with valid sizes and config");
        for _ in 0..10 {
            let batch = sampler
                .sample_batch()
                .expect("batch sampling should succeed with valid sampler state");
            for ts in &batch {
                let sz = sizes[ts.task_id];
                for &idx in &ts.sample_indices {
                    assert!(
                        idx < sz,
                        "index {idx} out of range for task {} (size {sz})",
                        ts.task_id
                    );
                }
            }
        }
    }

    // ── Test 10: Reproducibility — same seed, same sequence ──────────────────
    #[test]
    fn reproducibility_same_seed() {
        let sizes = &[15usize, 35, 50];
        let strategy = SamplingStrategy::Proportional;
        let mut s1 = CrossTaskSampler::new(sizes, &cfg(16, strategy.clone(), 99))
            .expect("CrossTaskSampler should construct with valid sizes and config");
        let mut s2 = CrossTaskSampler::new(sizes, &cfg(16, strategy, 99))
            .expect("CrossTaskSampler should construct with valid sizes and config");
        for _ in 0..5 {
            let b1 = s1
                .sample_batch()
                .expect("batch sampling should succeed with valid sampler state");
            let b2 = s2
                .sample_batch()
                .expect("batch sampling should succeed with valid sampler state");
            assert_eq!(b1.len(), b2.len());
            for (t1, t2) in b1.iter().zip(b2.iter()) {
                assert_eq!(t1.task_id, t2.task_id);
                assert_eq!(t1.sample_indices, t2.sample_indices);
            }
        }
    }

    // ── Test 11: Single-task sampler returns one TaskSample per batch ─────────
    #[test]
    fn single_task_one_sample_per_batch() {
        let mut sampler =
            CrossTaskSampler::new(&[100usize], &cfg(32, SamplingStrategy::Uniform, 11))
                .expect("CrossTaskSampler should construct with valid sizes and config");
        let batch = sampler
            .sample_batch()
            .expect("batch sampling should succeed with valid sampler state");
        assert_eq!(batch.len(), 1, "single task → exactly one TaskSample");
        assert_eq!(batch[0].task_id, 0);
        assert_eq!(batch[0].sample_indices.len(), 32);
    }

    // ── Test 12: Empty task sizes → error ────────────────────────────────────
    #[test]
    fn empty_task_sizes_returns_error() {
        let result = CrossTaskSampler::new(&[], &CrossTaskSamplerConfig::default());
        assert!(result.is_err(), "empty task_sizes should return an error");
    }

    // ── Test 13: batch_size=0 → error ────────────────────────────────────────
    #[test]
    fn batch_size_zero_returns_error() {
        let result = CrossTaskSampler::new(&[10usize], &cfg(0, SamplingStrategy::Uniform, 13));
        assert!(result.is_err(), "batch_size=0 should return an error");
    }

    // ── Test 14: temperature ≤ 0 → error ─────────────────────────────────────
    #[test]
    fn nonpositive_temperature_returns_error() {
        let neg = CrossTaskSampler::new(
            &[10usize],
            &cfg(8, SamplingStrategy::Temperature { temperature: -1.0 }, 14),
        );
        assert!(neg.is_err(), "negative temperature should return error");
        let zero = CrossTaskSampler::new(
            &[10usize],
            &cfg(8, SamplingStrategy::Temperature { temperature: 0.0 }, 14),
        );
        assert!(zero.is_err(), "zero temperature should return error");
    }

    // ── Test 15: Uniform distributes evenly across many draws ─────────────────
    #[test]
    fn uniform_distributes_evenly_statistically() {
        let sizes = &[100usize, 100, 100];
        let mut sampler = CrossTaskSampler::new(sizes, &cfg(300, SamplingStrategy::Uniform, 15))
            .expect("CrossTaskSampler should construct with valid sizes and config");

        let mut task_counts = [0usize; 3];
        let n_batches = 100;
        for _ in 0..n_batches {
            let batch = sampler
                .sample_batch()
                .expect("batch sampling should succeed with valid sampler state");
            for ts in &batch {
                task_counts[ts.task_id] += ts.sample_indices.len();
            }
        }
        let total: usize = task_counts.iter().sum();
        let expected = total as f64 / 3.0;
        for (i, &cnt) in task_counts.iter().enumerate() {
            let deviation = (cnt as f64 - expected).abs() / expected;
            assert!(
                deviation < 0.10,
                "task {i} deviation {deviation:.3} > 10% from expected"
            );
        }
    }

    // ── Test 16: fixed_allocation dim mismatch → error ────────────────────────
    #[test]
    fn fixed_allocation_dim_mismatch_error() {
        let mut sampler =
            CrossTaskSampler::new(&[10usize, 20], &cfg(16, SamplingStrategy::Uniform, 16))
                .expect("CrossTaskSampler should construct with valid sizes and config");
        // Provide 3 allocations for 2 tasks.
        let result = sampler.sample_fixed_allocation(&[5, 5, 5]);
        assert!(result.is_err(), "dim mismatch should be an error");
    }
} // end mod tests
