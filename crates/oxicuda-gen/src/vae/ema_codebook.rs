//! EMA codebook with decay warm-up and dead-code reinitialisation.
//!
//! Extends the plain EMA vector-quantiser (see [`crate::vae::quantize`] and
//! [`crate::vae::vq_vae2`]) with the two training-stability techniques that the
//! base implementations lack:
//!
//! 1. **Decay warm-up.**  Early in training the EMA statistics are dominated by
//!    their (arbitrary) initialisation.  Ramping the decay `γ` from a small
//!    value up to its target over `warmup_steps` lets the codebook adapt quickly
//!    at first, then settle into slow, stable averaging — analogous to a
//!    learning-rate warm-up.
//!
//! 2. **Dead-code reinitialisation ("random restart").**  Codebook entries that
//!    have not been selected for `dead_patience` consecutive updates are
//!    re-seeded from a random encoder output in the current batch (as in
//!    Jukebox / DALL·E).  This revives collapsed codes and improves codebook
//!    utilisation.
//!
//! The EMA update itself uses the Laplace-smoothed normalisation of
//! Razavi et al. (2019, VQ-VAE-2):
//!
//! ```text
//! N_k ← (n_k + ε) / (Σ_j n_j + Kε) · Σ_j n_j ,   e_k ← m_k / N_k
//! ```
//!
//! where `n_k` is the smoothed cluster size and `m_k` the smoothed embedding sum.

use crate::error::{GenError, GenResult};
use crate::handle::LcgRng;

// ─── RNG helper ──────────────────────────────────────────────────────────────────

/// Uniform sample in `[-scale, scale)` using the full 31-bit RNG range.
///
/// [`LcgRng::next_f32`] only spans `[0, 0.5)`; we rebuild a true `[0, 1)`
/// uniform from the integer output directly.
#[inline]
fn uniform_sym(rng: &mut LcgRng, scale: f32) -> f32 {
    let u = rng.next_u32() as f32 / 4_294_967_296.0_f32; // 2^31
    (u - 0.5) * 2.0 * scale
}

// ─── EmaCodebookConfig ───────────────────────────────────────────────────────────

/// Configuration for [`EmaCodebook`].
#[derive(Debug, Clone, Copy)]
pub struct EmaCodebookConfig {
    /// Target EMA decay `γ ∈ (0, 1)` reached after warm-up (e.g. `0.99`).
    pub base_decay: f32,
    /// Laplace smoothing constant for cluster-size normalisation.
    pub eps: f32,
    /// Number of updates over which the decay ramps from ~0 up to `base_decay`.
    pub warmup_steps: u64,
    /// A code unused for this many consecutive updates is considered dead.
    pub dead_patience: u64,
    /// Half-width of the symmetric uniform embedding initialisation.
    pub init_scale: f32,
}

impl Default for EmaCodebookConfig {
    fn default() -> Self {
        Self {
            base_decay: 0.99,
            eps: 1e-5,
            warmup_steps: 10,
            dead_patience: 4,
            init_scale: 0.1,
        }
    }
}

// ─── EmaCodebook ─────────────────────────────────────────────────────────────────

/// Vector-quantisation codebook with EMA updates, decay warm-up and dead-code
/// reinitialisation.
#[derive(Debug, Clone)]
pub struct EmaCodebook {
    /// Active embeddings, row-major `[n_codes × dim]`.
    embeddings: Vec<f32>,
    /// EMA smoothed embedding sums `m_k`, row-major `[n_codes × dim]`.
    ema_embeddings: Vec<f32>,
    /// EMA smoothed cluster sizes `n_k`, `[n_codes]`.
    ema_cluster_size: Vec<f32>,
    /// Cumulative number of assignments per code.
    usage_count: Vec<u64>,
    /// Consecutive updates since each code was last used.
    steps_since_used: Vec<u64>,
    n_codes: usize,
    dim: usize,
    base_decay: f32,
    eps: f32,
    warmup_steps: u64,
    dead_patience: u64,
    /// Number of EMA updates performed so far.
    step_count: u64,
}

impl EmaCodebook {
    /// Create a codebook with random uniform embeddings.
    ///
    /// # Errors
    /// * [`GenError::InvalidCodebookSize`] if `n_codes < 2`.
    /// * [`GenError::EmptyInput`] if `dim == 0`.
    pub fn new(
        n_codes: usize,
        dim: usize,
        config: EmaCodebookConfig,
        rng: &mut LcgRng,
    ) -> GenResult<Self> {
        if n_codes < 2 {
            return Err(GenError::InvalidCodebookSize(n_codes));
        }
        if dim == 0 {
            return Err(GenError::EmptyInput("dim must be > 0"));
        }
        let embeddings: Vec<f32> = (0..n_codes * dim)
            .map(|_| uniform_sym(rng, config.init_scale))
            .collect();
        Ok(Self::assemble(embeddings, n_codes, dim, config))
    }

    /// Create a codebook from explicit embeddings (row-major `[n_codes × dim]`).
    ///
    /// # Errors
    /// * [`GenError::InvalidCodebookSize`] if `n_codes < 2`.
    /// * [`GenError::EmptyInput`] if `dim == 0`.
    /// * [`GenError::DimensionMismatch`] if `embeddings.len() != n_codes * dim`.
    pub fn from_embeddings(
        embeddings: Vec<f32>,
        n_codes: usize,
        dim: usize,
        config: EmaCodebookConfig,
    ) -> GenResult<Self> {
        if n_codes < 2 {
            return Err(GenError::InvalidCodebookSize(n_codes));
        }
        if dim == 0 {
            return Err(GenError::EmptyInput("dim must be > 0"));
        }
        if embeddings.len() != n_codes * dim {
            return Err(GenError::DimensionMismatch {
                expected: n_codes * dim,
                got: embeddings.len(),
            });
        }
        Ok(Self::assemble(embeddings, n_codes, dim, config))
    }

    fn assemble(
        embeddings: Vec<f32>,
        n_codes: usize,
        dim: usize,
        config: EmaCodebookConfig,
    ) -> Self {
        let ema_embeddings = embeddings.clone();
        Self {
            embeddings,
            ema_embeddings,
            ema_cluster_size: vec![1.0_f32; n_codes],
            usage_count: vec![0_u64; n_codes],
            steps_since_used: vec![0_u64; n_codes],
            n_codes,
            dim,
            base_decay: config.base_decay,
            eps: config.eps,
            warmup_steps: config.warmup_steps,
            dead_patience: config.dead_patience,
            step_count: 0,
        }
    }

    /// Effective EMA decay for the *next* update, given the warm-up schedule.
    ///
    /// `γ_t = base_decay · min(1, (t + 1) / warmup_steps)`.
    #[must_use]
    pub fn current_decay(&self) -> f32 {
        if self.warmup_steps == 0 {
            return self.base_decay;
        }
        let ramp = ((self.step_count + 1) as f32 / self.warmup_steps as f32).min(1.0);
        self.base_decay * ramp
    }

    /// Quantise a batch of encoder outputs (read-only).
    ///
    /// Returns `(quantised, indices, commitment_loss)` where `commitment_loss`
    /// is the mean squared distance from each input to its nearest code.
    ///
    /// # Errors
    /// [`GenError::DimensionMismatch`] if `z.len() != batch * dim`.
    pub fn quantize(&self, z: &[f32], batch: usize) -> GenResult<(Vec<f32>, Vec<usize>, f32)> {
        if z.len() != batch * self.dim {
            return Err(GenError::DimensionMismatch {
                expected: batch * self.dim,
                got: z.len(),
            });
        }
        let mut quantised = vec![0.0_f32; batch * self.dim];
        let mut indices = vec![0_usize; batch];
        let mut total = 0.0_f32;
        for b in 0..batch {
            let zrow = &z[b * self.dim..(b + 1) * self.dim];
            let mut best = 0_usize;
            let mut best_dist = f32::INFINITY;
            for k in 0..self.n_codes {
                let e = &self.embeddings[k * self.dim..(k + 1) * self.dim];
                let dist: f32 = zrow
                    .iter()
                    .zip(e)
                    .map(|(&a, &c)| {
                        let d = a - c;
                        d * d
                    })
                    .sum();
                if dist < best_dist {
                    best_dist = dist;
                    best = k;
                }
            }
            indices[b] = best;
            let e = &self.embeddings[best * self.dim..(best + 1) * self.dim];
            quantised[b * self.dim..(b + 1) * self.dim].copy_from_slice(e);
            total += best_dist;
        }
        let commit = if batch > 0 {
            total / (batch as f32)
        } else {
            0.0
        };
        Ok((quantised, indices, commit))
    }

    /// EMA update of the codebook from `z` and its `indices`.
    ///
    /// Applies the warm-up decay, accumulates batch statistics, performs the
    /// Laplace-smoothed normalisation, and updates usage / recency counters.
    ///
    /// # Errors
    /// [`GenError::DimensionMismatch`] if `z.len() != batch * dim` or
    /// `indices.len() != batch`.
    pub fn update(&mut self, z: &[f32], indices: &[usize], batch: usize) -> GenResult<()> {
        if z.len() != batch * self.dim {
            return Err(GenError::DimensionMismatch {
                expected: batch * self.dim,
                got: z.len(),
            });
        }
        if indices.len() != batch {
            return Err(GenError::DimensionMismatch {
                expected: batch,
                got: indices.len(),
            });
        }
        let dim = self.dim;
        let gamma = self.current_decay();

        // Accumulate per-code batch statistics.
        let mut batch_count = vec![0.0_f32; self.n_codes];
        let mut batch_sum = vec![0.0_f32; self.n_codes * dim];
        for (b, &k) in indices.iter().enumerate() {
            if k < self.n_codes {
                batch_count[k] += 1.0;
                let zrow = &z[b * dim..(b + 1) * dim];
                let dst = &mut batch_sum[k * dim..(k + 1) * dim];
                for (acc, &val) in dst.iter_mut().zip(zrow) {
                    *acc += val;
                }
            }
        }

        // EMA on cluster sizes and embedding sums.
        for (k, (cluster, emb_chunk)) in self
            .ema_cluster_size
            .iter_mut()
            .zip(self.ema_embeddings.chunks_mut(dim))
            .enumerate()
        {
            *cluster = gamma * *cluster + (1.0 - gamma) * batch_count[k];
            let src = &batch_sum[k * dim..(k + 1) * dim];
            for (e, &s) in emb_chunk.iter_mut().zip(src) {
                *e = gamma * *e + (1.0 - gamma) * s;
            }
        }

        // Laplace-smoothed normalisation: e_k = m_k / N_k.
        let total: f32 = self.ema_cluster_size.iter().sum();
        let denom_base = total + (self.n_codes as f32) * self.eps;
        for (k, emb_chunk) in self.embeddings.chunks_mut(dim).enumerate() {
            let n_k = (self.ema_cluster_size[k] + self.eps) / denom_base * total;
            let denom = n_k.max(1e-8);
            let src = &self.ema_embeddings[k * dim..(k + 1) * dim];
            for (e, &s) in emb_chunk.iter_mut().zip(src) {
                *e = s / denom;
            }
        }

        // Usage / recency tracking.
        for (k, &cnt) in batch_count.iter().enumerate() {
            if cnt > 0.0 {
                self.usage_count[k] = self.usage_count[k].saturating_add(cnt as u64);
                self.steps_since_used[k] = 0;
            } else {
                self.steps_since_used[k] = self.steps_since_used[k].saturating_add(1);
            }
        }

        self.step_count += 1;
        Ok(())
    }

    /// Reinitialise dead codes from random encoder outputs in `z`.
    ///
    /// A code is *dead* if it has not been used for at least `dead_patience`
    /// updates.  Each dead code is re-seeded from a uniformly-chosen row of `z`,
    /// its EMA state reset, and its recency counter cleared.  Returns the indices
    /// of the codes that were revived.
    ///
    /// # Errors
    /// * [`GenError::EmptyInput`] if `batch == 0`.
    /// * [`GenError::DimensionMismatch`] if `z.len() != batch * dim`.
    pub fn revive_dead_codes(
        &mut self,
        z: &[f32],
        batch: usize,
        rng: &mut LcgRng,
    ) -> GenResult<Vec<usize>> {
        if batch == 0 {
            return Err(GenError::EmptyInput("batch must be > 0"));
        }
        if z.len() != batch * self.dim {
            return Err(GenError::DimensionMismatch {
                expected: batch * self.dim,
                got: z.len(),
            });
        }
        let dim = self.dim;
        let mut revived = Vec::new();
        for k in 0..self.n_codes {
            if self.steps_since_used[k] >= self.dead_patience {
                let r = rng.next_usize(batch);
                let src = &z[r * dim..(r + 1) * dim];
                self.embeddings[k * dim..(k + 1) * dim].copy_from_slice(src);
                self.ema_embeddings[k * dim..(k + 1) * dim].copy_from_slice(src);
                self.ema_cluster_size[k] = 1.0;
                self.steps_since_used[k] = 0;
                revived.push(k);
            }
        }
        Ok(revived)
    }

    /// Convenience: quantise → EMA update → revive dead codes.
    ///
    /// Returns `(indices, commitment_loss, revived_codes)`.
    ///
    /// # Errors
    /// Propagates errors from the individual steps.
    pub fn train_step(
        &mut self,
        z: &[f32],
        batch: usize,
        rng: &mut LcgRng,
    ) -> GenResult<(Vec<usize>, f32, Vec<usize>)> {
        let (_, indices, commit) = self.quantize(z, batch)?;
        self.update(z, &indices, batch)?;
        let revived = self.revive_dead_codes(z, batch, rng)?;
        Ok((indices, commit, revived))
    }

    /// Number of codebook entries.
    #[must_use]
    pub fn n_codes(&self) -> usize {
        self.n_codes
    }

    /// Embedding dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Full embedding matrix (row-major `[n_codes × dim]`).
    #[must_use]
    pub fn embeddings(&self) -> &[f32] {
        &self.embeddings
    }

    /// Embedding vector of code `k`.
    #[must_use]
    pub fn code(&self, k: usize) -> &[f32] {
        &self.embeddings[k * self.dim..(k + 1) * self.dim]
    }

    /// Cumulative usage counts per code.
    #[must_use]
    pub fn usage_count(&self) -> &[u64] {
        &self.usage_count
    }

    /// Consecutive updates since each code was last used.
    #[must_use]
    pub fn steps_since_used(&self) -> &[u64] {
        &self.steps_since_used
    }

    /// Number of EMA updates performed.
    #[must_use]
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Target EMA decay (after warm-up).
    #[must_use]
    pub fn base_decay(&self) -> f32 {
        self.base_decay
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dist(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| {
                let d = x - y;
                d * d
            })
            .sum::<f32>()
            .sqrt()
    }

    fn repeat_vec(v: &[f32], times: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(v.len() * times);
        for _ in 0..times {
            out.extend_from_slice(v);
        }
        out
    }

    #[test]
    fn ema_update_moves_code_toward_assigned_vectors() {
        let cfg = EmaCodebookConfig::default();
        let mut rng = LcgRng::new(11);
        let mut cb = EmaCodebook::new(4, 2, cfg, &mut rng).expect("new should succeed");
        let target = [3.0_f32, -2.0];
        let batch = 6;
        let z = repeat_vec(&target, batch);
        let (_, idx, _) = cb.quantize(&z, batch).expect("quantize should succeed");
        let k0 = idx[0];
        let before = dist(cb.code(k0), &target);
        cb.update(&z, &idx, batch).expect("update should succeed");
        let after = dist(cb.code(k0), &target);
        assert!(
            after < before,
            "EMA update should move code toward target: {before} → {after}"
        );
    }

    #[test]
    fn dead_code_reinitialised_used_persists() {
        let cfg = EmaCodebookConfig {
            dead_patience: 3,
            ..EmaCodebookConfig::default()
        };
        let patience = cfg.dead_patience;
        let mut rng = LcgRng::new(21);
        let mut cb = EmaCodebook::new(4, 2, cfg, &mut rng).expect("new should succeed");
        let target = [5.0_f32, 5.0];
        let batch = 4;
        let z = repeat_vec(&target, batch);

        // Drive enough updates that the unused codes exceed dead_patience.
        for _ in 0..(patience + 1) {
            let (_, idx, _) = cb.quantize(&z, batch).expect("quantize should succeed");
            cb.update(&z, &idx, batch).expect("update should succeed");
        }
        let (_, idx, _) = cb.quantize(&z, batch).expect("quantize should succeed");
        let used = idx[0];

        let before: Vec<Vec<f32>> = (0..cb.n_codes()).map(|k| cb.code(k).to_vec()).collect();
        let revived = cb
            .revive_dead_codes(&z, batch, &mut rng)
            .expect("revive_dead_codes should succeed");

        assert!(!revived.is_empty(), "some dead code must be revived");
        assert!(!revived.contains(&used), "the used code must persist");
        // Used code is byte-for-byte unchanged by revival.
        assert_eq!(cb.code(used), before[used].as_slice());
        // A revived code actually changed.
        let rk = revived[0];
        let changed = before[rk]
            .iter()
            .zip(cb.code(rk))
            .any(|(&a, &b)| (a - b).abs() > 1e-6);
        assert!(changed, "revived code {rk} should change");
    }

    #[test]
    fn usage_counts_tracked() {
        let cfg = EmaCodebookConfig::default();
        let mut rng = LcgRng::new(31);
        let mut cb = EmaCodebook::new(4, 2, cfg, &mut rng).expect("new should succeed");
        let target = [2.0_f32, 2.0];
        let batch = 5;
        let z = repeat_vec(&target, batch);
        let updates = 3_u64;
        for _ in 0..updates {
            let (_, idx, _) = cb.quantize(&z, batch).expect("quantize should succeed");
            cb.update(&z, &idx, batch).expect("update should succeed");
        }
        let (_, idx, _) = cb.quantize(&z, batch).expect("quantize should succeed");
        let used = idx[0];
        assert_eq!(cb.usage_count()[used], (batch as u64) * updates);
        let total_unused: u64 = (0..cb.n_codes())
            .filter(|&k| k != used)
            .map(|k| cb.usage_count()[k])
            .sum();
        assert_eq!(total_unused, 0, "unused codes must have zero usage");
    }

    #[test]
    fn decay_warmup_ramps_up() {
        let cfg = EmaCodebookConfig {
            warmup_steps: 8,
            base_decay: 0.9,
            ..EmaCodebookConfig::default()
        };
        let mut rng = LcgRng::new(41);
        let mut cb = EmaCodebook::new(4, 2, cfg, &mut rng).expect("new should succeed");
        let z = repeat_vec(&[1.0_f32, 0.0], 3);
        let d0 = cb.current_decay();
        let (_, idx, _) = cb.quantize(&z, 3).expect("quantize should succeed");
        cb.update(&z, &idx, 3).expect("update should succeed");
        let d1 = cb.current_decay();
        assert!(d0 < d1, "decay must increase during warm-up: {d0} → {d1}");
        assert!(d0 < cfg.base_decay, "initial decay below target");
        for _ in 0..20 {
            let (_, idx, _) = cb.quantize(&z, 3).expect("quantize should succeed");
            cb.update(&z, &idx, 3).expect("update should succeed");
        }
        assert!(
            (cb.current_decay() - cfg.base_decay).abs() < 1e-6,
            "decay should reach target after warm-up"
        );
    }

    #[test]
    fn quantize_finite_and_valid_indices() {
        let cfg = EmaCodebookConfig::default();
        let mut rng = LcgRng::new(51);
        let cb = EmaCodebook::new(8, 3, cfg, &mut rng).expect("new should succeed");
        let batch = 5;
        let mut z = vec![0.0_f32; batch * 3];
        let mut rng2 = LcgRng::new(52);
        z.iter_mut().for_each(|v| *v = uniform_sym(&mut rng2, 1.0));
        let (q, idx, commit) = cb.quantize(&z, batch).expect("quantize should succeed");
        assert_eq!(q.len(), batch * 3);
        assert_eq!(idx.len(), batch);
        assert!(commit.is_finite() && commit >= 0.0);
        for &i in &idx {
            assert!(i < cb.n_codes(), "index {i} out of range");
        }
    }

    #[test]
    fn train_step_round_trip() {
        let cfg = EmaCodebookConfig::default();
        let mut rng = LcgRng::new(61);
        let mut cb = EmaCodebook::new(4, 2, cfg, &mut rng).expect("new should succeed");
        let batch = 4;
        let z = repeat_vec(&[1.5_f32, -0.5], batch);
        let (idx, commit, _revived) = cb
            .train_step(&z, batch, &mut rng)
            .expect("train_step should succeed");
        assert_eq!(idx.len(), batch);
        assert!(commit.is_finite());
        assert!(cb.embeddings().iter().all(|v| v.is_finite()));
        assert_eq!(cb.step_count(), 1);
    }

    #[test]
    fn rejects_bad_dimensions() {
        let cfg = EmaCodebookConfig::default();
        let mut rng = LcgRng::new(71);
        assert!(EmaCodebook::new(1, 4, cfg, &mut rng).is_err());
        assert!(EmaCodebook::new(4, 0, cfg, &mut rng).is_err());
        let cb = EmaCodebook::new(4, 2, cfg, &mut rng).expect("new should succeed");
        assert!(cb.quantize(&[0.0; 3], 2).is_err());
    }

    /// Training simulation: data drawn from `n_codes` well-separated clusters
    /// should drive codebook *usage* above 80 % once the EMA updates and
    /// dead-code revival have had time to spread the codes over the clusters.
    /// This exercises the full `train_step` loop end-to-end and asserts the
    /// healthy-utilisation property that motivates EMA + revival.
    #[test]
    fn codebook_usage_exceeds_80_percent_after_training() {
        let n_codes = 16usize;
        let dim = 2usize;
        let cfg = EmaCodebookConfig {
            base_decay: 0.95,
            warmup_steps: 20,
            dead_patience: 3,
            init_scale: 0.05,
            ..EmaCodebookConfig::default()
        };
        let mut rng = LcgRng::new(2024);
        let mut cb = EmaCodebook::new(n_codes, dim, cfg, &mut rng).expect("new should succeed");

        // 16 cluster centres on a 4×4 grid, spacing 4.0 (well separated).
        let centers: Vec<[f32; 2]> = (0..n_codes)
            .map(|k| {
                let gx = (k % 4) as f32 * 4.0 - 6.0;
                let gy = (k / 4) as f32 * 4.0 - 6.0;
                [gx, gy]
            })
            .collect();

        let mut data_rng = LcgRng::new(7777);
        let batch = 32usize;
        for _ in 0..400 {
            // Each sample: pick a random centre, add small jitter.
            let mut z = vec![0.0_f32; batch * dim];
            for b in 0..batch {
                let c = centers[data_rng.next_usize(n_codes)];
                let jx = (data_rng.next_f32() - 0.5) * 0.6;
                let jy = (data_rng.next_f32() - 0.5) * 0.6;
                z[b * dim] = c[0] + jx;
                z[b * dim + 1] = c[1] + jy;
            }
            cb.train_step(&z, batch, &mut rng)
                .expect("train_step should succeed");
        }

        // Usage = fraction of codes that have been assigned at least once.
        let used = cb.usage_count().iter().filter(|&&c| c > 0).count();
        let usage = used as f32 / n_codes as f32;
        assert!(
            usage > 0.80,
            "codebook usage should exceed 80%, got {:.1}% ({used}/{n_codes})",
            usage * 100.0
        );
        // Codes should track their clusters: embeddings remain finite & bounded.
        assert!(cb.embeddings().iter().all(|v| v.is_finite()));
    }
}
