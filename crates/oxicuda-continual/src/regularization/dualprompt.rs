//! DualPrompt — complementary G-Prompt + E-Prompt continual learning
//! (Wang 2022).
//!
//! Reference: Wang, Z., Zhang, Z., Ebrahimi, S., Sun, R., Zhang, H., Lee, C.-Y.,
//! Ren, X., Su, G., Perot, V., Dy, J. & Pfister, T. (2022). "DualPrompt:
//! Complementary Prompting for Rehearsal-free Continual Learning." *European
//! Conference on Computer Vision* (ECCV 2022).
//!
//! # Overview
//!
//! DualPrompt extends L2P with **two** kinds of prompts attached to a frozen
//! backbone:
//!
//! - a single **G(eneral)-Prompt** `g`, shared across *all* tasks, which
//!   captures task-invariant instructions;
//! - a pool of **E(xpert)-Prompts** `{e_k}`, one per task, capturing
//!   task-specific knowledge, each paired with a learnable **key** `κ_k`.
//!
//! At inference the input feature `q` (a query, typically the frozen backbone's
//! `[CLS]` embedding) is matched against the E-Prompt keys by cosine similarity;
//! the highest-scoring expert prompt is selected and concatenated with the
//! G-Prompt to condition the backbone. Because the two prompt families should
//! encode *disjoint* (general vs. specific) information, DualPrompt adds an
//! **orthogonality regulariser** that drives the inner product between the
//! G-Prompt and each E-Prompt toward zero:
//!
//! ```text
//!   L_orth = Σ_k ( ⟨ḡ, ē_k⟩ )²          (ḡ, ē_k are flattened prompt vectors)
//! ```
//!
//! All tensors are FP32 and stored row-major. Prompt pools are initialised
//! from `N(0, 1)` via the deterministic [`ContRng`].

use crate::error::{ContinualError, ContinualResult};
use crate::handle::LcgRng;

/// Type alias for the crate-local RNG used in DualPrompt operations.
pub type ContRng = LcgRng;

/// Configuration for [`DualPrompt`].
#[derive(Debug, Clone)]
pub struct DualPromptConfig {
    /// Number of tokens in the shared G-Prompt (`L_g`).
    pub g_length: usize,
    /// Number of tokens in each task-specific E-Prompt (`L_e`).
    pub e_length: usize,
    /// Embedding dimension of every prompt token / query (`d`).
    pub d_model: usize,
    /// Number of tasks (E-Prompts) in the pool.
    pub n_tasks: usize,
}

impl DualPromptConfig {
    fn validate(&self) -> ContinualResult<()> {
        if self.d_model == 0 {
            return Err(ContinualError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if self.g_length == 0 || self.e_length == 0 {
            return Err(ContinualError::EmptyInput);
        }
        if self.n_tasks == 0 {
            return Err(ContinualError::NoTasksInStream);
        }
        Ok(())
    }
}

/// DualPrompt state: a single shared G-Prompt and a per-task E-Prompt pool with
/// matching keys.
#[derive(Debug, Clone)]
pub struct DualPrompt {
    /// Shared G-Prompt, flat `[g_length × d_model]`.
    g_prompt: Vec<f32>,
    /// E-Prompt pool, flat `[n_tasks × e_length × d_model]`.
    e_prompts: Vec<f32>,
    /// L2-normalised E-Prompt keys, flat `[n_tasks × d_model]`.
    keys: Vec<f32>,
    config: DualPromptConfig,
}

impl DualPrompt {
    /// Create a new DualPrompt with `N(0, 1)`-initialised prompts and
    /// L2-normalised keys.
    ///
    /// # Errors
    /// - [`ContinualError::DimensionMismatch`] if `d_model == 0`.
    /// - [`ContinualError::EmptyInput`] if `g_length == 0` or `e_length == 0`.
    /// - [`ContinualError::NoTasksInStream`] if `n_tasks == 0`.
    pub fn new(config: DualPromptConfig, rng: &mut ContRng) -> ContinualResult<Self> {
        config.validate()?;

        let mut g_prompt = vec![0.0_f32; config.g_length * config.d_model];
        rng.fill_normal(&mut g_prompt);

        let mut e_prompts = vec![0.0_f32; config.n_tasks * config.e_length * config.d_model];
        rng.fill_normal(&mut e_prompts);

        let mut keys = vec![0.0_f32; config.n_tasks * config.d_model];
        rng.fill_normal(&mut keys);
        for k in 0..config.n_tasks {
            let s = k * config.d_model;
            let e = s + config.d_model;
            l2_normalize_in_place(&mut keys[s..e]);
        }

        Ok(Self {
            g_prompt,
            e_prompts,
            keys,
            config,
        })
    }

    /// Number of tasks in the E-Prompt pool.
    #[must_use]
    pub fn n_tasks(&self) -> usize {
        self.config.n_tasks
    }

    /// Borrow the shared G-Prompt (flat `[g_length × d_model]`).
    #[must_use]
    pub fn g_prompt(&self) -> &[f32] {
        &self.g_prompt
    }

    /// Borrow task `task`'s E-Prompt (flat `[e_length × d_model]`).
    ///
    /// # Errors
    /// - [`ContinualError::TaskIndexOutOfRange`] if `task >= n_tasks`.
    pub fn e_prompt(&self, task: usize) -> ContinualResult<&[f32]> {
        if task >= self.config.n_tasks {
            return Err(ContinualError::TaskIndexOutOfRange {
                index: task,
                n_tasks: self.config.n_tasks,
            });
        }
        let span = self.config.e_length * self.config.d_model;
        let s = task * span;
        Ok(&self.e_prompts[s..s + span])
    }

    /// Select the best-matching E-Prompt index for a query `q` by maximal cosine
    /// similarity against the keys. Returns `(task_index, similarity)`.
    ///
    /// # Errors
    /// - [`ContinualError::DimensionMismatch`] if `q.len() != d_model`.
    pub fn match_task(&self, q: &[f32]) -> ContinualResult<(usize, f32)> {
        if q.len() != self.config.d_model {
            return Err(ContinualError::DimensionMismatch {
                expected: self.config.d_model,
                got: q.len(),
            });
        }
        let qn = l2_normalize(q);
        let mut best_idx = 0usize;
        let mut best_sim = f32::NEG_INFINITY;
        for k in 0..self.config.n_tasks {
            let s = k * self.config.d_model;
            let key = &self.keys[s..s + self.config.d_model];
            let sim: f32 = qn.iter().zip(key.iter()).map(|(&a, &b)| a * b).sum();
            if sim > best_sim {
                best_sim = sim;
                best_idx = k;
            }
        }
        Ok((best_idx, best_sim))
    }

    /// Assemble the conditioning prompt for `task` by concatenating the shared
    /// G-Prompt followed by that task's E-Prompt. The result has length
    /// `(g_length + e_length) · d_model`.
    ///
    /// # Errors
    /// - [`ContinualError::TaskIndexOutOfRange`] if `task >= n_tasks`.
    pub fn assemble_prompt(&self, task: usize) -> ContinualResult<Vec<f32>> {
        let e = self.e_prompt(task)?;
        let mut out = Vec::with_capacity(self.g_prompt.len() + e.len());
        out.extend_from_slice(&self.g_prompt);
        out.extend_from_slice(e);
        Ok(out)
    }

    /// Orthogonality regulariser `L_orth = Σ_k ⟨ḡ, ē_k⟩²`, where `ḡ` and `ē_k`
    /// are the flattened G/E prompt vectors truncated to their common length
    /// `min(|g|, |e|)` (the overlap over which orthogonality is enforced).
    ///
    /// A value near `0` means the general and expert prompts are (approximately)
    /// orthogonal, i.e. encode disjoint information.
    #[must_use]
    pub fn orthogonality_penalty(&self) -> f32 {
        let g_len = self.g_prompt.len();
        let e_span = self.config.e_length * self.config.d_model;
        let overlap = g_len.min(e_span);
        let mut total = 0.0_f32;
        for k in 0..self.config.n_tasks {
            let s = k * e_span;
            let ek = &self.e_prompts[s..s + e_span];
            let dot: f32 = self.g_prompt[..overlap]
                .iter()
                .zip(ek[..overlap].iter())
                .map(|(&a, &b)| a * b)
                .sum();
            total += dot * dot;
        }
        total
    }

    /// Overwrite task `task`'s E-Prompt and key (e.g. after learning a task).
    /// The provided key is L2-normalised on entry.
    ///
    /// # Errors
    /// - [`ContinualError::TaskIndexOutOfRange`] if `task >= n_tasks`.
    /// - [`ContinualError::DimensionMismatch`] if `prompt`/`key` lengths are
    ///   wrong.
    pub fn set_task(&mut self, task: usize, prompt: &[f32], key: &[f32]) -> ContinualResult<()> {
        if task >= self.config.n_tasks {
            return Err(ContinualError::TaskIndexOutOfRange {
                index: task,
                n_tasks: self.config.n_tasks,
            });
        }
        let span = self.config.e_length * self.config.d_model;
        if prompt.len() != span {
            return Err(ContinualError::DimensionMismatch {
                expected: span,
                got: prompt.len(),
            });
        }
        if key.len() != self.config.d_model {
            return Err(ContinualError::DimensionMismatch {
                expected: self.config.d_model,
                got: key.len(),
            });
        }
        let s = task * span;
        self.e_prompts[s..s + span].copy_from_slice(prompt);
        let ks = task * self.config.d_model;
        self.keys[ks..ks + self.config.d_model].copy_from_slice(key);
        l2_normalize_in_place(&mut self.keys[ks..ks + self.config.d_model]);
        Ok(())
    }
}

// ─── helpers ──────────────────────────────────────────────────────────────────

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let n = l2_norm(v).max(1e-12);
    v.iter().map(|&x| x / n).collect()
}

fn l2_normalize_in_place(v: &mut [f32]) {
    let n = l2_norm(v).max(1e-12);
    for x in v.iter_mut() {
        *x /= n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> DualPromptConfig {
        DualPromptConfig {
            g_length: 3,
            e_length: 2,
            d_model: 4,
            n_tasks: 3,
        }
    }

    // -------------------- construction / validation ------------------------

    #[test]
    fn new_ok_shapes() {
        let mut rng = ContRng::new(1);
        let dp = DualPrompt::new(cfg(), &mut rng)
            .expect("DualPrompt should construct with valid config");
        assert_eq!(dp.g_prompt().len(), 3 * 4);
        assert_eq!(dp.n_tasks(), 3);
        assert_eq!(
            dp.e_prompt(0)
                .expect("e-prompt should be accessible for valid task index")
                .len(),
            2 * 4
        );
    }

    #[test]
    fn new_d_model_zero_error() {
        let mut rng = ContRng::new(1);
        let c = DualPromptConfig {
            d_model: 0,
            ..cfg()
        };
        assert!(matches!(
            DualPrompt::new(c, &mut rng),
            Err(ContinualError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn new_zero_length_error() {
        let mut rng = ContRng::new(1);
        let c = DualPromptConfig {
            g_length: 0,
            ..cfg()
        };
        assert!(matches!(
            DualPrompt::new(c, &mut rng),
            Err(ContinualError::EmptyInput)
        ));
    }

    #[test]
    fn new_zero_tasks_error() {
        let mut rng = ContRng::new(1);
        let c = DualPromptConfig {
            n_tasks: 0,
            ..cfg()
        };
        assert!(matches!(
            DualPrompt::new(c, &mut rng),
            Err(ContinualError::NoTasksInStream)
        ));
    }

    // -------------------- keys / matching ----------------------------------

    #[test]
    fn keys_l2_normalised() {
        let mut rng = ContRng::new(7);
        let dp = DualPrompt::new(cfg(), &mut rng)
            .expect("DualPrompt should construct with valid config");
        for k in 0..dp.n_tasks() {
            let s = k * 4;
            let norm = l2_norm(&dp.keys[s..s + 4]);
            assert!((norm - 1.0).abs() < 1e-5, "key {k} norm = {norm}");
        }
    }

    #[test]
    fn match_task_in_range() {
        let mut rng = ContRng::new(9);
        let dp = DualPrompt::new(cfg(), &mut rng)
            .expect("DualPrompt should construct with valid config");
        let q = vec![0.5_f32, -0.2, 0.1, 0.3];
        let (idx, sim) = dp
            .match_task(&q)
            .expect("task matching should succeed with valid query");
        assert!(idx < dp.n_tasks());
        assert!((-1.0001..=1.0001).contains(&sim), "cosine sim {sim}");
    }

    #[test]
    fn match_task_recovers_planted_key() {
        // Set task 1's key to a known direction and query with it.
        let mut rng = ContRng::new(11);
        let mut dp = DualPrompt::new(cfg(), &mut rng)
            .expect("DualPrompt should construct with valid config");
        let prompt = vec![0.0_f32; 2 * 4];
        let key = vec![1.0_f32, 0.0, 0.0, 0.0];
        dp.set_task(1, &prompt, &key)
            .expect("setting task prompt should succeed");
        let (idx, sim) = dp
            .match_task(&[10.0, 0.0, 0.0, 0.0])
            .expect("task matching should succeed with valid query");
        assert_eq!(idx, 1, "should match planted task 1");
        assert!((sim - 1.0).abs() < 1e-5, "cosine should be 1, got {sim}");
    }

    #[test]
    fn match_task_dim_mismatch_error() {
        let mut rng = ContRng::new(9);
        let dp = DualPrompt::new(cfg(), &mut rng)
            .expect("DualPrompt should construct with valid config");
        let r = dp.match_task(&[0.1, 0.2]); // wrong dim
        assert!(matches!(r, Err(ContinualError::DimensionMismatch { .. })));
    }

    // -------------------- assembly -----------------------------------------

    #[test]
    fn assemble_prompt_length() {
        let mut rng = ContRng::new(3);
        let dp = DualPrompt::new(cfg(), &mut rng)
            .expect("DualPrompt should construct with valid config");
        let p = dp
            .assemble_prompt(2)
            .expect("e-prompt should be accessible for valid task index");
        // (g_length + e_length) * d_model = (3 + 2) * 4 = 20
        assert_eq!(p.len(), 20);
        // first g_length*d_model entries are exactly the G-Prompt.
        assert_eq!(&p[..12], dp.g_prompt());
    }

    #[test]
    fn assemble_prompt_out_of_range_error() {
        let mut rng = ContRng::new(3);
        let dp = DualPrompt::new(cfg(), &mut rng)
            .expect("DualPrompt should construct with valid config");
        let r = dp.assemble_prompt(99);
        assert!(matches!(r, Err(ContinualError::TaskIndexOutOfRange { .. })));
    }

    #[test]
    fn e_prompt_out_of_range_error() {
        let mut rng = ContRng::new(3);
        let dp = DualPrompt::new(cfg(), &mut rng)
            .expect("DualPrompt should construct with valid config");
        assert!(matches!(
            dp.e_prompt(5),
            Err(ContinualError::TaskIndexOutOfRange { .. })
        ));
    }

    // -------------------- orthogonality ------------------------------------

    #[test]
    fn orthogonality_penalty_nonneg() {
        let mut rng = ContRng::new(21);
        let dp = DualPrompt::new(cfg(), &mut rng)
            .expect("DualPrompt should construct with valid config");
        assert!(dp.orthogonality_penalty() >= 0.0);
    }

    #[test]
    fn orthogonality_zero_when_e_orthogonal_to_g() {
        // Force every E-Prompt orthogonal to the G-Prompt by zeroing them.
        let mut rng = ContRng::new(33);
        let mut dp = DualPrompt::new(cfg(), &mut rng)
            .expect("DualPrompt should construct with valid config");
        let zero_prompt = vec![0.0_f32; 2 * 4];
        let key = vec![1.0_f32, 0.0, 0.0, 0.0];
        for k in 0..dp.n_tasks() {
            dp.set_task(k, &zero_prompt, &key)
                .expect("setting task prompt should succeed");
        }
        assert!(
            dp.orthogonality_penalty().abs() < 1e-6,
            "zero E-prompts ⇒ zero penalty"
        );
    }

    #[test]
    fn orthogonality_positive_when_aligned() {
        // Make all E-Prompts equal to a copy of the G-Prompt overlap → dot > 0.
        let mut rng = ContRng::new(44);
        let mut dp = DualPrompt::new(cfg(), &mut rng)
            .expect("DualPrompt should construct with valid config");
        // e_span = 2*4 = 8 == g_len = 3*4 = 12? No: overlap = min(12, 8) = 8.
        // Build an E-Prompt equal to the first 8 entries of g, padded length 8.
        let g = dp.g_prompt().to_vec();
        let e: Vec<f32> = g[..8].to_vec();
        let key = vec![1.0_f32, 0.0, 0.0, 0.0];
        dp.set_task(0, &e, &key)
            .expect("setting task prompt should succeed");
        assert!(
            dp.orthogonality_penalty() > 0.0,
            "aligned prompts ⇒ positive penalty"
        );
    }

    // -------------------- set_task -----------------------------------------

    #[test]
    fn set_task_normalises_key_and_replaces_prompt() {
        let mut rng = ContRng::new(55);
        let mut dp = DualPrompt::new(cfg(), &mut rng)
            .expect("DualPrompt should construct with valid config");
        let prompt: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let key = vec![3.0_f32, 4.0, 0.0, 0.0]; // norm 5 → normalised
        dp.set_task(0, &prompt, &key)
            .expect("setting task prompt should succeed");
        assert_eq!(
            dp.e_prompt(0)
                .expect("e-prompt should be accessible for valid task index"),
            prompt.as_slice()
        );
        let s = 0;
        let kn = l2_norm(&dp.keys[s..s + 4]);
        assert!((kn - 1.0).abs() < 1e-5, "key should be unit norm, got {kn}");
    }

    #[test]
    fn set_task_dim_mismatch_error() {
        let mut rng = ContRng::new(66);
        let mut dp = DualPrompt::new(cfg(), &mut rng)
            .expect("DualPrompt should construct with valid config");
        let r = dp.set_task(0, &[0.0; 3], &[1.0, 0.0, 0.0, 0.0]); // wrong prompt len
        assert!(matches!(r, Err(ContinualError::DimensionMismatch { .. })));
    }

    #[test]
    fn deterministic_construction() {
        let mut a = ContRng::new(1234);
        let mut b = ContRng::new(1234);
        let dp_a =
            DualPrompt::new(cfg(), &mut a).expect("DualPrompt should construct with valid config");
        let dp_b =
            DualPrompt::new(cfg(), &mut b).expect("DualPrompt should construct with valid config");
        assert_eq!(dp_a.g_prompt(), dp_b.g_prompt());
        assert_eq!(dp_a.orthogonality_penalty(), dp_b.orthogonality_penalty());
    }
}
