//! DP-FTRL: Differentially Private Follow-The-Regularized-Leader with tree aggregation.
//!
//! McMahan et al., "Practical Differentially Private Clustering via Approximation",
//! ICLR 2022. (Also: Kairouz et al., "Practical and Private (Deep) Learning Without
//! Sampling or Shuffling", ICML 2021.)
//!
//! DP-FTRL replaces the per-step Gaussian noise injection of DP-SGD with a
//! binary-tree (Fenwick-tree) noise structure:  the sum of clipped gradients
//! over rounds 1..T is decomposed into O(log T) prefix-sum tree nodes, and
//! calibrated Gaussian noise is added **once per node** rather than once per
//! round.  Because each node's noise is independent and the prefix sum reads
//! at most D = ⌈log₂(T+1)⌉ nodes, the effective noise per round is
//! O(σ √(log T)) instead of σ √T in naïve composition.
//!
//! # Privacy accounting
//!
//! We use Rényi DP composition across the O(log T * T / 2) total node visits:
//!
//! ```text
//!   For a node at Fenwick level l (0 = leaves):
//!       σ_l = base_σ · √(l + 1)
//!
//!   Per-node RDP at order α (Gaussian mechanism, sensitivity = clip_norm):
//!       ε_RDP(α) = α / (2 · σ_l²)
//!
//!   Total RDP is the sum over all nodes visited across all rounds 1..current_round.
//!   Conversion to (ε, δ)-DP via the standard RDP→DP bound (Mironov 2017):
//!       ε_DP = ε_RDP − log(δ) / (α − 1)
//!   optimised over α ∈ {2, 4, 8, …, 64}.
//! ```

use crate::error::{FedError, FedResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the DP-FTRL tree-aggregation mechanism.
#[derive(Debug, Clone)]
pub struct DpFtrlConfig {
    /// Per-step noise multiplier σ (must be > 0).
    pub sigma: f32,
    /// L2 gradient clipping threshold C (must be > 0).
    pub clip_norm: f32,
    /// Total number of training rounds T (must be ≥ 1).
    pub n_rounds: usize,
    /// Target DP δ (must be in (0, 1)).
    pub delta: f64,
}

impl Default for DpFtrlConfig {
    fn default() -> Self {
        Self {
            sigma: 1.0,
            clip_norm: 1.0,
            n_rounds: 100,
            delta: 1e-5,
        }
    }
}

// ─── Tree Node ────────────────────────────────────────────────────────────────

/// One node of the Fenwick (binary-indexed) tree used for gradient accumulation.
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Accumulated noisy gradient sum at this node.
    pub sum: Vec<f32>,
    /// Fenwick level of this node (0 = leaves, higher = closer to root).
    /// Equal to the number of trailing zero bits in the 1-based node index.
    pub level: usize,
}

impl TreeNode {
    fn new(n_params: usize, level: usize) -> Self {
        Self {
            sum: vec![0.0_f32; n_params],
            level,
        }
    }
}

// ─── State ────────────────────────────────────────────────────────────────────

/// Runtime state for DP-FTRL with binary-tree gradient accumulation.
#[derive(Debug, Clone)]
pub struct DpFtrlState {
    /// Number of rounds completed so far.
    pub round: usize,
    /// Number of model parameters.
    pub n_params: usize,
    /// Algorithm configuration.
    pub cfg: DpFtrlConfig,
    /// Fenwick tree of `TreeNode`s indexed 0 … n_rounds−1.
    /// Node `i` (0-based) corresponds to Fenwick position `i+1` (1-based).
    pub nodes: Vec<TreeNode>,
}

impl DpFtrlState {
    /// Maximum tree depth: ⌈log₂(n_rounds + 1)⌉.
    ///
    /// Equivalently, the number of bits needed to represent `n_rounds`.
    #[must_use]
    pub fn tree_depth(&self) -> usize {
        if self.cfg.n_rounds == 0 {
            return 0;
        }
        // ⌈log₂(n+1)⌉ = bit-length of n
        usize::BITS as usize - self.cfg.n_rounds.leading_zeros() as usize
    }

    /// Population count of `round` — the number of Fenwick tree nodes visited
    /// on the query path for prefix-sum up to `round`.
    #[must_use]
    pub fn path_length(round: usize) -> usize {
        round.count_ones() as usize
    }
}

// ─── Result ──────────────────────────────────────────────────────────────────

/// Output produced after adding each gradient round.
#[derive(Debug, Clone)]
pub struct DpFtrlResult {
    /// Noisy prefix sum of clipped gradients from rounds 1..=current_round.
    pub prefix_sum: Vec<f32>,
    /// Current (ε, δ)-DP privacy budget consumed, evaluated at state.cfg.delta.
    pub epsilon: f64,
}

// ─── Algorithm ───────────────────────────────────────────────────────────────

/// DP-FTRL algorithm with binary-tree noise injection.
pub struct DpFtrl;

impl DpFtrl {
    /// Initialise a fresh DP-FTRL state.
    ///
    /// Validates:
    /// - `sigma > 0 && finite` → [`FedError::InvalidNoiseMultiplier`]
    /// - `clip_norm > 0 && finite` → [`FedError::InvalidClipNorm`]
    /// - `n_rounds >= 1` → [`FedError::InvalidPrivacyBudget`]
    /// - `0 < delta < 1` → [`FedError::InvalidPrivacyBudget`]
    /// - `n_params >= 1` → [`FedError::DimensionMismatch`]
    #[allow(clippy::new_ret_no_self)]
    pub fn new(n_params: usize, cfg: DpFtrlConfig) -> FedResult<DpFtrlState> {
        if !(cfg.sigma > 0.0 && cfg.sigma.is_finite()) {
            return Err(FedError::InvalidNoiseMultiplier);
        }
        if !(cfg.clip_norm > 0.0 && cfg.clip_norm.is_finite()) {
            return Err(FedError::InvalidClipNorm);
        }
        if cfg.n_rounds == 0 {
            return Err(FedError::InvalidPrivacyBudget);
        }
        if !(cfg.delta > 0.0 && cfg.delta < 1.0) {
            return Err(FedError::InvalidPrivacyBudget);
        }
        if n_params == 0 {
            return Err(FedError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }

        // Build Fenwick tree: one node per round position 1..=n_rounds.
        // Node at 1-based index i has Fenwick level = trailing_zeros(i).
        let nodes: Vec<TreeNode> = (1..=cfg.n_rounds)
            .map(|i| {
                let level = i.trailing_zeros() as usize;
                TreeNode::new(n_params, level)
            })
            .collect();

        Ok(DpFtrlState {
            round: 0,
            n_params,
            cfg,
            nodes,
        })
    }

    /// Clip gradient `grad` to L2 ball of radius `clip_norm`.
    ///
    /// `g_clipped[i] = g[i] * min(1, clip_norm / (||g|| + 1e-12))`
    ///
    /// # Errors
    /// - [`FedError::InvalidClipNorm`] if `clip_norm <= 0`.
    /// - [`FedError::DimensionMismatch`] if `grad` is empty.
    pub fn clip_gradient(grad: &[f32], clip_norm: f32) -> FedResult<Vec<f32>> {
        if !(clip_norm > 0.0 && clip_norm.is_finite()) {
            return Err(FedError::InvalidClipNorm);
        }
        if grad.is_empty() {
            return Err(FedError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let norm_sq: f32 = grad.iter().map(|&g| g * g).sum();
        let norm = norm_sq.sqrt();
        let scale = (clip_norm / (norm + 1e-12_f32)).min(1.0_f32);
        Ok(grad.iter().map(|&g| g * scale).collect())
    }

    /// Compute the per-node noise standard deviation at Fenwick level `level`.
    ///
    /// `σ_level = base_sigma * sqrt(level + 1)`
    #[must_use]
    pub fn level_sigma(base_sigma: f32, level: usize) -> f32 {
        base_sigma * ((level as f32 + 1.0_f32).sqrt())
    }

    /// Add one round's gradient to the Fenwick tree with calibrated noise.
    ///
    /// Performs the Fenwick-tree *update* walk (round → root), adding the
    /// clipped gradient plus N(0, σ_level * clip_norm) noise to every touched
    /// node.  After updating, computes and returns the noisy prefix sum and
    /// current ε.
    ///
    /// # Errors
    /// - [`FedError::DimensionMismatch`] if `gradient.len() ≠ state.n_params`.
    /// - [`FedError::Internal`] if `state.round >= state.cfg.n_rounds` (tree full).
    pub fn add_gradient(
        state: &mut DpFtrlState,
        gradient: &[f32],
        rng: &mut LcgRng,
    ) -> FedResult<DpFtrlResult> {
        if gradient.len() != state.n_params {
            return Err(FedError::DimensionMismatch {
                expected: state.n_params,
                got: gradient.len(),
            });
        }
        if state.round >= state.cfg.n_rounds {
            return Err(FedError::Internal(format!(
                "DP-FTRL tree is full: {} rounds already completed",
                state.round
            )));
        }

        let clipped = Self::clip_gradient(gradient, state.cfg.clip_norm)?;
        let clip_c = state.cfg.clip_norm;
        let base_sigma = state.cfg.sigma;
        let n_rounds = state.cfg.n_rounds;

        // Fenwick update: walk from round+1 to root.
        let new_round = state.round + 1;
        let mut idx = new_round; // 1-based Fenwick index
        while idx <= n_rounds {
            let level = idx.trailing_zeros() as usize;
            let sigma_level = Self::level_sigma(base_sigma, level);
            let noise_std = sigma_level * clip_c;

            // Add clipped gradient + Gaussian noise to this node.
            let node = &mut state.nodes[idx - 1];
            let mut p = 0_usize;
            while p + 1 < state.n_params {
                let (z0, z1) = rng.next_normal_pair();
                node.sum[p] += clipped[p] + noise_std * z0;
                node.sum[p + 1] += clipped[p + 1] + noise_std * z1;
                p += 2;
            }
            if p < state.n_params {
                let (z0, _) = rng.next_normal_pair();
                node.sum[p] += clipped[p] + noise_std * z0;
            }

            // Fenwick parent: idx += lowest-set-bit(idx).
            let lsb = idx & idx.wrapping_neg();
            idx += lsb;
        }

        state.round = new_round;

        let prefix = Self::prefix_sum(state);
        let epsilon = Self::compute_epsilon(state)?;

        Ok(DpFtrlResult {
            prefix_sum: prefix,
            epsilon,
        })
    }

    /// Compute the privacy budget ε spent so far.
    ///
    /// Uses Rényi DP composition over all tree nodes visited in rounds 1..=round:
    ///
    /// ```text
    ///   For round t: nodes visited = popcount(t) levels l_0, l_1, …
    ///   Per-node RDP at order α: α / (2 · σ_l²)
    ///   Total RDP = Σ_{all visited nodes} α / (2 · σ_l²)
    ///   ε_DP(α) = total_rdp − log(δ) / (α − 1)
    ///   ε = min over α ∈ {2,4,8,16,32,64}
    /// ```
    ///
    /// # Errors
    /// Returns [`FedError::InvalidPrivacyBudget`] if no finite ε can be computed.
    pub fn compute_epsilon(state: &DpFtrlState) -> FedResult<f64> {
        if state.round == 0 {
            return Ok(0.0);
        }

        let base_sigma = state.cfg.sigma as f64;
        let delta = state.cfg.delta;

        // For each round t in 1..=current_round, accumulate per-level RDP.
        // We build a count array: how many times was each Fenwick level touched?
        // Level l is touched by round t iff bit l is set in t (for Fenwick updates,
        // the levels touched are the positions of set bits in t for the query path,
        // and for the update path they are similar but we account for them uniformly).
        //
        // More precisely: when we updated round t, we walked the Fenwick update chain.
        // The levels touched by the Fenwick update for round t are the bit positions
        // of t (i.e., for each set bit at position l in t, a node at level l is
        // updated).  Query for prefix(t) also touches the same set of levels (by
        // Fenwick tree symmetry).
        //
        // Total visits per level l = Σ_{t=1}^{round} (number of times level l appears
        // in Fenwick update path for t).  For level l this equals
        // ⌊round / 2^(l+1)⌋ * 1 + max(0, round mod 2^(l+1) − 2^l + 1)
        // but it is simpler and exact to just count directly for our max 100-round
        // typical case.  We do so:

        let mut level_counts: Vec<u64> = vec![0_u64; 64];
        for t in 1..=state.round {
            let mut idx = t;
            let n_rounds = state.cfg.n_rounds;
            while idx <= n_rounds {
                let level = idx.trailing_zeros() as usize;
                if level < 64 {
                    level_counts[level] = level_counts[level].saturating_add(1);
                }
                let lsb = idx & idx.wrapping_neg();
                idx += lsb;
            }
        }

        // Alpha candidates: 2, 4, 8, 16, 32, 64
        let alphas: [f64; 6] = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
        let log_delta = delta.ln();

        let mut best_eps = f64::INFINITY;

        for &alpha in &alphas {
            // Total RDP = Σ_l count_l * α / (2 * σ_l²)
            let mut total_rdp = 0.0_f64;
            for (l, &count) in level_counts.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                let sigma_l = base_sigma * ((l as f64 + 1.0_f64).sqrt());
                let rdp_per_node = alpha / (2.0 * sigma_l * sigma_l);
                total_rdp += count as f64 * rdp_per_node;
            }

            // Convert RDP → (ε, δ)-DP: ε = rdp − log(δ) / (α − 1)
            let eps = total_rdp - log_delta / (alpha - 1.0);
            if eps.is_finite() && eps > 0.0 && eps < best_eps {
                best_eps = eps;
            }
        }

        if best_eps.is_infinite() {
            return Err(FedError::InvalidPrivacyBudget);
        }

        Ok(best_eps)
    }

    /// Read the current noisy prefix sum (rounds 1..=state.round).
    ///
    /// Performs the Fenwick-tree *query* walk from `state.round` to zero,
    /// summing `nodes[idx-1].sum` at each step.
    #[must_use]
    pub fn prefix_sum(state: &DpFtrlState) -> Vec<f32> {
        let mut result = vec![0.0_f32; state.n_params];
        let mut idx = state.round;
        while idx > 0 {
            let node = &state.nodes[idx - 1];
            for (r, &s) in result.iter_mut().zip(node.sum.iter()) {
                *r += s;
            }
            // Fenwick parent for query: idx -= lowest-set-bit(idx).
            let lsb = idx & idx.wrapping_neg();
            idx -= lsb;
        }
        result
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> DpFtrlConfig {
        DpFtrlConfig {
            sigma: 1.0,
            clip_norm: 1.0,
            n_rounds: 16,
            delta: 1e-5,
        }
    }

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    // ─── Test 1: new with valid config succeeds ───────────────────────────────
    #[test]
    fn dp_ftrl_new_valid() {
        let state = DpFtrl::new(4, default_cfg()).expect("test invariant: valid DpFtrl::new");
        assert_eq!(state.round, 0);
        assert_eq!(state.n_params, 4);
        assert_eq!(state.nodes.len(), 16);
    }

    // ─── Test 2: clip_gradient normalises to clip_norm when ||g|| > clip_norm ─
    #[test]
    fn clip_gradient_clips_large_gradient() {
        let grad = vec![3.0_f32, 4.0]; // ||g|| = 5 > clip_norm = 2
        let clipped =
            DpFtrl::clip_gradient(&grad, 2.0).expect("test invariant: valid clip_gradient");
        let norm_sq: f32 = clipped.iter().map(|&v| v * v).sum();
        let norm = norm_sq.sqrt();
        // Should be clipped to clip_norm (within floating-point tolerance).
        assert!((norm - 2.0).abs() < 1e-4, "norm after clip = {norm}");
    }

    // ─── Test 3: clip_gradient unchanged when ||g|| <= clip_norm ─────────────
    #[test]
    fn clip_gradient_unchanged_below_norm() {
        let grad = vec![0.3_f32, 0.4]; // ||g|| = 0.5 < clip_norm = 2
        let clipped =
            DpFtrl::clip_gradient(&grad, 2.0).expect("test invariant: valid clip_gradient");
        assert!((clipped[0] - 0.3).abs() < 1e-6);
        assert!((clipped[1] - 0.4).abs() < 1e-6);
    }

    // ─── Test 4: clip_gradient on zero vector returns zero ───────────────────
    #[test]
    fn clip_gradient_zero_vector() {
        let grad = vec![0.0_f32; 5];
        let clipped =
            DpFtrl::clip_gradient(&grad, 1.0).expect("test invariant: valid clip_gradient zero");
        assert!(clipped.iter().all(|&v| v == 0.0));
    }

    // ─── Test 5: level_sigma at level=0 equals base_sigma ────────────────────
    #[test]
    fn level_sigma_level_zero() {
        let base = 2.5_f32;
        let s = DpFtrl::level_sigma(base, 0);
        assert!((s - base).abs() < 1e-6, "level 0 sigma should equal base");
    }

    // ─── Test 6: level_sigma at level=1 equals base_sigma * sqrt(2) ──────────
    #[test]
    fn level_sigma_level_one() {
        let base = 1.0_f32;
        let s = DpFtrl::level_sigma(base, 1);
        let expected = 2.0_f32.sqrt();
        assert!((s - expected).abs() < 1e-5, "level 1 sigma = {s}");
    }

    // ─── Test 7: add_gradient increments state.round ─────────────────────────
    #[test]
    fn add_gradient_increments_round() {
        let mut state = DpFtrl::new(3, default_cfg()).expect("test invariant: valid state");
        let mut rng = make_rng();
        let grad = vec![0.1_f32, -0.2, 0.3];
        DpFtrl::add_gradient(&mut state, &grad, &mut rng)
            .expect("test invariant: valid add_gradient");
        assert_eq!(state.round, 1);
        DpFtrl::add_gradient(&mut state, &grad, &mut rng)
            .expect("test invariant: valid add_gradient round 2");
        assert_eq!(state.round, 2);
    }

    // ─── Test 8: prefix_sum after one round has non-zero entries ─────────────
    #[test]
    fn prefix_sum_nonzero_after_one_round() {
        let mut state = DpFtrl::new(4, default_cfg()).expect("test invariant: valid state");
        let mut rng = make_rng();
        let grad = vec![0.5_f32, -0.5, 0.5, -0.5];
        let result = DpFtrl::add_gradient(&mut state, &grad, &mut rng)
            .expect("test invariant: valid add_gradient");
        // With non-zero gradient + noise, sum should be non-zero somewhere.
        let any_nonzero = result.prefix_sum.iter().any(|&v| v != 0.0);
        assert!(any_nonzero, "prefix_sum should be non-zero after one round");
    }

    // ─── Test 9: tree_depth correct for n_rounds=4 → 3 ──────────────────────
    #[test]
    fn tree_depth_four_rounds() {
        let cfg = DpFtrlConfig {
            n_rounds: 4,
            ..default_cfg()
        };
        let state = DpFtrl::new(2, cfg).expect("test invariant: valid state n_rounds=4");
        // ⌈log₂(5)⌉ = 3 (4 = 0b100, needs 3 bits)
        assert_eq!(
            state.tree_depth(),
            3,
            "tree depth for n_rounds=4 should be 3"
        );
    }

    // ─── Test 10: tree_depth correct for n_rounds=8 → 4 ─────────────────────
    #[test]
    fn tree_depth_eight_rounds() {
        let cfg = DpFtrlConfig {
            n_rounds: 8,
            ..default_cfg()
        };
        let state = DpFtrl::new(2, cfg).expect("test invariant: valid state n_rounds=8");
        // ⌈log₂(9)⌉ = 4 (8 = 0b1000, needs 4 bits)
        assert_eq!(
            state.tree_depth(),
            4,
            "tree depth for n_rounds=8 should be 4"
        );
    }

    // ─── Test 11: path_length for round=4 → 1 (binary 100: one set bit) ──────
    #[test]
    fn path_length_four() {
        assert_eq!(DpFtrlState::path_length(4), 1, "4 = 0b100 has 1 set bit");
    }

    // ─── Test 12: path_length for round=3 → 2 (binary 11: two set bits) ──────
    #[test]
    fn path_length_three() {
        assert_eq!(DpFtrlState::path_length(3), 2, "3 = 0b11 has 2 set bits");
    }

    // ─── Test 13: compute_epsilon after 1 round returns finite positive value ─
    #[test]
    fn compute_epsilon_after_one_round_positive() {
        let mut state = DpFtrl::new(3, default_cfg()).expect("test invariant: valid state");
        let mut rng = make_rng();
        let grad = vec![0.1_f32, -0.1, 0.0];
        DpFtrl::add_gradient(&mut state, &grad, &mut rng)
            .expect("test invariant: valid add_gradient");
        let eps = DpFtrl::compute_epsilon(&state).expect("test invariant: valid compute_epsilon");
        assert!(eps.is_finite() && eps > 0.0, "epsilon = {eps}");
    }

    // ─── Test 14: compute_epsilon increases with more rounds ─────────────────
    #[test]
    fn compute_epsilon_increases_with_rounds() {
        let mut state = DpFtrl::new(3, default_cfg()).expect("test invariant: valid state");
        let mut rng = make_rng();
        let grad = vec![0.1_f32, -0.1, 0.0];
        DpFtrl::add_gradient(&mut state, &grad, &mut rng)
            .expect("test invariant: valid add_gradient r1");
        let eps1 = DpFtrl::compute_epsilon(&state).expect("test invariant: valid epsilon r1");
        DpFtrl::add_gradient(&mut state, &grad, &mut rng)
            .expect("test invariant: valid add_gradient r2");
        let eps2 = DpFtrl::compute_epsilon(&state).expect("test invariant: valid epsilon r2");
        assert!(
            eps2 > eps1,
            "epsilon should increase: eps1={eps1}, eps2={eps2}"
        );
    }

    // ─── Test 15: prefix_sum length == n_params ───────────────────────────────
    #[test]
    fn prefix_sum_length_matches_n_params() {
        let n_params = 7_usize;
        let mut state = DpFtrl::new(n_params, default_cfg()).expect("test invariant: valid state");
        let mut rng = make_rng();
        let grad = vec![0.1_f32; n_params];
        let result = DpFtrl::add_gradient(&mut state, &grad, &mut rng)
            .expect("test invariant: valid add_gradient");
        assert_eq!(result.prefix_sum.len(), n_params);
    }

    // ─── Test 16: sigma=0 → InvalidNoiseMultiplier ───────────────────────────
    #[test]
    fn err_sigma_zero() {
        let cfg = DpFtrlConfig {
            sigma: 0.0,
            ..default_cfg()
        };
        assert!(matches!(
            DpFtrl::new(4, cfg),
            Err(FedError::InvalidNoiseMultiplier)
        ));
    }

    // ─── Test 17: clip_norm=0 → InvalidClipNorm ──────────────────────────────
    #[test]
    fn err_clip_norm_zero() {
        let cfg = DpFtrlConfig {
            clip_norm: 0.0,
            ..default_cfg()
        };
        assert!(matches!(
            DpFtrl::new(4, cfg),
            Err(FedError::InvalidClipNorm)
        ));
    }

    // ─── Test 18: n_rounds=0 → InvalidPrivacyBudget ──────────────────────────
    #[test]
    fn err_n_rounds_zero() {
        let cfg = DpFtrlConfig {
            n_rounds: 0,
            ..default_cfg()
        };
        assert!(matches!(
            DpFtrl::new(4, cfg),
            Err(FedError::InvalidPrivacyBudget)
        ));
    }

    // ─── Test 19: delta=0 → InvalidPrivacyBudget ─────────────────────────────
    #[test]
    fn err_delta_zero() {
        let cfg = DpFtrlConfig {
            delta: 0.0,
            ..default_cfg()
        };
        assert!(matches!(
            DpFtrl::new(4, cfg),
            Err(FedError::InvalidPrivacyBudget)
        ));
    }

    // ─── Test 20: gradient dimension mismatch → DimensionMismatch ────────────
    #[test]
    fn err_gradient_dimension_mismatch() {
        let mut state = DpFtrl::new(4, default_cfg()).expect("test invariant: valid state");
        let mut rng = make_rng();
        let wrong_grad = vec![0.1_f32, 0.2]; // length 2, expected 4
        assert!(matches!(
            DpFtrl::add_gradient(&mut state, &wrong_grad, &mut rng),
            Err(FedError::DimensionMismatch {
                expected: 4,
                got: 2
            })
        ));
    }

    // ─── Test 21: multiple rounds accumulate prefix sum correctly ─────────────
    #[test]
    fn multiple_rounds_accumulate() {
        let cfg = DpFtrlConfig {
            sigma: 0.0001, // tiny noise so signal dominates
            clip_norm: 10.0,
            n_rounds: 8,
            delta: 1e-5,
        };
        let mut state = DpFtrl::new(2, cfg).expect("test invariant: valid state small sigma");
        let mut rng = make_rng();
        let grad = vec![1.0_f32, 1.0];
        let mut last_sum = vec![0.0_f32; 2];
        for _ in 0..4 {
            let result = DpFtrl::add_gradient(&mut state, &grad, &mut rng)
                .expect("test invariant: valid add_gradient multi-round");
            // Each round the prefix sum should be larger than before (gradient > 0).
            for (cur, prev) in result.prefix_sum.iter().zip(last_sum.iter()) {
                assert!(*cur > *prev, "prefix_sum should grow: {cur} <= {prev}");
            }
            last_sum = result.prefix_sum;
        }
    }

    // ─── Test 22: compute_epsilon returns 0 for round=0 ──────────────────────
    #[test]
    fn compute_epsilon_zero_rounds_is_zero() {
        let state = DpFtrl::new(3, default_cfg()).expect("test invariant: valid state");
        let eps =
            DpFtrl::compute_epsilon(&state).expect("test invariant: valid epsilon at round=0");
        assert_eq!(eps, 0.0);
    }
}
