use crate::handle::LcgRng;
use crate::lora::lora::mat_vec_mul;

/// Configuration for an AdaLoRA adapter.
#[derive(Debug, Clone)]
pub struct AdaloraConfig {
    /// Initial (full) rank of the decomposition.
    pub r: usize,
    /// Scaling factor α.
    pub alpha: f32,
    /// Target rank after importance-based pruning; must be ≤ `r`.
    pub target_r: usize,
}

/// AdaLoRA adapter: ΔW = P · diag(Λ) · Q where P∈ℝ^{d×r}, Λ∈ℝ^r, Q∈ℝ^{r×k}.
///
/// W shape: `[out_features × in_features]`.
/// P shape: `[out_features × rank]` (column-wise orthonormal-like).
/// Q shape: `[rank × in_features]` (row-wise orthonormal-like).
/// Λ shape: `[rank]` (singular value vector).
#[derive(Debug, Clone)]
pub struct AdaloraLinear {
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
    /// Current (full) rank.
    pub rank: usize,
    /// Target rank after pruning.
    pub target_rank: usize,
    /// Effective scale α/r.
    pub scale: f32,
    /// Base weight, shape `[out_features × in_features]`.
    pub w: Vec<f32>,
    /// Left singular vectors P, shape `[out_features × rank]`.
    pub p: Vec<f32>,
    /// Singular values Λ, shape `[rank]`.
    pub lambda: Vec<f32>,
    /// Right singular vectors Q, shape `[rank × in_features]`.
    pub q: Vec<f32>,
}

impl AdaloraLinear {
    /// Construct a new `AdaloraLinear`.
    ///
    /// W = zeros. P and Q are initialised with random normal values then normalised
    /// column/row-wise (orthonormal-like). Λ = 0.01 · ones.
    #[must_use]
    pub fn new(
        in_features: usize,
        out_features: usize,
        cfg: &AdaloraConfig,
        rng: &mut LcgRng,
    ) -> Self {
        let scale = cfg.alpha / cfg.r as f32;
        let w = vec![0.0_f32; out_features * in_features];

        // Initialise P with random normal, normalise each column
        let mut p = vec![0.0_f32; out_features * cfg.r];
        rng.fill_normal(&mut p);
        normalise_columns(&mut p, out_features, cfg.r);

        let lambda = vec![0.01_f32; cfg.r];

        // Initialise Q with random normal, normalise each row
        let mut q = vec![0.0_f32; cfg.r * in_features];
        rng.fill_normal(&mut q);
        normalise_rows(&mut q, cfg.r, in_features);

        Self {
            in_features,
            out_features,
            rank: cfg.r,
            target_rank: cfg.target_r,
            scale,
            w,
            p,
            lambda,
            q,
        }
    }

    /// Compute the forward pass: `(W + ΔW) · x` where `ΔW = P · diag(Λ) · Q`.
    ///
    /// `x` must have length `in_features`. Returns a vector of length `out_features`.
    #[must_use]
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let delta = self.reconstruct_delta();
        let w_eff: Vec<f32> = self
            .w
            .iter()
            .zip(delta.iter())
            .map(|(w, d)| w + d)
            .collect();
        mat_vec_mul(&w_eff, x, self.out_features, self.in_features)
    }

    /// Compute per-rank importance scores: `|λ_i| · ‖P[:,i]‖₂ · ‖Q[i,:]‖₂`.
    #[must_use]
    pub fn importance_scores(&self) -> Vec<f32> {
        (0..self.rank)
            .map(|i| {
                let lambda_abs = self.lambda[i].abs();
                // ‖P[:,i]‖₂
                let p_col_norm = (0..self.out_features)
                    .map(|r| {
                        let v = self.p[r * self.rank + i];
                        v * v
                    })
                    .sum::<f32>()
                    .sqrt();
                // ‖Q[i,:]‖₂
                let q_row_norm = (0..self.in_features)
                    .map(|c| {
                        let v = self.q[i * self.in_features + c];
                        v * v
                    })
                    .sum::<f32>()
                    .sqrt();
                lambda_abs * p_col_norm * q_row_norm
            })
            .collect()
    }

    /// Zero-out the `rank - target_rank` least important singular values (mask by importance).
    pub fn prune_to_target(&mut self) {
        let scores = self.importance_scores();
        // Collect indices sorted by score ascending (lowest first)
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let to_prune = self.rank.saturating_sub(self.target_rank);
        for (i, _) in indexed.iter().take(to_prune) {
            self.lambda[*i] = 0.0;
        }
    }

    /// Reconstruct the full delta matrix `P · diag(Λ) · Q` as a flat `[out × in]` matrix.
    #[must_use]
    pub fn reconstruct_delta(&self) -> Vec<f32> {
        // result = P * diag(Λ) * Q
        // Compute P_scaled = P * diag(Λ) first, then multiply by Q
        let mut result = vec![0.0_f32; self.out_features * self.in_features];
        for i in 0..self.out_features {
            for k in 0..self.rank {
                let p_ik = self.p[i * self.rank + k];
                let lambda_k = self.lambda[k];
                let p_scaled = self.scale * p_ik * lambda_k;
                if p_scaled == 0.0 {
                    continue;
                }
                for j in 0..self.in_features {
                    result[i * self.in_features + j] += p_scaled * self.q[k * self.in_features + j];
                }
            }
        }
        result
    }
}

/// Normalise each column of an `[rows × cols]` row-major matrix to unit L2 norm.
fn normalise_columns(m: &mut [f32], rows: usize, cols: usize) {
    for c in 0..cols {
        let norm_sq: f32 = (0..rows).map(|r| m[r * cols + c].powi(2)).sum();
        let norm = norm_sq.sqrt().max(1e-12);
        for r in 0..rows {
            m[r * cols + c] /= norm;
        }
    }
}

/// Normalise each row of an `[rows × cols]` row-major matrix to unit L2 norm.
fn normalise_rows(m: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let norm_sq: f32 = (0..cols).map(|c| m[r * cols + c].powi(2)).sum();
        let norm = norm_sq.sqrt().max(1e-12);
        for c in 0..cols {
            m[r * cols + c] /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn cfg(r: usize, alpha: f32, target_r: usize) -> AdaloraConfig {
        AdaloraConfig { r, alpha, target_r }
    }

    // -----------------------------------------------------------------------
    // Test 1: reconstruct_delta matches hand-computed scale · P · diag(Λ) · Q
    // -----------------------------------------------------------------------
    #[test]
    fn reconstruct_delta_analytic() {
        // out=2, in=2, rank=1, alpha=1 → scale=1
        let mut rng = LcgRng::new(0);
        let mut layer = AdaloraLinear::new(2, 2, &cfg(1, 1.0, 1), &mut rng);

        // P = [[1.0], [0.0]]  (out=2, rank=1, row-major)
        layer.p = vec![1.0, 0.0];
        layer.lambda = vec![2.0];
        // Q = [[0.5, 0.5]]    (rank=1, in=2, row-major)
        layer.q = vec![0.5, 0.5];

        // Expected: scale * P * diag(Λ) * Q
        //   = 1 * [[1*2*0.5, 1*2*0.5], [0*2*0.5, 0*2*0.5]]
        //   = [[1.0, 1.0], [0.0, 0.0]]
        let delta = layer.reconstruct_delta();
        assert_eq!(delta.len(), 4, "delta shape must be out*in");
        assert!(
            (delta[0] - 1.0).abs() < 1e-5,
            "delta[0,0] expected 1.0, got {}",
            delta[0]
        );
        assert!(
            (delta[1] - 1.0).abs() < 1e-5,
            "delta[0,1] expected 1.0, got {}",
            delta[1]
        );
        assert!(
            delta[2].abs() < 1e-5,
            "delta[1,0] expected 0.0, got {}",
            delta[2]
        );
        assert!(
            delta[3].abs() < 1e-5,
            "delta[1,1] expected 0.0, got {}",
            delta[3]
        );
    }

    // -----------------------------------------------------------------------
    // Test 2: forward(x) = (W + ΔW) · x  where ΔW = scale · P · diag(Λ) · Q
    // -----------------------------------------------------------------------
    #[test]
    fn forward_equals_base_plus_delta() {
        let mut rng = LcgRng::new(42);
        let mut layer = AdaloraLinear::new(2, 2, &cfg(1, 1.0, 1), &mut rng);

        // Identity base weight
        layer.w = vec![1.0, 0.0, 0.0, 1.0];
        layer.p = vec![1.0, 0.0];
        layer.lambda = vec![2.0];
        layer.q = vec![0.5, 0.5]; // delta = [[1,1],[0,0]]

        let x = vec![1.0_f32, 1.0];
        // base: [1, 1]; delta·x: [2, 0]; total: [3, 1]
        let y = layer.forward(&x);
        assert_eq!(y.len(), 2, "output length must equal out_features");
        assert!((y[0] - 3.0).abs() < 1e-5, "y[0] expected 3.0, got {}", y[0]);
        assert!((y[1] - 1.0).abs() < 1e-5, "y[1] expected 1.0, got {}", y[1]);
    }

    // -----------------------------------------------------------------------
    // Test 3: output shape is [out_features]
    // -----------------------------------------------------------------------
    #[test]
    fn forward_output_shape() {
        let mut rng = LcgRng::new(99);
        let layer = AdaloraLinear::new(8, 5, &cfg(2, 4.0, 1), &mut rng);
        let x = vec![0.5_f32; 8];
        assert_eq!(layer.forward(&x).len(), 5);
    }

    // -----------------------------------------------------------------------
    // Test 4: same seed and input produce identical output (determinism)
    // -----------------------------------------------------------------------
    #[test]
    fn forward_deterministic() {
        let mut rng1 = LcgRng::new(7);
        let mut rng2 = LcgRng::new(7);
        let l1 = AdaloraLinear::new(4, 4, &cfg(3, 6.0, 2), &mut rng1);
        let l2 = AdaloraLinear::new(4, 4, &cfg(3, 6.0, 2), &mut rng2);
        let x = vec![0.1_f32, -0.3, 0.7, 0.2];
        assert_eq!(l1.forward(&x), l2.forward(&x));
    }

    // -----------------------------------------------------------------------
    // Test 5: importance_scores matches |λ_i| · ‖P[:,i]‖₂ · ‖Q[i,:]‖₂
    // -----------------------------------------------------------------------
    #[test]
    fn importance_scores_formula() {
        let mut rng = LcgRng::new(55);
        let mut layer = AdaloraLinear::new(3, 3, &cfg(2, 4.0, 1), &mut rng);

        // P (3×2 row-major): column 0 = [1,0,0] norm=1, column 1 = [0,1,0] norm=1
        layer.p = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        layer.lambda = vec![3.0, 5.0];
        // Q (2×3 row-major): row 0 = [1,0,0] norm=1, row 1 = [0,1,0] norm=1
        layer.q = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

        // importance[0] = |3.0| * 1.0 * 1.0 = 3.0
        // importance[1] = |5.0| * 1.0 * 1.0 = 5.0
        let scores = layer.importance_scores();
        assert_eq!(scores.len(), 2);
        assert!(
            (scores[0] - 3.0).abs() < 1e-5,
            "score[0] expected 3.0, got {}",
            scores[0]
        );
        assert!(
            (scores[1] - 5.0).abs() < 1e-5,
            "score[1] expected 5.0, got {}",
            scores[1]
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: prune_to_target zeroes the rank - target_rank least-important lambdas
    // -----------------------------------------------------------------------
    #[test]
    fn prune_to_target_zeroes_least_important() {
        // rank=3, target=1 → 2 lowest-importance lambdas must become 0
        let mut rng = LcgRng::new(11);
        let mut layer = AdaloraLinear::new(3, 3, &cfg(3, 3.0, 1), &mut rng);

        // Identity-like P and Q so importance_score[i] = |lambda[i]|
        layer.p = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        layer.lambda = vec![10.0, 1.0, 5.0]; // importances: 10, 1, 5
        layer.q = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        layer.prune_to_target();

        // Highest importance (index 0, score=10) must survive
        assert!(
            (layer.lambda[0] - 10.0).abs() < 1e-5,
            "lambda[0] should survive pruning"
        );
        // Two lowest (index 1 score=1, index 2 score=5) must be zeroed
        assert!(
            layer.lambda[1].abs() < 1e-5,
            "lambda[1] (score=1) should be pruned to 0"
        );
        assert!(
            layer.lambda[2].abs() < 1e-5,
            "lambda[2] (score=5) should be pruned to 0"
        );
    }

    // -----------------------------------------------------------------------
    // Test 7: prune_to_target leaves at most target_rank non-zero lambdas
    // -----------------------------------------------------------------------
    #[test]
    fn prune_to_target_effective_rank() {
        let r = 4;
        let target_r = 2;
        let mut rng = LcgRng::new(22);
        let mut layer = AdaloraLinear::new(4, 4, &cfg(r, 4.0, target_r), &mut rng);
        // Assign distinguishable lambda values
        layer.lambda = vec![0.5, 2.0, 0.1, 1.5];

        layer.prune_to_target();

        let nonzero = layer.lambda.iter().filter(|&&v| v != 0.0).count();
        assert!(
            nonzero <= target_r,
            "expected ≤{target_r} non-zero lambdas after pruning, got {nonzero}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 8: forward outputs are finite for random initialisation
    // -----------------------------------------------------------------------
    #[test]
    fn forward_finite_outputs() {
        let mut rng = LcgRng::new(33);
        let layer = AdaloraLinear::new(6, 5, &cfg(4, 8.0, 2), &mut rng);
        let x = vec![0.3_f32, -0.5, 1.2, -0.7, 0.9, 0.1];
        for &v in layer.forward(&x).iter() {
            assert!(v.is_finite(), "output must be finite, got {v}");
        }
    }
}
