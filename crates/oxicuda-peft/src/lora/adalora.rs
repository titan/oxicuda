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
