//! Hetero-associative memory (correlation-matrix key→value mapping).
//!
//! Unlike the auto-associative (Hopfield-style) memory in [`crate::memory::assoc_memory`],
//! where the key and value share a single space (`key == value`), this module maps a KEY
//! space to a *distinct* VALUE space. The two spaces may have different dimensionalities
//! (`key_dim ≠ value_dim` is allowed), producing a rectangular weight matrix.
//!
//! # Storage model
//!
//! The memory holds a weight matrix `W` of shape `value_dim × key_dim` (row-major). Two
//! storage regimes are supported:
//!
//! - **Hebbian** (default): each `store` accumulates the outer product `value ⊗ key`,
//!   i.e. `W[v][k] += value[v] * key[k]`. Recall is `W · key` (length `value_dim`). For
//!   orthonormal keys this recovers the stored value exactly; for correlated keys the
//!   recall is a noisy superposition.
//!
//! - **Ridge pseudo-inverse**: stored `(key, value)` pairs are buffered, and [`finalize`]
//!   rebuilds `W` via the ridge-regularised least-squares solution
//!   `W = V Kᵀ (K Kᵀ + λI)⁻¹` (with `λ = 1e-6`). This recovers exact values for
//!   *non-orthogonal* keys (up to numerical precision).
//!
//! [`finalize`]: HeteroAssociativeMemory::finalize

use crate::error::{HdcError, HdcResult};

/// Ridge regularisation strength used by the pseudo-inverse `finalize` solve.
const RIDGE_LAMBDA: f64 = 1e-6;

/// Configuration for a [`HeteroAssociativeMemory`].
#[derive(Debug, Clone)]
pub struct HeteroAssocConfig {
    /// Dimensionality of the key space (must be ≥ 1).
    pub key_dim: usize,
    /// Dimensionality of the value space (must be ≥ 1).
    pub value_dim: usize,
    /// If `true`, store buffers `(key, value)` pairs and [`finalize`] rebuilds the weight
    /// matrix via ridge least-squares for exact recall of (possibly correlated) keys.
    /// If `false`, only the Hebbian accumulator is used.
    ///
    /// [`finalize`]: HeteroAssociativeMemory::finalize
    pub use_pseudo_inverse: bool,
}

impl Default for HeteroAssocConfig {
    fn default() -> Self {
        Self {
            key_dim: 1,
            value_dim: 1,
            use_pseudo_inverse: false,
        }
    }
}

/// Correlation-matrix hetero-associative memory mapping keys to values.
///
/// The weight matrix is stored row-major as `value_dim` rows of `key_dim` columns, so
/// `recall(key)[v] = Σ_k W[v][k] * key[k]`.
pub struct HeteroAssociativeMemory {
    cfg: HeteroAssocConfig,
    /// Weight matrix, `value_dim × key_dim`, row-major.
    weight: Vec<f32>,
    /// Buffered keys (only retained when `use_pseudo_inverse`), each `key_dim` long.
    stored_keys: Vec<Vec<f32>>,
    /// Buffered values (only retained when `use_pseudo_inverse`), each `value_dim` long.
    stored_values: Vec<Vec<f32>>,
    /// Number of `(key, value)` pairs stored so far.
    n_stored: usize,
}

impl HeteroAssociativeMemory {
    /// Create a new, empty hetero-associative memory.
    ///
    /// # Errors
    ///
    /// Returns [`HdcError::ZeroDimension`] if `key_dim == 0` or `value_dim == 0`.
    pub fn new(cfg: HeteroAssocConfig) -> HdcResult<Self> {
        if cfg.key_dim == 0 || cfg.value_dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        let weight = vec![0f32; cfg.value_dim * cfg.key_dim];
        Ok(Self {
            cfg,
            weight,
            stored_keys: Vec::new(),
            stored_values: Vec::new(),
            n_stored: 0,
        })
    }

    /// Store a single `(key, value)` pair.
    ///
    /// In Hebbian mode this accumulates the outer product `W += value ⊗ key`. In
    /// pseudo-inverse mode the pair is also buffered for the later [`finalize`] solve.
    ///
    /// # Errors
    ///
    /// - [`HdcError::DimensionMismatch`] if `key.len() != key_dim` or `value.len() != value_dim`.
    ///
    /// [`finalize`]: HeteroAssociativeMemory::finalize
    pub fn store(&mut self, key: &[f32], value: &[f32]) -> HdcResult<()> {
        self.check_key_len(key)?;
        self.check_value_len(value)?;
        self.accumulate_hebbian(key, value);
        if self.cfg.use_pseudo_inverse {
            self.stored_keys.push(key.to_vec());
            self.stored_values.push(value.to_vec());
        }
        self.n_stored += 1;
        Ok(())
    }

    /// Store `n` `(key, value)` pairs supplied as flat row-major slices.
    ///
    /// `keys` must hold `n * key_dim` elements (row `i` is `keys[i*key_dim .. (i+1)*key_dim]`)
    /// and `values` must hold `n * value_dim` elements. The effect is identical to calling
    /// [`store`] once per row, in order.
    ///
    /// # Errors
    ///
    /// - [`HdcError::DimensionMismatch`] if `keys.len() != n * key_dim` or
    ///   `values.len() != n * value_dim`.
    ///
    /// [`store`]: HeteroAssociativeMemory::store
    pub fn store_batch(&mut self, keys: &[f32], values: &[f32], n: usize) -> HdcResult<()> {
        let expected_keys = n.saturating_mul(self.cfg.key_dim);
        if keys.len() != expected_keys {
            return Err(HdcError::DimensionMismatch {
                expected: expected_keys,
                got: keys.len(),
            });
        }
        let expected_values = n.saturating_mul(self.cfg.value_dim);
        if values.len() != expected_values {
            return Err(HdcError::DimensionMismatch {
                expected: expected_values,
                got: values.len(),
            });
        }
        for i in 0..n {
            let key = &keys[i * self.cfg.key_dim..(i + 1) * self.cfg.key_dim];
            let value = &values[i * self.cfg.value_dim..(i + 1) * self.cfg.value_dim];
            self.store(key, value)?;
        }
        Ok(())
    }

    /// Recall the value associated with `key` as `W · key` (length `value_dim`).
    ///
    /// # Errors
    ///
    /// - [`HdcError::DimensionMismatch`] if `key.len() != key_dim`.
    pub fn recall(&self, key: &[f32]) -> HdcResult<Vec<f32>> {
        self.check_key_len(key)?;
        let mut out = vec![0f32; self.cfg.value_dim];
        for (v, slot) in out.iter_mut().enumerate() {
            let row = &self.weight[v * self.cfg.key_dim..(v + 1) * self.cfg.key_dim];
            let mut acc = 0f64;
            for (&w, &k) in row.iter().zip(key.iter()) {
                acc += (w as f64) * (k as f64);
            }
            *slot = acc as f32;
        }
        Ok(out)
    }

    /// Recall then return the index of the codebook row most similar (cosine) to `W · key`.
    ///
    /// The `codebook` is a flat row-major matrix of `n_codes` rows, each `value_dim` long;
    /// row `i` is `codebook[i*value_dim .. (i+1)*value_dim]`. Ties (equal cosine) are broken
    /// in favour of the lower index.
    ///
    /// # Errors
    ///
    /// - [`HdcError::DimensionMismatch`] if `key.len() != key_dim` or
    ///   `codebook.len() != n_codes * value_dim`.
    /// - [`HdcError::EmptyInput`] if `n_codes == 0`.
    pub fn recall_cleanup(
        &self,
        key: &[f32],
        codebook: &[f32],
        n_codes: usize,
    ) -> HdcResult<usize> {
        if n_codes == 0 {
            return Err(HdcError::EmptyInput);
        }
        let expected = n_codes.saturating_mul(self.cfg.value_dim);
        if codebook.len() != expected {
            return Err(HdcError::DimensionMismatch {
                expected,
                got: codebook.len(),
            });
        }
        let recalled = self.recall(key)?;
        let recalled_norm = l2_norm(&recalled);

        let mut best_idx = 0usize;
        let mut best_sim = f64::NEG_INFINITY;
        for i in 0..n_codes {
            let row = &codebook[i * self.cfg.value_dim..(i + 1) * self.cfg.value_dim];
            let sim = cosine_sim(&recalled, recalled_norm, row);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }
        Ok(best_idx)
    }

    /// Number of `(key, value)` pairs stored so far.
    #[must_use]
    pub fn n_stored(&self) -> usize {
        self.n_stored
    }

    /// Configuration this memory was built with.
    #[must_use]
    pub fn config(&self) -> &HeteroAssocConfig {
        &self.cfg
    }

    /// Rebuild the weight matrix for exact recall of buffered keys (pseudo-inverse mode).
    ///
    /// Solves the ridge-regularised least-squares problem `W = V Kᵀ (K Kᵀ + λI)⁻¹`, where
    /// `K` is `key_dim × n` (its columns are the stored keys) and `V` is `value_dim × n`
    /// (its columns are the stored values), with `λ = 1e-6`. The Gram system
    /// `(K Kᵀ + λI) Wᵀ = (V Kᵀ)ᵀ` is solved by partial-pivot Gaussian elimination.
    ///
    /// In Hebbian mode (`use_pseudo_inverse == false`) this is a no-op (the Hebbian
    /// accumulator is left untouched). With no stored pairs it is also a no-op.
    ///
    /// # Errors
    ///
    /// - [`HdcError::DivisionByZero`] if the Gram matrix is numerically singular even after
    ///   ridge regularisation (degenerate / all-zero keys).
    pub fn finalize(&mut self) -> HdcResult<()> {
        if !self.cfg.use_pseudo_inverse || self.stored_keys.is_empty() {
            return Ok(());
        }
        let key_dim = self.cfg.key_dim;
        let value_dim = self.cfg.value_dim;
        let n = self.stored_keys.len();

        // Gram matrix G = K Kᵀ + λI  (key_dim × key_dim, symmetric).
        // G[a][b] = Σ_p key_p[a] * key_p[b]  (+ λ on the diagonal).
        let mut gram = vec![0f64; key_dim * key_dim];
        for key in &self.stored_keys {
            for a in 0..key_dim {
                let ka = key[a] as f64;
                let row = &mut gram[a * key_dim..(a + 1) * key_dim];
                for (b, slot) in row.iter_mut().enumerate() {
                    *slot += ka * (key[b] as f64);
                }
            }
        }
        for a in 0..key_dim {
            gram[a * key_dim + a] += RIDGE_LAMBDA;
        }

        // Right-hand side B = (V Kᵀ)ᵀ = K Vᵀ, shape key_dim × value_dim (row-major).
        // (V Kᵀ)[v][a] = Σ_p value_p[v] * key_p[a]; we want its transpose so columns map to
        // value rows. B[a][v] = Σ_p key_p[a] * value_p[v].
        let mut rhs = vec![0f64; key_dim * value_dim];
        for p in 0..n {
            let key = &self.stored_keys[p];
            let value = &self.stored_values[p];
            for a in 0..key_dim {
                let ka = key[a] as f64;
                let row = &mut rhs[a * value_dim..(a + 1) * value_dim];
                for (v, slot) in row.iter_mut().enumerate() {
                    *slot += ka * (value[v] as f64);
                }
            }
        }

        // Solve G X = B for X (key_dim × value_dim). X[a][v] = Wᵀ[a][v] = W[v][a].
        let solution = solve_linear_system(gram, rhs, key_dim, value_dim)?;

        // Scatter Xᵀ into the row-major weight matrix W (value_dim × key_dim).
        for v in 0..value_dim {
            for a in 0..key_dim {
                self.weight[v * key_dim + a] = solution[a * value_dim + v] as f32;
            }
        }
        Ok(())
    }

    // ── Private helpers ─────────────────────────────────────────────────────

    /// Add the outer product `value ⊗ key` into the weight matrix.
    fn accumulate_hebbian(&mut self, key: &[f32], value: &[f32]) {
        let key_dim = self.cfg.key_dim;
        for (v, &vv) in value.iter().enumerate() {
            let row = &mut self.weight[v * key_dim..(v + 1) * key_dim];
            for (slot, &kv) in row.iter_mut().zip(key.iter()) {
                *slot += vv * kv;
            }
        }
    }

    fn check_key_len(&self, key: &[f32]) -> HdcResult<()> {
        if key.len() != self.cfg.key_dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.cfg.key_dim,
                got: key.len(),
            });
        }
        Ok(())
    }

    fn check_value_len(&self, value: &[f32]) -> HdcResult<()> {
        if value.len() != self.cfg.value_dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.cfg.value_dim,
                got: value.len(),
            });
        }
        Ok(())
    }
}

/// L2 norm of an `f32` slice, computed in `f64` for stability.
fn l2_norm(v: &[f32]) -> f64 {
    v.iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt()
}

/// Cosine similarity between a recalled vector (with pre-computed norm) and a codebook row.
///
/// Returns `f64::NEG_INFINITY` when either vector has (near-)zero norm so that such
/// degenerate rows never win the argmax over well-defined candidates.
fn cosine_sim(recalled: &[f32], recalled_norm: f64, row: &[f32]) -> f64 {
    let row_norm = l2_norm(row);
    let denom = recalled_norm * row_norm;
    if denom < f64::EPSILON {
        return f64::NEG_INFINITY;
    }
    let dot: f64 = recalled
        .iter()
        .zip(row.iter())
        .map(|(&a, &b)| (a as f64) * (b as f64))
        .sum();
    dot / denom
}

/// Solve the linear system `A X = B` for `X`, where `A` is `dim × dim` (row-major) and `B`
/// is `dim × n_rhs` (row-major), via Gaussian elimination with partial pivoting.
///
/// `A` and `B` are consumed (taken by value) and used as scratch space. Returns `X` as a
/// `dim × n_rhs` row-major matrix.
///
/// # Errors
///
/// - [`HdcError::DivisionByZero`] if `A` is numerically singular (a pivot is ~0).
fn solve_linear_system(
    mut a: Vec<f64>,
    mut b: Vec<f64>,
    dim: usize,
    n_rhs: usize,
) -> HdcResult<Vec<f64>> {
    // Forward elimination with partial pivoting.
    for col in 0..dim {
        // Find the pivot row (max abs value in this column at or below the diagonal).
        let mut pivot_row = col;
        let mut pivot_abs = a[col * dim + col].abs();
        for r in (col + 1)..dim {
            let val = a[r * dim + col].abs();
            if val > pivot_abs {
                pivot_abs = val;
                pivot_row = r;
            }
        }
        if pivot_abs < f64::EPSILON {
            return Err(HdcError::DivisionByZero);
        }
        // Swap pivot row into place (in both A and B).
        if pivot_row != col {
            for c in 0..dim {
                a.swap(pivot_row * dim + c, col * dim + c);
            }
            for c in 0..n_rhs {
                b.swap(pivot_row * n_rhs + c, col * n_rhs + c);
            }
        }
        let pivot = a[col * dim + col];
        // Eliminate this column from all rows below.
        for r in (col + 1)..dim {
            let factor = a[r * dim + col] / pivot;
            if factor == 0.0 {
                continue;
            }
            for c in col..dim {
                let sub = factor * a[col * dim + c];
                a[r * dim + c] -= sub;
            }
            for c in 0..n_rhs {
                let sub = factor * b[col * n_rhs + c];
                b[r * n_rhs + c] -= sub;
            }
        }
    }

    // Back substitution.
    let mut x = vec![0f64; dim * n_rhs];
    for col in 0..n_rhs {
        for r in (0..dim).rev() {
            let mut acc = b[r * n_rhs + col];
            for c in (r + 1)..dim {
                acc -= a[r * dim + c] * x[c * n_rhs + col];
            }
            let pivot = a[r * dim + r];
            if pivot.abs() < f64::EPSILON {
                return Err(HdcError::DivisionByZero);
            }
            x[r * n_rhs + col] = acc / pivot;
        }
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Build a deterministic Gaussian vector via the LCG's Box-Muller sampler.
    fn gaussian_vec(dim: usize, rng: &mut LcgRng) -> Vec<f32> {
        let mut v = Vec::with_capacity(dim);
        while v.len() < dim {
            let (a, b) = rng.normal_pair_f32();
            v.push(a);
            if v.len() < dim {
                v.push(b);
            }
        }
        v
    }

    fn cfg(key_dim: usize, value_dim: usize, pinv: bool) -> HeteroAssocConfig {
        HeteroAssocConfig {
            key_dim,
            value_dim,
            use_pseudo_inverse: pinv,
        }
    }

    #[test]
    fn store_one_recall_proportional_to_value() {
        // Hebbian: store(key, value); recall(key) = (key·key) * value (exact direction).
        let key = vec![1.0f32, 0.0, 0.0, 0.0];
        let value = vec![2.0f32, -3.0, 1.0];
        let mut mem = HeteroAssociativeMemory::new(cfg(4, 3, false)).unwrap();
        mem.store(&key, &value).unwrap();
        let recalled = mem.recall(&key).unwrap();
        // key·key = 1, so recall == value exactly here.
        for (r, v) in recalled.iter().zip(value.iter()) {
            assert!((r - v).abs() < 1e-5, "r={r} v={v}");
        }
    }

    #[test]
    fn hebbian_recall_scales_with_key_norm_squared() {
        // Non-unit key: recall = ||key||^2 * value (direction preserved, scaled).
        let key = vec![2.0f32, 0.0]; // ||key||^2 = 4
        let value = vec![1.0f32, -1.0, 0.5];
        let mut mem = HeteroAssociativeMemory::new(cfg(2, 3, false)).unwrap();
        mem.store(&key, &value).unwrap();
        let recalled = mem.recall(&key).unwrap();
        for (r, v) in recalled.iter().zip(value.iter()) {
            assert!((r - 4.0 * v).abs() < 1e-4, "r={r} expected={}", 4.0 * v);
        }
    }

    #[test]
    fn orthonormal_keys_recall_exact_hebbian() {
        // Orthonormal keys (standard basis) → Hebbian recall is exact.
        let mut mem = HeteroAssociativeMemory::new(cfg(3, 2, false)).unwrap();
        let keys = [
            vec![1.0f32, 0.0, 0.0],
            vec![0.0f32, 1.0, 0.0],
            vec![0.0f32, 0.0, 1.0],
        ];
        let values = [vec![5.0f32, -2.0], vec![-1.0f32, 3.0], vec![0.5f32, 0.5]];
        for (k, v) in keys.iter().zip(values.iter()) {
            mem.store(k, v).unwrap();
        }
        for (k, v) in keys.iter().zip(values.iter()) {
            let recalled = mem.recall(k).unwrap();
            for (r, e) in recalled.iter().zip(v.iter()) {
                assert!((r - e).abs() < 1e-4, "recall {r} != {e}");
            }
        }
    }

    #[test]
    fn orthonormal_rotated_keys_recall_exact_hebbian() {
        // A rotated orthonormal basis is still orthonormal → Hebbian recall exact.
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let mut mem = HeteroAssociativeMemory::new(cfg(2, 2, false)).unwrap();
        let keys = [vec![s, s], vec![s, -s]];
        let values = [vec![3.0f32, 7.0], vec![-4.0f32, 1.0]];
        for (k, v) in keys.iter().zip(values.iter()) {
            mem.store(k, v).unwrap();
        }
        for (k, v) in keys.iter().zip(values.iter()) {
            let recalled = mem.recall(k).unwrap();
            for (r, e) in recalled.iter().zip(v.iter()) {
                assert!((r - e).abs() < 1e-4, "recall {r} != {e}");
            }
        }
    }

    #[test]
    fn pseudo_inverse_exact_recall_nonorthogonal_keys() {
        // Correlated (non-orthogonal) keys: Hebbian crosstalk would corrupt recall, but
        // the ridge pseudo-inverse recovers exact values.
        let mut mem = HeteroAssociativeMemory::new(cfg(3, 2, true)).unwrap();
        let keys = [
            vec![1.0f32, 0.5, 0.0],
            vec![0.5f32, 1.0, 0.5],
            vec![0.0f32, 0.5, 1.0],
        ];
        let values = [vec![1.0f32, 2.0], vec![3.0f32, -1.0], vec![-2.0f32, 0.5]];
        for (k, v) in keys.iter().zip(values.iter()) {
            mem.store(k, v).unwrap();
        }
        mem.finalize().unwrap();
        for (k, v) in keys.iter().zip(values.iter()) {
            let recalled = mem.recall(k).unwrap();
            for (r, e) in recalled.iter().zip(v.iter()) {
                assert!((r - e).abs() < 1e-3, "pinv recall {r} != {e}");
            }
        }
    }

    #[test]
    fn pseudo_inverse_random_keys_exact_recall() {
        // Random (generically non-orthogonal) keys; pseudo-inverse still exact.
        let mut rng = LcgRng::new(0xA11CE);
        let key_dim = 8;
        let value_dim = 5;
        let n = 6; // fewer pairs than key_dim → well-conditioned
        let mut mem = HeteroAssociativeMemory::new(cfg(key_dim, value_dim, true)).unwrap();
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for _ in 0..n {
            let k = gaussian_vec(key_dim, &mut rng);
            let v = gaussian_vec(value_dim, &mut rng);
            mem.store(&k, &v).unwrap();
            keys.push(k);
            values.push(v);
        }
        mem.finalize().unwrap();
        for (k, v) in keys.iter().zip(values.iter()) {
            let recalled = mem.recall(k).unwrap();
            for (r, e) in recalled.iter().zip(v.iter()) {
                assert!((r - e).abs() < 1e-3, "recall {r} != {e}");
            }
        }
    }

    #[test]
    fn recall_cleanup_picks_correct_codebook_index() {
        // Store orthonormal-keyed pairs, then clean up the recall against a codebook.
        let mut mem = HeteroAssociativeMemory::new(cfg(3, 3, false)).unwrap();
        let keys = [
            vec![1.0f32, 0.0, 0.0],
            vec![0.0f32, 1.0, 0.0],
            vec![0.0f32, 0.0, 1.0],
        ];
        let values = [
            vec![1.0f32, 0.0, 0.0],
            vec![0.0f32, 1.0, 0.0],
            vec![0.0f32, 0.0, 1.0],
        ];
        for (k, v) in keys.iter().zip(values.iter()) {
            mem.store(k, v).unwrap();
        }
        // Codebook = the three value prototypes flattened.
        let codebook: Vec<f32> = values.iter().flatten().copied().collect();
        for (i, k) in keys.iter().enumerate() {
            let idx = mem.recall_cleanup(k, &codebook, 3).unwrap();
            assert_eq!(idx, i, "cleanup picked {idx}, expected {i}");
        }
    }

    #[test]
    fn recall_cleanup_robust_to_scaling() {
        // Cosine cleanup is scale-invariant: a scaled codebook still matches.
        let mut mem = HeteroAssociativeMemory::new(cfg(2, 2, false)).unwrap();
        mem.store(&[1.0, 0.0], &[1.0, 0.0]).unwrap();
        mem.store(&[0.0, 1.0], &[0.0, 1.0]).unwrap();
        // Codebook scaled by 10 — cosine ignores magnitude.
        let codebook = vec![10.0f32, 0.0, 0.0, 10.0];
        assert_eq!(mem.recall_cleanup(&[1.0, 0.0], &codebook, 2).unwrap(), 0);
        assert_eq!(mem.recall_cleanup(&[0.0, 1.0], &codebook, 2).unwrap(), 1);
    }

    #[test]
    fn recall_length_equals_value_dim() {
        let mem = HeteroAssociativeMemory::new(cfg(7, 4, false)).unwrap();
        let recalled = mem.recall(&[0.0; 7]).unwrap();
        assert_eq!(recalled.len(), 4);
    }

    #[test]
    fn rectangular_weight_key_ne_value() {
        // key_dim != value_dim: rectangular W, recall maps 5→3.
        let mut mem = HeteroAssociativeMemory::new(cfg(5, 3, false)).unwrap();
        let key = vec![1.0f32, 0.0, 0.0, 0.0, 0.0];
        let value = vec![9.0f32, 8.0, 7.0];
        mem.store(&key, &value).unwrap();
        let recalled = mem.recall(&key).unwrap();
        assert_eq!(recalled.len(), 3);
        for (r, v) in recalled.iter().zip(value.iter()) {
            assert!((r - v).abs() < 1e-5);
        }
    }

    #[test]
    fn n_stored_counts_pairs() {
        let mut mem = HeteroAssociativeMemory::new(cfg(2, 2, false)).unwrap();
        assert_eq!(mem.n_stored(), 0);
        mem.store(&[1.0, 0.0], &[1.0, 0.0]).unwrap();
        mem.store(&[0.0, 1.0], &[0.0, 1.0]).unwrap();
        assert_eq!(mem.n_stored(), 2);
    }

    #[test]
    fn store_batch_equals_repeated_store() {
        let key_dim = 3;
        let value_dim = 2;
        let keys = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let values = vec![5.0f32, 6.0, 7.0, 8.0];

        let mut batched = HeteroAssociativeMemory::new(cfg(key_dim, value_dim, false)).unwrap();
        batched.store_batch(&keys, &values, 2).unwrap();

        let mut single = HeteroAssociativeMemory::new(cfg(key_dim, value_dim, false)).unwrap();
        single.store(&keys[0..3], &values[0..2]).unwrap();
        single.store(&keys[3..6], &values[2..4]).unwrap();

        assert_eq!(batched.n_stored(), single.n_stored());
        let probe = vec![0.3f32, -0.7, 1.1];
        let rb = batched.recall(&probe).unwrap();
        let rs = single.recall(&probe).unwrap();
        for (a, b) in rb.iter().zip(rs.iter()) {
            assert!((a - b).abs() < 1e-5, "batch {a} != single {b}");
        }
    }

    #[test]
    fn capacity_degrades_gracefully_cleanup_still_correct() {
        // Store many random pairs into a Hebbian memory; for a *small* well-separated
        // probe set, cleanup against the stored codebook still resolves correctly.
        let mut rng = LcgRng::new(0xC0FFEE);
        let key_dim = 256;
        let value_dim = 64;
        let n = 5;
        let mut mem = HeteroAssociativeMemory::new(cfg(key_dim, value_dim, false)).unwrap();
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for _ in 0..n {
            let k = gaussian_vec(key_dim, &mut rng);
            let v = gaussian_vec(value_dim, &mut rng);
            mem.store(&k, &v).unwrap();
            keys.push(k);
            values.push(v);
        }
        let codebook: Vec<f32> = values.iter().flatten().copied().collect();
        for (i, k) in keys.iter().enumerate() {
            let idx = mem.recall_cleanup(k, &codebook, n).unwrap();
            assert_eq!(idx, i, "high-D cleanup picked {idx}, expected {i}");
        }
    }

    #[test]
    fn deterministic_recall() {
        let build = || {
            let mut m = HeteroAssociativeMemory::new(cfg(4, 3, false)).unwrap();
            m.store(&[1.0, 2.0, 3.0, 4.0], &[0.1, 0.2, 0.3]).unwrap();
            m.store(&[4.0, 3.0, 2.0, 1.0], &[0.3, 0.2, 0.1]).unwrap();
            m
        };
        let a = build();
        let b = build();
        let ra = a.recall(&[1.0, 1.0, 1.0, 1.0]).unwrap();
        let rb = b.recall(&[1.0, 1.0, 1.0, 1.0]).unwrap();
        assert_eq!(ra, rb);
    }

    #[test]
    fn err_key_wrong_length() {
        let mut mem = HeteroAssociativeMemory::new(cfg(4, 2, false)).unwrap();
        let res = mem.store(&[1.0, 0.0, 0.0], &[1.0, 0.0]);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
        let res2 = mem.recall(&[1.0, 0.0]);
        assert!(matches!(res2, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_value_wrong_length() {
        let mut mem = HeteroAssociativeMemory::new(cfg(2, 4, false)).unwrap();
        let res = mem.store(&[1.0, 0.0], &[1.0, 0.0, 0.0]);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_codebook_length_mismatch() {
        let mem = HeteroAssociativeMemory::new(cfg(2, 3, false)).unwrap();
        // n_codes=2 expects 2*3=6 elems; supply 5.
        let res = mem.recall_cleanup(&[1.0, 0.0], &[0.0; 5], 2);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_codebook_empty() {
        let mem = HeteroAssociativeMemory::new(cfg(2, 3, false)).unwrap();
        let res = mem.recall_cleanup(&[1.0, 0.0], &[], 0);
        assert!(matches!(res, Err(HdcError::EmptyInput)));
    }

    #[test]
    fn err_key_dim_zero() {
        let res = HeteroAssociativeMemory::new(cfg(0, 4, false));
        assert!(matches!(res, Err(HdcError::ZeroDimension)));
    }

    #[test]
    fn err_value_dim_zero() {
        let res = HeteroAssociativeMemory::new(cfg(4, 0, false));
        assert!(matches!(res, Err(HdcError::ZeroDimension)));
    }

    #[test]
    fn err_batch_keys_length_mismatch() {
        let mut mem = HeteroAssociativeMemory::new(cfg(3, 2, false)).unwrap();
        // n=2 expects keys.len()=6; supply 5.
        let res = mem.store_batch(&[0.0; 5], &[0.0; 4], 2);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_batch_values_length_mismatch() {
        let mut mem = HeteroAssociativeMemory::new(cfg(3, 2, false)).unwrap();
        // n=2 expects values.len()=4; supply 3.
        let res = mem.store_batch(&[0.0; 6], &[0.0; 3], 2);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn empty_memory_recall_is_zeros() {
        let mem = HeteroAssociativeMemory::new(cfg(5, 3, false)).unwrap();
        let recalled = mem.recall(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(recalled, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn cosine_cleanup_tie_breaks_to_lower_index() {
        // Empty memory → recall is zeros → all cosines are NEG_INFINITY (tie); the
        // argmax keeps the first index (lower index wins).
        let mem = HeteroAssociativeMemory::new(cfg(2, 2, false)).unwrap();
        let codebook = vec![1.0f32, 0.0, 0.0, 1.0];
        let idx = mem.recall_cleanup(&[0.0, 0.0], &codebook, 2).unwrap();
        assert_eq!(idx, 0);
    }

    #[test]
    fn finalize_hebbian_mode_is_noop() {
        // use_pseudo_inverse=false: finalize must not alter the Hebbian weights.
        let mut mem = HeteroAssociativeMemory::new(cfg(2, 2, false)).unwrap();
        mem.store(&[1.0, 0.0], &[3.0, 4.0]).unwrap();
        let before = mem.recall(&[1.0, 0.0]).unwrap();
        mem.finalize().unwrap();
        let after = mem.recall(&[1.0, 0.0]).unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn finalize_no_stored_pairs_is_noop() {
        // Pseudo-inverse mode but nothing stored → finalize is a clean no-op.
        let mut mem = HeteroAssociativeMemory::new(cfg(3, 2, true)).unwrap();
        mem.finalize().unwrap();
        let recalled = mem.recall(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(recalled, vec![0.0, 0.0]);
    }

    #[test]
    fn pseudo_inverse_single_pair_exact() {
        // Single pair, pseudo-inverse: recall(key) recovers value exactly.
        let mut mem = HeteroAssociativeMemory::new(cfg(3, 2, true)).unwrap();
        mem.store(&[2.0, 1.0, 0.0], &[5.0, -3.0]).unwrap();
        mem.finalize().unwrap();
        let recalled = mem.recall(&[2.0, 1.0, 0.0]).unwrap();
        assert!((recalled[0] - 5.0).abs() < 1e-3, "got {}", recalled[0]);
        assert!((recalled[1] + 3.0).abs() < 1e-3, "got {}", recalled[1]);
    }

    #[test]
    fn solve_linear_system_identity() {
        // Sanity check on the inline solver: I·X = B ⇒ X = B.
        let a = vec![1.0, 0.0, 0.0, 1.0]; // 2×2 identity
        let b = vec![3.0, 4.0, 5.0, 6.0]; // 2×2 rhs
        let x = solve_linear_system(a, b.clone(), 2, 2).unwrap();
        assert_eq!(x, b);
    }

    #[test]
    fn solve_linear_system_singular_errors() {
        // Singular matrix → DivisionByZero.
        let a = vec![1.0, 1.0, 1.0, 1.0]; // rank 1
        let b = vec![1.0, 2.0];
        let res = solve_linear_system(a, b, 2, 1);
        assert!(matches!(res, Err(HdcError::DivisionByZero)));
    }
}
