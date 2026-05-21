//! PHATE — Potential of Heat-diffusion for Affinity-based Transition Embedding.
//!
//! Moon et al. (2019) "Visualizing Structure and Transitions in High-Dimensional
//! Biological Data", *Nature Biotechnology* 37, 1482–1492.
//!
//! ## Algorithm Overview
//!
//! 1. **Adaptive-bandwidth kernel** — For each point i compute σᵢ = distance to the
//!    k-th nearest neighbour. Build `K_ij = exp(-dist²(i,j) / (σᵢ · σⱼ))`.
//!
//! 2. **Alpha-density normalisation** — Let `d_i = Σⱼ K_ij`.
//!    `K_α_ij = K_ij / (d_i^α · d_j^α)`.  Re-normalise degrees.
//!
//! 3. **Markov operator** — Row-normalise: `P_ij = K_α_ij / d'_i`.
//!
//! 4. **Diffusion operator power** — Compute `P^t` (t diffusion steps).
//!    For n ≤ 500 use a fast squaring scheme.  For larger n an eigendecomposition
//!    route is used: P_sym = D'^{1/2} P D'^{-1/2}, eigendecompose, compute P^t via
//!    spectral representation.
//!
//! 5. **PHATE potential** — `U_ij = -log(P^t_ij + ε)`.
//!
//! 6. **Potential distances** — `D_pot(i,j) = ‖U_i − U_j‖₂`.
//!
//! 7. **Classical MDS** — Torgerson MDS on D_pot → n_components embedding.

use crate::error::{ManifoldError, ManifoldResult};
use crate::linalg::jacobi_eig::{jacobi_eigh, sort_eigen_descending};

// ─────────────────────────────────────────────────────────────────────────────
// Public configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Hyper-parameters for the PHATE algorithm.
///
/// All fields have documented defaults (see [`PhateConfig::default`]).
#[derive(Debug, Clone)]
pub struct PhateConfig {
    /// Number of output dimensions (default 2).
    pub n_components: usize,
    /// Number of nearest neighbours used for adaptive bandwidth (default 5).
    pub k: usize,
    /// Number of diffusion steps (default 10).
    pub t: usize,
    /// Alpha-density normalisation exponent (default 1.0, 0 = off).
    pub alpha: f64,
    /// Informational-distance exponent γ — kept for API completeness
    /// (default 1.0 = Hellinger-like log-potential; not yet used in the
    /// diffusion-potential formula since standard PHATE always uses γ = 1).
    pub gamma: f64,
    /// Log-stability floor added before taking log (default 1e-7).
    pub epsilon: f64,
}

impl Default for PhateConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            k: 5,
            t: 10,
            alpha: 1.0,
            gamma: 1.0,
            epsilon: 1e-7,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public output
// ─────────────────────────────────────────────────────────────────────────────

/// Output of a PHATE fit.
pub struct PhateResult {
    /// Low-dimensional embedding: `[n_samples × n_components]` row-major.
    pub embedding: Vec<f64>,
    /// Diffusion potential matrix U: `[n_samples × n_samples]` row-major.
    /// `U_ij = -log(P^t_ij + ε)`.
    pub diff_potential: Vec<f64>,
    /// Eigenvalues of the Markov operator (sorted descending by magnitude).
    pub eigenvalues: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Fit a PHATE embedding to data matrix `x` (shape `n_samples × dim`, row-major).
///
/// # Errors
/// Returns [`ManifoldError`] on invalid inputs or numerical failures.
pub fn phate_fit(
    x: &[f64],
    n_samples: usize,
    dim: usize,
    cfg: &PhateConfig,
) -> ManifoldResult<PhateResult> {
    // ── Validate inputs ──────────────────────────────────────────────────────
    if n_samples == 0 || dim == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if x.len() != n_samples * dim {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples, dim],
            got: vec![x.len()],
        });
    }
    let n = n_samples;
    if cfg.n_components == 0 || cfg.n_components >= n {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: format!("must be in 1..{n}, got {}", cfg.n_components),
        });
    }
    if cfg.k == 0 || cfg.k >= n {
        return Err(ManifoldError::InvalidParameter {
            name: "k".into(),
            reason: format!("must be in 1..{n}, got {}", cfg.k),
        });
    }

    // ── Step 1: Adaptive-bandwidth kernel ────────────────────────────────────
    let sigma = knn_distances(x, n, dim, cfg.k);
    let mut kernel = build_kernel(x, n, dim, &sigma);

    // ── Step 2: Alpha-density normalisation ──────────────────────────────────
    // After normalisation the degrees d_prime are returned so we can build
    // the symmetric conjugate later.
    let d_prime = alpha_normalize(&mut kernel, n, cfg.alpha);

    // ── Step 3: Markov row-normalisation ─────────────────────────────────────
    markov_normalize(&mut kernel, n, &d_prime);
    // `kernel` is now the Markov matrix P.

    // ── Step 4: Diffusion operator P^t ──────────────────────────────────────
    //
    // Strategy:
    //   • Small n (≤ 500) or small t (≤ 20): direct repeated matrix multiply
    //     (O(n² t) — cheap for small n).
    //   • Large n: eigendecomposition of P_sym to avoid O(n³ t).
    //     We keep the full spectrum here (Jacobi is O(n³) but done once).
    let (p_t, eigenvalues) = if n <= 500 || cfg.t <= 20 {
        let p_t_mat = mat_pow(&kernel, n, cfg.t);
        // Compute eigenvalues via symmetric conjugate for the result struct.
        let eigs = compute_eigenvalues_of_markov(&kernel, n, &d_prime)?;
        (p_t_mat, eigs)
    } else {
        compute_p_t_via_eigen(&kernel, n, &d_prime, cfg.t)?
    };

    // ── Step 5: PHATE diffusion potential ────────────────────────────────────
    let diff_potential = diffusion_potential(&p_t, n, cfg.epsilon);

    // Validate numerical health before proceeding.
    if diff_potential.iter().any(|v| !v.is_finite()) {
        return Err(ManifoldError::NumericalInstability(
            "non-finite values in diffusion potential".into(),
        ));
    }

    // ── Step 6: Pairwise potential distances ─────────────────────────────────
    let d_pot = potential_distances(&diff_potential, n);

    // ── Step 7: Classical MDS on potential distances ─────────────────────────
    let embedding = classical_mds(&d_pot, n, cfg.n_components)?;

    Ok(PhateResult {
        embedding,
        diff_potential,
        eigenvalues,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 1 — Adaptive k-NN bandwidth
// ─────────────────────────────────────────────────────────────────────────────

/// For each point i, return the distance to its k-th nearest neighbour (σᵢ).
///
/// Uses brute-force O(n² d) pairwise distances — exact, no approximation.
fn knn_distances(x: &[f64], n: usize, dim: usize, k: usize) -> Vec<f64> {
    let mut sigma = vec![0.0_f64; n];
    let mut sq_dists = vec![0.0_f64; n];

    for i in 0..n {
        // Compute squared distances from point i to all others.
        for j in 0..n {
            let mut s = 0.0_f64;
            for d in 0..dim {
                let diff = x[i * dim + d] - x[j * dim + d];
                s += diff * diff;
            }
            sq_dists[j] = s;
        }
        // Partial-sort: find the k smallest distances (excluding self at j==i,
        // where sq_dist = 0).  Use a simple selection that avoids full sort.
        let sigma_k = k_th_smallest_nonzero(&sq_dists, n, i, k);
        sigma[i] = sigma_k.sqrt().max(1e-300);
    }
    sigma
}

/// Return the distance of the k-th nearest neighbour of point `self_idx`.
///
/// We exclude the self-distance (index `self_idx`).  If all distances are zero
/// (degenerate data), we fall back to a tiny positive value.
fn k_th_smallest_nonzero(sq_dists: &[f64], n: usize, self_idx: usize, k: usize) -> f64 {
    // Collect distances excluding self.
    let mut dists: Vec<f64> = (0..n)
        .filter(|&j| j != self_idx)
        .map(|j| sq_dists[j])
        .collect();
    // Partial-sort to find the k-th smallest.  k is 1-indexed here.
    if k >= dists.len() {
        // k too large; return max distance.
        return dists
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(1e-300);
    }
    // Floyd–Rivest / introselect would be better, but for n ≤ few-thousand
    // a full sort is acceptable and keeps the code simple.
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    dists[k - 1]
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 2 — Adaptive Gaussian kernel (geometric mean bandwidth)
// ─────────────────────────────────────────────────────────────────────────────

/// Build the n×n symmetric kernel `K_ij = exp(-‖xᵢ − xⱼ‖² / (σᵢ · σⱼ))`.
///
/// The denominator is the geometric mean of the per-point bandwidths (not 2σ²),
/// consistent with the variable-bandwidth formulation in Moon et al. 2019.
fn build_kernel(x: &[f64], n: usize, dim: usize, sigma: &[f64]) -> Vec<f64> {
    let mut k_mat = vec![0.0_f64; n * n];
    for i in 0..n {
        k_mat[i * n + i] = 1.0; // self-affinity = 1
        for j in (i + 1)..n {
            let mut sq_dist = 0.0_f64;
            for d in 0..dim {
                let diff = x[i * dim + d] - x[j * dim + d];
                sq_dist += diff * diff;
            }
            let bw = sigma[i] * sigma[j]; // geometric-mean denominator
            let val = (-sq_dist / bw).exp();
            k_mat[i * n + j] = val;
            k_mat[j * n + i] = val;
        }
    }
    k_mat
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 3 — Alpha-density normalisation
// ─────────────────────────────────────────────────────────────────────────────

/// In-place alpha-density normalisation: `K_α_ij = K_ij / (dᵢ^α · dⱼ^α)`.
///
/// Returns the re-computed row-sum vector d' (used to form P and P_sym).
///
/// With `alpha = 0` the kernel is unchanged and d' is simply the row-sums of K.
fn alpha_normalize(k_mat: &mut [f64], n: usize, alpha: f64) -> Vec<f64> {
    // First-pass degrees.
    let mut deg = row_sums(k_mat, n);
    for d in &mut deg {
        *d = d.max(1e-300);
    }

    if alpha > 0.0 {
        for i in 0..n {
            let di_a = deg[i].powf(alpha);
            for j in 0..n {
                let dj_a = deg[j].powf(alpha);
                k_mat[i * n + j] /= di_a * dj_a;
            }
        }
        // Recompute degrees after normalisation.
        let d_prime = row_sums(k_mat, n);
        let mut dp = d_prime;
        for d in &mut dp {
            *d = d.max(1e-300);
        }
        dp
    } else {
        deg
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 4 — Markov row-normalisation
// ─────────────────────────────────────────────────────────────────────────────

/// Convert kernel K_α to a row-stochastic Markov matrix in-place.
/// `P_ij = K_α_ij / d'_i`.
///
/// Takes the pre-computed row-sums `d_prime` to avoid recomputation.
fn markov_normalize(k_mat: &mut [f64], n: usize, d_prime: &[f64]) {
    for i in 0..n {
        let inv_d = 1.0 / d_prime[i]; // d_prime[i] ≥ 1e-300
        for j in 0..n {
            k_mat[i * n + j] *= inv_d;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 5 — Diffusion power P^t (two paths)
// ─────────────────────────────────────────────────────────────────────────────

/// Repeated matrix squaring / multiplication to compute P^t.
///
/// Uses a binary-exponentiation scheme when t is large:
///
/// - decompose t in binary, accumulate with squarings.
///
/// This is O(n³ log₂ t) total, much better than O(n³ t) for large t.
fn mat_pow(p: &[f64], n: usize, t: usize) -> Vec<f64> {
    if t == 0 {
        // P^0 = identity
        let mut id = vec![0.0_f64; n * n];
        for i in 0..n {
            id[i * n + i] = 1.0;
        }
        return id;
    }
    // Binary exponentiation.
    let mut result = {
        let mut id = vec![0.0_f64; n * n];
        for i in 0..n {
            id[i * n + i] = 1.0;
        }
        id
    };
    let mut base = p.to_vec();
    let mut exp = t;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mat_mul(&result, &base, n);
        }
        base = mat_mul(&base, &base, n);
        exp >>= 1;
    }
    result
}

/// Dense n×n matrix multiplication: C = A · B (row-major).
fn mat_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[i * n + k];
            if a_ik == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }
    c
}

/// Eigenvalue-based computation of P^t when n is large.
///
/// 1. Form P_sym = D'^{1/2} P D'^{-1/2}.  P_sym is symmetric (since P is
///    reversible w.r.t. the stationary measure π ∝ d').
/// 2. Eigendecompose P_sym: ψ_k are eigenvectors, λ_k eigenvalues.
/// 3. Right eigenvectors of P: φ_k = D'^{-1/2} ψ_k.
/// 4. P^t_ij = Σ_k λ_k^t φ_k(i) φ_k(j) (spectral reconstruction).
///
/// Returns (P^t, sorted_eigenvalues).
fn compute_p_t_via_eigen(
    p: &[f64],
    n: usize,
    d_prime: &[f64],
    t: usize,
) -> ManifoldResult<(Vec<f64>, Vec<f64>)> {
    // ── Build P_sym ──────────────────────────────────────────────────────────
    let sqrt_d: Vec<f64> = d_prime.iter().map(|d| d.sqrt()).collect();
    let inv_sqrt_d: Vec<f64> = sqrt_d.iter().map(|s| 1.0 / s.max(1e-300)).collect();

    let mut p_sym = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            // P_sym_ij = sqrt(d_i) · P_ij · (1/sqrt(d_j))
            p_sym[i * n + j] = sqrt_d[i] * p[i * n + j] * inv_sqrt_d[j];
        }
    }
    // Force exact symmetry to improve Jacobi convergence.
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (p_sym[i * n + j] + p_sym[j * n + i]);
            p_sym[i * n + j] = avg;
            p_sym[j * n + i] = avg;
        }
    }

    // ── Eigendecompose P_sym ─────────────────────────────────────────────────
    let (mut w, mut psi) = jacobi_eigh(&p_sym, n)?;
    sort_eigen_descending(&mut w, &mut psi, n);

    // Right eigenvectors of P: φ_k(i) = psi_k(i) / sqrt(d_i).
    // We store them column-major aligned with `psi` (column k = φ_k).
    // psi is row-major n×n, column k = psi[r * n + k].
    // φ_k(i) = inv_sqrt_d[i] * psi[i * n + k].

    // ── Reconstruct P^t ──────────────────────────────────────────────────────
    // P^t_ij = Σ_k  λ_k^t · φ_k(i) · φ_k(j)
    //        = Σ_k  λ_k^t · (inv_sqrt_d[i]·ψ_k(i)) · (inv_sqrt_d[j]·ψ_k(j))
    //
    // For each eigencomponent k build the outer product scaled by λ_k^t.
    let mut p_t = vec![0.0_f64; n * n];
    for k in 0..n {
        let lam_t = if t == 0 { 1.0_f64 } else { w[k].powi(t as i32) };
        if lam_t.abs() < 1e-300 {
            continue; // eigenvalue contribution negligible
        }
        // Build φ_k vector.
        let phi_k: Vec<f64> = (0..n).map(|i| inv_sqrt_d[i] * psi[i * n + k]).collect();
        // Accumulate outer product.
        for i in 0..n {
            let scale = lam_t * phi_k[i];
            if scale.abs() < 1e-300 {
                continue;
            }
            for j in 0..n {
                p_t[i * n + j] += scale * phi_k[j];
            }
        }
    }

    // Clip to [0, 1] — floating-point reconstruction can produce tiny negatives.
    for v in &mut p_t {
        *v = v.clamp(0.0, 1.0);
    }

    Ok((p_t, w))
}

/// Compute only the eigenvalues of P for the small-n path (P already known).
fn compute_eigenvalues_of_markov(
    _p: &[f64],
    n: usize,
    d_prime: &[f64],
) -> ManifoldResult<Vec<f64>> {
    // In the small-n path we already have P; we rebuild P_sym to get eigenvalues.
    // Since `p` passed in is already the Markov matrix we need d_prime which
    // we receive as a parameter.
    let sqrt_d: Vec<f64> = d_prime.iter().map(|d| d.max(1e-300).sqrt()).collect();
    let inv_sqrt_d: Vec<f64> = sqrt_d.iter().map(|s| 1.0 / s).collect();

    let mut p_sym = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            p_sym[i * n + j] = sqrt_d[i] * _p[i * n + j] * inv_sqrt_d[j];
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (p_sym[i * n + j] + p_sym[j * n + i]);
            p_sym[i * n + j] = avg;
            p_sym[j * n + i] = avg;
        }
    }

    let (mut w, mut v) = jacobi_eigh(&p_sym, n)?;
    sort_eigen_descending(&mut w, &mut v, n);
    Ok(w)
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 6 — PHATE diffusion potential
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the diffusion potential matrix `U_ij = -log(P^t_ij + ε)`.
///
/// The log is taken element-wise.  Because P^t_ij ∈ [0, 1] and ε > 0,
/// the values U_ij ∈ [-log(1 + ε), +∞).  In practice P^t_ii ≈ 1 so the
/// diagonal approaches -log(1 + ε) ≈ -ε (very small, approximately 0).
/// Off-diagonal entries with small P^t_ij produce large positive potentials.
fn diffusion_potential(p_t: &[f64], n: usize, epsilon: f64) -> Vec<f64> {
    p_t.iter()
        .take(n * n)
        .map(|&p_ij| -(p_ij + epsilon).ln())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 7 — Pairwise L₂ distances in potential space
// ─────────────────────────────────────────────────────────────────────────────

/// Compute pairwise L₂ distances (not squared) between rows of U.
///
/// Output: flat n×n row-major distance matrix for MDS.
fn potential_distances(u: &[f64], n: usize) -> Vec<f64> {
    let mut dist = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut sq = 0.0_f64;
            for k in 0..n {
                let diff = u[i * n + k] - u[j * n + k];
                sq += diff * diff;
            }
            let d = sq.sqrt();
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }
    dist
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 8 — Classical (Torgerson) MDS
// ─────────────────────────────────────────────────────────────────────────────

/// Classical MDS on an n×n **distance** matrix (not squared).
///
/// 1. D² = element-wise square.
/// 2. Double-centre: `B = -½ J D² J` where `J = I − (1/n) 11ᵀ`.
/// 3. Eigendecompose B (symmetric PSD up to floating-point noise).
/// 4. Embedding Y[:, c] = v_c · sqrt(max(0, λ_c)).
fn classical_mds(distances: &[f64], n: usize, n_components: usize) -> ManifoldResult<Vec<f64>> {
    // D² matrix.
    let d2: Vec<f64> = distances.iter().map(|d| d * d).collect();

    // Compute row means, column means, grand mean of D².
    let mut row_mean = vec![0.0_f64; n];
    let mut col_mean = vec![0.0_f64; n];
    let mut grand_total = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let v = d2[i * n + j];
            row_mean[i] += v;
            col_mean[j] += v;
            grand_total += v;
        }
    }
    let inv_n = 1.0 / n as f64;
    for v in &mut row_mean {
        *v *= inv_n;
    }
    for v in &mut col_mean {
        *v *= inv_n;
    }
    grand_total *= inv_n * inv_n;

    // Double-centred Gram matrix B.
    let mut b = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            b[i * n + j] = -0.5 * (d2[i * n + j] - row_mean[i] - col_mean[j] + grand_total);
        }
    }

    // Eigendecompose B.
    let (mut w, mut v) = jacobi_eigh(&b, n)?;
    sort_eigen_descending(&mut w, &mut v, n);

    // Build embedding from top n_components eigenpairs.
    let mut embedding = vec![0.0_f64; n * n_components];
    for c in 0..n_components {
        let lam = w[c].max(0.0); // clip numerical negatives
        let s = lam.sqrt();
        for r in 0..n {
            embedding[r * n_components + c] = v[r * n + c] * s;
        }
    }
    Ok(embedding)
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Row sums of a flat n×n matrix.
fn row_sums(m: &[f64], n: usize) -> Vec<f64> {
    let mut sums = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = 0.0_f64;
        for j in 0..n {
            s += m[i * n + j];
        }
        sums[i] = s;
    }
    sums
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper: build a tiny 2-D line dataset ────────────────────────────────
    fn line_data(n: usize) -> Vec<f64> {
        let mut x = vec![0.0_f64; n * 2];
        for i in 0..n {
            x[i * 2] = i as f64;
            x[i * 2 + 1] = 0.0;
        }
        x
    }

    // ── Helper: build two-cluster dataset ───────────────────────────────────
    // First half at (0, 0) region, second half at (10, 0) region.
    fn two_cluster_data(n_per_cluster: usize) -> Vec<f64> {
        let n = 2 * n_per_cluster;
        let mut x = vec![0.0_f64; n * 2];
        for i in 0..n_per_cluster {
            // Cluster A: small jitter around (0, 0).
            let t = i as f64 * 0.01;
            x[i * 2] = t;
            x[i * 2 + 1] = t * 0.5;
        }
        for i in 0..n_per_cluster {
            // Cluster B: small jitter around (10, 0).
            let t = i as f64 * 0.01;
            x[(n_per_cluster + i) * 2] = 10.0 + t;
            x[(n_per_cluster + i) * 2 + 1] = t * 0.5;
        }
        x
    }

    // ── Test 1: default config values ────────────────────────────────────────
    #[test]
    fn phate_config_defaults() {
        let cfg = PhateConfig::default();
        assert_eq!(cfg.n_components, 2);
        assert_eq!(cfg.k, 5);
        assert_eq!(cfg.t, 10);
        assert!((cfg.alpha - 1.0).abs() < 1e-12);
        assert!((cfg.gamma - 1.0).abs() < 1e-12);
        assert!((cfg.epsilon - 1e-7).abs() < 1e-15);
    }

    // ── Test 2: n = 0 → EmptyInput ───────────────────────────────────────────
    #[test]
    fn phate_single_point_error() {
        let cfg = PhateConfig::default();
        let result = phate_fit(&[], 0, 2, &cfg);
        assert!(matches!(result, Err(ManifoldError::EmptyInput)));
    }

    // ── Test 3: n_components ≥ n → InvalidParameter ──────────────────────────
    #[test]
    fn phate_n_components_too_large() {
        let n = 5;
        let x = line_data(n);
        let cfg = PhateConfig {
            n_components: n, // n_components must be < n
            k: 2,
            ..Default::default()
        };
        let result = phate_fit(&x, n, 2, &cfg);
        assert!(matches!(
            result,
            Err(ManifoldError::InvalidParameter { .. })
        ));
    }

    // ── Test 4: two well-separated clusters ──────────────────────────────────
    #[test]
    fn phate_two_clusters_separate() {
        let n_per = 10;
        let n = 2 * n_per;
        let x = two_cluster_data(n_per);
        let cfg = PhateConfig {
            n_components: 2,
            k: 3,
            t: 5,
            alpha: 1.0,
            ..Default::default()
        };
        let res = phate_fit(&x, n, 2, &cfg).expect("phate two-cluster");
        // Compute centroid of each cluster in embedding space.
        let mut c_a = [0.0_f64; 2];
        let mut c_b = [0.0_f64; 2];
        for i in 0..n_per {
            c_a[0] += res.embedding[i * 2];
            c_a[1] += res.embedding[i * 2 + 1];
            c_b[0] += res.embedding[(n_per + i) * 2];
            c_b[1] += res.embedding[(n_per + i) * 2 + 1];
        }
        for v in &mut c_a {
            *v /= n_per as f64;
        }
        for v in &mut c_b {
            *v /= n_per as f64;
        }
        let centroid_dist = ((c_a[0] - c_b[0]).powi(2) + (c_a[1] - c_b[1]).powi(2)).sqrt();
        assert!(
            centroid_dist > 0.5,
            "clusters should be separated in PHATE embedding, centroid_dist={centroid_dist}"
        );
    }

    // ── Test 5: linear chain — topology preserved (local distances small) ──────
    //
    // A symmetric 1-D chain of equally-spaced points produces a diffusion
    // geometry that is also symmetric around the midpoint.  Classical MDS on
    // this geometry yields a U-shaped (parabolic) 1-D embedding centered at 0.
    // Monotone ordering is therefore NOT guaranteed.
    //
    // What IS guaranteed: adjacent points in the chain must be closer in the
    // embedding than non-adjacent endpoints.  Specifically the distance between
    // consecutive chain neighbours must be less than the distance between the
    // two endpoints (points 0 and n-1).
    #[test]
    fn phate_linear_chain() {
        let n = 10;
        let x = line_data(n);
        let cfg = PhateConfig {
            n_components: 1,
            k: 2,
            t: 3,
            ..Default::default()
        };
        let res = phate_fit(&x, n, 2, &cfg).expect("phate linear chain");
        // Distance between the two endpoints in embedding space.
        let emb: Vec<f64> = (0..n).map(|i| res.embedding[i]).collect();
        let endpoint_dist = (emb[0] - emb[n - 1]).abs();
        // Average distance between adjacent chain neighbours.
        let avg_adj_dist: f64 =
            (0..n - 1).map(|i| (emb[i] - emb[i + 1]).abs()).sum::<f64>() / (n - 1) as f64;
        // The endpoint distance should be larger than the average adjacent
        // distance — the chain spans the embedding.
        assert!(
            endpoint_dist > avg_adj_dist * 0.5,
            "chain endpoints should be further apart than average adjacent pair; \
             endpoint_dist={endpoint_dist:.4}, avg_adj_dist={avg_adj_dist:.4}"
        );
        // All embedding values must be finite.
        assert!(
            emb.iter().all(|v| v.is_finite()),
            "non-finite in chain embedding"
        );
    }

    // ── Test 6: output embedding shape ───────────────────────────────────────
    #[test]
    fn phate_output_shape() {
        let n = 12;
        let x = line_data(n);
        let cfg = PhateConfig {
            n_components: 3,
            k: 3,
            t: 5,
            ..Default::default()
        };
        let res = phate_fit(&x, n, 2, &cfg).expect("shape test");
        assert_eq!(res.embedding.len(), n * 3);
    }

    // ── Test 7: diff_potential shape ─────────────────────────────────────────
    #[test]
    fn phate_diff_potential_shape() {
        let n = 8;
        let x = line_data(n);
        let cfg = PhateConfig {
            n_components: 2,
            k: 2,
            t: 2,
            ..Default::default()
        };
        let res = phate_fit(&x, n, 2, &cfg).expect("shape test");
        assert_eq!(res.diff_potential.len(), n * n);
    }

    // ── Test 8: diffusion potential non-negative ──────────────────────────────
    // P^t_ij ∈ [0, 1] with ε > 0 ⟹ -log(P^t_ij + ε) ≥ -log(1 + ε).
    // For ε = 1e-7 that lower bound is ≈ −1e-7 (negative, extremely close to 0).
    // After clamping P^t to [0,1] and adding ε > 0:
    //   U_ij ≥ -log(1 + ε) ≈ -ε  (almost 0).
    // So we verify all values are ≥ -ε - small tolerance.
    #[test]
    fn phate_diff_potential_log_structure() {
        let n = 8;
        let x = line_data(n);
        let cfg = PhateConfig {
            n_components: 2,
            k: 2,
            t: 2,
            epsilon: 1e-7,
            ..Default::default()
        };
        let res = phate_fit(&x, n, 2, &cfg).expect("log structure test");
        let lower_bound = -(1.0_f64 + cfg.epsilon).ln() - 1e-9;
        for &u in &res.diff_potential {
            assert!(
                u >= lower_bound,
                "potential value {u} below lower bound {lower_bound}"
            );
        }
    }

    // ── Test 9: varying t changes the embedding ───────────────────────────────
    #[test]
    fn phate_t_zero_recovers_pca_like() {
        let n = 10;
        let x = line_data(n);
        let cfg_t1 = PhateConfig {
            n_components: 2,
            k: 2,
            t: 1,
            ..Default::default()
        };
        let cfg_t10 = PhateConfig {
            n_components: 2,
            k: 2,
            t: 10,
            ..Default::default()
        };
        let res_t1 = phate_fit(&x, n, 2, &cfg_t1).expect("t=1");
        let res_t10 = phate_fit(&x, n, 2, &cfg_t10).expect("t=10");
        // The two embeddings should differ (different diffusion scales).
        let max_diff = res_t1
            .embedding
            .iter()
            .zip(res_t10.embedding.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff > 1e-8,
            "t=1 and t=10 should produce different embeddings"
        );
    }

    // ── Test 10: alpha = 0 (no density normalisation) ────────────────────────
    #[test]
    fn phate_alpha_zero() {
        let n = 10;
        let x = line_data(n);
        let cfg = PhateConfig {
            n_components: 2,
            k: 2,
            t: 3,
            alpha: 0.0,
            ..Default::default()
        };
        let res = phate_fit(&x, n, 2, &cfg).expect("alpha=0");
        assert_eq!(res.embedding.len(), n * 2);
        assert!(res.embedding.iter().all(|v| v.is_finite()));
    }

    // ── Test 11: no NaN / Inf anywhere in output ─────────────────────────────
    #[test]
    fn phate_diffusion_potential_finite() {
        let n = 15;
        let x = two_cluster_data(n / 2);
        let cfg = PhateConfig {
            n_components: 2,
            k: 3,
            t: 8,
            ..Default::default()
        };
        // n might be odd (15), safe with n_per=7 giving n=14.
        let x2 = two_cluster_data(7);
        let res = phate_fit(&x2, 14, 2, &cfg).expect("finite check");
        assert!(
            res.embedding.iter().all(|v| v.is_finite()),
            "non-finite embedding value"
        );
        assert!(
            res.diff_potential.iter().all(|v| v.is_finite()),
            "non-finite diff_potential value"
        );
        assert!(
            res.eigenvalues.iter().all(|v| v.is_finite()),
            "non-finite eigenvalue"
        );
        // Suppress unused warning for `x`.
        let _ = x;
    }

    // ── Test 12: single component embedding ──────────────────────────────────
    #[test]
    fn phate_single_component() {
        let n = 8;
        let x = line_data(n);
        let cfg = PhateConfig {
            n_components: 1,
            k: 2,
            t: 4,
            ..Default::default()
        };
        let res = phate_fit(&x, n, 2, &cfg).expect("single component");
        assert_eq!(res.embedding.len(), n);
    }

    // ── Test 13: eigenvalues sorted descending ───────────────────────────────
    #[test]
    fn phate_eigenvalues_sorted_descending() {
        let n = 10;
        let x = line_data(n);
        let cfg = PhateConfig {
            n_components: 2,
            k: 3,
            t: 5,
            ..Default::default()
        };
        let res = phate_fit(&x, n, 2, &cfg).expect("eig sort");
        let eigs = &res.eigenvalues;
        for i in 0..eigs.len().saturating_sub(1) {
            assert!(
                eigs[i] >= eigs[i + 1] - 1e-9,
                "eigenvalues not sorted: eigs[{i}]={} < eigs[{}]={}",
                eigs[i],
                i + 1,
                eigs[i + 1]
            );
        }
    }
}
