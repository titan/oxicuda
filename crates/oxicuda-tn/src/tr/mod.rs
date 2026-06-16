//! Tensor Ring (TR) decomposition (Zhao et al., 2016).
//!
//! A tensor ring represents a `d`-way array as a *cyclic* chain of rank-3 cores
//! `G_0, …, G_{d-1}`, each `G_k` of shape `(r_k, n_k, r_{k+1})`, with the
//! **periodic** boundary `r_d = r_0`. Reconstruction takes the trace over the
//! closing bond:
//! `A[i_0, …, i_{d-1}] = Tr( G_0[:, i_0, :] · G_1[:, i_1, :] ⋯ G_{d-1}[:, i_{d-1}, :] )`.
//!
//! Unlike the Tensor Train (which forces `r_0 = r_d = 1`), the ring's extra
//! closing bond distributes representational power more evenly around the chain
//! and is invariant under cyclic relabelling of the modes. This module provides
//! a TR-SVD that splits a dense tensor sequentially: the first unfolding fixes
//! the ring rank `r_0`, and the remaining cores are peeled off by truncated
//! SVDs exactly as in TT-SVD, with the final core carrying the closing bond back
//! to `r_0`.

use crate::svd::svd_jacobi;
use crate::{TnError, TnResult};

/// A single TR core of shape `(r_l, n, r_r)`, row-major.
#[derive(Debug, Clone)]
pub struct TrCore {
    pub r_l: usize,
    pub n: usize,
    pub r_r: usize,
    pub data: Vec<f64>,
}

impl TrCore {
    /// Construct a core, validating the data length.
    ///
    /// # Errors
    /// * [`TnError::InvalidBondDimension`] if any dimension is zero.
    /// * [`TnError::ShapeMismatch`] if `data.len() != r_l·n·r_r`.
    pub fn new(r_l: usize, n: usize, r_r: usize, data: Vec<f64>) -> TnResult<Self> {
        if r_l == 0 || n == 0 || r_r == 0 {
            return Err(TnError::InvalidBondDimension(0));
        }
        if data.len() != r_l * n * r_r {
            return Err(TnError::ShapeMismatch {
                expected: vec![r_l, n, r_r],
                got: vec![data.len()],
            });
        }
        Ok(Self { r_l, n, r_r, data })
    }

    /// Slice `G[:, i, :]` as an `(r_l × r_r)` matrix, row-major.
    fn slice(&self, i: usize) -> Vec<f64> {
        let mut m = vec![0.0; self.r_l * self.r_r];
        for a in 0..self.r_l {
            for b in 0..self.r_r {
                m[a * self.r_r + b] = self.data[(a * self.n + i) * self.r_r + b];
            }
        }
        m
    }
}

/// A Tensor Ring: a cyclic chain of cores with `r_d = r_0`.
#[derive(Debug, Clone)]
pub struct TrTensor {
    pub cores: Vec<TrCore>,
}

/// Multiply `a` (`m×k`) by `b` (`k×n`) row-major into `m×n`.
fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i * k + p];
            if aip == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += aip * b[p * n + j];
            }
        }
    }
    c
}

impl TrTensor {
    /// Build a TR from cores, validating the cyclic bond consistency.
    ///
    /// # Errors
    /// * [`TnError::EmptyInput`] if `cores` is empty.
    /// * [`TnError::DimensionMismatch`] if adjacent bonds disagree or the ring
    ///   does not close (`r_d != r_0`).
    pub fn new(cores: Vec<TrCore>) -> TnResult<Self> {
        if cores.is_empty() {
            return Err(TnError::EmptyInput);
        }
        for w in cores.windows(2) {
            if w[0].r_r != w[1].r_l {
                return Err(TnError::DimensionMismatch {
                    a: w[0].r_r,
                    b: w[1].r_l,
                });
            }
        }
        let first_rl = cores[0].r_l;
        let last_rr = cores.last().ok_or(TnError::EmptyInput)?.r_r;
        if first_rl != last_rr {
            return Err(TnError::DimensionMismatch {
                a: last_rr,
                b: first_rl,
            });
        }
        Ok(Self { cores })
    }

    /// The ring rank `r_0` (= the closing bond dimension).
    #[must_use]
    pub fn ring_rank(&self) -> usize {
        self.cores[0].r_l
    }

    /// The per-mode dimensions `n_k`.
    #[must_use]
    pub fn dims(&self) -> Vec<usize> {
        self.cores.iter().map(|c| c.n).collect()
    }

    /// Reconstruct the full dense tensor in C-order (slowest index `i_0`).
    ///
    /// Each element is the trace of the product of the per-mode slice matrices.
    ///
    /// # Errors
    /// Returns [`TnError::EmptyInput`] for a degenerate (empty) ring.
    pub fn reconstruct(&self) -> TnResult<Vec<f64>> {
        if self.cores.is_empty() {
            return Err(TnError::EmptyInput);
        }
        let dims = self.dims();
        let total: usize = dims.iter().product();
        let d = dims.len();
        let r0 = self.ring_rank();
        let mut out = vec![0.0; total];
        // Iterate over all multi-indices.
        let mut multi = vec![0usize; d];
        for (flat, out_elem) in out.iter_mut().enumerate() {
            // Decode flat → multi (C-order).
            let mut rem = flat;
            for k in (0..d).rev() {
                multi[k] = rem % dims[k];
                rem /= dims[k];
            }
            // Product of slice matrices, then trace. The left bond stays `r0`
            // throughout (the product is always r0 × cols).
            let mut acc = self.cores[0].slice(multi[0]);
            let mut cols = self.cores[0].r_r;
            for (core, &m) in self.cores[1..].iter().zip(&multi[1..]) {
                let s = core.slice(m);
                let next_cols = core.r_r;
                acc = matmul(&acc, &s, r0, cols, next_cols);
                cols = next_cols;
            }
            // Trace of the (r0 × r0) closing product.
            let mut tr = 0.0;
            for a in 0..r0 {
                tr += acc[a * cols + a];
            }
            *out_elem = tr;
        }
        Ok(out)
    }
}

/// Tensor-Ring SVD: decompose a flat C-order tensor of shape `dims` into a TR
/// with ring rank at most `ring_rank` and internal bonds at most `r_max`,
/// truncating singular values below `tol·σ₀`.
///
/// The first mode's left/right unfolding is split so that the *closing* bond has
/// dimension `ring_rank`; the interior cores are then extracted by sequential
/// truncated SVDs (TT-style). With `ring_rank = 1` this reduces exactly to a
/// Tensor Train.
///
/// # Errors
/// * [`TnError::EmptyInput`] if `dims` is empty or `ring_rank == 0`.
/// * [`TnError::ShapeMismatch`] if `data.len()` differs from `prod(dims)`.
pub fn tr_svd(
    data: &[f64],
    dims: &[usize],
    ring_rank: usize,
    r_max: usize,
    tol: f64,
) -> TnResult<TrTensor> {
    if dims.is_empty() || ring_rank == 0 {
        return Err(TnError::EmptyInput);
    }
    let total: usize = dims.iter().product();
    if data.len() != total {
        return Err(TnError::ShapeMismatch {
            expected: vec![total],
            got: vec![data.len()],
        });
    }
    let d = dims.len();
    if d == 1 {
        // A single mode: the ring is just one core (r0, n0, r0) — only consistent
        // with ring_rank = 1 (a scalar-weighted vector). Embed as a 1×n×1 core.
        let core = TrCore::new(1, dims[0], 1, data.to_vec())?;
        return TrTensor::new(vec![core]);
    }

    // Step 1: open the ring. SVD the first unfolding M0[(i_0), (rest)] and choose
    // the closing-bond size `r0`. The closing index is carried along the chain as
    // a *trailing* dimension so that, after the last mode is peeled, the working
    // matrix is exactly `(r_left · n_{d-1}) × r0` — i.e. the final core.
    let n0 = dims[0];
    let rest: usize = total / n0;
    let svd0 = svd_jacobi(data, n0, rest)?;
    let s_max0 = svd0.s.first().copied().unwrap_or(0.0);
    let abs_tol0 = tol * s_max0.max(1.0);
    // Number of significant singular values, capped by r_max.
    let mut keep_total = 0usize;
    for &s in &svd0.s {
        if keep_total >= r_max || s < abs_tol0 {
            break;
        }
        keep_total += 1;
    }
    keep_total = keep_total.max(1);
    // Split keep_total = r0 · r1 (closing bond × first internal bond).
    let r0 = ring_rank.min(keep_total).max(1);
    let r1 = keep_total.div_ceil(r0);

    // Core 0: (r0, n0, r1). Column c = a·r1 + b of the padded U maps to
    // (closing a, internal b). U_padded[i, c] = U[i, c] for c < keep_total else 0.
    let mut g0 = vec![0.0; r0 * n0 * r1];
    for i in 0..n0 {
        for a in 0..r0 {
            for b in 0..r1 {
                let c = a * r1 + b;
                let val = if c < keep_total {
                    svd0.u[i * svd0.k + c]
                } else {
                    0.0
                };
                g0[(a * n0 + i) * r1 + b] = val;
            }
        }
    }
    let core0 = TrCore::new(r0, n0, r1, g0)?;

    // Carried tensor C with modes (internal b = r1, rest, closing a = r0). We
    // build it as a flat matrix `carried` of shape (r1, rest·r0) so the closing
    // bond `r0` is the fastest-varying trailing block.
    //   C[b, j, a] = s_padded[a·r1 + b] · Vᵀ[a·r1 + b, j]
    let mut carried = vec![0.0; r1 * rest * r0];
    for b in 0..r1 {
        for a in 0..r0 {
            let c = a * r1 + b;
            let sc = if c < keep_total { svd0.s[c] } else { 0.0 };
            if sc == 0.0 {
                continue;
            }
            for j in 0..rest {
                let v = svd0.vt[c * rest + j];
                carried[(b * rest + j) * r0 + a] = sc * v;
            }
        }
    }

    // Step 2: TT-sweep modes 1..d-1, keeping `r0` as the trailing block of every
    // working matrix. `remaining` tracks the product of the still-unpeeled middle
    // dims (dims[k..d]); the working matrix is (r_left · n_k) × (remaining/n_k · r0).
    let mut cores = vec![core0];
    let mut r_left = r1;
    let mut remaining = rest; // product of dims[1..]
    for (k, &n_k) in dims.iter().enumerate().take(d).skip(1) {
        if k == d - 1 {
            // remaining == n_k. carried is (r_left, n_k · r0). Reshape directly to
            // the final core (r_left, n_k, r0).
            debug_assert_eq!(carried.len(), r_left * n_k * r0);
            cores.push(TrCore::new(r_left, n_k, r0, carried.clone())?);
            break;
        }
        // Working matrix: (r_left · n_k) × ((remaining/n_k) · r0).
        let rows = r_left * n_k;
        let cols = (remaining / n_k) * r0;
        let svd = svd_jacobi(&carried, rows, cols)?;
        let s_max = svd.s.first().copied().unwrap_or(0.0);
        let abs_tol = tol * s_max.max(1.0);
        let mut keep = 0usize;
        for &s in &svd.s {
            if keep >= r_max || s < abs_tol {
                break;
            }
            keep += 1;
        }
        keep = keep.max(1);
        // Core k: (r_left, n_k, keep) from U[:, :keep].
        let mut core_data = vec![0.0; r_left * n_k * keep];
        for i in 0..rows {
            for j in 0..keep {
                core_data[i * keep + j] = svd.u[i * svd.k + j];
            }
        }
        cores.push(TrCore::new(r_left, n_k, keep, core_data)?);
        // carried = diag(s)[:keep] · Vᵀ, shape (keep, cols) — the trailing r0
        // block of `cols` rides along unchanged.
        let mut next = vec![0.0; keep * cols];
        for a in 0..keep {
            let sa = svd.s[a];
            for j in 0..cols {
                next[a * cols + j] = sa * svd.vt[a * cols + j];
            }
        }
        carried = next;
        r_left = keep;
        remaining /= n_k;
    }

    TrTensor::new(cores)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn fro(a: &[f64]) -> f64 {
        a.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    #[test]
    fn tr_core_validates_shape() {
        assert!(TrCore::new(2, 3, 2, vec![0.0; 12]).is_ok());
        assert!(TrCore::new(2, 3, 2, vec![0.0; 11]).is_err());
        assert!(TrCore::new(0, 3, 2, vec![]).is_err());
    }

    #[test]
    fn tr_core_slice_extracts_matrix() {
        // (r_l=2, n=2, r_r=2). Set G[0,1,1] = 5.
        let mut data = vec![0.0; 8];
        // G[alpha=0, n_k=1, beta=1]: index = (0*2 + 1)*2 + 1 = 3
        data[3] = 5.0;
        let core = TrCore::new(2, 2, 2, data).expect("new should succeed");
        let s = core.slice(1);
        // slice(1) gives G[:,1,:]; entry [alpha=0, beta=1] = 0*r_r + 1 = 1
        assert_eq!(s[1], 5.0);
        assert_eq!(s[0], 0.0);
    }

    #[test]
    fn tr_tensor_rejects_unclosed_ring() {
        let c0 = TrCore::new(2, 2, 3, vec![0.0; 12]).expect("new should succeed");
        let c1 = TrCore::new(3, 2, 4, vec![0.0; 24]).expect("new should succeed"); // r_r=4 ≠ r_0=2
        assert!(TrTensor::new(vec![c0, c1]).is_err());
    }

    #[test]
    fn tr_tensor_accepts_closed_ring() {
        let c0 = TrCore::new(2, 2, 3, vec![0.0; 12]).expect("new should succeed");
        let c1 = TrCore::new(3, 2, 2, vec![0.0; 12]).expect("new should succeed"); // closes to r_0=2
        let tr = TrTensor::new(vec![c0, c1]).expect("new should succeed");
        assert_eq!(tr.ring_rank(), 2);
        assert_eq!(tr.dims(), vec![2, 2]);
    }

    #[test]
    fn matmul_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = matmul(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn tr_svd_ring_rank_one_is_tt_roundtrip() {
        // ring_rank = 1 ⇒ TR reduces to TT; reconstruction should match input.
        let mut rng = LcgRng::new(7);
        let dims = vec![3, 4, 2];
        let total: usize = dims.iter().product();
        let data: Vec<f64> = (0..total).map(|_| rng.next_normal()).collect();
        let tr = tr_svd(&data, &dims, 1, 20, 1e-14).expect("tr_svd should succeed");
        assert_eq!(tr.ring_rank(), 1);
        let rec = tr.reconstruct().expect("reconstruct should succeed");
        let diff: Vec<f64> = data.iter().zip(&rec).map(|(a, b)| a - b).collect();
        assert!(fro(&diff) < 1e-7, "fro diff = {}", fro(&diff));
    }

    #[test]
    fn tr_svd_reconstruction_shape() {
        let mut rng = LcgRng::new(11);
        let dims = vec![2, 3, 2];
        let total: usize = dims.iter().product();
        let data: Vec<f64> = (0..total).map(|_| rng.next_normal()).collect();
        let tr = tr_svd(&data, &dims, 2, 10, 1e-12).expect("tr_svd should succeed");
        let rec = tr.reconstruct().expect("reconstruct should succeed");
        assert_eq!(rec.len(), total);
    }

    #[test]
    fn tr_svd_ring_rank_two_reconstructs_exactly() {
        // With ring_rank > 1 and no truncation (large r_max, tiny tol), the trace
        // reconstruction must still reproduce the original tensor exactly.
        let mut rng = LcgRng::new(31);
        let dims = vec![3, 2, 4];
        let total: usize = dims.iter().product();
        let data: Vec<f64> = (0..total).map(|_| rng.next_normal()).collect();
        let tr = tr_svd(&data, &dims, 2, 64, 1e-15).expect("tr_svd should succeed");
        assert!(tr.ring_rank() >= 1);
        let rec = tr.reconstruct().expect("reconstruct should succeed");
        let diff: Vec<f64> = data.iter().zip(&rec).map(|(a, b)| a - b).collect();
        assert!(fro(&diff) < 1e-7, "fro diff = {}", fro(&diff));
    }

    #[test]
    fn tr_svd_rejects_bad_shape() {
        assert!(tr_svd(&[1.0, 2.0], &[3, 2], 1, 4, 1e-12).is_err());
        assert!(tr_svd(&[], &[], 1, 4, 1e-12).is_err());
        assert!(tr_svd(&[1.0], &[1], 0, 4, 1e-12).is_err());
    }

    #[test]
    fn tr_svd_single_mode() {
        let tr = tr_svd(&[1.0, 2.0, 3.0], &[3], 1, 4, 1e-12).expect("tr_svd should succeed");
        assert_eq!(tr.cores.len(), 1);
        let rec = tr.reconstruct().expect("reconstruct should succeed");
        assert_eq!(rec, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn tr_svd_ring_rank_capped_by_dimension() {
        // Requesting a huge ring rank is capped by min(n0, rest) and stays valid.
        let mut rng = LcgRng::new(3);
        let dims = vec![2, 2, 2];
        let total: usize = dims.iter().product();
        let data: Vec<f64> = (0..total).map(|_| rng.next_normal()).collect();
        let tr = tr_svd(&data, &dims, 100, 10, 1e-12).expect("tr_svd should succeed");
        assert!(tr.ring_rank() <= 2);
    }

    #[test]
    fn tr_svd_cores_form_valid_ring() {
        let mut rng = LcgRng::new(99);
        let dims = vec![3, 2, 3];
        let total: usize = dims.iter().product();
        let data: Vec<f64> = (0..total).map(|_| rng.next_normal()).collect();
        let tr = tr_svd(&data, &dims, 2, 8, 1e-12).expect("tr_svd should succeed");
        // Adjacent bonds match and the ring closes.
        for w in tr.cores.windows(2) {
            assert_eq!(w[0].r_r, w[1].r_l);
        }
        assert_eq!(
            tr.cores[0].r_l,
            tr.cores.last().expect("last should succeed").r_r
        );
    }

    #[test]
    fn tr_svd_dims_preserved() {
        let mut rng = LcgRng::new(5);
        let dims = vec![2, 4, 3];
        let total: usize = dims.iter().product();
        let data: Vec<f64> = (0..total).map(|_| rng.next_normal()).collect();
        let tr = tr_svd(&data, &dims, 2, 10, 1e-12).expect("tr_svd should succeed");
        assert_eq!(tr.dims(), dims);
    }
}
