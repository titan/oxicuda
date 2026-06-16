//! Tensor Renormalization Group (TRG).
//!
//! TRG (Levin & Nave, 2007) is a real-space coarse-graining scheme for two-
//! dimensional tensor networks — the canonical example being the partition
//! function of a classical lattice model (e.g. the 2D Ising model) written as a
//! uniform network of rank-4 tensors on a square lattice.
//!
//! Each RG step halves the number of tensors (and rotates the lattice by 45°)
//! while keeping the bond dimension bounded by a cutoff `chi`:
//!
//! 1. **Split.** The square-lattice tensor `T[u, l, d, r]` is decomposed two
//!    ways by SVD, grouping legs `(u, l)|(d, r)` on one sublattice and
//!    `(l, d)|(r, u)` on the other. Each SVD is truncated to `chi` singular
//!    values, producing two rank-3 "half" tensors `S1, S2` and `S3, S4`.
//! 2. **Contract.** Four half tensors are glued around a plaquette into a new
//!    rank-4 tensor `T'` on the coarse lattice.
//!
//! Iterating contracts an exponentially large network. The free energy per site
//! of a translation-invariant model is `ln Z / N = Σ_n (1/2^n) ln(λ_n)`, where
//! `λ_n` is the tensor norm factored out at RG step `n` (the recursion halves
//! the site count each step). [`trg_partition_log`] returns exactly this sum.

pub mod ising;

use crate::svd::svd_jacobi;
use crate::{TnError, TnResult};

/// A rank-4 lattice tensor `T[u, l, d, r]` stored row-major with leg dimensions
/// `(du, dl, dd, dr)`. The four legs are *up, left, down, right* in that order.
#[derive(Debug, Clone)]
pub struct LatticeTensor {
    pub du: usize,
    pub dl: usize,
    pub dd: usize,
    pub dr: usize,
    /// Flat data of length `du · dl · dd · dr`, index `((u·dl + l)·dd + d)·dr + r`.
    pub data: Vec<f64>,
}

impl LatticeTensor {
    /// Construct a lattice tensor, validating the data length.
    ///
    /// # Errors
    /// * [`TnError::EmptyInput`] if any dimension is zero.
    /// * [`TnError::ShapeMismatch`] if `data.len() != du·dl·dd·dr`.
    pub fn new(du: usize, dl: usize, dd: usize, dr: usize, data: Vec<f64>) -> TnResult<Self> {
        if du == 0 || dl == 0 || dd == 0 || dr == 0 {
            return Err(TnError::EmptyInput);
        }
        if data.len() != du * dl * dd * dr {
            return Err(TnError::ShapeMismatch {
                expected: vec![du, dl, dd, dr],
                got: vec![data.len()],
            });
        }
        Ok(Self {
            du,
            dl,
            dd,
            dr,
            data,
        })
    }

    #[inline]
    fn at(&self, u: usize, l: usize, d: usize, r: usize) -> f64 {
        self.data[((u * self.dl + l) * self.dd + d) * self.dr + r]
    }

    /// Maximum absolute element (used to factor out the tensor scale safely).
    fn max_abs(&self) -> f64 {
        self.data.iter().fold(0.0, |m, &x| m.max(x.abs()))
    }
}

/// Truncated SVD-split of a matrix `M` (`rows×cols`) into `U·√S` (`rows×r`) and
/// `√S·Vᵀ` (`r×cols`), keeping at most `chi` singular values above `tol·σ₀`.
///
/// Splitting the singular weights symmetrically across the two factors yields
/// the two rank-3 half-tensors of the TRG decomposition.
fn split_svd(m: &[f64], rows: usize, cols: usize, chi: usize, tol: f64) -> TnResult<SplitParts> {
    let svd = svd_jacobi(m, rows, cols)?;
    let s_max = svd.s.first().copied().unwrap_or(0.0);
    let abs_tol = tol * s_max.max(1.0);
    let mut keep = 0usize;
    for &s in &svd.s {
        if keep >= chi || s < abs_tol {
            break;
        }
        keep += 1;
    }
    keep = keep.max(1);
    // left[i, a] = U[i, a] · √s_a ; right[a, j] = √s_a · Vᵀ[a, j]
    let mut left = vec![0.0; rows * keep];
    let mut right = vec![0.0; keep * cols];
    for a in 0..keep {
        let root = svd.s[a].max(0.0).sqrt();
        for i in 0..rows {
            left[i * keep + a] = svd.u[i * svd.k + a] * root;
        }
        for j in 0..cols {
            right[a * cols + j] = root * svd.vt[a * cols + j];
        }
    }
    Ok(SplitParts {
        left,
        right,
        rank: keep,
    })
}

struct SplitParts {
    left: Vec<f64>,
    right: Vec<f64>,
    rank: usize,
}

/// One TRG coarse-graining step on a *uniform* network of `tensor`.
///
/// Returns the coarse-grained tensor together with the scale `factor` that was
/// divided out (the maximum absolute element of the new tensor before
/// normalisation). The new tensor's legs again label *up, left, down, right* on
/// the rotated coarse lattice. `chi` bounds the coarse bond dimension.
///
/// # Errors
/// Propagated from the internal SVDs.
pub fn trg_step(tensor: &LatticeTensor, chi: usize, tol: f64) -> TnResult<(LatticeTensor, f64)> {
    let (du, dl, dd, dr) = (tensor.du, tensor.dl, tensor.dd, tensor.dr);

    // Sublattice A split: group (u, l) | (d, r) ⇒ M_A[(u,l), (d,r)].
    let rows_a = du * dl;
    let cols_a = dd * dr;
    let mut m_a = vec![0.0; rows_a * cols_a];
    for u in 0..du {
        for l in 0..dl {
            for d in 0..dd {
                for r in 0..dr {
                    m_a[(u * dl + l) * cols_a + (d * dr + r)] = tensor.at(u, l, d, r);
                }
            }
        }
    }
    // S1[(u,l), a] , S3[a, (d,r)] with new bond `a` of size r_a.
    let part_a = split_svd(&m_a, rows_a, cols_a, chi, tol)?;
    let r_a = part_a.rank;

    // Sublattice B split: group (l, d) | (r, u) ⇒ M_B[(l,d), (r,u)].
    let rows_b = dl * dd;
    let cols_b = dr * du;
    let mut m_b = vec![0.0; rows_b * cols_b];
    for u in 0..du {
        for l in 0..dl {
            for d in 0..dd {
                for r in 0..dr {
                    m_b[(l * dd + d) * cols_b + (r * du + u)] = tensor.at(u, l, d, r);
                }
            }
        }
    }
    // S2[(l,d), b] , S4[b, (r,u)] with new bond `b` of size r_b.
    let part_b = split_svd(&m_b, rows_b, cols_b, chi, tol)?;
    let r_b = part_b.rank;

    // Contract the four half-tensors around a plaquette into the new tensor
    // T'[a1, b1, a2, b2] where each new leg is one of the SVD bonds. The standard
    // Levin-Nave gluing contracts the shared physical legs (the original u,l,d,r)
    // between adjacent halves. Concretely, with the four corner tensors
    // S1 (=A.left), S3 (=A.right), S2 (=B.left), S4 (=B.right):
    //
    //   T'[a, b, a', b'] = Σ_{i,j,k,l}
    //        S3[a, (j,k)] · S2[(k,l), b] · S1[(l,i), ... ] ...
    //
    // We implement the symmetric contraction used for the isotropic square
    // lattice: each coarse leg connects an `a`-bond to a `b`-bond through one of
    // the four original legs. We build T' by summing over the four original
    // half-edges.
    //
    // Reshape the parts for indexed access.
    // S1: rows (u,l) × r_a  → left_a[u][l][a]
    // S3: r_a × cols (d,r)  → right_a[a][d][r]
    // S2: rows (l,d) × r_b  → left_b[l][d][b]
    // S4: r_b × cols (r,u)  → right_b[b][r][u]
    let left_a = &part_a.left; // index (u*dl + l)*r_a + a
    let right_a = &part_a.right; // index a*(dd*dr) + (d*dr + r)
    let left_b = &part_b.left; // index (l*dd + d)*r_b + b
    let right_b = &part_b.right; // index b*(dr*du) + (r*du + u)

    // New tensor legs: T'[A_up, B_left, A_down, B_right] of size (r_a, r_b, r_a, r_b).
    // Contraction (isotropic TRG plaquette):
    //   T'[a_up, b_left, a_down, b_right]
    //     = Σ_{u,l,d,r} right_a[a_up; d, r] · left_b[l, d; b_left]
    //                   · left_a[u, l; a_down] · right_b[b_right; r, u]
    // The four original legs u, l, d, r each appear in exactly two of the half
    // tensors, forming a closed loop around the plaquette:
    //   left_a(u,l) — left_b(l,d) — right_a(d,r) — right_b(r,u) — back to left_a.
    // Contracting all four halves in a single nest is Θ(χ⁸). We instead pair the
    // halves through two Θ(χ⁵) intermediates and one Θ(χ⁶) final contraction —
    // the standard Levin–Nave cost. The reassociation is exact (identical summand):
    //   P[l, a_down, b_right, r] = Σ_u left_a[u,l; a_down] · right_b[b_right; r,u]
    //   Q[a_up, r, l, b_left]    = Σ_d right_a[a_up; d,r]  · left_b[l,d; b_left]
    //   T'[a_up,b_left,a_down,b_right] = Σ_{l,r} P[l,a_down,b_right,r]·Q[a_up,r,l,b_left]
    let mut p = vec![0.0; dl * r_a * r_b * dr];
    for l in 0..dl {
        for a_down in 0..r_a {
            for b_right in 0..r_b {
                for r in 0..dr {
                    let mut acc = 0.0;
                    for u in 0..du {
                        acc += left_a[(u * dl + l) * r_a + a_down]
                            * right_b[b_right * (dr * du) + (r * du + u)];
                    }
                    p[((l * r_a + a_down) * r_b + b_right) * dr + r] = acc;
                }
            }
        }
    }
    let mut q = vec![0.0; r_a * dr * dl * r_b];
    for a_up in 0..r_a {
        for r in 0..dr {
            for l in 0..dl {
                for b_left in 0..r_b {
                    let mut acc = 0.0;
                    for d in 0..dd {
                        acc += right_a[a_up * (dd * dr) + (d * dr + r)]
                            * left_b[(l * dd + d) * r_b + b_left];
                    }
                    q[((a_up * dr + r) * dl + l) * r_b + b_left] = acc;
                }
            }
        }
    }
    let mut new_data = vec![0.0; r_a * r_b * r_a * r_b];
    for a_up in 0..r_a {
        for b_left in 0..r_b {
            for a_down in 0..r_a {
                for b_right in 0..r_b {
                    let mut acc = 0.0;
                    for l in 0..dl {
                        for r in 0..dr {
                            acc += p[((l * r_a + a_down) * r_b + b_right) * dr + r]
                                * q[((a_up * dr + r) * dl + l) * r_b + b_left];
                        }
                    }
                    new_data[((a_up * r_b + b_left) * r_a + a_down) * r_b + b_right] = acc;
                }
            }
        }
    }

    let mut new_tensor = LatticeTensor::new(r_a, r_b, r_a, r_b, new_data)?;
    let factor = new_tensor.max_abs();
    if factor > 0.0 {
        let inv = 1.0 / factor;
        for x in &mut new_tensor.data {
            *x *= inv;
        }
    }
    Ok((new_tensor, factor))
}

/// Compute `ln Z / N` (free-energy-related log) of a uniform 2D tensor network by
/// iterating TRG for `n_steps` coarse-graining sweeps.
///
/// Each step factors out the tensor scale `c_n` and *halves* the number of
/// remaining tensors, so the per-site log accumulates as `Σ_n ln(c_n) / 2^{n+1}`,
/// plus the residual contribution of the final (small) tensor traced over its
/// boundary. The returned value converges to `ln Z / N` as `n_steps → ∞` and
/// `chi → ∞`.
///
/// # Errors
/// * [`TnError::InvalidConfiguration`] if `n_steps == 0`.
/// * Propagated from [`trg_step`].
pub fn trg_partition_log(
    tensor: &LatticeTensor,
    chi: usize,
    tol: f64,
    n_steps: usize,
) -> TnResult<f64> {
    if n_steps == 0 {
        return Err(TnError::InvalidConfiguration(
            "n_steps must be ≥ 1".to_string(),
        ));
    }
    let mut current = tensor.clone();
    let mut log_z = 0.0f64;
    // Weight 1/2^{n+1}: after n full coarse-grainings the lattice has N/2^n sites,
    // so factoring c_n out of every tensor contributes ln(c_n)·(N/2^n) to ln Z.
    let mut weight = 0.5f64;
    for _ in 0..n_steps {
        let (next, factor) = trg_step(&current, chi, tol)?;
        if factor > 0.0 {
            log_z += weight * factor.ln();
        }
        weight *= 0.5;
        current = next;
    }
    // Trace the final tensor over periodic boundaries: Σ_{u,l} T[u, l, u, l].
    let mut trace = 0.0f64;
    let du = current.du.min(current.dd);
    let dl = current.dl.min(current.dr);
    for u in 0..du {
        for l in 0..dl {
            trace += current.at(u, l, u, l);
        }
    }
    if trace > 0.0 {
        log_z += weight * 2.0 * trace.ln();
    }
    Ok(log_z)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Multiply `a` (`m×k`) by `b` (`k×n`), both row-major, into an `m×n` matrix.
    fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for p in 0..k {
                let aip = a[i * k + p];
                for j in 0..n {
                    c[i * n + j] += aip * b[p * n + j];
                }
            }
        }
        c
    }

    /// A trivial all-ones rank-4 tensor with bond dimension 1 (Z = product of 1s).
    fn ones_d1() -> LatticeTensor {
        LatticeTensor::new(1, 1, 1, 1, vec![1.0]).expect("new should succeed")
    }

    #[test]
    fn lattice_tensor_validates_shape() {
        assert!(LatticeTensor::new(2, 2, 2, 2, vec![0.0; 16]).is_ok());
        assert!(LatticeTensor::new(2, 2, 2, 2, vec![0.0; 8]).is_err());
        assert!(LatticeTensor::new(0, 2, 2, 2, vec![]).is_err());
    }

    #[test]
    fn lattice_tensor_indexing() {
        let mut data = vec![0.0; 2 * 2 * 2 * 2];
        // Set T[i=1,j=0,k=1,l=0] = 7; index = ((1*2+0)*2+1)*2+0 = 10
        data[10] = 7.0;
        let t = LatticeTensor::new(2, 2, 2, 2, data).expect("new should succeed");
        assert_eq!(t.at(1, 0, 1, 0), 7.0);
        assert_eq!(t.at(0, 0, 0, 0), 0.0);
    }

    #[test]
    fn matmul_identity() {
        let id = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let c = matmul(&id, &b, 2, 2, 2);
        assert_eq!(c, b);
    }

    #[test]
    fn split_svd_reconstructs_matrix() {
        // Rank-2 matrix; full split should reconstruct it.
        let m = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let parts = split_svd(&m, 2, 3, 10, 1e-14).expect("split_svd should succeed");
        let rec = matmul(&parts.left, &parts.right, 2, parts.rank, 3);
        for (a, b) in m.iter().zip(rec.iter()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn split_svd_respects_chi() {
        let m = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let parts = split_svd(&m, 3, 3, 1, 1e-14).expect("split_svd should succeed");
        assert_eq!(parts.rank, 1);
    }

    #[test]
    fn trg_step_d1_is_d1() {
        // Coarse-graining the trivial tensor stays trivial (norm 1 each step).
        let (next, factor) = trg_step(&ones_d1(), 4, 1e-12).expect("value should be present");
        assert_eq!(next.du, 1);
        assert!((factor - 1.0).abs() < 1e-9, "factor={factor}");
        assert!((next.data[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn trg_step_preserves_leg_symmetry() {
        // For an isotropic input the coarse tensor is square in its leg dims.
        let mut data = vec![0.0; 16];
        for (i, item) in data.iter_mut().enumerate() {
            *item = 1.0 + (i as f64) * 0.01;
        }
        let t = LatticeTensor::new(2, 2, 2, 2, data).expect("new should succeed");
        let (next, _) = trg_step(&t, 4, 1e-12).expect("trg_step should succeed");
        assert_eq!(next.du, next.dd);
        assert_eq!(next.dl, next.dr);
    }

    #[test]
    fn trg_partition_log_requires_steps() {
        assert!(trg_partition_log(&ones_d1(), 4, 1e-12, 0).is_err());
    }

    #[test]
    fn trg_partition_log_trivial_is_zero() {
        // All-ones bond-1 network: Z = 1 ⇒ ln Z / N = 0.
        let log = trg_partition_log(&ones_d1(), 4, 1e-12, 5).expect("value should be present");
        assert!(log.abs() < 1e-9, "log={log}");
    }

    #[test]
    fn trg_partition_log_uniform_scale() {
        // A bond-1 tensor with value c at every site gives Z = c^N ⇒ ln Z/N = ln c.
        let c = 2.5;
        let t = LatticeTensor::new(1, 1, 1, 1, vec![c]).expect("new should succeed");
        let log = trg_partition_log(&t, 4, 1e-12, 8).expect("trg_partition_log should succeed");
        assert!((log - c.ln()).abs() < 1e-6, "log={log}, ln c={}", c.ln());
    }

    #[test]
    fn trg_partition_log_is_finite_for_random() {
        let mut data = vec![0.0; 16];
        for (i, x) in data.iter_mut().enumerate() {
            *x = ((i * 37 + 11) % 13) as f64 / 13.0 + 0.1;
        }
        let t = LatticeTensor::new(2, 2, 2, 2, data).expect("new should succeed");
        let log = trg_partition_log(&t, 6, 1e-12, 6).expect("trg_partition_log should succeed");
        assert!(log.is_finite());
    }

    #[test]
    fn trg_step_factor_nonnegative() {
        let mut data = vec![0.0; 16];
        for (i, x) in data.iter_mut().enumerate() {
            *x = (i as f64).sin().abs() + 0.05;
        }
        let t = LatticeTensor::new(2, 2, 2, 2, data).expect("new should succeed");
        let (_, factor) = trg_step(&t, 4, 1e-12).expect("trg_step should succeed");
        assert!(factor >= 0.0);
    }
}
