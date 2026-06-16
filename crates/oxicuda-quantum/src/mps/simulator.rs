//! Matrix Product State (MPS) simulator for low-entanglement circuits.
//!
//! An MPS represents an `n`-qubit pure state as a chain of rank-3 site tensors
//!
//! ```text
//!   A^[0]  A^[1]  …  A^[n-1]
//!     │      │          │
//!    p_0    p_1        p_{n-1}
//! ```
//!
//! where each site tensor `A^[k]` has shape `(left_bond, 2, right_bond)`, the
//! middle index `p_k ∈ {0,1}` is the physical (qubit) index, and adjacent
//! tensors share a virtual *bond* index. The two boundary bonds (left of site 0
//! and right of site n-1) are fixed to dimension 1, so the chain contracts down
//! to a single `2^n` amplitude vector.
//!
//! ## Storage layout
//!
//! Every site tensor is held as a flat row-major `Vec<Complex32>` with
//! dimensions `(left_bond L, 2, right_bond R)`. The element `A[l, p, r]` lives at
//!
//! ```text
//!   offset = (l * 2 + p) * R + r
//! ```
//!
//! i.e. strides `[2 * R, R, 1]` for the `(left, physical, right)` axes.
//!
//! ## Qubit / bit ordering
//!
//! Site `k` corresponds to qubit `k`. To stay compatible with [`StateVector`],
//! where qubit `k` is bit `k` of the amplitude index (qubit 0 = least
//! significant bit), [`MatrixProductState::to_statevector`] places site 0's
//! physical index in the least-significant position.
//!
//! ## Truncation
//!
//! Two-qubit gates increase the shared bond. After applying a gate we perform a
//! singular-value decomposition of the merged two-site block and keep at most
//! `max_bond_dim` (χ) singular values, additionally discarding any singular
//! value below `svd_cutoff`. The retained Σ is folded into the left tensor,
//! yielding a left-canonical split.

use num_complex::Complex;

use crate::error::{QuantumError, QuantumResult};
use crate::statevec::state::StateVector;

type Complex32 = Complex<f32>;

#[inline]
fn c0() -> Complex32 {
    Complex32::new(0.0, 0.0)
}

/// Configuration for an [`MatrixProductState`] simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MpsConfig {
    /// Number of qubits (sites) in the chain.
    pub n_qubits: usize,
    /// Maximum retained bond dimension χ after a truncated SVD.
    pub max_bond_dim: usize,
    /// Singular values strictly below this threshold are dropped.
    pub svd_cutoff: f32,
}

impl MpsConfig {
    /// Construct a config, validating the qubit count and bond dimension.
    pub fn new(n_qubits: usize, max_bond_dim: usize, svd_cutoff: f32) -> QuantumResult<Self> {
        if n_qubits == 0 || n_qubits > 30 {
            return Err(QuantumError::InvalidQubitCount { n: n_qubits });
        }
        if max_bond_dim == 0 {
            return Err(QuantumError::InvalidParameter {
                name: "max_bond_dim must be >= 1".into(),
            });
        }
        if !(svd_cutoff.is_finite() && svd_cutoff >= 0.0) {
            return Err(QuantumError::InvalidParameter {
                name: "svd_cutoff must be finite and >= 0".into(),
            });
        }
        Ok(Self {
            n_qubits,
            max_bond_dim,
            svd_cutoff,
        })
    }
}

/// A single rank-3 site tensor with shape `(left, 2, right)`.
#[derive(Debug, Clone)]
struct SiteTensor {
    left: usize,
    right: usize,
    /// Flat row-major data, length `left * 2 * right`.
    data: Vec<Complex32>,
}

impl SiteTensor {
    fn zeros(left: usize, right: usize) -> Self {
        Self {
            left,
            right,
            data: vec![c0(); left * 2 * right],
        }
    }

    #[inline]
    fn idx(&self, l: usize, p: usize, r: usize) -> usize {
        (l * 2 + p) * self.right + r
    }

    #[inline]
    fn get(&self, l: usize, p: usize, r: usize) -> Complex32 {
        self.data[self.idx(l, p, r)]
    }

    #[inline]
    fn set(&mut self, l: usize, p: usize, r: usize, v: Complex32) {
        let i = self.idx(l, p, r);
        self.data[i] = v;
    }
}

/// Matrix Product State representation of an `n`-qubit pure state.
#[derive(Debug, Clone)]
pub struct MatrixProductState {
    config: MpsConfig,
    sites: Vec<SiteTensor>,
}

impl MatrixProductState {
    /// Build the product state |0…0⟩ with all bond dimensions equal to 1.
    pub fn new_zero_state(config: MpsConfig) -> QuantumResult<Self> {
        let n = config.n_qubits;
        let mut sites = Vec::with_capacity(n);
        for _ in 0..n {
            // Bonds are all 1; physical 0 amplitude = 1, physical 1 = 0.
            let mut t = SiteTensor::zeros(1, 1);
            t.set(0, 0, 0, Complex32::new(1.0, 0.0));
            sites.push(t);
        }
        Ok(Self { config, sites })
    }

    /// Number of qubits / sites.
    #[must_use]
    pub fn n_qubits(&self) -> usize {
        self.config.n_qubits
    }

    /// The active configuration.
    #[must_use]
    pub fn config(&self) -> MpsConfig {
        self.config
    }

    /// The right-bond dimension of each site (one entry per site).
    ///
    /// `bond_dims()[k]` is the dimension of the bond between site `k` and
    /// site `k+1`; the final entry is the right boundary bond (always 1).
    #[must_use]
    pub fn bond_dims(&self) -> Vec<usize> {
        self.sites.iter().map(|s| s.right).collect()
    }

    /// Apply a 1-qubit gate to `qubit`, contracting it into the physical index.
    ///
    /// This leaves both bond dimensions unchanged.
    pub fn apply_1q(&mut self, gate: &[[Complex32; 2]; 2], qubit: usize) -> QuantumResult<()> {
        if qubit >= self.config.n_qubits {
            return Err(QuantumError::QubitIndexOutOfRange {
                index: qubit,
                n_qubits: self.config.n_qubits,
            });
        }
        let site = &mut self.sites[qubit];
        let left = site.left;
        let right = site.right;
        for l in 0..left {
            for r in 0..right {
                let a0 = site.get(l, 0, r);
                let a1 = site.get(l, 1, r);
                let n0 = gate[0][0] * a0 + gate[0][1] * a1;
                let n1 = gate[1][0] * a0 + gate[1][1] * a1;
                site.set(l, 0, r, n0);
                site.set(l, 1, r, n1);
            }
        }
        Ok(())
    }

    /// Apply a 4×4 two-qubit gate to **adjacent** qubits `q` and `q+1`.
    ///
    /// The gate is indexed in the basis `|q, q+1⟩` with `q` the more-significant
    /// of the two physical indices, i.e. the matrix row/column index is
    /// `2 * bit(q) + bit(q+1)` — matching
    /// [`apply_2q_inplace`](crate::statevec::apply_2q::apply_2q_inplace).
    ///
    /// # Errors
    /// Returns an error if the qubits are not adjacent with `q_plus_1 == q + 1`,
    /// or if either index is out of range.
    pub fn apply_2q(
        &mut self,
        gate: &[[Complex32; 4]; 4],
        q: usize,
        q_plus_1: usize,
    ) -> QuantumResult<()> {
        let n = self.config.n_qubits;
        if q >= n {
            return Err(QuantumError::QubitIndexOutOfRange {
                index: q,
                n_qubits: n,
            });
        }
        if q_plus_1 >= n {
            return Err(QuantumError::QubitIndexOutOfRange {
                index: q_plus_1,
                n_qubits: n,
            });
        }
        if q_plus_1 != q + 1 {
            return Err(QuantumError::InvalidParameter {
                name: "apply_2q requires adjacent qubits with q_plus_1 == q + 1".into(),
            });
        }

        // Merge sites q and q+1 into a single block tensor with shape
        // (left, 2, 2, right), held as a matrix of rows = left*2 and
        // cols = 2*right after applying the gate.
        let left = self.sites[q].left;
        let mid = self.sites[q].right; // == self.sites[q+1].left
        let right = self.sites[q_plus_1].right;
        debug_assert_eq!(mid, self.sites[q_plus_1].left);

        // theta[l, p, p2, r] = Σ_m A_q[l, p, m] * A_{q+1}[m, p2, r]
        // then apply gate over (p, p2).
        // Store theta as a matrix `mat` of shape (left*2, 2*right):
        //   row = l * 2 + p,  col = p2 * right + r.
        let rows = left * 2;
        let cols = 2 * right;
        let mut theta = vec![c0(); rows * cols];
        {
            let sq = &self.sites[q];
            let sq1 = &self.sites[q_plus_1];
            for l in 0..left {
                for p in 0..2 {
                    for p2 in 0..2 {
                        for r in 0..right {
                            let mut acc = c0();
                            for m in 0..mid {
                                acc += sq.get(l, p, m) * sq1.get(m, p2, r);
                            }
                            let row = l * 2 + p;
                            let col = p2 * right + r;
                            theta[row * cols + col] = acc;
                        }
                    }
                }
            }
        }

        // Apply the 4×4 gate over the combined physical pair (p, p2).
        // gate index = 2 * p + p2 (p == bit(q), p2 == bit(q+1)).
        // new_theta[l, (P,P2), r] = Σ_{p,p2} gate[2P+P2, 2p+p2] * theta[l,(p,p2),r]
        let mut gated = vec![c0(); rows * cols];
        for l in 0..left {
            for r in 0..right {
                // gather the 2×2 physical block for this (l, r)
                let v00 = theta[(l * 2) * cols + r];
                let v01 = theta[(l * 2) * cols + (right + r)];
                let v10 = theta[(l * 2 + 1) * cols + r];
                let v11 = theta[(l * 2 + 1) * cols + (right + r)];
                // input vector ordered as [00, 01, 10, 11] = [v00, v01, v10, v11]
                let input = [v00, v01, v10, v11];
                for (big, grow) in gate.iter().enumerate() {
                    let mut acc = c0();
                    for (small, &iv) in input.iter().enumerate() {
                        acc += grow[small] * iv;
                    }
                    let big_p = big >> 1; // bit(q)
                    let big_p2 = big & 1; // bit(q+1)
                    let row = l * 2 + big_p;
                    let col = if big_p2 == 1 { right + r } else { r };
                    gated[row * cols + col] = acc;
                }
            }
        }

        // SVD of `gated` (rows × cols): gated = U Σ V†.
        let svd = svd_dense(&gated, rows, cols)?;
        let full_rank = svd.singular_values.len();

        // Determine retained rank: drop σ < cutoff, then cap at χ.
        let mut keep = 0usize;
        for &s in &svd.singular_values {
            if s >= self.config.svd_cutoff && keep < self.config.max_bond_dim {
                keep += 1;
            } else {
                break;
            }
        }
        if keep == 0 {
            // Degenerate (e.g. the block became ~0). Keep one bond to stay valid.
            keep = if full_rank == 0 { 0 } else { 1 };
        }
        let new_bond = keep;

        // Left tensor A_q'[l, p, k] = U[(l*2+p), k] * Σ_k  (fold Σ into left).
        // U has shape (rows × full_rank); we use its first `keep` columns.
        let mut left_tensor = SiteTensor::zeros(left, new_bond);
        for l in 0..left {
            for p in 0..2 {
                let row = l * 2 + p;
                for k in 0..new_bond {
                    let u = svd.u[row * full_rank + k];
                    let val = u * svd.singular_values[k];
                    left_tensor.set(l, p, k, val);
                }
            }
        }

        // Right tensor A_{q+1}'[k, p2, r] = V†[k, (p2*right+r)] = conj(V[(p2*right+r), k]).
        // V has shape (cols × full_rank).
        let mut right_tensor = SiteTensor::zeros(new_bond, right);
        for k in 0..new_bond {
            for p2 in 0..2 {
                for r in 0..right {
                    let col = p2 * right + r;
                    let v = svd.v[col * full_rank + k];
                    right_tensor.set(k, p2, r, v.conj());
                }
            }
        }

        self.sites[q] = left_tensor;
        self.sites[q_plus_1] = right_tensor;
        Ok(())
    }

    /// Contract the whole chain into a dense `2^n` [`StateVector`].
    ///
    /// Intended for verification on small systems; cost is exponential in `n`.
    pub fn to_statevector(&self) -> QuantumResult<StateVector> {
        let n = self.config.n_qubits;
        // Running tensor as a matrix of shape (acc_phys_dim, cur_right_bond),
        // where row encodes the physical indices contracted so far with qubit 0
        // in the least-significant position.
        //
        // Start with site 0: shape (left=1, 2, right) -> matrix (2, right).
        let s0 = &self.sites[0];
        if s0.left != 1 {
            return Err(QuantumError::Internal {
                msg: "left boundary bond must be 1".into(),
            });
        }
        let mut acc_dim = 2usize;
        let mut acc_right = s0.right;
        let mut acc = vec![c0(); acc_dim * acc_right];
        for p in 0..2 {
            for r in 0..acc_right {
                acc[p * acc_right + r] = s0.get(0, p, r);
            }
        }

        for site in self.sites.iter().skip(1) {
            if site.left != acc_right {
                return Err(QuantumError::Internal {
                    msg: "bond mismatch during contraction".into(),
                });
            }
            let new_right = site.right;
            let new_dim = acc_dim * 2;
            let mut next = vec![c0(); new_dim * new_right];
            // next[(phys_old + p_new * acc_dim), r2] =
            //     Σ_m acc[phys_old, m] * site[m, p_new, r2]
            // Placing the new physical index in the *higher* bits keeps qubit 0
            // (the first contracted site) in the least-significant position.
            for phys_old in 0..acc_dim {
                for p_new in 0..2 {
                    let new_phys = phys_old + p_new * acc_dim;
                    for r2 in 0..new_right {
                        let mut val = c0();
                        for m in 0..acc_right {
                            val += acc[phys_old * acc_right + m] * site.get(m, p_new, r2);
                        }
                        next[new_phys * new_right + r2] = val;
                    }
                }
            }
            acc = next;
            acc_dim = new_dim;
            acc_right = new_right;
        }

        if acc_right != 1 {
            return Err(QuantumError::Internal {
                msg: "right boundary bond must be 1".into(),
            });
        }
        // acc is now (2^n, 1); flatten to amplitudes.
        let dim = 1usize << n;
        if acc_dim != dim {
            return Err(QuantumError::DimensionMismatch {
                expected: dim,
                got: acc_dim,
            });
        }
        let mut amps = vec![c0(); dim];
        for (i, slot) in amps.iter_mut().enumerate() {
            *slot = acc[i];
        }
        Ok(StateVector { amps, n_qubits: n })
    }

    /// Squared norm ⟨ψ|ψ⟩ of the represented state.
    #[must_use]
    pub fn norm_sq(&self) -> f32 {
        // Contract ⟨ψ|ψ⟩ via the transfer-matrix sweep:
        //   E_{a,b} accumulates Σ over contracted physical legs of
        //   conj(A)·A. Start with the 1×1 left boundary E = [1].
        let mut env = vec![Complex32::new(1.0, 0.0); 1]; // shape (1, 1) flattened
        let mut dim_a = 1usize; // bra bond dim
        let mut dim_b = 1usize; // ket bond dim
        for site in &self.sites {
            let l = site.left;
            let r = site.right;
            // new_env[a', b'] = Σ_{a,b,p} conj(A[a,p,a']) * env[a,b] * A[b,p,b']
            let mut new_env = vec![c0(); r * r];
            for ap in 0..r {
                for bp in 0..r {
                    let mut acc = c0();
                    for p in 0..2 {
                        for a in 0..l.min(dim_a) {
                            let bra = site.get(a, p, ap).conj();
                            for b in 0..l.min(dim_b) {
                                acc += bra * env[a * dim_b + b] * site.get(b, p, bp);
                            }
                        }
                    }
                    new_env[ap * r + bp] = acc;
                }
            }
            env = new_env;
            dim_a = r;
            dim_b = r;
        }
        // Final env is (1,1).
        if env.is_empty() {
            return 0.0;
        }
        env[0].re.max(0.0)
    }

    /// Euclidean norm ‖ψ‖.
    #[must_use]
    pub fn norm(&self) -> f32 {
        self.norm_sq().sqrt()
    }

    /// Normalize the state in place (scales the first site tensor).
    pub fn normalize(&mut self) {
        let nrm = self.norm();
        if nrm > 1e-12 {
            let inv = 1.0 / nrm;
            if let Some(first) = self.sites.first_mut() {
                for v in &mut first.data {
                    *v *= inv;
                }
            }
        }
    }

    /// Expectation value ⟨ψ| Z_qubit |ψ⟩.
    pub fn expectation_z(&self, qubit: usize) -> QuantumResult<f32> {
        if qubit >= self.config.n_qubits {
            return Err(QuantumError::QubitIndexOutOfRange {
                index: qubit,
                n_qubits: self.config.n_qubits,
            });
        }
        // Transfer-matrix sweep with a Z insertion (phase +1 on p=0, -1 on p=1)
        // at the target site; divide by ⟨ψ|ψ⟩ for an unnormalized MPS.
        let mut env = vec![Complex32::new(1.0, 0.0); 1];
        let mut dim_a = 1usize;
        let mut dim_b = 1usize;
        for (idx, site) in self.sites.iter().enumerate() {
            let l = site.left;
            let r = site.right;
            let mut new_env = vec![c0(); r * r];
            for ap in 0..r {
                for bp in 0..r {
                    let mut acc = c0();
                    for p in 0..2 {
                        let z_phase = if idx == qubit && p == 1 { -1.0 } else { 1.0 };
                        for a in 0..l.min(dim_a) {
                            let bra = site.get(a, p, ap).conj() * z_phase;
                            for b in 0..l.min(dim_b) {
                                acc += bra * env[a * dim_b + b] * site.get(b, p, bp);
                            }
                        }
                    }
                    new_env[ap * r + bp] = acc;
                }
            }
            env = new_env;
            dim_a = r;
            dim_b = r;
        }
        let numer = if env.is_empty() { 0.0 } else { env[0].re };
        let denom = self.norm_sq();
        if denom <= 1e-12 {
            return Err(QuantumError::MeasurementFailed);
        }
        Ok(numer / denom)
    }
}

// ─── Self-contained dense complex SVD ────────────────────────────────────────

/// Result of [`svd_dense`]: `M = U Σ V†` with σ in descending order.
struct SvdResult {
    /// Left singular vectors, shape `(m × k)` row-major (`k = min(m,n)` rank).
    u: Vec<Complex32>,
    /// Right singular vectors `V` (NOT V†), shape `(n × k)` row-major.
    v: Vec<Complex32>,
    /// Singular values, length `k`, descending.
    singular_values: Vec<f32>,
}

/// Compute a singular value decomposition of an `m × n` complex matrix.
///
/// Strategy: build the smaller Hermitian Gram matrix and diagonalise it with a
/// complex cyclic Jacobi sweep (eigenvalues = σ²). When `m >= n` we use
/// `G = MᴴM` (n×n) to obtain `V`, then `U = M V Σ⁻¹`; otherwise we use
/// `G = MMᴴ` (m×m) to obtain `U`, then `V = Mᴴ U Σ⁻¹`. A guard handles σ ≈ 0.
fn svd_dense(mat: &[Complex32], m: usize, n: usize) -> QuantumResult<SvdResult> {
    if m == 0 || n == 0 || mat.len() != m * n {
        return Err(QuantumError::DimensionMismatch {
            expected: m * n,
            got: mat.len(),
        });
    }
    let k = m.min(n);

    if m >= n {
        // G = Mᴴ M  (n × n)
        let g = gram_ata(mat, m, n);
        let (evals, evecs) = jacobi_hermitian(&g, n)?;
        // Sort eigen-pairs descending by eigenvalue.
        let order = sorted_desc_indices(&evals);
        let mut singular_values = Vec::with_capacity(k);
        let mut v = vec![c0(); n * k]; // (n × k)
        let mut u = vec![c0(); m * k]; // (m × k)
        for (col, &ei) in order.iter().take(k).enumerate() {
            let sigma = evals[ei].max(0.0).sqrt();
            singular_values.push(sigma);
            // V column = eigenvector ei.
            for row in 0..n {
                v[row * k + col] = evecs[row * n + ei];
            }
            // U column = M v / σ (guard σ ≈ 0).
            if sigma > 1e-12 {
                let inv = 1.0 / sigma;
                for i in 0..m {
                    let mut acc = c0();
                    for j in 0..n {
                        acc += mat[i * n + j] * v[j * k + col];
                    }
                    u[i * k + col] = acc * inv;
                }
            }
        }
        fill_orthonormal_u(&mut u, &singular_values, m, k);
        Ok(SvdResult {
            u,
            v,
            singular_values,
        })
    } else {
        // G = M Mᴴ  (m × m)
        let g = gram_aat(mat, m, n);
        let (evals, evecs) = jacobi_hermitian(&g, m)?;
        let order = sorted_desc_indices(&evals);
        let mut singular_values = Vec::with_capacity(k);
        let mut u = vec![c0(); m * k]; // (m × k)
        let mut v = vec![c0(); n * k]; // (n × k)
        for (col, &ei) in order.iter().take(k).enumerate() {
            let sigma = evals[ei].max(0.0).sqrt();
            singular_values.push(sigma);
            for row in 0..m {
                u[row * k + col] = evecs[row * m + ei];
            }
            // V column = Mᴴ u / σ  (guard σ ≈ 0).
            if sigma > 1e-12 {
                let inv = 1.0 / sigma;
                for j in 0..n {
                    let mut acc = c0();
                    for i in 0..m {
                        acc += mat[i * n + j].conj() * u[i * k + col];
                    }
                    v[j * k + col] = acc * inv;
                }
            }
        }
        Ok(SvdResult {
            u,
            v,
            singular_values,
        })
    }
}

/// For columns with σ ≈ 0, replace the (left-undetermined) U column with an
/// orthonormal complement so that U has orthonormal columns. This keeps the
/// reconstruction `U Σ Vᴴ` correct (those columns are multiplied by σ ≈ 0) and
/// makes the left tensor well-formed.
fn fill_orthonormal_u(u: &mut [Complex32], sigma: &[f32], m: usize, k: usize) {
    for col in 0..k.min(sigma.len()) {
        if sigma[col] > 1e-12 {
            continue;
        }
        // Build a candidate from standard basis vectors, orthogonalised against
        // the already-fixed columns (Gram-Schmidt).
        let mut placed = false;
        for seed in 0..m {
            let mut cand = vec![c0(); m];
            cand[seed] = Complex32::new(1.0, 0.0);
            for prev in 0..col {
                // proj = <prev|cand> ; cand -= proj * prev
                let mut dot = c0();
                for i in 0..m {
                    dot += u[i * k + prev].conj() * cand[i];
                }
                for i in 0..m {
                    cand[i] -= dot * u[i * k + prev];
                }
            }
            let nrm: f32 = cand.iter().map(|x| x.norm_sqr()).sum::<f32>().sqrt();
            if nrm > 1e-6 {
                let inv = 1.0 / nrm;
                for i in 0..m {
                    u[i * k + col] = cand[i] * inv;
                }
                placed = true;
                break;
            }
        }
        if !placed {
            // Fall back to a zero column (only reachable for a fully rank-0 M).
            for i in 0..m {
                u[i * k + col] = c0();
            }
        }
    }
}

/// Gram matrix `Mᴴ M` (n × n), Hermitian PSD.
fn gram_ata(mat: &[Complex32], m: usize, n: usize) -> Vec<Complex32> {
    let mut g = vec![c0(); n * n];
    for a in 0..n {
        for b in 0..n {
            let mut acc = c0();
            for i in 0..m {
                acc += mat[i * n + a].conj() * mat[i * n + b];
            }
            g[a * n + b] = acc;
        }
    }
    g
}

/// Gram matrix `M Mᴴ` (m × m), Hermitian PSD.
fn gram_aat(mat: &[Complex32], m: usize, n: usize) -> Vec<Complex32> {
    let mut g = vec![c0(); m * m];
    for a in 0..m {
        for b in 0..m {
            let mut acc = c0();
            for j in 0..n {
                acc += mat[a * n + j] * mat[b * n + j].conj();
            }
            g[a * m + b] = acc;
        }
    }
    g
}

/// Indices that sort `vals` in descending order.
fn sorted_desc_indices(vals: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..vals.len()).collect();
    idx.sort_by(|&i, &j| {
        vals[j]
            .partial_cmp(&vals[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx
}

/// Cyclic Jacobi eigenvalue iteration for a complex Hermitian `n × n` matrix.
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvectors are stored column-
/// major-by-column in a row-major `(n × n)` buffer: column `j` of the result is
/// `evecs[row * n + j]`, and `A · v_j = λ_j v_j`.
///
/// The complex Hermitian Jacobi rotation zeroes the off-diagonal pair `(p, q)`
/// using a phase rotation followed by a real Jacobi (Givens) rotation; see
/// Golub & Van Loan, *Matrix Computations*, §8.5. This mirrors the small-matrix
/// eigensolver style used in [`crate::density::metrics`].
fn jacobi_hermitian(a_in: &[Complex32], n: usize) -> QuantumResult<(Vec<f32>, Vec<Complex32>)> {
    if a_in.len() != n * n {
        return Err(QuantumError::DimensionMismatch {
            expected: n * n,
            got: a_in.len(),
        });
    }
    let mut a = a_in.to_vec();
    // Eigenvector accumulator, initialised to identity.
    let mut v = vec![c0(); n * n];
    for i in 0..n {
        v[i * n + i] = Complex32::new(1.0, 0.0);
    }
    if n == 1 {
        return Ok((vec![a[0].re], v));
    }

    let max_sweeps = 100usize;
    for _sweep in 0..max_sweeps {
        // Off-diagonal Frobenius norm.
        let mut off = 0.0f32;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p * n + q].norm_sqr();
            }
        }
        if off.sqrt() < 1e-12 {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.norm() < 1e-18 {
                    continue;
                }
                let app = a[p * n + p].re;
                let aqq = a[q * n + q].re;

                // Phase φ so that e^{-iφ} a_pq is real-positive.
                let phi = apq.im.atan2(apq.re);
                let (sin_phi, cos_phi) = phi.sin_cos();
                let e_neg = Complex32::new(cos_phi, -sin_phi); // e^{-iφ}
                let e_pos = Complex32::new(cos_phi, sin_phi); // e^{+iφ}
                let abs_apq = apq.norm();

                // Real Jacobi angle θ for the real symmetric problem
                //   [[app, |a_pq|], [|a_pq|, aqq]].
                let tau = (aqq - app) / (2.0 * abs_apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let cos_t = 1.0 / (1.0 + t * t).sqrt();
                let sin_t = t * cos_t;

                // Combined rotation columns: the unitary acting on (p, q) is
                //   J = [[ c,        s·e^{-iφ} ],
                //        [ -s·e^{+iφ}, c        ]].
                let c = cos_t;
                let s_em = sin_phi_combine(sin_t, e_neg);
                let s_ep = sin_phi_combine(sin_t, e_pos);

                // Apply J on the left (rows p, q) and Jᴴ on the right
                // (cols p, q) to A.
                apply_jacobi_rotation(&mut a, n, p, q, c, s_em, s_ep);
                // Accumulate eigenvectors: V = V · Jᴴ (rotate columns p, q).
                accumulate_eigenvectors(&mut v, n, p, q, c, s_em, s_ep);
            }
        }
    }

    let eigenvalues: Vec<f32> = (0..n).map(|i| a[i * n + i].re).collect();
    Ok((eigenvalues, v))
}

#[inline]
fn sin_phi_combine(sin_t: f32, e: Complex32) -> Complex32 {
    Complex32::new(sin_t * e.re, sin_t * e.im)
}

/// Apply the Hermitian Jacobi similarity `A ← J A Jᴴ` for the rotation acting on
/// the `(p, q)` 2×2 subspace with
///   J = [[c, s_em], [-s_ep, c]]   (s_em = s·e^{-iφ}, s_ep = s·e^{+iφ}).
fn apply_jacobi_rotation(
    a: &mut [Complex32],
    n: usize,
    p: usize,
    q: usize,
    c: f32,
    s_em: Complex32,
    s_ep: Complex32,
) {
    let cc = Complex32::new(c, 0.0);
    // Left multiply: rows p and q. row_p' = c·row_p + s_em·row_q
    //                               row_q' = -s_ep·row_p + c·row_q
    for j in 0..n {
        let rp = a[p * n + j];
        let rq = a[q * n + j];
        a[p * n + j] = cc * rp + s_em * rq;
        a[q * n + j] = -s_ep * rp + cc * rq;
    }
    // Right multiply by Jᴴ: cols p and q.
    // Jᴴ = [[c, -conj(s_ep)],[conj(s_em), c]]  acting on columns.
    // col_p' = c·col_p + conj(s_em)·col_q
    // col_q' = -conj(s_ep)·col_p + c·col_q
    let s_em_c = s_em.conj();
    let s_ep_c = s_ep.conj();
    for i in 0..n {
        let cp = a[i * n + p];
        let cq = a[i * n + q];
        a[i * n + p] = cc * cp + s_em_c * cq;
        a[i * n + q] = -s_ep_c * cp + cc * cq;
    }
}

/// Accumulate `V ← V · Jᴴ` (rotate eigenvector columns p and q).
fn accumulate_eigenvectors(
    v: &mut [Complex32],
    n: usize,
    p: usize,
    q: usize,
    c: f32,
    s_em: Complex32,
    s_ep: Complex32,
) {
    let cc = Complex32::new(c, 0.0);
    let s_em_c = s_em.conj();
    let s_ep_c = s_ep.conj();
    for i in 0..n {
        let cp = v[i * n + p];
        let cq = v[i * n + q];
        v[i * n + p] = cc * cp + s_em_c * cq;
        v[i * n + q] = -s_ep_c * cp + cc * cq;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::hadamard::gate_h;
    use crate::gates::parametric::gate_ry;
    use crate::gates::pauli::{gate_x, gate_z};
    use crate::handle::LcgRng;
    use crate::statevec::apply_1q::{apply_1q_controlled, apply_1q_inplace};
    use crate::statevec::state::StateVector;

    fn cfg(n: usize, chi: usize) -> MpsConfig {
        MpsConfig::new(n, chi, 1e-12).expect("test config parameters are valid")
    }

    fn c(re: f32, im: f32) -> Complex32 {
        Complex32::new(re, im)
    }

    /// 4×4 CNOT with control = q (more significant), target = q+1.
    /// Row/col index = 2*bit(ctrl) + bit(tgt).
    fn cnot_4x4() -> [[Complex32; 4]; 4] {
        let o = c(1.0, 0.0);
        let z = c0();
        // |00>->|00>, |01>->|01>, |10>->|11>, |11>->|10>
        [[o, z, z, z], [z, o, z, z], [z, z, z, o], [z, z, o, z]]
    }

    fn max_abs_diff(a: &[Complex32], b: &[Complex32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).norm())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn t01_product_state_is_zero_ket() {
        let mps = MatrixProductState::new_zero_state(cfg(3, 4))
            .expect("valid config produces zero state");
        let sv = mps.to_statevector().expect("MPS converts to statevector");
        assert!((sv.amps[0].re - 1.0).abs() < 1e-6);
        for a in &sv.amps[1..] {
            assert!(a.norm() < 1e-6);
        }
    }

    #[test]
    fn t02_pauli_x_flips_amplitude() {
        let mut mps = MatrixProductState::new_zero_state(cfg(1, 2))
            .expect("valid config produces zero state");
        mps.apply_1q(&gate_x(), 0)
            .expect("qubit 0 is in range for 1-qubit MPS");
        let sv = mps.to_statevector().expect("MPS converts to statevector");
        assert!(sv.amps[0].norm() < 1e-6);
        assert!((sv.amps[1].re - 1.0).abs() < 1e-6);
    }

    #[test]
    fn t03_hadamard_uniform_superposition() {
        let mut mps = MatrixProductState::new_zero_state(cfg(1, 2))
            .expect("valid config produces zero state");
        mps.apply_1q(&gate_h(), 0)
            .expect("qubit 0 is in range for 1-qubit MPS");
        let sv = mps.to_statevector().expect("MPS converts to statevector");
        let inv = std::f32::consts::FRAC_1_SQRT_2;
        assert!((sv.amps[0].re - inv).abs() < 1e-5);
        assert!((sv.amps[1].re - inv).abs() < 1e-5);
    }

    #[test]
    fn t04_cnot_on_plus_zero_makes_bell_bond2() {
        // |+0>: H on qubit 0, then CNOT(0->1). With qubit 0 the control and the
        // less-significant index in the state vector, build the gate in the
        // (q=0 control MSB-of-pair, q+1=1 target) convention.
        let mut mps = MatrixProductState::new_zero_state(cfg(2, 4))
            .expect("valid config produces zero state");
        mps.apply_1q(&gate_h(), 0)
            .expect("qubit 0 is in range for 2-qubit MPS");
        mps.apply_2q(&cnot_4x4(), 0, 1)
            .expect("qubits 0 and 1 are adjacent and in range");
        let sv = mps.to_statevector().expect("MPS converts to statevector");
        let inv = std::f32::consts::FRAC_1_SQRT_2;
        // Bell |00> + |11>.
        assert!((sv.amps[0].re - inv).abs() < 1e-5, "amps={:?}", sv.amps);
        assert!((sv.amps[3].re - inv).abs() < 1e-5, "amps={:?}", sv.amps);
        assert!(sv.amps[1].norm() < 1e-5);
        assert!(sv.amps[2].norm() < 1e-5);
        // Bond between the two sites must be 2.
        assert_eq!(mps.bond_dims()[0], 2);
    }

    #[test]
    fn t05_full_chi_matches_statevector_random_circuit() {
        // Build the same Clifford+Ry circuit on both an MPS (full χ) and a
        // dense StateVector, and compare amplitudes to ~1e-4.
        let n = 5usize;
        let chi = 1usize << n; // full bond, no truncation
        let mut mps = MatrixProductState::new_zero_state(cfg(n, chi))
            .expect("valid config produces zero state");
        let mut sv =
            StateVector::new_zero_state(n).expect("valid qubit count produces zero statevector");

        let mut rng = LcgRng::new(12345);
        let angles = [0.3f32, 1.1, -0.7, 2.0, 0.45, -1.3];
        // Layer of Ry + H on each qubit.
        for (q, &ang) in angles.iter().take(n).enumerate() {
            let ry = gate_ry(ang);
            mps.apply_1q(&ry, q)
                .expect("qubit index q is within the n-qubit MPS");
            apply_1q_inplace(&mut sv, q, &ry)
                .expect("qubit index q is within the n-qubit statevector");
            mps.apply_1q(&gate_h(), q)
                .expect("qubit index q is within the n-qubit MPS");
            apply_1q_inplace(&mut sv, q, &gate_h())
                .expect("qubit index q is within the n-qubit statevector");
        }
        // Ladder of CNOTs on adjacent pairs.
        for q in 0..(n - 1) {
            mps.apply_2q(&cnot_4x4(), q, q + 1)
                .expect("adjacent qubits q and q+1 are in range");
            // Equivalent dense CNOT(control=q, target=q+1).
            apply_1q_controlled(&mut sv, q, q + 1, &gate_x())
                .expect("adjacent qubits q and q+1 are valid for controlled gate");
        }
        // A second Ry layer to entangle further.
        for (q, &ang) in angles.iter().take(n).enumerate() {
            let ry = gate_ry(ang * 0.5 + 0.2);
            mps.apply_1q(&ry, q)
                .expect("qubit index q is within the n-qubit MPS");
            apply_1q_inplace(&mut sv, q, &ry)
                .expect("qubit index q is within the n-qubit statevector");
        }
        let _ = &mut rng;

        let mps_sv = mps.to_statevector().expect("MPS converts to statevector");
        let diff = max_abs_diff(&mps_sv.amps, &sv.amps);
        assert!(diff < 1e-4, "max amplitude diff {diff}");
    }

    #[test]
    fn t06_truncation_reduces_bond() {
        // Create a maximally entangling pair, then re-split with χ = 1 to force
        // truncation of the bond from 2 down to 1.
        let mut mps = MatrixProductState::new_zero_state(cfg(2, 1))
            .expect("valid config produces zero state");
        mps.apply_1q(&gate_h(), 0)
            .expect("qubit 0 is in range for 2-qubit MPS");
        // χ = 1 ⇒ the Bell bond (rank 2) is truncated to 1.
        mps.apply_2q(&cnot_4x4(), 0, 1)
            .expect("qubits 0 and 1 are adjacent and in range");
        assert_eq!(mps.bond_dims()[0], 1, "bond should be truncated to χ=1");
    }

    #[test]
    fn t07_expectation_z_matches_statevector() {
        let n = 3usize;
        let chi = 1usize << n;
        let mut mps = MatrixProductState::new_zero_state(cfg(n, chi))
            .expect("valid config produces zero state");
        let mut sv =
            StateVector::new_zero_state(n).expect("valid qubit count produces zero statevector");
        let angs = [0.6f32, -1.2, 0.9];
        for (q, &a) in angs.iter().enumerate() {
            let g = gate_ry(a);
            mps.apply_1q(&g, q)
                .expect("qubit index q is within the n-qubit MPS");
            apply_1q_inplace(&mut sv, q, &g)
                .expect("qubit index q is within the n-qubit statevector");
        }
        for q in 0..n {
            let z_mps = mps
                .expectation_z(q)
                .expect("qubit index q is within the n-qubit MPS");
            // ⟨Z_q⟩ from statevector = Σ |amp|² (−1)^{bit q}.
            let mask = 1usize << q;
            let z_sv: f32 = sv
                .amps
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    let sign = if i & mask != 0 { -1.0 } else { 1.0 };
                    sign * a.norm_sqr()
                })
                .sum();
            assert!((z_mps - z_sv).abs() < 1e-4, "q={q} mps={z_mps} sv={z_sv}");
        }
    }

    #[test]
    fn t08_unitary_gates_preserve_norm() {
        let mut mps = MatrixProductState::new_zero_state(cfg(3, 8))
            .expect("n=3 and chi=8 are valid MPS configuration parameters");
        mps.apply_1q(&gate_h(), 0)
            .expect("qubit 0 is within the 3-qubit MPS range");
        mps.apply_2q(&cnot_4x4(), 0, 1)
            .expect("qubits 0 and 1 are adjacent and both within the 3-qubit MPS range");
        mps.apply_1q(&gate_ry(0.8), 2)
            .expect("qubit 2 is within the 3-qubit MPS range");
        mps.apply_2q(&cnot_4x4(), 1, 2)
            .expect("qubits 1 and 2 are adjacent and both within the 3-qubit MPS range");
        let nrm = mps.norm();
        assert!((nrm - 1.0).abs() < 1e-4, "norm={nrm}");
    }

    #[test]
    fn t09_svd_reconstruction() {
        // Random-ish 4×3 complex matrix; check U Σ Vᴴ ≈ M.
        let m = 4usize;
        let n = 3usize;
        let mut mat = vec![c0(); m * n];
        let mut rng = LcgRng::new(99);
        for v in &mut mat {
            *v = c(rng.next_f32() - 0.5, rng.next_f32() - 0.5);
        }
        let svd = svd_dense(&mat, m, n)
            .expect("mat has exactly m*n=12 elements and m=4, n=3 are both positive");
        let k = svd.singular_values.len();
        // Reconstruct.
        let mut recon = vec![c0(); m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = c0();
                for t in 0..k {
                    acc += svd.u[i * k + t]
                        * Complex32::new(svd.singular_values[t], 0.0)
                        * svd.v[j * k + t].conj();
                }
                recon[i * n + j] = acc;
            }
        }
        let diff = max_abs_diff(&recon, &mat);
        assert!(diff < 1e-4, "reconstruction diff {diff}");
        // Singular values descending and non-negative.
        for w in svd.singular_values.windows(2) {
            assert!(w[0] >= w[1] - 1e-6);
            assert!(w[1] >= -1e-6);
        }
    }

    #[test]
    fn t10_svd_reconstruction_wide() {
        // Wide matrix m < n exercises the MMᴴ branch.
        let m = 2usize;
        let n = 5usize;
        let mut mat = vec![c0(); m * n];
        let mut rng = LcgRng::new(7);
        for v in &mut mat {
            *v = c(rng.next_f32() - 0.5, rng.next_f32() - 0.5);
        }
        let svd = svd_dense(&mat, m, n)
            .expect("mat has exactly m*n=10 elements and m=2, n=5 are both positive");
        let k = svd.singular_values.len();
        let mut recon = vec![c0(); m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = c0();
                for t in 0..k {
                    acc += svd.u[i * k + t]
                        * Complex32::new(svd.singular_values[t], 0.0)
                        * svd.v[j * k + t].conj();
                }
                recon[i * n + j] = acc;
            }
        }
        let diff = max_abs_diff(&recon, &mat);
        assert!(diff < 1e-4, "wide reconstruction diff {diff}");
    }

    #[test]
    fn t11_non_adjacent_apply_2q_errors() {
        let mut mps = MatrixProductState::new_zero_state(cfg(3, 4))
            .expect("n=3 and chi=4 are valid MPS configuration parameters");
        assert!(mps.apply_2q(&cnot_4x4(), 0, 2).is_err());
    }

    #[test]
    fn t12_bond_never_exceeds_chi() {
        let n = 6usize;
        let chi = 3usize;
        let mut mps = MatrixProductState::new_zero_state(cfg(n, chi))
            .expect("n=6 and chi=3 are valid MPS configuration parameters");
        for q in 0..n {
            mps.apply_1q(&gate_h(), q)
                .expect("q iterates over 0..n so it is always within the n-qubit MPS range");
        }
        for q in 0..(n - 1) {
            mps.apply_2q(&cnot_4x4(), q, q + 1)
                .expect("q < n-1 so both q and q+1 are within range and adjacent");
            mps.apply_2q(&cnot_4x4(), q, q + 1)
                .expect("q < n-1 so both q and q+1 are within range and adjacent");
        }
        for b in mps.bond_dims() {
            assert!(b <= chi, "bond {b} exceeds χ={chi}");
        }
    }

    #[test]
    fn t13_product_state_bonds_all_one() {
        let mps = MatrixProductState::new_zero_state(cfg(5, 4))
            .expect("n=5 and chi=4 are valid MPS configuration parameters");
        for b in mps.bond_dims() {
            assert_eq!(b, 1);
        }
    }

    #[test]
    fn t14_out_of_range_qubit_errors() {
        let mut mps = MatrixProductState::new_zero_state(cfg(2, 4))
            .expect("n=2 and chi=4 are valid MPS configuration parameters");
        assert!(mps.apply_1q(&gate_x(), 5).is_err());
        assert!(mps.expectation_z(9).is_err());
    }

    #[test]
    fn t15_zero_qubits_config_errors() {
        assert!(MpsConfig::new(0, 4, 1e-12).is_err());
        assert!(MpsConfig::new(2, 0, 1e-12).is_err());
    }

    #[test]
    fn t16_normalize_restores_unit_norm() {
        let mut mps = MatrixProductState::new_zero_state(cfg(2, 4))
            .expect("n=2 and chi=4 are valid MPS configuration parameters");
        mps.apply_1q(&gate_h(), 0)
            .expect("qubit 0 is within the 2-qubit MPS range");
        // Scale the first site to break normalization.
        if let Some(first) = mps.sites.first_mut() {
            for v in &mut first.data {
                *v *= 3.0;
            }
        }
        assert!((mps.norm() - 3.0).abs() < 1e-4);
        mps.normalize();
        assert!((mps.norm() - 1.0).abs() < 1e-4, "norm={}", mps.norm());
    }

    #[test]
    fn t17_expectation_z_on_basis_states() {
        // |0> ⇒ +1, |1> ⇒ −1.
        let mut mps = MatrixProductState::new_zero_state(cfg(1, 2))
            .expect("n=1 and chi=2 are valid MPS configuration parameters");
        assert!(
            (mps.expectation_z(0)
                .expect("qubit 0 is the only qubit in this 1-qubit MPS so it is always in range")
                - 1.0)
                .abs()
                < 1e-5
        );
        mps.apply_1q(&gate_x(), 0)
            .expect("qubit 0 is the only qubit in this 1-qubit MPS so it is always in range");
        assert!(
            (mps.expectation_z(0)
                .expect("qubit 0 is the only qubit in this 1-qubit MPS so it is always in range")
                + 1.0)
                .abs()
                < 1e-5
        );
    }

    #[test]
    fn t18_gate_z_phase_then_statevector() {
        // Apply H then Z on qubit 0 of a 2-qubit register; compare to dense.
        let mut mps = MatrixProductState::new_zero_state(cfg(2, 4))
            .expect("n=2 and chi=4 are valid MPS configuration parameters");
        let mut sv = StateVector::new_zero_state(2)
            .expect("n=2 is a valid positive qubit count for a statevector");
        mps.apply_1q(&gate_h(), 1)
            .expect("qubit 1 is within the 2-qubit MPS range");
        apply_1q_inplace(&mut sv, 1, &gate_h())
            .expect("qubit 1 is within the 2-qubit statevector range");
        mps.apply_1q(&gate_z(), 1)
            .expect("qubit 1 is within the 2-qubit MPS range");
        apply_1q_inplace(&mut sv, 1, &gate_z())
            .expect("qubit 1 is within the 2-qubit statevector range");
        let mps_sv = mps
            .to_statevector()
            .expect("MPS was constructed with valid boundary bonds so contraction must succeed");
        let diff = max_abs_diff(&mps_sv.amps, &sv.amps);
        assert!(diff < 1e-5, "diff={diff}");
    }
}
