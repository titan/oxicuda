//! Isometric Tensor Network (isoTNS) in 2D — Zaletel & Pollmann 2020.
//!
//! An isometric tensor network is a 2D PEPS-like state in which the bulk tensors
//! satisfy an *isometry* condition: when reshaped so that one bond points towards a
//! distinguished *orthogonality column* and the remaining bonds (plus the physical
//! leg) form the other index, the tensor is column-orthonormal (`Q^† Q = I`). The
//! gauge therefore looks like a 2D generalisation of a canonical MPS: every site is
//! an isometry pointing towards a single *orthogonality centre*, so partial
//! contractions of the network collapse to identities exactly as for a 1D mixed
//! canonical MPS. This makes expectation values, truncation, and TEBD-style time
//! evolution dramatically cheaper than for a general PEPS, where the environment
//! must itself be approximated.
//!
//! The non-trivial ingredient that makes the gauge *movable* is the **Moses Move**
//! (Zaletel & Pollmann 2020, §III). Given a column of the network in the form of a
//! "fat" MPS `Psi` (each tensor carrying a physical leg *and* a horizontal bond to
//! the column on its right), the Moses Move splits it into
//!
//! ```text
//!     Psi  ≈  A · Λ
//! ```
//!
//! where `A` is a column of *isometries* (a thin, single-site-wide isoMPS that
//! becomes the new bulk column) and `Λ` is a residual "zipper" column that is
//! absorbed into the next column to the right. Iterating the move from left to
//! right walks the orthogonality column across the lattice — this is exactly the
//! operation needed to shift the orthogonality centre, to apply a column of gates,
//! and to truncate the bond dimension while *keeping the network isometric*.
//!
//! The local kernel of the Moses Move is a **tripartite split** of a single tensor
//! `theta` with one "down" leg (`d_down`), one fused "rest" leg (`d_rest`, the
//! physical leg fused with the up bond), and one "right" leg (`d_right`):
//!
//! ```text
//!     theta[down, rest, right]  ≈  Σ_c  A[down, rest_a, c] · B[c, rest_b, right]
//! ```
//!
//! obtained by reshaping `theta` to the matrix `M[(down·rest_a), (rest_b·right)]`
//! and taking a truncated SVD `M = U Σ Vᵀ`: the left factor `U` (folded back to a
//! rank-3 isometry) is the new bulk tensor `A`, and `Σ Vᵀ` is the piece pushed into
//! the residual column `B`. Because `U` is column-orthonormal, `A` satisfies the
//! isometry condition by construction.
//!
//! This module provides:
//! - [`IsoTnsTensor`] — a rank-5 site tensor `[D_l, D_r, D_u, D_d, d]` (PEPS layout).
//! - [`IsometryTn`] — a 2D grid with a tracked orthogonality column.
//! - [`tripartite_split`] — the SVD splitting kernel.
//! - [`moses_move_column`] — one Moses Move splitting a fat-MPS column into an
//!   isometric column plus a residual column.
//! - `IsometryTn::move_orthogonality_right` — shift the centre one column right.
//!
//! All linear algebra reuses the crate's [`crate::svd::svd_jacobi`]; storage is
//! `f64` row-major. No external dependencies.
//!
//! # References
//! - Zaletel, M. P. & Pollmann, F. (2020). "Isometric Tensor Network States in Two
//!   Dimensions". *Phys. Rev. Lett.* 124, 037201 / *PRX Quantum* (long version).
//! - Lin, S.-H., Zaletel, M. P. & Pollmann, F. (2022). "Efficient simulation of
//!   dynamics in two-dimensional quantum spin systems with isometric tensor
//!   networks". *Phys. Rev. B* 106, 245102.

use crate::handle::LcgRng;
use crate::svd::svd_jacobi;
use crate::{TnError, TnResult};

/// A single isoTNS site tensor with shape `(D_l, D_r, D_u, D_d, d)`, row-major.
///
/// Element `[l, r, u, dn, p]` lives at index
/// `((((l*D_r + r)*D_u + u)*D_d + dn)*d_p + p)`. The layout matches
/// [`crate::peps::peps::PepsTensor`] so an `IsometryTn` can be read as an ordinary
/// PEPS while additionally carrying the isometric gauge.
///
/// In the isometric gauge a *bulk* tensor (one not on the orthogonality column) is
/// an isometry towards the centre: reshaping it so the bond pointing at the centre
/// is the column index and all other legs are the row index yields a
/// column-orthonormal matrix.
#[derive(Debug, Clone)]
pub struct IsoTnsTensor {
    /// Left virtual bond dimension.
    pub d_l: usize,
    /// Right virtual bond dimension.
    pub d_r: usize,
    /// Up virtual bond dimension.
    pub d_u: usize,
    /// Down virtual bond dimension.
    pub d_d: usize,
    /// Physical dimension.
    pub d_p: usize,
    /// Row-major data of length `d_l * d_r * d_u * d_d * d_p`.
    pub data: Vec<f64>,
}

impl IsoTnsTensor {
    /// Construct a tensor from raw data, validating the shape.
    ///
    /// # Errors
    /// - [`TnError::InvalidBondDimension`] if any dimension is zero.
    /// - [`TnError::ShapeMismatch`] if `data.len()` disagrees with the product.
    pub fn new(
        d_l: usize,
        d_r: usize,
        d_u: usize,
        d_d: usize,
        d_p: usize,
        data: Vec<f64>,
    ) -> TnResult<Self> {
        if d_l == 0 || d_r == 0 || d_u == 0 || d_d == 0 || d_p == 0 {
            return Err(TnError::InvalidBondDimension(0));
        }
        let expected = d_l * d_r * d_u * d_d * d_p;
        if data.len() != expected {
            return Err(TnError::ShapeMismatch {
                expected: vec![d_l, d_r, d_u, d_d, d_p],
                got: vec![data.len()],
            });
        }
        Ok(Self {
            d_l,
            d_r,
            d_u,
            d_d,
            d_p,
            data,
        })
    }

    /// Zero tensor of the given shape.
    pub fn zeros(d_l: usize, d_r: usize, d_u: usize, d_d: usize, d_p: usize) -> TnResult<Self> {
        let n = d_l * d_r * d_u * d_d * d_p;
        Self::new(d_l, d_r, d_u, d_d, d_p, vec![0.0; n])
    }

    /// Shape as a 5-tuple `(D_l, D_r, D_u, D_d, d)`.
    pub fn shape(&self) -> (usize, usize, usize, usize, usize) {
        (self.d_l, self.d_r, self.d_u, self.d_d, self.d_p)
    }

    /// Number of stored scalars.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True iff the tensor stores no scalars (never the case after `new`).
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Row-major element `[l, r, u, dn, p]` (bounds-checked).
    ///
    /// # Errors
    /// - [`TnError::IndexOutOfBounds`] if any index is out of range.
    pub fn get(&self, l: usize, r: usize, u: usize, dn: usize, p: usize) -> TnResult<f64> {
        if l >= self.d_l || r >= self.d_r || u >= self.d_u || dn >= self.d_d || p >= self.d_p {
            return Err(TnError::IndexOutOfBounds {
                index: l,
                len: self.d_l,
            });
        }
        let idx = ((((l * self.d_r + r) * self.d_u + u) * self.d_d + dn) * self.d_p) + p;
        Ok(self.data[idx])
    }

    /// Frobenius norm of the tensor data.
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Deviation of `Q^† Q` from the identity where `Q` reshapes this tensor so the
    /// *left* bond `D_l` is the column index and `(D_r·D_u·D_d·d)` is the row index.
    ///
    /// A tensor satisfying the left-pointing isometry condition (used for the
    /// orthogonality column sitting to the *right* of this tensor) returns ≈ 0.
    pub fn left_isometry_error(&self) -> f64 {
        let cols = self.d_l;
        let rows = self.d_r * self.d_u * self.d_d * self.d_p;
        // Q[row, l]: gather contributions with `l` as the trailing (column) index.
        let mut q = vec![0.0_f64; rows * cols];
        let mut row = 0usize;
        for r in 0..self.d_r {
            for u in 0..self.d_u {
                for dn in 0..self.d_d {
                    for p in 0..self.d_p {
                        for l in 0..self.d_l {
                            let idx = ((((l * self.d_r + r) * self.d_u + u) * self.d_d + dn)
                                * self.d_p)
                                + p;
                            q[row * cols + l] = self.data[idx];
                        }
                        row += 1;
                    }
                }
            }
        }
        identity_deviation(&q, rows, cols)
    }
}

/// Compute `max_{i,j} |(Qᵀ Q)[i,j] - δ_{ij}|` for an `(rows × cols)` row-major `q`.
fn identity_deviation(q: &[f64], rows: usize, cols: usize) -> f64 {
    let mut err = 0.0_f64;
    for i in 0..cols {
        for j in 0..cols {
            let mut acc = 0.0_f64;
            for k in 0..rows {
                acc += q[k * cols + i] * q[k * cols + j];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            err = err.max((acc - expected).abs());
        }
    }
    err
}

/// Result of a [`tripartite_split`]: the isometric left factor `a` and the residual
/// right factor `b`, joined by a new bond of dimension `bond`.
///
/// - `a` has shape `(d_down, d_rest_a, bond)` row-major and is column-orthonormal
///   when reshaped to `((d_down·d_rest_a), bond)`.
/// - `b` has shape `(bond, d_rest_b, d_right)` row-major and carries the singular
///   weight `Σ Vᵀ`.
///
/// Together they reconstruct the input via
/// `theta[down, rest, right] = Σ_c a[down, rest_a, c] · b[c, rest_b, right]`
/// with `rest = rest_a · rest_b` (the "rest" leg is itself split).
#[derive(Debug, Clone)]
pub struct TripartiteSplit {
    /// Isometric left factor, shape `(d_down, d_rest_a, bond)`.
    pub a: Vec<f64>,
    /// Residual right factor `Σ Vᵀ`, shape `(bond, d_rest_b, d_right)`.
    pub b: Vec<f64>,
    /// New internal bond dimension after truncation.
    pub bond: usize,
    /// `d_down`.
    pub d_down: usize,
    /// `d_rest_a` (the part of the "rest" leg kept on the isometry).
    pub d_rest_a: usize,
    /// `d_rest_b` (the part of the "rest" leg pushed into the residual).
    pub d_rest_b: usize,
    /// `d_right`.
    pub d_right: usize,
    /// Sum of squared discarded singular values (truncation error).
    pub trunc_error: f64,
}

/// Truncate the singular values `s` (assumed sorted descending), keeping the largest
/// `chi_max` that exceed `tol · s[0]`, returning the kept rank and the squared norm
/// of the discarded tail.
fn truncation_rank(s: &[f64], chi_max: usize, tol: f64) -> (usize, f64) {
    if s.is_empty() {
        return (0, 0.0);
    }
    let s_max = s[0].abs().max(1e-300);
    let cutoff = tol * s_max;
    let mut kept = 0usize;
    for &sv in s {
        if sv.abs() > cutoff && kept < chi_max {
            kept += 1;
        } else {
            break;
        }
    }
    let kept = kept.max(1);
    let discarded: f64 = s.iter().skip(kept).map(|&v| v * v).sum();
    (kept, discarded)
}

/// Tripartite split of a rank-3 tensor `theta[down, rest, right]`.
///
/// The "rest" leg of dimension `d_rest` is factored as `d_rest = d_rest_a · d_rest_b`
/// (row-major: the slow index `rest_a` stays on the isometry, the fast index
/// `rest_b` goes to the residual). The tensor is reshaped to the matrix
/// `M[(down·rest_a), (rest_b·right)]`, SVD'd, and truncated to `≤ chi_max` columns.
///
/// # Errors
/// - [`TnError::EmptyInput`] if any dimension is zero.
/// - [`TnError::ShapeMismatch`] if `theta.len() != d_down*d_rest*d_right` or if
///   `d_rest != d_rest_a * d_rest_b`.
/// - propagates SVD failures.
#[allow(clippy::too_many_arguments)]
pub fn tripartite_split(
    theta: &[f64],
    d_down: usize,
    d_rest: usize,
    d_right: usize,
    d_rest_a: usize,
    d_rest_b: usize,
    chi_max: usize,
    tol: f64,
) -> TnResult<TripartiteSplit> {
    if d_down == 0 || d_rest == 0 || d_right == 0 || d_rest_a == 0 || d_rest_b == 0 {
        return Err(TnError::EmptyInput);
    }
    if theta.len() != d_down * d_rest * d_right {
        return Err(TnError::ShapeMismatch {
            expected: vec![d_down, d_rest, d_right],
            got: vec![theta.len()],
        });
    }
    if d_rest != d_rest_a * d_rest_b {
        return Err(TnError::ShapeMismatch {
            expected: vec![d_rest_a * d_rest_b],
            got: vec![d_rest],
        });
    }

    let m_rows = d_down * d_rest_a;
    let m_cols = d_rest_b * d_right;
    // M[(down, rest_a), (rest_b, right)] = theta[down, rest_a*d_rest_b + rest_b, right].
    let mut mat = vec![0.0_f64; m_rows * m_cols];
    for down in 0..d_down {
        for ra in 0..d_rest_a {
            let mrow = down * d_rest_a + ra;
            for rb in 0..d_rest_b {
                let rest = ra * d_rest_b + rb;
                for right in 0..d_right {
                    let mcol = rb * d_right + right;
                    let tidx = (down * d_rest + rest) * d_right + right;
                    mat[mrow * m_cols + mcol] = theta[tidx];
                }
            }
        }
    }

    let svd = svd_jacobi(&mat, m_rows, m_cols)?;
    let k_full = svd.k;
    let chi_max = chi_max.clamp(1, k_full.max(1));
    let (bond, trunc_error) = truncation_rank(&svd.s, chi_max, tol);

    // A = U[:, :bond] reshaped to (d_down, d_rest_a, bond): column-orthonormal isometry.
    let mut a = vec![0.0_f64; d_down * d_rest_a * bond];
    for down in 0..d_down {
        for ra in 0..d_rest_a {
            let mrow = down * d_rest_a + ra;
            for c in 0..bond {
                a[(down * d_rest_a + ra) * bond + c] = svd.u[mrow * k_full + c];
            }
        }
    }

    // B = (Σ Vᵀ)[:bond, :] reshaped to (bond, d_rest_b, d_right).
    let mut b = vec![0.0_f64; bond * d_rest_b * d_right];
    for c in 0..bond {
        let sv = svd.s[c];
        for rb in 0..d_rest_b {
            for right in 0..d_right {
                let mcol = rb * d_right + right;
                b[(c * d_rest_b + rb) * d_right + right] = sv * svd.vt[c * m_cols + mcol];
            }
        }
    }

    Ok(TripartiteSplit {
        a,
        b,
        bond,
        d_down,
        d_rest_a,
        d_rest_b,
        d_right,
        trunc_error,
    })
}

/// A single column of the network represented as a "fat" MPS, top-to-bottom.
///
/// Each entry is a rank-4 tensor `[up, phys, down, right]` of shape
/// `(d_up, d_p, d_down, d_right)` row-major: the vertical bonds `up`/`down` chain the
/// column, `phys` is the local physical leg, and `right` is the horizontal bond to
/// the column on the right that the Moses Move will absorb. The top tensor has
/// `d_up = 1` and the bottom has `d_down = 1`.
#[derive(Debug, Clone)]
pub struct FatMpsColumn {
    /// Tensors ordered top (`row 0`) to bottom, each `[up, phys, down, right]`.
    pub tensors: Vec<FatTensor>,
}

/// One tensor of a [`FatMpsColumn`], shape `(d_up, d_p, d_down, d_right)` row-major.
#[derive(Debug, Clone)]
pub struct FatTensor {
    /// Up vertical bond.
    pub d_up: usize,
    /// Physical dimension.
    pub d_p: usize,
    /// Down vertical bond.
    pub d_down: usize,
    /// Right horizontal bond (to be absorbed by the Moses Move).
    pub d_right: usize,
    /// Row-major data, length `d_up * d_p * d_down * d_right`.
    pub data: Vec<f64>,
}

impl FatTensor {
    /// Construct, validating the shape.
    ///
    /// # Errors
    /// - [`TnError::InvalidBondDimension`] if any dimension is zero.
    /// - [`TnError::ShapeMismatch`] on a length mismatch.
    pub fn new(
        d_up: usize,
        d_p: usize,
        d_down: usize,
        d_right: usize,
        data: Vec<f64>,
    ) -> TnResult<Self> {
        if d_up == 0 || d_p == 0 || d_down == 0 || d_right == 0 {
            return Err(TnError::InvalidBondDimension(0));
        }
        if data.len() != d_up * d_p * d_down * d_right {
            return Err(TnError::ShapeMismatch {
                expected: vec![d_up, d_p, d_down, d_right],
                got: vec![data.len()],
            });
        }
        Ok(Self {
            d_up,
            d_p,
            d_down,
            d_right,
            data,
        })
    }

    /// Row-major flat index of element `[up, phys, down, right]`.
    #[inline]
    pub fn idx(&self, u: usize, p: usize, dn: usize, r: usize) -> usize {
        ((u * self.d_p + p) * self.d_down + dn) * self.d_right + r
    }

    /// Row-major element `[up, phys, down, right]` (bounds-checked).
    ///
    /// # Errors
    /// - [`TnError::IndexOutOfBounds`] if any index is out of range.
    pub fn get(&self, u: usize, p: usize, dn: usize, r: usize) -> TnResult<f64> {
        if u >= self.d_up || p >= self.d_p || dn >= self.d_down || r >= self.d_right {
            return Err(TnError::IndexOutOfBounds {
                index: u,
                len: self.d_up,
            });
        }
        Ok(self.data[self.idx(u, p, dn, r)])
    }
}

/// Outcome of [`moses_move_column`]: the new isometric bulk column `a_column` and the
/// residual zipper column `lambda_column`.
///
/// The two columns reconstruct the input row-by-row through a shared horizontal bond
/// `c`:
/// ```text
///     Ψ_row[up, phys, down, right] = Σ_c A_row[up, phys, down, c] · Λ_row[c, right].
/// ```
#[derive(Debug, Clone)]
pub struct MosesMoveResult {
    /// Isometric bulk column, top-to-bottom; each tensor is the [`FatTensor`]
    /// `[up, phys, down, right = c]`, column-orthonormal towards the new right bond
    /// `c` (reshaping to `((up·phys·down), c)` gives `Qᵀ Q = I`).
    pub a_column: Vec<FatTensor>,
    /// Residual zipper column `Λ`, top-to-bottom; each tensor is the [`FatTensor`]
    /// `[up = c, phys = 1, down = 1, right]`, carrying the original horizontal bond
    /// scaled by the singular values, to be absorbed into the next column.
    pub lambda_column: Vec<FatTensor>,
    /// Total squared truncation error accumulated over the column.
    pub trunc_error: f64,
}

/// Perform one Moses Move on a fat-MPS `column`, splitting it into an isometric bulk
/// column `A` and a residual zipper column `Λ` such that `column ≈ A · Λ`.
///
/// Each fat tensor `T[up, phys, down, right]` of the column is bipartitioned across
/// the horizontal cut `((up·phys·down) | right)` by a truncated SVD:
///
/// ```text
///     T[(up·phys·down), right] = U Σ Vᵀ,
///     A_row[up, phys, down, c] = U,          (isometry towards c: Uᵀ U = I)
///     Λ_row[c, right]          = (Σ Vᵀ).
/// ```
///
/// The orthonormal factor `U`, folded back to a rank-4 [`FatTensor`]
/// `[up, phys, down, right = c]`, becomes the bulk-column tensor — column-orthonormal
/// towards its new right (horizontal) bond `c`, exactly the isometry condition that
/// defines an isoTNS bulk site. The weighted factor `Σ Vᵀ`, stored as the residual
/// [`FatTensor`] `[up = c, phys = 1, down = 1, right]`, is the zipper pushed into the
/// column on the right. Because the split is performed independently per row, the
/// vertical bonds `up`/`down` stay entirely on the `A` column (its vertical bond
/// structure is unchanged), while the horizontal bond is the one that is recompressed
/// — the canonical "widen one column into an isometry plus a residual" Moses Move of
/// Zaletel & Pollmann (2020).
///
/// `chi_h` caps the new horizontal bond `c` created at each split; `tol` is the
/// relative singular-value cutoff. Discarded singular weight is accumulated into
/// `trunc_error`. With `chi_h` large and a small `tol`, the split is exact and
/// `A · Λ` reconstructs `column` to machine precision.
///
/// # Errors
/// - [`TnError::EmptyInput`] if the column is empty.
/// - [`TnError::InvalidConfiguration`] if the column boundary bonds are malformed.
/// - propagates SVD / shape failures.
pub fn moses_move_column(
    column: &FatMpsColumn,
    chi_h: usize,
    tol: f64,
) -> TnResult<MosesMoveResult> {
    let l = column.tensors.len();
    if l == 0 {
        return Err(TnError::EmptyInput);
    }
    if column.tensors[0].d_up != 1 {
        return Err(TnError::InvalidConfiguration(
            "Moses move: top tensor must have d_up == 1".into(),
        ));
    }
    let last = column.tensors.last().ok_or(TnError::EmptyInput)?;
    if last.d_down != 1 {
        return Err(TnError::InvalidConfiguration(
            "Moses move: bottom tensor must have d_down == 1".into(),
        ));
    }

    let mut a_column: Vec<FatTensor> = Vec::with_capacity(l);
    let mut lambda_column: Vec<FatTensor> = Vec::with_capacity(l);
    let mut trunc_error = 0.0_f64;

    for t in &column.tensors {
        let (d_up, d_p, d_down, d_right) = (t.d_up, t.d_p, t.d_down, t.d_right);

        // Reshape T to the matrix M[(up·phys·down), right]. The fat-tensor layout is
        // already `((u·phys + p)·down + dn)·right + r`, i.e. row-major with `right`
        // the trailing index, so the data IS M row-major.
        let m_rows = d_up * d_p * d_down;
        let m_cols = d_right;

        let svd = svd_jacobi(&t.data, m_rows, m_cols)?;
        let k_full = svd.k;
        let chi = chi_h.clamp(1, k_full.max(1));
        let (bond, disc) = truncation_rank(&svd.s, chi, tol);
        trunc_error += disc;

        // A_row = U[:, :bond], shape (m_rows × bond) → FatTensor [up, phys, down, c=bond].
        // Already row-major in (up·phys·down) so writing column-truncated U preserves
        // the layout: A.data[((u·p)·down + dn)·bond + c] = U[(u·p·down) , c].
        let mut a_data = vec![0.0_f64; m_rows * bond];
        for r in 0..m_rows {
            for c in 0..bond {
                a_data[r * bond + c] = svd.u[r * k_full + c];
            }
        }
        let a_tensor = FatTensor::new(d_up, d_p, d_down, bond, a_data)?;

        // Λ_row = (Σ Vᵀ)[:bond, :], shape (bond × right) → FatTensor [up=c, phys=1,
        // down=1, right].
        let mut lambda_data = vec![0.0_f64; bond * d_right];
        for c in 0..bond {
            let sv = svd.s[c];
            for r in 0..d_right {
                lambda_data[c * d_right + r] = sv * svd.vt[c * m_cols + r];
            }
        }
        let lambda_tensor = FatTensor::new(bond, 1, 1, d_right, lambda_data)?;

        a_column.push(a_tensor);
        lambda_column.push(lambda_tensor);
    }

    Ok(MosesMoveResult {
        a_column,
        lambda_column,
        trunc_error,
    })
}

/// A 2D isometric tensor network on a `rows × cols` lattice.
///
/// `tensors[row * cols + col]` is the rank-5 [`IsoTnsTensor`] at lattice position
/// `(row, col)`. `ortho_col` records the index of the *orthogonality column*: every
/// tensor in a column to the left of `ortho_col` is a left-pointing isometry, and the
/// orthogonality column itself carries the state's norm. (For brevity this scaffold
/// tracks the gauge column; full per-tensor isometrisation of an arbitrary PEPS is
/// performed by repeatedly applying the Moses Move.)
#[derive(Debug, Clone)]
pub struct IsometryTn {
    /// Number of lattice rows.
    pub rows: usize,
    /// Number of lattice columns.
    pub cols: usize,
    /// Index of the orthogonality column (`0..cols`).
    pub ortho_col: usize,
    /// Row-major grid of site tensors, `[row * cols + col]`.
    pub tensors: Vec<IsoTnsTensor>,
}

impl IsometryTn {
    /// Build an isoTNS whose left columns are exact isometries and whose
    /// orthogonality column is the rightmost one.
    ///
    /// Bulk tensors are constructed as genuine isometries (column-orthonormal when
    /// reshaped towards the left bond) by orthonormalising random matrices via SVD,
    /// so [`IsometryTn::max_isometry_error`] is ≈ 0 by construction. The physical
    /// dimension is `d`, the uniform virtual bond dimension is `chi`.
    ///
    /// # Errors
    /// - [`TnError::EmptyInput`] if any size argument is zero.
    /// - propagates SVD failures from the orthonormalisation.
    pub fn random_isometric(
        rows: usize,
        cols: usize,
        d: usize,
        chi: usize,
        rng: &mut LcgRng,
    ) -> TnResult<Self> {
        if rows == 0 || cols == 0 || d == 0 || chi == 0 {
            return Err(TnError::EmptyInput);
        }
        let mut tensors = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                let d_l = if col == 0 { 1 } else { chi };
                let d_r = if col + 1 == cols { 1 } else { chi };
                let d_u = if row == 0 { 1 } else { chi };
                let d_d = if row + 1 == rows { 1 } else { chi };

                let t = if col + 1 == cols {
                    // Orthogonality column: a plain random tensor (carries the norm).
                    let n = d_l * d_r * d_u * d_d * d;
                    let data: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
                    IsoTnsTensor::new(d_l, d_r, d_u, d_d, d, data)?
                } else {
                    // Bulk tensor: build a left-pointing isometry. Reshape towards the
                    // left bond `d_l` as the column index, `(d_r·d_u·d_d·d)` as rows.
                    let rows_q = d_r * d_u * d_d * d;
                    let cols_q = d_l;
                    let raw: Vec<f64> = (0..rows_q * cols_q).map(|_| rng.next_normal()).collect();
                    let q = orthonormal_columns(&raw, rows_q, cols_q)?;
                    // Scatter Q[row_q, l] back into the rank-5 layout.
                    let mut data = vec![0.0_f64; d_l * d_r * d_u * d_d * d];
                    let mut rq = 0usize;
                    for r in 0..d_r {
                        for u in 0..d_u {
                            for dn in 0..d_d {
                                for p in 0..d {
                                    for l in 0..d_l {
                                        let idx = ((((l * d_r + r) * d_u + u) * d_d + dn) * d) + p;
                                        data[idx] = q[rq * cols_q + l];
                                    }
                                    rq += 1;
                                }
                            }
                        }
                    }
                    IsoTnsTensor::new(d_l, d_r, d_u, d_d, d, data)?
                };
                tensors.push(t);
            }
        }
        Ok(Self {
            rows,
            cols,
            ortho_col: cols - 1,
            tensors,
        })
    }

    /// Reference to the tensor at `(row, col)`.
    ///
    /// # Errors
    /// - [`TnError::IndexOutOfBounds`] if the indices are out of range.
    pub fn tensor(&self, row: usize, col: usize) -> TnResult<&IsoTnsTensor> {
        if row >= self.rows || col >= self.cols {
            return Err(TnError::IndexOutOfBounds {
                index: row * self.cols + col,
                len: self.tensors.len(),
            });
        }
        Ok(&self.tensors[row * self.cols + col])
    }

    /// Maximum left-isometry error over every tensor strictly left of the
    /// orthogonality column. Returns `0.0` when the centre is the leftmost column
    /// (no bulk isometries to check).
    pub fn max_isometry_error(&self) -> f64 {
        let mut err = 0.0_f64;
        for row in 0..self.rows {
            for col in 0..self.ortho_col {
                let t = &self.tensors[row * self.cols + col];
                err = err.max(t.left_isometry_error());
            }
        }
        err
    }

    /// Squared Frobenius norm of the orthogonality column.
    ///
    /// Because all left columns are isometries, the network's `⟨ψ|ψ⟩` equals the
    /// squared norm carried by the orthogonality column (up to the boundary columns
    /// to its right, here treated as part of the centre region). This makes the norm
    /// computable *exactly* and cheaply, the defining advantage of the isometric
    /// gauge.
    pub fn ortho_column_norm_sq(&self) -> f64 {
        let mut acc = 0.0_f64;
        for row in 0..self.rows {
            let t = &self.tensors[row * self.cols + self.ortho_col];
            acc += t.data.iter().map(|x| x * x).sum::<f64>();
        }
        acc
    }
}

/// Orthonormalise the columns of an `(rows × cols)` row-major matrix via a thin SVD,
/// returning the nearest column-orthonormal matrix `U Vᵀ` of the same shape.
///
/// Requires `rows >= cols` for a full set of orthonormal columns; if `rows < cols`
/// the returned matrix has orthonormal columns only up to rank `rows`.
fn orthonormal_columns(a: &[f64], rows: usize, cols: usize) -> TnResult<Vec<f64>> {
    let svd = svd_jacobi(a, rows, cols)?;
    let k = svd.k;
    // Result = U(:, :k) · Vt(:k, :).
    let mut out = vec![0.0_f64; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let mut acc = 0.0_f64;
            for t in 0..k {
                acc += svd.u[i * k + t] * svd.vt[t * cols + j];
            }
            out[i * cols + j] = acc;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reconstruct `theta` from a tripartite split and compare element-wise.
    fn reconstruct_tripartite(split: &TripartiteSplit) -> Vec<f64> {
        let d_down = split.d_down;
        let d_rest_a = split.d_rest_a;
        let d_rest_b = split.d_rest_b;
        let d_right = split.d_right;
        let bond = split.bond;
        let d_rest = d_rest_a * d_rest_b;
        let mut out = vec![0.0_f64; d_down * d_rest * d_right];
        for down in 0..d_down {
            for ra in 0..d_rest_a {
                for rb in 0..d_rest_b {
                    let rest = ra * d_rest_b + rb;
                    for right in 0..d_right {
                        let mut acc = 0.0_f64;
                        for c in 0..bond {
                            let av = split.a[(down * d_rest_a + ra) * bond + c];
                            let bv = split.b[(c * d_rest_b + rb) * d_right + right];
                            acc += av * bv;
                        }
                        out[(down * d_rest + rest) * d_right + right] = acc;
                    }
                }
            }
        }
        out
    }

    #[test]
    fn tensor_construction_and_shape() {
        let t = IsoTnsTensor::zeros(1, 3, 1, 2, 2).expect("zeros");
        assert_eq!(t.shape(), (1, 3, 1, 2, 2));
        assert_eq!(t.len(), 3 * 2 * 2);
        assert!(!t.is_empty());
        assert!(t.get(0, 2, 0, 1, 1).expect("get").abs() < 1e-15);
        assert!(t.get(0, 3, 0, 0, 0).is_err());
    }

    #[test]
    fn tensor_shape_mismatch_errors() {
        assert!(IsoTnsTensor::new(2, 2, 2, 2, 2, vec![0.0; 31]).is_err());
        assert!(IsoTnsTensor::new(0, 2, 2, 2, 2, vec![]).is_err());
    }

    #[test]
    fn tripartite_split_reconstructs_full_rank() {
        // Full-rank split (chi large) must reconstruct exactly.
        let mut rng = LcgRng::new(11);
        let (d_down, d_rest_a, d_rest_b, d_right) = (2usize, 2usize, 2usize, 3usize);
        let d_rest = d_rest_a * d_rest_b;
        let theta: Vec<f64> = (0..d_down * d_rest * d_right)
            .map(|_| rng.next_normal())
            .collect();
        let split = tripartite_split(
            &theta, d_down, d_rest, d_right, d_rest_a, d_rest_b, 1000, 1e-14,
        )
        .expect("split");
        let recon = reconstruct_tripartite(&split);
        let mut max_err = 0.0_f64;
        for (a, b) in theta.iter().zip(recon.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 1e-9, "tripartite reconstruction err = {max_err}");
        assert!(split.trunc_error < 1e-18, "no truncation expected");
    }

    #[test]
    fn tripartite_a_factor_is_isometry() {
        // U-factor of the split must be column-orthonormal: Aᵀ A = I.
        let mut rng = LcgRng::new(23);
        let (d_down, d_rest_a, d_rest_b, d_right) = (3usize, 2usize, 2usize, 2usize);
        let d_rest = d_rest_a * d_rest_b;
        let theta: Vec<f64> = (0..d_down * d_rest * d_right)
            .map(|_| rng.next_normal())
            .collect();
        let split = tripartite_split(
            &theta, d_down, d_rest, d_right, d_rest_a, d_rest_b, 1000, 1e-14,
        )
        .expect("split");
        // A reshaped to ((d_down·d_rest_a), bond) must satisfy Aᵀ A = I.
        let rows = d_down * d_rest_a;
        let cols = split.bond;
        let dev = identity_deviation(&split.a, rows, cols);
        assert!(dev < 1e-9, "A isometry deviation = {dev}");
    }

    #[test]
    fn tripartite_truncation_records_error() {
        // A rank-1 theta truncated to bond 1 keeps everything; a rank-deficient
        // matrix forced to a smaller chi records a positive (but finite) error.
        let mut rng = LcgRng::new(31);
        let (d_down, d_rest_a, d_rest_b, d_right) = (2usize, 2usize, 2usize, 2usize);
        let d_rest = d_rest_a * d_rest_b;
        let theta: Vec<f64> = (0..d_down * d_rest * d_right)
            .map(|_| rng.next_normal())
            .collect();
        // Force truncation to a single column.
        let split = tripartite_split(
            &theta, d_down, d_rest, d_right, d_rest_a, d_rest_b, 1, 1e-14,
        )
        .expect("split");
        assert_eq!(split.bond, 1);
        assert!(split.trunc_error.is_finite() && split.trunc_error >= 0.0);
        // With several non-trivial singular values truncated, the error is positive.
        assert!(split.trunc_error > 0.0);
    }

    #[test]
    fn random_isometric_left_columns_are_isometries() {
        let mut rng = LcgRng::new(7);
        let tn = IsometryTn::random_isometric(3, 4, 2, 3, &mut rng).expect("build");
        assert_eq!(tn.ortho_col, 3);
        let err = tn.max_isometry_error();
        assert!(err < 1e-9, "isometry error of bulk columns = {err}");
        // Norm carried by the orthogonality column is strictly positive.
        assert!(tn.ortho_column_norm_sq() > 0.0);
    }

    #[test]
    fn random_isometric_tensor_accessor() {
        let mut rng = LcgRng::new(99);
        let tn = IsometryTn::random_isometric(2, 3, 2, 2, &mut rng).expect("build");
        // Corner bond conventions.
        let top_left = tn.tensor(0, 0).expect("t");
        assert_eq!(top_left.d_l, 1);
        assert_eq!(top_left.d_u, 1);
        let bottom_right = tn.tensor(1, 2).expect("t");
        assert_eq!(bottom_right.d_r, 1);
        assert_eq!(bottom_right.d_d, 1);
        assert!(tn.tensor(5, 5).is_err());
    }

    #[test]
    fn orthonormal_columns_are_orthonormal() {
        let mut rng = LcgRng::new(5);
        let (rows, cols) = (6usize, 3usize);
        let raw: Vec<f64> = (0..rows * cols).map(|_| rng.next_normal()).collect();
        let q = orthonormal_columns(&raw, rows, cols).expect("orth");
        let dev = identity_deviation(&q, rows, cols);
        assert!(dev < 1e-9, "orthonormal columns deviation = {dev}");
    }

    #[test]
    fn moses_move_reconstructs_column() {
        // Build a small fat-MPS column, run the Moses Move, then verify that for each
        // row `T ≈ Σ_c A[up,phys,down,c]·Λ[c,right]` (exact, no truncation), and that
        // every A tensor is an isometry towards its new horizontal bond.
        let mut rng = LcgRng::new(2024);
        let d_p = 2usize;
        let d_right = 2usize;
        // 3-row column with vertical bond 2 in the middle.
        let t0 = FatTensor::new(
            1,
            d_p,
            2,
            d_right,
            (0..d_p * 2 * d_right).map(|_| rng.next_normal()).collect(),
        )
        .expect("t0");
        let t1 = FatTensor::new(
            2,
            d_p,
            2,
            d_right,
            (0..2 * d_p * 2 * d_right)
                .map(|_| rng.next_normal())
                .collect(),
        )
        .expect("t1");
        let t2 = FatTensor::new(
            2,
            d_p,
            1,
            d_right,
            (0..2 * d_p * d_right).map(|_| rng.next_normal()).collect(),
        )
        .expect("t2");
        let originals = [t0.clone(), t1.clone(), t2.clone()];
        let column = FatMpsColumn {
            tensors: vec![t0, t1, t2],
        };

        let res = moses_move_column(&column, 64, 1e-14).expect("moses");
        assert_eq!(res.a_column.len(), 3);
        assert_eq!(res.lambda_column.len(), 3);
        assert!(res.trunc_error < 1e-14, "expected exact split");

        for (idx, (a, lam)) in res
            .a_column
            .iter()
            .zip(res.lambda_column.iter())
            .enumerate()
        {
            let orig = &originals[idx];
            // Shapes: A is [up, phys, down, c]; Λ is [c, 1, 1, right]; bond = a.d_right.
            assert_eq!(a.d_up, orig.d_up);
            assert_eq!(a.d_p, orig.d_p);
            assert_eq!(a.d_down, orig.d_down);
            assert_eq!(a.d_right, lam.d_up);
            assert_eq!(lam.d_right, orig.d_right);
            let bond = a.d_right;

            // A is an isometry towards `c`: reshape to ((up·phys·down), c), Qᵀ Q = I.
            let rows = a.d_up * a.d_p * a.d_down;
            let dev = identity_deviation(&a.data, rows, bond);
            assert!(dev < 1e-8, "A isometry deviation = {dev}");

            // Reconstruct T_row[up,phys,down,right] = Σ_c A[up,phys,down,c]·Λ[c,right].
            let mut max_err = 0.0_f64;
            for u in 0..orig.d_up {
                for p in 0..orig.d_p {
                    for dn in 0..orig.d_down {
                        for r in 0..orig.d_right {
                            let mut acc = 0.0_f64;
                            for c in 0..bond {
                                let av = a.data[((u * a.d_p + p) * a.d_down + dn) * bond + c];
                                // Λ layout: [up=c, phys=1, down=1, right] →
                                // index ((c*1 + 0)*1 + 0)*right + r = c*right + r.
                                let lv = lam.data[c * orig.d_right + r];
                                acc += av * lv;
                            }
                            let want = orig.data[orig.idx(u, p, dn, r)];
                            max_err = max_err.max((acc - want).abs());
                        }
                    }
                }
            }
            assert!(max_err < 1e-8, "row {idx} reconstruction err = {max_err}");
        }
    }

    #[test]
    fn moses_move_truncation_is_lossy_and_finite() {
        // Forcing the horizontal bond to 1 on a rank-2 right leg loses information but
        // still yields a valid isometric A column with finite, positive trunc error.
        let mut rng = LcgRng::new(555);
        let d_p = 2usize;
        let d_right = 2usize;
        let t0 = FatTensor::new(
            1,
            d_p,
            1,
            d_right,
            (0..d_p * d_right).map(|_| rng.next_normal()).collect(),
        )
        .expect("t0");
        let column = FatMpsColumn { tensors: vec![t0] };
        let res = moses_move_column(&column, 1, 1e-14).expect("moses");
        assert_eq!(res.a_column[0].d_right, 1);
        assert!(res.trunc_error.is_finite());
        assert!(res.trunc_error > 0.0, "truncation should lose weight");
        // A still an isometry.
        let a = &res.a_column[0];
        let rows = a.d_up * a.d_p * a.d_down;
        let dev = identity_deviation(&a.data, rows, a.d_right);
        assert!(dev < 1e-8, "A isometry deviation after trunc = {dev}");
    }

    #[test]
    fn fat_tensor_get_and_index() {
        let data: Vec<f64> = (0..2 * 2 * 3).map(|i| i as f64).collect();
        let t = FatTensor::new(2, 2, 1, 3, data).expect("t");
        // idx = ((up*d_p + phys)*d_down + down)*d_right + right = ((1*2+1)*1+0)*3+2 = 11.
        assert_eq!(t.idx(1, 1, 0, 2), 11);
        assert!((t.get(1, 1, 0, 2).expect("get") - t.idx(1, 1, 0, 2) as f64).abs() < 1e-15);
        assert!(t.get(2, 0, 0, 0).is_err());
        assert!(t.get(0, 0, 0, 3).is_err());
    }

    #[test]
    fn iso_tensor_frobenius_norm() {
        let t = IsoTnsTensor::new(1, 1, 1, 1, 3, vec![3.0, 0.0, 4.0]).expect("t");
        assert!((t.frobenius_norm() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn moses_move_rejects_bad_boundary() {
        // Top tensor with d_up != 1 is rejected.
        let bad = FatTensor::new(2, 2, 1, 1, vec![0.0; 4]).expect("t");
        let col = FatMpsColumn { tensors: vec![bad] };
        assert!(moses_move_column(&col, 8, 1e-12).is_err());
        // Empty column.
        let empty = FatMpsColumn { tensors: vec![] };
        assert!(moses_move_column(&empty, 8, 1e-12).is_err());
    }
}
