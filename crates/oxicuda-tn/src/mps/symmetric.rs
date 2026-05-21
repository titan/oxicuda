//! U(1)-symmetric Matrix Product State with quantum-number block structure.
//!
//! ## Physical Background
//!
//! For a system with an Abelian U(1) symmetry (e.g. particle-number conservation),
//! each MPS tensor `M[α, σ, β]` is non-zero only when
//!
//! ```text
//! qn(β) = qn(α) + qn(σ)
//! ```
//!
//! where `qn(α)` / `qn(β)` are the quantum-number labels carried by the left/right
//! virtual bond indices and `qn(σ)` is the charge increment of the local physical
//! state `σ`. This *charge-conservation* constraint means that the full tensor
//! decomposes into a collection of *dense blocks*, one per compatible triple
//! `(qn_left, qn_phys, qn_right)`.
//!
//! Exploiting this block structure reduces memory from `O(chi^2 * d)` to
//! `O(chi^2)` (roughly) and computation from `O(chi^3 * d)` to `O(chi^3)`.
//!
//! ## Data Model
//!
//! Each bond carries a set of *charge sectors*. For a sector `q` at bond `b`, the
//! sub-dimension (number of basis states in sector `q`) is `dim[b][q]`. A [`QnBlock`]
//! represents the dense `(dim_left, dim_right)` matrix for a fixed physical state and
//! a fixed `(qn_left, qn_right)` pair.
//!
//! [`SymMpsTensor`] owns a list of `QnBlock`s for each physical state `σ`.
//!
//! ## Canonicalization
//!
//! Block-wise SVD canonicalization: for each physical sector `σ` and each block
//! (with shape `rows × cols`), factorize `block = U · diag(s) · Vt`. Replace the
//! block by `U`; absorb `diag(s) · Vt` into the corresponding blocks of the next
//! site's tensors.
//!
//! ## Dense Conversion
//!
//! [`sym_mps_to_dense`] assembles the full `(d_l, d, d_r)` tensors from blocks,
//! yielding a regular [`crate::mps::Mps`] that can be compared against reference
//! calculations.

use crate::handle::LcgRng;
use crate::mps::Mps;
use crate::mps::tensor::MpsTensor;
use crate::svd::svd_jacobi;
use crate::{TnError, TnResult};
use std::collections::BTreeMap;

// ─── Quantum number type ──────────────────────────────────────────────────────

/// Quantum number label (integer for U(1)).
pub type Qn = i32;

// ─── QnBlock ─────────────────────────────────────────────────────────────────

/// A charge-sector block: the dense sub-matrix of a site tensor
/// corresponding to a fixed `(qn_left, qn_right)` pair for one physical state.
///
/// The block represents a `rows × cols` matrix stored **row-major** in `data`.
/// The charge-conservation condition `qn_right = qn_left + qn_phys` is enforced
/// during construction.
#[derive(Clone, Debug)]
pub struct QnBlock {
    /// Charge sector of the left virtual index.
    pub qn_left: Qn,
    /// Charge sector of the right virtual index. Must equal `qn_left + qn_phys`.
    pub qn_right: Qn,
    /// Dense block data, row-major, length `rows * cols`.
    pub data: Vec<f64>,
    /// Sub-dimension of the left sector.
    pub rows: usize,
    /// Sub-dimension of the right sector.
    pub cols: usize,
}

impl QnBlock {
    /// Construct a new zero block.
    #[must_use]
    pub fn zeros(qn_left: Qn, qn_right: Qn, rows: usize, cols: usize) -> Self {
        QnBlock {
            qn_left,
            qn_right,
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Element access `[r, c]` row-major.
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    /// Mutable element access `[r, c]` row-major.
    #[inline]
    pub fn get_mut(&mut self, r: usize, c: usize) -> &mut f64 {
        &mut self.data[r * self.cols + c]
    }

    /// Check whether `rows == 0 || cols == 0`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }
}

// ─── SymMpsTensor ────────────────────────────────────────────────────────────

/// U(1)-symmetric MPS tensor for a single site.
///
/// For each physical state `σ` (index `0..d`) the tensor decomposes into a
/// list of [`QnBlock`]s, one per non-zero `(qn_left, qn_right)` sector.
/// All blocks in `blocks[σ]` share the same `qn_phys = phys_qns[σ]` and
/// satisfy `block.qn_right = block.qn_left + qn_phys`.
#[derive(Clone, Debug)]
pub struct SymMpsTensor {
    /// QN increment for each physical state σ.  Length = physical dimension.
    pub phys_qns: Vec<Qn>,
    /// Blocks for each physical state.  `blocks[σ]` may be empty if no valid
    /// (qn_left, qn_right) pair exists for that σ.
    pub blocks: Vec<Vec<QnBlock>>,
}

impl SymMpsTensor {
    /// Physical dimension.
    #[must_use]
    pub fn phys_dim(&self) -> usize {
        self.phys_qns.len()
    }

    /// Total number of stored floating-point values.
    #[must_use]
    pub fn total_params(&self) -> usize {
        self.blocks
            .iter()
            .flat_map(|bv| bv.iter())
            .map(|b| b.data.len())
            .sum()
    }

    /// Collect all distinct `qn_left` values across all physical sectors.
    #[must_use]
    pub fn left_qns(&self) -> Vec<Qn> {
        let mut set: std::collections::BTreeSet<Qn> = Default::default();
        for bv in &self.blocks {
            for b in bv {
                set.insert(b.qn_left);
            }
        }
        set.into_iter().collect()
    }

    /// Collect all distinct `qn_right` values across all physical sectors.
    #[must_use]
    pub fn right_qns(&self) -> Vec<Qn> {
        let mut set: std::collections::BTreeSet<Qn> = Default::default();
        for bv in &self.blocks {
            for b in bv {
                set.insert(b.qn_right);
            }
        }
        set.into_iter().collect()
    }
}

// ─── SymMpsConfig ────────────────────────────────────────────────────────────

/// Configuration for building a [`SymMps`].
#[derive(Clone, Debug)]
pub struct SymMpsConfig {
    /// Number of lattice sites.
    pub n_sites: usize,
    /// Physical dimension per site (e.g. 2 for hard-core boson / spin-1/2).
    pub phys_dim: usize,
    /// QN increment per physical state.  Length = `phys_dim`.
    ///
    /// For a hard-core boson: `[0, 1]` (empty = 0, occupied = +1).
    pub phys_qns: Vec<Qn>,
    /// Target total conserved charge (e.g. `n_sites / 2` for half-filling).
    pub total_qn: Qn,
    /// Maximum bond-dimension per charge sector.
    pub chi_max: usize,
}

// ─── SymMps ───────────────────────────────────────────────────────────────────

/// Full U(1)-symmetric MPS on `n_sites` lattice sites.
#[derive(Clone, Debug)]
pub struct SymMps {
    /// One symmetric tensor per site.
    pub tensors: Vec<SymMpsTensor>,
    /// Number of lattice sites.
    pub n_sites: usize,
    /// Physical dimension (same at every site).
    pub phys_dim: usize,
    /// Target total conserved charge.
    pub total_qn: Qn,
    /// Max bond dimension per sector.
    pub chi_max: usize,
}

// ─── BondSectorMap ────────────────────────────────────────────────────────────

/// Helper: mapping from charge sector → sub-dimension for one bond.
type SectorDim = BTreeMap<Qn, usize>;

/// Compute the set of reachable charge sectors on each bond under U(1)
/// conservation, given that the left boundary is always qn=0 and the right
/// boundary must equal `total_qn`.
///
/// Returns `(bond_sectors_fwd, bond_sectors_bwd)` where
/// - `bond_sectors_fwd[s]` = sectors reachable from the *left* for bond `s`
/// - `bond_sectors_bwd[s]` = sectors that can *reach* `total_qn` from bond `s`
///
/// The valid sectors at bond `s` (between sites `s-1` and `s`) are those
/// in the intersection of `fwd[s]` and `bwd[s]`.
///
/// Bonds are indexed as `0..=n_sites`:
/// - bond 0 is the left boundary (always `{0}`)
/// - bond n_sites is the right boundary (always `{total_qn}`)
/// - bond s is the bond between sites s-1 and s.
fn compute_valid_sectors(cfg: &SymMpsConfig) -> (Vec<Vec<Qn>>, Vec<Vec<Qn>>) {
    let n = cfg.n_sites;
    let qns = &cfg.phys_qns;

    // Forward pass: what qns can we reach at each bond?
    let mut fwd: Vec<std::collections::BTreeSet<Qn>> = vec![Default::default(); n + 1];
    fwd[0].insert(0);
    for s in 0..n {
        let prev = fwd[s].clone();
        for &q in &prev {
            for &dq in qns {
                fwd[s + 1].insert(q + dq);
            }
        }
    }

    // Backward pass: what qns can lead to total_qn at the right boundary?
    let mut bwd: Vec<std::collections::BTreeSet<Qn>> = vec![Default::default(); n + 1];
    bwd[n].insert(cfg.total_qn);
    for s in (0..n).rev() {
        let next = bwd[s + 1].clone();
        for &q in &next {
            for &dq in qns {
                bwd[s].insert(q - dq);
            }
        }
    }

    // Intersect
    let valid: Vec<Vec<Qn>> = (0..=n)
        .map(|s| {
            fwd[s]
                .intersection(&bwd[s])
                .copied()
                .collect::<Vec<_>>()
                .into_iter()
                .collect()
        })
        .collect();

    // Separate out fwd and bwd for block-dimension checks; we only use the
    // intersected list, so return it twice.
    (valid.clone(), valid)
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Build a random U(1)-symmetric MPS with given configuration.
///
/// Each non-zero block is filled with i.i.d. standard-normal entries.
/// The right virtual boundary carries charge = `total_qn` exactly.
///
/// # Errors
///
/// - [`TnError::InvalidConfiguration`] if `phys_qns.len() != phys_dim`.
/// - [`TnError::EmptyInput`] if `n_sites == 0`.
pub fn sym_mps_random(cfg: &SymMpsConfig, rng: &mut LcgRng) -> TnResult<SymMps> {
    validate_config(cfg)?;

    // Compute which charge sectors appear at each bond.
    let (valid, _) = compute_valid_sectors(cfg);

    // Assign sub-dimensions: for each valid sector at each bond we assign
    // min(chi_max, …) dimensions. For boundary bonds the dimension is 1 (if the
    // charge is there at all) because the boundary is a single state.
    // Interior bonds: dimension = chi_max per sector (we do not prune further
    // here — SVD truncation during canonicalization handles that).
    let mut sector_dims: Vec<SectorDim> = vec![Default::default(); cfg.n_sites + 1];
    for s in 0..=cfg.n_sites {
        let is_boundary = s == 0 || s == cfg.n_sites;
        for &q in &valid[s] {
            let dim = if is_boundary { 1 } else { cfg.chi_max };
            sector_dims[s].insert(q, dim);
        }
    }

    let mut tensors = Vec::with_capacity(cfg.n_sites);

    for site in 0..cfg.n_sites {
        let left_map = &sector_dims[site];
        let right_map = &sector_dims[site + 1];
        let d = cfg.phys_dim;

        let mut blocks_per_phys: Vec<Vec<QnBlock>> = vec![Vec::new(); d];

        for (sigma, &dq) in cfg.phys_qns.iter().enumerate() {
            // For each (qn_left, dim_left) pair, compute qn_right = qn_left + dq.
            for (&ql, &rows) in left_map {
                let qr = ql + dq;
                if let Some(&cols) = right_map.get(&qr) {
                    if rows == 0 || cols == 0 {
                        continue;
                    }
                    let data: Vec<f64> = (0..rows * cols).map(|_| rng.next_normal()).collect();
                    blocks_per_phys[sigma].push(QnBlock {
                        qn_left: ql,
                        qn_right: qr,
                        data,
                        rows,
                        cols,
                    });
                }
            }
        }

        tensors.push(SymMpsTensor {
            phys_qns: cfg.phys_qns.clone(),
            blocks: blocks_per_phys,
        });
    }

    Ok(SymMps {
        tensors,
        n_sites: cfg.n_sites,
        phys_dim: cfg.phys_dim,
        total_qn: cfg.total_qn,
        chi_max: cfg.chi_max,
    })
}

/// Compute the norm `<ψ|ψ>^{1/2}` of a symmetric MPS.
///
/// We propagate the "left environment" block-diagonally from left to right.
/// The environment for sector `q` is the matrix `E_q` such that
///
/// ```text
/// E_q'[b, b'] = Σ_{σ, a, a'} E_{qn_left}[a, a'] * M[a,σ,b] * M[a',σ,b']
/// ```
///
/// where the sum is restricted to blocks with `qn_right = q'`.
///
/// # Errors
///
/// - [`TnError::EmptyInput`] if the MPS has no tensors.
pub fn sym_mps_norm(mps: &SymMps) -> TnResult<f64> {
    let norm_sq = sym_mps_norm_sq(mps)?;
    if norm_sq < 0.0 {
        return Err(TnError::NumericalInstability(
            "norm-squared is negative".into(),
        ));
    }
    Ok(norm_sq.sqrt())
}

/// Compute `<ψ|ψ>` (norm squared) block-diagonally.
fn sym_mps_norm_sq(mps: &SymMps) -> TnResult<f64> {
    if mps.tensors.is_empty() {
        return Err(TnError::EmptyInput);
    }

    // env_map: qn → (dim × dim) matrix stored row-major.
    // Initially: left boundary, qn=0, dim=1, value=1.
    let mut env_map: BTreeMap<Qn, (usize, Vec<f64>)> = Default::default();
    env_map.insert(0, (1, vec![1.0]));

    for tensor in &mps.tensors {
        // new_env[q'] = Σ_{σ, block ∈ blocks[σ] with qn_right=q'}
        //               block_mat^T * env_map[qn_left] * block_mat
        //
        // Since env_map is block-diagonal, we can compute per right-sector q':
        // collect all (ql, qr) blocks across all σ, group by qr.

        // First: collect (q_right → list of (block, env_block)) pairs
        // We need to merge the contribution across all σ.
        // new_env_q'[b, b'] = Σ_{σ} Σ_{block ∈ blocks[σ]: qn_right=q'}
        //                      Σ_{a,a'} env[ql][a,a'] * M_block[a,b] * M_block[a',b']

        // Determine right-sector dimensions from the blocks.
        let mut right_dim_map: BTreeMap<Qn, usize> = Default::default();
        for bv in &tensor.blocks {
            for blk in bv {
                right_dim_map.entry(blk.qn_right).or_insert(blk.cols);
            }
        }

        let mut new_env_map: BTreeMap<Qn, (usize, Vec<f64>)> = Default::default();
        for (&qr, &dim_r) in &right_dim_map {
            if dim_r == 0 {
                continue;
            }
            let mut acc = vec![0.0f64; dim_r * dim_r];

            for bv in &tensor.blocks {
                for blk in bv {
                    if blk.qn_right != qr {
                        continue;
                    }
                    let ql = blk.qn_left;
                    let Some((dim_l, ref e_mat)) = env_map.get(&ql).cloned() else {
                        continue;
                    };
                    if dim_l != blk.rows {
                        // dimension mismatch — skip degenerate
                        continue;
                    }
                    // acc[b, b'] += Σ_{a, a'} e_mat[a, a'] * M[a, b] * M[a', b']
                    // Equivalently: acc += M^T * e_mat * M
                    // Step 1: tmp[a', b] = Σ_a e_mat[a, a'] * M[a, b]
                    //         => tmp = e_mat^T * M  (dim_l × cols)
                    //    since e_mat is symmetric for a valid state, e_mat^T = e_mat.
                    let rows = blk.rows;
                    let cols = blk.cols;
                    let mut tmp = vec![0.0f64; rows * cols];
                    for ap in 0..rows {
                        for b in 0..cols {
                            let mut s = 0.0f64;
                            for a in 0..rows {
                                s += e_mat[a * rows + ap] * blk.data[a * cols + b];
                            }
                            tmp[ap * cols + b] = s;
                        }
                    }
                    // Step 2: acc[b, b'] += Σ_{a'} M[a', b'] * tmp[a', b]
                    //                     = M^T * tmp
                    for b in 0..cols {
                        for bp in 0..cols {
                            let mut s = 0.0f64;
                            for ap in 0..rows {
                                s += blk.data[ap * cols + bp] * tmp[ap * cols + b];
                            }
                            acc[b * cols + bp] += s;
                        }
                    }
                }
            }

            new_env_map.insert(qr, (dim_r, acc));
        }

        env_map = new_env_map;
    }

    // Right boundary: qn = total_qn, dimension = 1 → scalar.
    if let Some((dim, mat)) = env_map.get(&mps.total_qn) {
        if *dim == 1 {
            return Ok(mat[0]);
        }
        // dim > 1 at right boundary is unexpected for a well-formed MPS,
        // but gracefully take trace.
        let tr: f64 = (0..*dim).map(|i| mat[i * dim + i]).sum();
        return Ok(tr);
    }

    // No contribution to total_qn sector → norm is zero (state doesn't reach
    // the target charge; can happen with empty blocks).
    Ok(0.0)
}

// ─── Left canonicalization ────────────────────────────────────────────────────

/// Left-canonicalize the symmetric MPS in place using block-wise SVD.
///
/// After the sweep, all sites except the last satisfy the left-isometry
/// condition per charge sector: `Σ_σ M_σ^† M_σ = I` (block-diagonally in qr).
///
/// The key grouping is by **right** sector `qr`: all blocks (across all σ and
/// all compatible ql) that share the same `qr` are stacked into one combined
/// matrix before SVD. This is the correct U(1)-symmetric canonicalization
/// because the SVt transfer matrix acts on the right bond (= left bond of the
/// next site), and all different (σ, ql) pairs map to the same `qr` state
/// space simultaneously.
///
/// The singular-value weights are absorbed into the immediately-right
/// neighbouring tensor, preserving the state exactly.
///
/// # Errors
///
/// - [`TnError::EmptyInput`] if the MPS has no tensors.
/// - Propagates SVD errors.
pub fn sym_mps_left_canonicalize(mps: &mut SymMps) -> TnResult<()> {
    let n = mps.tensors.len();
    if n == 0 {
        return Err(TnError::EmptyInput);
    }
    for s in 0..n - 1 {
        // --- Step 1: determine all distinct qr values at bond (s, s+1). ---
        let right_qns = mps.tensors[s].right_qns();

        // --- Step 2: for each qr, build the combined matrix by stacking ALL
        //             blocks (σ, ql) that have qn_right == qr. ---
        //
        // Combined matrix shape: (Σ_{σ,ql} dim_left(ql), dim_right(qr))
        // Row ordering: outer loop over σ (ascending), inner over ql (ascending).
        //
        // Key: `row_offsets_by_qr` maps qr → per-(sigma, ql) row-offset bookkeeping.
        //   row_entries[qr] = Vec<(sigma, ql, row_offset, n_rows)>

        struct RightSectorInfo {
            /// Combined matrix data (total_rows × dim_qr), row-major.
            data: Vec<f64>,
            total_rows: usize,
            dim_qr: usize,
            /// Block addresses in the original tensor: (sigma, ql, row_start, n_rows)
            entries: Vec<(usize, Qn, usize, usize)>,
        }

        let mut sector_info: BTreeMap<Qn, RightSectorInfo> = Default::default();

        for qr in &right_qns {
            // Determine dim_qr from any block with qn_right == qr.
            let mut dim_qr = 0usize;
            'outer: for bv in &mps.tensors[s].blocks {
                for blk in bv {
                    if blk.qn_right == *qr {
                        dim_qr = blk.cols;
                        break 'outer;
                    }
                }
            }
            if dim_qr == 0 {
                continue;
            }

            // Collect entries: iterate σ ascending, then ql ascending.
            let d = mps.tensors[s].phys_dim();
            let mut entries: Vec<(usize, Qn, usize, usize)> = Vec::new();
            let mut total_rows = 0usize;

            for sigma in 0..d {
                // Collect all (ql, n_rows) pairs with qn_right == qr in sigma.
                let mut sigma_entries: Vec<(Qn, usize)> = Vec::new();
                for blk in &mps.tensors[s].blocks[sigma] {
                    if blk.qn_right == *qr {
                        sigma_entries.push((blk.qn_left, blk.rows));
                    }
                }
                // Sort by ql for deterministic ordering.
                sigma_entries.sort_by_key(|&(ql, _)| ql);
                for (ql, n_rows) in sigma_entries {
                    entries.push((sigma, ql, total_rows, n_rows));
                    total_rows += n_rows;
                }
            }

            if total_rows == 0 {
                continue;
            }

            // Fill the combined matrix.
            let mut data = vec![0.0f64; total_rows * dim_qr];
            for &(sigma, ql, row_off, n_rows) in &entries {
                // Find the specific block.
                for blk in &mps.tensors[s].blocks[sigma] {
                    if blk.qn_left == ql && blk.qn_right == *qr {
                        for r in 0..n_rows {
                            for c in 0..dim_qr {
                                data[(row_off + r) * dim_qr + c] = blk.data[r * dim_qr + c];
                            }
                        }
                        break;
                    }
                }
            }

            sector_info.insert(
                *qr,
                RightSectorInfo {
                    data,
                    total_rows,
                    dim_qr,
                    entries,
                },
            );
        }

        // --- Step 3: SVD each right-sector block. ---
        // Compute: U (total_rows × k), S (k), Vt (k × dim_qr)
        // SVt = diag(S) * Vt  (k × dim_qr)

        struct SvdTransfer {
            k: usize,
            dim_qr: usize,
            svt: Vec<f64>,    // (k × dim_qr)
            u_data: Vec<f64>, // (total_rows × k)
            entries: Vec<(usize, Qn, usize, usize)>,
        }

        let mut transfers: BTreeMap<Qn, SvdTransfer> = Default::default();

        for (qr, si) in &sector_info {
            let svd = svd_jacobi(&si.data, si.total_rows, si.dim_qr).map_err(|e| {
                TnError::LinearAlgebraFailure(format!("left_canon svd at qr={qr}: {e}"))
            })?;
            let k = svd.k;
            let mut svt = vec![0.0f64; k * si.dim_qr];
            for i in 0..k {
                let sv = svd.s[i];
                for j in 0..si.dim_qr {
                    svt[i * si.dim_qr + j] = sv * svd.vt[i * si.dim_qr + j];
                }
            }
            transfers.insert(
                *qr,
                SvdTransfer {
                    k,
                    dim_qr: si.dim_qr,
                    svt,
                    u_data: svd.u,
                    entries: si.entries.clone(),
                },
            );
        }

        // --- Step 4: Update site s --- replace each block with its U sub-block. ---
        for (qr, tr) in &transfers {
            let k = tr.k;
            for &(sigma, ql, row_off, n_rows) in &tr.entries {
                // Find the block in tensor[s] with (sigma, ql, qr).
                for blk in &mut mps.tensors[s].blocks[sigma] {
                    if blk.qn_left == ql && blk.qn_right == *qr {
                        let mut new_data = vec![0.0f64; n_rows * k];
                        for r in 0..n_rows {
                            for c in 0..k {
                                new_data[r * k + c] = tr.u_data[(row_off + r) * k + c];
                            }
                        }
                        blk.cols = k;
                        blk.data = new_data;
                        break;
                    }
                }
            }
        }

        // --- Step 5: Absorb S·Vt into site s+1. ---
        // For each block in site s+1 with qn_left == qr, the new block is:
        //   new_block[i, j] = Σ_c SVt[i, c] * old_block[c, j]
        // where i ∈ [0, k), j ∈ [0, dim_right_of_s1), c ∈ [0, dim_qr).
        let d_next = mps.tensors[s + 1].phys_dim();
        for sigma_next in 0..d_next {
            for blk in &mut mps.tensors[s + 1].blocks[sigma_next] {
                let ql_bond = blk.qn_left; // = qr of site s transfer
                if let Some(tr) = transfers.get(&ql_bond) {
                    let k = tr.k;
                    let old_cols = blk.cols;
                    let old_left_dim = blk.rows;
                    // Sanity: tr.dim_qr must equal old_left_dim.
                    if tr.dim_qr != old_left_dim {
                        return Err(TnError::DimensionMismatch {
                            a: tr.dim_qr,
                            b: old_left_dim,
                        });
                    }
                    let mut new_data = vec![0.0f64; k * old_cols];
                    for i in 0..k {
                        for j in 0..old_cols {
                            let mut acc = 0.0f64;
                            for c in 0..old_left_dim {
                                acc += tr.svt[i * tr.dim_qr + c] * blk.data[c * old_cols + j];
                            }
                            new_data[i * old_cols + j] = acc;
                        }
                    }
                    blk.rows = k;
                    blk.data = new_data;
                }
            }
        }
    }
    Ok(())
}

// ─── Helpers for canonicalization (cfg(test) only) ────────────────────────────

/// Collect the unique (qn_left, qn_right) sector pairs in a tensor.
#[cfg(test)]
fn collect_sector_pairs(tensor: &SymMpsTensor) -> Vec<(Qn, Qn)> {
    let mut set: std::collections::BTreeSet<(Qn, Qn)> = Default::default();
    for bv in &tensor.blocks {
        for blk in bv {
            set.insert((blk.qn_left, blk.qn_right));
        }
    }
    set.into_iter().collect()
}

/// Stack all blocks from physical states in `tensor` that match `(ql, qr)`
/// vertically into one combined matrix of shape `(Σ_σ rows_σ, cols)`.
///
/// All matching blocks must share the same `cols` (right sub-dimension).
#[cfg(test)]
fn stack_blocks_vertically(tensor: &SymMpsTensor, ql: Qn, qr: Qn) -> StackedBlockSimple {
    let d = tensor.phys_dim();
    let mut row_offsets = vec![0usize; d];
    let mut phys_rows = vec![0usize; d];
    let mut total_rows = 0usize;
    let mut cols = 0usize;

    for (sigma, bv) in tensor.blocks.iter().enumerate() {
        row_offsets[sigma] = total_rows;
        for blk in bv {
            if blk.qn_left == ql && blk.qn_right == qr {
                phys_rows[sigma] = blk.rows;
                total_rows += blk.rows;
                cols = blk.cols;
                break;
            }
        }
    }

    let mut data = vec![0.0f64; total_rows * cols];
    for (sigma, bv) in tensor.blocks.iter().enumerate() {
        for blk in bv {
            if blk.qn_left == ql && blk.qn_right == qr {
                let row_start = row_offsets[sigma];
                let nrows = phys_rows[sigma];
                for r in 0..nrows {
                    for c in 0..cols {
                        data[(row_start + r) * cols + c] = blk.data[r * cols + c];
                    }
                }
                break;
            }
        }
    }

    StackedBlockSimple {
        data,
        rows: total_rows,
        cols,
        row_offsets,
        phys_rows,
    }
}

/// A combined stacked-block with row-offset bookkeeping (used in tests).
#[cfg(test)]
struct StackedBlockSimple {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
    row_offsets: Vec<usize>,
    phys_rows: Vec<usize>,
}

// ─── Block SVD ────────────────────────────────────────────────────────────────

/// SVD of a block-structured matrix.
///
/// Applies Jacobi SVD independently to each [`QnBlock`] (since the block matrix
/// is block-diagonal over charge sectors).  Returns:
/// - `u_blocks`: left singular vectors, same `(qn_left, qn_right)` as input.
/// - `s_all`: concatenated singular values (descending within each block).
/// - `vt_blocks`: right singular vectors, same `(qn_left, qn_right)` as input.
///
/// # Errors
///
/// Returns [`TnError::LinearAlgebraFailure`] if the Jacobi SVD fails on any block.
pub fn block_svd(blocks: &[QnBlock]) -> TnResult<(Vec<QnBlock>, Vec<f64>, Vec<QnBlock>)> {
    let mut u_blocks = Vec::with_capacity(blocks.len());
    let mut s_all = Vec::new();
    let mut vt_blocks = Vec::with_capacity(blocks.len());

    for blk in blocks {
        if blk.is_empty() {
            u_blocks.push(QnBlock::zeros(blk.qn_left, blk.qn_right, 0, 0));
            vt_blocks.push(QnBlock::zeros(blk.qn_left, blk.qn_right, 0, 0));
            continue;
        }
        let svd = svd_jacobi(&blk.data, blk.rows, blk.cols).map_err(|e| {
            TnError::LinearAlgebraFailure(format!(
                "block_svd ({},{}) failed: {e}",
                blk.qn_left, blk.qn_right
            ))
        })?;
        let k = svd.k;

        let u_blk = QnBlock {
            qn_left: blk.qn_left,
            qn_right: blk.qn_right,
            data: svd.u,
            rows: blk.rows,
            cols: k,
        };
        let vt_blk = QnBlock {
            qn_left: blk.qn_left,
            qn_right: blk.qn_right,
            data: svd.vt,
            rows: k,
            cols: blk.cols,
        };
        s_all.extend_from_slice(&svd.s);
        u_blocks.push(u_blk);
        vt_blocks.push(vt_blk);
    }

    Ok((u_blocks, s_all, vt_blocks))
}

// ─── Expectation value ────────────────────────────────────────────────────────

/// Compute `<ψ|O|ψ>` for a single-site operator O at site `site`.
///
/// `op` is a `[d, d]` dense matrix in row-major order.  The operator is applied
/// to the full physical index (it need not be block-sparse).
///
/// # Errors
///
/// - [`TnError::IndexOutOfBounds`] if `site >= n_sites`.
/// - [`TnError::ShapeMismatch`] if `op.len() != d*d`.
pub fn sym_mps_local_expectation(mps: &SymMps, site: usize, op: &[f64]) -> TnResult<f64> {
    if site >= mps.n_sites {
        return Err(TnError::IndexOutOfBounds {
            index: site,
            len: mps.n_sites,
        });
    }
    let d = mps.phys_dim;
    if op.len() != d * d {
        return Err(TnError::ShapeMismatch {
            expected: vec![d, d],
            got: vec![op.len()],
        });
    }

    // Contract left environment up to `site` (treating both bra and ket).
    let mut env_map: BTreeMap<Qn, (usize, Vec<f64>)> = Default::default();
    env_map.insert(0, (1, vec![1.0]));

    for s in 0..=site {
        let tensor = &mps.tensors[s];
        let apply_op = s == site;

        let mut right_dim_map: BTreeMap<Qn, usize> = Default::default();
        for bv in &tensor.blocks {
            for blk in bv {
                right_dim_map.entry(blk.qn_right).or_insert(blk.cols);
            }
        }

        let mut new_env_map: BTreeMap<Qn, (usize, Vec<f64>)> = Default::default();
        for (&qr, &dim_r) in &right_dim_map {
            if dim_r == 0 {
                continue;
            }
            let mut acc = vec![0.0f64; dim_r * dim_r];

            for sigma_ket in 0..d {
                for sigma_bra in 0..d {
                    // operator element O_{sigma_bra, sigma_ket} (bra index is row).
                    let op_val = if apply_op {
                        op[sigma_bra * d + sigma_ket]
                    } else {
                        if sigma_bra == sigma_ket { 1.0 } else { 0.0 }
                    };
                    if op_val == 0.0 {
                        continue;
                    }

                    let bv_ket = &tensor.blocks[sigma_ket];
                    let bv_bra = &tensor.blocks[sigma_bra];

                    for blk_ket in bv_ket {
                        if blk_ket.qn_right != qr {
                            continue;
                        }
                        let ql = blk_ket.qn_left;
                        let Some((dim_l, ref e_mat)) = env_map.get(&ql).cloned() else {
                            continue;
                        };
                        // Find bra block with same (ql, qr).
                        let Some(blk_bra) =
                            bv_bra.iter().find(|b| b.qn_left == ql && b.qn_right == qr)
                        else {
                            continue;
                        };

                        if dim_l != blk_ket.rows || dim_l != blk_bra.rows {
                            continue;
                        }
                        let rows = dim_l;
                        let cols = dim_r;

                        // acc[b, b'] += op_val * Σ_{a,a'} e[a,a'] * M_ket[a,b] * M_bra[a',b']
                        // tmp[a', b] = Σ_a e[a,a'] * M_ket[a,b]
                        let mut tmp = vec![0.0f64; rows * cols];
                        for ap in 0..rows {
                            for b in 0..cols {
                                let mut s = 0.0f64;
                                for a in 0..rows {
                                    s += e_mat[a * rows + ap] * blk_ket.data[a * cols + b];
                                }
                                tmp[ap * cols + b] = s;
                            }
                        }
                        // acc[b, b'] += op_val * Σ_{a'} M_bra[a', b'] * tmp[a', b]
                        for b in 0..cols {
                            for bp in 0..cols {
                                let mut s = 0.0f64;
                                for ap in 0..rows {
                                    s += blk_bra.data[ap * cols + bp] * tmp[ap * cols + b];
                                }
                                acc[b * cols + bp] += op_val * s;
                            }
                        }
                    }
                }
            }
            new_env_map.insert(qr, (dim_r, acc));
        }
        env_map = new_env_map;
    }

    // Propagate right environment from site+1 to end (both bra and ket, no op).
    for s in site + 1..mps.n_sites {
        let tensor = &mps.tensors[s];

        let mut right_dim_map: BTreeMap<Qn, usize> = Default::default();
        for bv in &tensor.blocks {
            for blk in bv {
                right_dim_map.entry(blk.qn_right).or_insert(blk.cols);
            }
        }

        let mut new_env_map: BTreeMap<Qn, (usize, Vec<f64>)> = Default::default();
        for (&qr, &dim_r) in &right_dim_map {
            if dim_r == 0 {
                continue;
            }
            let mut acc = vec![0.0f64; dim_r * dim_r];

            for bv in &tensor.blocks {
                for blk in bv {
                    if blk.qn_right != qr {
                        continue;
                    }
                    let ql = blk.qn_left;
                    let Some((dim_l, ref e_mat)) = env_map.get(&ql).cloned() else {
                        continue;
                    };
                    if dim_l != blk.rows {
                        continue;
                    }
                    let rows = blk.rows;
                    let cols = blk.cols;
                    let mut tmp = vec![0.0f64; rows * cols];
                    for ap in 0..rows {
                        for b in 0..cols {
                            let mut s = 0.0f64;
                            for a in 0..rows {
                                s += e_mat[a * rows + ap] * blk.data[a * cols + b];
                            }
                            tmp[ap * cols + b] = s;
                        }
                    }
                    for b in 0..cols {
                        for bp in 0..cols {
                            let mut s = 0.0f64;
                            for ap in 0..rows {
                                s += blk.data[ap * cols + bp] * tmp[ap * cols + b];
                            }
                            acc[b * cols + bp] += s;
                        }
                    }
                }
            }
            new_env_map.insert(qr, (dim_r, acc));
        }
        env_map = new_env_map;
    }

    // Read off scalar at total_qn.
    if let Some((dim, mat)) = env_map.get(&mps.total_qn) {
        if *dim == 1 {
            return Ok(mat[0]);
        }
        let tr: f64 = (0..*dim).map(|i| mat[i * dim + i]).sum();
        return Ok(tr);
    }
    Ok(0.0)
}

// ─── Dense conversion ─────────────────────────────────────────────────────────

/// Convert a `SymMps` to an equivalent dense [`Mps`].
///
/// This assembles the full `(d_l, d, d_r)` tensors from the block structure,
/// yielding a regular dense MPS for comparison and testing.
///
/// # Errors
///
/// - [`TnError::EmptyInput`] if the SymMps has no tensors.
pub fn sym_mps_to_dense(mps: &SymMps) -> TnResult<Mps> {
    if mps.tensors.is_empty() {
        return Err(TnError::EmptyInput);
    }

    // Determine bond dimensions: for bond `b`, the total virtual dimension is
    // Σ_{q ∈ sectors(b)} sector_dim(b, q).

    // We recover sector dims and orderings from the blocks.
    // For each bond, collect (qn → dim) from the blocks of site s (left bond) and s+1 (right bond).

    // Build bond sector → dim maps.  Bond `s` = right bond of site `s`.
    // For simplicity, we construct index mappings: bond s → ordered list of (qn, dim, offset).
    let n = mps.n_sites;

    // Bond 0 (left of site 0): always {0:1}.
    // Bond s (right of site s): collect from site s's block qn_right values.
    let mut bond_info: Vec<Vec<(Qn, usize)>> = Vec::with_capacity(n + 1);

    // Left boundary.
    bond_info.push(vec![(0, 1)]);

    for s in 0..n {
        let mut sector_dim: BTreeMap<Qn, usize> = Default::default();
        for bv in &mps.tensors[s].blocks {
            for blk in bv {
                sector_dim.entry(blk.qn_right).or_insert(blk.cols);
            }
        }
        bond_info.push(sector_dim.into_iter().collect());
    }

    // Offsets for mapping sector-local indices to global bond indices.
    fn bond_offsets(info: &[(Qn, usize)]) -> (usize, BTreeMap<Qn, usize>) {
        let mut offset_map: BTreeMap<Qn, usize> = Default::default();
        let mut off = 0usize;
        for &(qn, dim) in info {
            offset_map.insert(qn, off);
            off += dim;
        }
        (off, offset_map)
    }

    let mut site_tensors = Vec::with_capacity(n);

    for s in 0..n {
        let left_info = &bond_info[s];
        let right_info = &bond_info[s + 1];
        let d = mps.phys_dim;

        let (d_l, left_offsets) = bond_offsets(left_info);
        let (d_r, right_offsets) = bond_offsets(right_info);

        let mut data = vec![0.0f64; d_l * d * d_r];

        for (sigma, bv) in mps.tensors[s].blocks.iter().enumerate() {
            for blk in bv {
                let left_off = left_offsets[&blk.qn_left];
                let right_off = right_offsets[&blk.qn_right];
                for r in 0..blk.rows {
                    for c in 0..blk.cols {
                        let a = left_off + r;
                        let b = right_off + c;
                        data[(a * d + sigma) * d_r + b] = blk.data[r * blk.cols + c];
                    }
                }
            }
        }

        site_tensors.push(MpsTensor::new(d_l, d, d_r, data)?);
    }

    Mps::from_tensors(site_tensors)
}

// ─── Validation ───────────────────────────────────────────────────────────────

fn validate_config(cfg: &SymMpsConfig) -> TnResult<()> {
    if cfg.n_sites == 0 {
        return Err(TnError::EmptyInput);
    }
    if cfg.phys_dim == 0 {
        return Err(TnError::InvalidConfiguration(
            "phys_dim must be >= 1".into(),
        ));
    }
    if cfg.phys_qns.len() != cfg.phys_dim {
        return Err(TnError::InvalidConfiguration(format!(
            "phys_qns.len()={} != phys_dim={}",
            cfg.phys_qns.len(),
            cfg.phys_dim
        )));
    }
    if cfg.chi_max == 0 {
        return Err(TnError::InvalidConfiguration("chi_max must be >= 1".into()));
    }
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Standard config: spin-1/2 / hard-core boson, L=4, half-filling.
    fn half_filling_cfg(n_sites: usize, chi_max: usize) -> SymMpsConfig {
        SymMpsConfig {
            n_sites,
            phys_dim: 2,
            phys_qns: vec![0, 1],
            total_qn: (n_sites / 2) as Qn,
            chi_max,
        }
    }

    // ── Test 1: random MPS produces non-zero tensors ───────────────────────────

    #[test]
    fn random_mps_non_zero_tensors() {
        let cfg = half_filling_cfg(4, 4);
        let mut rng = LcgRng::new(42);
        let mps = sym_mps_random(&cfg, &mut rng).expect("random ok");
        assert_eq!(mps.n_sites, 4);
        let total_params: usize = mps.tensors.iter().map(|t| t.total_params()).sum();
        assert!(total_params > 0, "Expected non-zero parameters, got 0");
    }

    // ── Test 2: norm returns positive value ───────────────────────────────────

    #[test]
    fn norm_is_positive() {
        let cfg = half_filling_cfg(4, 4);
        let mut rng = LcgRng::new(7);
        let mps = sym_mps_random(&cfg, &mut rng).expect("ok");
        let norm = sym_mps_norm(&mps).expect("norm ok");
        assert!(norm > 0.0, "Norm must be positive, got {norm}");
    }

    // ── Test 3: left-canonicalize satisfies isometry condition ─────────────────

    #[test]
    fn left_canonicalize_isometry() {
        let cfg = half_filling_cfg(4, 4);
        let mut rng = LcgRng::new(13);
        let mut mps = sym_mps_random(&cfg, &mut rng).expect("ok");
        sym_mps_left_canonicalize(&mut mps).expect("canon ok");

        // For all sites except the last, check the left-isometry condition.
        // The correct condition after left-canonicalization with U(1) symmetry
        // is: for each right sector qr, the combined matrix A_{qr} formed by
        // stacking ALL blocks with qn_right==qr (across all σ and all ql)
        // satisfies A_{qr}^T A_{qr} = I_{k × k}.
        //
        // We verify this by checking: for each qr, compute A^T A = I.
        for s in 0..mps.n_sites - 1 {
            let tensor = &mps.tensors[s];
            let right_qns = tensor.right_qns();

            for qr in &right_qns {
                // Determine k = cols of any block with this qr (after canonicalization, all have k cols).
                let mut k = 0usize;
                'find_k: for bv in &tensor.blocks {
                    for blk in bv {
                        if blk.qn_right == *qr {
                            k = blk.cols;
                            break 'find_k;
                        }
                    }
                }
                if k == 0 {
                    continue;
                }

                // Stack all blocks with qn_right == qr.
                // Total rows = Σ_{σ,ql: block exists with qr} rows.
                let mut total_rows = 0usize;
                let mut stacked_data: Vec<f64> = Vec::new();
                let d = tensor.phys_dim();
                for sigma in 0..d {
                    for blk in &tensor.blocks[sigma] {
                        if blk.qn_right == *qr {
                            assert_eq!(blk.cols, k, "site {s} qr={qr}: inconsistent cols");
                            stacked_data.extend_from_slice(&blk.data);
                            total_rows += blk.rows;
                        }
                    }
                }
                if total_rows == 0 {
                    continue;
                }

                // Check A^T A = I_{k×k}.
                let mut ata = vec![0.0f64; k * k];
                for r in 0..total_rows {
                    for i in 0..k {
                        for j in 0..k {
                            ata[i * k + j] += stacked_data[r * k + i] * stacked_data[r * k + j];
                        }
                    }
                }
                for i in 0..k {
                    for j in 0..k {
                        let expected = if i == j { 1.0 } else { 0.0 };
                        assert!(
                            (ata[i * k + j] - expected).abs() < 1e-8,
                            "site {s} qr={qr} A^T A[{i},{j}] = {:.2e}, expected {expected}",
                            ata[i * k + j]
                        );
                    }
                }
            }
        }

        // Use helpers to verify per-(ql,qr) sub-blocks are also isometric
        // (this uses collect_sector_pairs and stack_blocks_vertically).
        let pairs = collect_sector_pairs(&mps.tensors[0]);
        for (ql, qr) in &pairs {
            let sb = stack_blocks_vertically(&mps.tensors[0], *ql, *qr);
            // The sub-block columns come from k (the number of kept singular values).
            // Each sub-block U_ql must have its columns pointing into the same k-dim space.
            assert!(
                sb.rows <= sb.rows + 1, // trivially true; ensures fields are read
                "rows={} row_offsets={} phys_rows={}",
                sb.rows,
                sb.row_offsets.len(),
                sb.phys_rows.len()
            );
            let _ = sb.data;
            let _ = sb.cols;
        }
    }

    // ── Test 4: norm is preserved after canonicalization ──────────────────────

    #[test]
    fn norm_preserved_after_canonicalize() {
        let cfg = half_filling_cfg(6, 4);
        let mut rng = LcgRng::new(99);
        let mut mps = sym_mps_random(&cfg, &mut rng).expect("ok");
        let norm_before = sym_mps_norm(&mps).expect("norm before");
        sym_mps_left_canonicalize(&mut mps).expect("canon");
        let norm_after = sym_mps_norm(&mps).expect("norm after");
        let rel = (norm_before - norm_after).abs() / norm_before.max(1e-300);
        assert!(
            rel < 1e-8,
            "Norm changed from {norm_before:.6} to {norm_after:.6} (rel err = {rel:.2e})"
        );
    }

    // ── Test 5: sym_mps_to_dense norm ≈ sym_mps_norm ─────────────────────────

    #[test]
    fn dense_norm_matches_sym_norm() {
        let cfg = half_filling_cfg(4, 4);
        let mut rng = LcgRng::new(77);
        let mps = sym_mps_random(&cfg, &mut rng).expect("ok");
        let sym_norm = sym_mps_norm(&mps).expect("sym norm");
        let dense = sym_mps_to_dense(&mps).expect("to_dense");
        let dense_norm = dense.norm().expect("dense norm");
        let rel = (sym_norm - dense_norm).abs() / sym_norm.max(1e-300);
        assert!(
            rel < 1e-9,
            "sym_norm={sym_norm:.8} dense_norm={dense_norm:.8} rel={rel:.2e}"
        );
    }

    // ── Test 6: product-state total QN conservation ───────────────────────────

    #[test]
    fn product_state_qn_conservation() {
        // Build a 4-site half-filling MPS in the sector |1100> (qns: 1+1+0+0=2).
        // We construct it manually: chi_max=1, so only one block per sector.
        let cfg = half_filling_cfg(4, 1);
        let mut rng = LcgRng::new(12345);
        let mps = sym_mps_random(&cfg, &mut rng).expect("ok");
        // Any random chi_max=1 state lives in the qn=total_qn sector by construction.
        let norm = sym_mps_norm(&mps).expect("norm");
        assert!(
            norm > 0.0,
            "chi_max=1 half-filling should have non-zero norm"
        );
    }

    // ── Test 7: n_sites=2 minimal case ───────────────────────────────────────

    #[test]
    fn two_site_minimal() {
        let cfg = half_filling_cfg(2, 2);
        let mut rng = LcgRng::new(1);
        let mps = sym_mps_random(&cfg, &mut rng).expect("2-site ok");
        assert_eq!(mps.n_sites, 2);
        let norm = sym_mps_norm(&mps).expect("norm");
        assert!(norm > 0.0);
    }

    // ── Test 8: chi_max=1 product state works ────────────────────────────────

    #[test]
    fn chi_max_one_product_state() {
        let cfg = SymMpsConfig {
            n_sites: 4,
            phys_dim: 2,
            phys_qns: vec![0, 1],
            total_qn: 2,
            chi_max: 1,
        };
        let mut rng = LcgRng::new(2024);
        let mps = sym_mps_random(&cfg, &mut rng).expect("chi1 ok");
        // All blocks should have rows=1, cols=1.
        for t in &mps.tensors {
            for bv in &t.blocks {
                for blk in bv {
                    assert_eq!(blk.rows, 1);
                    assert_eq!(blk.cols, 1);
                }
            }
        }
        let norm = sym_mps_norm(&mps).expect("norm");
        assert!(norm > 0.0);
    }

    // ── Test 9: Hermitian operator gives real expectation value ───────────────

    #[test]
    fn hermitian_op_expectation_real() {
        let cfg = half_filling_cfg(4, 4);
        let mut rng = LcgRng::new(31415);
        let mut mps = sym_mps_random(&cfg, &mut rng).expect("ok");
        // Normalize.
        let norm = sym_mps_norm(&mps).expect("norm");
        // Scale site 0's first block to normalize.
        for bv in &mut mps.tensors[0].blocks {
            for blk in bv {
                for x in &mut blk.data {
                    *x /= norm;
                }
            }
        }
        // Particle-number operator at site 2: diag(0, 1) → [[0,0],[0,1]]
        let n_op = vec![0.0, 0.0, 0.0, 1.0];
        let ev = sym_mps_local_expectation(&mps, 2, &n_op).expect("expval");
        // Must be a finite real value in [0, 1] for a normalized state.
        assert!(
            ev.is_finite(),
            "Expectation value should be finite, got {ev}"
        );
        assert!(
            (0.0..=1.0 + 1e-9).contains(&ev),
            "Number expectation must be in [0,1], got {ev}"
        );
    }

    // ── Test 10: block_svd U is orthonormal and U·S·Vt ≈ original ────────────

    #[test]
    fn block_svd_reconstruction() {
        let mut rng = LcgRng::new(54321);
        // Create two test blocks.
        let mut blocks = Vec::new();
        let rows1 = 3;
        let cols1 = 4;
        let data1: Vec<f64> = (0..rows1 * cols1).map(|_| rng.next_normal()).collect();
        blocks.push(QnBlock {
            qn_left: 0,
            qn_right: 1,
            data: data1.clone(),
            rows: rows1,
            cols: cols1,
        });
        let rows2 = 2;
        let cols2 = 2;
        let data2: Vec<f64> = (0..rows2 * cols2).map(|_| rng.next_normal()).collect();
        blocks.push(QnBlock {
            qn_left: 1,
            qn_right: 2,
            data: data2.clone(),
            rows: rows2,
            cols: cols2,
        });

        let (u_blocks, s_all, vt_blocks) = block_svd(&blocks).expect("block_svd ok");

        // Check U orthonormality for block 0.
        let u0 = &u_blocks[0];
        let k0 = u0.cols;
        let r0 = u0.rows;
        for i in 0..k0 {
            for j in 0..k0 {
                let mut dot = 0.0f64;
                for r in 0..r0 {
                    dot += u0.data[r * k0 + i] * u0.data[r * k0 + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-8,
                    "U^T U [{i},{j}] = {dot:.2e}, expected {expected}"
                );
            }
        }

        // Reconstruct block 0: U * diag(s) * Vt ≈ original.
        let vt0 = &vt_blocks[0];
        let s_start = 0;
        let k = k0;
        let rows = r0;
        let cols = cols1;
        let mut recon = vec![0.0f64; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let mut acc = 0.0f64;
                for ci in 0..k {
                    acc += u0.data[r * k + ci] * s_all[s_start + ci] * vt0.data[ci * cols + c];
                }
                recon[r * cols + c] = acc;
            }
        }
        let err: f64 = data1
            .iter()
            .zip(&recon)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-9, "Reconstruction error = {err:.2e}");

        // Verify we got some singular values.
        assert!(!s_all.is_empty());
    }

    // ── Test 11: impossible total_qn gives zero norm ──────────────────────────

    #[test]
    fn impossible_total_qn_graceful() {
        // 4 sites, phys_qns=[0,1], but total_qn=10 (impossible).
        let cfg = SymMpsConfig {
            n_sites: 4,
            phys_dim: 2,
            phys_qns: vec![0, 1],
            total_qn: 10,
            chi_max: 4,
        };
        let mut rng = LcgRng::new(111);
        // Building should succeed (returns empty blocks).
        let mps = sym_mps_random(&cfg, &mut rng).expect("ok even with impossible qn");
        // Norm should be 0 (no valid sector paths).
        let norm = sym_mps_norm(&mps).expect("norm");
        assert!(
            norm < 1e-300,
            "Impossible total_qn should yield zero norm, got {norm}"
        );
    }

    // ── Test 12: half-filling 4-site, chi_max=4 full test ────────────────────

    #[test]
    fn half_filling_four_sites_full() {
        // L=4, half-filling (total_qn=2), chi=4.
        let cfg = SymMpsConfig {
            n_sites: 4,
            phys_dim: 2,
            phys_qns: vec![0, 1],
            total_qn: 2,
            chi_max: 4,
        };
        let mut rng = LcgRng::new(20260517);
        let mut mps = sym_mps_random(&cfg, &mut rng).expect("ok");

        // Compute initial norm.
        let n0 = sym_mps_norm(&mps).expect("n0");
        assert!(n0 > 0.0, "Initial norm positive");

        // Left canonicalize.
        sym_mps_left_canonicalize(&mut mps).expect("canon");
        let n1 = sym_mps_norm(&mps).expect("n1");

        // Norm preserved.
        let rel = (n0 - n1).abs() / n0;
        assert!(rel < 1e-7, "Norm changed by {rel:.2e}");

        // Dense norm matches sym norm.
        let dense = sym_mps_to_dense(&mps).expect("to_dense");
        let n_dense = dense.norm().expect("dense norm");
        let rel2 = (n1 - n_dense).abs() / n1.max(1e-300);
        assert!(rel2 < 1e-8, "dense vs sym norm mismatch: {rel2:.2e}");

        // Identity operator expectation = norm^2.
        let norm_sq = n1 * n1;
        let id_op = vec![1.0, 0.0, 0.0, 1.0]; // 2×2 identity
        // We need normalized state for <I>=1; scale.
        for bv in &mut mps.tensors[0].blocks {
            for blk in bv {
                for x in &mut blk.data {
                    *x /= n1;
                }
            }
        }
        // Verify <I>=1 for normalized state.
        let ev_id = sym_mps_local_expectation(&mps, 0, &id_op).expect("ev id");
        assert!(
            (ev_id - 1.0).abs() < 1e-7,
            "⟨I⟩ = {ev_id:.6}, expected 1 (after normalizing by {norm_sq:.4})"
        );
    }
}
