//! PARDISO-compatible sparse direct solver interface.
//!
//! Provides a [`PardisoCompatSolver`] whose API mirrors the classic **PARDISO**
//! direct solver (Schenk & Gärtner 2004, "Solving unsymmetric sparse systems of
//! linear equations with PARDISO", Future Gener. Comput. Syst. 20(3), 475–487).
//! PARDISO is driven by a single integer `phase` argument that selects which
//! stage(s) of the solve pipeline to execute:
//!
//! | `phase` | Stage(s)                                            |
//! |---------|-----------------------------------------------------|
//! | `11`    | Analysis: fill-reducing reordering + symbolic setup |
//! | `22`    | Numerical factorization                             |
//! | `33`    | Forward/backward solve (back-substitution)          |
//! | `12`    | Analysis + numerical factorization                  |
//! | `13`    | Analysis + factorization + solve (everything)       |
//! | `23`    | Factorization + solve                               |
//!
//! Splitting the work this way lets a caller pay the (expensive) analysis once
//! and then re-factorize / re-solve many times as the matrix values or
//! right-hand sides change — exactly PARDISO's value proposition.
//!
//! # Pipeline
//!
//! 1. **Reordering** (analysis): a fill-reducing permutation `P` is computed with
//!    nested dissection (falling back to the natural ordering for tiny systems),
//!    reusing [`crate::sparse::nested_dissection::NestedDissectionOrdering`].
//! 2. **Symbolic + numerical factorization**: the symmetrically permuted matrix
//!    `P·A·Pᵀ` is factored as `Pₚ·(P·A·Pᵀ) = L·U` with the left-looking sparse LU
//!    of [`crate::sparse::superlu_left_looking::LeftLookingLu`], which performs its
//!    own partial pivoting (`Pₚ`) for numerical stability on top of the
//!    fill-reducing order.
//! 3. **Solve**: for each right-hand side `b`, the solution is recovered as
//!    `x = Pᵀ · (L·U)⁻¹ · (P · b)`.
//!
//! This mirrors PARDISO's separation of the *fill-reducing* permutation (chosen
//! once from the structure) and the *pivoting* permutation (chosen during numeric
//! factorization), and supports unsymmetric matrices.

use crate::error::{SolverError, SolverResult};
use crate::sparse::nested_dissection::{AdjacencyGraph, NestedDissectionOrdering, Permutation};
use crate::sparse::superlu_left_looking::LeftLookingLu;

/// PARDISO `phase` selector.
///
/// The numeric values match the PARDISO convention so existing call sites that
/// pass raw integers can be ported mechanically via [`Phase::from_code`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Analysis only (`11`): reordering + symbolic structure.
    Analysis,
    /// Numerical factorization only (`22`).
    Factorize,
    /// Solve only (`33`): forward/backward substitution.
    Solve,
    /// Analysis + factorization (`12`).
    AnalysisFactorize,
    /// Factorization + solve (`23`).
    FactorizeSolve,
    /// Full pipeline (`13`): analysis + factorization + solve.
    All,
}

impl Phase {
    /// Map a raw PARDISO phase integer to a [`Phase`].
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::InternalError`] for an unrecognised code.
    pub fn from_code(code: i32) -> SolverResult<Self> {
        match code {
            11 => Ok(Phase::Analysis),
            22 => Ok(Phase::Factorize),
            33 => Ok(Phase::Solve),
            12 => Ok(Phase::AnalysisFactorize),
            23 => Ok(Phase::FactorizeSolve),
            13 => Ok(Phase::All),
            other => Err(SolverError::InternalError(format!(
                "pardiso: unsupported phase code {other} (expected 11/22/33/12/23/13)"
            ))),
        }
    }

    /// Does this phase include the analysis (reordering) stage?
    fn does_analysis(self) -> bool {
        matches!(
            self,
            Phase::Analysis | Phase::AnalysisFactorize | Phase::All
        )
    }

    /// Does this phase include the numerical factorization stage?
    fn does_factorize(self) -> bool {
        matches!(
            self,
            Phase::Factorize | Phase::AnalysisFactorize | Phase::FactorizeSolve | Phase::All
        )
    }

    /// Does this phase include the solve stage?
    fn does_solve(self) -> bool {
        matches!(self, Phase::Solve | Phase::FactorizeSolve | Phase::All)
    }
}

/// A PARDISO-compatible sparse direct solver.
///
/// Hold the matrix structure across phases: run [`Phase::Analysis`] once, then
/// [`Phase::Factorize`] whenever the numerical values change, then
/// [`Phase::Solve`] for each right-hand side. The combined phases
/// ([`Phase::All`], …) are provided for convenience.
#[derive(Debug)]
pub struct PardisoCompatSolver {
    /// Matrix dimension.
    n: usize,
    /// Fill-reducing permutation from the analysis phase (`None` until analysed).
    fill_perm: Option<Permutation>,
    /// Numeric factors of `P·A·Pᵀ` (`None` until factorized).
    factors: Option<LeftLookingLu>,
    /// Threshold pivoting parameter forwarded to the left-looking LU.
    pivot_tol: f64,
}

impl PardisoCompatSolver {
    /// Create a fresh solver for an `n × n` matrix.
    ///
    /// `pivot_tol` is forwarded to the underlying left-looking LU (use `1.0` for
    /// classical partial pivoting; a smaller value in `(0, 1]` trades a little
    /// stability for less fill).
    pub fn new(n: usize, pivot_tol: f64) -> Self {
        Self {
            n,
            fill_perm: None,
            factors: None,
            pivot_tol: pivot_tol.clamp(0.0, 1.0),
        }
    }

    /// Whether the analysis (reordering) phase has been completed.
    pub fn is_analyzed(&self) -> bool {
        self.fill_perm.is_some()
    }

    /// Whether numeric factors are available.
    pub fn is_factorized(&self) -> bool {
        self.factors.is_some()
    }

    /// The fill-reducing permutation chosen during analysis, if any.
    pub fn fill_reducing_permutation(&self) -> Option<&[usize]> {
        self.fill_perm.as_ref().map(|p| p.perm.as_slice())
    }

    /// Run one (or several) PARDISO phases.
    ///
    /// * `row_offsets`, `col_indices`, `values` — the matrix in CSR form. Required
    ///   for the analysis and factorization phases; may be empty slices for a
    ///   solve-only call.
    /// * `rhs` — the right-hand side(s). For a solve phase, length must be a
    ///   multiple of `n` (one or more stacked RHS vectors). For non-solve phases
    ///   it is ignored and may be empty.
    ///
    /// Returns the stacked solution vector(s) when the phase includes a solve,
    /// otherwise an empty `Vec`.
    ///
    /// # Errors
    ///
    /// * [`SolverError::DimensionMismatch`] for bad CSR / RHS lengths.
    /// * [`SolverError::InternalError`] if a phase is requested out of order
    ///   (e.g. solve before factorize).
    /// * Any factorization error ([`SolverError::SingularMatrix`], …).
    pub fn run(
        &mut self,
        phase: Phase,
        row_offsets: &[usize],
        col_indices: &[usize],
        values: &[f64],
        rhs: &[f64],
    ) -> SolverResult<Vec<f64>> {
        let n = self.n;

        // ---- Analysis: compute the fill-reducing permutation. ----
        if phase.does_analysis() {
            if row_offsets.len() != n + 1 {
                return Err(SolverError::DimensionMismatch(format!(
                    "pardiso analysis: row_offsets length {} != n+1 = {}",
                    row_offsets.len(),
                    n + 1
                )));
            }
            self.fill_perm = Some(compute_fill_reducing(row_offsets, col_indices, n)?);
            // Structure changed: invalidate any stale factors.
            self.factors = None;
        }

        // ---- Factorization: factor P·A·Pᵀ. ----
        if phase.does_factorize() {
            let perm = self.fill_perm.as_ref().ok_or_else(|| {
                SolverError::InternalError(
                    "pardiso: factorization requested before analysis".into(),
                )
            })?;
            if row_offsets.len() != n + 1 {
                return Err(SolverError::DimensionMismatch(format!(
                    "pardiso factorize: row_offsets length {} != n+1 = {}",
                    row_offsets.len(),
                    n + 1
                )));
            }
            let (pro, pci, pva) = permute_csr_symmetric(row_offsets, col_indices, values, n, perm);
            let lu = LeftLookingLu::factorize(&pro, &pci, &pva, n, self.pivot_tol)?;
            self.factors = Some(lu);
        }

        // ---- Solve: x = Pᵀ (LU)⁻¹ P b for each stacked RHS. ----
        if phase.does_solve() {
            let perm = self.fill_perm.as_ref().ok_or_else(|| {
                SolverError::InternalError("pardiso: solve requested before analysis".into())
            })?;
            let lu = self.factors.as_ref().ok_or_else(|| {
                SolverError::InternalError("pardiso: solve requested before factorization".into())
            })?;
            if n == 0 {
                return Ok(Vec::new());
            }
            if rhs.len() % n != 0 || rhs.is_empty() {
                return Err(SolverError::DimensionMismatch(format!(
                    "pardiso solve: rhs length {} is not a positive multiple of n = {n}",
                    rhs.len()
                )));
            }
            let nrhs = rhs.len() / n;
            let mut out = vec![0.0f64; rhs.len()];
            for k in 0..nrhs {
                let b = &rhs[k * n..(k + 1) * n];
                // Permute RHS into the fill-reducing order: pb[i] = b[perm[i]].
                let pb: Vec<f64> = (0..n).map(|i| b[perm.perm[i]]).collect();
                let py = lu.solve(&pb)?;
                // Un-permute: x[perm[i]] = py[i].
                let dst = &mut out[k * n..(k + 1) * n];
                for i in 0..n {
                    dst[perm.perm[i]] = py[i];
                }
            }
            return Ok(out);
        }

        Ok(Vec::new())
    }

    /// Convenience: run the entire pipeline ([`Phase::All`]) and return the
    /// solution of `A · x = b`.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`PardisoCompatSolver::run`].
    pub fn solve(
        &mut self,
        row_offsets: &[usize],
        col_indices: &[usize],
        values: &[f64],
        rhs: &[f64],
    ) -> SolverResult<Vec<f64>> {
        self.run(Phase::All, row_offsets, col_indices, values, rhs)
    }
}

/// One-call convenience mirroring `pardiso(..., phase=13, ...)`: factor and solve
/// `A · x = b` for a CSR matrix.
///
/// # Errors
///
/// Propagates any error from [`PardisoCompatSolver::run`].
pub fn pardiso_solve(
    row_offsets: &[usize],
    col_indices: &[usize],
    values: &[f64],
    n: usize,
    rhs: &[f64],
) -> SolverResult<Vec<f64>> {
    let mut solver = PardisoCompatSolver::new(n, 1.0);
    solver.run(Phase::All, row_offsets, col_indices, values, rhs)
}

// ---------------------------------------------------------------------------
// Reordering & permutation helpers
// ---------------------------------------------------------------------------

/// Compute a fill-reducing permutation for the (possibly unsymmetric) CSR matrix.
///
/// Nested dissection operates on the symmetrized structure `A + Aᵀ`. For very
/// small systems (where dissection brings no benefit and the recursive
/// separators are degenerate) the natural ordering is used.
fn compute_fill_reducing(
    row_offsets: &[usize],
    col_indices: &[usize],
    n: usize,
) -> SolverResult<Permutation> {
    if n <= 2 {
        return Ok(Permutation::identity(n));
    }

    // Build the symmetrized adjacency (A + Aᵀ) in i32 CSR for AdjacencyGraph.
    let (sym_ptr, sym_idx) = symmetrize_structure_i32(row_offsets, col_indices, n);
    let graph = AdjacencyGraph::from_symmetric_csr(&sym_ptr, &sym_idx, n);
    NestedDissectionOrdering::compute(&graph)
}

/// Build the symmetrized sparsity structure `struct(A) ∪ struct(Aᵀ)` as i32 CSR
/// (values dropped; the diagonal is implied and excluded by the graph builder).
fn symmetrize_structure_i32(
    row_offsets: &[usize],
    col_indices: &[usize],
    n: usize,
) -> (Vec<i32>, Vec<i32>) {
    let mut rows: Vec<std::collections::BTreeSet<usize>> =
        vec![std::collections::BTreeSet::new(); n];
    for i in 0..n {
        let rs = row_offsets.get(i).copied().unwrap_or(0);
        let re = row_offsets.get(i + 1).copied().unwrap_or(rs);
        for idx in rs..re {
            if let Some(&j) = col_indices.get(idx) {
                if j < n {
                    rows[i].insert(j);
                    rows[j].insert(i); // symmetrize
                }
            }
        }
    }
    let mut ptr = Vec::with_capacity(n + 1);
    let mut idx = Vec::new();
    ptr.push(0i32);
    for set in &rows {
        for &c in set {
            idx.push(c as i32);
        }
        ptr.push(idx.len() as i32);
    }
    (ptr, idx)
}

/// Apply a symmetric permutation `P` to a CSR matrix, returning the CSR triple of
/// `P · A · Pᵀ`.
///
/// With `perm` denoting the forward map (`new_row i = old_row perm[i]`) and
/// `iperm` its inverse, entry `A[r, c]` is placed at `(iperm[r], iperm[c])`.
fn permute_csr_symmetric(
    row_offsets: &[usize],
    col_indices: &[usize],
    values: &[f64],
    n: usize,
    perm: &Permutation,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let iperm = &perm.iperm;
    // Gather permuted entries per new row.
    let mut new_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for r in 0..n {
        let rs = row_offsets.get(r).copied().unwrap_or(0);
        let re = row_offsets.get(r + 1).copied().unwrap_or(rs);
        let nr = iperm[r];
        for idx in rs..re {
            let c = match col_indices.get(idx) {
                Some(&c) if c < n => c,
                _ => continue,
            };
            let v = values.get(idx).copied().unwrap_or(0.0);
            let nc = iperm[c];
            new_rows[nr].push((nc, v));
        }
    }
    // Flatten into CSR with column-sorted rows (and summed duplicates, if any).
    let mut new_ptr = vec![0usize; n + 1];
    let mut new_idx = Vec::new();
    let mut new_val = Vec::new();
    for (i, row) in new_rows.iter_mut().enumerate() {
        row.sort_by_key(|&(c, _)| c);
        let mut last_col: Option<usize> = None;
        for &(c, v) in row.iter() {
            match (Some(c) == last_col, new_val.last_mut()) {
                // Duplicate column in the same row: accumulate into the last cell.
                (true, Some(slot)) => *slot += v,
                // New column (or empty accumulator): start a fresh entry.
                _ => {
                    new_idx.push(c);
                    new_val.push(v);
                    last_col = Some(c);
                }
            }
        }
        new_ptr[i + 1] = new_idx.len();
    }
    (new_ptr, new_idx, new_val)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_to_csr(a: &[Vec<f64>]) -> (Vec<usize>, Vec<usize>, Vec<f64>, usize) {
        let n = a.len();
        let mut row_offsets = vec![0usize; n + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for (i, row) in a.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                if v != 0.0 {
                    col_indices.push(j);
                    values.push(v);
                }
            }
            row_offsets[i + 1] = col_indices.len();
        }
        (row_offsets, col_indices, values, n)
    }

    fn residual_norm(a: &[Vec<f64>], x: &[f64], b: &[f64]) -> f64 {
        let n = b.len();
        let mut s = 0.0;
        for i in 0..n {
            let mut ax = 0.0;
            for j in 0..n {
                ax += a[i][j] * x[j];
            }
            s += (ax - b[i]).powi(2);
        }
        s.sqrt()
    }

    #[test]
    fn phase_code_mapping() {
        assert_eq!(Phase::from_code(11).unwrap(), Phase::Analysis);
        assert_eq!(Phase::from_code(22).unwrap(), Phase::Factorize);
        assert_eq!(Phase::from_code(33).unwrap(), Phase::Solve);
        assert_eq!(Phase::from_code(13).unwrap(), Phase::All);
        assert!(Phase::from_code(99).is_err());
    }

    #[test]
    fn full_pipeline_spd_tridiagonal() {
        let n = 16;
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            a[i][i] = 4.0;
            if i > 0 {
                a[i][i - 1] = -1.0;
            }
            if i + 1 < n {
                a[i][i + 1] = -1.0;
            }
        }
        let x_exact: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        let mut b = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                b[i] += a[i][j] * x_exact[j];
            }
        }
        let (ro, ci, va, _) = dense_to_csr(&a);
        let x = pardiso_solve(&ro, &ci, &va, n, &b).unwrap();
        for i in 0..n {
            assert!(
                (x[i] - x_exact[i]).abs() < 1e-9,
                "x[{i}] = {} exp {}",
                x[i],
                x_exact[i]
            );
        }
        assert!(residual_norm(&a, &x, &b) < 1e-9);
    }

    #[test]
    fn full_pipeline_nonsymmetric() {
        let a = vec![
            vec![0.0, 2.0, 1.0],
            vec![4.0, 1.0, 0.0],
            vec![1.0, 1.0, 3.0],
        ];
        let x_exact = [3.0, 1.0, 2.0];
        let mut b = vec![0.0; 3];
        for i in 0..3 {
            for j in 0..3 {
                b[i] += a[i][j] * x_exact[j];
            }
        }
        let (ro, ci, va, n) = dense_to_csr(&a);
        let x = pardiso_solve(&ro, &ci, &va, n, &b).unwrap();
        for i in 0..3 {
            assert!(
                (x[i] - x_exact[i]).abs() < 1e-10,
                "x[{i}] = {} exp {}",
                x[i],
                x_exact[i]
            );
        }
    }

    #[test]
    fn phased_reuse_refactorize_resolve() {
        // Analyse once, factor, solve; then change values, re-factor, re-solve
        // without re-analysing — the PARDISO reuse pattern.
        let a1 = vec![
            vec![3.0, 1.0, 0.0],
            vec![1.0, 4.0, 1.0],
            vec![0.0, 1.0, 5.0],
        ];
        let (ro, ci, va1, n) = dense_to_csr(&a1);
        let mut solver = PardisoCompatSolver::new(n, 1.0);

        // Phase 11: analysis only.
        let empty = solver.run(Phase::Analysis, &ro, &ci, &va1, &[]).unwrap();
        assert!(empty.is_empty());
        assert!(solver.is_analyzed());
        assert!(!solver.is_factorized());

        // Phase 22: factor.
        solver.run(Phase::Factorize, &ro, &ci, &va1, &[]).unwrap();
        assert!(solver.is_factorized());

        // Phase 33: solve.
        let x_exact = [1.0, 2.0, 3.0];
        let mut b = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                b[i] += a1[i][j] * x_exact[j];
            }
        }
        let x = solver.run(Phase::Solve, &[], &[], &[], &b).unwrap();
        for i in 0..n {
            assert!(
                (x[i] - x_exact[i]).abs() < 1e-10,
                "phase33 x[{i}] = {}",
                x[i]
            );
        }

        // Change values (same structure): re-factor + re-solve with the same
        // analysis. a2 doubles the diagonal.
        let a2 = vec![
            vec![6.0, 1.0, 0.0],
            vec![1.0, 8.0, 1.0],
            vec![0.0, 1.0, 10.0],
        ];
        let (_, _, va2, _) = dense_to_csr(&a2);
        let mut b2 = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                b2[i] += a2[i][j] * x_exact[j];
            }
        }
        let x2 = solver
            .run(Phase::FactorizeSolve, &ro, &ci, &va2, &b2)
            .unwrap();
        for i in 0..n {
            assert!(
                (x2[i] - x_exact[i]).abs() < 1e-10,
                "refactor x[{i}] = {}",
                x2[i]
            );
        }
    }

    #[test]
    fn multiple_right_hand_sides() {
        let a = vec![vec![2.0, 0.0], vec![0.0, 4.0]];
        let (ro, ci, va, n) = dense_to_csr(&a);
        // Two stacked RHS: b0 -> x=[1,1], b1 -> x=[2,3].
        let rhs = vec![2.0, 4.0, /* | */ 4.0, 12.0];
        let x = pardiso_solve(&ro, &ci, &va, n, &rhs).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 1.0).abs() < 1e-12);
        assert!((x[2] - 2.0).abs() < 1e-12 && (x[3] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn solve_before_factorize_errors() {
        let mut solver = PardisoCompatSolver::new(2, 1.0);
        let res = solver.run(Phase::Solve, &[], &[], &[], &[1.0, 2.0]);
        assert!(matches!(res, Err(SolverError::InternalError(_))));
    }

    #[test]
    fn permute_csr_roundtrip_identity() {
        // Permuting with identity must reproduce the matrix structure.
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let (ro, ci, va, n) = dense_to_csr(&a);
        let perm = Permutation::identity(n);
        let (pro, pci, pva) = permute_csr_symmetric(&ro, &ci, &va, n, &perm);
        assert_eq!(pro, ro);
        assert_eq!(pci, ci);
        assert_eq!(pva, va);
    }
}
