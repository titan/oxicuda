//! Two-parameter (bigraded) persistence: the rank invariant of a bi-filtration.
//!
//! A *bi-filtration* assigns to every simplex `σ` a pair of real birth parameters
//! `(a_σ, b_σ)`.  For a comparable pair `(a, b)` in a finite 2-grid the *sub-complex*
//!
//! ```text
//!   K_{a,b} = { σ : a_σ ≤ a  and  b_σ ≤ b }
//! ```
//!
//! is a genuine simplicial complex (closed under faces, because a face never appears
//! after a co-face in either parameter).  As `(a, b)` ranges over the grid these
//! complexes form a commutative bigraded diagram of homology vector spaces, and the
//! whole module `H_k(K_{·,·})` is — unlike the 1-parameter case — generally *not*
//! decomposable into intervals (Carlsson & Zomorodian, *The theory of multidimensional
//! persistence*, 2009).  Two computable invariants are exposed here:
//!
//! * the **Hilbert function** / bigraded Betti grid `dim H_k(K_{a,b})` at every grid
//!   point ([`MultiParameterPersistence::hilbert_function`]); and
//! * the **rank invariant** `rank( H_k(K_u) → H_k(K_v) )` for any comparable pair
//!   `u ≤ v` in the grid ([`MultiParameterPersistence::rank_invariant`]), which is the
//!   complete discrete invariant studied by Carlsson–Zomorodian.
//!
//! ## How the rank invariant is computed
//!
//! For `u ≤ v` the inclusion `K_u ⊆ K_v` induces a linear map on `H_k`.  Build the
//! two-step filtration that inserts every simplex of `K_u` at time `0` and every
//! remaining simplex of `K_v` at time `1`; reduce it with the existing
//! Edelsbrunner–Letscher–Zomorodian column reduction.  A class of `H_k(K_u)` survives
//! into `H_k(K_v)` **iff** it is an *essential* class of this two-step filtration that
//! is born at time `0`; one that dies when the `K_v∖K_u` simplices are added lies in
//! the kernel of the inclusion.  Hence
//!
//! ```text
//!   rank( H_k(K_u) → H_k(K_v) ) = #{ essential k-classes born at time 0 }.
//! ```
//!
//! This reuses [`BoundaryMatrix`], [`reduce_boundary_matrix`] and
//! [`extract_persistence_pairs`] verbatim — no reduction routine is duplicated.

use crate::complex::filtration::{FilteredSimplex, Filtration};
use crate::complex::simplex::Simplex;
use crate::error::{TdaError, TdaResult};
use crate::homology::boundary::BoundaryMatrix;
use crate::homology::persistent::extract_persistence_pairs;
use crate::homology::reduction::reduce_boundary_matrix;
use std::collections::HashSet;

/// Tolerance used for grid-threshold and time-label comparisons.
const GRID_EPS: f64 = 1e-9;

/// A simplex together with its two filtration (birth) parameters `(a_value, b_value)`.
///
/// In a valid [`BiFiltration`] every face of the simplex must carry parameters that
/// are component-wise no larger than `(a_value, b_value)`.
#[derive(Debug, Clone)]
pub struct BigradedSimplex {
    /// The underlying simplex (sorted vertex indices).
    pub simplex: Simplex,
    /// Birth value along the first parameter (e.g. Vietoris–Rips radius).
    pub a_value: f64,
    /// Birth value along the second parameter (e.g. density / codensity).
    pub b_value: f64,
}

/// A bi-filtered simplicial complex evaluated on a finite 2-grid.
///
/// The grids `a_grid` and `b_grid` are the (ascending) parameter values at which the
/// sub-complexes `K_{a,b}` are inspected.
#[derive(Debug, Clone)]
pub struct BiFiltration {
    simplices: Vec<BigradedSimplex>,
    a_grid: Vec<f64>,
    b_grid: Vec<f64>,
}

/// Validate a parameter grid: non-empty, free of NaN, sorted ascending.
fn validate_grid(grid: &[f64], name: &str) -> TdaResult<()> {
    if grid.is_empty() {
        return Err(TdaError::ParameterOutOfRange(format!(
            "{name}-grid is empty"
        )));
    }
    for &v in grid {
        if v.is_nan() {
            return Err(TdaError::NanFiltrationValue);
        }
    }
    for w in grid.windows(2) {
        if w[1] < w[0] {
            return Err(TdaError::FiltrationNotSorted);
        }
    }
    Ok(())
}

impl BiFiltration {
    /// Build a bi-filtration from an explicit list of bigraded simplices and two grids.
    ///
    /// # Errors
    /// * [`TdaError::EmptyComplex`] if `simplices` is empty.
    /// * [`TdaError::ParameterOutOfRange`] if either grid is empty.
    /// * [`TdaError::FiltrationNotSorted`] if either grid is not ascending.
    /// * [`TdaError::NanFiltrationValue`] for any NaN grid value or simplex parameter.
    /// * [`TdaError::InvalidSimplex`] if the same simplex is listed twice.
    /// * [`TdaError::ClosureViolation`] if a face is missing or appears *after* its
    ///   co-face in either parameter (which would break the sub-complex property).
    pub fn new(
        simplices: Vec<BigradedSimplex>,
        a_grid: Vec<f64>,
        b_grid: Vec<f64>,
    ) -> TdaResult<Self> {
        if simplices.is_empty() {
            return Err(TdaError::EmptyComplex);
        }
        validate_grid(&a_grid, "a")?;
        validate_grid(&b_grid, "b")?;
        for bs in &simplices {
            if bs.a_value.is_nan() || bs.b_value.is_nan() {
                return Err(TdaError::NanFiltrationValue);
            }
        }

        // Sorted lookup vertices → (a, b) with duplicate detection.
        let mut lookup: Vec<(Vec<usize>, (f64, f64))> = simplices
            .iter()
            .map(|bs| (bs.simplex.vertices.clone(), (bs.a_value, bs.b_value)))
            .collect();
        lookup.sort_by(|x, y| x.0.cmp(&y.0));
        for w in lookup.windows(2) {
            if w[0].0 == w[1].0 {
                return Err(TdaError::InvalidSimplex(format!(
                    "duplicate simplex {:?}",
                    w[0].0
                )));
            }
        }
        let get = |verts: &[usize]| -> Option<(f64, f64)> {
            lookup
                .binary_search_by(|(v, _)| v.as_slice().cmp(verts))
                .ok()
                .map(|p| lookup[p].1)
        };

        // Closure + bigraded monotonicity: every face present with smaller-or-equal
        // parameters in both coordinates.
        for bs in &simplices {
            for face in bs.simplex.faces() {
                match get(&face.vertices) {
                    None => {
                        return Err(TdaError::ClosureViolation(format!(
                            "face {:?} of {:?} missing from the bi-filtration",
                            face.vertices, bs.simplex.vertices
                        )));
                    }
                    Some((fa, fb)) => {
                        if fa > bs.a_value + GRID_EPS || fb > bs.b_value + GRID_EPS {
                            return Err(TdaError::ClosureViolation(format!(
                                "face {:?} appears after its co-face {:?} in the bi-grading",
                                face.vertices, bs.simplex.vertices
                            )));
                        }
                    }
                }
            }
        }

        Ok(Self {
            simplices,
            a_grid,
            b_grid,
        })
    }

    /// Build a *function–Rips* bi-filtration: parameter `a` is the Vietoris–Rips
    /// radius (the simplex diameter) and parameter `b` is the maximum of a per-vertex
    /// scalar `fn_values` over the simplex's vertices (e.g. a density / codensity).
    ///
    /// The Vietoris–Rips skeleton up to `max_dim` is built with the existing
    /// [`Filtration::vietoris_rips_from_points`] routine and re-graded with the second
    /// parameter, so no new Rips enumeration is introduced.
    ///
    /// # Errors
    /// Grid-validation errors as in [`BiFiltration::new`], plus
    /// [`TdaError::DimensionMismatch`] if `fn_values.len()` differs from the point
    /// count, and any error raised by the Vietoris–Rips construction.
    pub fn vietoris_rips_function(
        points: &[f64],
        n_dims: usize,
        fn_values: &[f64],
        a_grid: Vec<f64>,
        b_grid: Vec<f64>,
        max_dim: usize,
    ) -> TdaResult<Self> {
        validate_grid(&a_grid, "a")?;
        validate_grid(&b_grid, "b")?;
        if n_dims == 0 {
            return Err(TdaError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if points.is_empty() {
            return Err(TdaError::EmptyPointCloud);
        }
        if !points.len().is_multiple_of(n_dims) {
            return Err(TdaError::DimensionMismatch {
                expected: (points.len() / n_dims) * n_dims,
                got: points.len(),
            });
        }
        let n_pts = points.len() / n_dims;
        if fn_values.len() != n_pts {
            return Err(TdaError::DimensionMismatch {
                expected: n_pts,
                got: fn_values.len(),
            });
        }
        for &v in fn_values {
            if v.is_nan() {
                return Err(TdaError::NanFiltrationValue);
            }
        }

        let a_max = a_grid
            .last()
            .copied()
            .ok_or_else(|| TdaError::ParameterOutOfRange("a-grid is empty".to_owned()))?;
        let rips = Filtration::vietoris_rips_from_points(points, n_dims, a_max, max_dim)?;
        let simplices: Vec<BigradedSimplex> = rips
            .simplices
            .iter()
            .map(|fs| {
                let b_value = fs
                    .simplex
                    .vertices
                    .iter()
                    .map(|&v| fn_values[v])
                    .fold(f64::NEG_INFINITY, f64::max);
                BigradedSimplex {
                    simplex: fs.simplex.clone(),
                    a_value: fs.value,
                    b_value,
                }
            })
            .collect();
        Self::new(simplices, a_grid, b_grid)
    }

    /// The ascending grid of first-parameter values.
    pub fn a_grid(&self) -> &[f64] {
        &self.a_grid
    }

    /// The ascending grid of second-parameter values.
    pub fn b_grid(&self) -> &[f64] {
        &self.b_grid
    }

    /// Total number of bigraded simplices.
    pub fn n_simplices(&self) -> usize {
        self.simplices.len()
    }

    /// Indices (into `self.simplices`) of the sub-complex `K_{a,b}`.
    fn subcomplex_indices(&self, a: f64, b: f64) -> Vec<usize> {
        self.simplices
            .iter()
            .enumerate()
            .filter(|(_, bs)| bs.a_value <= a + GRID_EPS && bs.b_value <= b + GRID_EPS)
            .map(|(i, _)| i)
            .collect()
    }

    /// Ordinary `k`-th Betti number of the sub-complex spanned by `indices`
    /// (all placed at a single filtration time), via the standard reduction.
    fn betti_of_subset(&self, indices: &[usize], k: usize) -> TdaResult<usize> {
        if indices.is_empty() {
            return Ok(0);
        }
        let fs: Vec<FilteredSimplex> = indices
            .iter()
            .map(|&i| FilteredSimplex {
                simplex: self.simplices[i].simplex.clone(),
                value: 0.0,
            })
            .collect();
        let filt = Filtration::new(fs)?;
        let mut bm = BoundaryMatrix::from_filtration(&filt)?;
        reduce_boundary_matrix(&mut bm);
        let pairs = extract_persistence_pairs(&bm, &filt)?;
        // For a complex placed at one filtration time the essential classes of
        // dimension k are exactly a basis of H_k, so their count is β_k.
        Ok(pairs
            .iter()
            .filter(|p| p.death.is_none() && p.dim == k)
            .count())
    }

    /// `rank( H_k(K_u) → H_k(K_v) )` for grid indices `u = (ia, ib) ≤ v = (ja, jb)`.
    ///
    /// # Errors
    /// [`TdaError::ParameterOutOfRange`] if an index is out of range or `u ⩽̸ v`.
    fn rank_invariant_impl(
        &self,
        k: usize,
        u: (usize, usize),
        v: (usize, usize),
    ) -> TdaResult<usize> {
        let (ia, ib) = u;
        let (ja, jb) = v;
        let na = self.a_grid.len();
        let nb = self.b_grid.len();
        if ia >= na || ja >= na || ib >= nb || jb >= nb {
            return Err(TdaError::ParameterOutOfRange(
                "grid index out of range".to_owned(),
            ));
        }
        if ia > ja || ib > jb {
            return Err(TdaError::ParameterOutOfRange(
                "u must be component-wise ≤ v in the 2-grid".to_owned(),
            ));
        }

        let u_set: HashSet<usize> = self
            .subcomplex_indices(self.a_grid[ia], self.b_grid[ib])
            .into_iter()
            .collect();
        let v_idx = self.subcomplex_indices(self.a_grid[ja], self.b_grid[jb]);
        if v_idx.is_empty() {
            return Ok(0);
        }

        // Two-step filtration: K_u at time 0, K_v∖K_u at time 1.
        let fs: Vec<FilteredSimplex> = v_idx
            .iter()
            .map(|&i| FilteredSimplex {
                simplex: self.simplices[i].simplex.clone(),
                value: if u_set.contains(&i) { 0.0 } else { 1.0 },
            })
            .collect();
        let filt = Filtration::new(fs)?;
        let mut bm = BoundaryMatrix::from_filtration(&filt)?;
        reduce_boundary_matrix(&mut bm);
        let pairs = extract_persistence_pairs(&bm, &filt)?;

        // Essential k-classes born at time 0 = classes of H_k(K_u) that survive in K_v.
        Ok(pairs
            .iter()
            .filter(|p| p.death.is_none() && p.dim == k && p.birth.abs() < GRID_EPS)
            .count())
    }
}

/// The Hilbert function (bigraded Betti numbers) of a bi-filtration in one homological
/// dimension: `dim H_dim(K_{a_i, b_j})` for every grid cell `(i, j)`.
#[derive(Debug, Clone)]
pub struct HilbertFunction {
    /// Homological dimension `k`.
    pub dim: usize,
    /// Number of first-parameter grid values.
    pub n_a: usize,
    /// Number of second-parameter grid values.
    pub n_b: usize,
    /// Row-major grid: `values[i * n_b + j] = dim H_dim(K_{a_i, b_j})`.
    pub values: Vec<usize>,
}

impl HilbertFunction {
    /// Betti number at grid cell `(ia, ib)`, or `None` if out of range.
    pub fn get(&self, ia: usize, ib: usize) -> Option<usize> {
        if ia < self.n_a && ib < self.n_b {
            Some(self.values[ia * self.n_b + ib])
        } else {
            None
        }
    }

    /// Sum of all Betti numbers over the grid (the total bigraded dimension).
    pub fn total(&self) -> usize {
        self.values.iter().sum()
    }
}

/// Two-parameter persistence computation over a [`BiFiltration`].
///
/// Bundles the bi-filtration with a maximum homological dimension and exposes the
/// Hilbert function and the rank invariant.
#[derive(Debug, Clone)]
pub struct MultiParameterPersistence {
    bifiltration: BiFiltration,
    max_dim: usize,
}

impl MultiParameterPersistence {
    /// Wrap a bi-filtration; `max_dim` is the largest homological dimension that may
    /// be queried.
    ///
    /// # Errors
    /// [`TdaError::DimensionTooLarge`] if `max_dim > 6`.
    pub fn new(bifiltration: BiFiltration, max_dim: usize) -> TdaResult<Self> {
        if max_dim > 6 {
            return Err(TdaError::DimensionTooLarge(max_dim));
        }
        Ok(Self {
            bifiltration,
            max_dim,
        })
    }

    /// Borrow the underlying bi-filtration.
    pub fn bifiltration(&self) -> &BiFiltration {
        &self.bifiltration
    }

    /// The largest queryable homological dimension.
    pub fn max_dim(&self) -> usize {
        self.max_dim
    }

    /// The Hilbert function `dim H_k(K_{a_i, b_j})` over the whole grid.
    ///
    /// # Errors
    /// [`TdaError::DimensionTooLarge`] if `k > max_dim`; reduction errors are
    /// propagated.
    pub fn hilbert_function(&self, k: usize) -> TdaResult<HilbertFunction> {
        if k > self.max_dim {
            return Err(TdaError::DimensionTooLarge(k));
        }
        let bf = &self.bifiltration;
        let n_a = bf.a_grid.len();
        let n_b = bf.b_grid.len();
        let mut values = vec![0usize; n_a * n_b];
        for (i, &av) in bf.a_grid.iter().enumerate() {
            for (j, &bv) in bf.b_grid.iter().enumerate() {
                let idx = bf.subcomplex_indices(av, bv);
                values[i * n_b + j] = bf.betti_of_subset(&idx, k)?;
            }
        }
        Ok(HilbertFunction {
            dim: k,
            n_a,
            n_b,
            values,
        })
    }

    /// The rank invariant `rank( H_k(K_u) → H_k(K_v) )` for comparable grid indices
    /// `u = (ia, ib) ≤ v = (ja, jb)`.
    ///
    /// Note `rank_invariant(k, u, u)` equals `dim H_k(K_u)`, i.e. the Hilbert value at
    /// `u`.
    ///
    /// # Errors
    /// [`TdaError::DimensionTooLarge`] if `k > max_dim`; [`TdaError::ParameterOutOfRange`]
    /// if an index is out of range or `u ⩽̸ v`.
    pub fn rank_invariant(
        &self,
        k: usize,
        u: (usize, usize),
        v: (usize, usize),
    ) -> TdaResult<usize> {
        if k > self.max_dim {
            return Err(TdaError::DimensionTooLarge(k));
        }
        self.bifiltration.rank_invariant_impl(k, u, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A "filled triangle" bi-filtration that is trivial in the second parameter:
    /// 3 vertices at a=0; 3 edges at a=1 (closing a loop); the 2-cell at a=2 (filling
    /// it).  All b-values are 0, so it factors as a 1-parameter filtration in `a`.
    fn triangle_bifiltration() -> BiFiltration {
        let v = |i: usize| Simplex::new(vec![i]).expect("vertex");
        let e = |i: usize, j: usize| Simplex::new(vec![i, j]).expect("edge");
        let simplices = vec![
            BigradedSimplex {
                simplex: v(0),
                a_value: 0.0,
                b_value: 0.0,
            },
            BigradedSimplex {
                simplex: v(1),
                a_value: 0.0,
                b_value: 0.0,
            },
            BigradedSimplex {
                simplex: v(2),
                a_value: 0.0,
                b_value: 0.0,
            },
            BigradedSimplex {
                simplex: e(0, 1),
                a_value: 1.0,
                b_value: 0.0,
            },
            BigradedSimplex {
                simplex: e(1, 2),
                a_value: 1.0,
                b_value: 0.0,
            },
            BigradedSimplex {
                simplex: e(0, 2),
                a_value: 1.0,
                b_value: 0.0,
            },
            BigradedSimplex {
                simplex: Simplex::new(vec![0, 1, 2]).expect("triangle"),
                a_value: 2.0,
                b_value: 0.0,
            },
        ];
        BiFiltration::new(simplices, vec![0.0, 1.0, 2.0], vec![0.0]).expect("valid bifiltration")
    }

    /// Independent 1-parameter persistent Betti number β_k^{a_i, a_j} from a filtration.
    fn pers_betti_1param(filt: &Filtration, k: usize, ai: f64, aj: f64) -> usize {
        let mut bm = BoundaryMatrix::from_filtration(filt).expect("bm");
        reduce_boundary_matrix(&mut bm);
        let pairs = extract_persistence_pairs(&bm, filt).expect("pairs");
        pairs
            .iter()
            .filter(|p| {
                p.dim == k
                    && p.birth <= ai + GRID_EPS
                    && match p.death {
                        Some(d) => d > aj + GRID_EPS,
                        None => true,
                    }
            })
            .count()
    }

    fn triangle_1param() -> Filtration {
        let v = |i: usize| Simplex::new(vec![i]).expect("vertex");
        let e = |i: usize, j: usize| Simplex::new(vec![i, j]).expect("edge");
        Filtration::new(vec![
            FilteredSimplex {
                simplex: v(0),
                value: 0.0,
            },
            FilteredSimplex {
                simplex: v(1),
                value: 0.0,
            },
            FilteredSimplex {
                simplex: v(2),
                value: 0.0,
            },
            FilteredSimplex {
                simplex: e(0, 1),
                value: 1.0,
            },
            FilteredSimplex {
                simplex: e(1, 2),
                value: 1.0,
            },
            FilteredSimplex {
                simplex: e(0, 2),
                value: 1.0,
            },
            FilteredSimplex {
                simplex: Simplex::new(vec![0, 1, 2]).expect("triangle"),
                value: 2.0,
            },
        ])
        .expect("filt")
    }

    // (a) Product / factoring bi-filtration ⇒ rank invariant matches 1-parameter
    //     persistence along the active axis.
    #[test]
    fn rank_invariant_matches_one_parameter() {
        let bf = triangle_bifiltration();
        let mpp = MultiParameterPersistence::new(bf, 2).expect("mpp");
        let filt = triangle_1param();

        // a-axis is index 0..3 (values 0,1,2); b is fixed at index 0.
        let checks = [
            (0usize, 0.0, 2.0, (0usize, 0usize), (2usize, 0usize)),
            (0, 0.0, 0.0, (0, 0), (0, 0)),
            (1, 1.0, 1.0, (1, 0), (1, 0)),
            (1, 1.0, 2.0, (1, 0), (2, 0)),
        ];
        for (k, ai, aj, u, v) in checks {
            let ri = mpp.rank_invariant(k, u, v).expect("ri");
            let pb = pers_betti_1param(&filt, k, ai, aj);
            assert_eq!(ri, pb, "k={k} ai={ai} aj={aj}: rank {ri} != 1-param β {pb}");
        }

        // Concrete topology: loop born at a=1, killed at a=2.
        assert_eq!(mpp.rank_invariant(1, (1, 0), (1, 0)).expect("ri"), 1);
        assert_eq!(mpp.rank_invariant(1, (1, 0), (2, 0)).expect("ri"), 0);
        // Three components merge into one image class.
        assert_eq!(mpp.rank_invariant(0, (0, 0), (2, 0)).expect("ri"), 1);
    }

    // (b) A single point across the grid: H_0 rank 1 everywhere, no higher homology.
    #[test]
    fn single_point_trivial_higher_homology() {
        let bf = BiFiltration::new(
            vec![BigradedSimplex {
                simplex: Simplex::new(vec![0]).expect("vertex"),
                a_value: 0.0,
                b_value: 0.0,
            }],
            vec![0.0, 1.0],
            vec![0.0, 1.0],
        )
        .expect("bifiltration");
        let mpp = MultiParameterPersistence::new(bf, 2).expect("mpp");

        let h0 = mpp.hilbert_function(0).expect("h0");
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(h0.get(i, j), Some(1), "H0 must be 1 at ({i},{j})");
            }
        }
        let h1 = mpp.hilbert_function(1).expect("h1");
        assert_eq!(h1.total(), 0, "no H1 for a single point");

        assert_eq!(mpp.rank_invariant(0, (0, 0), (1, 1)).expect("ri"), 1);
        assert_eq!(mpp.rank_invariant(1, (0, 0), (1, 1)).expect("ri"), 0);
        // Self rank equals the Hilbert value.
        assert_eq!(mpp.rank_invariant(0, (1, 1), (1, 1)).expect("ri"), 1);
    }

    // (c) Monotonicity: for comparable u ≤ v ≤ w the rank is non-increasing as the
    //     interval grows, rank(u→w) ≤ min(rank(u→v), rank(v→w)).
    #[test]
    fn rank_invariant_monotone() {
        let bf = triangle_bifiltration();
        let mpp = MultiParameterPersistence::new(bf, 2).expect("mpp");

        let a_pts = [(0usize, 0usize), (1, 0), (2, 0)];
        for k in 0..=1usize {
            for &u in &a_pts {
                for &v in &a_pts {
                    if v.0 < u.0 {
                        continue;
                    }
                    for &w in &a_pts {
                        if w.0 < v.0 {
                            continue;
                        }
                        let ruv = mpp.rank_invariant(k, u, v).expect("ruv");
                        let rvw = mpp.rank_invariant(k, v, w).expect("rvw");
                        let ruw = mpp.rank_invariant(k, u, w).expect("ruw");
                        assert!(
                            ruw <= ruv && ruw <= rvw,
                            "k={k}: rank({u:?}->{w:?})={ruw} not ≤ min({ruv},{rvw})"
                        );
                    }
                }
            }
        }
        // Explicit strict drop: the loop present at a=1 is filled by a=2.
        assert_eq!(mpp.rank_invariant(1, (1, 0), (1, 0)).expect("ri"), 1);
        assert_eq!(mpp.rank_invariant(1, (1, 0), (2, 0)).expect("ri"), 0);
    }

    // (d) A circle (regular hexagon) with a non-trivial second parameter: the H1 loop
    //     only appears once the radius admits all edges AND the density threshold
    //     admits the last vertex.
    #[test]
    fn circle_h1_in_expected_region() {
        // Regular hexagon, circumradius 1 ⇒ side = 1, short diagonal = √3 ≈ 1.732.
        let n = 6usize;
        let pts: Vec<f64> = (0..n)
            .flat_map(|i| {
                let t = std::f64::consts::PI * (i as f64) / 3.0;
                vec![t.cos(), t.sin()]
            })
            .collect();
        // Vertex 5 carries a higher codensity, so it only enters at the top b-level.
        let fn_values = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        // a-grid: 0 (vertices only) < 1.05 (all sides, no diagonals) < 1.5 (still no diagonals).
        let a_grid = vec![0.0, 1.05, 1.5];
        let b_grid = vec![0.0, 1.0];
        let bf = BiFiltration::vietoris_rips_function(&pts, 2, &fn_values, a_grid, b_grid, 2)
            .expect("bf");
        let mpp = MultiParameterPersistence::new(bf, 2).expect("mpp");

        let h1 = mpp.hilbert_function(1).expect("h1");
        // a index 1 = radius 1.05; b index 1 = full density ⇒ full hexagon loop.
        assert_eq!(
            h1.get(1, 1),
            Some(1),
            "expected one H1 loop at (1.05, full)"
        );
        // Same radius but vertex 5 excluded ⇒ broken loop, no H1.
        assert_eq!(h1.get(1, 0), Some(0), "no loop when a vertex is missing");
        // No radius ⇒ no loop.
        assert_eq!(h1.get(0, 1), Some(0), "no loop at radius 0");

        // The loop persists as the radius grows (b at full density).
        assert_eq!(mpp.rank_invariant(1, (1, 1), (2, 1)).expect("ri"), 1);
    }

    // (e) Malformed grids / dimensions / comparability errors.
    #[test]
    fn malformed_inputs_error() {
        let vtx = || BigradedSimplex {
            simplex: Simplex::new(vec![0]).expect("v"),
            a_value: 0.0,
            b_value: 0.0,
        };

        // Empty grid.
        assert!(BiFiltration::new(vec![vtx()], vec![], vec![0.0]).is_err());
        // Unsorted grid.
        assert!(BiFiltration::new(vec![vtx()], vec![1.0, 0.0], vec![0.0]).is_err());
        // NaN grid value.
        assert!(BiFiltration::new(vec![vtx()], vec![f64::NAN], vec![0.0]).is_err());
        // Empty simplex set.
        assert!(BiFiltration::new(vec![], vec![0.0], vec![0.0]).is_err());
        // Missing face (edge without its vertices).
        let bad_edge = BigradedSimplex {
            simplex: Simplex::new(vec![0, 1]).expect("e"),
            a_value: 1.0,
            b_value: 0.0,
        };
        assert!(BiFiltration::new(vec![bad_edge], vec![0.0, 1.0], vec![0.0]).is_err());

        // Valid bi-filtration but bad queries.
        let bf = triangle_bifiltration();
        let mpp = MultiParameterPersistence::new(bf, 2).expect("mpp");
        // u not ≤ v.
        assert!(mpp.rank_invariant(0, (2, 0), (0, 0)).is_err());
        // index out of range.
        assert!(mpp.rank_invariant(0, (0, 0), (9, 0)).is_err());
        // dimension beyond max_dim.
        assert!(mpp.rank_invariant(5, (0, 0), (0, 0)).is_err());
        assert!(mpp.hilbert_function(5).is_err());

        // max_dim too large.
        let bf2 = triangle_bifiltration();
        assert!(MultiParameterPersistence::new(bf2, 7).is_err());
    }
}
