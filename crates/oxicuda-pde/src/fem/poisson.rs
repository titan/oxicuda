//! 2D Poisson equation −Δu = f on `[0,lx]×[0,ly]` with Dirichlet BC u=0 on ∂Ω.
//!
//! Uses piecewise-linear (P1) triangular finite elements on a structured
//! Cartesian mesh. The global stiffness matrix is stored as a dense [n_dof×n_dof]
//! array (row-major) and the linear system is solved via an inline dense Cholesky
//! (LL^T) decomposition – no external crates needed.
//!
//! # Reference
//! Brenner & Scott, "The Mathematical Theory of Finite Element Methods", §3.

use crate::error::{PdeError, PdeResult};
use crate::fem::p1_triangle::p1_local_stiffness;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for a 2D Poisson FEM solver on a rectangular domain.
#[derive(Debug, Clone)]
pub struct PoissonFemConfig {
    /// Number of grid points (nodes) in the x direction (≥ 2).
    pub n_x: usize,
    /// Number of grid points (nodes) in the y direction (≥ 2).
    pub n_y: usize,
    /// Domain width:  x ∈ [0, lx].
    pub lx: f64,
    /// Domain height: y ∈ [0, ly].
    pub ly: f64,
}

// ─── Solver struct ────────────────────────────────────────────────────────────

/// 2D Poisson FEM solver using P1 triangles on a structured Cartesian mesh.
///
/// Nodes are numbered row-major: `node_id(i,j) = j*n_x + i`
/// where `i ∈ [0, n_x)` (x-direction) and `j ∈ [0, n_y)` (y-direction).
///
/// Boundary nodes (i==0, i==n_x-1, j==0, j==n_y-1) carry a Dirichlet
/// condition u = 0 and are **eliminated** before assembly, so `n_dof`
/// counts only the interior nodes.
#[derive(Debug)]
pub struct PoissonFem {
    /// Problem configuration.
    pub config: PoissonFemConfig,
    /// Coordinates of every node in the mesh, indexed by global node id.
    pub nodes: Vec<(f64, f64)>,
    /// Triangulation: each entry lists the 3 global node ids of a triangle.
    pub elements: Vec<[usize; 3]>,
    /// Number of interior (free) degrees of freedom.
    pub n_dof: usize,

    /// Global node ids of the interior nodes, in ascending order.
    interior_ids: Vec<usize>,
    /// `global_to_dof[k]` = DOF index of global node `k`, or `usize::MAX`
    /// if node `k` is a boundary node (treated as sentinel for "no DOF").
    global_to_dof: Vec<usize>,
}

impl PoissonFem {
    /// Construct the mesh and DOF numbering for the given configuration.
    ///
    /// # Errors
    /// Returns [`PdeError::InvalidParameter`] if `n_x < 2` or `n_y < 2`.
    pub fn new(config: PoissonFemConfig) -> PdeResult<Self> {
        let n_x = config.n_x;
        let n_y = config.n_y;

        if n_x < 2 {
            return Err(PdeError::InvalidParameter {
                name: "n_x".into(),
                reason: format!("must be >= 2, got {n_x}"),
            });
        }
        if n_y < 2 {
            return Err(PdeError::InvalidParameter {
                name: "n_y".into(),
                reason: format!("must be >= 2, got {n_y}"),
            });
        }

        let dx = config.lx / (n_x - 1) as f64;
        let dy = config.ly / (n_y - 1) as f64;
        let n_total = n_x * n_y;

        // Build node coordinate list (row-major: id = j*n_x + i).
        let mut nodes = Vec::with_capacity(n_total);
        for j in 0..n_y {
            for i in 0..n_x {
                nodes.push((i as f64 * dx, j as f64 * dy));
            }
        }

        // Build triangulation: each cell (i,j)..(i+1,j+1) → 2 triangles.
        //   lower: (i,j), (i+1,j), (i+1,j+1)
        //   upper: (i,j), (i+1,j+1), (i,j+1)
        let n_cells = (n_x - 1) * (n_y - 1);
        let mut elements = Vec::with_capacity(2 * n_cells);
        for j in 0..n_y - 1 {
            for i in 0..n_x - 1 {
                let id = |ii: usize, jj: usize| jj * n_x + ii;
                let a = id(i, j);
                let b = id(i + 1, j);
                let c = id(i + 1, j + 1);
                let d = id(i, j + 1);
                elements.push([a, b, c]); // lower triangle
                elements.push([a, c, d]); // upper triangle
            }
        }

        // Identify interior nodes and build DOF map.
        let mut global_to_dof = vec![usize::MAX; n_total];
        let mut interior_ids = Vec::new();
        for j in 0..n_y {
            for i in 0..n_x {
                let is_boundary = i == 0 || i == n_x - 1 || j == 0 || j == n_y - 1;
                if !is_boundary {
                    let id = j * n_x + i;
                    global_to_dof[id] = interior_ids.len();
                    interior_ids.push(id);
                }
            }
        }
        let n_dof = interior_ids.len();

        Ok(Self {
            config,
            nodes,
            elements,
            n_dof,
            interior_ids,
            global_to_dof,
        })
    }

    /// Total number of nodes in the mesh (including boundary nodes).
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Total number of triangular elements.
    #[inline]
    pub fn n_elements(&self) -> usize {
        self.elements.len()
    }

    // ─── Matrix / vector assembly ──────────────────────────────────────────

    /// Assemble the global stiffness matrix restricted to interior DOFs.
    ///
    /// Returns a dense `n_dof × n_dof` matrix stored row-major.
    ///
    /// # Errors
    /// Propagates any error from [`p1_local_stiffness`].
    pub fn assemble_stiffness(&self) -> PdeResult<Vec<f64>> {
        let nd = self.n_dof;
        let mut k_global = vec![0.0_f64; nd * nd];

        for &tri in &self.elements {
            let (x0, y0) = self.nodes[tri[0]];
            let (x1, y1) = self.nodes[tri[1]];
            let (x2, y2) = self.nodes[tri[2]];

            let k_local = p1_local_stiffness(x0, y0, x1, y1, x2, y2)?;

            for a in 0..3 {
                let dof_a = self.global_to_dof[tri[a]];
                if dof_a == usize::MAX {
                    continue; // boundary node, skip
                }
                for b in 0..3 {
                    let dof_b = self.global_to_dof[tri[b]];
                    if dof_b == usize::MAX {
                        continue; // boundary node, skip
                    }
                    k_global[dof_a * nd + dof_b] += k_local[a * 3 + b];
                }
            }
        }

        Ok(k_global)
    }

    /// Assemble the load vector for a right-hand side function `f(x, y)`.
    ///
    /// Uses the centroid quadrature rule:
    /// `b_i ≈ f(x_c, y_c) * Area / 3` for each interior node `i` in
    /// the triangle.
    ///
    /// Returns a `[n_dof]` vector.
    ///
    /// # Errors
    /// Returns [`PdeError::InvalidGrid`] if a triangle has zero area
    /// (degenerate mesh).
    pub fn assemble_load(&self, f_fn: impl Fn(f64, f64) -> f64) -> PdeResult<Vec<f64>> {
        let nd = self.n_dof;
        let mut load = vec![0.0_f64; nd];

        for &tri in &self.elements {
            let (x0, y0) = self.nodes[tri[0]];
            let (x1, y1) = self.nodes[tri[1]];
            let (x2, y2) = self.nodes[tri[2]];

            let twice_area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
            if twice_area.abs() < 1.0e-15 {
                return Err(PdeError::InvalidGrid(format!(
                    "degenerate triangle with vertices ({x0},{y0}), ({x1},{y1}), ({x2},{y2})"
                )));
            }
            let area = 0.5 * twice_area.abs();

            // Centroid of the triangle.
            let xc = (x0 + x1 + x2) / 3.0;
            let yc = (y0 + y1 + y2) / 3.0;
            let f_val = f_fn(xc, yc);
            let contrib = f_val * area / 3.0;

            for &gn in tri.iter() {
                let dof = self.global_to_dof[gn];
                if dof != usize::MAX {
                    load[dof] += contrib;
                }
            }
        }

        Ok(load)
    }

    // ─── Solver ────────────────────────────────────────────────────────────

    /// Solve −Δu = f(x,y) in Ω, u = 0 on ∂Ω.
    ///
    /// Assembles the stiffness matrix K and load vector b, then solves
    /// `K u = b` by dense Cholesky (LL^T) factorisation.
    ///
    /// Returns the `[n_dof]` coefficient vector of the discrete solution
    /// at the interior nodes (indexed by [`Self::interior_ids`]).
    ///
    /// # Errors
    /// - [`PdeError::InvalidParameter`] if the mesh has 0 DOFs (pure boundary mesh).
    /// - [`PdeError::SingularMatrix`] if Cholesky detects a non-positive pivot.
    /// - Propagates any assembly error.
    pub fn solve(&self, f_fn: impl Fn(f64, f64) -> f64) -> PdeResult<Vec<f64>> {
        if self.n_dof == 0 {
            return Err(PdeError::InvalidParameter {
                name: "n_dof".into(),
                reason: "no interior degrees of freedom; mesh is too coarse".into(),
            });
        }
        let k = self.assemble_stiffness()?;
        let b = self.assemble_load(f_fn)?;
        cholesky_solve(&k, &b, self.n_dof)
    }

    /// Access the DOF-index of a global node id (returns `None` for boundary nodes).
    #[inline]
    pub fn dof_of(&self, global_id: usize) -> Option<usize> {
        let d = self.global_to_dof[global_id];
        if d == usize::MAX { None } else { Some(d) }
    }

    /// Ordered list of global node ids for the interior DOFs.
    #[inline]
    pub fn interior_ids(&self) -> &[usize] {
        &self.interior_ids
    }
}

// ─── Inline Cholesky solver ────────────────────────────────────────────────────

/// Dense LL^T Cholesky factorisation and solve for an n×n SPD matrix `a`
/// stored row-major, and a right-hand side vector `b` of length n.
///
/// Returns x s.t. A x = b.
///
/// # Errors
/// Returns [`PdeError::SingularMatrix`] if any pivot is ≤ 0 or its square
/// root is not finite (matrix is not positive definite).
fn cholesky_solve(a: &[f64], b: &[f64], n: usize) -> PdeResult<Vec<f64>> {
    // ── Factorisation: compute L (lower triangular) in-place ──────────────
    // We store L as a flat n×n row-major array.
    let mut l = vec![0.0_f64; n * n];

    for j in 0..n {
        // Diagonal element L[j,j].
        let mut diag = a[j * n + j];
        for k in 0..j {
            diag -= l[j * n + k] * l[j * n + k];
        }
        if diag <= 0.0 {
            return Err(PdeError::SingularMatrix(format!(
                "Cholesky: non-positive pivot {diag:.3e} at column {j}"
            )));
        }
        let ljj = diag.sqrt();
        if !ljj.is_finite() {
            return Err(PdeError::SingularMatrix(format!(
                "Cholesky: sqrt of pivot is not finite at column {j}"
            )));
        }
        l[j * n + j] = ljj;

        // Sub-diagonal elements L[i,j] for i > j.
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / ljj;
        }
    }

    // ── Forward substitution: L y = b ─────────────────────────────────────
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }

    // ── Back substitution: L^T x = y ──────────────────────────────────────
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in i + 1..n {
            s -= l[k * n + i] * x[k]; // L^T[i,k] = L[k,i]
        }
        x[i] = s / l[i * n + i];
    }

    Ok(x)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config_3x3() -> PoissonFemConfig {
        PoissonFemConfig {
            n_x: 3,
            n_y: 3,
            lx: 1.0,
            ly: 1.0,
        }
    }

    // ── Mesh topology ──────────────────────────────────────────────────────

    #[test]
    fn n_nodes_correct() {
        let fem = PoissonFem::new(default_config_3x3()).expect("construct 3x3");
        assert_eq!(fem.n_nodes(), 9, "3x3 grid must have 9 nodes");
    }

    #[test]
    fn n_elements_correct() {
        let fem = PoissonFem::new(default_config_3x3()).expect("construct 3x3");
        // 2 cells × 2 cells × 2 triangles/cell = 8
        assert_eq!(fem.n_elements(), 8, "3x3 grid must have 8 triangles");
    }

    // ── Stiffness matrix ───────────────────────────────────────────────────

    #[test]
    fn stiffness_shape() {
        let fem = PoissonFem::new(default_config_3x3()).expect("ok");
        let k = fem.assemble_stiffness().expect("assemble stiffness");
        let nd = fem.n_dof;
        assert_eq!(k.len(), nd * nd, "stiffness must be n_dof × n_dof");
    }

    #[test]
    fn stiffness_spd() {
        // Use a larger mesh so n_dof > 1 for the symmetry check.
        let cfg = PoissonFemConfig {
            n_x: 5,
            n_y: 5,
            lx: 1.0,
            ly: 1.0,
        };
        let fem = PoissonFem::new(cfg).expect("ok");
        let k = fem.assemble_stiffness().expect("assemble stiffness");
        let nd = fem.n_dof;

        // Check symmetry: |K[i,j] - K[j,i]| < eps for all i,j.
        for i in 0..nd {
            for j in 0..nd {
                assert!(
                    (k[i * nd + j] - k[j * nd + i]).abs() < 1.0e-12,
                    "K not symmetric at ({i},{j}): {} vs {}",
                    k[i * nd + j],
                    k[j * nd + i]
                );
            }
        }

        // Check all diagonal entries are positive.
        for i in 0..nd {
            assert!(
                k[i * nd + i] > 0.0,
                "diagonal K[{i},{i}] = {} ≤ 0",
                k[i * nd + i]
            );
        }
    }

    // ── Load vector ────────────────────────────────────────────────────────

    #[test]
    fn load_shape() {
        let fem = PoissonFem::new(default_config_3x3()).expect("ok");
        let f = fem.assemble_load(|_, _| 1.0).expect("assemble load");
        assert_eq!(f.len(), fem.n_dof, "load must have n_dof entries");
    }

    // ── Solve ──────────────────────────────────────────────────────────────

    #[test]
    fn solve_shape() {
        let fem = PoissonFem::new(default_config_3x3()).expect("ok");
        let u = fem.solve(|_, _| 1.0).expect("solve");
        assert_eq!(u.len(), fem.n_dof, "solution must have n_dof entries");
    }

    #[test]
    fn solve_finite() {
        let cfg = PoissonFemConfig {
            n_x: 5,
            n_y: 5,
            lx: 1.0,
            ly: 1.0,
        };
        let fem = PoissonFem::new(cfg).expect("ok");
        let u = fem.solve(|_, _| 1.0).expect("solve");
        for (k, &v) in u.iter().enumerate() {
            assert!(v.is_finite(), "u[{k}] = {v} is not finite");
        }
    }

    #[test]
    fn constant_rhs_monotone() {
        // For f=1 with zero Dirichlet BC the solution is ≥ 0 inside the domain.
        let cfg = PoissonFemConfig {
            n_x: 5,
            n_y: 5,
            lx: 1.0,
            ly: 1.0,
        };
        let fem = PoissonFem::new(cfg).expect("ok");
        let u = fem.solve(|_, _| 1.0).expect("solve");
        for (k, &v) in u.iter().enumerate() {
            assert!(
                v > 0.0,
                "interior solution u[{k}] = {v} should be > 0 for f=1"
            );
        }
    }

    #[test]
    fn zero_rhs_zero_solution() {
        let cfg = PoissonFemConfig {
            n_x: 5,
            n_y: 5,
            lx: 1.0,
            ly: 1.0,
        };
        let fem = PoissonFem::new(cfg).expect("ok");
        let u = fem.solve(|_, _| 0.0).expect("solve with f=0");
        for (k, &v) in u.iter().enumerate() {
            assert!(
                v.abs() < 1.0e-12,
                "u[{k}] = {v:.2e} should be zero when f=0"
            );
        }
    }

    // ── Error paths ────────────────────────────────────────────────────────

    #[test]
    fn n_x_1_error() {
        let cfg = PoissonFemConfig {
            n_x: 1,
            n_y: 3,
            lx: 1.0,
            ly: 1.0,
        };
        let result = PoissonFem::new(cfg);
        assert!(result.is_err(), "n_x=1 must return Err");
        match result {
            Err(PdeError::InvalidParameter { name, .. }) => {
                assert_eq!(name, "n_x");
            }
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }

    #[test]
    fn n_y_1_error() {
        let cfg = PoissonFemConfig {
            n_x: 3,
            n_y: 1,
            lx: 1.0,
            ly: 1.0,
        };
        let result = PoissonFem::new(cfg);
        assert!(result.is_err(), "n_y=1 must return Err");
    }

    // ── Accuracy check ─────────────────────────────────────────────────────

    #[test]
    fn unit_source_solution_bounded() {
        // For −Δu = 1 on [0,1]² with u=0 on ∂Ω, the maximum principle
        // guarantees 0 < u < 1/8 (the bound for a disk of the same area).
        let cfg = PoissonFemConfig {
            n_x: 7,
            n_y: 7,
            lx: 1.0,
            ly: 1.0,
        };
        let fem = PoissonFem::new(cfg).expect("ok");
        let u = fem.solve(|_, _| 1.0).expect("solve");
        for &v in &u {
            assert!(
                v > 0.0 && v < 0.15,
                "solution {v} out of expected range (0, 0.15)"
            );
        }
    }
}
