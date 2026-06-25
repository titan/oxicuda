//! Spectral graph sketch: effective-resistance spectral sparsifier.
//!
//! Implements the Spielman–Srivastava (STOC 2008 / SICOMP 2011) spectral
//! sparsifier with the Johnson–Lindenstrauss effective-resistance estimator,
//! following the streaming-friendly viewpoint of Kelner & Levin (2013). Given a
//! weighted undirected graph `G = (V, E, w)` with combinatorial Laplacian
//! `L = Bᵀ W B` (where `B` is the signed edge–vertex incidence matrix and `W`
//! the diagonal weight matrix), the sketch produces a re-weighted subgraph `H`
//! whose Laplacian `L_H` satisfies, with high probability,
//!
//! ```text
//! (1 − ε) · xᵀ L x  ≤  xᵀ L_H x  ≤  (1 + ε) · xᵀ L x      for all x ∈ ℝⁿ .
//! ```
//!
//! # Effective resistance and the Spielman–Srivastava trick
//!
//! The *effective resistance* between `u` and `v` is
//! `R_eff(u, v) = b_{uv}ᵀ L⁺ b_{uv}` with `b_{uv} = e_u − e_v` and `L⁺` the
//! Moore–Penrose pseudoinverse. Spielman & Srivastava observe that with the
//! weighted incidence matrix `B_w = W^{1/2} B` (so `L = B_wᵀ B_w`),
//!
//! ```text
//! R_eff(u, v) = ‖B_w L⁺ b_{uv}‖²  =  ‖(B_w L⁺) (e_u − e_v)‖² .
//! ```
//!
//! Projecting the `m × n` matrix `B_w L⁺` from the left by a random
//! `k × m` Johnson–Lindenstrauss matrix `Q` (entries `±1/√k`) with
//! `k = O(log n / ε²)` preserves all pairwise distances up to `(1 ± ε)`, so
//! defining `Z = Q B_w L⁺ ∈ ℝ^{k×n}` gives
//!
//! ```text
//! R_eff(u, v) ≈ ‖Z (e_u − e_v)‖² = ‖z_u − z_v‖² ,
//! ```
//! where `z_u` is the `u`-th column of `Z`. Each **row** of `Z` is
//! `z_{(i)} = (Q B_w)_{(i)} L⁺`, obtained by solving the Laplacian system
//! `L y = (Q B_w)_{(i)}ᵀ` for `y`. We solve these `k` systems with a
//! null-space-projected **Conjugate Gradient** routine (`L` is singular with
//! null space `span{𝟙}` on each connected component).
//!
//! # Importance sampling and re-weighting
//!
//! With `p_e ∝ w_e · R_eff(e)` and `Σ_e w_e R_eff(e) = rank(L) = n − c`
//! (`c` = number of connected components), each edge is sampled independently
//! with probability `p_e = min(1, q · w_e R_eff(e))` for an oversampling factor
//! `q = ⌈ C · log n / ε² ⌉`; sampled edges are re-weighted by `w_e / p_e`. The
//! re-weighting makes `L_H` an **unbiased** estimator of `L`
//! (`𝔼[L_H] = L`), and the JL/effective-resistance sampling guarantees the
//! spectral bound above with high probability (Spielman–Srivastava 2011,
//! Thm 1; via the matrix-Chernoff analysis of Tropp 2012).

use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;

/// A single weighted undirected edge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Edge {
    /// Lower endpoint (always `< v`).
    pub u: usize,
    /// Higher endpoint.
    pub v: usize,
    /// Positive edge weight.
    pub w: f64,
}

/// Configuration for a [`GraphSketch`].
#[derive(Debug, Clone, Copy)]
pub struct GraphSketchConfig {
    /// Target spectral accuracy `ε ∈ (0, 1)`. Controls both the JL projection
    /// dimension and the edge over-sampling factor.
    pub epsilon: f64,
    /// Multiplicative constant `C` in the JL dimension `k = ⌈C·log n / ε²⌉`.
    /// A value around `4`–`8` is typical; larger ⇒ more accurate, more work.
    pub jl_constant: f64,
    /// Multiplicative constant in the over-sampling factor
    /// `q = ⌈C_s·log n / ε²⌉`. Larger ⇒ denser, more accurate sparsifier.
    pub sample_constant: f64,
    /// Maximum Conjugate-Gradient iterations per Laplacian solve.
    pub cg_max_iter: usize,
    /// Relative residual tolerance for the Conjugate-Gradient solver.
    pub cg_tol: f64,
}

impl Default for GraphSketchConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.5,
            jl_constant: 4.0,
            sample_constant: 8.0,
            cg_max_iter: 0, // 0 ⇒ defaulted to a multiple of n at solve time.
            cg_tol: 1e-10,
        }
    }
}

/// The re-weighted sparsifier returned by [`GraphSketch::sparsify`].
#[derive(Debug, Clone)]
pub struct SparsifiedGraph {
    /// Number of vertices (unchanged from the original graph).
    pub n_vertices: usize,
    /// Sampled, re-weighted edges (a subset of the originals with new weights).
    pub edges: Vec<Edge>,
}

impl SparsifiedGraph {
    /// Evaluate the Laplacian quadratic form `xᵀ L_H x` for this sparsifier.
    ///
    /// # Errors
    /// [`SketchError::DimensionMismatch`] if `x.len() != n_vertices`.
    pub fn quadratic_form(&self, x: &[f64]) -> SketchResult<f64> {
        if x.len() != self.n_vertices {
            return Err(SketchError::DimensionMismatch {
                a: self.n_vertices,
                b: x.len(),
            });
        }
        Ok(laplacian_quadratic_form(&self.edges, x))
    }
}

/// Spectral graph sketch supporting effective-resistance estimation and
/// effective-resistance spectral sparsification.
#[derive(Debug, Clone)]
pub struct GraphSketch {
    /// Number of vertices `n`.
    n: usize,
    /// Configuration.
    cfg: GraphSketchConfig,
    /// Accumulated edges (endpoints kept canonical with `u < v`).
    edges: Vec<Edge>,
    /// Index of an existing edge by `(u, v)` key for weight accumulation.
    edge_index: std::collections::HashMap<(usize, usize), usize>,
    /// RNG for the JL projection and importance sampling.
    rng: LcgRng,
    /// Cached `Z = Q B_w L⁺` (`k × n`, row-major), computed by `build_z`.
    z_cache: Option<Vec<f64>>,
    /// Number of JL rows `k` matching `z_cache`.
    z_rows: usize,
}

impl GraphSketch {
    /// Create a sketch for an `n`-vertex graph.
    ///
    /// # Errors
    /// * [`SketchError::InvalidParameter`] — `n < 2`, or `ε ∉ (0, 1)`, or any
    ///   structural constant non-positive / non-finite.
    pub fn new(n_vertices: usize, cfg: GraphSketchConfig, rng: LcgRng) -> SketchResult<Self> {
        if n_vertices < 2 {
            return Err(SketchError::InvalidParameter {
                name: "n_vertices".to_string(),
                reason: "must be >= 2".to_string(),
            });
        }
        if !(cfg.epsilon.is_finite() && cfg.epsilon > 0.0 && cfg.epsilon < 1.0) {
            return Err(SketchError::InvalidParameter {
                name: "epsilon".to_string(),
                reason: "must lie in (0, 1)".to_string(),
            });
        }
        if !(cfg.jl_constant.is_finite() && cfg.jl_constant > 0.0) {
            return Err(SketchError::InvalidParameter {
                name: "jl_constant".to_string(),
                reason: "must be finite and > 0".to_string(),
            });
        }
        if !(cfg.sample_constant.is_finite() && cfg.sample_constant > 0.0) {
            return Err(SketchError::InvalidParameter {
                name: "sample_constant".to_string(),
                reason: "must be finite and > 0".to_string(),
            });
        }
        if !(cfg.cg_tol.is_finite() && cfg.cg_tol > 0.0) {
            return Err(SketchError::InvalidParameter {
                name: "cg_tol".to_string(),
                reason: "must be finite and > 0".to_string(),
            });
        }
        Ok(Self {
            n: n_vertices,
            cfg,
            edges: Vec::new(),
            edge_index: std::collections::HashMap::new(),
            rng,
            z_cache: None,
            z_rows: 0,
        })
    }

    /// Number of vertices.
    #[must_use]
    pub fn n_vertices(&self) -> usize {
        self.n
    }

    /// Number of distinct edges currently held.
    #[must_use]
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// A read-only view of the accumulated edges.
    #[must_use]
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Add (or accumulate the weight of) an undirected edge `(u, v)` with
    /// positive weight `w`. Parallel edges are merged by summing weights, which
    /// preserves the Laplacian. Invalidates any cached projection.
    ///
    /// # Errors
    /// * [`SketchError::IndexOutOfBounds`] — `u` or `v` outside `[0, n)`.
    /// * [`SketchError::InvalidParameter`] — self-loop (`u == v`), or `w` not
    ///   finite and positive.
    pub fn add_edge(&mut self, u: usize, v: usize, w: f64) -> SketchResult<()> {
        if u >= self.n {
            return Err(SketchError::IndexOutOfBounds {
                index: u,
                len: self.n,
            });
        }
        if v >= self.n {
            return Err(SketchError::IndexOutOfBounds {
                index: v,
                len: self.n,
            });
        }
        if u == v {
            return Err(SketchError::InvalidParameter {
                name: "(u,v)".to_string(),
                reason: "self-loops are not allowed".to_string(),
            });
        }
        if !(w.is_finite() && w > 0.0) {
            return Err(SketchError::InvalidParameter {
                name: "w".to_string(),
                reason: "edge weight must be finite and > 0".to_string(),
            });
        }
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        self.z_cache = None; // structure changed
        match self.edge_index.get(&(a, b)) {
            Some(&idx) => {
                self.edges[idx].w += w;
            }
            None => {
                let idx = self.edges.len();
                self.edges.push(Edge { u: a, v: b, w });
                self.edge_index.insert((a, b), idx);
            }
        }
        Ok(())
    }

    /// Johnson–Lindenstrauss projection dimension `k = ⌈C·ln n / ε²⌉`.
    ///
    /// `k` is the number of independent `±1/√k` projection rows; more rows only
    /// reduce the estimator variance (`E[‖Qx‖²] = ‖x‖²`, variance `∝ 1/k`), so —
    /// unlike a column-compressing sketch — `k` may legitimately exceed the edge
    /// count `m` for small dense graphs. It is capped at an absolute bound to
    /// keep the per-row Laplacian solves affordable.
    fn jl_dim(&self) -> usize {
        const K_CAP: usize = 4096;
        let n = self.n as f64;
        let eps = self.cfg.epsilon;
        let raw = (self.cfg.jl_constant * n.ln().max(1.0) / (eps * eps)).ceil();
        let k = if raw.is_finite() && raw >= 1.0 {
            raw as usize
        } else {
            1
        };
        k.clamp(1, K_CAP)
    }

    /// Effective resistance `R_eff(u, v)` between two vertices via the cached JL
    /// estimator `Z`: `R_eff ≈ ‖z_u − z_v‖²`.
    ///
    /// # Errors
    /// * [`SketchError::IndexOutOfBounds`] — `u` or `v` outside `[0, n)`.
    /// * [`SketchError::EmptyStream`] — no edges have been added.
    /// * [`SketchError::NotConverged`] — a Laplacian solve failed to converge.
    pub fn effective_resistance(&mut self, u: usize, v: usize) -> SketchResult<f64> {
        if u >= self.n {
            return Err(SketchError::IndexOutOfBounds {
                index: u,
                len: self.n,
            });
        }
        if v >= self.n {
            return Err(SketchError::IndexOutOfBounds {
                index: v,
                len: self.n,
            });
        }
        if u == v {
            return Ok(0.0);
        }
        if self.edges.is_empty() {
            return Err(SketchError::EmptyStream);
        }
        self.ensure_z()?;
        let z = self
            .z_cache
            .as_ref()
            .ok_or_else(|| SketchError::NumericalInstability("Z projection missing".to_string()))?;
        let k = self.z_rows;
        let mut acc = 0.0_f64;
        for row in 0..k {
            let diff = z[row * self.n + u] - z[row * self.n + v];
            acc += diff * diff;
        }
        Ok(acc)
    }

    /// Produce a spectral sparsifier `H` of the accumulated graph by
    /// effective-resistance importance sampling.
    ///
    /// # Errors
    /// * [`SketchError::EmptyStream`] — no edges have been added.
    /// * [`SketchError::NotConverged`] — a Laplacian solve failed to converge.
    /// * [`SketchError::NumericalInstability`] — non-finite intermediate values.
    pub fn sparsify(&mut self) -> SketchResult<SparsifiedGraph> {
        if self.edges.is_empty() {
            return Err(SketchError::EmptyStream);
        }
        self.ensure_z()?;
        // Effective resistance of every edge from the cached projection.
        let m = self.edges.len();
        let mut reff = vec![0.0_f64; m];
        for (e_idx, edge) in self.edges.iter().enumerate() {
            reff[e_idx] = self.reff_from_cache(edge.u, edge.v)?;
        }
        // Over-sampling factor q = C_s · ln n / ε².
        let n = self.n as f64;
        let eps = self.cfg.epsilon;
        let q = (self.cfg.sample_constant * n.ln().max(1.0) / (eps * eps)).max(1.0);

        let mut sampled: Vec<Edge> = Vec::new();
        for (e_idx, edge) in self.edges.iter().enumerate() {
            let leverage = edge.w * reff[e_idx];
            if !leverage.is_finite() || leverage < 0.0 {
                return Err(SketchError::NumericalInstability(
                    "non-finite leverage score".to_string(),
                ));
            }
            // Keep-probability p_e = min(1, q · w_e · R_eff_e).
            let p = (q * leverage).min(1.0);
            if p >= 1.0 {
                // Always kept, weight unchanged.
                sampled.push(*edge);
            } else if p > 0.0 && self.rng.next_f64() < p {
                // Re-weight by 1/p to keep L_H unbiased.
                sampled.push(Edge {
                    u: edge.u,
                    v: edge.v,
                    w: edge.w / p,
                });
            }
        }
        Ok(SparsifiedGraph {
            n_vertices: self.n,
            edges: sampled,
        })
    }

    /// Exact Laplacian quadratic form `xᵀ L x` of the **original** graph.
    ///
    /// # Errors
    /// [`SketchError::DimensionMismatch`] if `x.len() != n`.
    pub fn quadratic_form(&self, x: &[f64]) -> SketchResult<f64> {
        if x.len() != self.n {
            return Err(SketchError::DimensionMismatch {
                a: self.n,
                b: x.len(),
            });
        }
        Ok(laplacian_quadratic_form(&self.edges, x))
    }

    // ── Internal: build / cache the JL effective-resistance projection ────────

    /// Ensure `z_cache` holds a valid `Z = Q B_w L⁺` for the current graph.
    fn ensure_z(&mut self) -> SketchResult<()> {
        if self.z_cache.is_some() {
            return Ok(());
        }
        let z = self.build_z()?;
        self.z_cache = Some(z);
        Ok(())
    }

    /// `R_eff(u, v) = ‖z_u − z_v‖²` from the cached `Z` (assumes `ensure_z`).
    fn reff_from_cache(&self, u: usize, v: usize) -> SketchResult<f64> {
        let z = self
            .z_cache
            .as_ref()
            .ok_or_else(|| SketchError::NumericalInstability("Z projection missing".to_string()))?;
        let mut acc = 0.0_f64;
        for row in 0..self.z_rows {
            let diff = z[row * self.n + u] - z[row * self.n + v];
            acc += diff * diff;
        }
        Ok(acc)
    }

    /// Build `Z = Q B_w L⁺` (`k × n`, row-major).
    ///
    /// For each of the `k` JL rows we form `y = (Q B_w)_{(i)}` (length `n`) and
    /// solve `L z = y` with the projected CG, storing `z` as row `i` of `Z`.
    fn build_z(&mut self) -> SketchResult<Vec<f64>> {
        let k = self.jl_dim();
        let n = self.n;
        let m = self.edges.len();
        let scale = 1.0 / (k as f64).sqrt();
        let cg_iter = if self.cfg.cg_max_iter == 0 {
            // A safe default: more than n iterations guarantees CG convergence
            // in exact arithmetic; the residual tolerance usually stops earlier.
            (2 * n + 50).max(100)
        } else {
            self.cfg.cg_max_iter
        };

        let mut z = vec![0.0_f64; k * n];
        for row in 0..k {
            // y = (Q B_w)_{row}: for each edge e=(u,v,w) draw q ∈ {−1,+1}/√k and
            // add q·√w·(e_u − e_v) to y.
            let mut y = vec![0.0_f64; n];
            for edge in &self.edges {
                let sign = if self.rng.next_bool() { scale } else { -scale };
                let sw = edge.w.sqrt() * sign;
                y[edge.u] += sw;
                y[edge.v] -= sw;
            }
            // The incidence rows are all orthogonal to 𝟙 by construction, so y is
            // already in range(L) for a connected graph; project defensively.
            project_off_ones(&mut y);
            let sol = solve_laplacian_cg(&self.edges, n, &y, cg_iter, self.cfg.cg_tol)?;
            z[row * n..(row + 1) * n].copy_from_slice(&sol);
        }
        let _ = m; // m documented above; silence unused in case of future edits.
        self.z_rows = k;
        Ok(z)
    }
}

/// Evaluate `xᵀ L x = Σ_{(u,v,w)} w · (x_u − x_v)²` over an edge list.
fn laplacian_quadratic_form(edges: &[Edge], x: &[f64]) -> f64 {
    let mut acc = 0.0_f64;
    for e in edges {
        let d = x[e.u] - x[e.v];
        acc += e.w * d * d;
    }
    acc
}

/// Apply the Laplacian to a vector: `out = L x = Σ_e w_e (x_u − x_v)(e_u − e_v)`.
fn laplacian_matvec(edges: &[Edge], x: &[f64], out: &mut [f64]) {
    for o in out.iter_mut() {
        *o = 0.0;
    }
    for e in edges {
        let d = e.w * (x[e.u] - x[e.v]);
        out[e.u] += d;
        out[e.v] -= d;
    }
}

/// Subtract the mean so the vector becomes orthogonal to the all-ones vector
/// (the Laplacian's null space on a connected component).
fn project_off_ones(v: &mut [f64]) {
    let n = v.len();
    if n == 0 {
        return;
    }
    let mean = v.iter().sum::<f64>() / n as f64;
    for x in v.iter_mut() {
        *x -= mean;
    }
}

/// Solve `L x = b` for the minimum-norm solution (`x ⊥ 𝟙`) via null-space
/// projected Conjugate Gradient.
///
/// `L` is symmetric PSD and singular (null space ⊇ `span{𝟙}`). We keep every
/// iterate and residual orthogonal to `𝟙` by mean-subtraction, which restricts
/// CG to `range(L)` and recovers the pseudoinverse action `x = L⁺ b`.
///
/// # Errors
/// [`SketchError::NotConverged`] if the relative residual does not fall below
/// `tol` within `max_iter` iterations (only returned when the residual is still
/// materially large, guarding against a genuinely ill-posed system).
fn solve_laplacian_cg(
    edges: &[Edge],
    n: usize,
    b: &[f64],
    max_iter: usize,
    tol: f64,
) -> SketchResult<Vec<f64>> {
    let mut x = vec![0.0_f64; n];
    let mut r = b.to_vec();
    project_off_ones(&mut r);
    let b_norm = dot(&r, &r).sqrt();
    if b_norm <= tol {
        // Right-hand side is (numerically) zero ⇒ solution is zero.
        return Ok(x);
    }
    let mut p = r.clone();
    let mut rs_old = dot(&r, &r);
    let mut ap = vec![0.0_f64; n];

    for _ in 0..max_iter {
        laplacian_matvec(edges, &p, &mut ap);
        project_off_ones(&mut ap);
        let p_ap = dot(&p, &ap);
        if !(p_ap.is_finite()) || p_ap <= 0.0 {
            // Curvature exhausted (p in the null space) — residual is as small as
            // this Krylov space allows; accept the current iterate.
            break;
        }
        let alpha = rs_old / p_ap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        project_off_ones(&mut r);
        let rs_new = dot(&r, &r);
        if rs_new.sqrt() <= tol * b_norm {
            return Ok(x);
        }
        let beta = rs_new / rs_old;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }

    // Final residual check; tolerate a loose-but-finite residual since CG on a
    // singular system can stall once the Krylov space is spanned.
    let mut final_ax = vec![0.0_f64; n];
    laplacian_matvec(edges, &x, &mut final_ax);
    let mut resid = vec![0.0_f64; n];
    for i in 0..n {
        resid[i] = b[i] - final_ax[i];
    }
    project_off_ones(&mut resid);
    let rel = dot(&resid, &resid).sqrt() / b_norm;
    if rel.is_finite() && rel <= (tol.sqrt()).max(1e-4) {
        Ok(x)
    } else {
        Err(SketchError::NotConverged { iter: max_iter })
    }
}

/// Euclidean inner product.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A high-accuracy config: a large JL constant drives the effective-
    /// resistance estimator variance (`∝ 1/k`) down so the estimates match the
    /// closed forms closely.
    fn accurate_cfg() -> GraphSketchConfig {
        GraphSketchConfig {
            epsilon: 0.5,
            jl_constant: 400.0,
            sample_constant: 8.0,
            cg_max_iter: 0,
            cg_tol: 1e-12,
        }
    }

    /// Closed-form effective resistance of a single edge of weight `w` is `1/w`.
    #[test]
    fn single_edge_effective_resistance() {
        let mut g = GraphSketch::new(2, accurate_cfg(), LcgRng::new(1)).expect("ok");
        g.add_edge(0, 1, 2.0).expect("ok");
        let r = g.effective_resistance(0, 1).expect("ok");
        // R_eff = 1/w = 0.5 (exact: with 1 edge the JL projection is norm-exact).
        assert!((r - 0.5).abs() < 1e-6, "single-edge R_eff {r} vs 0.5");
    }

    /// Path graph 0-1-2 with unit weights: R_eff(0,2) = 2 (series resistors).
    #[test]
    fn path_graph_series_resistance() {
        let mut g = GraphSketch::new(3, accurate_cfg(), LcgRng::new(2)).expect("ok");
        g.add_edge(0, 1, 1.0).expect("ok");
        g.add_edge(1, 2, 1.0).expect("ok");
        let r02 = g.effective_resistance(0, 2).expect("ok");
        // Two unit resistors in series ⇒ R = 2.
        assert!((r02 - 2.0).abs() < 0.12, "path R_eff(0,2) {r02} vs 2.0");
        let r01 = g.effective_resistance(0, 1).expect("ok");
        assert!((r01 - 1.0).abs() < 0.12, "path R_eff(0,1) {r01} vs 1.0");
    }

    /// Triangle (complete graph K3) with unit weights: R_eff between any pair is
    /// 2/3 (two parallel paths: direct resistor 1 ∥ series-2 path).
    #[test]
    fn triangle_effective_resistance() {
        let mut g = GraphSketch::new(3, accurate_cfg(), LcgRng::new(3)).expect("ok");
        g.add_edge(0, 1, 1.0).expect("ok");
        g.add_edge(1, 2, 1.0).expect("ok");
        g.add_edge(0, 2, 1.0).expect("ok");
        let r = g.effective_resistance(0, 1).expect("ok");
        // 1 ∥ 2 = 2/3.
        assert!((r - 2.0 / 3.0).abs() < 0.1, "triangle R_eff {r} vs 0.667");
    }

    /// K4 (complete graph on 4 vertices) unit weights: R_eff between any pair is
    /// 1/2.
    #[test]
    fn complete_graph_k4_resistance() {
        let mut g = GraphSketch::new(4, accurate_cfg(), LcgRng::new(4)).expect("ok");
        for u in 0..4 {
            for v in (u + 1)..4 {
                g.add_edge(u, v, 1.0).expect("ok");
            }
        }
        let r = g.effective_resistance(0, 2).expect("ok");
        // For K_n, R_eff = 2/n = 1/2 for n=4.
        assert!((r - 0.5).abs() < 0.1, "K4 R_eff {r} vs 0.5");
    }

    /// The sum of `w_e · R_eff(e)` over all edges equals `rank(L) = n − 1` for a
    /// connected graph.
    #[test]
    fn leverage_scores_sum_to_rank() {
        let mut g = GraphSketch::new(5, accurate_cfg(), LcgRng::new(5)).expect("ok");
        // A connected graph on 5 vertices.
        let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 4)];
        for &(u, v) in &edges {
            g.add_edge(u, v, 1.0).expect("ok");
        }
        let mut total = 0.0;
        for &(u, v) in &edges {
            total += g.effective_resistance(u, v).expect("ok");
        }
        // n - 1 = 4. JL is approximate, so allow a generous band.
        assert!((total - 4.0).abs() < 0.6, "Σ w·R_eff {total} vs 4.0");
    }

    /// Spectral guarantee: for a moderately dense random graph, the sparsifier's
    /// quadratic form lies within the (1 ± ε) band of the original for several
    /// random test vectors. We verify within ε = 0.5.
    #[test]
    fn spectral_form_within_band() {
        let n = 12;
        let cfg = GraphSketchConfig {
            epsilon: 0.5,
            jl_constant: 6.0,
            sample_constant: 16.0,
            cg_max_iter: 0,
            cg_tol: 1e-10,
        };
        let mut g = GraphSketch::new(n, cfg, LcgRng::new(777)).expect("ok");
        // Build a connected, fairly dense graph (cycle + random chords).
        for i in 0..n {
            g.add_edge(i, (i + 1) % n, 1.0).expect("ok");
        }
        let mut chord_rng = LcgRng::new(9001);
        for _ in 0..30 {
            let a = chord_rng.next_usize(n);
            let b = chord_rng.next_usize(n);
            if a != b {
                g.add_edge(a, b, 1.0 + chord_rng.next_f64()).expect("ok");
            }
        }
        let original_edges = g.n_edges();
        let h = g.sparsify().expect("ok");
        // Sparsifier should have no more edges than the original.
        assert!(
            h.edges.len() <= original_edges,
            "sparsifier has {} edges > original {}",
            h.edges.len(),
            original_edges
        );

        // Check the spectral band on several random vectors.
        let mut vec_rng = LcgRng::new(424242);
        let eps = 0.5;
        let mut checked = 0;
        for _ in 0..20 {
            let mut x: Vec<f64> = (0..n).map(|_| vec_rng.next_range(-1.0, 1.0)).collect();
            project_off_ones(&mut x);
            let orig = g.quadratic_form(&x).expect("ok");
            if orig < 1e-9 {
                continue;
            }
            let spar = h.quadratic_form(&x).expect("ok");
            let ratio = spar / orig;
            assert!(
                ratio > 1.0 - eps && ratio < 1.0 + eps,
                "spectral ratio {ratio} outside (1±{eps}) for orig {orig}"
            );
            checked += 1;
        }
        assert!(
            checked >= 5,
            "too few non-degenerate test vectors: {checked}"
        );
    }

    /// The sparsifier of a sufficiently large dense graph has fewer edges than
    /// the original. For a complete graph `K_n` every edge has leverage
    /// `w·R_eff = 2/n`, so the keep-probability `min(1, q·2/n)` is `< 1` once
    /// `n` is large relative to the over-sampling factor `q`.
    #[test]
    fn sparsifier_reduces_edges_on_dense_graph() {
        let n = 40;
        let cfg = GraphSketchConfig {
            epsilon: 0.5,
            jl_constant: 4.0,
            sample_constant: 1.0,
            cg_max_iter: 0,
            cg_tol: 1e-10,
        };
        let mut g = GraphSketch::new(n, cfg, LcgRng::new(31337)).expect("ok");
        // Complete graph K_40: n(n-1)/2 = 780 edges.
        for u in 0..n {
            for v in (u + 1)..n {
                g.add_edge(u, v, 1.0).expect("ok");
            }
        }
        let original = g.n_edges();
        let h = g.sparsify().expect("ok");
        assert!(
            h.edges.len() < original,
            "dense sparsifier {} not < original {}",
            h.edges.len(),
            original
        );
    }

    #[test]
    fn rejects_bad_construction() {
        // n < 2.
        assert!(GraphSketch::new(1, GraphSketchConfig::default(), LcgRng::new(0)).is_err());
        // epsilon out of (0,1).
        let bad_lo = GraphSketchConfig {
            epsilon: 0.0,
            ..GraphSketchConfig::default()
        };
        assert!(GraphSketch::new(5, bad_lo, LcgRng::new(0)).is_err());
        let bad_hi = GraphSketchConfig {
            epsilon: 1.5,
            ..GraphSketchConfig::default()
        };
        assert!(GraphSketch::new(5, bad_hi, LcgRng::new(0)).is_err());
    }

    #[test]
    fn rejects_bad_edges() {
        let mut g = GraphSketch::new(4, GraphSketchConfig::default(), LcgRng::new(0)).expect("ok");
        assert!(g.add_edge(0, 4, 1.0).is_err()); // v out of range
        assert!(g.add_edge(5, 1, 1.0).is_err()); // u out of range
        assert!(g.add_edge(2, 2, 1.0).is_err()); // self-loop
        assert!(g.add_edge(0, 1, 0.0).is_err()); // non-positive weight
        assert!(g.add_edge(0, 1, -1.0).is_err());
        assert!(g.add_edge(0, 1, f64::NAN).is_err());
    }

    #[test]
    fn empty_graph_errors() {
        let mut g = GraphSketch::new(3, GraphSketchConfig::default(), LcgRng::new(0)).expect("ok");
        assert!(g.sparsify().is_err());
        assert!(g.effective_resistance(0, 1).is_err());
    }

    #[test]
    fn parallel_edges_accumulate_weight() {
        let mut g = GraphSketch::new(2, GraphSketchConfig::default(), LcgRng::new(0)).expect("ok");
        g.add_edge(0, 1, 1.0).expect("ok");
        g.add_edge(1, 0, 3.0).expect("ok"); // same undirected edge, reversed
        assert_eq!(g.n_edges(), 1);
        // Combined weight 4 ⇒ R_eff = 1/4.
        let r = g.effective_resistance(0, 1).expect("ok");
        assert!((r - 0.25).abs() < 1e-6, "merged-weight R_eff {r} vs 0.25");
    }

    #[test]
    fn quadratic_form_dimension_check() {
        let mut g = GraphSketch::new(3, GraphSketchConfig::default(), LcgRng::new(0)).expect("ok");
        g.add_edge(0, 1, 1.0).expect("ok");
        assert!(g.quadratic_form(&[1.0, 2.0]).is_err()); // wrong length
        assert!(g.quadratic_form(&[1.0, 2.0, 3.0]).is_ok());
    }

    #[test]
    fn cg_solves_path_laplacian() {
        // Directly exercise the CG solver on a path Laplacian: L x = b with
        // b ⊥ 𝟙 must satisfy L x ≈ b.
        let edges = [
            Edge { u: 0, v: 1, w: 1.0 },
            Edge { u: 1, v: 2, w: 1.0 },
            Edge { u: 2, v: 3, w: 1.0 },
        ];
        let n = 4;
        let mut b = vec![1.0, -1.0, 0.5, -0.5];
        project_off_ones(&mut b);
        let x = solve_laplacian_cg(&edges, n, &b, 100, 1e-12).expect("cg ok");
        let mut lx = vec![0.0; n];
        laplacian_matvec(&edges, &x, &mut lx);
        for i in 0..n {
            assert!(
                (lx[i] - b[i]).abs() < 1e-6,
                "L x [{i}] {} vs {}",
                lx[i],
                b[i]
            );
        }
    }

    #[test]
    fn sparsifier_unbiased_quadratic_form_in_expectation() {
        // Averaging xᵀ L_H x over many independent sparsifiers should concentrate
        // on xᵀ L x because re-weighting makes L_H unbiased.
        let n = 8;
        let cfg = GraphSketchConfig {
            epsilon: 0.5,
            jl_constant: 6.0,
            sample_constant: 6.0,
            cg_max_iter: 0,
            cg_tol: 1e-10,
        };
        let mut builder = GraphSketch::new(n, cfg, LcgRng::new(2024)).expect("ok");
        for i in 0..n {
            builder.add_edge(i, (i + 1) % n, 1.0).expect("ok");
            builder.add_edge(i, (i + 2) % n, 1.0).expect("ok");
        }
        let x: Vec<f64> = {
            let mut v: Vec<f64> = (0..n).map(|i| (i as f64) - (n as f64) / 2.0).collect();
            project_off_ones(&mut v);
            v
        };
        let orig = builder.quadratic_form(&x).expect("ok");
        let reps = 40;
        let mut acc = 0.0;
        for _ in 0..reps {
            let h = builder.sparsify().expect("ok");
            acc += h.quadratic_form(&x).expect("ok");
        }
        let mean = acc / reps as f64;
        let rel = (mean - orig).abs() / orig;
        assert!(
            rel < 0.25,
            "mean sparsifier form {mean} vs orig {orig} (rel {rel})"
        );
    }
}
