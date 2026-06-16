//! Immersed boundary method (Peskin, 2002).
//!
//! Implements the core operators of the immersed boundary (IB) method on a
//! uniform two-dimensional Eulerian grid coupled to a set of Lagrangian boundary
//! markers:
//!
//! * the **regularised discrete delta** `δ_h` built from Peskin's 4-point cosine
//!   kernel (with the 3-point kernel available as an alternative),
//! * the **spread** operator `S` that scatters a Lagrangian force to the grid,
//! * the **interpolate** operator `J` that gathers an Eulerian field onto the
//!   markers, and
//! * a **direct-forcing** step that imposes a prescribed boundary velocity
//!   (no-slip) by adding the localized body force needed to cancel the residual.
//!
//! # Discrete delta
//!
//! The 1-D kernel `φ(r)` is a smooth, compactly supported function satisfying the
//! moment conditions that make the IB coupling second-order accurate for
//! interpolation of constants and linears:
//!
//! ```text
//! Σ_j φ(r − j) = 1                 (partition of unity / reproduces constants)
//! Σ_j (r − j) φ(r − j) = 0         (first moment → reproduces linears)
//! ```
//!
//! The 2-D delta is the tensor product `δ_h(x, y) = h⁻² φ(x/h) φ(y/h)`.
//!
//! # Spread / interpolate
//!
//! With grid spacing `h` and cell volume `h²`,
//!
//! ```text
//! spread:       f_{ij} = Σ_k F_k · δ_h(x_{ij} − X_k)
//! interpolate:  U_k    = Σ_{ij} u_{ij} · δ_h(x_{ij} − X_k) · h²
//! ```
//!
//! are exact transposes with respect to the cell-weighted grid inner product
//! `⟨a, b⟩_grid = h² Σ_{ij} a_{ij} b_{ij}` and the Euclidean Lagrangian inner
//! product `⟨A, B⟩_L = Σ_k A_k B_k`, i.e. `⟨S F, u⟩_grid = ⟨F, J u⟩_L`.
//!
//! Reference: C. S. Peskin, "The immersed boundary method", *Acta Numerica* 11
//! (2002) 479–517.

use crate::error::{PdeError, PdeResult};

/// Choice of regularised 1-D kernel for the discrete delta.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaKernel {
    /// Peskin's standard 4-point kernel (support `|r| < 2`).
    ///
    /// This is the canonical IB kernel of Peskin (2002): it satisfies both the
    /// partition-of-unity *and* the first-moment (zero) condition, so it
    /// reproduces constants **and** linear fields exactly.
    FourPoint,
    /// Peskin's 4-point **cosine** kernel `¼(1 + cos(πr/2))` (support `|r| < 2`).
    ///
    /// Cheaper to evaluate but only a partition of unity — it does **not**
    /// satisfy the first-moment condition, so it reproduces constants but not
    /// linear fields.
    FourPointCosine,
    /// Peskin's 3-point kernel (support `|r| < 1.5`).
    ///
    /// Satisfies the partition-of-unity and first-moment conditions (constants
    /// and linears reproduced exactly).
    ThreePoint,
}

impl DeltaKernel {
    /// Half-width of the kernel support in grid cells (`φ(r) = 0` for `|r| ≥ radius`).
    #[must_use]
    pub fn radius(self) -> f64 {
        match self {
            Self::FourPoint | Self::FourPointCosine => 2.0,
            Self::ThreePoint => 1.5,
        }
    }

    /// Evaluate the 1-D kernel `φ(r)` (dimensionless argument `r = distance / h`).
    #[must_use]
    pub fn phi(self, r: f64) -> f64 {
        match self {
            Self::FourPoint => phi_4pt(r),
            Self::FourPointCosine => phi_4pt_cosine(r),
            Self::ThreePoint => phi_3pt(r),
        }
    }
}

/// Peskin's standard 4-point kernel.
///
/// ```text
///         ⅛ (3 − 2|r| + √(1 + 4|r| − 4r²))      0 ≤ |r| ≤ 1
/// φ(r) =  ⅛ (5 − 2|r| − √(−7 + 12|r| − 4r²))     1 ≤ |r| ≤ 2
///         0                                       otherwise
/// ```
///
/// It satisfies `Σ_j φ(r − j) = 1`, `Σ_j (r − j) φ(r − j) = 0`, and the
/// sum-of-squares condition, giving exact reproduction of constant and linear
/// fields.
#[must_use]
fn phi_4pt(r: f64) -> f64 {
    let a = r.abs();
    if a <= 1.0 {
        (3.0 - 2.0 * a + (1.0 + 4.0 * a - 4.0 * a * a).sqrt()) / 8.0
    } else if a <= 2.0 {
        (5.0 - 2.0 * a - (-7.0 + 12.0 * a - 4.0 * a * a).sqrt()) / 8.0
    } else {
        0.0
    }
}

/// Peskin's 4-point cosine kernel.
///
/// ```text
/// φ(r) = ¼ (1 + cos(π r / 2))   for |r| ≤ 2,   else 0.
/// ```
///
/// A partition of unity but **not** first-moment accurate.
#[must_use]
fn phi_4pt_cosine(r: f64) -> f64 {
    let a = r.abs();
    if a <= 2.0 {
        0.25 * (1.0 + (std::f64::consts::FRAC_PI_2 * r).cos())
    } else {
        0.0
    }
}

/// Peskin's 3-point kernel.
///
/// ```text
///         ⅓ (1 + √(1 − 3 r²))                       |r| ≤ ½
/// φ(r) =  ⅙ (5 − 3|r| − √(1 − 3(1 − |r|)²))         ½ ≤ |r| ≤ 3⁄2
///         0                                          otherwise
/// ```
#[must_use]
fn phi_3pt(r: f64) -> f64 {
    let a = r.abs();
    if a <= 0.5 {
        (1.0 + (1.0 - 3.0 * a * a).sqrt()) / 3.0
    } else if a <= 1.5 {
        let t = 1.0 - a;
        (5.0 - 3.0 * a - (1.0 - 3.0 * t * t).sqrt()) / 6.0
    } else {
        0.0
    }
}

/// A uniform 2-D Eulerian grid for the immersed-boundary coupling.
///
/// Node `(i, j)` (with `i` the `x`-index in `0..nx`, `j` the `y`-index in
/// `0..ny`) sits at `(x0 + i·h, y0 + j·h)`. Fields are stored row-major in `j`
/// then `i`, i.e. the flat index is `j·nx + i`.
#[derive(Debug, Clone)]
pub struct ImmersedBoundary {
    /// Number of nodes along `x`.
    pub nx: usize,
    /// Number of nodes along `y`.
    pub ny: usize,
    /// Uniform grid spacing `h > 0`.
    pub h: f64,
    /// `x`-coordinate of node `i = 0`.
    pub x0: f64,
    /// `y`-coordinate of node `j = 0`.
    pub y0: f64,
    /// Regularised-delta kernel.
    pub kernel: DeltaKernel,
}

impl ImmersedBoundary {
    /// Construct a grid. Validates `nx, ny ≥ 1`, finite origin, and `h > 0`.
    ///
    /// # Errors
    /// Returns [`PdeError::InvalidGrid`] for an empty grid and
    /// [`PdeError::InvalidParameter`] for a non-positive or non-finite `h`.
    pub fn new(
        nx: usize,
        ny: usize,
        h: f64,
        x0: f64,
        y0: f64,
        kernel: DeltaKernel,
    ) -> PdeResult<Self> {
        if nx == 0 || ny == 0 {
            return Err(PdeError::InvalidGrid(format!(
                "grid must be non-empty, got {nx}×{ny}"
            )));
        }
        if !(h.is_finite() && h > 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "h".to_string(),
                reason: format!("must be > 0 and finite, got {h}"),
            });
        }
        if !(x0.is_finite() && y0.is_finite()) {
            return Err(PdeError::InvalidParameter {
                name: "origin".to_string(),
                reason: "origin coordinates must be finite".to_string(),
            });
        }
        Ok(Self {
            nx,
            ny,
            h,
            x0,
            y0,
            kernel,
        })
    }

    /// Number of grid nodes (`nx · ny`).
    #[must_use]
    pub fn n_nodes(&self) -> usize {
        self.nx * self.ny
    }

    /// Flat row-major index of node `(i, j)`.
    #[must_use]
    pub fn index(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    /// Physical coordinates of node `(i, j)`.
    #[must_use]
    pub fn node_coord(&self, i: usize, j: usize) -> (f64, f64) {
        (self.x0 + i as f64 * self.h, self.y0 + j as f64 * self.h)
    }

    /// Inclusive index window `[i_lo, i_hi] × [j_lo, j_hi]` of grid nodes inside
    /// the kernel support around marker `(mx, my)`, clamped to the grid.
    fn support_window(&self, mx: f64, my: f64) -> (usize, usize, usize, usize) {
        let radius = self.kernel.radius();
        // Continuous index of the marker along each axis.
        let ix = (mx - self.x0) / self.h;
        let iy = (my - self.y0) / self.h;
        let i_lo = ((ix - radius).ceil()).max(0.0) as isize;
        let i_hi = ((ix + radius).floor()).min((self.nx - 1) as f64) as isize;
        let j_lo = ((iy - radius).ceil()).max(0.0) as isize;
        let j_hi = ((iy + radius).floor()).min((self.ny - 1) as f64) as isize;
        (
            i_lo.max(0) as usize,
            i_hi.max(-1) as usize,
            j_lo.max(0) as usize,
            j_hi.max(-1) as usize,
        )
    }

    /// Discrete delta `δ_h(x_{ij} − X_k)` for node `(i, j)` and marker `(mx, my)`.
    ///
    /// `δ_h = h⁻² φ((x_i − mx)/h) φ((y_j − my)/h)`.
    #[must_use]
    pub fn delta(&self, i: usize, j: usize, mx: f64, my: f64) -> f64 {
        let (x, y) = self.node_coord(i, j);
        let phi_x = self.kernel.phi((x - mx) / self.h);
        let phi_y = self.kernel.phi((y - my) / self.h);
        phi_x * phi_y / (self.h * self.h)
    }

    /// Sum of the discrete delta over **all** grid nodes for one marker, i.e.
    /// `h² Σ_{ij} δ_h(x_{ij} − X_k)`. For a marker whose full support lies inside
    /// the grid this equals `1` (partition of unity).
    #[must_use]
    pub fn delta_partition_sum(&self, mx: f64, my: f64) -> f64 {
        let (i_lo, i_hi, j_lo, j_hi) = self.support_window(mx, my);
        if i_hi < i_lo || j_hi < j_lo {
            return 0.0;
        }
        let cell = self.h * self.h;
        let mut acc = 0.0;
        for j in j_lo..=j_hi {
            for i in i_lo..=i_hi {
                acc += self.delta(i, j, mx, my) * cell;
            }
        }
        acc
    }

    /// **Spread** a scalar Lagrangian quantity to the grid.
    ///
    /// `f_{ij} = Σ_k F_k · δ_h(x_{ij} − X_k)`.
    ///
    /// # Arguments
    /// * `markers` — interleaved marker coordinates `[x_0, y_0, x_1, y_1, …]`.
    /// * `values` — one Lagrangian value `F_k` per marker.
    ///
    /// # Errors
    /// Returns [`PdeError`] when `markers.len()` is odd, when `values.len()`
    /// disagrees with the number of markers, or when a coordinate/value is not
    /// finite.
    pub fn spread(&self, markers: &[f64], values: &[f64]) -> PdeResult<Vec<f64>> {
        let n_markers = self.check_markers(markers)?;
        if values.len() != n_markers {
            return Err(PdeError::DimensionMismatch {
                a: values.len(),
                b: n_markers,
            });
        }
        let mut grid = vec![0.0_f64; self.n_nodes()];
        for k in 0..n_markers {
            let mx = markers[2 * k];
            let my = markers[2 * k + 1];
            let fk = values[k];
            if !fk.is_finite() {
                return Err(PdeError::NumericalInstability(format!(
                    "non-finite Lagrangian value at marker {k}"
                )));
            }
            let (i_lo, i_hi, j_lo, j_hi) = self.support_window(mx, my);
            if i_hi < i_lo || j_hi < j_lo {
                continue;
            }
            for j in j_lo..=j_hi {
                for i in i_lo..=i_hi {
                    grid[self.index(i, j)] += fk * self.delta(i, j, mx, my);
                }
            }
        }
        Ok(grid)
    }

    /// **Interpolate** a scalar Eulerian field onto the markers.
    ///
    /// `U_k = Σ_{ij} u_{ij} · δ_h(x_{ij} − X_k) · h²`.
    ///
    /// # Arguments
    /// * `field` — grid field of length `nx · ny`, row-major in `j` then `i`.
    /// * `markers` — interleaved marker coordinates `[x_0, y_0, …]`.
    ///
    /// # Errors
    /// Returns [`PdeError`] when `field.len() != nx·ny`, when `markers.len()` is
    /// odd, or when a coordinate is not finite.
    pub fn interpolate(&self, field: &[f64], markers: &[f64]) -> PdeResult<Vec<f64>> {
        if field.len() != self.n_nodes() {
            return Err(PdeError::ShapeMismatch {
                expected: vec![self.n_nodes()],
                got: vec![field.len()],
            });
        }
        let n_markers = self.check_markers(markers)?;
        let cell = self.h * self.h;
        let mut out = vec![0.0_f64; n_markers];
        for k in 0..n_markers {
            let mx = markers[2 * k];
            let my = markers[2 * k + 1];
            let (i_lo, i_hi, j_lo, j_hi) = self.support_window(mx, my);
            if i_hi < i_lo || j_hi < j_lo {
                continue;
            }
            let mut acc = 0.0;
            for j in j_lo..=j_hi {
                for i in i_lo..=i_hi {
                    acc += field[self.index(i, j)] * self.delta(i, j, mx, my) * cell;
                }
            }
            out[k] = acc;
        }
        Ok(out)
    }

    /// One **direct-forcing** step that drives the interpolated boundary velocity
    /// toward a prescribed value (no-slip enforcement).
    ///
    /// Given the current Eulerian velocity component `u`, target marker velocities
    /// `u_target` (one per marker), and step `dt`, the method:
    ///
    /// 1. interpolates `u` onto the markers, `U_k = J u`;
    /// 2. forms the Lagrangian forcing `F_k = (u_target,k − U_k) / dt`;
    /// 3. spreads it back, `f = S F`, and applies `u ← u + dt · f`.
    ///
    /// The returned field is the corrected velocity; the residual
    /// `‖J u_new − u_target‖` is reduced relative to `‖J u − u_target‖` because
    /// `S` and `J` are positive-coupled adjoints.
    ///
    /// # Arguments
    /// * `u` — Eulerian velocity component, length `nx · ny`.
    /// * `markers` — interleaved marker coordinates.
    /// * `u_target` — prescribed marker velocity, one per marker.
    /// * `dt` — positive time step.
    ///
    /// # Errors
    /// Returns [`PdeError`] on a shape mismatch, an odd `markers.len()`, a
    /// `u_target.len()` mismatch, a non-positive `dt`, or non-finite input.
    pub fn direct_forcing_step(
        &self,
        u: &[f64],
        markers: &[f64],
        u_target: &[f64],
        dt: f64,
    ) -> PdeResult<Vec<f64>> {
        if u.len() != self.n_nodes() {
            return Err(PdeError::ShapeMismatch {
                expected: vec![self.n_nodes()],
                got: vec![u.len()],
            });
        }
        let n_markers = self.check_markers(markers)?;
        if u_target.len() != n_markers {
            return Err(PdeError::DimensionMismatch {
                a: u_target.len(),
                b: n_markers,
            });
        }
        if !(dt.is_finite() && dt > 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "dt".to_string(),
                reason: format!("must be > 0 and finite, got {dt}"),
            });
        }

        let u_interp = self.interpolate(u, markers)?;
        let mut force = vec![0.0_f64; n_markers];
        for k in 0..n_markers {
            if !u_target[k].is_finite() {
                return Err(PdeError::NumericalInstability(format!(
                    "non-finite target velocity at marker {k}"
                )));
            }
            force[k] = (u_target[k] - u_interp[k]) / dt;
        }
        let spread_force = self.spread(markers, &force)?;
        let mut u_new = u.to_vec();
        for (un, &f) in u_new.iter_mut().zip(spread_force.iter()) {
            *un += dt * f;
        }
        Ok(u_new)
    }

    /// Cell-weighted grid inner product `⟨a, b⟩_grid = h² Σ_{ij} a_{ij} b_{ij}`.
    ///
    /// # Errors
    /// Returns [`PdeError::ShapeMismatch`] when either field is the wrong length.
    pub fn grid_inner(&self, a: &[f64], b: &[f64]) -> PdeResult<f64> {
        if a.len() != self.n_nodes() || b.len() != self.n_nodes() {
            return Err(PdeError::ShapeMismatch {
                expected: vec![self.n_nodes()],
                got: vec![a.len().max(b.len())],
            });
        }
        let cell = self.h * self.h;
        Ok(cell * a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum::<f64>())
    }

    /// Validate interleaved markers and return the marker count.
    fn check_markers(&self, markers: &[f64]) -> PdeResult<usize> {
        if markers.len() % 2 != 0 {
            return Err(PdeError::InvalidParameter {
                name: "markers".to_string(),
                reason: format!("length {} must be even (x, y pairs)", markers.len()),
            });
        }
        for (i, &c) in markers.iter().enumerate() {
            if !c.is_finite() {
                return Err(PdeError::NumericalInstability(format!(
                    "non-finite marker coordinate at position {i}"
                )));
            }
        }
        Ok(markers.len() / 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid(kernel: DeltaKernel) -> ImmersedBoundary {
        // 21×21 nodes, h=0.5, origin (-5,-5): covers [-5, 5]².
        ImmersedBoundary::new(21, 21, 0.5, -5.0, -5.0, kernel).expect("new should succeed")
    }

    // ----------------------------------------------------------------------
    // (a) δ_h sums to 1 over the grid (partition of unity), incl. off-grid.
    // ----------------------------------------------------------------------

    const ALL_KERNELS: [DeltaKernel; 3] = [
        DeltaKernel::FourPoint,
        DeltaKernel::FourPointCosine,
        DeltaKernel::ThreePoint,
    ];

    /// Kernels with the first-moment (linear-accuracy) property.
    const LINEAR_KERNELS: [DeltaKernel; 2] = [DeltaKernel::FourPoint, DeltaKernel::ThreePoint];

    #[test]
    fn delta_partition_of_unity_on_node() {
        for kernel in ALL_KERNELS {
            let g = make_grid(kernel);
            // Marker exactly on a node.
            let sum = g.delta_partition_sum(0.0, 0.0);
            assert!((sum - 1.0).abs() < 1e-12, "{kernel:?} on-node sum {sum}");
        }
    }

    #[test]
    fn delta_partition_of_unity_off_grid() {
        for kernel in ALL_KERNELS {
            let g = make_grid(kernel);
            // Several markers at arbitrary off-grid positions, all with full
            // support strictly inside the domain.
            for &(mx, my) in &[
                (0.123, -0.789),
                (1.5, 2.0),
                (-2.314, 0.951),
                (0.5, 0.5),
                (-1.0, -1.0),
            ] {
                let sum = g.delta_partition_sum(mx, my);
                assert!(
                    (sum - 1.0).abs() < 1e-12,
                    "{kernel:?} off-grid sum at ({mx},{my}) = {sum}"
                );
            }
        }
    }

    #[test]
    fn one_d_kernel_first_moment_is_zero() {
        // Σ_j (r − j) φ(r − j) = 0 → linear reproduction. Check on a fine sweep.
        for kernel in LINEAR_KERNELS {
            let radius = kernel.radius() as isize + 1;
            for step in 0..20 {
                let r = step as f64 / 20.0; // r ∈ [0, 1)
                let mut s0 = 0.0;
                let mut s1 = 0.0;
                for j in -radius..=radius {
                    let arg = r - j as f64;
                    let w = kernel.phi(arg);
                    s0 += w;
                    s1 += arg * w;
                }
                assert!((s0 - 1.0).abs() < 1e-12, "{kernel:?} sum φ = {s0}");
                assert!(s1.abs() < 1e-12, "{kernel:?} first moment = {s1}");
            }
        }
    }

    #[test]
    fn cosine_kernel_is_partition_of_unity_but_not_linear() {
        // The cosine kernel sums to one but has a non-zero first moment.
        let kernel = DeltaKernel::FourPointCosine;
        let r = 0.37_f64;
        let mut s0 = 0.0;
        let mut s1 = 0.0;
        for j in -3..=3 {
            let arg = r - j as f64;
            let w = kernel.phi(arg);
            s0 += w;
            s1 += arg * w;
        }
        assert!((s0 - 1.0).abs() < 1e-12, "cosine sum φ = {s0}");
        assert!(s1.abs() > 1e-3, "cosine first moment unexpectedly ~0: {s1}");
    }

    // ----------------------------------------------------------------------
    // (b) adjoint consistency: ⟨spread(F), u⟩_grid = ⟨F, interp(u)⟩_L.
    // ----------------------------------------------------------------------

    #[test]
    fn spread_interpolate_are_adjoint() {
        for kernel in ALL_KERNELS {
            let g = make_grid(kernel);
            let markers = [0.3, -0.4, 1.1, 0.7, -2.2, 1.9];
            let forces = [1.0, -0.5, 2.0];

            // Arbitrary smooth grid field.
            let mut field = vec![0.0_f64; g.n_nodes()];
            for j in 0..g.ny {
                for i in 0..g.nx {
                    let (x, y) = g.node_coord(i, j);
                    field[g.index(i, j)] = 0.7 + 0.3 * x - 0.2 * y + 0.1 * x * y;
                }
            }

            let spread = g.spread(&markers, &forces).expect("spread should succeed");
            let lhs = g
                .grid_inner(&spread, &field)
                .expect("grid_inner should succeed");

            let interp = g
                .interpolate(&field, &markers)
                .expect("interpolate should succeed");
            let rhs: f64 = forces.iter().zip(interp.iter()).map(|(&f, &u)| f * u).sum();

            assert!(
                (lhs - rhs).abs() < 1e-12,
                "{kernel:?} adjoint mismatch: lhs={lhs}, rhs={rhs}"
            );
        }
    }

    // ----------------------------------------------------------------------
    // (c) interpolating a constant field reproduces the constant.
    // ----------------------------------------------------------------------

    #[test]
    fn interpolate_reproduces_constant() {
        for kernel in ALL_KERNELS {
            let g = make_grid(kernel);
            let c = 1.234_567_89;
            let field = vec![c; g.n_nodes()];
            let markers = [0.123, -0.456, 1.0, 1.0, -1.7, 2.3];
            let interp = g
                .interpolate(&field, &markers)
                .expect("interpolate should succeed");
            for (k, &v) in interp.iter().enumerate() {
                assert!((v - c).abs() < 1e-12, "marker {k}: {v} != {c}");
            }
        }
    }

    // ----------------------------------------------------------------------
    // (d) interpolating a linear field reproduces it at the markers.
    // ----------------------------------------------------------------------

    #[test]
    fn interpolate_reproduces_linear() {
        for kernel in LINEAR_KERNELS {
            let g = make_grid(kernel);
            // u(x, y) = a + b x + c y.
            let (a, b, c) = (0.5, -1.25, 0.8);
            let mut field = vec![0.0_f64; g.n_nodes()];
            for j in 0..g.ny {
                for i in 0..g.nx {
                    let (x, y) = g.node_coord(i, j);
                    field[g.index(i, j)] = a + b * x + c * y;
                }
            }
            let markers = [0.321, -0.654, 1.111, 0.222, -2.0, 1.5];
            let interp = g
                .interpolate(&field, &markers)
                .expect("interpolate should succeed");
            for k in 0..(markers.len() / 2) {
                let mx = markers[2 * k];
                let my = markers[2 * k + 1];
                let exact = a + b * mx + c * my;
                assert!(
                    (interp[k] - exact).abs() < 1e-12,
                    "{kernel:?} marker {k}: {} != {exact}",
                    interp[k]
                );
            }
        }
    }

    // ----------------------------------------------------------------------
    // (e) the spread is localized (zero beyond the kernel support).
    // ----------------------------------------------------------------------

    #[test]
    fn spread_is_localized() {
        let g = make_grid(DeltaKernel::FourPoint);
        // Single marker at the centre node (0, 0).
        let markers = [0.0, 0.0];
        let forces = [1.0];
        let spread = g.spread(&markers, &forces).expect("spread should succeed");
        let radius = g.kernel.radius();
        for j in 0..g.ny {
            for i in 0..g.nx {
                let (x, y) = g.node_coord(i, j);
                let val = spread[g.index(i, j)];
                // Strictly outside the 2-D support box → exactly zero.
                if x.abs() >= radius * g.h || y.abs() >= radius * g.h {
                    assert_eq!(val, 0.0, "non-zero spread at ({x},{y}) outside support");
                }
            }
        }
        // The contribution is concentrated near the marker (centre is positive).
        let centre = spread[g.index(10, 10)];
        assert!(
            centre > 0.0,
            "centre spread should be positive, got {centre}"
        );
    }

    // ----------------------------------------------------------------------
    // (f) a direct-forcing step drives the interpolated boundary velocity toward
    //     the prescribed value (no-slip residual decreases).
    // ----------------------------------------------------------------------

    #[test]
    fn direct_forcing_reduces_no_slip_residual() {
        for kernel in ALL_KERNELS {
            let g = make_grid(kernel);
            // Start from a non-zero ambient flow u ≡ 1 everywhere.
            let u = vec![1.0_f64; g.n_nodes()];
            // Boundary markers along a small circle near the centre.
            let mut markers = Vec::new();
            let n_pts = 12;
            for p in 0..n_pts {
                let theta = std::f64::consts::TAU * p as f64 / n_pts as f64;
                markers.push(1.5 * theta.cos());
                markers.push(1.5 * theta.sin());
            }
            // No-slip: prescribe zero boundary velocity.
            let u_target = vec![0.0_f64; n_pts];

            let before = g
                .interpolate(&u, &markers)
                .expect("interpolate should succeed");
            let res_before: f64 = before
                .iter()
                .zip(u_target.iter())
                .map(|(&v, &t)| (v - t) * (v - t))
                .sum::<f64>()
                .sqrt();

            let dt = 0.5;
            let u_new = g
                .direct_forcing_step(&u, &markers, &u_target, dt)
                .expect("direct_forcing_step should succeed");

            let after = g
                .interpolate(&u_new, &markers)
                .expect("interpolate should succeed");
            let res_after: f64 = after
                .iter()
                .zip(u_target.iter())
                .map(|(&v, &t)| (v - t) * (v - t))
                .sum::<f64>()
                .sqrt();

            assert!(
                res_before > 0.0,
                "{kernel:?} initial residual should be > 0"
            );
            assert!(
                res_after < res_before,
                "{kernel:?} residual did not decrease: {res_before} → {res_after}"
            );
        }
    }

    #[test]
    fn direct_forcing_iterates_toward_no_slip() {
        // Repeated direct forcing drives the residual steadily down.
        let g = make_grid(DeltaKernel::FourPoint);
        let mut u = vec![1.0_f64; g.n_nodes()];
        let mut markers = Vec::new();
        for p in 0..16 {
            let theta = std::f64::consts::TAU * p as f64 / 16.0;
            markers.push(2.0 * theta.cos());
            markers.push(2.0 * theta.sin());
        }
        let u_target = vec![0.0_f64; 16];
        let residual = |u: &[f64]| -> f64 {
            g.interpolate(u, &markers)
                .expect("value should be present")
                .iter()
                .map(|&v| v * v)
                .sum::<f64>()
                .sqrt()
        };
        let mut prev = residual(&u);
        for _ in 0..8 {
            u = g
                .direct_forcing_step(&u, &markers, &u_target, 1.0)
                .expect("direct_forcing_step should succeed");
            let now = residual(&u);
            assert!(now <= prev + 1e-12, "residual increased: {prev} → {now}");
            prev = now;
        }
        assert!(
            prev < residual(&vec![1.0_f64; g.n_nodes()]),
            "no progress made"
        );
    }

    // ----------------------------------------------------------------------
    // Kernel values and error paths.
    // ----------------------------------------------------------------------

    #[test]
    fn four_point_kernel_values() {
        // φ(0) = ½, φ(±1) = ¼, φ(±2) = 0.
        assert!((phi_4pt(0.0) - 0.5).abs() < 1e-12);
        assert!((phi_4pt(1.0) - 0.25).abs() < 1e-12);
        assert!((phi_4pt(-1.0) - 0.25).abs() < 1e-12);
        assert!(phi_4pt(2.0).abs() < 1e-12);
        assert_eq!(phi_4pt(2.5), 0.0);
    }

    #[test]
    fn three_point_kernel_continuous_at_half() {
        // Both pieces must agree at the breakpoint |r| = ½.
        // First branch:  (1 + √(1 − 3·¼))/3 = (1 + ½)/3 = ½.
        // Second branch: (5 − 1.5 − √(1 − 3·¼))/6 = (3.5 − 0.5)/6 = ½.
        let left = phi_3pt(0.5);
        let right = {
            let a = 0.5_f64;
            let t = 1.0 - a;
            (5.0 - 3.0 * a - (1.0 - 3.0 * t * t).sqrt()) / 6.0
        };
        assert!(
            (left - right).abs() < 1e-12,
            "3pt discontinuous: {left} vs {right}"
        );
        assert!(
            (left - 0.5).abs() < 1e-12,
            "3pt(0.5) = {left}, expected 0.5"
        );
        // φ(0) = ⅓·(1 + 1) = ⅔ at the centre.
        assert!((phi_3pt(0.0) - 2.0 / 3.0).abs() < 1e-12);
        // Zero at and beyond the support edge |r| = 3⁄2.
        assert!(phi_3pt(1.5).abs() < 1e-12);
        assert_eq!(phi_3pt(2.0), 0.0);
    }

    #[test]
    fn new_rejects_bad_args() {
        assert!(ImmersedBoundary::new(0, 4, 0.5, 0.0, 0.0, DeltaKernel::FourPoint).is_err());
        assert!(ImmersedBoundary::new(4, 4, 0.0, 0.0, 0.0, DeltaKernel::FourPoint).is_err());
        assert!(ImmersedBoundary::new(4, 4, -1.0, 0.0, 0.0, DeltaKernel::FourPoint).is_err());
    }

    #[test]
    fn spread_rejects_odd_markers_and_bad_values() {
        let g = make_grid(DeltaKernel::FourPoint);
        assert!(g.spread(&[0.0, 0.0, 1.0], &[1.0]).is_err()); // odd length
        assert!(g.spread(&[0.0, 0.0], &[1.0, 2.0]).is_err()); // value count mismatch
    }

    #[test]
    fn interpolate_rejects_wrong_field_length() {
        let g = make_grid(DeltaKernel::FourPoint);
        assert!(g.interpolate(&[0.0, 1.0, 2.0], &[0.0, 0.0]).is_err());
    }

    #[test]
    fn direct_forcing_rejects_non_positive_dt() {
        let g = make_grid(DeltaKernel::FourPoint);
        let u = vec![0.0_f64; g.n_nodes()];
        assert!(g.direct_forcing_step(&u, &[0.0, 0.0], &[0.0], 0.0).is_err());
    }
}
