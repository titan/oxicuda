//! Betti curves: a piecewise-constant vector summary of a barcode.
//!
//! Reference: Yuriy Mileyko, Sayan Mukherjee, John Harer, "Probability measures on
//! the space of persistence diagrams", Inverse Problems 27 (2011), 124007; and
//! Umeda, "Time Series Classification via Topological Data Analysis", Trans. JSAI
//! 32 (2017), which popularised the *Betti curve* (a.k.a. *Betti sequence*) as a
//! fixed-length feature vector for downstream machine learning.
//!
//! The Betti number `β_d(t)` of a filtration at parameter `t` is the number of
//! homology classes of dimension `d` that are *alive* at `t`, i.e. the number of
//! persistence pairs `(b, d)` with `b ≤ t < d`.  Sampling `β_d` on a grid yields the
//! **Betti curve**, a vector in `ℕ^{|grid|}` (here stored as `usize`, with an `f32`
//! view for integration / distances).
//!
//! Convention (matching `persistence/landscape_p.rs`): the sampling grid is `f32`,
//! births/deaths are read as `f64` and cast to `f32`.  Essential classes
//! (`death == None`) are treated as living forever (`death = +∞`), so they contribute
//! `1` at every grid point `≥ birth`.

use crate::error::{TdaError, TdaResult};
use crate::persistence::barcode::Barcode;
use crate::persistence::diagram::PersistenceDiagram;

// ─── Result type ────────────────────────────────────────────────────────────────

/// A Betti curve for a single homological dimension, sampled on a shared grid.
///
/// `values[i]` is the Betti number `β_dim(grid[i])` — the count of dimension-`dim`
/// classes alive at parameter `grid[i]`.
#[derive(Debug, Clone)]
pub struct BettiCurve {
    /// Homological dimension this curve represents.
    pub dim: usize,
    /// Sampling grid (filtration parameters), length = `values.len()`.
    pub grid: Vec<f32>,
    /// Sampled Betti numbers, one per grid point.
    pub values: Vec<usize>,
}

impl BettiCurve {
    /// The sampled Betti numbers as `f32` (for integration, distances, ML features).
    pub fn as_f32(&self) -> Vec<f32> {
        self.values.iter().map(|&v| v as f32).collect()
    }

    /// Area under the Betti curve via the trapezoidal rule over the (possibly
    /// non-uniform) grid.  Returns `0` if fewer than two sample points.
    pub fn area(&self) -> f32 {
        if self.grid.len() < 2 || self.values.len() < 2 {
            return 0.0;
        }
        let n = self.grid.len().min(self.values.len());
        let mut acc = 0.0_f32;
        for i in 1..n {
            let dt = self.grid[i] - self.grid[i - 1];
            let avg = 0.5 * (self.values[i] as f32 + self.values[i - 1] as f32);
            acc += avg * dt;
        }
        acc
    }

    /// The L² distance between two Betti curves, evaluated by the trapezoidal rule
    /// over the shared grid: `(∫ |β₁(t) − β₂(t)|² dt)^{1/2}`.
    ///
    /// # Errors
    /// Returns [`TdaError::DimensionMismatch`] if the two grids have different lengths.
    pub fn l2_distance(&self, other: &BettiCurve) -> TdaResult<f32> {
        if self.grid.len() != other.grid.len() {
            return Err(TdaError::DimensionMismatch {
                expected: self.grid.len(),
                got: other.grid.len(),
            });
        }
        let n = self.grid.len();
        if n < 2 {
            return Ok(0.0);
        }
        let mut acc = 0.0_f32;
        let mut prev = {
            let a = self.values.first().copied().unwrap_or(0) as f32;
            let b = other.values.first().copied().unwrap_or(0) as f32;
            (a - b) * (a - b)
        };
        for i in 1..n {
            let dt = self.grid[i] - self.grid[i - 1];
            let a = self.values.get(i).copied().unwrap_or(0) as f32;
            let b = other.values.get(i).copied().unwrap_or(0) as f32;
            let cur = (a - b) * (a - b);
            acc += 0.5 * (prev + cur) * dt;
            prev = cur;
        }
        Ok(acc.sqrt())
    }
}

// ─── Construction ───────────────────────────────────────────────────────────────

/// Compute the Betti curve of `diagram` for homological dimension `dim` on `grid`.
///
/// A persistence pair `(b, d)` of dimension `dim` contributes `1` at grid value `t`
/// iff `b ≤ t < d` (the standard half-open convention; essential classes use
/// `d = +∞`).  Pairs of any other dimension are ignored.
///
/// The implementation is a deliberately simple double loop over (grid points ×
/// pairs); for the grid/diagram sizes used in TDA pipelines this is more than
/// adequate and keeps the half-open boundary logic transparent.
///
/// # Errors
/// Returns [`TdaError::NanFiltrationValue`] if any grid value is NaN.
pub fn betti_curve(
    diagram: &PersistenceDiagram,
    dim: usize,
    grid: &[f32],
) -> TdaResult<BettiCurve> {
    for &t in grid {
        if t.is_nan() {
            return Err(TdaError::NanFiltrationValue);
        }
    }

    // Pre-extract (birth, death) as f32 for the requested dimension only.
    let intervals: Vec<(f32, f32)> = diagram
        .pairs
        .iter()
        .filter(|p| p.dim == dim)
        .map(|p| {
            let birth = p.birth as f32;
            let death = p.death.map(|d| d as f32).unwrap_or(f32::INFINITY);
            (birth, death)
        })
        .collect();

    // Naive double loop: for each grid point, count alive intervals (b ≤ t < d).
    let mut values: Vec<usize> = Vec::with_capacity(grid.len());
    for &t in grid {
        let mut count = 0usize;
        for &(b, d) in &intervals {
            if b <= t && t < d {
                count += 1;
            }
        }
        values.push(count);
    }

    Ok(BettiCurve {
        dim,
        grid: grid.to_vec(),
        values,
    })
}

/// Compute the Betti curve of a [`Barcode`] for homological dimension `dim` on `grid`.
///
/// Identical semantics to [`betti_curve`], reading bars instead of pairs.  Essential
/// bars (`death == f64::INFINITY`) are alive at every `t ≥ birth`.
///
/// # Errors
/// Returns [`TdaError::NanFiltrationValue`] if any grid value is NaN.
pub fn betti_curve_from_barcode(
    barcode: &Barcode,
    dim: usize,
    grid: &[f32],
) -> TdaResult<BettiCurve> {
    for &t in grid {
        if t.is_nan() {
            return Err(TdaError::NanFiltrationValue);
        }
    }

    let intervals: Vec<(f32, f32)> = barcode
        .bars
        .iter()
        .filter(|bar| bar.dim == dim)
        .map(|bar| (bar.birth as f32, bar.death as f32))
        .collect();

    let mut values: Vec<usize> = Vec::with_capacity(grid.len());
    for &t in grid {
        let mut count = 0usize;
        for &(b, d) in &intervals {
            if b <= t && t < d {
                count += 1;
            }
        }
        values.push(count);
    }

    Ok(BettiCurve {
        dim,
        grid: grid.to_vec(),
        values,
    })
}

/// Compute Betti curves for every dimension `0..=max_dim` on a shared `grid`.
///
/// Returns a `Vec` of length `max_dim + 1` where index `d` is `β_d`'s curve.
///
/// # Errors
/// Propagates [`TdaError::NanFiltrationValue`] from [`betti_curve`].
pub fn betti_curves_all_dims(
    diagram: &PersistenceDiagram,
    max_dim: usize,
    grid: &[f32],
) -> TdaResult<Vec<BettiCurve>> {
    let mut curves = Vec::with_capacity(max_dim + 1);
    for d in 0..=max_dim {
        curves.push(betti_curve(diagram, d, grid)?);
    }
    Ok(curves)
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::homology::persistent::PersistencePair;

    fn diag(pairs: &[(usize, f64, Option<f64>)]) -> PersistenceDiagram {
        let ps = pairs
            .iter()
            .map(|&(dim, b, d)| PersistencePair {
                dim,
                birth: b,
                death: d,
            })
            .collect();
        PersistenceDiagram::new(ps, 0)
    }

    #[test]
    fn decisive_h0_h1_curves() {
        // H0 class [0, 2); H1 class [1, 3). Grid at midpoints 0.5, 1.5, 2.5.
        let d = diag(&[(0, 0.0, Some(2.0)), (1, 1.0, Some(3.0))]);
        let grid = [0.5_f32, 1.5, 2.5];

        let h0 = betti_curve(&d, 0, &grid).expect("h0");
        assert_eq!(h0.values, vec![1, 1, 0], "H0 alive on [0,2)");

        let h1 = betti_curve(&d, 1, &grid).expect("h1");
        assert_eq!(h1.values, vec![0, 1, 1], "H1 alive on [1,3)");
    }

    #[test]
    fn half_open_boundary() {
        // Pair [1, 2): alive at t=1 (1 ≤ 1 < 2), dead at t=2 (2 ≮ 2).
        let d = diag(&[(0, 1.0, Some(2.0))]);
        let grid = [1.0_f32, 2.0];
        let c = betti_curve(&d, 0, &grid).expect("curve");
        assert_eq!(c.values, vec![1, 0]);
    }

    #[test]
    fn essential_class_lives_forever() {
        // Essential [0, ∞): alive at every grid point ≥ 0.
        let d = diag(&[(0, 0.0, None)]);
        let grid = [0.0_f32, 5.0, 100.0, 1e6];
        let c = betti_curve(&d, 0, &grid).expect("curve");
        assert_eq!(c.values, vec![1, 1, 1, 1]);
    }

    #[test]
    fn dimension_filtering() {
        // One H0 and one H1 pair both alive on [0, 10). Only the requested dim counts.
        let d = diag(&[(0, 0.0, Some(10.0)), (1, 0.0, Some(10.0))]);
        let grid = [1.0_f32, 2.0];
        let h0 = betti_curve(&d, 0, &grid).expect("h0");
        let h1 = betti_curve(&d, 1, &grid).expect("h1");
        let h2 = betti_curve(&d, 2, &grid).expect("h2");
        assert_eq!(h0.values, vec![1, 1]);
        assert_eq!(h1.values, vec![1, 1]);
        assert_eq!(h2.values, vec![0, 0], "no H2 pairs");
    }

    #[test]
    fn empty_diagram_is_all_zero() {
        let d = diag(&[]);
        let grid = [0.0_f32, 1.0, 2.0, 3.0];
        let c = betti_curve(&d, 0, &grid).expect("curve");
        assert_eq!(c.values, vec![0, 0, 0, 0]);
    }

    #[test]
    fn all_dims_has_correct_length() {
        let d = diag(&[
            (0, 0.0, Some(2.0)),
            (1, 1.0, Some(3.0)),
            (2, 0.5, Some(4.0)),
        ]);
        let grid = [0.5_f32, 1.5, 2.5];
        let curves = betti_curves_all_dims(&d, 2, &grid).expect("curves");
        assert_eq!(curves.len(), 3);
        assert_eq!(curves[0].dim, 0);
        assert_eq!(curves[1].dim, 1);
        assert_eq!(curves[2].dim, 2);
    }

    #[test]
    fn area_trapezoid() {
        // values [1, 1, 0] on grid [0, 1, 2]:
        //   trap(0→1) = 0.5*(1+1)*1 = 1.0; trap(1→2) = 0.5*(1+0)*1 = 0.5; total = 1.5.
        let c = BettiCurve {
            dim: 0,
            grid: vec![0.0, 1.0, 2.0],
            values: vec![1, 1, 0],
        };
        assert!((c.area() - 1.5).abs() < 1e-6, "area = {}", c.area());
    }

    #[test]
    fn area_too_few_points_is_zero() {
        let c = BettiCurve {
            dim: 0,
            grid: vec![1.0],
            values: vec![3],
        };
        assert_eq!(c.area(), 0.0);
    }

    #[test]
    fn l2_self_is_zero_and_mismatch_errors() {
        let d = diag(&[(0, 0.0, Some(2.0))]);
        let grid = [0.0_f32, 1.0, 2.0];
        let c = betti_curve(&d, 0, &grid).expect("curve");
        let self_dist = c.l2_distance(&c).expect("self");
        assert!(self_dist.abs() < 1e-6, "self distance = {self_dist}");

        let short_grid = [0.0_f32, 1.0];
        let c2 = betti_curve(&d, 0, &short_grid).expect("curve2");
        assert!(
            c.l2_distance(&c2).is_err(),
            "grid length mismatch must error"
        );
    }

    #[test]
    fn nan_grid_errors() {
        let d = diag(&[(0, 0.0, Some(2.0))]);
        let grid = [0.0_f32, f32::NAN, 2.0];
        assert!(betti_curve(&d, 0, &grid).is_err());
    }

    #[test]
    fn from_barcode_matches_diagram() {
        let d = diag(&[(0, 0.0, Some(2.0)), (1, 1.0, None)]);
        let grid = [0.5_f32, 1.5, 2.5];
        let bc = Barcode::from_diagram(&d, 0.0);

        for dim in 0..=1 {
            let from_diag = betti_curve(&d, dim, &grid).expect("diag");
            let from_bar = betti_curve_from_barcode(&bc, dim, &grid).expect("barcode");
            assert_eq!(
                from_diag.values, from_bar.values,
                "barcode curve must match diagram curve for dim {dim}"
            );
        }
    }
}
