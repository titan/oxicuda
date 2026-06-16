//! Refinement indicators and marking strategies for adaptive mesh refinement.
//!
//! Given a scalar field sampled once per leaf of a [`crate::amr::octree::Quadtree`],
//! we compute a per-cell *refinement indicator* that estimates where the solution
//! is under-resolved, then *mark* cells for refinement or coarsening.
//!
//! # Indicators
//!
//! * **Jump indicator** — `ind_K = max_{K' ∈ N(K)} |u_K − u_{K'}|` over the
//!   face neighbours `N(K)`. Large inter-cell jumps flag steep gradients /
//!   discontinuities; this is the discrete analogue of a gradient-recovery
//!   estimator and is robust on non-conforming AMR meshes.
//! * **Gradient×size indicator** — `ind_K = h_K · |∇u|_K`, where the cell
//!   gradient is estimated from neighbour finite differences. Scaling by the
//!   cell size `h_K` makes the indicator consistent with an `O(h)` truncation
//!   contribution, so refining high-indicator cells equidistributes error.
//!
//! # Marking
//!
//! * **Fixed-fraction (Dörfler) marking** — choose the *smallest* set `M` of
//!   cells whose summed (squared) indicator covers a fraction `θ ∈ (0,1]` of the
//!   total: `Σ_{K∈M} ind_K² ≥ θ Σ_K ind_K²`. This bulk criterion drives provably
//!   optimal AMR convergence (Dörfler, SIAM J. Numer. Anal. 33, 1996).
//! * **Threshold marking** — refine cells above `refine_frac · max_ind` and
//!   coarsen cells below `coarsen_frac · max_ind`.

use crate::amr::octree::Quadtree;
use crate::error::{PdeError, PdeResult};

/// Per-cell indicator field paired with the leaf indices it was computed on.
///
/// `leaves[k]` is the quadtree cell index whose indicator is `values[k]`.
#[derive(Debug, Clone)]
pub struct Indicators {
    /// Quadtree leaf indices, in the order returned by [`Quadtree::leaves`].
    pub leaves: Vec<usize>,
    /// Indicator value per leaf (same length / order as `leaves`).
    pub values: Vec<f64>,
}

impl Indicators {
    /// Total summed-squared indicator `Σ_K ind_K²` (the AMR error proxy).
    #[must_use]
    pub fn total_squared(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum()
    }

    /// Maximum indicator value (`0` if empty).
    #[must_use]
    pub fn max(&self) -> f64 {
        self.values.iter().copied().fold(0.0, f64::max)
    }
}

/// A subset of leaves selected by a marking strategy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarkedCells {
    /// Quadtree leaf indices marked for **refinement**.
    pub refine: Vec<usize>,
    /// Quadtree leaf indices marked for **coarsening**.
    pub coarsen: Vec<usize>,
}

/// Jump-based refinement indicator: `ind_K = max_{K'} |u_K − u_{K'}|`.
///
/// `field[k]` is the scalar value on leaf `tree.leaves()[k]`. The result is
/// aligned with the same leaf ordering.
///
/// # Errors
/// [`PdeError::DimensionMismatch`] if `field.len()` differs from the leaf count.
pub fn jump_indicator(tree: &Quadtree, field: &[f64]) -> PdeResult<Indicators> {
    let leaves = tree.leaves();
    if field.len() != leaves.len() {
        return Err(PdeError::DimensionMismatch {
            a: field.len(),
            b: leaves.len(),
        });
    }
    // Map leaf cell-index -> position in `field`.
    let mut pos_of = vec![usize::MAX; tree.cell_count()];
    for (k, &cell) in leaves.iter().enumerate() {
        pos_of[cell] = k;
    }
    let mut values = vec![0.0; leaves.len()];
    for (k, &cell) in leaves.iter().enumerate() {
        let uk = field[k];
        let mut max_jump = 0.0_f64;
        for n in tree.face_neighbors(cell) {
            let p = pos_of[n];
            if p != usize::MAX {
                let jump = (uk - field[p]).abs();
                if jump > max_jump {
                    max_jump = jump;
                }
            }
        }
        values[k] = max_jump;
    }
    Ok(Indicators { leaves, values })
}

/// Gradient-magnitude × cell-size indicator: `ind_K = h_K · |∇u|_K`.
///
/// The gradient is estimated by central/one-sided differences against face
/// neighbours using cell-centre separations. On a uniform patch this reduces to
/// `h · |Δu| / Δx ≈ |Δu|`, i.e. the local solution variation.
///
/// # Errors
/// [`PdeError::DimensionMismatch`] if `field.len()` differs from the leaf count.
pub fn gradient_indicator(tree: &Quadtree, field: &[f64]) -> PdeResult<Indicators> {
    let leaves = tree.leaves();
    if field.len() != leaves.len() {
        return Err(PdeError::DimensionMismatch {
            a: field.len(),
            b: leaves.len(),
        });
    }
    let centers = tree.leaf_centers();
    let sizes = tree.leaf_sizes();
    let mut pos_of = vec![usize::MAX; tree.cell_count()];
    for (k, &cell) in leaves.iter().enumerate() {
        pos_of[cell] = k;
    }

    let mut values = vec![0.0; leaves.len()];
    for (k, &cell) in leaves.iter().enumerate() {
        let (cx, cy) = centers[k];
        let uk = field[k];
        // Least-squares gradient from neighbour differences:
        //   minimise Σ_n ( g·(x_n − x_k) − (u_n − u_k) )².
        // Normal equations: (Σ d dᵀ) g = Σ d (Δu).
        let mut a11 = 0.0;
        let mut a12 = 0.0;
        let mut a22 = 0.0;
        let mut b1 = 0.0;
        let mut b2 = 0.0;
        for n in tree.face_neighbors(cell) {
            let p = pos_of[n];
            if p == usize::MAX {
                continue;
            }
            let (nx, ny) = centers[p];
            let dx = nx - cx;
            let dy = ny - cy;
            let du = field[p] - uk;
            a11 += dx * dx;
            a12 += dx * dy;
            a22 += dy * dy;
            b1 += dx * du;
            b2 += dy * du;
        }
        let det = a11 * a22 - a12 * a12;
        let grad_mag = if det.abs() > 1.0e-30 {
            let gx = (b1 * a22 - b2 * a12) / det;
            let gy = (a11 * b2 - a12 * b1) / det;
            (gx * gx + gy * gy).sqrt()
        } else {
            0.0
        };
        values[k] = sizes[k] * grad_mag;
    }
    Ok(Indicators { leaves, values })
}

/// Fixed-fraction (Dörfler / bulk-chasing) marking for refinement.
///
/// Selects the smallest set `M` of leaves (largest indicators first) such that
/// `Σ_{K∈M} ind_K² ≥ θ · total`, where `total = Σ_K ind_K²`. This is the
/// canonical bulk criterion guaranteeing the marked cells carry at least a
/// fraction `θ` of the global error.
///
/// # Errors
/// * [`PdeError::InvalidParameter`] if `theta ∉ (0, 1]`.
pub fn dorfler_mark(indicators: &Indicators, theta: f64) -> PdeResult<MarkedCells> {
    if !(theta > 0.0 && theta <= 1.0) {
        return Err(PdeError::InvalidParameter {
            name: "theta".into(),
            reason: "Dörfler fraction must lie in (0, 1]".into(),
        });
    }
    let total: f64 = indicators.total_squared();
    if total <= 0.0 {
        // A flat field: nothing to refine.
        return Ok(MarkedCells {
            refine: Vec::new(),
            coarsen: Vec::new(),
        });
    }
    // Sort leaf positions by descending squared indicator.
    let mut order: Vec<usize> = (0..indicators.values.len()).collect();
    order.sort_by(|&i, &j| {
        let vi = indicators.values[i] * indicators.values[i];
        let vj = indicators.values[j] * indicators.values[j];
        vj.partial_cmp(&vi).unwrap_or(std::cmp::Ordering::Equal)
    });
    let target = theta * total;
    let mut acc = 0.0;
    let mut refine = Vec::new();
    for &k in &order {
        if acc >= target {
            break;
        }
        acc += indicators.values[k] * indicators.values[k];
        refine.push(indicators.leaves[k]);
    }
    Ok(MarkedCells {
        refine,
        coarsen: Vec::new(),
    })
}

/// Threshold marking relative to the maximum indicator.
///
/// Marks for refinement every leaf with `ind_K ≥ refine_frac · max_ind`, and for
/// coarsening every leaf with `ind_K ≤ coarsen_frac · max_ind`.
///
/// # Errors
/// [`PdeError::InvalidParameter`] if the fractions are out of `[0,1]` or if
/// `coarsen_frac > refine_frac` (overlapping bands).
pub fn threshold_mark(
    indicators: &Indicators,
    refine_frac: f64,
    coarsen_frac: f64,
) -> PdeResult<MarkedCells> {
    if !(0.0..=1.0).contains(&refine_frac) || !(0.0..=1.0).contains(&coarsen_frac) {
        return Err(PdeError::InvalidParameter {
            name: "frac".into(),
            reason: "fractions must lie in [0, 1]".into(),
        });
    }
    if coarsen_frac > refine_frac {
        return Err(PdeError::InvalidParameter {
            name: "coarsen_frac".into(),
            reason: "coarsen_frac must not exceed refine_frac".into(),
        });
    }
    let max_ind = indicators.max();
    let refine_thr = refine_frac * max_ind;
    let coarsen_thr = coarsen_frac * max_ind;
    let mut refine = Vec::new();
    let mut coarsen = Vec::new();
    for (k, &v) in indicators.values.iter().enumerate() {
        if max_ind > 0.0 && v >= refine_thr {
            refine.push(indicators.leaves[k]);
        } else if v <= coarsen_thr {
            coarsen.push(indicators.leaves[k]);
        }
    }
    Ok(MarkedCells { refine, coarsen })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amr::octree::Quadtree;

    /// Build a uniformly 2×2 refined unit quadtree and sample `f(x,y)` per leaf.
    fn refined_grid_field<F: Fn(f64, f64) -> f64>(levels: usize, f: F) -> (Quadtree, Vec<f64>) {
        let mut t = Quadtree::new(0.0, 0.0, 1.0, 1.0).expect("root");
        for _ in 0..levels {
            for l in t.leaves() {
                // refine() invalidates appended indices only; safe to refine all
                // current leaves in a fresh snapshot.
                let _ = t.refine(l);
            }
        }
        let field: Vec<f64> = t.leaf_centers().into_iter().map(|(x, y)| f(x, y)).collect();
        (t, field)
    }

    #[test]
    fn jump_indicator_flags_step_not_flat() {
        // Step field: u = 1 for x > 0.5, else 0. The jump indicator must be large
        // for cells straddling x = 0.5 and zero deep inside each constant region.
        let (tree, field) = refined_grid_field(3, |x, _y| if x > 0.5 { 1.0 } else { 0.0 });
        let ind = jump_indicator(&tree, &field).expect("ind");
        let centers = tree.leaf_centers();

        let mut near_interface_max = 0.0_f64;
        let mut flat_region_max = 0.0_f64;
        for (k, &(x, _y)) in centers.iter().enumerate() {
            if (x - 0.5).abs() < 0.1 {
                near_interface_max = near_interface_max.max(ind.values[k]);
            }
            if !(0.25..=0.75).contains(&x) {
                flat_region_max = flat_region_max.max(ind.values[k]);
            }
        }
        assert!(
            near_interface_max > 0.5,
            "interface cells must have large jump, got {near_interface_max}"
        );
        assert!(
            flat_region_max < 1e-12,
            "flat regions must have ~zero jump, got {flat_region_max}"
        );
    }

    #[test]
    fn gradient_indicator_larger_for_steeper_field() {
        // A linear ramp u = x has constant gradient; a flat field has zero.
        let (tree, ramp) = refined_grid_field(3, |x, _y| 3.0 * x);
        let (_t2, flat) = refined_grid_field(3, |_x, _y| 1.0);
        let ramp_ind = gradient_indicator(&tree, &ramp).expect("ind");
        let flat_ind = gradient_indicator(&tree, &flat).expect("ind");
        assert!(ramp_ind.max() > 1e-3, "ramp gradient must be detected");
        assert!(flat_ind.max() < 1e-12, "flat field gradient ≈ 0");
    }

    #[test]
    fn dorfler_selects_minimal_covering_set() {
        // Indicators with a clear heavy tail: one big, several tiny.
        let leaves = vec![0, 1, 2, 3, 4];
        let values = vec![10.0, 1.0, 1.0, 1.0, 1.0];
        let ind = Indicators { leaves, values };
        let total = ind.total_squared(); // 100 + 4 = 104.
        // θ = 0.9 → need ≥ 93.6; the single big cell (100) already covers it.
        let marked = dorfler_mark(&ind, 0.9).expect("mark");
        assert_eq!(marked.refine, vec![0], "the single dominant cell suffices");
        // The selected set indeed covers the fraction.
        let covered: f64 = marked
            .refine
            .iter()
            .map(|&c| {
                let p = ind.leaves.iter().position(|&l| l == c).expect("pos");
                ind.values[p] * ind.values[p]
            })
            .sum();
        assert!(covered >= 0.9 * total);
        // Removing any marked cell would drop below the target (minimality).
        assert!(covered - 100.0 < 0.9 * total);
    }

    #[test]
    fn dorfler_full_fraction_marks_all_nonzero() {
        let ind = Indicators {
            leaves: vec![0, 1, 2],
            values: vec![2.0, 3.0, 4.0],
        };
        let marked = dorfler_mark(&ind, 1.0).expect("mark");
        assert_eq!(
            marked.refine.len(),
            3,
            "θ=1 must mark every contributing cell"
        );
    }

    #[test]
    fn dorfler_flat_field_marks_nothing() {
        let ind = Indicators {
            leaves: vec![0, 1, 2],
            values: vec![0.0, 0.0, 0.0],
        };
        let marked = dorfler_mark(&ind, 0.5).expect("mark");
        assert!(marked.refine.is_empty());
    }

    #[test]
    fn dorfler_rejects_bad_theta() {
        let ind = Indicators {
            leaves: vec![0],
            values: vec![1.0],
        };
        assert!(dorfler_mark(&ind, 0.0).is_err());
        assert!(dorfler_mark(&ind, 1.5).is_err());
    }

    #[test]
    fn threshold_mark_partitions_high_and_low() {
        let ind = Indicators {
            leaves: vec![0, 1, 2, 3],
            values: vec![10.0, 8.0, 1.0, 0.5],
        };
        let marked = threshold_mark(&ind, 0.7, 0.1).expect("mark");
        // refine threshold = 7.0 -> {10, 8}; coarsen threshold = 1.0 -> {1.0, 0.5}.
        assert!(marked.refine.contains(&0) && marked.refine.contains(&1));
        assert!(marked.coarsen.contains(&2) && marked.coarsen.contains(&3));
        assert!(!marked.refine.contains(&2));
    }

    #[test]
    fn jump_indicator_dimension_mismatch_errs() {
        let tree = Quadtree::new(0.0, 0.0, 1.0, 1.0).expect("root");
        assert!(jump_indicator(&tree, &[1.0, 2.0]).is_err());
    }
}
