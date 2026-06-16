//! Tropical coordinates on the space of persistence diagrams (Kalisnik 2019;
//! Monod et al. 2019).
//!
//! A persistence diagram is an unordered *multiset* of birth-death points, so any
//! feature vector fed to a downstream learner must be invariant to the order in
//! which the points are listed. **Tropical functions** — polynomials in the
//! max-plus (tropical) semiring `(ℝ ∪ {−∞}, max, +)` — are naturally symmetric in
//! their arguments and therefore give a stable, permutation-invariant
//! vectorisation of a diagram.
//!
//! For a point `(b, d)` write its *persistence* `λ = d − b` and the *midpoint
//! coordinate* `m = (b + d) / 2`. Following Kalisnik's construction, the
//! tropical coordinates of a diagram `{(bᵢ, dᵢ)}` are the symmetric "max-of-sums"
//! statistics
//! ```text
//! F_j  =  max over j-subsets S ⊆ points  of  Σ_{i ∈ S} min(2·λᵢ, μᵢ)
//! ```
//! together with the simpler order statistics of the persistences. Concretely
//! this module returns, for the first [`TropicalConfig::n_coords`] orders `j`:
//! - the `j`-th largest persistence `λ_(j)` (the `max`-plus elementary symmetric
//!   value of degree 1 restricted to rank `j`), and
//! - the running tropical sum `Σ_{i ≤ j} λ_(i)` of the `j` largest persistences,
//!   which is the value of the degree-`j` tropical *power sum*.
//!
//! These coordinates are (i) symmetric in the diagram points, (ii) 1-Lipschitz
//! with respect to the bottleneck/Wasserstein geometry on the persistences, and
//! (iii) stable under the addition of low-persistence noise points, which is
//! exactly the behaviour Monod et al. exploit for statistical inference on
//! diagrams.
//!
//! References:
//! - S. Kalisnik, "Tropical coordinates on the space of persistence barcodes",
//!   Foundations of Computational Mathematics, 2019.
//! - S. Monod, S. Kalisnik, et al., "Tropical sufficient statistics for
//!   persistent homology", SIAM J. Applied Algebra & Geometry, 2019.

use crate::error::{TdaError, TdaResult};
use crate::persistence::diagram::PersistenceDiagram;

/// Configuration for [`tropical_coordinates`].
#[derive(Debug, Clone)]
pub struct TropicalConfig {
    /// Number of coordinate orders `j = 1..=n_coords` to compute. Each order
    /// contributes two values (a max-plus order statistic and a tropical power
    /// sum), so the output vector has length `2 · n_coords`.
    pub n_coords: usize,
    /// Whether to treat essential (infinite-persistence) classes as having a
    /// finite persistence equal to `essential_value`. If `false`, essential
    /// classes are skipped.
    pub include_essential: bool,
    /// Replacement persistence assigned to essential classes when
    /// `include_essential` is set.
    pub essential_value: f64,
}

impl Default for TropicalConfig {
    fn default() -> Self {
        Self {
            n_coords: 5,
            include_essential: false,
            essential_value: 0.0,
        }
    }
}

/// Compute the tropical coordinate vector of a persistence diagram.
///
/// The result has length `2 · cfg.n_coords`: for each order `j` (1-indexed) it
/// stores the `j`-th largest persistence followed by the tropical power sum of
/// the `j` largest persistences. Orders beyond the number of available points are
/// padded with `0`, so the output length is deterministic regardless of diagram
/// size.
///
/// # Errors
/// - [`TdaError::ParameterOutOfRange`] when `cfg.n_coords == 0`.
/// - [`TdaError::NanFiltrationValue`] when a finite pair has a NaN birth or death.
pub fn tropical_coordinates(
    diagram: &PersistenceDiagram,
    cfg: &TropicalConfig,
) -> TdaResult<Vec<f64>> {
    if cfg.n_coords == 0 {
        return Err(TdaError::ParameterOutOfRange(
            "n_coords must be >= 1".to_string(),
        ));
    }

    // Collect persistences λ = death − birth.
    let mut persistences = collect_persistences(diagram, cfg)?;
    // Sort descending so index j-1 is the j-th largest.
    persistences.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut coords = vec![0.0; 2 * cfg.n_coords];
    let mut running = 0.0;
    for j in 0..cfg.n_coords {
        let lam_j = persistences.get(j).copied().unwrap_or(0.0);
        running += lam_j;
        coords[2 * j] = lam_j; // j-th largest persistence (max-plus order stat)
        coords[2 * j + 1] = running; // tropical power sum of top (j+1)
    }
    Ok(coords)
}

/// Total persistence of a diagram, `Σ λᵢ` — the degree-`∞` tropical power sum,
/// also recoverable as the last running sum when `n_coords ≥ |points|`.
///
/// # Errors
/// [`TdaError::NanFiltrationValue`] when a finite pair has a NaN birth or death.
pub fn tropical_total_persistence(
    diagram: &PersistenceDiagram,
    cfg: &TropicalConfig,
) -> TdaResult<f64> {
    let persistences = collect_persistences(diagram, cfg)?;
    Ok(persistences.iter().sum())
}

/// Tropical max-plus polynomial value `max_i (cⱼ + λᵢ)` for a vector of
/// coefficients `coeffs` — a single tropical monomial evaluated against the
/// diagram. Returns `f64::NEG_INFINITY` (the tropical additive identity) for an
/// empty point set or empty coefficient list.
///
/// This is the elementary building block of Kalisnik's tropical functions and is
/// exposed for callers that wish to assemble custom symmetric features.
///
/// # Errors
/// [`TdaError::NanFiltrationValue`] when a finite pair has a NaN birth or death.
pub fn tropical_max_plus(
    diagram: &PersistenceDiagram,
    coeffs: &[f64],
    cfg: &TropicalConfig,
) -> TdaResult<f64> {
    let persistences = collect_persistences(diagram, cfg)?;
    if persistences.is_empty() || coeffs.is_empty() {
        return Ok(f64::NEG_INFINITY);
    }
    let mut best = f64::NEG_INFINITY;
    for &c in coeffs {
        for &lam in &persistences {
            let v = c + lam;
            if v > best {
                best = v;
            }
        }
    }
    Ok(best)
}

/// Read the (non-negative) persistences from a diagram, honouring the
/// essential-class policy in `cfg`.
fn collect_persistences(diagram: &PersistenceDiagram, cfg: &TropicalConfig) -> TdaResult<Vec<f64>> {
    let mut out = Vec::with_capacity(diagram.pairs.len());
    for p in &diagram.pairs {
        match p.death {
            Some(d) => {
                if p.birth.is_nan() || d.is_nan() {
                    return Err(TdaError::NanFiltrationValue);
                }
                let lam = (d - p.birth).max(0.0);
                out.push(lam);
            }
            None => {
                if cfg.include_essential {
                    if p.birth.is_nan() {
                        return Err(TdaError::NanFiltrationValue);
                    }
                    out.push(cfg.essential_value.max(0.0));
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::homology::persistent::PersistencePair;

    fn diagram(points: &[(f64, Option<f64>)]) -> PersistenceDiagram {
        let pairs = points
            .iter()
            .map(|&(b, d)| PersistencePair {
                dim: 1,
                birth: b,
                death: d,
            })
            .collect();
        PersistenceDiagram::new(pairs, 1)
    }

    fn cfg() -> TropicalConfig {
        TropicalConfig::default()
    }

    // 1. Output length is exactly 2 * n_coords.
    #[test]
    fn output_shape() {
        let d = diagram(&[(0.0, Some(3.0)), (1.0, Some(2.0))]);
        let v = tropical_coordinates(&d, &cfg()).expect("value should be present");
        assert_eq!(v.len(), 2 * cfg().n_coords);
    }

    // 2. All coordinates are finite and non-negative.
    #[test]
    fn nonneg_finite() {
        let d = diagram(&[(0.0, Some(5.0)), (2.0, Some(8.0)), (1.0, Some(1.5))]);
        let v = tropical_coordinates(&d, &cfg()).expect("value should be present");
        for &x in &v {
            assert!(x.is_finite() && x >= 0.0, "bad coordinate {x}");
        }
    }

    // 3. An empty diagram gives an all-zero vector of the right length.
    #[test]
    fn empty_diagram_zero() {
        let d = diagram(&[]);
        let v = tropical_coordinates(&d, &cfg()).expect("value should be present");
        assert_eq!(v.len(), 2 * cfg().n_coords);
        for &x in &v {
            assert_eq!(x, 0.0);
        }
    }

    // 4. The first order statistic equals the maximum persistence.
    #[test]
    fn first_coord_is_max_persistence() {
        let d = diagram(&[(0.0, Some(3.0)), (1.0, Some(7.0)), (2.0, Some(4.0))]);
        let v = tropical_coordinates(&d, &cfg()).expect("value should be present");
        // Persistences: 3, 6, 2 → max = 6.
        assert!((v[0] - 6.0).abs() < 1e-12, "max persistence {}", v[0]);
    }

    // 5. The tropical power sum at order j equals the sum of the j largest
    //    persistences and is non-decreasing in j.
    #[test]
    fn power_sum_monotone() {
        let d = diagram(&[(0.0, Some(3.0)), (0.0, Some(6.0)), (0.0, Some(2.0))]);
        let v = tropical_coordinates(&d, &cfg()).expect("value should be present");
        // Sorted persistences: 6, 3, 2 → running sums 6, 9, 11, 11, 11.
        assert!((v[1] - 6.0).abs() < 1e-12);
        assert!((v[3] - 9.0).abs() < 1e-12);
        assert!((v[5] - 11.0).abs() < 1e-12);
        // Power sums non-decreasing.
        for j in 1..cfg().n_coords {
            assert!(v[2 * j + 1] >= v[2 * (j - 1) + 1] - 1e-12);
        }
    }

    // 6. Permutation invariance: reordering the diagram points leaves the
    //    coordinates unchanged.
    #[test]
    fn permutation_invariant() {
        let a = diagram(&[(0.0, Some(3.0)), (1.0, Some(7.0)), (2.0, Some(4.0))]);
        let b = diagram(&[(2.0, Some(4.0)), (0.0, Some(3.0)), (1.0, Some(7.0))]);
        let va = tropical_coordinates(&a, &cfg()).expect("value should be present");
        let vb = tropical_coordinates(&b, &cfg()).expect("value should be present");
        for (x, y) in va.iter().zip(&vb) {
            assert!((x - y).abs() < 1e-12, "{x} vs {y}");
        }
    }

    // 7. n_coords == 0 errors.
    #[test]
    fn n_coords_0_error() {
        let d = diagram(&[(0.0, Some(1.0))]);
        let c = TropicalConfig {
            n_coords: 0,
            ..cfg()
        };
        let err = tropical_coordinates(&d, &c);
        assert!(
            matches!(err, Err(TdaError::ParameterOutOfRange(_))),
            "got {err:?}"
        );
    }

    // 8. NaN filtration value errors.
    #[test]
    fn nan_value_error() {
        let d = diagram(&[(0.0, Some(f64::NAN))]);
        let err = tropical_coordinates(&d, &cfg());
        assert!(
            matches!(err, Err(TdaError::NanFiltrationValue)),
            "got {err:?}"
        );
    }

    // 9. Essential classes are included only when requested.
    #[test]
    fn essential_policy() {
        let d = diagram(&[(0.0, Some(2.0)), (0.0, None)]);
        // Skipped by default: only the finite persistence (2) contributes.
        let v_skip = tropical_coordinates(&d, &cfg()).expect("value should be present");
        assert!((v_skip[0] - 2.0).abs() < 1e-12);
        assert!(
            v_skip[2].abs() < 1e-12,
            "second order should be 0, got {}",
            v_skip[2]
        );
        // Included with a large replacement value: it becomes the max.
        let c = TropicalConfig {
            include_essential: true,
            essential_value: 10.0,
            ..cfg()
        };
        let v_inc = tropical_coordinates(&d, &c).expect("tropical_coordinates should succeed");
        assert!(
            (v_inc[0] - 10.0).abs() < 1e-12,
            "essential not included: {}",
            v_inc[0]
        );
    }

    // 10. Total persistence equals the last running sum when n_coords covers all
    //     points, and the max-plus monomial agrees with the manual maximum.
    #[test]
    fn total_and_max_plus() {
        let d = diagram(&[(0.0, Some(3.0)), (0.0, Some(6.0)), (0.0, Some(2.0))]);
        let total = tropical_total_persistence(&d, &cfg()).expect("value should be present");
        assert!((total - 11.0).abs() < 1e-12, "total {total}");
        // max_i (c + λ_i) with c = 1 over persistences {3,6,2} = 7.
        let mp = tropical_max_plus(&d, &[1.0], &cfg()).expect("value should be present");
        assert!((mp - 7.0).abs() < 1e-12, "max-plus {mp}");
        // Empty coefficient list → tropical additive identity.
        let empty = tropical_max_plus(&d, &[], &cfg()).expect("value should be present");
        assert_eq!(empty, f64::NEG_INFINITY);
    }

    // 11. Adding a tiny-persistence noise point does not change the leading
    //     coordinates (stability under low-persistence noise).
    #[test]
    fn stable_under_noise() {
        let clean = diagram(&[(0.0, Some(5.0)), (1.0, Some(4.0))]);
        let noisy = diagram(&[(0.0, Some(5.0)), (1.0, Some(4.0)), (2.0, Some(2.001))]);
        let vc = tropical_coordinates(&clean, &cfg()).expect("value should be present");
        let vn = tropical_coordinates(&noisy, &cfg()).expect("value should be present");
        // The two largest persistences (5, 3) are identical; first two order
        // statistics must match exactly.
        assert!((vc[0] - vn[0]).abs() < 1e-12, "{} vs {}", vc[0], vn[0]);
        assert!((vc[2] - vn[2]).abs() < 1e-12, "{} vs {}", vc[2], vn[2]);
    }
}
