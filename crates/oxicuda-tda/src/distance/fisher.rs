//! Persistence Fisher distance and kernel between persistence diagrams.
//!
//! Le & Yamada, *Persistence Fisher Kernel: A Riemannian Manifold Kernel for
//! Persistence Diagrams* (NeurIPS 2018).  A diagram is smoothed into a probability
//! density on the birth–death plane; the **Fisher information metric** on the
//! statistical manifold of such densities then yields a genuine (geodesic) distance,
//! and exponentiating it gives a positive-definite kernel.
//!
//! ## Construction
//!
//! Let `D_i`, `D_j` be the finite off-diagonal points of two diagrams and let
//! `Δ(D)` be the orthogonal projection of every point of `D` onto the diagonal
//! (`(b, d) ↦ ((b+d)/2, (b+d)/2)`).  Form the two *augmented* point sets
//!
//! ```text
//!   P = D_i ∪ Δ(D_j),     Q = D_j ∪ Δ(D_i)
//! ```
//!
//! (augmenting each diagram with the diagonal shadow of the other makes the two
//! comparable even when they have different cardinalities).  Smooth each set into a
//! density with an isotropic Gaussian of bandwidth `σ`, evaluate both densities on the
//! common support `Θ = P ∪ Q`, normalise them to sum to one over `Θ`, and take the
//! Fisher (Bhattacharyya/Hellinger-arccos) distance
//!
//! ```text
//!   d_FIM(D_i, D_j) = arccos( Σ_{θ∈Θ} sqrt( ρ_P(θ) · ρ_Q(θ) ) ) ∈ [0, π/2].
//! ```
//!
//! The inner sum is the Bhattacharyya coefficient of the two normalised densities, so
//! it lies in `[0, 1]` (Cauchy–Schwarz) and the arccos is well defined.  The kernel is
//!
//! ```text
//!   k_FIM(D_i, D_j) = exp( −t · d_FIM(D_i, D_j) ),   t > 0,
//! ```
//!
//! which equals `1` exactly on the diagonal and lies in `(0, 1]`.  Only the finite
//! off-diagonal points participate; points on the diagonal carry no topological signal
//! and are dropped, exactly as in [`crate::distance::kernel`] and
//! [`mod@crate::persistence::persistence_image`].

use crate::error::{TdaError, TdaResult};
use crate::persistence::diagram::PersistenceDiagram;

/// Configuration for the persistence Fisher distance and kernel.
#[derive(Debug, Clone)]
pub struct PersistenceFisherConfig {
    /// Gaussian smoothing bandwidth `σ > 0` used to turn a diagram into a density.
    pub sigma: f64,
    /// Kernel bandwidth `t > 0` in `k = exp(−t · d_FIM)` (unused by the distance).
    pub bandwidth: f64,
}

impl Default for PersistenceFisherConfig {
    fn default() -> Self {
        Self {
            sigma: 1.0,
            bandwidth: 1.0,
        }
    }
}

/// Collect the finite off-diagonal points `(birth, death)` of a diagram
/// (`death > birth`).
fn finite_points(diag: &PersistenceDiagram) -> Vec<(f64, f64)> {
    diag.finite_pairs()
        .iter()
        .filter_map(|p| {
            let d = p.death?;
            if d > p.birth {
                Some((p.birth, d))
            } else {
                None
            }
        })
        .collect()
}

/// Squared Euclidean distance between two birth–death points.
#[inline]
fn dist_sq(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}

/// Orthogonal projection of a birth–death point onto the diagonal.
#[inline]
fn diagonal_projection(p: (f64, f64)) -> (f64, f64) {
    let m = 0.5 * (p.0 + p.1);
    (m, m)
}

/// Fisher information distance between two finite point sets, following the Le–Yamada
/// construction described in the module documentation.
fn fisher_distance_points(pts_i: &[(f64, f64)], pts_j: &[(f64, f64)], sigma: f64) -> f64 {
    // P = D_i ∪ Δ(D_j),  Q = D_j ∪ Δ(D_i).
    let mut p_set: Vec<(f64, f64)> = pts_i.to_vec();
    p_set.extend(pts_j.iter().map(|&q| diagonal_projection(q)));
    let mut q_set: Vec<(f64, f64)> = pts_j.to_vec();
    q_set.extend(pts_i.iter().map(|&p| diagonal_projection(p)));

    // Common support Θ = P ∪ Q (as a list; duplicates are harmless).
    let theta: Vec<(f64, f64)> = p_set.iter().chain(q_set.iter()).copied().collect();
    if theta.is_empty() {
        // Two empty diagrams are identical.
        return 0.0;
    }

    let denom = 2.0 * sigma * sigma;
    let density = |set: &[(f64, f64)], x: (f64, f64)| -> f64 {
        set.iter().map(|&u| (-dist_sq(x, u) / denom).exp()).sum()
    };

    let rho_p: Vec<f64> = theta.iter().map(|&x| density(&p_set, x)).collect();
    let rho_q: Vec<f64> = theta.iter().map(|&x| density(&q_set, x)).collect();
    let sum_p: f64 = rho_p.iter().sum();
    let sum_q: f64 = rho_q.iter().sum();

    // theta is non-empty ⇒ both P and Q are non-empty ⇒ both sums are > 0.
    if sum_p <= 0.0 || sum_q <= 0.0 {
        return 0.0;
    }

    let mut bhattacharyya = 0.0_f64;
    for (rp, rq) in rho_p.iter().zip(rho_q.iter()) {
        bhattacharyya += ((rp / sum_p) * (rq / sum_q)).sqrt();
    }
    bhattacharyya.clamp(0.0, 1.0).acos()
}

/// Persistence Fisher distance `d_FIM(D₁, D₂) ∈ [0, π/2]` (Le & Yamada 2018).
///
/// Only [`PersistenceFisherConfig::sigma`] is used.
///
/// # Errors
/// [`TdaError::ParameterOutOfRange`] if `sigma` is not strictly positive and finite.
pub fn persistence_fisher_distance(
    d1: &PersistenceDiagram,
    d2: &PersistenceDiagram,
    cfg: &PersistenceFisherConfig,
) -> TdaResult<f64> {
    if cfg.sigma <= 0.0 || !cfg.sigma.is_finite() {
        return Err(TdaError::ParameterOutOfRange(format!(
            "sigma must be > 0 and finite, got {}",
            cfg.sigma
        )));
    }
    let p1 = finite_points(d1);
    let p2 = finite_points(d2);
    Ok(fisher_distance_points(&p1, &p2, cfg.sigma))
}

/// Persistence Fisher kernel `k_FIM(D₁, D₂) = exp(−t · d_FIM(D₁, D₂)) ∈ (0, 1]`.
///
/// Uses both [`PersistenceFisherConfig::sigma`] (for the distance) and
/// [`PersistenceFisherConfig::bandwidth`] (`t`).
///
/// # Errors
/// [`TdaError::ParameterOutOfRange`] if `sigma` or `bandwidth` is not strictly positive
/// and finite.
pub fn persistence_fisher_kernel(
    d1: &PersistenceDiagram,
    d2: &PersistenceDiagram,
    cfg: &PersistenceFisherConfig,
) -> TdaResult<f64> {
    if cfg.bandwidth <= 0.0 || !cfg.bandwidth.is_finite() {
        return Err(TdaError::ParameterOutOfRange(format!(
            "bandwidth must be > 0 and finite, got {}",
            cfg.bandwidth
        )));
    }
    let distance = persistence_fisher_distance(d1, d2, cfg)?;
    Ok((-cfg.bandwidth * distance).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::homology::persistent::PersistencePair;
    use std::f64::consts::FRAC_PI_2;

    fn diag(pts: &[(f64, f64)]) -> PersistenceDiagram {
        let pairs = pts
            .iter()
            .map(|&(b, d)| PersistencePair {
                dim: 0,
                birth: b,
                death: Some(d),
            })
            .collect();
        PersistenceDiagram::new(pairs, 0)
    }

    #[test]
    fn self_distance_zero_self_kernel_one() {
        let d = diag(&[(0.0, 2.0), (1.0, 4.0)]);
        let cfg = PersistenceFisherConfig::default();
        let dist = persistence_fisher_distance(&d, &d, &cfg).expect("dist");
        assert!(dist < 1e-9, "self-distance must be ~0, got {dist}");
        let k = persistence_fisher_kernel(&d, &d, &cfg).expect("k");
        assert!((k - 1.0).abs() < 1e-9, "self-kernel must be ~1, got {k}");
    }

    #[test]
    fn symmetric() {
        let a = diag(&[(0.0, 2.0)]);
        let b = diag(&[(0.5, 3.0), (1.0, 4.0)]);
        let cfg = PersistenceFisherConfig::default();
        let dab = persistence_fisher_distance(&a, &b, &cfg).expect("dab");
        let dba = persistence_fisher_distance(&b, &a, &cfg).expect("dba");
        assert!((dab - dba).abs() < 1e-9, "distance must be symmetric");
        let kab = persistence_fisher_kernel(&a, &b, &cfg).expect("kab");
        let kba = persistence_fisher_kernel(&b, &a, &cfg).expect("kba");
        assert!((kab - kba).abs() < 1e-9, "kernel must be symmetric");
    }

    #[test]
    fn distance_in_range_kernel_in_range() {
        let a = diag(&[(0.0, 2.0)]);
        let b = diag(&[(3.0, 9.0), (1.0, 2.0)]);
        let cfg = PersistenceFisherConfig::default();
        let dist = persistence_fisher_distance(&a, &b, &cfg).expect("dist");
        assert!(
            (0.0..=FRAC_PI_2 + 1e-9).contains(&dist),
            "distance {dist} out of [0, π/2]"
        );
        let k = persistence_fisher_kernel(&a, &b, &cfg).expect("k");
        assert!(k > 0.0 && k <= 1.0 + 1e-12, "kernel {k} out of (0, 1]");
    }

    #[test]
    fn distinct_diagrams_positive_distance() {
        let a = diag(&[(0.0, 2.0)]);
        let b = diag(&[(0.0, 8.0)]);
        let cfg = PersistenceFisherConfig::default();
        let dist = persistence_fisher_distance(&a, &b, &cfg).expect("dist");
        assert!(dist > 0.0, "distinct diagrams must have positive distance");
        let k = persistence_fisher_kernel(&a, &b, &cfg).expect("k");
        assert!(k < 1.0, "distinct diagrams must have kernel < 1, got {k}");
    }

    #[test]
    fn distance_grows_with_separation() {
        // B is a small perturbation of A; C is far from A.
        let a = diag(&[(0.0, 2.0)]);
        let b = diag(&[(0.0, 2.6)]);
        let c = diag(&[(0.0, 7.0)]);
        let cfg = PersistenceFisherConfig::default();
        let dab = persistence_fisher_distance(&a, &b, &cfg).expect("dab");
        let dac = persistence_fisher_distance(&a, &c, &cfg).expect("dac");
        assert!(
            dac > dab,
            "farther diagram must be more distant: d(A,C)={dac} ≤ d(A,B)={dab}"
        );
        // The kernel decreases as the distance grows.
        let kab = persistence_fisher_kernel(&a, &b, &cfg).expect("kab");
        let kac = persistence_fisher_kernel(&a, &c, &cfg).expect("kac");
        assert!(kac < kab, "kernel must shrink with distance");
    }

    #[test]
    fn empty_diagrams() {
        let empty = diag(&[]);
        let nonempty = diag(&[(0.0, 3.0)]);
        let cfg = PersistenceFisherConfig::default();
        // Two empty diagrams are identical.
        let d_ee = persistence_fisher_distance(&empty, &empty, &cfg).expect("d_ee");
        assert!(d_ee < 1e-12, "empty vs empty must be 0, got {d_ee}");
        let k_ee = persistence_fisher_kernel(&empty, &empty, &cfg).expect("k_ee");
        assert!((k_ee - 1.0).abs() < 1e-12);
        // Empty vs non-empty differ.
        let d_en = persistence_fisher_distance(&empty, &nonempty, &cfg).expect("d_en");
        assert!(d_en > 0.0, "empty vs non-empty must differ, got {d_en}");
    }

    #[test]
    fn bad_parameters_error() {
        let d = diag(&[(0.0, 2.0)]);
        // sigma ≤ 0.
        let cfg = PersistenceFisherConfig {
            sigma: 0.0,
            bandwidth: 1.0,
        };
        assert!(persistence_fisher_distance(&d, &d, &cfg).is_err());
        assert!(persistence_fisher_kernel(&d, &d, &cfg).is_err());
        // bandwidth ≤ 0 (kernel only).
        let cfg = PersistenceFisherConfig {
            sigma: 1.0,
            bandwidth: 0.0,
        };
        assert!(persistence_fisher_kernel(&d, &d, &cfg).is_err());
        // Distance still fine with a valid sigma.
        assert!(persistence_fisher_distance(&d, &d, &cfg).is_ok());
    }

    #[test]
    fn diagonal_points_dropped() {
        // A point on the diagonal carries no signal: distance to the empty diagram is 0.
        let on_diag = diag(&[(2.0, 2.0)]);
        let empty = diag(&[]);
        let cfg = PersistenceFisherConfig::default();
        let dist = persistence_fisher_distance(&on_diag, &empty, &cfg).expect("dist");
        assert!(
            dist < 1e-12,
            "diagonal-only diagram must equal empty, got {dist}"
        );
    }
}
