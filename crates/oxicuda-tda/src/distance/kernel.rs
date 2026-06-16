//! Positive-definite kernels on the space of persistence diagrams.
//!
//! These turn persistence diagrams into elements of a reproducing-kernel Hilbert
//! space (RKHS), enabling kernel SVMs, kernel PCA, two-sample tests, etc.  Three
//! kernels are implemented here, all over the **f64** birth–death coordinates of the
//! *finite* points of a diagram (off-diagonal points only; diagonal points are
//! filtered out — they carry no topological signal and are required to vanish for
//! the scale-space construction below).
//!
//! ## Persistence Scale-Space Kernel (PSSK)
//!
//! Reininghaus, Huber, Bauer, Kwitt, "A Stable Multi-Scale Kernel for Topological
//! Machine Learning", CVPR 2015.  Each diagram `D` is mapped to the solution of a
//! heat-diffusion problem on the half-plane with a Dirichlet boundary condition on
//! the diagonal; the resulting feature map has the closed-form inner product
//!
//! ```text
//!   k_σ(D, E) = (1 / (8πσ)) · Σ_{p∈D} Σ_{q∈E}
//!                 [ exp(−‖p − q‖² / (8σ)) − exp(−‖p − q̄‖² / (8σ)) ]
//! ```
//!
//! where `q̄ = (q_death, q_birth)` is the mirror of `q` across the diagonal.  The
//! second (subtracted) term enforces the Dirichlet condition, so a point on the
//! diagonal (`p = p̄`) contributes nothing.
//!
//! ## Persistence Weighted Gaussian Kernel (PWGK)
//!
//! Kusano, Hiraoka, Fukumizu, "Persistence Weighted Gaussian Kernel for Topological
//! Data Analysis", ICML 2016.  A weighted kernel-mean embedding:
//!
//! ```text
//!   k(D, E) = Σ_{x∈D} Σ_{y∈E} w(x) w(y) exp(−‖x − y‖² / (2τ²)),
//!   w(x)    = arctan(C · pers(x)^p),   pers(x) = death − birth.
//! ```
//!
//! ## Sliced Wasserstein Kernel
//!
//! Carrière, Cuturi, Oudot, "Sliced Wasserstein Kernel for Persistence Diagrams",
//! ICML 2017.  Built from the (negative-definite) sliced Wasserstein distance `SW`
//! already available in [`mod@crate::persistence::wasserstein_p`] as
//! `k(D, E) = exp(−SW(D, E) / (2η²))`.

use crate::error::{TdaError, TdaResult};
use crate::handle::LcgRng;
use crate::persistence::diagram::PersistenceDiagram;
use crate::persistence::wasserstein_p::sliced_wasserstein;
use std::f64::consts::PI;

// ─── Configuration ──────────────────────────────────────────────────────────────

/// Configuration for the persistence scale-space kernel.
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Scale parameter σ > 0.  Larger σ ⇒ smoother feature map (coarser scale).
    pub sigma: f64,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self { sigma: 1.0 }
    }
}

/// Configuration for the persistence weighted Gaussian kernel.
#[derive(Debug, Clone)]
pub struct PwgkConfig {
    /// Gaussian bandwidth τ > 0 on birth–death space.
    pub tau: f64,
    /// Weight scale `C` in `w(x) = arctan(C · pers^p)`.
    pub weight_c: f64,
    /// Weight exponent `p` in `w(x) = arctan(C · pers^p)`.
    pub weight_p: f64,
}

impl Default for PwgkConfig {
    fn default() -> Self {
        Self {
            tau: 1.0,
            weight_c: 1.0,
            weight_p: 1.0,
        }
    }
}

// ─── Shared helpers ─────────────────────────────────────────────────────────────

/// Collect the finite off-diagonal points `(birth, death)` of a diagram, dropping
/// any point on (or below) the diagonal (`death ≤ birth`).  Mirrors the extraction
/// used in `persistence_image.rs`.
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

// ─── Persistence scale-space kernel (Reininghaus et al. 2015) ───────────────────

/// Persistence scale-space kernel `k_σ(D, E)` (Reininghaus et al. 2015).
///
/// See the module documentation for the closed form.  Diagonal points contribute
/// nothing by construction.
///
/// # Errors
/// Returns [`TdaError::ParameterOutOfRange`] if `sigma` is not strictly positive and
/// finite.
pub fn persistence_scale_space_kernel(
    d1: &PersistenceDiagram,
    d2: &PersistenceDiagram,
    cfg: &KernelConfig,
) -> TdaResult<f64> {
    if cfg.sigma <= 0.0 || !cfg.sigma.is_finite() {
        return Err(TdaError::ParameterOutOfRange(format!(
            "sigma must be > 0 and finite, got {}",
            cfg.sigma
        )));
    }

    let pts1 = finite_points(d1);
    let pts2 = finite_points(d2);
    let denom = 8.0 * cfg.sigma;
    let prefactor = 1.0 / (8.0 * PI * cfg.sigma);

    let mut acc = 0.0_f64;
    for &p in &pts1 {
        for &q in &pts2 {
            let q_bar = (q.1, q.0); // mirror across the diagonal
            let direct = (-dist_sq(p, q) / denom).exp();
            let mirror = (-dist_sq(p, q_bar) / denom).exp();
            acc += direct - mirror;
        }
    }
    Ok(prefactor * acc)
}

/// Distance induced by the scale-space kernel:
/// `d(D, E) = sqrt(k(D,D) + k(E,E) − 2·k(D,E))`.
///
/// The argument of the square root is clamped at `0` to absorb floating-point noise.
///
/// # Errors
/// Propagates errors from [`persistence_scale_space_kernel`].
pub fn persistence_scale_space_distance(
    d1: &PersistenceDiagram,
    d2: &PersistenceDiagram,
    cfg: &KernelConfig,
) -> TdaResult<f64> {
    let k11 = persistence_scale_space_kernel(d1, d1, cfg)?;
    let k22 = persistence_scale_space_kernel(d2, d2, cfg)?;
    let k12 = persistence_scale_space_kernel(d1, d2, cfg)?;
    let d2_val = k11 + k22 - 2.0 * k12;
    Ok(d2_val.max(0.0).sqrt())
}

// ─── Persistence weighted Gaussian kernel (Kusano et al. 2016) ──────────────────

/// Arctangent persistence weight `w(x) = arctan(C · pers^p)`.
#[inline]
fn pwgk_weight(pers: f64, cfg: &PwgkConfig) -> f64 {
    (cfg.weight_c * pers.powf(cfg.weight_p)).atan()
}

/// Persistence weighted Gaussian kernel `k(D, E)` (Kusano et al. 2016).
///
/// See the module documentation for the closed form.
///
/// # Errors
/// Returns [`TdaError::ParameterOutOfRange`] if `tau` is not strictly positive and
/// finite.
pub fn persistence_weighted_gaussian_kernel(
    d1: &PersistenceDiagram,
    d2: &PersistenceDiagram,
    cfg: &PwgkConfig,
) -> TdaResult<f64> {
    if cfg.tau <= 0.0 || !cfg.tau.is_finite() {
        return Err(TdaError::ParameterOutOfRange(format!(
            "tau must be > 0 and finite, got {}",
            cfg.tau
        )));
    }

    let pts1 = finite_points(d1);
    let pts2 = finite_points(d2);
    let denom = 2.0 * cfg.tau * cfg.tau;

    // Precompute weights (pers = death − birth).
    let w1: Vec<f64> = pts1.iter().map(|&(b, d)| pwgk_weight(d - b, cfg)).collect();
    let w2: Vec<f64> = pts2.iter().map(|&(b, d)| pwgk_weight(d - b, cfg)).collect();

    let mut acc = 0.0_f64;
    for (i, &x) in pts1.iter().enumerate() {
        for (j, &y) in pts2.iter().enumerate() {
            let g = (-dist_sq(x, y) / denom).exp();
            acc += w1[i] * w2[j] * g;
        }
    }
    Ok(acc)
}

/// Distance induced by the persistence weighted Gaussian kernel:
/// `d(D, E) = sqrt(k(D,D) + k(E,E) − 2·k(D,E))` (clamped at `0`).
///
/// # Errors
/// Propagates errors from [`persistence_weighted_gaussian_kernel`].
pub fn persistence_weighted_gaussian_distance(
    d1: &PersistenceDiagram,
    d2: &PersistenceDiagram,
    cfg: &PwgkConfig,
) -> TdaResult<f64> {
    let k11 = persistence_weighted_gaussian_kernel(d1, d1, cfg)?;
    let k22 = persistence_weighted_gaussian_kernel(d2, d2, cfg)?;
    let k12 = persistence_weighted_gaussian_kernel(d1, d2, cfg)?;
    let d2_val = k11 + k22 - 2.0 * k12;
    Ok(d2_val.max(0.0).sqrt())
}

// ─── Sliced Wasserstein kernel (Carrière et al. 2017) ───────────────────────────

/// Sliced Wasserstein kernel `k(D, E) = exp(−SW_p(D, E) / (2η²))` (Carrière 2017).
///
/// `SW_p` is the sliced Wasserstein distance from
/// [`crate::persistence::wasserstein_p::sliced_wasserstein`]; `p`, `n_projections`
/// and `rng` are forwarded to it verbatim.  The kernel bandwidth is `eta > 0`.
///
/// # Errors
/// * [`TdaError::ParameterOutOfRange`] if `eta` is not strictly positive and finite.
/// * Any error raised by the underlying [`sliced_wasserstein`] computation.
pub fn sliced_wasserstein_kernel(
    d1: &PersistenceDiagram,
    d2: &PersistenceDiagram,
    p: f64,
    n_projections: usize,
    rng: &mut LcgRng,
    eta: f64,
) -> TdaResult<f64> {
    if eta <= 0.0 || !eta.is_finite() {
        return Err(TdaError::ParameterOutOfRange(format!(
            "eta must be > 0 and finite, got {eta}"
        )));
    }
    let sw = sliced_wasserstein(d1, d2, p, n_projections, rng)?;
    let denom = 2.0 * eta * eta;
    Ok((-sw / denom).exp())
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::homology::persistent::PersistencePair;

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

    // ── Scale-space kernel ──────────────────────────────────────────────────────

    #[test]
    fn scale_space_self_positive() {
        let d = diag(&[(0.0, 2.0), (1.0, 4.0)]);
        let cfg = KernelConfig::default();
        let k = persistence_scale_space_kernel(&d, &d, &cfg).expect("k");
        assert!(k > 0.0, "self-kernel must be positive, got {k}");
    }

    #[test]
    fn scale_space_symmetric() {
        let a = diag(&[(0.0, 2.0)]);
        let b = diag(&[(0.5, 3.0), (1.0, 4.0)]);
        let cfg = KernelConfig::default();
        let kab = persistence_scale_space_kernel(&a, &b, &cfg).expect("kab");
        let kba = persistence_scale_space_kernel(&b, &a, &cfg).expect("kba");
        assert!((kab - kba).abs() < 1e-12, "kernel must be symmetric");
    }

    #[test]
    fn scale_space_diagonal_only_is_zero() {
        // A pair with birth == death sits on the diagonal and is filtered out.
        let d = diag(&[(1.0, 1.0)]);
        let cfg = KernelConfig::default();
        let k = persistence_scale_space_kernel(&d, &d, &cfg).expect("k");
        assert!(
            k.abs() < 1e-15,
            "diagonal-only diagram must give 0, got {k}"
        );
    }

    #[test]
    fn scale_space_identical_distance_zero() {
        let d = diag(&[(0.0, 2.0), (1.0, 5.0)]);
        let cfg = KernelConfig::default();
        let dist = persistence_scale_space_distance(&d, &d, &cfg).expect("dist");
        assert!(dist < 1e-9, "self-distance must be ~0, got {dist}");
    }

    #[test]
    fn scale_space_distinct_distance_positive() {
        let a = diag(&[(0.0, 2.0)]);
        let b = diag(&[(0.0, 8.0)]);
        let cfg = KernelConfig::default();
        let dist = persistence_scale_space_distance(&a, &b, &cfg).expect("dist");
        assert!(dist > 0.0, "distinct diagrams must have positive distance");
    }

    #[test]
    fn scale_space_sigma_zero_errors() {
        let d = diag(&[(0.0, 2.0)]);
        let cfg = KernelConfig { sigma: 0.0 };
        assert!(persistence_scale_space_kernel(&d, &d, &cfg).is_err());
    }

    #[test]
    fn scale_space_single_point_exact() {
        // D = E = {(0,2)}, σ = 1.
        //   ‖p − q‖² = 0  ⇒ direct = 1.
        //   q̄ = (2,0); ‖(0,2) − (2,0)‖² = 4 + 4 = 8 ⇒ mirror = exp(−8/8) = e^{−1}.
        //   k = (1/(8π)) · (1 − e^{−1}).
        let d = diag(&[(0.0, 2.0)]);
        let cfg = KernelConfig { sigma: 1.0 };
        let k = persistence_scale_space_kernel(&d, &d, &cfg).expect("k");
        let expected = (1.0 - (-1.0_f64).exp()) / (8.0 * PI);
        assert!(
            (k - expected).abs() < 1e-12,
            "k = {k}, expected = {expected}"
        );
    }

    // ── Persistence weighted Gaussian kernel ────────────────────────────────────

    #[test]
    fn pwgk_self_positive() {
        let d = diag(&[(0.0, 2.0), (1.0, 4.0)]);
        let cfg = PwgkConfig::default();
        let k = persistence_weighted_gaussian_kernel(&d, &d, &cfg).expect("k");
        assert!(k > 0.0, "PWGK self-kernel must be positive, got {k}");
    }

    #[test]
    fn pwgk_symmetric() {
        let a = diag(&[(0.0, 3.0)]);
        let b = diag(&[(0.5, 2.0), (1.0, 5.0)]);
        let cfg = PwgkConfig::default();
        let kab = persistence_weighted_gaussian_kernel(&a, &b, &cfg).expect("kab");
        let kba = persistence_weighted_gaussian_kernel(&b, &a, &cfg).expect("kba");
        assert!((kab - kba).abs() < 1e-12, "PWGK must be symmetric");
    }

    #[test]
    fn pwgk_identical_distance_zero() {
        let d = diag(&[(0.0, 2.0), (1.0, 6.0)]);
        let cfg = PwgkConfig::default();
        let dist = persistence_weighted_gaussian_distance(&d, &d, &cfg).expect("dist");
        assert!(dist < 1e-9, "PWGK self-distance must be ~0, got {dist}");
    }

    #[test]
    fn pwgk_tau_zero_errors() {
        let d = diag(&[(0.0, 2.0)]);
        let cfg = PwgkConfig {
            tau: 0.0,
            ..Default::default()
        };
        assert!(persistence_weighted_gaussian_kernel(&d, &d, &cfg).is_err());
    }

    #[test]
    fn pwgk_zero_persistence_is_zero() {
        // birth == death ⇒ pers = 0 ⇒ weight arctan(0) = 0 ⇒ kernel 0.
        let d = diag(&[(2.0, 2.0)]);
        let cfg = PwgkConfig::default();
        let k = persistence_weighted_gaussian_kernel(&d, &d, &cfg).expect("k");
        assert!(
            k.abs() < 1e-15,
            "zero-persistence diagram must give 0, got {k}"
        );
    }

    // ── Sliced Wasserstein kernel ───────────────────────────────────────────────

    #[test]
    fn sliced_wasserstein_kernel_self_is_one() {
        // SW(D, D) = 0 ⇒ exp(0) = 1.
        let d = diag(&[(0.0, 2.0), (1.0, 4.0)]);
        let mut rng = LcgRng::new(11);
        let k = sliced_wasserstein_kernel(&d, &d, 2.0, 50, &mut rng, 1.0).expect("k");
        assert!((k - 1.0).abs() < 1e-9, "SW kernel self must be ~1, got {k}");
    }

    #[test]
    fn sliced_wasserstein_kernel_symmetric() {
        let a = diag(&[(0.0, 4.0)]);
        let b = diag(&[(1.0, 2.0)]);
        // Use fresh RNGs seeded identically so the random projections match.
        let mut rng_ab = LcgRng::new(123);
        let mut rng_ba = LcgRng::new(123);
        let kab = sliced_wasserstein_kernel(&a, &b, 2.0, 200, &mut rng_ab, 1.5).expect("kab");
        let kba = sliced_wasserstein_kernel(&b, &a, 2.0, 200, &mut rng_ba, 1.5).expect("kba");
        assert!((kab - kba).abs() < 1e-9, "SW kernel must be symmetric");
    }

    #[test]
    fn sliced_wasserstein_kernel_eta_zero_errors() {
        let d = diag(&[(0.0, 2.0)]);
        let mut rng = LcgRng::new(1);
        assert!(sliced_wasserstein_kernel(&d, &d, 2.0, 10, &mut rng, 0.0).is_err());
    }
}
