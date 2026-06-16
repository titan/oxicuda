//! Random projection (Johnson-Lindenstrauss; Achlioptas 2003).
//!
//! Random projection reduces dimensionality by multiplying the data by a random
//! matrix `R ∈ ℝ^{dim × k}`:
//! ```text
//! Y = X · R ,   X ∈ ℝ^{n × dim},   Y ∈ ℝ^{n × k}.
//! ```
//! The **Johnson-Lindenstrauss lemma** guarantees that, with high probability, a
//! suitably scaled random `R` approximately preserves all pairwise Euclidean
//! distances: for any `0 < ε < 1`, choosing
//! ```text
//! k ≥ 4 · ln(n) / (ε²/2 − ε³/3)
//! ```
//! ensures `(1 − ε)‖xᵢ − xⱼ‖² ≤ ‖yᵢ − yⱼ‖² ≤ (1 + ε)‖xᵢ − xⱼ‖²` for all pairs.
//!
//! Two projection families are provided:
//! - [`RandomProjectionKind::Gaussian`] — entries are i.i.d. `N(0, 1/k)`, the
//!   classic dense JL transform.
//! - [`RandomProjectionKind::Sparse`] — Achlioptas' sparse `{-1, 0, +1}`
//!   construction with density `1/s`: each entry is `±√(s/k)` with probability
//!   `1/(2s)` each and `0` otherwise. With `s = 3` (the original proposal) two
//!   thirds of the matrix is zero, giving a 3× speed-up; `s = √dim` (Li et al.
//!   2006) yields very sparse projections.
//!
//! All randomness comes from the reproducible workspace [`LcgRng`], so a fixed
//! seed yields a fixed projection.
//!
//! References:
//! - W. B. Johnson & J. Lindenstrauss, "Extensions of Lipschitz mappings into a
//!   Hilbert space", 1984.
//! - D. Achlioptas, "Database-friendly random projections", JCSS, 2003.

use crate::error::{ManifoldError, ManifoldResult};
use crate::handle::LcgRng;

/// Family of random projection matrix to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RandomProjectionKind {
    /// Dense Gaussian matrix with entries `~ N(0, 1/k)`.
    Gaussian,
    /// Sparse Achlioptas `{-1, 0, +1}` matrix; `density = 1/s`.
    Sparse,
}

/// Hyper-parameters for [`random_projection`].
#[derive(Debug, Clone)]
pub struct RandomProjectionConfig {
    /// Target dimensionality `k`.
    pub n_components: usize,
    /// Projection family.
    pub kind: RandomProjectionKind,
    /// Sparsity parameter `s` for [`RandomProjectionKind::Sparse`] (`density = 1/s`,
    /// `s ≥ 1`). Ignored for the Gaussian kind.
    pub sparse_s: f64,
    /// RNG seed (reproducible projections).
    pub seed: u64,
}

impl Default for RandomProjectionConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            kind: RandomProjectionKind::Gaussian,
            sparse_s: 3.0,
            seed: 0x9E_37_79_B9,
        }
    }
}

/// Smallest JL embedding dimension that preserves pairwise distances within a
/// factor of `1 ± eps` for `n` samples, per the Johnson-Lindenstrauss bound
/// ```text
/// k_min = ⌈ 4 ln(n) / (eps²/2 − eps³/3) ⌉.
/// ```
///
/// # Errors
/// [`ManifoldError::InvalidParameter`] when `n < 1` or `eps ∉ (0, 1)`.
pub fn johnson_lindenstrauss_min_dim(n: usize, eps: f64) -> ManifoldResult<usize> {
    if n < 1 {
        return Err(ManifoldError::InvalidParameter {
            name: "n".into(),
            reason: "must be >= 1".into(),
        });
    }
    if !(eps > 0.0 && eps < 1.0) {
        return Err(ManifoldError::InvalidParameter {
            name: "eps".into(),
            reason: format!("must be in (0, 1), got {eps}"),
        });
    }
    let denom = eps * eps / 2.0 - eps * eps * eps / 3.0;
    let k = (4.0 * (n as f64).ln() / denom).ceil();
    Ok(k.max(1.0) as usize)
}

/// Project `n` points of dimensionality `dim` into `cfg.n_components` dimensions
/// using a random Johnson-Lindenstrauss transform.
///
/// `data` is a row-major `[n × dim]` slice. The result is row-major
/// `[n × n_components]`.
///
/// # Errors
/// - [`ManifoldError::EmptyInput`] when `n == 0` or `dim == 0`.
/// - [`ManifoldError::ShapeMismatch`] when `data.len() != n * dim`.
/// - [`ManifoldError::InvalidParameter`] when `n_components == 0`, or when
///   `sparse_s < 1` for a sparse projection.
pub fn random_projection(
    data: &[f64],
    n: usize,
    dim: usize,
    cfg: &RandomProjectionConfig,
) -> ManifoldResult<Vec<f64>> {
    if n == 0 || dim == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if data.len() != n * dim {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, dim],
            got: vec![data.len()],
        });
    }
    let k = cfg.n_components;
    if k == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: "must be >= 1".into(),
        });
    }
    if cfg.kind == RandomProjectionKind::Sparse && cfg.sparse_s < 1.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "sparse_s".into(),
            reason: format!("must be >= 1, got {}", cfg.sparse_s),
        });
    }

    let r = build_matrix(dim, k, cfg);

    // Y = X · R, row-major.
    let mut y = vec![0.0; n * k];
    for i in 0..n {
        let x_row = &data[i * dim..(i + 1) * dim];
        for (d, &x) in x_row.iter().enumerate() {
            if x == 0.0 {
                continue;
            }
            let r_row = &r[d * k..(d + 1) * k];
            let y_row = &mut y[i * k..(i + 1) * k];
            for (yc, &rc) in y_row.iter_mut().zip(r_row.iter()) {
                *yc += x * rc;
            }
        }
    }
    Ok(y)
}

/// Construct the `[dim × k]` (row-major) projection matrix.
fn build_matrix(dim: usize, k: usize, cfg: &RandomProjectionConfig) -> Vec<f64> {
    let mut rng = LcgRng::new(cfg.seed);
    let mut r = vec![0.0; dim * k];
    match cfg.kind {
        RandomProjectionKind::Gaussian => {
            let scale = 1.0 / (k as f64).sqrt();
            for v in &mut r {
                *v = rng.next_normal() * scale;
            }
        }
        RandomProjectionKind::Sparse => {
            // Achlioptas: value ±√(s/k) with prob 1/(2s) each, else 0.
            let s = cfg.sparse_s;
            let value = (s / k as f64).sqrt();
            let p = 1.0 / (2.0 * s); // probability of +value (and of −value)
            for v in &mut r {
                let u = rng.next_f64();
                if u < p {
                    *v = value;
                } else if u < 2.0 * p {
                    *v = -value;
                } else {
                    *v = 0.0;
                }
            }
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> RandomProjectionConfig {
        RandomProjectionConfig::default()
    }

    /// `n` points in `dim` dimensions drawn from a reproducible RNG.
    fn random_data(n: usize, dim: usize, seed: u64) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        (0..n * dim).map(|_| rng.next_normal()).collect()
    }

    // 1. Output shape is [n × n_components].
    #[test]
    fn output_shape() {
        let n = 20;
        let dim = 50;
        let data = random_data(n, dim, 1);
        let c = RandomProjectionConfig {
            n_components: 8,
            ..cfg()
        };
        let y = random_projection(&data, n, dim, &c).expect("random_projection should succeed");
        assert_eq!(y.len(), n * 8);
    }

    // 2. All projected coordinates are finite.
    #[test]
    fn finite() {
        let data = random_data(15, 40, 2);
        let y = random_projection(&data, 15, 40, &cfg()).expect("value should be present");
        for &v in &y {
            assert!(v.is_finite(), "non-finite value {v}");
        }
    }

    // 3. Empty input errors.
    #[test]
    fn empty_input_error() {
        let err = random_projection(&[], 0, 0, &cfg());
        assert!(matches!(err, Err(ManifoldError::EmptyInput)), "got {err:?}");
    }

    // 4. Shape mismatch errors.
    #[test]
    fn shape_mismatch_error() {
        let err = random_projection(&[1.0, 2.0], 2, 3, &cfg());
        assert!(
            matches!(err, Err(ManifoldError::ShapeMismatch { .. })),
            "got {err:?}"
        );
    }

    // 5. n_components == 0 errors.
    #[test]
    fn n_components_0_error() {
        let data = random_data(5, 10, 3);
        let c = RandomProjectionConfig {
            n_components: 0,
            ..cfg()
        };
        let err = random_projection(&data, 5, 10, &c);
        assert!(
            matches!(err, Err(ManifoldError::InvalidParameter { .. })),
            "got {err:?}"
        );
    }

    // 6. Deterministic for a fixed seed.
    #[test]
    fn deterministic() {
        let data = random_data(12, 30, 4);
        let a = random_projection(&data, 12, 30, &cfg()).expect("value should be present");
        let b = random_projection(&data, 12, 30, &cfg()).expect("value should be present");
        assert_eq!(a, b);
    }

    // 7. Different seeds give different projections.
    #[test]
    fn seed_changes_projection() {
        let data = random_data(10, 25, 5);
        let a = random_projection(&data, 10, 25, &cfg()).expect("value should be present");
        let c2 = RandomProjectionConfig {
            seed: cfg().seed ^ 0xFFFF,
            ..cfg()
        };
        let b = random_projection(&data, 10, 25, &c2).expect("random_projection should succeed");
        let differ = a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-9);
        assert!(differ, "projections with different seeds should differ");
    }

    // 8. Gaussian JL approximately preserves pairwise distances on average.
    #[test]
    fn gaussian_preserves_distances() {
        let n = 40;
        let dim = 200;
        let data = random_data(n, dim, 6);
        let c = RandomProjectionConfig {
            n_components: 80,
            kind: RandomProjectionKind::Gaussian,
            ..cfg()
        };
        let y = random_projection(&data, n, dim, &c).expect("random_projection should succeed");
        let dist = |buf: &[f64], d: usize, i: usize, j: usize| -> f64 {
            let mut s = 0.0;
            for k in 0..d {
                let diff = buf[i * d + k] - buf[j * d + k];
                s += diff * diff;
            }
            s.sqrt()
        };
        let mut ratio_sum = 0.0;
        let mut count = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                let orig = dist(&data, dim, i, j);
                let proj = dist(&y, 80, i, j);
                if orig > 1e-9 {
                    ratio_sum += proj / orig;
                    count += 1;
                }
            }
        }
        let mean_ratio = ratio_sum / count as f64;
        // Mean preserved-distance ratio should be near 1 (JL isometry on average).
        assert!((mean_ratio - 1.0).abs() < 0.25, "mean ratio {mean_ratio}");
    }

    // 9. Sparse projection runs and yields finite output.
    #[test]
    fn sparse_projection_finite() {
        let data = random_data(15, 60, 7);
        let c = RandomProjectionConfig {
            n_components: 20,
            kind: RandomProjectionKind::Sparse,
            sparse_s: 3.0,
            ..cfg()
        };
        let y = random_projection(&data, 15, 60, &c).expect("random_projection should succeed");
        assert_eq!(y.len(), 15 * 20);
        for &v in &y {
            assert!(v.is_finite(), "non-finite sparse value {v}");
        }
    }

    // 10. Invalid sparse density errors.
    #[test]
    fn sparse_s_too_small_error() {
        let data = random_data(5, 10, 8);
        let c = RandomProjectionConfig {
            kind: RandomProjectionKind::Sparse,
            sparse_s: 0.5,
            ..cfg()
        };
        let err = random_projection(&data, 5, 10, &c);
        assert!(
            matches!(err, Err(ManifoldError::InvalidParameter { .. })),
            "got {err:?}"
        );
    }

    // 11. JL minimum-dimension helper is monotone and validates its arguments.
    #[test]
    fn jl_min_dim() {
        let k_small_eps = johnson_lindenstrauss_min_dim(1000, 0.1)
            .expect("johnson_lindenstrauss_min_dim should succeed");
        let k_large_eps = johnson_lindenstrauss_min_dim(1000, 0.5)
            .expect("johnson_lindenstrauss_min_dim should succeed");
        assert!(k_small_eps > k_large_eps, "{k_small_eps} <= {k_large_eps}");
        assert!(johnson_lindenstrauss_min_dim(0, 0.1).is_err());
        assert!(johnson_lindenstrauss_min_dim(100, 1.5).is_err());
    }
}
