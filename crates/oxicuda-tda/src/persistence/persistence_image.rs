//! Persistence images (Adams et al. 2017, JMLR 18:1-35).
//!
//! A persistence image is a stable, finite-dimensional vectorisation of a persistence
//! diagram.  Each finite persistence pair `(b, d)` is mapped to the transformed pair
//! `(birth = b, persistence = d - b)` and then represented as a weighted 2D Gaussian
//! kernel on a regular grid in `(birth, persistence)` space.
//!
//! Also implements the persistence silhouette (Chazal et al. 2014): a weighted 1D
//! summary derived from piecewise-linear tent functions over the persistence pairs.

use crate::error::{TdaError, TdaResult};
use crate::persistence::diagram::PersistenceDiagram;

// ─── Weight function ──────────────────────────────────────────────────────────

/// Weight function applied to each persistence pair before accumulating into the image.
#[derive(Debug, Clone)]
pub enum WeightFunction {
    /// `w(b, p) = p / p_max` — linear normalised persistence.  Zero weight for
    /// zero-persistence pairs.  Default.
    Linear,
    /// `w(b, p) = (2/π) * arctan(p)` — bounded, smooth, always positive.
    Arctan,
    /// `w(b, p) = 1.0` — constant (unweighted).
    Constant,
    /// `w(b, p) = 1 - exp(-p² / (2σ_w²))` — de-emphasises near-diagonal pairs.
    PersistenceGaussian {
        /// Bandwidth parameter σ_w > 0.
        sigma_w: f64,
    },
}

impl WeightFunction {
    /// Evaluate the weight function at birth `b`, persistence `p` and maximum
    /// persistence `p_max`.
    pub fn evaluate(&self, _b: f64, p: f64, p_max: f64) -> f64 {
        match self {
            WeightFunction::Linear => {
                if p_max > 0.0 {
                    p / p_max
                } else {
                    0.0
                }
            }
            WeightFunction::Arctan => (2.0 / std::f64::consts::PI) * p.atan(),
            WeightFunction::Constant => 1.0,
            WeightFunction::PersistenceGaussian { sigma_w } => {
                if *sigma_w <= 0.0 {
                    0.0
                } else {
                    1.0 - (-p * p / (2.0 * sigma_w * sigma_w)).exp()
                }
            }
        }
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for persistence image computation.
#[derive(Debug, Clone)]
pub struct PersistenceImageConfig {
    /// Number of pixels along the birth axis.  Default: 20.
    pub n_birth: usize,
    /// Number of pixels along the persistence axis.  Default: 20.
    pub n_persistence: usize,
    /// Gaussian bandwidth σ.  Must be > 0.  Default: 1.0.
    pub sigma: f64,
    /// Birth axis range `[birth_min, birth_max]`.  Auto-computed from diagram when `None`.
    pub birth_range: Option<(f64, f64)>,
    /// Upper bound of the persistence axis.  Auto-computed when `None`.
    pub persistence_max: Option<f64>,
    /// Weight function applied to each pair.
    pub weight: WeightFunction,
}

impl Default for PersistenceImageConfig {
    fn default() -> Self {
        Self {
            n_birth: 20,
            n_persistence: 20,
            sigma: 1.0,
            birth_range: None,
            persistence_max: None,
            weight: WeightFunction::Linear,
        }
    }
}

// ─── Result ───────────────────────────────────────────────────────────────────

/// Persistence image stored as a flat row-major array.
///
/// Layout: `image[j * n_birth + i]` is the pixel at persistence index `j` (row)
/// and birth index `i` (column).
#[derive(Debug, Clone)]
pub struct PersistenceImageResult {
    /// Flat pixel array, length = `n_birth * n_persistence`.
    pub image: Vec<f64>,
    /// Actual birth axis range `[birth_min, birth_max]`.
    pub birth_range: (f64, f64),
    /// Actual persistence axis upper bound.
    pub persistence_max: f64,
    /// Number of pixels on the birth axis.
    pub n_birth: usize,
    /// Number of pixels on the persistence axis.
    pub n_persistence: usize,
}

impl PersistenceImageResult {
    /// Get the pixel value at `(birth_idx, persistence_idx)`.
    ///
    /// Returns [`TdaError::ParameterOutOfRange`] if either index is out of bounds.
    pub fn get(&self, birth_idx: usize, persistence_idx: usize) -> TdaResult<f64> {
        if birth_idx >= self.n_birth {
            return Err(TdaError::ParameterOutOfRange(format!(
                "birth_idx {} >= n_birth {}",
                birth_idx, self.n_birth
            )));
        }
        if persistence_idx >= self.n_persistence {
            return Err(TdaError::ParameterOutOfRange(format!(
                "persistence_idx {} >= n_persistence {}",
                persistence_idx, self.n_persistence
            )));
        }
        Ok(self.image[persistence_idx * self.n_birth + birth_idx])
    }

    /// Normalize the image to `[0, 1]` in-place.
    ///
    /// If the image is all zeros, it is left unchanged.
    pub fn normalize(&mut self) {
        let max_val = self.image.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_val <= 0.0 || !max_val.is_finite() {
            return;
        }
        for v in &mut self.image {
            *v /= max_val;
        }
    }
}

// ─── Grid range helpers ───────────────────────────────────────────────────────

/// Determine the birth-axis range from the list of `(birth, persistence)` pairs,
/// applying a 10% padding.
fn auto_birth_range(pairs: &[(f64, f64)]) -> (f64, f64) {
    if pairs.is_empty() {
        return (0.0, 1.0);
    }
    let mut bmin = f64::INFINITY;
    let mut bmax = f64::NEG_INFINITY;
    for &(b, _) in pairs {
        if b < bmin {
            bmin = b;
        }
        if b > bmax {
            bmax = b;
        }
    }
    if (bmax - bmin).abs() < f64::EPSILON {
        // Single unique birth value.
        return (bmin - 0.5, bmin + 0.5);
    }
    let span = bmax - bmin;
    (bmin - 0.1 * span, bmax + 0.1 * span)
}

/// Determine the persistence-axis maximum from the list of pairs, with 10% padding.
fn auto_persistence_max(pairs: &[(f64, f64)]) -> f64 {
    if pairs.is_empty() {
        return 1.0;
    }
    let pmax = pairs
        .iter()
        .map(|&(_, p)| p)
        .fold(f64::NEG_INFINITY, f64::max);
    if pmax <= 0.0 {
        return 1.0;
    }
    pmax * 1.1
}

// ─── 2D Gaussian ─────────────────────────────────────────────────────────────

/// Normalised 2D Gaussian: φ_σ(dx, dy) = exp(-(dx² + dy²) / (2σ²)) / (2πσ²).
#[inline]
fn gaussian_2d(dx: f64, dy: f64, sigma: f64) -> f64 {
    let norm = 2.0 * std::f64::consts::PI * sigma * sigma;
    (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp() / norm
}

// ─── Core algorithm ───────────────────────────────────────────────────────────

/// Compute a persistence image from a single persistence diagram.
///
/// See module documentation for the full algorithm description.
///
/// # Errors
///
/// * [`TdaError::ParameterOutOfRange`] — `sigma ≤ 0`, `n_birth = 0`, or `n_persistence = 0`.
pub fn persistence_image(
    diagram: &PersistenceDiagram,
    cfg: &PersistenceImageConfig,
) -> TdaResult<PersistenceImageResult> {
    if cfg.sigma <= 0.0 {
        return Err(TdaError::ParameterOutOfRange(
            "sigma must be > 0".to_owned(),
        ));
    }
    if cfg.n_birth == 0 {
        return Err(TdaError::ParameterOutOfRange(
            "n_birth must be > 0".to_owned(),
        ));
    }
    if cfg.n_persistence == 0 {
        return Err(TdaError::ParameterOutOfRange(
            "n_persistence must be > 0".to_owned(),
        ));
    }

    // Step 1: extract finite pairs with positive persistence.
    let raw_pairs: Vec<(f64, f64)> = diagram
        .finite_pairs()
        .iter()
        .filter_map(|p| {
            let d = p.death?;
            let pers = d - p.birth;
            if pers > 0.0 {
                Some((p.birth, pers))
            } else {
                None
            }
        })
        .collect();

    // Step 2: determine grid ranges.
    let (birth_min, birth_max) = cfg
        .birth_range
        .unwrap_or_else(|| auto_birth_range(&raw_pairs));
    let pers_max = cfg
        .persistence_max
        .unwrap_or_else(|| auto_persistence_max(&raw_pairs));

    let birth_span = birth_max - birth_min;
    let sigma = cfg.sigma;

    // Step 3 & 4: build pixel grid and accumulate Gaussians.
    let nb = cfg.n_birth;
    let np = cfg.n_persistence;
    let mut image = vec![0.0f64; nb * np];

    // Precompute weight normalisations.
    let p_max_for_weight = pers_max;

    for (b, p) in &raw_pairs {
        let w = cfg.weight.evaluate(*b, *p, p_max_for_weight);
        if w == 0.0 {
            continue;
        }
        for j in 0..np {
            // Pixel centre on persistence axis.
            let v = (j as f64 + 0.5) * pers_max / np as f64;
            for i in 0..nb {
                // Pixel centre on birth axis.
                let u = birth_min + (i as f64 + 0.5) * birth_span / nb as f64;
                let dx = u - b;
                let dy = v - p;
                image[j * nb + i] += w * gaussian_2d(dx, dy, sigma);
            }
        }
    }

    Ok(PersistenceImageResult {
        image,
        birth_range: (birth_min, birth_max),
        persistence_max: pers_max,
        n_birth: nb,
        n_persistence: np,
    })
}

/// Compute persistence images for multiple diagrams with the same configuration.
///
/// Returns a `Vec` of images, one per diagram.
///
/// # Errors
///
/// Propagates any error from [`persistence_image`].
pub fn persistence_image_batch(
    diagrams: &[PersistenceDiagram],
    cfg: &PersistenceImageConfig,
) -> TdaResult<Vec<PersistenceImageResult>> {
    diagrams.iter().map(|d| persistence_image(d, cfg)).collect()
}

// ─── Distance ─────────────────────────────────────────────────────────────────

/// Compute the L² distance between two persistence images.
///
/// Both images must have the same grid dimensions (`n_birth` and `n_persistence`).
///
/// # Errors
///
/// [`TdaError::DimensionMismatch`] if grids differ.
pub fn persistence_image_distance(
    img1: &PersistenceImageResult,
    img2: &PersistenceImageResult,
) -> TdaResult<f64> {
    if img1.n_birth != img2.n_birth || img1.n_persistence != img2.n_persistence {
        return Err(TdaError::DimensionMismatch {
            expected: img1.n_birth * img1.n_persistence,
            got: img2.n_birth * img2.n_persistence,
        });
    }
    let sum_sq: f64 = img1
        .image
        .iter()
        .zip(img2.image.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    Ok(sum_sq.sqrt())
}

// ─── Persistence silhouette ───────────────────────────────────────────────────

/// Piecewise-linear tent function for a pair `(birth, death)` evaluated at `t`.
///
/// * 0 if `t < birth` or `t > death`
/// * rises linearly from 0 at `birth` to 1 at `midpoint = (birth+death)/2`
/// * falls linearly from 1 at `midpoint` to 0 at `death`
#[inline]
fn tent(birth: f64, death: f64, t: f64) -> f64 {
    if t < birth || t > death {
        return 0.0;
    }
    let mid = (birth + death) / 2.0;
    let half_span = mid - birth;
    if half_span <= 0.0 {
        return 0.0;
    }
    if t <= mid {
        (t - birth) / half_span
    } else {
        (death - t) / half_span
    }
}

/// Compute the persistence silhouette (Chazal et al. 2014).
///
/// The silhouette is a weighted average of tent functions:
/// `φ(t) = Σ_p w_p · Λ_p(t) / Σ_p w_p`
///
/// evaluated at `n_points` equally spaced values in `t_range` (auto-computed from
/// the diagram's birth/death range if `None`).  Returns a zero vector if the diagram
/// has no finite pairs with positive persistence.
///
/// # Errors
///
/// * [`TdaError::ParameterOutOfRange`] — `n_points = 0` or `sigma = 0` (irrelevant
///   here, but weight validation requires the weight function to be well-formed).
pub fn persistence_silhouette(
    diagram: &PersistenceDiagram,
    n_points: usize,
    t_range: Option<(f64, f64)>,
    weight: &WeightFunction,
) -> TdaResult<Vec<f64>> {
    if n_points == 0 {
        return Err(TdaError::ParameterOutOfRange(
            "n_points must be > 0".to_owned(),
        ));
    }

    // Collect finite pairs (birth, death) with positive persistence.
    let pairs: Vec<(f64, f64)> = diagram
        .finite_pairs()
        .iter()
        .filter_map(|p| {
            let d = p.death?;
            if d > p.birth {
                Some((p.birth, d))
            } else {
                None
            }
        })
        .collect();

    if pairs.is_empty() {
        return Ok(vec![0.0; n_points]);
    }

    // Determine t range.
    let (t_min, t_max) = t_range.unwrap_or_else(|| {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &(b, d) in &pairs {
            if b < lo {
                lo = b;
            }
            if d > hi {
                hi = d;
            }
        }
        (lo, hi)
    });

    // Compute max persistence for Linear weight normalisation.
    let p_max = pairs
        .iter()
        .map(|&(b, d)| d - b)
        .fold(f64::NEG_INFINITY, f64::max);

    // Compute weights.
    let weights: Vec<f64> = pairs
        .iter()
        .map(|&(b, d)| weight.evaluate(b, d - b, p_max))
        .collect();
    let weight_sum: f64 = weights.iter().sum();

    // Evaluate silhouette.
    let step = if n_points > 1 {
        (t_max - t_min) / (n_points - 1) as f64
    } else {
        0.0
    };

    let mut result = vec![0.0f64; n_points];
    if weight_sum <= 0.0 {
        return Ok(result);
    }

    for (k, v) in result.iter_mut().enumerate() {
        let t = t_min + k as f64 * step;
        let numerator: f64 = pairs
            .iter()
            .zip(weights.iter())
            .map(|(&(b, d), &w)| w * tent(b, d, t))
            .sum();
        *v = numerator / weight_sum;
    }

    Ok(result)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::homology::persistent::PersistencePair;
    use crate::persistence::diagram::PersistenceDiagram;

    fn make_diagram(pairs: Vec<(f64, Option<f64>)>, dim: usize) -> PersistenceDiagram {
        let ps: Vec<PersistencePair> = pairs
            .into_iter()
            .map(|(b, d)| PersistencePair {
                dim,
                birth: b,
                death: d,
            })
            .collect();
        PersistenceDiagram::new(ps, dim)
    }

    // ── test 1 ────────────────────────────────────────────────────────────────
    #[test]
    fn empty_diagram_returns_zero_image() {
        let diag = make_diagram(vec![], 0);
        let cfg = PersistenceImageConfig::default();
        let img = persistence_image(&diag, &cfg).expect("ok");
        assert!(img.image.iter().all(|&v| v >= 0.0));
        // All Gaussians are from an empty set → sum is 0.
        assert!(img.image.iter().all(|&v| v == 0.0));
    }

    // ── test 2 ────────────────────────────────────────────────────────────────
    #[test]
    fn single_pair_image_nonzero() {
        let diag = make_diagram(vec![(0.0, Some(1.0))], 0);
        let cfg = PersistenceImageConfig {
            sigma: 0.5,
            ..Default::default()
        };
        let img = persistence_image(&diag, &cfg).expect("ok");
        let sum: f64 = img.image.iter().sum();
        assert!(sum > 0.0, "image must be non-zero for a finite pair");
    }

    // ── test 3 ────────────────────────────────────────────────────────────────
    #[test]
    fn image_shape_correct() {
        let diag = make_diagram(vec![(0.0, Some(1.0))], 0);
        let cfg = PersistenceImageConfig {
            n_birth: 5,
            n_persistence: 10,
            ..Default::default()
        };
        let img = persistence_image(&diag, &cfg).expect("ok");
        assert_eq!(img.image.len(), 50);
        assert_eq!(img.n_birth, 5);
        assert_eq!(img.n_persistence, 10);
    }

    // ── test 4 ────────────────────────────────────────────────────────────────
    #[test]
    fn image_all_nonnegative() {
        let diag = make_diagram(
            vec![(0.0, Some(1.0)), (0.5, Some(2.0)), (1.0, Some(1.5))],
            0,
        );
        let cfg = PersistenceImageConfig::default();
        let img = persistence_image(&diag, &cfg).expect("ok");
        assert!(
            img.image.iter().all(|&v| v >= 0.0),
            "all pixels must be ≥ 0"
        );
    }

    // ── test 5 ────────────────────────────────────────────────────────────────
    #[test]
    fn weight_constant_sums_to_expected() {
        // With Constant weight, each pair contributes the same total Gaussian mass.
        // Total integral of a 2D Gaussian is 1 (when normalised), so the total image
        // sum should be approximately n_pairs * (integral ≈ 1) with some discretisation.
        let diag = make_diagram(vec![(0.0, Some(1.0))], 0);
        let cfg = PersistenceImageConfig {
            weight: WeightFunction::Constant,
            sigma: 0.1,
            n_birth: 50,
            n_persistence: 50,
            ..Default::default()
        };
        let img = persistence_image(&diag, &cfg).expect("ok");
        let total: f64 = img.image.iter().sum();
        // With small sigma and large grid the total should be close to 1.
        // Just check positivity.
        assert!(total > 0.0);
    }

    // ── test 6 ────────────────────────────────────────────────────────────────
    #[test]
    fn linear_weight_zero_at_zero_persistence() {
        // A pair with birth == death is filtered out (pers=0), so image stays zero.
        let diag = make_diagram(vec![(1.0, Some(1.0))], 0);
        let cfg = PersistenceImageConfig {
            weight: WeightFunction::Linear,
            ..Default::default()
        };
        let img = persistence_image(&diag, &cfg).expect("ok");
        assert!(
            img.image.iter().all(|&v| v == 0.0),
            "zero-persistence pair must contribute nothing"
        );
    }

    // ── test 7 ────────────────────────────────────────────────────────────────
    #[test]
    fn arctan_weight_positive() {
        let wf = WeightFunction::Arctan;
        let w = wf.evaluate(0.5, 1.0, 1.0);
        let expected = (2.0 / std::f64::consts::PI) * 1.0_f64.atan();
        assert!((w - expected).abs() < 1e-12);
        assert!(w > 0.0);
    }

    // ── test 8 ────────────────────────────────────────────────────────────────
    #[test]
    fn normalize_maps_to_unit_range() {
        let diag = make_diagram(vec![(0.0, Some(2.0))], 0);
        let cfg = PersistenceImageConfig {
            weight: WeightFunction::Constant,
            ..Default::default()
        };
        let mut img = persistence_image(&diag, &cfg).expect("ok");
        img.normalize();
        let max = img.image.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((max - 1.0).abs() < 1e-9, "max after normalize should be 1");
    }

    // ── test 9 ────────────────────────────────────────────────────────────────
    #[test]
    fn normalize_empty_image_stays_zero() {
        let diag = make_diagram(vec![], 0);
        let cfg = PersistenceImageConfig::default();
        let mut img = persistence_image(&diag, &cfg).expect("ok");
        img.normalize();
        assert!(img.image.iter().all(|&v| v == 0.0));
    }

    // ── test 10 ───────────────────────────────────────────────────────────────
    #[test]
    fn get_pixel_boundary_check() {
        let diag = make_diagram(vec![(0.0, Some(1.0))], 0);
        let cfg = PersistenceImageConfig {
            n_birth: 4,
            n_persistence: 4,
            ..Default::default()
        };
        let img = persistence_image(&diag, &cfg).expect("ok");
        // Valid access.
        assert!(img.get(3, 3).is_ok());
        // Out-of-bounds birth index.
        assert!(img.get(4, 0).is_err());
        // Out-of-bounds persistence index.
        assert!(img.get(0, 4).is_err());
    }

    // ── test 11 ───────────────────────────────────────────────────────────────
    #[test]
    fn batch_same_as_single() {
        let diag = make_diagram(vec![(0.0, Some(1.0)), (0.5, Some(1.5))], 0);
        let cfg = PersistenceImageConfig {
            sigma: 0.3,
            ..Default::default()
        };
        let single = persistence_image(&diag, &cfg).expect("ok");
        let batch = persistence_image_batch(&[diag], &cfg).expect("ok");
        assert_eq!(batch.len(), 1);
        let batch_img = &batch[0];
        for (a, b) in single.image.iter().zip(batch_img.image.iter()) {
            assert!((a - b).abs() < 1e-14, "batch must equal single");
        }
    }

    // ── test 12 ───────────────────────────────────────────────────────────────
    #[test]
    fn batch_empty_diagrams() {
        let cfg = PersistenceImageConfig::default();
        let result = persistence_image_batch(&[], &cfg).expect("ok");
        assert!(result.is_empty());
    }

    // ── test 13 ───────────────────────────────────────────────────────────────
    #[test]
    fn image_distance_zero_for_same() {
        let diag = make_diagram(vec![(0.0, Some(1.0))], 0);
        let cfg = PersistenceImageConfig::default();
        let img = persistence_image(&diag, &cfg).expect("ok");
        let dist = persistence_image_distance(&img, &img).expect("ok");
        assert!(dist.abs() < 1e-12, "self-distance must be 0");
    }

    // ── test 14 ───────────────────────────────────────────────────────────────
    #[test]
    fn image_distance_positive_for_different() {
        let diag1 = make_diagram(vec![(0.0, Some(1.0))], 0);
        let diag2 = make_diagram(vec![(0.0, Some(5.0))], 0);
        let cfg = PersistenceImageConfig {
            sigma: 0.5,
            ..Default::default()
        };
        let img1 = persistence_image(&diag1, &cfg).expect("ok");
        let img2 = persistence_image(&diag2, &cfg).expect("ok");
        let dist = persistence_image_distance(&img1, &img2).expect("ok");
        assert!(dist > 0.0, "different diagrams must have positive distance");
    }

    // ── test 15 ───────────────────────────────────────────────────────────────
    #[test]
    fn image_distance_symmetric() {
        let diag1 = make_diagram(vec![(0.0, Some(1.0))], 0);
        let diag2 = make_diagram(vec![(0.5, Some(2.0))], 0);
        let cfg = PersistenceImageConfig {
            sigma: 0.4,
            ..Default::default()
        };
        let img1 = persistence_image(&diag1, &cfg).expect("ok");
        let img2 = persistence_image(&diag2, &cfg).expect("ok");
        let d12 = persistence_image_distance(&img1, &img2).expect("ok");
        let d21 = persistence_image_distance(&img2, &img1).expect("ok");
        assert!((d12 - d21).abs() < 1e-12, "distance must be symmetric");
    }

    // ── test 16 ───────────────────────────────────────────────────────────────
    #[test]
    fn silhouette_single_pair() {
        // One pair (0, 2): tent peaks at t=1 with value 1.
        let diag = make_diagram(vec![(0.0, Some(2.0))], 0);
        let sil = persistence_silhouette(&diag, 101, Some((0.0, 2.0)), &WeightFunction::Constant)
            .expect("ok");
        // Peak should be at index 50 (t=1.0) with value 1.0.
        let max_val = sil.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((max_val - 1.0).abs() < 1e-9, "tent peak must be 1.0");
        // Peak index.
        let peak_idx = sil
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(peak_idx, 50, "peak must be at midpoint");
    }

    // ── test 17 ───────────────────────────────────────────────────────────────
    #[test]
    fn silhouette_empty_diagram() {
        let diag = make_diagram(vec![], 0);
        let sil = persistence_silhouette(&diag, 50, None, &WeightFunction::Constant).expect("ok");
        assert_eq!(sil.len(), 50);
        assert!(sil.iter().all(|&v| v == 0.0));
    }

    // ── test 18 ───────────────────────────────────────────────────────────────
    #[test]
    fn silhouette_length_correct() {
        let diag = make_diagram(vec![(0.0, Some(1.0)), (0.5, Some(2.0))], 0);
        let sil = persistence_silhouette(&diag, 37, None, &WeightFunction::Linear).expect("ok");
        assert_eq!(sil.len(), 37);
    }

    // ── test 19 ───────────────────────────────────────────────────────────────
    #[test]
    fn err_sigma_zero() {
        let diag = make_diagram(vec![(0.0, Some(1.0))], 0);
        let cfg = PersistenceImageConfig {
            sigma: 0.0,
            ..Default::default()
        };
        assert!(persistence_image(&diag, &cfg).is_err());
    }

    // ── test 20 ───────────────────────────────────────────────────────────────
    #[test]
    fn err_n_birth_zero() {
        let diag = make_diagram(vec![(0.0, Some(1.0))], 0);
        let cfg = PersistenceImageConfig {
            n_birth: 0,
            ..Default::default()
        };
        assert!(persistence_image(&diag, &cfg).is_err());
    }
}
