//! SMOTE: Synthetic Minority Over-sampling Technique (Chawla et al. 2002, JAIR).
//!
//! Also implements Borderline-SMOTE (Han et al. 2005).
//!
//! Algorithm (Basic SMOTE):
//! 1. Extract minority class M = {xᵢ : yᵢ = minority_class}.
//! 2. Compute k-NN for each minority sample within M (brute-force L2 distance).
//! 3. Compute n_synthetic:
//!    - target_min = round(oversampling_ratio * n_total)
//!    - n_synthetic = max(0, target_min - |M|)
//! 4. For each synthetic sample: pick seed xᵢ (cycled), pick random neighbor xⱼ,
//!    - x_new = xᵢ + λ · (xⱼ - xᵢ),  λ ∈ [0, 1) uniform.
//!
//! Algorithm (Borderline-SMOTE):
//!   1. Compute k-NN for each minority sample within the FULL dataset.
//!   2. Count m' = majority-class neighbors.
//!   3. NOISE: m' == k → skip.  SAFE: m' < k/2 → skip.  DANGER: k/2 ≤ m' < k → keep.
//!   4. Generate synthetics only from DANGER samples; neighbors drawn within full M.

use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Which SMOTE variant to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmoteVariant {
    /// Standard SMOTE (Chawla 2002).
    Basic,
    /// Borderline-SMOTE (Han 2005): only DANGER minority samples are used as seeds.
    Borderline,
}

/// Configuration for SMOTE / Borderline-SMOTE.
#[derive(Debug, Clone)]
pub struct SmoteConfig {
    /// Number of nearest neighbors to consider (≥ 1).
    pub k_neighbors: usize,
    /// Target minority fraction of total dataset.
    /// `n_synthetic = max(0, round(oversampling_ratio * n_total) - n_minority)`.
    pub oversampling_ratio: f64,
    /// SMOTE variant.
    pub variant: SmoteVariant,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for SmoteConfig {
    fn default() -> Self {
        Self {
            k_neighbors: 5,
            oversampling_ratio: 0.5,
            variant: SmoteVariant::Basic,
            seed: 0,
        }
    }
}

// ─── Result ──────────────────────────────────────────────────────────────────

/// Output of a SMOTE run.
#[derive(Debug, Clone)]
pub struct SmoteResult {
    /// Synthetic samples in row-major layout, shape `[n_synthetic, n_features]`.
    pub synthetic_x: Vec<f64>,
    /// Number of synthetic samples generated.
    pub n_synthetic: usize,
    /// Number of features per sample.
    pub n_features: usize,
    /// Index (within the original minority subset) of the seed sample for each synthetic.
    pub minority_seed_indices: Vec<usize>,
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Compute squared L2 distance between two rows of equal length.
#[inline]
fn squared_l2(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let d = ai - bi;
            d * d
        })
        .sum()
}

/// Brute-force k-NN search: for `query` (slice of length `n_features`),
/// return indices of `k` nearest samples from `pool` (excluding `exclude_idx`
/// if `Some`).
///
/// `pool` is row-major, shape `[pool_n, n_features]`.
fn knn_indices(
    query: &[f64],
    pool: &[f64],
    pool_n: usize,
    n_features: usize,
    k: usize,
    exclude_idx: Option<usize>,
) -> Vec<usize> {
    // Gather (distance, idx) pairs
    let mut dists: Vec<(f64, usize)> = (0..pool_n)
        .filter(|&i| exclude_idx != Some(i))
        .map(|i| {
            let row = &pool[i * n_features..(i + 1) * n_features];
            (squared_l2(query, row), i)
        })
        .collect();

    // Partial sort: bring k smallest to the front
    let k_actual = k.min(dists.len());
    dists.select_nth_unstable_by(k_actual.saturating_sub(1), |a, b| {
        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
    });
    dists[..k_actual].iter().map(|&(_, idx)| idx).collect()
}

/// Partial Fisher-Yates: draw one uniform element from `0..n`.
#[inline]
fn rand_usize(rng: &mut LcgRng, n: usize) -> usize {
    rng.next_usize(n)
}

// ─── Core algorithm ──────────────────────────────────────────────────────────

/// Run SMOTE or Borderline-SMOTE.
///
/// - `x`: row-major feature matrix, shape `[n, n_features]`.
/// - `y`: class labels (usize), length `n`.
/// - `n`: number of samples.
/// - `n_features`: number of features per sample.
/// - `minority_class`: label of the minority class to oversample.
/// - `config`: algorithm configuration.
pub fn smote(
    x: &[f64],
    y: &[usize],
    n: usize,
    n_features: usize,
    minority_class: usize,
    config: &SmoteConfig,
) -> TabularResult<SmoteResult> {
    // ── Validation ────────────────────────────────────────────────────────────
    if n < 2 {
        return Err(TabularError::InsufficientSamples { need: 2, got: n });
    }
    if n_features == 0 {
        return Err(TabularError::InvalidParameter {
            name: "n_features".to_string(),
            msg: "must be >= 1".to_string(),
        });
    }
    if x.len() != n * n_features {
        return Err(TabularError::DimensionMismatch {
            expected: n * n_features,
            got: x.len(),
        });
    }
    if config.oversampling_ratio <= 0.0 || config.oversampling_ratio > 1.0 {
        return Err(TabularError::InvalidParameter {
            name: "oversampling_ratio".to_string(),
            msg: "must be in (0, 1]".to_string(),
        });
    }
    if config.k_neighbors == 0 {
        return Err(TabularError::InvalidParameter {
            name: "k_neighbors".to_string(),
            msg: "must be >= 1".to_string(),
        });
    }

    // ── Extract minority samples ──────────────────────────────────────────────
    let minority_orig_indices: Vec<usize> = (0..n).filter(|&i| y[i] == minority_class).collect();
    let n_min = minority_orig_indices.len();

    // If minority class has 0 samples, return empty result (no error)
    if n_min == 0 {
        return Ok(SmoteResult {
            synthetic_x: Vec::new(),
            n_synthetic: 0,
            n_features,
            minority_seed_indices: Vec::new(),
        });
    }

    // Build dense minority matrix
    let mut minority_x = vec![0.0_f64; n_min * n_features];
    for (new_i, &orig_i) in minority_orig_indices.iter().enumerate() {
        minority_x[new_i * n_features..(new_i + 1) * n_features]
            .copy_from_slice(&x[orig_i * n_features..(orig_i + 1) * n_features]);
    }

    // ── Compute n_synthetic ───────────────────────────────────────────────────
    let target_min = (config.oversampling_ratio * n as f64).round() as usize;
    let n_synthetic = target_min.saturating_sub(n_min);

    if n_synthetic == 0 {
        return Ok(SmoteResult {
            synthetic_x: Vec::new(),
            n_synthetic: 0,
            n_features,
            minority_seed_indices: Vec::new(),
        });
    }

    // ── Validate k_neighbors ─────────────────────────────────────────────────
    // We need at least k neighbors within minority set (excluding self → n_min-1 candidates)
    if n_min.saturating_sub(1) < config.k_neighbors {
        return Err(TabularError::InsufficientSamples {
            need: config.k_neighbors + 1,
            got: n_min,
        });
    }

    let k = config.k_neighbors;
    let mut rng = LcgRng::new(config.seed);

    // ── Determine seed samples based on variant ───────────────────────────────
    let seed_indices: Vec<usize> = match config.variant {
        SmoteVariant::Basic => (0..n_min).collect(),
        SmoteVariant::Borderline => {
            // For each minority sample, find k-NN in the FULL dataset
            // then count majority-class neighbors.
            (0..n_min)
                .filter(|&mi| {
                    let orig_i = minority_orig_indices[mi];
                    let query = &x[orig_i * n_features..(orig_i + 1) * n_features];
                    // k-NN in full x excluding self
                    let neighbors = knn_indices(query, x, n, n_features, k, Some(orig_i));
                    let m_prime = neighbors
                        .iter()
                        .filter(|&&ni| y[ni] != minority_class)
                        .count();
                    // DANGER: k/2 ≤ m' < k
                    let half_k = k as f64 / 2.0;
                    m_prime >= half_k.ceil() as usize && m_prime < k
                })
                .collect()
        }
    };

    // If no seed samples (e.g., all NOISE/SAFE in Borderline), generate zero synthetics
    if seed_indices.is_empty() {
        return Ok(SmoteResult {
            synthetic_x: Vec::new(),
            n_synthetic: 0,
            n_features,
            minority_seed_indices: Vec::new(),
        });
    }

    // ── Pre-compute k-NN within minority for each seed ────────────────────────
    // neighbors_per_seed[i] = list of minority indices (neighbors of seed_indices[i])
    let neighbors_per_seed: Vec<Vec<usize>> = seed_indices
        .iter()
        .map(|&mi| {
            let query = &minority_x[mi * n_features..(mi + 1) * n_features];
            knn_indices(query, &minority_x, n_min, n_features, k, Some(mi))
        })
        .collect();

    // ── Generate synthetic samples ─────────────────────────────────────────────
    let mut synthetic_x = vec![0.0_f64; n_synthetic * n_features];
    let mut minority_seed_indices = Vec::with_capacity(n_synthetic);
    let n_seeds = seed_indices.len();

    for s in 0..n_synthetic {
        // Cycle through seeds
        let seed_pos = s % n_seeds;
        let mi = seed_indices[seed_pos];
        minority_seed_indices.push(mi);

        let seed_row = &minority_x[mi * n_features..(mi + 1) * n_features];

        // Pick random neighbor from precomputed list
        let nbrs = &neighbors_per_seed[seed_pos];
        let nbr_mi = nbrs[rand_usize(&mut rng, nbrs.len())];
        let nbr_row = &minority_x[nbr_mi * n_features..(nbr_mi + 1) * n_features];

        // Interpolate: x_new = seed + λ * (neighbor - seed), λ ∈ [0, 1)
        let lambda = rng.next_f32() as f64;
        let out_row = &mut synthetic_x[s * n_features..(s + 1) * n_features];
        for f in 0..n_features {
            out_row[f] = seed_row[f] + lambda * (nbr_row[f] - seed_row[f]);
        }
    }

    Ok(SmoteResult {
        synthetic_x,
        n_synthetic,
        n_features,
        minority_seed_indices,
    })
}

/// Augment the original dataset with the SMOTE-generated synthetic samples.
///
/// Returns `(augmented_x, augmented_y)` where the synthetic samples are
/// appended after the original `n` samples.
///
/// - `x`: original feature matrix, shape `[n, n_features]`.
/// - `y`: original labels, length `n`.
/// - `n`: number of original samples.
/// - `result`: output of `smote(...)`.
/// - `minority_class`: label assigned to all synthetic samples.
pub fn smote_augment(
    x: &[f64],
    y: &[usize],
    n: usize,
    result: &SmoteResult,
    minority_class: usize,
) -> (Vec<f64>, Vec<usize>) {
    let nf = result.n_features;
    let ns = result.n_synthetic;

    let mut aug_x = Vec::with_capacity((n + ns) * nf);
    aug_x.extend_from_slice(x);
    aug_x.extend_from_slice(&result.synthetic_x);

    let mut aug_y = Vec::with_capacity(n + ns);
    aug_y.extend_from_slice(y);
    aug_y.extend(std::iter::repeat_n(minority_class, ns));

    (aug_x, aug_y)
}

/// Compute the fraction of samples belonging to each class.
///
/// Returns a vector of length `n_classes` where `out[c] = count(y == c) / n`.
/// If `n_classes == 0` or `y` is empty, returns an empty vector.
#[must_use]
pub fn class_imbalance_ratio(y: &[usize], n_classes: usize) -> Vec<f64> {
    if n_classes == 0 || y.is_empty() {
        return Vec::new();
    }
    let n = y.len() as f64;
    let mut counts = vec![0usize; n_classes];
    for &label in y {
        if label < n_classes {
            counts[label] += 1;
        }
    }
    counts.iter().map(|&c| c as f64 / n).collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Build a simple imbalanced dataset: `n_minority` samples of class 0,
    /// `n_majority` samples of class 1.  Features are 1-D for simplicity.
    fn make_imbalanced_1d(n_minority: usize, n_majority: usize) -> (Vec<f64>, Vec<usize>) {
        let n = n_minority + n_majority;
        let mut x = vec![0.0_f64; n];
        let mut y = vec![0usize; n];
        for i in 0..n_minority {
            x[i] = i as f64 * 0.1;
            y[i] = 0;
        }
        for i in 0..n_majority {
            x[n_minority + i] = (i as f64 + 10.0) * 0.1;
            y[n_minority + i] = 1;
        }
        (x, y)
    }

    /// Build a 2-D imbalanced dataset.
    fn make_imbalanced_2d(n_minority: usize, n_majority: usize) -> (Vec<f64>, Vec<usize>) {
        let n = n_minority + n_majority;
        let mut x = vec![0.0_f64; n * 2];
        let mut y = vec![0usize; n];
        for i in 0..n_minority {
            x[i * 2] = i as f64 * 0.1;
            x[i * 2 + 1] = i as f64 * 0.05;
            y[i] = 0;
        }
        for i in 0..n_majority {
            x[(n_minority + i) * 2] = (i as f64 + 20.0) * 0.1;
            x[(n_minority + i) * 2 + 1] = 5.0;
            y[n_minority + i] = 1;
        }
        (x, y)
    }

    // ── Test 1: Interpolation correctness ────────────────────────────────────
    #[test]
    fn test_interpolation_within_bounds() {
        let n_min = 10;
        let n_maj = 30;
        let (x, y) = make_imbalanced_1d(n_min, n_maj);
        let n = n_min + n_maj;
        let cfg = SmoteConfig {
            k_neighbors: 3,
            oversampling_ratio: 0.4,
            variant: SmoteVariant::Basic,
            seed: 42,
        };
        let result = smote(&x, &y, n, 1, 0, &cfg).expect("smote should succeed");
        // For each synthetic sample, its value must lie within [min(seed,nbr), max(seed,nbr)]
        // Since we don't track neighbor pairs here, just verify values are in overall minority range
        let min_val = x[..n_min].iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = x[..n_min].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        for &sv in &result.synthetic_x {
            assert!(
                sv >= min_val - 1e-10 && sv <= max_val + 1e-10,
                "synthetic={sv} out of [{min_val}, {max_val}]"
            );
        }
    }

    // ── Test 2: synthetic_x.len() == n_synthetic * n_features ─────────────
    #[test]
    fn test_synthetic_x_len() {
        let n_min = 10;
        let n_maj = 30;
        let (x, y) = make_imbalanced_1d(n_min, n_maj);
        let n = n_min + n_maj;
        let cfg = SmoteConfig {
            oversampling_ratio: 0.4,
            ..Default::default()
        };
        let result = smote(&x, &y, n, 1, 0, &cfg).expect("smote should succeed");
        assert_eq!(
            result.synthetic_x.len(),
            result.n_synthetic * result.n_features
        );
    }

    // ── Test 3: minority count after augment ──────────────────────────────────
    #[test]
    fn test_augment_minority_count() {
        let n_min = 10;
        let n_maj = 30;
        let (x, y) = make_imbalanced_1d(n_min, n_maj);
        let n = n_min + n_maj;
        let cfg = SmoteConfig {
            oversampling_ratio: 0.4,
            ..Default::default()
        };
        let result = smote(&x, &y, n, 1, 0, &cfg).expect("smote should succeed");
        let (_, aug_y) = smote_augment(&x, &y, n, &result, 0);
        let new_minority_count = aug_y.iter().filter(|&&l| l == 0).count();
        assert_eq!(
            new_minority_count,
            n_min + result.n_synthetic,
            "minority count mismatch"
        );
    }

    // ── Test 4: class_imbalance_ratio sums to 1.0 ────────────────────────────
    #[test]
    fn test_class_imbalance_ratio_sums_to_one() {
        let y = vec![0usize, 0, 0, 1, 1, 1, 1, 1];
        let ratios = class_imbalance_ratio(&y, 2);
        let total: f64 = ratios.iter().sum();
        assert!((total - 1.0).abs() < 1e-10, "sum={total}");
    }

    // ── Test 5: k=1 works ────────────────────────────────────────────────────
    #[test]
    fn test_k1_works() {
        let n_min = 5;
        let n_maj = 15;
        let (x, y) = make_imbalanced_1d(n_min, n_maj);
        let n = n_min + n_maj;
        let cfg = SmoteConfig {
            k_neighbors: 1,
            oversampling_ratio: 0.4,
            ..Default::default()
        };
        let result = smote(&x, &y, n, 1, 0, &cfg);
        assert!(result.is_ok(), "k=1 should work: {:?}", result.err());
    }

    // ── Test 6: Borderline variant runs without error ─────────────────────────
    #[test]
    fn test_borderline_no_error() {
        let n_min = 10;
        let n_maj = 20;
        let (x, y) = make_imbalanced_1d(n_min, n_maj);
        let n = n_min + n_maj;
        let cfg = SmoteConfig {
            k_neighbors: 3,
            oversampling_ratio: 0.4,
            variant: SmoteVariant::Borderline,
            seed: 7,
        };
        let result = smote(&x, &y, n, 1, 0, &cfg);
        assert!(
            result.is_ok(),
            "Borderline should succeed: {:?}",
            result.err()
        );
    }

    // ── Test 7: Borderline n_synthetic ≤ Basic n_synthetic ────────────────────
    #[test]
    fn test_borderline_fewer_or_equal() {
        let n_min = 10;
        let n_maj = 20;
        let (x, y) = make_imbalanced_1d(n_min, n_maj);
        let n = n_min + n_maj;
        let base_cfg = SmoteConfig {
            k_neighbors: 3,
            oversampling_ratio: 0.4,
            variant: SmoteVariant::Basic,
            seed: 7,
        };
        let border_cfg = SmoteConfig {
            variant: SmoteVariant::Borderline,
            ..base_cfg.clone()
        };
        let basic = smote(&x, &y, n, 1, 0, &base_cfg).expect("smote should succeed");
        let borderline = smote(&x, &y, n, 1, 0, &border_cfg).expect("smote should succeed");
        assert!(
            borderline.n_synthetic <= basic.n_synthetic,
            "borderline {} > basic {}",
            borderline.n_synthetic,
            basic.n_synthetic
        );
    }

    // ── Test 8: already balanced → n_synthetic = 0 ───────────────────────────
    #[test]
    fn test_already_balanced_zero_synthetic() {
        // 10 minority, 10 majority, oversampling_ratio=0.4 →
        // target_min = round(0.4 * 20) = 8 < n_min=10 → n_synthetic=0
        let n_min = 10;
        let n_maj = 10;
        let (x, y) = make_imbalanced_1d(n_min, n_maj);
        let n = n_min + n_maj;
        let cfg = SmoteConfig {
            oversampling_ratio: 0.4,
            ..Default::default()
        };
        let result = smote(&x, &y, n, 1, 0, &cfg).expect("smote should succeed");
        assert_eq!(
            result.n_synthetic, 0,
            "should generate zero synthetics when already balanced"
        );
    }

    // ── Test 9: determinism (same seed) ──────────────────────────────────────
    #[test]
    fn test_determinism_same_seed() {
        let n_min = 10;
        let n_maj = 30;
        let (x, y) = make_imbalanced_1d(n_min, n_maj);
        let n = n_min + n_maj;
        let cfg = SmoteConfig {
            oversampling_ratio: 0.4,
            seed: 123,
            ..Default::default()
        };
        let r1 = smote(&x, &y, n, 1, 0, &cfg).expect("smote should succeed");
        let r2 = smote(&x, &y, n, 1, 0, &cfg).expect("smote should succeed");
        assert_eq!(
            r1.synthetic_x, r2.synthetic_x,
            "same seed must give identical results"
        );
    }

    // ── Test 10: different seeds → different results ──────────────────────────
    #[test]
    fn test_different_seeds_different_results() {
        let n_min = 12;
        let n_maj = 30;
        let (x, y) = make_imbalanced_1d(n_min, n_maj);
        let n = n_min + n_maj;
        let cfg_a = SmoteConfig {
            oversampling_ratio: 0.45,
            seed: 1,
            ..Default::default()
        };
        let cfg_b = SmoteConfig {
            oversampling_ratio: 0.45,
            seed: 2,
            ..Default::default()
        };
        let r_a = smote(&x, &y, n, 1, 0, &cfg_a).expect("smote should succeed");
        let r_b = smote(&x, &y, n, 1, 0, &cfg_b).expect("smote should succeed");
        if r_a.n_synthetic > 0 && r_b.n_synthetic > 0 {
            assert_ne!(
                r_a.synthetic_x, r_b.synthetic_x,
                "different seeds should give different results"
            );
        }
    }

    // ── Test 11: n_features=1 ────────────────────────────────────────────────
    #[test]
    fn test_n_features_one() {
        let n_min = 8;
        let n_maj = 20;
        let (x, y) = make_imbalanced_1d(n_min, n_maj);
        let n = n_min + n_maj;
        let cfg = SmoteConfig {
            oversampling_ratio: 0.4,
            ..Default::default()
        };
        let result = smote(&x, &y, n, 1, 0, &cfg);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be present").n_features, 1);
    }

    // ── Test 12: n_features=5 ────────────────────────────────────────────────
    #[test]
    fn test_n_features_five() {
        let n_min = 10;
        let n_maj = 30;
        let n_feat = 5;
        let n = n_min + n_maj;
        let x: Vec<f64> = (0..n * n_feat).map(|i| i as f64 * 0.01).collect();
        let y: Vec<usize> = (0..n).map(|i| if i < n_min { 0 } else { 1 }).collect();
        let cfg = SmoteConfig {
            oversampling_ratio: 0.4,
            ..Default::default()
        };
        let result = smote(&x, &y, n, n_feat, 0, &cfg);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be present").n_features, n_feat);
    }

    // ── Test 13: smote_augment total rows ────────────────────────────────────
    #[test]
    fn test_augment_total_rows() {
        let n_min = 10;
        let n_maj = 30;
        let (x, y) = make_imbalanced_1d(n_min, n_maj);
        let n = n_min + n_maj;
        let cfg = SmoteConfig {
            oversampling_ratio: 0.4,
            ..Default::default()
        };
        let result = smote(&x, &y, n, 1, 0, &cfg).expect("smote should succeed");
        let ns = result.n_synthetic;
        let (aug_x, aug_y) = smote_augment(&x, &y, n, &result, 0);
        assert_eq!(aug_x.len(), n + ns, "aug_x length mismatch");
        assert_eq!(aug_y.len(), n + ns, "aug_y length mismatch");
    }

    // ── Test 14: smote_augment minority labels correct ────────────────────────
    #[test]
    fn test_augment_minority_labels() {
        let n_min = 10;
        let n_maj = 30;
        let (x, y) = make_imbalanced_1d(n_min, n_maj);
        let n = n_min + n_maj;
        let cfg = SmoteConfig {
            oversampling_ratio: 0.4,
            ..Default::default()
        };
        let result = smote(&x, &y, n, 1, 0, &cfg).expect("smote should succeed");
        let ns = result.n_synthetic;
        let (_, aug_y) = smote_augment(&x, &y, n, &result, 0);
        // All added labels (last ns entries) must be minority_class = 0
        let added_labels = &aug_y[n..];
        assert_eq!(added_labels.len(), ns);
        assert!(
            added_labels.iter().all(|&l| l == 0),
            "added labels must all be minority_class"
        );
    }

    // ── Test 15: n_synthetic >= 0 ────────────────────────────────────────────
    #[test]
    fn test_n_synthetic_non_negative() {
        let n_min = 10;
        let n_maj = 30;
        let (x, y) = make_imbalanced_2d(n_min, n_maj);
        let n = n_min + n_maj;
        let cfg = SmoteConfig {
            oversampling_ratio: 0.4,
            ..Default::default()
        };
        let result = smote(&x, &y, n, 2, 0, &cfg).expect("smote should succeed");
        // n_synthetic is usize so always ≥ 0; but verify it matches the vec
        assert_eq!(result.synthetic_x.len(), result.n_synthetic * 2);
    }

    // ── Test 16: minority_seed_indices.len() == n_synthetic ──────────────────
    #[test]
    fn test_seed_indices_len() {
        let n_min = 10;
        let n_maj = 30;
        let (x, y) = make_imbalanced_1d(n_min, n_maj);
        let n = n_min + n_maj;
        let cfg = SmoteConfig {
            oversampling_ratio: 0.4,
            ..Default::default()
        };
        let result = smote(&x, &y, n, 1, 0, &cfg).expect("smote should succeed");
        assert_eq!(result.minority_seed_indices.len(), result.n_synthetic);
    }

    // ── Test 17: n < 2 → InsufficientSamples ─────────────────────────────────
    #[test]
    fn test_n_less_than_2_error() {
        let x = vec![1.0_f64];
        let y = vec![0usize];
        let cfg = SmoteConfig::default();
        let result = smote(&x, &y, 1, 1, 0, &cfg);
        assert!(matches!(
            result,
            Err(TabularError::InsufficientSamples { need: 2, got: 1 })
        ));
    }

    // ── Test 18: minority count < k → InsufficientSamples ────────────────────
    #[test]
    fn test_minority_too_small_for_k() {
        // 2 minority samples, k=5 → need k+1=6 but only have 2
        let n_min = 2;
        let n_maj = 20;
        let (x, y) = make_imbalanced_1d(n_min, n_maj);
        let n = n_min + n_maj;
        let cfg = SmoteConfig {
            k_neighbors: 5,
            oversampling_ratio: 0.4,
            ..Default::default()
        };
        let result = smote(&x, &y, n, 1, 0, &cfg);
        assert!(
            matches!(result, Err(TabularError::InsufficientSamples { .. })),
            "expected InsufficientSamples, got {:?}",
            result
        );
    }

    // ── Test 19: oversampling_ratio=0.0 → InvalidParameter ───────────────────
    #[test]
    fn test_oversampling_ratio_zero_error() {
        let n_min = 5;
        let n_maj = 15;
        let (x, y) = make_imbalanced_1d(n_min, n_maj);
        let n = n_min + n_maj;
        let cfg = SmoteConfig {
            oversampling_ratio: 0.0,
            ..Default::default()
        };
        let result = smote(&x, &y, n, 1, 0, &cfg);
        assert!(matches!(result, Err(TabularError::InvalidParameter { .. })));
    }

    // ── Test 20: k_neighbors=0 → InvalidParameter ────────────────────────────
    #[test]
    fn test_k_neighbors_zero_error() {
        let n_min = 5;
        let n_maj = 15;
        let (x, y) = make_imbalanced_1d(n_min, n_maj);
        let n = n_min + n_maj;
        let cfg = SmoteConfig {
            k_neighbors: 0,
            oversampling_ratio: 0.4,
            ..Default::default()
        };
        let result = smote(&x, &y, n, 1, 0, &cfg);
        assert!(matches!(result, Err(TabularError::InvalidParameter { .. })));
    }

    // ── Test 21: x.len() != n * n_features → DimensionMismatch ──────────────
    #[test]
    fn test_dimension_mismatch_error() {
        let x = vec![1.0_f64; 10]; // 10 elements but n=3, n_features=4 → expects 12
        let y = vec![0usize, 1, 0];
        let cfg = SmoteConfig::default();
        let result = smote(&x, &y, 3, 4, 0, &cfg);
        assert!(matches!(
            result,
            Err(TabularError::DimensionMismatch { .. })
        ));
    }

    // ── Test 22: minority_class with 0 samples → n_synthetic=0, no error ─────
    #[test]
    fn test_absent_minority_class_no_error() {
        // Class 2 does not appear in y
        let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let y = vec![0usize, 0, 1, 1, 1];
        let cfg = SmoteConfig {
            oversampling_ratio: 0.4,
            ..Default::default()
        };
        let result = smote(&x, &y, 5, 1, 2, &cfg);
        assert!(result.is_ok(), "absent minority class should not error");
        let result = result.expect("result should be present");
        assert_eq!(result.n_synthetic, 0);
    }
}
