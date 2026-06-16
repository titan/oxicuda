//! RS-Hash — Sathe & Aggarwal, ICDM 2016.
//!
//! *"Subspace Outlier Detection in Linear Time with Randomized Hashing."*
//!
//! RS-Hash scores points by how *crowded* the randomized grid cells they fall
//! into are.  An ensemble of independent hash components is built; each
//! component is the triple
//!
//! ```text
//!   (random subspace V  ×  random grid resolution f  ×  random shift α)
//! ```
//!
//! * **Random grid resolution** `f` — sampled uniformly in `[1/√s, 1 − 1/√s]`,
//!   where `s` is the (sub-sampled) training set size.  After min–max
//!   normalisation each selected feature is gridded into cells of width `f`.
//! * **Random subspace** `V` — a set of `r` features drawn without replacement,
//!   with the locality parameter `r` itself sampled uniformly in
//!   `[1 + ⌊½·log₍₁/f₎(s)⌋ , 1 + ⌊log₍₁/f₎(s)⌋]` (clamped to `[1, d]`).  This is
//!   the parameter range of the original paper / the reference PyOD
//!   implementation: it keeps the expected per-cell occupancy meaningful.
//! * **Random shift** `α` — one offset per selected feature, uniform in
//!   `[0, f)`, which randomises bin boundaries.
//!
//! A point hashes to a cell by `cellⱼ = ⌊ (x̃ⱼ + αⱼ) / f ⌋` over the selected
//! features `j ∈ V` (`x̃` is the normalised coordinate).  During `fit` every
//! training point's cell count is recorded per component.
//!
//! # Score
//!
//! The per-component score of a query is `log(count + 1)` of its cell — a
//! *crowdedness* measure (rare cell ⇒ low log-count ⇒ anomalous).  The reported
//! anomaly score is the **negated ensemble average** so that, consistent with
//! the rest of the crate, **higher ⇒ more anomalous**:
//!
//! ```text
//!   anomaly(x) = − (1/m) Σ_components log( count_component(cell(x)) + 1 )
//! ```
//!
//! The raw, paper-oriented crowdedness `(1/m) Σ log(count+1)` (lower ⇒ more
//! anomalous) is available via [`RsHash::log_count_score`].

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

// ─── Single hash component ──────────────────────────────────────────────────────

/// One randomized hashing component: a subspace, a resolution, a shift, and the
/// fitted cell-occupancy table.
#[derive(Debug, Clone)]
pub struct HashComponent {
    /// Selected feature indices (the random subspace `V`, size `r`).
    pub features: Vec<usize>,
    /// Grid resolution `f ∈ (0, 1)` (cell width in normalised space).
    pub f: f32,
    /// Per-selected-feature random shift `α ∈ [0, f)` (length `r`).
    pub shift: Vec<f32>,
    /// Cell → count association list, keyed by the per-feature bin indices.
    counts: Vec<(Vec<i64>, u64)>,
}

impl HashComponent {
    /// Compute the grid cell key of a *normalised* point over this component's
    /// subspace.
    fn cell(&self, x_norm: &[f32]) -> Vec<i64> {
        self.features
            .iter()
            .enumerate()
            .map(|(idx, &j)| {
                let v = x_norm[j] + self.shift[idx];
                (v / self.f).floor() as i64
            })
            .collect()
    }

    /// Add a normalised training point to this component's occupancy table.
    fn add(&mut self, x_norm: &[f32]) {
        let key = self.cell(x_norm);
        for (k, c) in self.counts.iter_mut() {
            if *k == key {
                *c += 1;
                return;
            }
        }
        self.counts.push((key, 1));
    }

    /// Occupancy count of the cell containing `x_norm` (0 if empty).
    fn count(&self, x_norm: &[f32]) -> u64 {
        let key = self.cell(x_norm);
        for (k, c) in &self.counts {
            if *k == key {
                return *c;
            }
        }
        0
    }
}

// ─── Configuration ──────────────────────────────────────────────────────────────

/// Hyper-parameters for [`RsHash`].
#[derive(Debug, Clone)]
pub struct RsHashConfig {
    /// Number of hash components in the ensemble `m` (default `100`).
    pub n_components: usize,
    /// Maximum sub-sample size `s` used to set `f`/`r` ranges (default `1000`).
    pub sample_size: usize,
    /// RNG seed (default `42`).
    pub seed: u64,
}

impl Default for RsHashConfig {
    fn default() -> Self {
        Self {
            n_components: 100,
            sample_size: 1000,
            seed: 42,
        }
    }
}

// ─── RsHash detector ─────────────────────────────────────────────────────────────

/// RS-Hash randomized-subspace-hashing anomaly detector.
///
/// # Usage
///
/// ```rust,ignore
/// let mut det = RsHash::new(RsHashConfig::default());
/// det.fit(&train, n_samples, n_features)?;
/// let score = det.score(&query)?;   // higher ⇒ more anomalous
/// ```
pub struct RsHash {
    config: RsHashConfig,
    components: Vec<HashComponent>,
    /// Per-feature training minimum (for min–max normalisation).
    feat_min: Vec<f32>,
    /// Per-feature training range `max − min` (0 ⇒ constant feature).
    feat_range: Vec<f32>,
    n_features: usize,
    fitted: bool,
}

impl RsHash {
    /// Create an unfitted detector from `config`.
    #[must_use]
    pub fn new(config: RsHashConfig) -> Self {
        Self {
            config,
            components: Vec::new(),
            feat_min: Vec::new(),
            feat_range: Vec::new(),
            n_features: 0,
            fitted: false,
        }
    }

    /// Min–max normalise a raw point into `[0, 1]^d` using fitted statistics.
    fn normalize(&self, x: &[f32]) -> Vec<f32> {
        x.iter()
            .enumerate()
            .map(|(j, &v)| {
                if self.feat_range[j] > 0.0 {
                    (v - self.feat_min[j]) / self.feat_range[j]
                } else {
                    0.0 // constant feature maps to a single coordinate
                }
            })
            .collect()
    }

    /// Sample the locality parameter `r` (subspace size) in the documented
    /// range `[1 + ⌊½·log₍₁/f₎(s)⌋ , 1 + ⌊log₍₁/f₎(s)⌋]`, clamped to `[1, d]`.
    fn sample_subspace_size(f: f32, s: usize, d: usize, rng: &mut LcgRng) -> usize {
        // log base (1/f) of s  ==  ln(s) / ln(1/f).
        let inv_f_ln = (1.0_f32 / f).ln().max(1e-6);
        let log_f_s = (s as f32).max(2.0).ln() / inv_f_ln;
        let low = (1.0 + 0.5 * log_f_s).floor().max(1.0) as usize;
        let high = (1.0 + log_f_s).floor().max(1.0) as usize;
        let low = low.min(d).max(1);
        let high = high.min(d).max(low);
        if high == low {
            low
        } else {
            low + rng.next_usize(high - low + 1)
        }
    }

    /// Fit on `data` (row-major `[n_samples × n_features]`).
    ///
    /// Min–max statistics are computed, the ensemble of components is sampled,
    /// and every training point is hashed into each component's grid.
    pub fn fit(&mut self, data: &[f32], n_samples: usize, n_features: usize) -> AnomalyResult<()> {
        if n_samples == 0 {
            return Err(AnomalyError::EmptyInput);
        }
        if n_features == 0 {
            return Err(AnomalyError::InvalidFeatureCount { n: 0 });
        }
        if data.len() != n_samples * n_features {
            return Err(AnomalyError::DimensionMismatch {
                expected: n_samples * n_features,
                got: data.len(),
            });
        }
        if self.config.n_components == 0 {
            return Err(AnomalyError::Internal {
                msg: "n_components must be > 0".into(),
            });
        }

        // ── Per-feature min / max for normalisation ────────────────────────────
        let mut feat_min = vec![f32::INFINITY; n_features];
        let mut feat_max = vec![f32::NEG_INFINITY; n_features];
        for i in 0..n_samples {
            for j in 0..n_features {
                let v = data[i * n_features + j];
                if v < feat_min[j] {
                    feat_min[j] = v;
                }
                if v > feat_max[j] {
                    feat_max[j] = v;
                }
            }
        }
        let feat_range: Vec<f32> = feat_min
            .iter()
            .zip(feat_max.iter())
            .map(|(lo, hi)| hi - lo)
            .collect();

        self.feat_min = feat_min;
        self.feat_range = feat_range;
        self.n_features = n_features;

        // Effective sample size used for f/r ranges.
        let s = self.config.sample_size.min(n_samples).max(2);
        let inv_sqrt_s = 1.0_f32 / (s as f32).sqrt();

        let mut rng = LcgRng::new(self.config.seed);
        let mut components = Vec::with_capacity(self.config.n_components);

        for _ in 0..self.config.n_components {
            // Grid resolution f ∈ [1/√s, 1 − 1/√s].
            let f = inv_sqrt_s + rng.next_f32() * (1.0 - 2.0 * inv_sqrt_s);
            let f = f.clamp(inv_sqrt_s, 1.0 - inv_sqrt_s).max(1e-4);

            // Subspace size r, then the subspace itself (without replacement).
            let r = Self::sample_subspace_size(f, s, n_features, &mut rng);
            let features = sample_without_replacement(n_features, r, &mut rng);

            // Per-feature random shift α ∈ [0, f).
            let shift: Vec<f32> = (0..features.len()).map(|_| rng.next_f32() * f).collect();

            let mut comp = HashComponent {
                features,
                f,
                shift,
                counts: Vec::new(),
            };

            // Hash every training point into this component.
            for i in 0..n_samples {
                let row = &data[i * n_features..(i + 1) * n_features];
                let norm = self.normalize(row);
                comp.add(&norm);
            }
            components.push(comp);
        }

        self.components = components;
        self.fitted = true;
        Ok(())
    }

    /// Raw crowdedness score `(1/m) Σ log(count + 1)` (paper orientation).
    ///
    /// **Lower ⇒ more anomalous** (rare cells have small counts).  The query is
    /// treated as out-of-sample (no self-subtraction).
    pub fn log_count_score(&self, x: &[f32]) -> AnomalyResult<f32> {
        self.log_count_internal(x, false)
    }

    /// In-sample variant of [`RsHash::log_count_score`]: subtracts 1 from every
    /// cell count (use when `x` is a training point).
    pub fn log_count_score_in_sample(&self, x: &[f32]) -> AnomalyResult<f32> {
        self.log_count_internal(x, true)
    }

    fn log_count_internal(&self, x: &[f32], query_self: bool) -> AnomalyResult<f32> {
        if !self.fitted {
            return Err(AnomalyError::NotFitted);
        }
        if x.len() != self.n_features {
            return Err(AnomalyError::FeatureCountMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }
        let norm = self.normalize(x);
        let mut sum = 0.0_f32;
        for comp in &self.components {
            let c = comp.count(&norm);
            let c = if query_self { c.saturating_sub(1) } else { c };
            sum += (c as f32 + 1.0).ln();
        }
        Ok(sum / self.components.len() as f32)
    }

    /// Anomaly score: `− log_count_score`, so **higher ⇒ more anomalous**
    /// (consistent with the rest of the crate).
    pub fn score(&self, x: &[f32]) -> AnomalyResult<f32> {
        Ok(-self.log_count_score(x)?)
    }

    /// In-sample anomaly score (`x` is a training point).
    pub fn score_in_sample(&self, x: &[f32]) -> AnomalyResult<f32> {
        Ok(-self.log_count_score_in_sample(x)?)
    }

    /// Batch anomaly scoring; `x` is row-major `[n × n_features]`; returns `[n]`.
    pub fn score_batch(&self, x: &[f32], n: usize) -> AnomalyResult<Vec<f32>> {
        if !self.fitted {
            return Err(AnomalyError::NotFitted);
        }
        if x.len() != n * self.n_features {
            return Err(AnomalyError::DimensionMismatch {
                expected: n * self.n_features,
                got: x.len(),
            });
        }
        let mut scores = Vec::with_capacity(n);
        for i in 0..n {
            let row = &x[i * self.n_features..(i + 1) * self.n_features];
            scores.push(self.score(row)?);
        }
        Ok(scores)
    }

    /// Read-only access to the fitted hash components (for inspection/testing).
    #[must_use]
    #[inline]
    pub fn components(&self) -> &[HashComponent] {
        &self.components
    }

    /// Number of features the detector was fitted on (0 if unfitted).
    #[must_use]
    #[inline]
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────────

/// Partial Fisher–Yates: sample `k` distinct indices from `0..n`.
fn sample_without_replacement(n: usize, k: usize, rng: &mut LcgRng) -> Vec<usize> {
    let k = k.min(n);
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + rng.next_usize(n - i);
        indices.swap(i, j);
    }
    indices.truncate(k);
    indices
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense cluster around the origin (stddev 0.2) plus a far outlier row at
    /// the end.  Returns `(data, n)`.
    fn cluster_with_outlier(n_inliers: usize, seed: u64) -> (Vec<f32>, usize) {
        let mut rng = LcgRng::new(seed);
        let mut data = Vec::with_capacity((n_inliers + 1) * 2);
        for _ in 0..n_inliers {
            data.push(rng.next_normal() * 0.2);
            data.push(rng.next_normal() * 0.2);
        }
        data.push(10.0);
        data.push(10.0);
        (data, n_inliers + 1)
    }

    // ── (a) Points in rare cells get HIGHER anomaly scores than dense cells ───
    #[test]
    fn rare_cell_scores_higher() {
        let (data, n) = cluster_with_outlier(300, 7);
        let mut det = RsHash::new(RsHashConfig {
            n_components: 100,
            sample_size: 300,
            seed: 11,
        });
        det.fit(&data, n, 2).expect("fit");

        let dense = [0.0_f32, 0.0]; // crowded centre
        let rare = [10.0_f32, 10.0]; // the outlier's rare cell

        let s_dense = det.score(&dense).expect("dense score");
        let s_rare = det.score(&rare).expect("rare score");
        assert!(
            s_rare > s_dense,
            "rare-cell score {s_rare} should exceed dense-cell score {s_dense}"
        );
    }

    // ── (d) A point in a dense cluster scores low ─────────────────────────────
    #[test]
    fn dense_cluster_point_scores_low() {
        let (data, n) = cluster_with_outlier(400, 23);
        let mut det = RsHash::new(RsHashConfig {
            n_components: 80,
            sample_size: 400,
            seed: 5,
        });
        det.fit(&data, n, 2).expect("fit");

        // A dense centre point's anomaly score must be below the outlier's.
        let centre = det.score(&[0.0_f32, 0.0]).expect("centre");
        let outlier = det.score(&[10.0_f32, 10.0]).expect("outlier");
        assert!(
            centre < outlier,
            "dense centre score {centre} should be below outlier {outlier}"
        );
        // And its raw crowdedness (log-count) should be comparatively large.
        let centre_lc = det.log_count_score(&[0.0_f32, 0.0]).expect("lc");
        let outlier_lc = det.log_count_score(&[10.0_f32, 10.0]).expect("lc");
        assert!(
            centre_lc > outlier_lc,
            "dense centre log-count {centre_lc} should exceed outlier {outlier_lc}"
        );
    }

    // ── (c) Grid hashing is consistent (same point → same cell per component) ─
    #[test]
    fn grid_hashing_consistent() {
        let mut rng = LcgRng::new(99);
        // Build a component over a 3-D subspace by hand.
        let comp = HashComponent {
            features: vec![0, 2],
            f: 0.25,
            shift: vec![0.1, 0.05],
            counts: Vec::new(),
        };
        let x = [0.37_f32, 999.0, 0.62]; // feature 1 is ignored (not in subspace)
        let c1 = comp.cell(&x);
        let c2 = comp.cell(&x);
        assert_eq!(c1, c2, "same point must map to the same cell");
        // Manually verify: cell0 = floor((0.37+0.1)/0.25)=floor(1.88)=1,
        //                  cell1 = floor((0.62+0.05)/0.25)=floor(2.68)=2.
        assert_eq!(c1, vec![1, 2], "cell key mismatch: {c1:?}");
        let _ = &mut rng; // silence unused
    }

    // ── (b) The ensemble averages over random subspaces and resolutions ───────
    #[test]
    fn ensemble_diverse_subspaces_and_resolutions() {
        // With enough features and components there should be a variety of both
        // subspace selections and resolutions f across the ensemble.
        let n = 200_usize;
        let d = 8_usize;
        let mut rng = LcgRng::new(321);
        let data: Vec<f32> = (0..n * d).map(|_| rng.next_f32()).collect();

        let mut det = RsHash::new(RsHashConfig {
            n_components: 60,
            sample_size: 200,
            seed: 4,
        });
        det.fit(&data, n, d).expect("fit");

        let comps = det.components();
        assert_eq!(comps.len(), 60);

        // Distinct resolutions f.
        let mut distinct_f = 0usize;
        for (i, a) in comps.iter().enumerate() {
            if comps[..i].iter().all(|b| (b.f - a.f).abs() > 1e-6) {
                distinct_f += 1;
            }
        }
        assert!(
            distinct_f > 5,
            "expected many distinct resolutions, got {distinct_f}"
        );

        // Distinct subspaces (feature sets).
        let mut distinct_sub = 0usize;
        for (i, a) in comps.iter().enumerate() {
            if comps[..i].iter().all(|b| b.features != a.features) {
                distinct_sub += 1;
            }
        }
        assert!(
            distinct_sub > 5,
            "expected many distinct subspaces, got {distinct_sub}"
        );

        // The ensemble average actually combines components (score uses all m).
        let s = det.score(&vec![0.5_f32; d]).expect("score");
        assert!(s.is_finite(), "ensemble score must be finite");
    }

    // ── (e) Subspace dimensionality is sampled in the documented range ────────
    #[test]
    fn subspace_size_in_documented_range() {
        let d = 20_usize;
        let s = 500_usize;
        let mut rng = LcgRng::new(77);
        for _ in 0..2000 {
            // Resolution f in the valid band.
            let inv_sqrt_s = 1.0 / (s as f32).sqrt();
            let f = (inv_sqrt_s + rng.next_f32() * (1.0 - 2.0 * inv_sqrt_s))
                .clamp(inv_sqrt_s, 1.0 - inv_sqrt_s)
                .max(1e-4);
            let r = RsHash::sample_subspace_size(f, s, d, &mut rng);

            // Re-derive the documented bounds and assert membership.
            let inv_f_ln = (1.0_f32 / f).ln().max(1e-6);
            let log_f_s = (s as f32).max(2.0).ln() / inv_f_ln;
            let low = ((1.0 + 0.5 * log_f_s).floor().max(1.0) as usize)
                .min(d)
                .max(1);
            let high = ((1.0 + log_f_s).floor().max(1.0) as usize).min(d).max(low);

            assert!(
                r >= low && r <= high,
                "subspace size {r} outside documented range [{low},{high}] (f={f})"
            );
            assert!(r >= 1 && r <= d, "subspace size {r} outside [1,{d}]");
        }
    }

    // ── (f) Log-count scoring is monotone (denser cell ⇒ lower score) ─────────
    #[test]
    fn log_count_monotone_in_density() {
        // Construct a single component and populate two cells with different
        // occupancies; the denser cell must yield a lower anomaly score.
        let mut comp = HashComponent {
            features: vec![0],
            f: 0.5,
            shift: vec![0.0],
            counts: Vec::new(),
        };
        // Cell A around x=0.1 (will get many points); cell B around x=0.9 (few).
        for _ in 0..50 {
            comp.add(&[0.1_f32]);
        }
        for _ in 0..2 {
            comp.add(&[0.9_f32]);
        }
        let dense_count = comp.count(&[0.1_f32]);
        let sparse_count = comp.count(&[0.9_f32]);
        assert_eq!(dense_count, 50, "dense cell count");
        assert_eq!(sparse_count, 2, "sparse cell count");

        // log(count+1) monotone: denser ⇒ larger log-count ⇒ lower anomaly.
        let dense_lc = (dense_count as f32 + 1.0).ln();
        let sparse_lc = (sparse_count as f32 + 1.0).ln();
        assert!(
            dense_lc > sparse_lc,
            "denser cell log-count {dense_lc} should exceed sparser {sparse_lc}"
        );

        // Wire through the full detector to confirm orientation end-to-end.
        let n = 52_usize;
        let mut data = vec![0.1_f32; 50];
        data.push(0.9_f32);
        data.push(0.9_f32);
        let mut det = RsHash::new(RsHashConfig {
            n_components: 40,
            sample_size: 52,
            seed: 13,
        });
        det.fit(&data, n, 1).expect("fit");
        let dense_score = det.score(&[0.1_f32]).expect("dense");
        let sparse_score = det.score(&[0.9_f32]).expect("sparse");
        assert!(
            sparse_score > dense_score,
            "sparser point score {sparse_score} should exceed denser {dense_score}"
        );
    }

    // ── Determinism: same seed ⇒ identical scores ─────────────────────────────
    #[test]
    fn deterministic_same_seed() {
        let (data, n) = cluster_with_outlier(80, 1);
        let cfg = RsHashConfig {
            n_components: 50,
            sample_size: 80,
            seed: 7,
        };
        let mut a = RsHash::new(cfg.clone());
        let mut b = RsHash::new(cfg);
        a.fit(&data, n, 2).expect("fit a");
        b.fit(&data, n, 2).expect("fit b");
        let sa = a.score(&[0.0_f32, 0.0]).expect("a");
        let sb = b.score(&[0.0_f32, 0.0]).expect("b");
        assert!((sa - sb).abs() < 1e-6, "scores differ: {sa} vs {sb}");
    }

    // ── Error handling ────────────────────────────────────────────────────────
    #[test]
    fn unfitted_score_errors() {
        let det = RsHash::new(RsHashConfig::default());
        match det.score(&[0.0_f32, 0.0]) {
            Err(AnomalyError::NotFitted) => {}
            other => panic!("expected NotFitted, got {other:?}"),
        }
    }

    #[test]
    fn empty_input_errors() {
        let mut det = RsHash::new(RsHashConfig::default());
        match det.fit(&[], 0, 2) {
            Err(AnomalyError::EmptyInput) => {}
            other => panic!("expected EmptyInput, got {other:?}"),
        }
    }

    #[test]
    fn feature_count_mismatch_on_score() {
        let (data, n) = cluster_with_outlier(20, 2);
        let mut det = RsHash::new(RsHashConfig {
            n_components: 10,
            sample_size: 20,
            seed: 3,
        });
        det.fit(&data, n, 2).expect("fit");
        match det.score(&[0.0_f32, 0.0, 0.0]) {
            Err(AnomalyError::FeatureCountMismatch {
                expected: 2,
                got: 3,
            }) => {}
            other => panic!("expected FeatureCountMismatch, got {other:?}"),
        }
    }

    #[test]
    fn dimension_mismatch_on_fit() {
        let mut det = RsHash::new(RsHashConfig::default());
        match det.fit(&[1.0_f32, 2.0, 3.0], 2, 2) {
            Err(AnomalyError::DimensionMismatch {
                expected: 4,
                got: 3,
            }) => {}
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn zero_components_errors() {
        let mut det = RsHash::new(RsHashConfig {
            n_components: 0,
            ..RsHashConfig::default()
        });
        match det.fit(&[0.1_f32, 0.2, 0.3, 0.4], 2, 2) {
            Err(AnomalyError::Internal { .. }) => {}
            other => panic!("expected Internal error, got {other:?}"),
        }
    }

    #[test]
    fn score_batch_length() {
        let (data, n) = cluster_with_outlier(50, 6);
        let mut det = RsHash::new(RsHashConfig {
            n_components: 30,
            sample_size: 50,
            seed: 4,
        });
        det.fit(&data, n, 2).expect("fit");
        let queries = vec![0.0_f32, 0.0, 1.0, 1.0, 10.0, 10.0];
        let scores = det.score_batch(&queries, 3).expect("batch");
        assert_eq!(scores.len(), 3);
        assert!(scores.iter().all(|s| s.is_finite()), "all finite");
    }

    #[test]
    fn constant_feature_no_panic() {
        // One constant feature should not panic during normalisation/hashing.
        let n = 30_usize;
        let mut data = Vec::with_capacity(n * 2);
        for i in 0..n {
            data.push(2.5_f32); // constant feature 0
            data.push(i as f32 * 0.1); // varying feature 1
        }
        let mut det = RsHash::new(RsHashConfig {
            n_components: 20,
            sample_size: 30,
            seed: 9,
        });
        det.fit(&data, n, 2).expect("fit");
        let s = det.score(&[2.5_f32, 1.0]).expect("score");
        assert!(
            s.is_finite(),
            "score on constant feature must be finite, got {s}"
        );
    }
}
