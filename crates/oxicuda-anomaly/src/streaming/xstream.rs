//! xStream — Manzoor, Lamba & Akoglu, KDD 2018.
//!
//! *"xStream: Outlier Detection in Feature-Evolving Data Streams."*
//!
//! xStream detects outliers in high-dimensional, possibly *feature-evolving*
//! streams using three ingredients:
//!
//! 1. **StreamHash** — a sparse random projection `R^d → R^K` to a fixed number
//!    of components `K`.  Each projected component is a sparse signed sum of the
//!    input features; the sparse signs are produced deterministically from a
//!    seed so that the projection of any point is reproducible and so that
//!    *new* features appearing later in the stream simply contribute additional
//!    terms (feature-evolving support).
//!
//! 2. **Half-space chains** — an ensemble of multi-scale grid-count structures.
//!    A chain of depth `D` fixes, for every level `ℓ = 1..D`, one projected
//!    component `f_ℓ` (sampled with replacement) and a random shift.  As the
//!    level deepens, the grid resolution for the chosen component is *doubled*
//!    (successive coordinate halving), giving a multi-scale family of bins.
//!    The number of times component `f` has been split by level `ℓ` is `k_f`,
//!    and the bin index is `⌊ (xⱼ / 2^{k_f}) + shift_f ⌋` where `xⱼ` is the
//!    projected coordinate.  Each level keeps a hash map from the bin *prefix*
//!    to a count.
//!
//! 3. **Multi-scale density & score** — the density of a point at level `ℓ` is
//!    its bin count scaled by `2^ℓ` (deeper, finer bins are up-weighted to
//!    compensate for the shrinking cell volume).  The point's per-chain score
//!    is the **minimum depth-scaled density over all levels**; the anomaly
//!    score is the negative of the average of this minimum over the ensemble:
//!
//!    ```text
//!    score(x) = − (1/C) Σ_chains  min_{ℓ}  2^ℓ · count_ℓ(bin_ℓ(x))
//!    ```
//!
//!    Outliers fall in sparsely populated regions ⇒ low minimum density ⇒ high
//!    (less negative) anomaly score.

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

// ─── StreamHash sparse projection ──────────────────────────────────────────────

/// Deterministic sparse random projection `R^d → R^K` (StreamHash).
///
/// Component `k` is `Σⱼ s_{k,j} · xⱼ` where each `s_{k,j} ∈ {−1, 0, +1}` with
/// `P(±1) = density/2` and `P(0) = 1 − density` (Achlioptas sparse projection).
/// The signs are generated from a per-`(k,j)` hash of the seed, so the same
/// `(k, j)` always yields the same sign even as the feature set grows.
#[derive(Debug, Clone)]
pub struct StreamHash {
    /// Output dimensionality `K`.
    pub k: usize,
    /// Input dimensionality `d` the projection was created for.
    pub d: usize,
    /// Sparse signs flattened row-major as `[K × d]`, values in `{−1, 0, +1}`.
    signs: Vec<f32>,
    /// Scale `1/√(density · d)` applied to every component.
    scale: f32,
}

impl StreamHash {
    /// Build a StreamHash projecting `d` features to `k` components.
    ///
    /// `density ∈ (0, 1]` is the expected fraction of non-zero signs per
    /// component (default in [`XStreamConfig`] is `1/3`, Achlioptas).
    #[must_use]
    pub fn new(d: usize, k: usize, density: f32, seed: u64) -> Self {
        let mut signs = vec![0.0_f32; k * d];
        // Per-(k,j) deterministic RNG so signs are stable & reproducible.
        for ki in 0..k {
            for j in 0..d {
                // Mix component and feature indices into the seed.
                let mut local = LcgRng::new(
                    seed ^ ((ki as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                        ^ ((j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)),
                );
                let u = local.next_f32();
                let s = if u < density * 0.5 {
                    1.0_f32
                } else if u < density {
                    -1.0_f32
                } else {
                    0.0_f32
                };
                signs[ki * d + j] = s;
            }
        }
        let scale = 1.0_f32 / (density * d as f32).max(1e-12).sqrt();
        Self { k, d, signs, scale }
    }

    /// Project a single point `x` (length ≤ `d`; missing trailing features are
    /// treated as `0`, supporting feature-evolving streams) to `K` components.
    #[must_use]
    pub fn project(&self, x: &[f32]) -> Vec<f32> {
        let used = x.len().min(self.d);
        (0..self.k)
            .map(|ki| {
                let row = &self.signs[ki * self.d..ki * self.d + used];
                let acc: f32 = row.iter().zip(x.iter()).map(|(s, xi)| s * xi).sum();
                acc * self.scale
            })
            .collect()
    }
}

// ─── Half-space chain ──────────────────────────────────────────────────────────

/// A single multi-scale half-space chain over the `K` projected components.
///
/// At every level `ℓ` (`0..depth`) the chain fixes a component `feature[ℓ]`
/// (sampled with replacement) and increments that component's resolution.  Each
/// level owns a hash map from the running bin prefix to a count.
#[derive(Debug, Clone)]
pub struct HalfSpaceChain {
    /// Component chosen at each level.
    feature: Vec<usize>,
    /// Per-component random shift in `[0, 1)`.
    shift: Vec<f32>,
    /// Per-level bin-prefix → count maps.
    counts: Vec<Vec<(Vec<i64>, u64)>>,
    /// Chain depth (number of levels).
    depth: usize,
    /// Number of projected components `K`.
    k: usize,
}

impl HalfSpaceChain {
    /// Construct an empty chain of `depth` levels over `k` components.
    fn new(k: usize, depth: usize, rng: &mut LcgRng) -> Self {
        let feature: Vec<usize> = (0..depth).map(|_| rng.next_usize(k.max(1))).collect();
        let shift: Vec<f32> = (0..k).map(|_| rng.next_f32()).collect();
        Self {
            feature,
            shift,
            counts: vec![Vec::new(); depth],
            depth,
            k,
        }
    }

    /// Number of times each component has been split *up to and including*
    /// level `ℓ`, i.e. how many of `feature[0..=ℓ]` equal each component.
    ///
    /// Returns the bin index of the projected point at every level as a growing
    /// prefix vector: `levels[ℓ]` is the length-`(ℓ+1)` bin id used at level ℓ.
    fn bin_prefixes(&self, z: &[f32]) -> Vec<Vec<i64>> {
        // Track how many times each component has been halved so far.
        let mut splits = vec![0u32; self.k];
        let mut prefix: Vec<i64> = Vec::with_capacity(self.depth);
        let mut levels: Vec<Vec<i64>> = Vec::with_capacity(self.depth);
        for &f in &self.feature {
            splits[f] += 1;
            // Finer grid: multiply coordinate by 2^(splits) (== /2 cell size).
            let factor = (1u64 << splits[f]) as f32;
            let shifted = z[f] * factor + self.shift[f];
            let bin = shifted.floor() as i64;
            prefix.push(bin);
            levels.push(prefix.clone());
        }
        levels
    }

    /// Add a projected point to every level's count map.
    fn add(&mut self, z: &[f32]) {
        let levels = self.bin_prefixes(z);
        for (l, key) in levels.into_iter().enumerate() {
            increment(&mut self.counts[l], key);
        }
    }

    /// Look up the count of the bin containing `z` at each level.  Returns a
    /// vector of length `depth`.
    fn level_counts(&self, z: &[f32]) -> Vec<u64> {
        let levels = self.bin_prefixes(z);
        levels
            .into_iter()
            .enumerate()
            .map(|(l, key)| lookup(&self.counts[l], &key))
            .collect()
    }

    /// Minimum depth-scaled density `min_ℓ 2^{ℓ+1} · count_ℓ` for `z`.
    ///
    /// `query_self == true` subtracts 1 from every count so that a *training*
    /// point is not counted as its own neighbour when scored in-sample.
    fn min_scaled_density(&self, z: &[f32], query_self: bool) -> f32 {
        let counts = self.level_counts(z);
        let mut best = f32::INFINITY;
        for (l, c) in counts.into_iter().enumerate() {
            let adj = if query_self { c.saturating_sub(1) } else { c };
            let scale = (1u64 << (l + 1)) as f32; // 2^{ℓ+1}
            let density = scale * adj as f32;
            if density < best {
                best = density;
            }
        }
        if best.is_finite() { best } else { 0.0 }
    }
}

/// Increment the count for `key` in a sorted-by-insertion association list.
fn increment(map: &mut Vec<(Vec<i64>, u64)>, key: Vec<i64>) {
    for (k, c) in map.iter_mut() {
        if *k == key {
            *c += 1;
            return;
        }
    }
    map.push((key, 1));
}

/// Look up the count for `key`, returning `0` when absent.
fn lookup(map: &[(Vec<i64>, u64)], key: &[i64]) -> u64 {
    for (k, c) in map {
        if k.as_slice() == key {
            return *c;
        }
    }
    0
}

// ─── Configuration ──────────────────────────────────────────────────────────────

/// Hyper-parameters for [`XStream`].
#[derive(Debug, Clone)]
pub struct XStreamConfig {
    /// Number of StreamHash projection components `K` (default `100`).
    pub n_components: usize,
    /// Number of half-space chains in the ensemble (default `50`).
    pub n_chains: usize,
    /// Depth (levels) of every chain (default `15`).
    pub depth: usize,
    /// Sparse-projection density (fraction of non-zero signs, default `1/3`).
    pub projection_density: f32,
    /// RNG seed (default `42`).
    pub seed: u64,
}

impl Default for XStreamConfig {
    fn default() -> Self {
        Self {
            n_components: 100,
            n_chains: 50,
            depth: 15,
            projection_density: 1.0 / 3.0,
            seed: 42,
        }
    }
}

// ─── XStream detector ────────────────────────────────────────────────────────────

/// xStream feature-evolving stream anomaly detector.
///
/// # Usage
///
/// ```rust,ignore
/// let mut det = XStream::new(XStreamConfig::default());
/// det.fit(&train, n_samples, n_features)?;     // populates chain bin counts
/// let score = det.score(&query)?;              // higher ⇒ more anomalous
/// ```
pub struct XStream {
    config: XStreamConfig,
    hash: Option<StreamHash>,
    chains: Vec<HalfSpaceChain>,
    n_features: usize,
    fitted: bool,
}

impl XStream {
    /// Create an unfitted detector from `config`.
    #[must_use]
    pub fn new(config: XStreamConfig) -> Self {
        Self {
            config,
            hash: None,
            chains: Vec::new(),
            n_features: 0,
            fitted: false,
        }
    }

    /// Fit on `data` (row-major `[n_samples × n_features]`): build the
    /// StreamHash, allocate the chain ensemble, and stream every training point
    /// into the chains.
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
        if self.config.n_chains == 0 {
            return Err(AnomalyError::Internal {
                msg: "n_chains must be > 0".into(),
            });
        }
        if self.config.depth == 0 {
            return Err(AnomalyError::Internal {
                msg: "depth must be > 0".into(),
            });
        }

        let hash = StreamHash::new(
            n_features,
            self.config.n_components,
            self.config.projection_density.clamp(1e-6, 1.0),
            self.config.seed,
        );

        // Allocate chains from a master RNG seeded distinctly from StreamHash.
        let mut rng = LcgRng::new(self.config.seed ^ 0xDEAD_BEEF_CAFE_F00D);
        let mut chains = Vec::with_capacity(self.config.n_chains);
        for _ in 0..self.config.n_chains {
            chains.push(HalfSpaceChain::new(
                self.config.n_components,
                self.config.depth,
                &mut rng,
            ));
        }

        // Stream every training point into the chains.
        for i in 0..n_samples {
            let row = &data[i * n_features..(i + 1) * n_features];
            let z = hash.project(row);
            for chain in chains.iter_mut() {
                chain.add(&z);
            }
        }

        self.hash = Some(hash);
        self.chains = chains;
        self.n_features = n_features;
        self.fitted = true;
        Ok(())
    }

    /// Stream a single new point into all chains (online update / feature-evolve).
    pub fn partial_fit(&mut self, x: &[f32]) -> AnomalyResult<()> {
        let hash = self.hash.as_ref().ok_or(AnomalyError::NotFitted)?;
        let z = hash.project(x);
        for chain in self.chains.iter_mut() {
            chain.add(&z);
        }
        Ok(())
    }

    /// Anomaly score of `x`: `− mean_chains( min_ℓ depth-scaled density )`.
    ///
    /// Higher ⇒ sparser local region ⇒ more anomalous.  The query point is
    /// treated as out-of-sample (its own contribution is *not* subtracted).
    pub fn score(&self, x: &[f32]) -> AnomalyResult<f32> {
        self.score_internal(x, false)
    }

    /// In-sample anomaly score: like [`XStream::score`] but subtracts the
    /// query's own contribution from every bin count (use when `x` is one of
    /// the training points).
    pub fn score_in_sample(&self, x: &[f32]) -> AnomalyResult<f32> {
        self.score_internal(x, true)
    }

    fn score_internal(&self, x: &[f32], query_self: bool) -> AnomalyResult<f32> {
        if !self.fitted {
            return Err(AnomalyError::NotFitted);
        }
        if x.len() != self.n_features {
            return Err(AnomalyError::FeatureCountMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }
        let hash = self.hash.as_ref().ok_or(AnomalyError::NotFitted)?;
        let z = hash.project(x);
        let mut sum_min_density = 0.0_f32;
        for chain in &self.chains {
            sum_min_density += chain.min_scaled_density(&z, query_self);
        }
        let mean_min_density = sum_min_density / self.chains.len() as f32;
        // Negative density ⇒ anomaly score increases as density falls.
        Ok(-mean_min_density)
    }

    /// Batch scoring; `x` is row-major `[n × n_features]`; returns `[n]`.
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

    /// Read-only access to the fitted StreamHash (if fitted).
    #[must_use]
    #[inline]
    pub fn stream_hash(&self) -> Option<&StreamHash> {
        self.hash.as_ref()
    }

    /// Number of features the detector was fitted on (0 if unfitted).
    #[must_use]
    #[inline]
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense 2-D cluster around the origin (stddev 0.3) plus distant outliers.
    fn dense_cluster(n: usize, seed: u64) -> (Vec<f32>, usize) {
        let mut rng = LcgRng::new(seed);
        let mut data = Vec::with_capacity(n * 2);
        for _ in 0..n {
            data.push(rng.next_normal() * 0.3);
            data.push(rng.next_normal() * 0.3);
        }
        (data, n)
    }

    // ── (d) StreamHash projection is deterministic / consistent ───────────────
    #[test]
    fn streamhash_deterministic() {
        let h1 = StreamHash::new(8, 16, 1.0 / 3.0, 99);
        let h2 = StreamHash::new(8, 16, 1.0 / 3.0, 99);
        let x = [0.1_f32, -0.4, 0.7, 0.2, -0.9, 0.3, 0.5, -0.1];
        let p1 = h1.project(&x);
        let p2 = h2.project(&x);
        assert_eq!(p1.len(), 16);
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert!(
                (a - b).abs() < 1e-7,
                "projection not deterministic: {a} vs {b}"
            );
        }
        // Same point projected twice through the same hash is identical.
        let p3 = h1.project(&x);
        assert_eq!(p1, p3, "repeat projection must match exactly");
    }

    // StreamHash supports feature-evolving input (shorter vectors zero-padded).
    #[test]
    fn streamhash_feature_evolving() {
        let h = StreamHash::new(6, 8, 1.0 / 3.0, 7);
        let full = [0.5_f32, 0.5, 0.5, 0.0, 0.0, 0.0];
        let short = [0.5_f32, 0.5, 0.5]; // trailing features absent
        let pf = h.project(&full);
        let ps = h.project(&short);
        for (a, b) in pf.iter().zip(ps.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "evolving projection mismatch: {a} vs {b}"
            );
        }
    }

    // ── (b) Half-space-chain bin counts update correctly as points stream in ──
    #[test]
    fn chain_bin_counts_update() {
        let mut rng = LcgRng::new(5);
        let mut chain = HalfSpaceChain::new(4, 6, &mut rng);
        // Insert the SAME projected point N times; its bin count at every level
        // must equal N.
        let z = vec![0.37_f32, -0.12, 0.5, 0.9];
        for _ in 0..7 {
            chain.add(&z);
        }
        let counts = chain.level_counts(&z);
        assert_eq!(counts.len(), 6, "one count per level");
        for (l, c) in counts.iter().enumerate() {
            assert_eq!(*c, 7, "level {l} count should be 7, got {c}");
        }
        // A point in an entirely different region has count 0 at the deepest level.
        let far = vec![1000.0_f32, -1000.0, 500.0, -500.0];
        let far_counts = chain.level_counts(&far);
        assert_eq!(
            *far_counts
                .last()
                .expect("far_counts vec should be non-empty"),
            0,
            "far point should be empty"
        );
    }

    // ── (c) Multi-scale aggregation takes the min density across levels ───────
    #[test]
    fn multiscale_takes_min_density() {
        let mut rng = LcgRng::new(11);
        let mut chain = HalfSpaceChain::new(3, 5, &mut rng);
        // Build counts that are large at coarse levels but become small at fine
        // levels for our query (by inserting points that share the coarse bin
        // but diverge in finer bins).
        let base = [0.2_f32, 0.3, 0.4];
        for k in 0..10 {
            // Perturb only slightly so coarse bins coincide but fine bins split.
            let z = [base[0] + k as f32 * 0.01, base[1], base[2]];
            chain.add(&z);
        }
        let query = [base[0], base[1], base[2]];
        let counts = chain.level_counts(&query);
        // Reconstruct the depth-scaled densities and confirm min_scaled_density
        // equals the minimum of 2^{ℓ+1} · count_ℓ.
        let mut expected_min = f32::INFINITY;
        for (l, c) in counts.iter().enumerate() {
            let d = (1u64 << (l + 1)) as f32 * *c as f32;
            if d < expected_min {
                expected_min = d;
            }
        }
        let got = chain.min_scaled_density(&query, false);
        assert!(
            (got - expected_min).abs() < 1e-3,
            "min density {got} should equal min over levels {expected_min}"
        );
    }

    // ── (a)+(e) Outlier (sparse) scores higher; dense point scores low ────────
    #[test]
    fn outlier_scores_higher_than_dense_point() {
        let (data, n) = dense_cluster(300, 21);
        let mut det = XStream::new(XStreamConfig {
            n_components: 50,
            n_chains: 60,
            depth: 12,
            projection_density: 1.0 / 3.0,
            seed: 3,
        });
        det.fit(&data, n, 2).expect("fit");

        let dense_point = [0.0_f32, 0.0]; // centre of the cluster
        let outlier = [15.0_f32, -15.0]; // far sparse region

        let s_dense = det.score(&dense_point).expect("dense score");
        let s_outlier = det.score(&outlier).expect("outlier score");

        assert!(
            s_outlier > s_dense,
            "outlier score {s_outlier} should exceed dense score {s_dense}"
        );
    }

    // ── (f) Score is monotone in local sparsity ──────────────────────────────
    #[test]
    fn score_monotone_in_sparsity() {
        // Cluster near origin; move the query progressively away.  As the query
        // leaves the dense region the (min) density should drop monotonically,
        // so the anomaly score should be non-decreasing.
        let (data, n) = dense_cluster(400, 33);
        let mut det = XStream::new(XStreamConfig {
            n_components: 40,
            n_chains: 80,
            depth: 12,
            projection_density: 1.0 / 3.0,
            seed: 9,
        });
        det.fit(&data, n, 2).expect("fit");

        // Distances increasing from the cluster centre.
        let near = det.score(&[0.0_f32, 0.0]).expect("near");
        let mid = det.score(&[3.0_f32, 3.0]).expect("mid");
        let far = det.score(&[30.0_f32, 30.0]).expect("far");

        assert!(
            near <= mid,
            "score should not decrease moving out: {near} → {mid}"
        );
        assert!(
            mid <= far,
            "score should not decrease moving out: {mid} → {far}"
        );
        assert!(
            far > near,
            "far point must be more anomalous than the centre"
        );
    }

    // ── partial_fit updates counts (a streamed point raises local density) ────
    #[test]
    fn partial_fit_increases_local_density() {
        let (data, n) = dense_cluster(120, 4);
        let mut det = XStream::new(XStreamConfig {
            n_components: 30,
            n_chains: 40,
            depth: 10,
            projection_density: 1.0 / 3.0,
            seed: 8,
        });
        det.fit(&data, n, 2).expect("fit");

        let probe = [8.0_f32, 8.0];
        let before = det.score(&probe).expect("before");
        // Stream the probe location many times → it becomes dense → lower score.
        for _ in 0..200 {
            det.partial_fit(&probe).expect("partial fit");
        }
        let after = det.score(&probe).expect("after");
        assert!(
            after < before,
            "after densifying, score {after} should drop below {before}"
        );
    }

    // ── Error handling ────────────────────────────────────────────────────────
    #[test]
    fn unfitted_score_errors() {
        let det = XStream::new(XStreamConfig::default());
        match det.score(&[0.0_f32, 0.0]) {
            Err(AnomalyError::NotFitted) => {}
            other => panic!("expected NotFitted, got {other:?}"),
        }
    }

    #[test]
    fn empty_input_errors() {
        let mut det = XStream::new(XStreamConfig::default());
        match det.fit(&[], 0, 2) {
            Err(AnomalyError::EmptyInput) => {}
            other => panic!("expected EmptyInput, got {other:?}"),
        }
    }

    #[test]
    fn feature_count_mismatch_on_score() {
        let (data, n) = dense_cluster(20, 1);
        let mut det = XStream::new(XStreamConfig {
            n_components: 16,
            n_chains: 8,
            depth: 6,
            projection_density: 1.0 / 3.0,
            seed: 2,
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
        let mut det = XStream::new(XStreamConfig::default());
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
        let mut det = XStream::new(XStreamConfig {
            n_components: 0,
            ..XStreamConfig::default()
        });
        let data = vec![0.1_f32, 0.2, 0.3, 0.4];
        match det.fit(&data, 2, 2) {
            Err(AnomalyError::Internal { .. }) => {}
            other => panic!("expected Internal error, got {other:?}"),
        }
    }

    #[test]
    fn score_batch_length() {
        let (data, n) = dense_cluster(50, 6);
        let mut det = XStream::new(XStreamConfig {
            n_components: 24,
            n_chains: 20,
            depth: 8,
            projection_density: 1.0 / 3.0,
            seed: 4,
        });
        det.fit(&data, n, 2).expect("fit");
        let queries = vec![0.0_f32, 0.0, 1.0, 1.0, 10.0, 10.0];
        let scores = det.score_batch(&queries, 3).expect("batch");
        assert_eq!(scores.len(), 3);
        assert!(scores.iter().all(|s| s.is_finite()), "all finite");
    }
}
