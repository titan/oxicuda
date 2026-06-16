//! ROCK and IDEC streaming anomaly detection trackers.
//!
//! # ROCK (Robust Competitive Coding)
//! Maintains a set of coding vectors (centroids). Each incoming point "competes"
//! to update the nearest centroid via Kohonen-style online learning. The
//! reconstruction error (distance to the nearest centroid) serves as the anomaly
//! score. Points consistently far from all centroids are flagged as anomalies.
//!
//! # IDEC (Incremental Density Estimation for Concept-drift)
//! Estimates the local density at each incoming point using a Gaussian kernel
//! density estimate over a growing set of stored exemplars (with weight decay).
//! Points landing in low-density regions are flagged as anomalies.

use std::collections::VecDeque;

use crate::{
    error::{AnomalyError, AnomalyResult},
    handle::LcgRng,
};

// ─────────────────────────────────────────────────────────────────────────────
// Utility: squared Euclidean distance between two equal-length slices
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi) * (ai - bi))
        .sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// ROCK tracker
// ─────────────────────────────────────────────────────────────────────────────

/// Hyper-parameters for the ROCK streaming anomaly tracker.
#[derive(Debug, Clone)]
pub struct RockConfig {
    /// Number of coding vectors (centroids).
    pub n_centroids: usize,
    /// Sliding-window size for the reference set (retained but informational).
    pub window_size: usize,
    /// Kohonen-style online update rate for centroids (`0 < η ≤ 1`).
    pub learning_rate: f64,
    /// Distance threshold above which a point is flagged as an anomaly.
    pub anomaly_threshold: f64,
    /// Feature dimensionality.
    pub n_dims: usize,
}

/// ROCK streaming anomaly tracker.
///
/// Centroids are initialised from the first `n_centroids` observations.
/// Subsequent observations update the nearest centroid using Kohonen's rule:
/// ```text
/// c_winner ← c_winner + η * (x − c_winner)
/// ```
/// The reconstruction error (Euclidean distance to the nearest centroid) is
/// returned as the anomaly score.
#[derive(Debug, Clone)]
pub struct RockDetector {
    centroids: Vec<Vec<f64>>,
    window: VecDeque<Vec<f64>>,
    n_seen: usize,
}

impl RockDetector {
    /// Construct a new detector; centroids are initialised lazily from the
    /// first `n_centroids` observations.
    ///
    /// `rng` is used to randomly perturb centroids if needed (currently reserved
    /// for future use — deterministic init from stream).
    pub fn new(config: &RockConfig, rng: &mut LcgRng) -> AnomalyResult<Self> {
        if config.n_centroids == 0 {
            return Err(AnomalyError::InvalidFeatureCount {
                n: config.n_centroids,
            });
        }
        if config.n_dims == 0 {
            return Err(AnomalyError::InvalidFeatureCount { n: config.n_dims });
        }
        // Pre-allocate centroids using small random values so updates can begin
        // immediately even if n_centroids > number of initial points.
        let mut centroids = Vec::with_capacity(config.n_centroids);
        for k in 0..config.n_centroids {
            let offset = (k as f64 * 0.01) / (config.n_centroids as f64).max(1.0);
            let c: Vec<f64> = (0..config.n_dims)
                .map(|_| (rng.next_f32() as f64 - 0.5) * 0.01 + offset)
                .collect();
            centroids.push(c);
        }
        Ok(Self {
            centroids,
            window: VecDeque::with_capacity(config.window_size),
            n_seen: 0,
        })
    }

    /// Find the index of the nearest centroid to `point`.
    fn nearest_centroid_idx(&self, point: &[f64]) -> usize {
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, sq_dist(c, point)))
            .min_by(|(_, da), (_, db)| da.partial_cmp(db).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Compute Euclidean distance to the nearest centroid (anomaly score).
    fn min_centroid_dist(&self, point: &[f64]) -> f64 {
        self.centroids
            .iter()
            .map(|c| sq_dist(c, point).sqrt())
            .fold(f64::INFINITY, f64::min)
    }

    /// Update the detector with a new observation.
    ///
    /// 1. Finds the winning centroid (nearest).
    /// 2. Performs a Kohonen online update: `c_w ← c_w + η·(x − c_w)`.
    /// 3. Maintains the sliding window.
    /// 4. Returns the distance to the **pre-update** winner as the anomaly score.
    pub fn update(&mut self, point: &[f64], config: &RockConfig) -> AnomalyResult<f64> {
        if point.len() != config.n_dims {
            return Err(AnomalyError::DimensionMismatch {
                expected: config.n_dims,
                got: point.len(),
            });
        }

        // Score using pre-update centroids
        let score = self.min_centroid_dist(point);

        // Warm-up: replace centroid verbatim with early observations
        let winner_idx = self.nearest_centroid_idx(point);
        if self.n_seen < config.n_centroids {
            // Verbatim initialisation: set centroid to this observation
            let centroid_to_init = self.n_seen % config.n_centroids;
            self.centroids[centroid_to_init][..config.n_dims]
                .copy_from_slice(&point[..config.n_dims]);
        } else {
            // Kohonen update
            let eta = config.learning_rate;
            let c = &mut self.centroids[winner_idx];
            for d in 0..config.n_dims {
                c[d] += eta * (point[d] - c[d]);
            }
        }

        // Maintain sliding window
        if self.window.len() >= config.window_size {
            self.window.pop_front();
        }
        self.window.push_back(point.to_vec());

        self.n_seen += 1;
        Ok(score)
    }

    /// Score a new point without modifying the detector state (inference mode).
    ///
    /// Returns the Euclidean distance to the nearest centroid.
    pub fn score(&self, point: &[f64]) -> AnomalyResult<f64> {
        if point.len() != self.centroids[0].len() {
            return Err(AnomalyError::DimensionMismatch {
                expected: self.centroids[0].len(),
                got: point.len(),
            });
        }
        Ok(self.min_centroid_dist(point))
    }

    /// Return `true` if `score` exceeds the configured threshold.
    pub fn is_anomaly(&self, score: f64, config: &RockConfig) -> bool {
        score > config.anomaly_threshold
    }

    /// Reference to the current centroid vectors.
    pub fn centroids(&self) -> &[Vec<f64>] {
        &self.centroids
    }

    /// Total number of observations processed (including the current one).
    pub fn n_seen(&self) -> usize {
        self.n_seen
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IDEC tracker
// ─────────────────────────────────────────────────────────────────────────────

/// Hyper-parameters for the IDEC incremental density tracker.
#[derive(Debug, Clone)]
pub struct IdecConfig {
    /// Gaussian kernel bandwidth (standard deviation of the kernel).
    pub bandwidth: f64,
    /// Maximum number of stored exemplars.
    pub max_components: usize,
    /// A point is anomalous if `density(x) / max_density < min_density_ratio`.
    pub min_density_ratio: f64,
    /// Weight decay applied to existing exemplar weights each update step.
    /// `forgetting_factor = 0.0` → no decay; `1.0` → complete forgetting.
    pub forgetting_factor: f64,
}

/// IDEC incremental density tracker.
///
/// Stores a set of `(exemplar_point, weight)` pairs. The density at a query
/// point is estimated via a weighted Gaussian kernel density estimator:
/// ```text
/// f(x) = Σ_i  w_i · K_h(x − x_i)
/// ```
/// where `K_h(u) = exp(−‖u‖²/(2h²)) / (h√(2π))^d`.
#[derive(Debug, Clone)]
pub struct IdecDetector {
    exemplars: Vec<(Vec<f64>, f64)>, // (point, weight)
    max_density: f64,
}

impl IdecDetector {
    /// Create an empty detector. The first point processed by `update` will
    /// **never** be flagged as an anomaly because `max_density` is not yet set.
    pub fn new() -> Self {
        Self {
            exemplars: Vec::new(),
            max_density: 0.0,
        }
    }

    /// Gaussian kernel density estimate at `point` over all exemplars.
    ///
    /// Formula for a single exemplar `(x_i, w_i)`:
    /// ```text
    /// w_i * exp(−‖x − x_i‖² / (2h²)) / (h * √(2π))^d
    /// ```
    /// where `h = bandwidth` and `d = dimensionality`.
    pub fn density(&self, point: &[f64], bandwidth: f64) -> f64 {
        if self.exemplars.is_empty() {
            return 0.0;
        }
        let d = point.len();
        let h2 = bandwidth * bandwidth * 2.0;
        let norm_const = (bandwidth * std::f64::consts::TAU.sqrt()).powi(d as i32);

        self.exemplars
            .iter()
            .map(|(xi, wi)| {
                let dist2: f64 = xi
                    .iter()
                    .zip(point.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                wi * (-dist2 / h2).exp() / norm_const
            })
            .sum()
    }

    /// Update the detector with a new observation.
    ///
    /// Steps:
    /// 1. Apply weight decay to all existing exemplars.
    /// 2. Compute density at the new point using current exemplars.
    /// 3. Determine if the point is an anomaly (`density / max_density < ratio`).
    /// 4. Store the point as a new exemplar (weight = 1.0).
    /// 5. Update `max_density`.
    /// 6. Prune to `max_components` if necessary.
    ///
    /// Returns `(density, is_anomaly)`.
    pub fn update(&mut self, point: &[f64], config: &IdecConfig) -> AnomalyResult<(f64, bool)> {
        if point.is_empty() {
            return Err(AnomalyError::EmptyInput);
        }

        // Step 1: weight decay on existing exemplars
        if config.forgetting_factor > 0.0 {
            let decay = 1.0 - config.forgetting_factor;
            for (_, w) in self.exemplars.iter_mut() {
                *w *= decay;
            }
        }

        // Step 2: compute density at new point before adding it
        let dens = self.density(point, config.bandwidth);

        // Step 3: anomaly decision
        let is_anomaly = if self.max_density <= 0.0 {
            // First point: no reference density yet → never anomaly
            false
        } else {
            (dens / self.max_density) < config.min_density_ratio
        };

        // Step 4: add point as new exemplar
        self.exemplars.push((point.to_vec(), 1.0));

        // Step 5: update max density (we recompute density including the new exemplar)
        let new_dens = self.density(point, config.bandwidth);
        if new_dens > self.max_density {
            self.max_density = new_dens;
        }

        // Step 6: prune if over capacity
        if self.exemplars.len() > config.max_components {
            self.prune(config.max_components);
        }

        Ok((dens, is_anomaly))
    }

    /// Prune exemplars to at most `max_components` by removing those with the
    /// lowest weight.
    fn prune(&mut self, max_components: usize) {
        if self.exemplars.len() <= max_components {
            return;
        }
        // Sort by weight descending and keep top `max_components`
        self.exemplars.sort_unstable_by(|(_, wa), (_, wb)| {
            wb.partial_cmp(wa).unwrap_or(std::cmp::Ordering::Equal)
        });
        self.exemplars.truncate(max_components);
    }

    /// Number of stored exemplars.
    pub fn n_exemplars(&self) -> usize {
        self.exemplars.len()
    }
}

impl Default for IdecDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn default_rock_config() -> RockConfig {
        RockConfig {
            n_centroids: 3,
            window_size: 20,
            learning_rate: 0.2,
            anomaly_threshold: 5.0,
            n_dims: 2,
        }
    }

    fn default_idec_config() -> IdecConfig {
        IdecConfig {
            bandwidth: 0.5,
            max_components: 10,
            min_density_ratio: 0.01,
            forgetting_factor: 0.0,
        }
    }

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    // ── ROCK tests ────────────────────────────────────────────────────────────

    #[test]
    fn rock_normal_scores_small_after_warmup() {
        let cfg = default_rock_config();
        let mut rng = make_rng();
        let mut det = RockDetector::new(&cfg, &mut rng)
            .expect("RockDetector construction should succeed with valid config");

        // Warm up with points near origin
        for i in 0..20 {
            let p = vec![(i as f64) * 0.01, (i as f64) * 0.01];
            det.update(&p, &cfg)
                .expect("ROCK update should succeed with valid input");
        }
        // A point near the cluster centre should have a low score
        let score = det
            .update(&[0.05, 0.05], &cfg)
            .expect("ROCK update should succeed with valid input");
        assert!(
            score < cfg.anomaly_threshold,
            "expected small score for inlier, got {score}"
        );
    }

    #[test]
    fn rock_outlier_scores_high() {
        let cfg = default_rock_config();
        let mut rng = make_rng();
        let mut det = RockDetector::new(&cfg, &mut rng)
            .expect("RockDetector construction should succeed with valid config");

        // Train on cluster around (1, 1)
        for _ in 0..30 {
            det.update(&[1.0, 1.0], &cfg)
                .expect("ROCK update should succeed with valid input");
        }
        // Far-away outlier
        let score = det
            .update(&[1000.0, 1000.0], &cfg)
            .expect("ROCK update should succeed with valid input");
        assert!(
            score > cfg.anomaly_threshold,
            "expected high score for outlier, got {score}"
        );
    }

    #[test]
    fn rock_update_changes_centroids() {
        let cfg = default_rock_config();
        let mut rng = make_rng();
        let mut det = RockDetector::new(&cfg, &mut rng)
            .expect("RockDetector construction should succeed with valid config");

        let before: Vec<Vec<f64>> = det.centroids().to_vec();
        det.update(&[10.0, 10.0], &cfg)
            .expect("ROCK update should succeed with valid input");
        let after: Vec<Vec<f64>> = det.centroids().to_vec();

        // At least one centroid must have changed
        let changed = before.iter().zip(after.iter()).any(|(b, a)| {
            b.iter()
                .zip(a.iter())
                .any(|(bi, ai)| (bi - ai).abs() > 1e-12)
        });
        assert!(changed, "centroids should change after update");
    }

    #[test]
    fn rock_score_is_deterministic() {
        let cfg = default_rock_config();
        let mut rng = make_rng();
        let mut det = RockDetector::new(&cfg, &mut rng)
            .expect("RockDetector construction should succeed with valid config");

        for _ in 0..10 {
            det.update(&[1.0, 1.0], &cfg)
                .expect("ROCK update should succeed with valid input");
        }
        let p = &[2.0, 2.0];
        let s1 = det
            .score(p)
            .expect("ROCK score should succeed with valid input");
        let s2 = det
            .score(p)
            .expect("ROCK score should succeed with valid input");
        assert_eq!(s1, s2, "score must be deterministic in inference mode");
    }

    #[test]
    fn rock_is_anomaly_flags_correctly() {
        let cfg = default_rock_config();
        let mut rng = make_rng();
        let det = RockDetector::new(&cfg, &mut rng)
            .expect("RockDetector construction should succeed with valid config");

        assert!(
            !det.is_anomaly(cfg.anomaly_threshold - 0.001, &cfg),
            "below threshold should not be anomaly"
        );
        assert!(
            det.is_anomaly(cfg.anomaly_threshold + 0.001, &cfg),
            "above threshold should be anomaly"
        );
    }

    #[test]
    fn rock_centroids_have_correct_shape() {
        let cfg = default_rock_config();
        let mut rng = make_rng();
        let det = RockDetector::new(&cfg, &mut rng)
            .expect("RockDetector construction should succeed with valid config");

        assert_eq!(det.centroids().len(), cfg.n_centroids);
        for c in det.centroids() {
            assert_eq!(c.len(), cfg.n_dims);
        }
    }

    #[test]
    fn rock_n_dims_mismatch_returns_error() {
        let cfg = default_rock_config();
        let mut rng = make_rng();
        let mut det = RockDetector::new(&cfg, &mut rng)
            .expect("RockDetector construction should succeed with valid config");

        let wrong_dim = vec![1.0_f64; cfg.n_dims + 1];
        let result = det.update(&wrong_dim, &cfg);
        assert!(result.is_err(), "mismatched dims should return error");
    }

    #[test]
    fn rock_n_seen_increments() {
        let cfg = default_rock_config();
        let mut rng = make_rng();
        let mut det = RockDetector::new(&cfg, &mut rng)
            .expect("RockDetector construction should succeed with valid config");

        for i in 0..5 {
            det.update(&[i as f64, i as f64], &cfg)
                .expect("ROCK update should succeed with valid input");
            assert_eq!(det.n_seen(), i + 1);
        }
    }

    #[test]
    fn rock_score_only_no_state_change() {
        let cfg = default_rock_config();
        let mut rng = make_rng();
        let mut det = RockDetector::new(&cfg, &mut rng)
            .expect("RockDetector construction should succeed with valid config");

        for _ in 0..5 {
            det.update(&[1.0, 1.0], &cfg)
                .expect("ROCK update should succeed with valid input");
        }
        let n_before = det.n_seen();
        let centroids_before = det.centroids().to_vec();
        let _ = det
            .score(&[1.0, 1.0])
            .expect("ROCK score should succeed with valid input");
        assert_eq!(det.n_seen(), n_before, "score() must not change n_seen");
        assert_eq!(
            det.centroids(),
            centroids_before,
            "score() must not change centroids"
        );
    }

    // ── IDEC tests ────────────────────────────────────────────────────────────

    #[test]
    fn idec_density_exemplar_higher_than_distant() {
        let cfg = default_idec_config();
        let mut det = IdecDetector::new();

        // Add a few exemplars near origin
        for _ in 0..5 {
            det.update(&[0.0, 0.0], &cfg)
                .expect("IDEC update should succeed");
        }
        let dens_near = det.density(&[0.0, 0.0], cfg.bandwidth);
        let dens_far = det.density(&[100.0, 100.0], cfg.bandwidth);
        assert!(
            dens_near > dens_far,
            "density near exemplars ({dens_near}) should exceed density far away ({dens_far})"
        );
    }

    #[test]
    fn idec_density_formula_gaussian_kernel() {
        // Verify formula: exp(-‖x-y‖²/(2h²)) / (h√(2π))^d
        let h = 1.0_f64;
        let d = 1_usize;
        let mut det = IdecDetector::new();
        let xi = vec![0.0_f64];
        det.exemplars.push((xi.clone(), 1.0));
        det.max_density = 1.0;

        let x = vec![1.0_f64];
        let dist2 = 1.0_f64;
        let expected =
            (-dist2 / (2.0 * h * h)).exp() / (h * std::f64::consts::TAU.sqrt()).powi(d as i32);
        let actual = det.density(&x, h);
        assert!(
            (actual - expected).abs() < 1e-12,
            "density formula mismatch: actual={actual} expected={expected}"
        );
    }

    #[test]
    fn idec_update_adds_exemplars() {
        let cfg = default_idec_config();
        let mut det = IdecDetector::new();

        assert_eq!(det.n_exemplars(), 0);
        det.update(&[1.0, 2.0], &cfg)
            .expect("IDEC update should succeed");
        assert_eq!(det.n_exemplars(), 1);
        det.update(&[3.0, 4.0], &cfg)
            .expect("IDEC update should succeed");
        assert_eq!(det.n_exemplars(), 2);
    }

    #[test]
    fn idec_prune_keeps_max_components() {
        let cfg = IdecConfig {
            max_components: 5,
            ..default_idec_config()
        };
        let mut det = IdecDetector::new();

        for i in 0..20 {
            det.update(&[i as f64, 0.0], &cfg)
                .expect("IDEC update should succeed");
        }
        assert!(
            det.n_exemplars() <= cfg.max_components,
            "n_exemplars={} should be ≤ max_components={}",
            det.n_exemplars(),
            cfg.max_components
        );
    }

    #[test]
    fn idec_forgetting_factor_decays_weights() {
        let cfg = IdecConfig {
            forgetting_factor: 0.5,
            max_components: 100,
            ..default_idec_config()
        };
        let mut det = IdecDetector::new();

        det.update(&[0.0, 0.0], &cfg)
            .expect("IDEC update should succeed");
        let w0 = det.exemplars[0].1;
        det.update(&[0.0, 0.0], &cfg)
            .expect("IDEC update should succeed");
        let w0_after = det.exemplars[0].1;
        // After one more update the first exemplar should have decayed
        assert!(
            w0_after < w0,
            "weight should decay: before={w0}, after={w0_after}"
        );
    }

    #[test]
    fn idec_outlier_detected() {
        let cfg = IdecConfig {
            min_density_ratio: 0.5, // strict threshold
            bandwidth: 0.1,
            max_components: 50,
            forgetting_factor: 0.0,
        };
        let mut det = IdecDetector::new();

        // Build up density near origin
        for _ in 0..20 {
            det.update(&[0.0, 0.0], &cfg)
                .expect("IDEC update should succeed");
        }
        // Point very far from all exemplars should be flagged
        let (dens, is_anom) = det
            .update(&[1000.0, 1000.0], &cfg)
            .expect("IDEC update for outlier should succeed");
        assert!(is_anom, "distant point density={dens} should be anomaly");
    }

    #[test]
    fn idec_first_point_never_anomaly() {
        let cfg = default_idec_config();
        let mut det = IdecDetector::new();

        let (_, is_anom) = det
            .update(&[999.0, 999.0], &cfg)
            .expect("IDEC update for first point should succeed");
        assert!(
            !is_anom,
            "first point should never be flagged as anomaly (no reference)"
        );
    }

    #[test]
    fn idec_empty_input_returns_error() {
        let cfg = default_idec_config();
        let mut det = IdecDetector::new();
        let result = det.update(&[], &cfg);
        assert!(result.is_err(), "empty point should return error");
    }
}
