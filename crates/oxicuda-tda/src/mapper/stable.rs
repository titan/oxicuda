//! Stable Mapper: the Mapper graph with a statistical-stability layer.
//!
//! Carrière & Oudot, *Structure and Stability of the One-Dimensional Mapper*
//! (Foundations of Computational Mathematics, 2018), show that the Mapper graph is a
//! statistically consistent estimator of the Reeb graph and propose a *bootstrap*
//! procedure to attach confidence to its topological features: resample the point
//! cloud with replacement, recompute the Mapper graph, and measure how often the
//! topological signature is reproduced.
//!
//! This module wraps the existing [`build_mapper`] machinery (cover + single-linkage
//! clustering + nerve) and adds that bootstrap layer.  The topological signature used
//! is the pair `(β₀, β₁)` of the Mapper graph — the number of connected components and
//! the number of independent loops — which are exactly the quantities the
//! one-dimensional Mapper is designed to recover.  The reported scores are
//!
//! * [`StableMapperResult::component_confidence`] — fraction of bootstrap replicates
//!   whose component count matches the reference graph;
//! * [`StableMapperResult::loop_confidence`] — fraction matching the loop count `β₁`;
//! * [`StableMapperResult::stability`] — fraction matching *both* (the joint
//!   confidence of the full signature).
//!
//! Each score lies in `[0, 1]`; values near `1` mean the recovered topology is robust
//! to resampling.  The bootstrap is driven by the deterministic [`LcgRng`] so results
//! are reproducible from [`StableMapperConfig::seed`].

use crate::error::{TdaError, TdaResult};
use crate::handle::LcgRng;
use crate::mapper::mapper::{MapperConfig, MapperGraph, build_mapper};

/// Configuration for [`stable_mapper`].
#[derive(Debug, Clone)]
pub struct StableMapperConfig {
    /// Base Mapper configuration (cover size, overlap, clustering scale, `min_pts`).
    pub mapper: MapperConfig,
    /// Number of bootstrap resamples (`> 0`).
    pub n_bootstrap: usize,
    /// Seed for the deterministic bootstrap RNG.
    pub seed: u64,
}

impl StableMapperConfig {
    /// Convenience constructor.
    pub fn new(mapper: MapperConfig, n_bootstrap: usize, seed: u64) -> Self {
        Self {
            mapper,
            n_bootstrap,
            seed,
        }
    }
}

/// Result of a Stable Mapper computation.
#[derive(Debug, Clone)]
pub struct StableMapperResult {
    /// The reference Mapper graph built on the full point cloud.
    pub graph: MapperGraph,
    /// Reference number of connected components (β₀).
    pub n_components: usize,
    /// Reference number of independent loops (β₁).
    pub betti_1: usize,
    /// Fraction of bootstraps reproducing the full `(β₀, β₁)` signature — in `[0, 1]`.
    pub stability: f64,
    /// Fraction of bootstraps reproducing the component count — in `[0, 1]`.
    pub component_confidence: f64,
    /// Fraction of bootstraps reproducing the loop count — in `[0, 1]`.
    pub loop_confidence: f64,
    /// The `(β₀, β₁)` signature of each bootstrap replicate.
    pub bootstrap_signatures: Vec<(usize, usize)>,
}

/// Build the Stable Mapper of a point cloud under a scalar filter.
///
/// The reference Mapper graph is built on the full data; then `cfg.n_bootstrap`
/// resamples (drawn with replacement) are each mapped and compared to the reference
/// `(β₀, β₁)` signature to produce bootstrap confidence scores.
///
/// `filter_fn` receives one point at a time as a `&[f64]` slice of length `n_dims`.
///
/// # Errors
/// * [`TdaError::EmptyPointCloud`] if `n_pts == 0`.
/// * [`TdaError::DimensionMismatch`] if `points.len() != n_pts * n_dims`.
/// * [`TdaError::InvalidCoverParameter`] if `n_intervals == 0` or `overlap_frac ∉ (0, 1)`.
/// * [`TdaError::ParameterOutOfRange`] if `n_bootstrap == 0`.
/// * any error raised by the underlying [`build_mapper`].
pub fn stable_mapper<F: Fn(&[f64]) -> f64>(
    points: &[f64],
    n_pts: usize,
    n_dims: usize,
    filter_fn: F,
    cfg: &StableMapperConfig,
) -> TdaResult<StableMapperResult> {
    if n_pts == 0 {
        return Err(TdaError::EmptyPointCloud);
    }
    if n_dims == 0 || points.len() != n_pts * n_dims {
        return Err(TdaError::DimensionMismatch {
            expected: n_pts * n_dims,
            got: points.len(),
        });
    }
    if cfg.mapper.n_intervals == 0 {
        return Err(TdaError::InvalidCoverParameter(
            "n_intervals must be > 0".to_owned(),
        ));
    }
    // Stable Mapper needs genuine — but proper — overlap: the open interval (0, 1).
    if !(cfg.mapper.overlap_frac > 0.0 && cfg.mapper.overlap_frac < 1.0) {
        return Err(TdaError::InvalidCoverParameter(format!(
            "overlap_frac must be in (0, 1), got {}",
            cfg.mapper.overlap_frac
        )));
    }
    if cfg.n_bootstrap == 0 {
        return Err(TdaError::ParameterOutOfRange(
            "n_bootstrap must be > 0".to_owned(),
        ));
    }

    // Reference graph and its topological signature.
    let graph = build_mapper(points, n_pts, n_dims, &filter_fn, &cfg.mapper)?;
    let ref_components = graph.connected_components().len();
    let ref_betti_1 = graph.betti_1();

    // Bootstrap: resample with replacement and compare signatures.
    let mut rng = LcgRng::new(cfg.seed);
    let mut signatures: Vec<(usize, usize)> = Vec::with_capacity(cfg.n_bootstrap);
    let mut component_match = 0usize;
    let mut loop_match = 0usize;
    let mut both_match = 0usize;
    let mut resampled = vec![0.0_f64; n_pts * n_dims];

    for _ in 0..cfg.n_bootstrap {
        for slot in 0..n_pts {
            let src = rng.next_usize(n_pts);
            resampled[slot * n_dims..(slot + 1) * n_dims]
                .copy_from_slice(&points[src * n_dims..(src + 1) * n_dims]);
        }
        let boot = build_mapper(&resampled, n_pts, n_dims, &filter_fn, &cfg.mapper)?;
        let components = boot.connected_components().len();
        let betti_1 = boot.betti_1();
        signatures.push((components, betti_1));
        if components == ref_components {
            component_match += 1;
        }
        if betti_1 == ref_betti_1 {
            loop_match += 1;
        }
        if components == ref_components && betti_1 == ref_betti_1 {
            both_match += 1;
        }
    }

    let denom = cfg.n_bootstrap as f64;
    Ok(StableMapperResult {
        graph,
        n_components: ref_components,
        betti_1: ref_betti_1,
        stability: both_match as f64 / denom,
        component_confidence: component_match as f64 / denom,
        loop_confidence: loop_match as f64 / denom,
        bootstrap_signatures: signatures,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn circle_points(n: usize) -> Vec<f64> {
        (0..n)
            .flat_map(|i| {
                let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
                vec![angle.cos(), angle.sin()]
            })
            .collect()
    }

    // (a) A circle with a coordinate filter: the reference Mapper graph has one loop.
    // (Dense sampling so that overlapping cover intervals reliably share points — with
    // too few points the discrete bands can miss each other and the loop never closes.)
    #[test]
    fn circle_reference_has_one_loop() {
        let n = 60usize;
        let pts = circle_points(n);
        let cfg = StableMapperConfig::new(
            MapperConfig {
                n_intervals: 6,
                overlap_frac: 0.4,
                cluster_eps: 0.5,
                min_pts: 1,
            },
            16,
            7,
        );
        let res = stable_mapper(&pts, n, 2, |p| p[0], &cfg).expect("stable mapper");
        assert_eq!(
            res.betti_1, 1,
            "circle Mapper graph must have exactly one loop"
        );
        assert_eq!(res.n_components, 1, "circle Mapper graph is connected");
    }

    // (b) Two well-separated clusters give two connected components.
    #[test]
    fn two_clusters_two_components() {
        // Cluster A around x ∈ [0, 1], cluster B around x ∈ [10, 11].
        let mut pts: Vec<f64> = Vec::new();
        for i in 0..10 {
            pts.push(i as f64 / 10.0);
            pts.push(0.0);
        }
        for i in 0..10 {
            pts.push(10.0 + i as f64 / 10.0);
            pts.push(0.0);
        }
        let n = 20usize;
        let cfg = StableMapperConfig::new(
            MapperConfig {
                n_intervals: 4,
                overlap_frac: 0.3,
                cluster_eps: 2.0,
                min_pts: 1,
            },
            16,
            3,
        );
        let res = stable_mapper(&pts, n, 2, |p| p[0], &cfg).expect("stable mapper");
        assert_eq!(res.n_components, 2, "two blobs ⇒ two components");
    }

    // (c) Stability score is in a sensible range; denser/cleaner data is more stable.
    #[test]
    fn stability_in_range_and_high_for_clean_data() {
        let make_two_blobs = |per: usize| -> (Vec<f64>, usize) {
            let mut pts = Vec::new();
            for i in 0..per {
                pts.push(i as f64 / per as f64);
                pts.push(0.0);
            }
            for i in 0..per {
                pts.push(10.0 + i as f64 / per as f64);
                pts.push(0.0);
            }
            (pts, 2 * per)
        };
        let mapper = MapperConfig {
            n_intervals: 4,
            overlap_frac: 0.3,
            cluster_eps: 2.0,
            min_pts: 1,
        };

        let (sparse, n_sparse) = make_two_blobs(4);
        let (dense, n_dense) = make_two_blobs(15);
        let res_sparse = stable_mapper(
            &sparse,
            n_sparse,
            2,
            |p| p[0],
            &StableMapperConfig::new(mapper.clone(), 24, 11),
        )
        .expect("sparse");
        let res_dense = stable_mapper(
            &dense,
            n_dense,
            2,
            |p| p[0],
            &StableMapperConfig::new(mapper, 24, 11),
        )
        .expect("dense");

        for s in [
            res_sparse.stability,
            res_sparse.component_confidence,
            res_sparse.loop_confidence,
            res_dense.stability,
        ] {
            assert!((0.0..=1.0).contains(&s), "score {s} out of [0,1]");
        }
        // The well-separated, denser cloud reproduces its signature most of the time.
        assert!(
            res_dense.stability >= 0.5,
            "dense clean data should be stable, got {}",
            res_dense.stability
        );
        // Bootstrap signatures were actually recorded.
        assert_eq!(res_dense.bootstrap_signatures.len(), 24);
    }

    // (d) A cover with a single interval collapses to one node.
    #[test]
    fn single_interval_single_node() {
        // One tight blob.
        let n = 8usize;
        let pts: Vec<f64> = (0..n).flat_map(|i| vec![i as f64 / 20.0, 0.0]).collect();
        let cfg = StableMapperConfig::new(
            MapperConfig {
                n_intervals: 1,
                overlap_frac: 0.3,
                cluster_eps: 5.0,
                min_pts: 1,
            },
            8,
            1,
        );
        let res = stable_mapper(&pts, n, 2, |p| p[0], &cfg).expect("stable mapper");
        assert_eq!(res.graph.n_nodes(), 1, "single interval ⇒ single node");
        assert_eq!(res.n_components, 1);
        assert_eq!(res.betti_1, 0);
    }

    // (e) Bad cover parameters are rejected.
    #[test]
    fn bad_cover_params_error() {
        let pts = circle_points(10);
        let base = MapperConfig {
            n_intervals: 4,
            overlap_frac: 0.3,
            cluster_eps: 0.7,
            min_pts: 1,
        };

        // n_intervals == 0.
        let mut c = base.clone();
        c.n_intervals = 0;
        assert!(stable_mapper(&pts, 10, 2, |p| p[0], &StableMapperConfig::new(c, 8, 0)).is_err());

        // overlap == 0 (not in the open interval).
        let mut c = base.clone();
        c.overlap_frac = 0.0;
        assert!(stable_mapper(&pts, 10, 2, |p| p[0], &StableMapperConfig::new(c, 8, 0)).is_err());

        // overlap == 1.
        let mut c = base.clone();
        c.overlap_frac = 1.0;
        assert!(stable_mapper(&pts, 10, 2, |p| p[0], &StableMapperConfig::new(c, 8, 0)).is_err());

        // overlap negative.
        let mut c = base.clone();
        c.overlap_frac = -0.2;
        assert!(stable_mapper(&pts, 10, 2, |p| p[0], &StableMapperConfig::new(c, 8, 0)).is_err());

        // overlap > 1.
        let mut c = base.clone();
        c.overlap_frac = 1.5;
        assert!(stable_mapper(&pts, 10, 2, |p| p[0], &StableMapperConfig::new(c, 8, 0)).is_err());

        // n_bootstrap == 0.
        assert!(
            stable_mapper(&pts, 10, 2, |p| p[0], &StableMapperConfig::new(base, 0, 0)).is_err()
        );
    }
}
