//! ES-HyperNEAT: Evolvable Substrate HyperNEAT.
//!
//! Reference: Risi & Stanley, "An Enhanced Hypercube-Based Encoding for Evolving the
//! Placement, Density, and Connectivity of Neurons", Artificial Life 18(4):331-363, 2012.
//!
//! ES-HyperNEAT extends HyperNEAT by automatically discovering the placement and
//! connectivity of hidden substrate neurons from CPPN output geometry, rather than
//! requiring the user to specify a fixed hidden-layer geometry.  The CPPN is queried
//! on a resolution×resolution grid; high-magnitude regions indicate where hidden
//! neurons should be placed.

#![allow(clippy::needless_range_loop)]

use super::hyperneat::{
    CppnConfig, CppnWeights, Substrate, cppn_forward_pub as cppn_forward, hyperneat_query_weights,
};
use crate::{EvolError, EvolResult, handle::LcgRng};

// ─── ES-HyperNEAT configuration ─────────────────────────────────────────────

/// Configuration for ES-HyperNEAT.
///
/// Extends `HyperNeatConfig` with substrate-discovery parameters.
#[derive(Debug, Clone)]
pub struct EsHyperNeatConfig {
    /// CPPN architecture.
    pub cppn: CppnConfig,
    /// Fixed input and output layer coordinates — the base substrate.
    /// The hidden layer is discovered automatically.
    pub input_coords: Vec<(f64, f64)>,
    /// Fixed output coordinates.
    pub output_coords: Vec<(f64, f64)>,
    /// Minimum CPPN response magnitude required to express a connection
    /// (same role as `expression_threshold` in HyperNEAT).
    pub expression_threshold: f64,
    /// Number of probe points per axis in the discovery grid (default 11).
    pub resolution: usize,
    /// Minimum CPPN response to place a hidden node at a probed location (default 0.1).
    pub placement_threshold: f64,
    /// Minimum |weight| to keep a connection after substrate query (default 0.05).
    pub prune_threshold: f64,
    /// Number of (μ+λ)-ES generations.
    pub n_evol_iters: usize,
    /// Initial perturbation standard deviation.
    pub sigma_init: f64,
    /// Per-generation multiplicative decay of sigma.
    pub sigma_decay: f64,
    /// Random seed.
    pub seed: u64,
}

impl EsHyperNeatConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// Returns `InvalidParameter` for any out-of-range value.
    pub fn validate(&self) -> EvolResult<()> {
        if self.input_coords.is_empty() {
            return Err(EvolError::InvalidParameter(
                "EsHyperNeatConfig: input_coords must not be empty".into(),
            ));
        }
        if self.output_coords.is_empty() {
            return Err(EvolError::InvalidParameter(
                "EsHyperNeatConfig: output_coords must not be empty".into(),
            ));
        }
        if self.resolution < 2 {
            return Err(EvolError::InvalidParameter(
                "EsHyperNeatConfig: resolution must be >= 2".into(),
            ));
        }
        if self.expression_threshold < 0.0 {
            return Err(EvolError::InvalidParameter(
                "expression_threshold must be >= 0".into(),
            ));
        }
        if self.placement_threshold < 0.0 {
            return Err(EvolError::InvalidParameter(
                "placement_threshold must be >= 0".into(),
            ));
        }
        if self.prune_threshold < 0.0 {
            return Err(EvolError::InvalidParameter(
                "prune_threshold must be >= 0".into(),
            ));
        }
        if self.n_evol_iters == 0 {
            return Err(EvolError::InvalidParameter(
                "n_evol_iters must be >= 1".into(),
            ));
        }
        if self.sigma_init <= 0.0 {
            return Err(EvolError::InvalidParameter("sigma_init must be > 0".into()));
        }
        if !(0.0..=1.0).contains(&self.sigma_decay) {
            return Err(EvolError::InvalidParameter(
                "sigma_decay must be in [0, 1]".into(),
            ));
        }
        Ok(())
    }

    /// Build a default configuration for quick tests.
    ///
    /// Uses 3 inputs at y=-0.5, 2 outputs at y=0.5, resolution 7.
    pub fn default_small(cppn: CppnConfig) -> Self {
        let n_in = 3usize;
        let n_out = 2usize;
        let input_coords: Vec<(f64, f64)> = (0..n_in)
            .map(|i| (-1.0 + 2.0 * i as f64 / (n_in - 1) as f64, -0.5))
            .collect();
        let output_coords: Vec<(f64, f64)> = (0..n_out)
            .map(|i| (-1.0 + 2.0 * i as f64 / (n_out - 1) as f64, 0.5))
            .collect();
        Self {
            cppn,
            input_coords,
            output_coords,
            expression_threshold: 0.1,
            resolution: 7,
            placement_threshold: 0.1,
            prune_threshold: 0.05,
            n_evol_iters: 10,
            sigma_init: 0.5,
            sigma_decay: 0.95,
            seed: 0,
        }
    }
}

// ─── Discovered substrate wrapper ────────────────────────────────────────────

/// A substrate discovered by ES-HyperNEAT.
///
/// Unlike the fixed `Substrate`, the hidden layer is populated by the
/// discovery algorithm rather than the user.
#[derive(Debug, Clone)]
pub struct EsSubstrate {
    /// Input coordinates (fixed, copied from config).
    pub input_coords: Vec<(f64, f64)>,
    /// Hidden coordinates discovered from the CPPN geometry.
    pub hidden_coords: Vec<(f64, f64)>,
    /// Output coordinates (fixed, copied from config).
    pub output_coords: Vec<(f64, f64)>,
}

impl EsSubstrate {
    /// Convert to a plain `Substrate` for use with standard HyperNEAT helpers.
    ///
    /// If hidden_coords is empty (nothing was discovered), inserts a single
    /// dummy node at the origin so downstream operations remain valid.
    #[must_use]
    pub fn to_substrate(&self) -> Substrate {
        let hidden = if self.hidden_coords.is_empty() {
            vec![(0.0, 0.0)]
        } else {
            self.hidden_coords.clone()
        };
        Substrate {
            input_coords: self.input_coords.clone(),
            hidden_coords: hidden,
            output_coords: self.output_coords.clone(),
        }
    }

    /// Total connection count (input→hidden + hidden→output).
    #[must_use]
    pub fn n_weights(&self) -> usize {
        let n_hid = if self.hidden_coords.is_empty() {
            1
        } else {
            self.hidden_coords.len()
        };
        self.input_coords.len() * n_hid + n_hid * self.output_coords.len()
    }
}

// ─── ES-HyperNEAT state ────────────────────────────────────────────────────

/// State produced by an ES-HyperNEAT run.
#[derive(Debug, Clone)]
pub struct EsHyperNeatState {
    /// Best CPPN weights found during evolution.
    pub cppn_weights: CppnWeights,
    /// Adaptive substrate discovered from the best CPPN.
    pub discovered_substrate: EsSubstrate,
    /// Weight matrix for the discovered substrate network.
    pub substrate_weights: Vec<f64>,
    /// Best fitness seen during evolution.
    pub best_fitness: f64,
    /// Number of generations completed.
    pub generation: usize,
}

// ─── Hidden-node discovery ────────────────────────────────────────────────────

/// Discover hidden-node positions from the CPPN's output geometry.
///
/// The algorithm probes a `resolution × resolution` grid in `[-1, 1]²` and
/// considers two passes:
/// 1. For each input coord (anchor at source) → grid point (candidate target):
///    if |CPPN(input_anchor, grid_pt)| > threshold → candidate.
/// 2. For each grid point (candidate source) → output coord (anchor at target):
///    if |CPPN(grid_pt, output_anchor)| > threshold → candidate.
///
/// Candidate points from both passes are merged and deduplicated by merging
/// any pair of candidates within `2/resolution` Euclidean distance.
///
/// Returns the discovered hidden-node coordinates.
pub fn discover_hidden_nodes(
    cppn: &CppnWeights,
    cppn_cfg: &CppnConfig,
    input_coords: &[(f64, f64)],
    output_coords: &[(f64, f64)],
    resolution: usize,
    threshold: f64,
) -> Vec<(f64, f64)> {
    // Build the grid of probe points in [-1, 1]²
    let grid: Vec<(f64, f64)> = grid_points(resolution);
    let merge_radius = 2.0 / resolution as f64;

    let mut candidates: Vec<(f64, f64)> = Vec::new();

    // Pass 1: input-anchor → grid candidate (discovering hidden nodes
    // that receive strong signals from the input layer).
    for &(xs, ys) in input_coords {
        for &(xt, yt) in &grid {
            let raw = cppn_forward(cppn, cppn_cfg, xs, ys, xt, yt);
            if raw.abs() > threshold {
                candidates.push((xt, yt));
            }
        }
    }

    // Pass 2: grid candidate → output-anchor (discovering hidden nodes
    // that send strong signals to the output layer).
    for &(xs, ys) in &grid {
        for &(xt, yt) in output_coords {
            let raw = cppn_forward(cppn, cppn_cfg, xs, ys, xt, yt);
            if raw.abs() > threshold {
                candidates.push((xs, ys));
            }
        }
    }

    // Deduplicate by merging positions within merge_radius.
    deduplicate_candidates(candidates, merge_radius)
}

/// Build a flat list of `resolution × resolution` grid points spanning `[-1,1]²`.
fn grid_points(resolution: usize) -> Vec<(f64, f64)> {
    let mut pts = Vec::with_capacity(resolution * resolution);
    for i in 0..resolution {
        let y = if resolution == 1 {
            0.0
        } else {
            -1.0 + 2.0 * i as f64 / (resolution - 1) as f64
        };
        for j in 0..resolution {
            let x = if resolution == 1 {
                0.0
            } else {
                -1.0 + 2.0 * j as f64 / (resolution - 1) as f64
            };
            pts.push((x, y));
        }
    }
    pts
}

/// Merge any two candidates within `radius` Euclidean distance using a
/// greedy scan: for each candidate, if it is close to an already-accepted
/// representative, skip it; otherwise add it as a new representative.
fn deduplicate_candidates(candidates: Vec<(f64, f64)>, radius: f64) -> Vec<(f64, f64)> {
    let mut result: Vec<(f64, f64)> = Vec::new();
    let r2 = radius * radius;
    for (cx, cy) in candidates {
        let too_close = result.iter().any(|&(rx, ry)| {
            let dx = cx - rx;
            let dy = cy - ry;
            dx * dx + dy * dy <= r2
        });
        if !too_close {
            result.push((cx, cy));
        }
    }
    result
}

// ─── Substrate discovery (public API) ────────────────────────────────────────

/// Discover the adaptive substrate from the CPPN and a base (input/output) substrate.
///
/// The hidden layer in `base_substrate` is **ignored** — it is replaced by the
/// set of nodes discovered by probing the CPPN on the `resolution × resolution` grid.
///
/// # Returns
/// A new `EsSubstrate` with the discovered hidden layer.
pub fn es_hyperneat_discover_substrate(
    cppn: &CppnWeights,
    cppn_cfg: &CppnConfig,
    base_substrate: &Substrate,
    resolution: usize,
    placement_threshold: f64,
) -> EsSubstrate {
    let hidden_coords = discover_hidden_nodes(
        cppn,
        cppn_cfg,
        &base_substrate.input_coords,
        &base_substrate.output_coords,
        resolution,
        placement_threshold,
    );
    EsSubstrate {
        input_coords: base_substrate.input_coords.clone(),
        hidden_coords,
        output_coords: base_substrate.output_coords.clone(),
    }
}

// ─── Substrate weight query with pruning ────────────────────────────────────

/// Query CPPN weights for the ES-discovered substrate and prune weak connections.
///
/// Builds the full weight matrix for the discovered substrate via the standard
/// HyperNEAT geometric query, then zeros out any connection with
/// `|weight| < prune_threshold`.
fn es_query_and_prune(
    cppn: &CppnWeights,
    cppn_cfg: &CppnConfig,
    substrate: &EsSubstrate,
    expression_threshold: f64,
    prune_threshold: f64,
) -> Vec<f64> {
    let plain = substrate.to_substrate();
    let mut weights = hyperneat_query_weights(cppn, cppn_cfg, &plain, expression_threshold);
    for w in weights.iter_mut() {
        if w.abs() < prune_threshold {
            *w = 0.0;
        }
    }
    weights
}

// ─── Forward pass through discovered substrate ───────────────────────────────

/// Run inference through the ES-discovered substrate network.
///
/// Architecture: input → (tanh) hidden → (tanh) output.
///
/// `state.substrate_weights` must match `state.discovered_substrate.n_weights()`.
///
/// # Errors
/// Returns `DimensionMismatch` if `x.len()` does not match the input-layer size.
pub fn es_hyperneat_forward(state: &EsHyperNeatState, x: &[f64]) -> EvolResult<Vec<f64>> {
    let n_in = state.discovered_substrate.input_coords.len();
    if x.len() != n_in {
        return Err(EvolError::DimensionMismatch {
            expected: n_in,
            got: x.len(),
        });
    }

    let plain = state.discovered_substrate.to_substrate();
    let n_hid = plain.hidden_coords.len();
    let n_out = plain.output_coords.len();
    let sw = &state.substrate_weights;

    let expected_len = n_in * n_hid + n_hid * n_out;
    if sw.len() != expected_len {
        return Err(EvolError::DimensionMismatch {
            expected: expected_len,
            got: sw.len(),
        });
    }

    // Input → hidden (tanh)
    let mut hidden = vec![0.0f64; n_hid];
    for h in 0..n_hid {
        let mut pre = 0.0;
        for i in 0..n_in {
            pre += x[i] * sw[i * n_hid + h];
        }
        hidden[h] = pre.tanh();
    }

    // Hidden → output (tanh)
    let ih_end = n_in * n_hid;
    let mut output = vec![0.0f64; n_out];
    for o in 0..n_out {
        let mut pre = 0.0;
        for h in 0..n_hid {
            pre += hidden[h] * sw[ih_end + h * n_out + o];
        }
        output[o] = pre.tanh();
    }

    Ok(output)
}

// ─── (μ+λ)-ES internals ──────────────────────────────────────────────────────

const MU: usize = 5;
const LAMBDA: usize = 20;

/// Perturb a flat parameter vector with Gaussian noise of std `sigma`.
fn perturb(params: &[f64], sigma: f64, rng: &mut LcgRng) -> Vec<f64> {
    params
        .iter()
        .map(|&p| p + rng.next_normal() * sigma)
        .collect()
}

/// Evaluate a flat CPPN parameter vector.
///
/// Returns `(fitness, es_substrate, substrate_weights)`.
fn evaluate_es_params(
    flat: &[f64],
    cfg: &EsHyperNeatConfig,
    fitness_fn: &impl Fn(&[f64], &EsSubstrate) -> f64,
) -> EvolResult<(f64, EsSubstrate, Vec<f64>)> {
    let cppn = CppnWeights::from_flat(flat, cfg.cppn.n_hidden)?;

    // Build a base substrate (hidden layer will be overwritten by discovery)
    let base = Substrate {
        input_coords: cfg.input_coords.clone(),
        hidden_coords: vec![(0.0, 0.0)], // placeholder
        output_coords: cfg.output_coords.clone(),
    };

    // Discover the hidden substrate
    let es_sub = es_hyperneat_discover_substrate(
        &cppn,
        &cfg.cppn,
        &base,
        cfg.resolution,
        cfg.placement_threshold,
    );

    // Query weights with pruning
    let sw = es_query_and_prune(
        &cppn,
        &cfg.cppn,
        &es_sub,
        cfg.expression_threshold,
        cfg.prune_threshold,
    );

    let fitness = fitness_fn(&sw, &es_sub);
    Ok((fitness, es_sub, sw))
}

// ─── Public API: run ES-HyperNEAT ────────────────────────────────────────────

/// Run ES-HyperNEAT via (μ+λ)-ES to optimise CPPN weights and discover the substrate.
///
/// `fitness_fn(substrate_weights, substrate) -> f64` — higher is better.
///
/// At each generation:
/// 1. Generate λ offspring from the μ best parents via Gaussian perturbation.
/// 2. For each offspring: discover the adaptive substrate, query weights, evaluate fitness.
/// 3. Select the top μ individuals (μ+λ selection).
///
/// # Errors
/// Returns `InvalidParameter` if `cfg.validate()` fails.
/// Returns `EmptyPopulation` if the initial population cannot be evaluated.
pub fn es_hyperneat_run(
    fitness_fn: impl Fn(&[f64], &EsSubstrate) -> f64,
    cfg: &EsHyperNeatConfig,
) -> EvolResult<EsHyperNeatState> {
    cfg.validate()?;

    let mut rng = LcgRng::new(cfg.seed);
    let n_params = cfg.cppn.n_params();

    // Initialise μ parents randomly
    let mut parents: Vec<Vec<f64>> = (0..MU)
        .map(|_| {
            (0..n_params)
                .map(|_| rng.next_normal() * cfg.sigma_init)
                .collect()
        })
        .collect();

    // Evaluate initial parents
    let parent_results: Vec<Option<(f64, EsSubstrate, Vec<f64>)>> = parents
        .iter()
        .map(|p| {
            evaluate_es_params(p, cfg, &fitness_fn)
                .ok()
                .filter(|(f, _, _)| f.is_finite())
        })
        .collect();

    // Track best overall
    let best_init_idx = parent_results
        .iter()
        .enumerate()
        .filter_map(|(i, r)| r.as_ref().map(|(f, _, _)| (i, *f)))
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .ok_or(EvolError::EmptyPopulation)?;

    let best_init_fit = parent_results[best_init_idx]
        .as_ref()
        .map(|(f, _, _)| *f)
        .unwrap_or(f64::NEG_INFINITY);

    let mut best_flat = parents[best_init_idx].clone();
    let mut best_fitness = best_init_fit;
    let mut sigma = cfg.sigma_init;

    // Extract parent fitnesses for selection
    let mut parent_fitness: Vec<f64> = parent_results
        .iter()
        .map(|r| r.as_ref().map(|(f, _, _)| *f).unwrap_or(f64::NEG_INFINITY))
        .collect();

    for _gen in 0..cfg.n_evol_iters {
        // Generate λ offspring
        let mut offspring: Vec<(Vec<f64>, f64)> = Vec::with_capacity(LAMBDA);
        for _ in 0..LAMBDA {
            let parent_idx = rng.next_usize(MU);
            let child = perturb(&parents[parent_idx], sigma, &mut rng);
            let fit = evaluate_es_params(&child, cfg, &fitness_fn)
                .ok()
                .filter(|(f, _, _)| f.is_finite())
                .map(|(f, _, _)| f)
                .unwrap_or(f64::NEG_INFINITY);
            offspring.push((child, fit));
        }

        // μ+λ selection
        let mut combined: Vec<(Vec<f64>, f64)> = parents
            .iter()
            .zip(parent_fitness.iter())
            .map(|(p, &f)| (p.clone(), f))
            .collect();
        combined.extend(offspring);
        combined
            .sort_by(|(_, fa), (_, fb)| fb.partial_cmp(fa).unwrap_or(std::cmp::Ordering::Equal));
        combined.truncate(MU);

        // Update best
        if combined[0].1 > best_fitness {
            best_fitness = combined[0].1;
            best_flat = combined[0].0.clone();
        }

        parents = combined.iter().map(|(p, _)| p.clone()).collect();
        parent_fitness = combined.iter().map(|(_, f)| *f).collect();

        // Sigma annealing
        sigma *= cfg.sigma_decay;
        sigma = sigma.max(1e-8);
    }

    // Final evaluation of the best CPPN to get the discovered substrate + weights
    let best_cppn = CppnWeights::from_flat(&best_flat, cfg.cppn.n_hidden)?;
    let base = Substrate {
        input_coords: cfg.input_coords.clone(),
        hidden_coords: vec![(0.0, 0.0)],
        output_coords: cfg.output_coords.clone(),
    };
    let discovered_substrate = es_hyperneat_discover_substrate(
        &best_cppn,
        &cfg.cppn,
        &base,
        cfg.resolution,
        cfg.placement_threshold,
    );
    let substrate_weights = es_query_and_prune(
        &best_cppn,
        &cfg.cppn,
        &discovered_substrate,
        cfg.expression_threshold,
        cfg.prune_threshold,
    );

    Ok(EsHyperNeatState {
        cppn_weights: best_cppn,
        discovered_substrate,
        substrate_weights,
        best_fitness,
        generation: cfg.n_evol_iters,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::hyperneat::{CppnActivation, CppnConfig, CppnWeights};
    use super::*;

    fn default_cppn_cfg() -> CppnConfig {
        CppnConfig::new(
            4,
            vec![
                CppnActivation::Tanh,
                CppnActivation::Gaussian,
                CppnActivation::Sine,
                CppnActivation::Sigmoid,
            ],
        )
        .expect("value should be present")
    }

    fn default_es_cfg() -> EsHyperNeatConfig {
        EsHyperNeatConfig::default_small(default_cppn_cfg())
    }

    type CoordPair = (Vec<(f64, f64)>, Vec<(f64, f64)>);

    fn make_input_output_coords(n_in: usize, n_out: usize) -> CoordPair {
        let input: Vec<(f64, f64)> = (0..n_in)
            .map(|i| {
                let x = if n_in == 1 {
                    0.0
                } else {
                    -1.0 + 2.0 * i as f64 / (n_in - 1) as f64
                };
                (x, -0.5)
            })
            .collect();
        let output: Vec<(f64, f64)> = (0..n_out)
            .map(|i| {
                let x = if n_out == 1 {
                    0.0
                } else {
                    -1.0 + 2.0 * i as f64 / (n_out - 1) as f64
                };
                (x, 0.5)
            })
            .collect();
        (input, output)
    }

    // ── 1: validate catches bad resolution ───────────────────────────────────

    #[test]
    fn validate_bad_resolution() {
        let mut cfg = default_es_cfg();
        cfg.resolution = 1;
        assert!(cfg.validate().is_err());
    }

    // ── 2: validate catches bad sigma_init ───────────────────────────────────

    #[test]
    fn validate_bad_sigma_init() {
        let mut cfg = default_es_cfg();
        cfg.sigma_init = 0.0;
        assert!(cfg.validate().is_err());
    }

    // ── 3: validate catches bad sigma_decay ──────────────────────────────────

    #[test]
    fn validate_bad_sigma_decay() {
        let mut cfg = default_es_cfg();
        cfg.sigma_decay = 1.5;
        assert!(cfg.validate().is_err());
    }

    // ── 4: validate catches empty input_coords ───────────────────────────────

    #[test]
    fn validate_empty_input_coords() {
        let mut cfg = default_es_cfg();
        cfg.input_coords = Vec::new();
        assert!(cfg.validate().is_err());
    }

    // ── 5: validate catches empty output_coords ──────────────────────────────

    #[test]
    fn validate_empty_output_coords() {
        let mut cfg = default_es_cfg();
        cfg.output_coords = Vec::new();
        assert!(cfg.validate().is_err());
    }

    // ── 6: grid_points produces correct count ────────────────────────────────

    #[test]
    fn grid_points_count() {
        let pts = grid_points(5);
        assert_eq!(pts.len(), 25, "5×5 grid should have 25 points");
    }

    // ── 7: grid_points span [-1, 1]² ─────────────────────────────────────────

    #[test]
    fn grid_points_range() {
        let pts = grid_points(7);
        for (x, y) in pts {
            assert!((-1.0 - 1e-10..=1.0 + 1e-10).contains(&x));
            assert!((-1.0 - 1e-10..=1.0 + 1e-10).contains(&y));
        }
    }

    // ── 8: deduplicate removes nearby points ─────────────────────────────────

    #[test]
    fn deduplication_works() {
        let candidates = vec![(0.0, 0.0), (0.01, 0.0), (0.9, 0.9), (0.91, 0.9)];
        let result = deduplicate_candidates(candidates, 0.2);
        // The (0.0,0.0) and (0.01,0.0) are within 0.2 → only 1 kept.
        // The (0.9,0.9) and (0.91,0.9) are within 0.2 → only 1 kept.
        assert_eq!(result.len(), 2);
    }

    // ── 9: discover_hidden_nodes returns finite coords ────────────────────────

    #[test]
    fn discover_hidden_nodes_finite() {
        let mut rng = LcgRng::new(42);
        let cppn_cfg = default_cppn_cfg();
        let cppn = CppnWeights::random(cppn_cfg.n_hidden, 1.0, &mut rng);
        let (input_coords, output_coords) = make_input_output_coords(3, 2);
        let hidden = discover_hidden_nodes(&cppn, &cppn_cfg, &input_coords, &output_coords, 7, 0.1);
        for (x, y) in hidden {
            assert!(x.is_finite() && y.is_finite());
        }
    }

    // ── 10: es_hyperneat_discover_substrate produces valid EsSubstrate ────────

    #[test]
    fn discover_substrate_valid() {
        let mut rng = LcgRng::new(77);
        let cppn_cfg = default_cppn_cfg();
        let cppn = CppnWeights::random(cppn_cfg.n_hidden, 0.5, &mut rng);
        let base = Substrate {
            input_coords: vec![(-1.0, -0.5), (0.0, -0.5), (1.0, -0.5)],
            hidden_coords: vec![(0.0, 0.0)],
            output_coords: vec![(-0.5, 0.5), (0.5, 0.5)],
        };
        let es_sub = es_hyperneat_discover_substrate(&cppn, &cppn_cfg, &base, 5, 0.05);
        assert_eq!(es_sub.input_coords, base.input_coords);
        assert_eq!(es_sub.output_coords, base.output_coords);
        // hidden may be empty or non-empty, just check it is finite
        for (x, y) in &es_sub.hidden_coords {
            assert!(x.is_finite() && y.is_finite());
        }
    }

    // ── 11: to_substrate inserts dummy hidden when empty ─────────────────────

    #[test]
    fn to_substrate_empty_hidden_fallback() {
        let es_sub = EsSubstrate {
            input_coords: vec![(0.0, -0.5)],
            hidden_coords: Vec::new(),
            output_coords: vec![(0.0, 0.5)],
        };
        let plain = es_sub.to_substrate();
        assert_eq!(plain.hidden_coords.len(), 1);
        assert_eq!(plain.hidden_coords[0], (0.0, 0.0));
    }

    // ── 12: es_hyperneat_forward output shape and tanh range ─────────────────

    #[test]
    fn es_forward_shape_and_range() {
        let mut rng = LcgRng::new(13);
        let cppn_cfg = default_cppn_cfg();
        let cppn = CppnWeights::random(cppn_cfg.n_hidden, 0.5, &mut rng);
        let base = Substrate {
            input_coords: vec![(-1.0, -0.5), (0.0, -0.5), (1.0, -0.5)],
            hidden_coords: vec![(0.0, 0.0)],
            output_coords: vec![(-0.5, 0.5), (0.5, 0.5)],
        };
        let es_sub = es_hyperneat_discover_substrate(&cppn, &cppn_cfg, &base, 5, 0.05);
        let sw = es_query_and_prune(&cppn, &cppn_cfg, &es_sub, 0.1, 0.05);
        let state = EsHyperNeatState {
            cppn_weights: cppn,
            discovered_substrate: es_sub,
            substrate_weights: sw,
            best_fitness: 0.0,
            generation: 0,
        };
        let x = vec![0.1, -0.2, 0.3];
        let out = es_hyperneat_forward(&state, &x).expect("es_hyperneat_forward should succeed");
        assert_eq!(out.len(), 2);
        for &v in &out {
            assert!(v.abs() <= 1.0 + 1e-10, "output must be in [-1,1]");
        }
    }

    // ── 13: es_hyperneat_forward dimension mismatch error ────────────────────

    #[test]
    fn es_forward_dim_mismatch() {
        let mut rng = LcgRng::new(99);
        let cppn_cfg = default_cppn_cfg();
        let cppn = CppnWeights::random(cppn_cfg.n_hidden, 0.3, &mut rng);
        let base = Substrate {
            input_coords: vec![(-1.0, -0.5), (0.0, -0.5), (1.0, -0.5)],
            hidden_coords: vec![(0.0, 0.0)],
            output_coords: vec![(-0.5, 0.5), (0.5, 0.5)],
        };
        let es_sub = es_hyperneat_discover_substrate(&cppn, &cppn_cfg, &base, 5, 0.05);
        let sw = es_query_and_prune(&cppn, &cppn_cfg, &es_sub, 0.1, 0.05);
        let state = EsHyperNeatState {
            cppn_weights: cppn,
            discovered_substrate: es_sub,
            substrate_weights: sw,
            best_fitness: 0.0,
            generation: 0,
        };
        // Wrong input length (4 instead of 3)
        let result = es_hyperneat_forward(&state, &[0.0, 0.1, 0.2, 0.3]);
        assert!(result.is_err());
    }

    // ── 14: es_hyperneat_run completes without error ──────────────────────────

    #[test]
    fn es_hyperneat_run_completes() {
        let cfg = EsHyperNeatConfig {
            n_evol_iters: 5,
            ..default_es_cfg()
        };
        let state =
            es_hyperneat_run(|_sw, _sub| 1.0, &cfg).expect("es_hyperneat_run should succeed");
        assert_eq!(state.generation, 5);
        assert!(state.best_fitness.is_finite());
    }

    // ── 15: es_hyperneat_run best_fitness is finite ───────────────────────────

    #[test]
    fn es_hyperneat_run_fitness_finite() {
        let cfg = EsHyperNeatConfig {
            n_evol_iters: 8,
            sigma_init: 0.3,
            sigma_decay: 0.9,
            seed: 7,
            ..default_es_cfg()
        };
        let state = es_hyperneat_run(|sw, _sub| -sw.iter().map(|w| w * w).sum::<f64>(), &cfg)
            .expect("value should be present");
        assert!(state.best_fitness.is_finite());
    }

    // ── 16: substrate_weights length matches discovered substrate ─────────────

    #[test]
    fn substrate_weights_length_matches() {
        let cfg = EsHyperNeatConfig {
            n_evol_iters: 3,
            ..default_es_cfg()
        };
        let state =
            es_hyperneat_run(|_sw, _sub| 0.0, &cfg).expect("es_hyperneat_run should succeed");
        let expected = state.discovered_substrate.n_weights();
        assert_eq!(
            state.substrate_weights.len(),
            expected,
            "substrate_weights length must match n_weights"
        );
    }

    // ── 17: prune_threshold zeros out small weights ───────────────────────────

    #[test]
    fn pruning_zeros_small_weights() {
        let mut rng = LcgRng::new(55);
        let cppn_cfg = default_cppn_cfg();
        // Use a CPPN with tiny weights so outputs are small
        let cppn = CppnWeights::random(cppn_cfg.n_hidden, 0.01, &mut rng);
        let es_sub = EsSubstrate {
            input_coords: vec![(-1.0, -0.5), (0.0, -0.5), (1.0, -0.5)],
            hidden_coords: vec![(0.0, 0.0)],
            output_coords: vec![(-0.5, 0.5), (0.5, 0.5)],
        };
        // Set prune_threshold very high to zero everything
        let sw = es_query_and_prune(&cppn, &cppn_cfg, &es_sub, 0.0, 1e9);
        assert!(sw.iter().all(|&w| w == 0.0), "all weights should be pruned");
    }
}
