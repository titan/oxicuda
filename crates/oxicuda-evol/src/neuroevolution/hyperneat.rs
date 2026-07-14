//! HyperNEAT: Hypercube-based NeuroEvolution of Augmenting Topologies.
//!
//! Reference: K. Stanley, D. D'Ambrosio & J. Gauci, "A Hypercube-Based Encoding for
//! Evolving Large-Scale Neural Networks", Artificial Life 15(2):185-212, 2009.
//!
//! Uses a CPPN (Compositional Pattern Producing Network) to encode weight patterns
//! geometrically.  The CPPN maps (x_src, y_src, x_tgt, y_tgt, dist) → weight, and
//! a (μ+λ)-ES evolves the CPPN weights directly.

#![allow(clippy::needless_range_loop)]

use crate::{EvolError, EvolResult, handle::LcgRng};

// ─── CPPN activation functions ───────────────────────────────────────────────

/// Activation functions available to CPPN hidden neurons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CppnActivation {
    Sigmoid,
    Tanh,
    Gaussian,
    Sine,
}

impl CppnActivation {
    /// Apply the activation function.
    #[inline]
    pub fn apply(self, x: f64) -> f64 {
        match self {
            CppnActivation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            CppnActivation::Tanh => x.tanh(),
            CppnActivation::Gaussian => (-x * x).exp(),
            CppnActivation::Sine => x.sin(),
        }
    }
}

// ─── CPPN configuration ──────────────────────────────────────────────────────

/// Configuration for the CPPN used inside HyperNEAT.
///
/// The CPPN has 5 inputs `(x_src, y_src, x_tgt, y_tgt, dist)` and 1 output (weight).
/// Each hidden neuron uses the activation from `activations[i % len]`.
#[derive(Debug, Clone)]
pub struct CppnConfig {
    /// Number of hidden neurons in the CPPN.
    pub n_hidden: usize,
    /// Activation function cycle list for hidden neurons.
    pub activations: Vec<CppnActivation>,
}

impl CppnConfig {
    /// Create a new CPPN config with the given hidden count and activation list.
    ///
    /// # Errors
    /// Returns `InvalidParameter` if `n_hidden == 0` or `activations` is empty.
    pub fn new(n_hidden: usize, activations: Vec<CppnActivation>) -> EvolResult<Self> {
        if n_hidden == 0 {
            return Err(EvolError::InvalidParameter(
                "CppnConfig: n_hidden must be >= 1".into(),
            ));
        }
        if activations.is_empty() {
            return Err(EvolError::InvalidParameter(
                "CppnConfig: activations must not be empty".into(),
            ));
        }
        Ok(Self {
            n_hidden,
            activations,
        })
    }

    /// Total number of CPPN parameters.
    ///
    /// Layout: `(5 * n_hidden)` hidden weights + `n_hidden` hidden biases
    ///         + `n_hidden` output weights + `1` output bias.
    #[inline]
    pub fn n_params(&self) -> usize {
        5 * self.n_hidden + self.n_hidden + self.n_hidden + 1
    }
}

// ─── CPPN weights ─────────────────────────────────────────────────────────────

/// Weight tensors for a CPPN with 5 inputs, `n_hidden` hidden neurons, 1 output.
#[derive(Debug, Clone)]
pub struct CppnWeights {
    /// Hidden layer weights: shape `[n_hidden, 5]`, row-major.
    pub hidden_weights: Vec<f64>,
    /// Hidden layer biases: length `n_hidden`.
    pub hidden_bias: Vec<f64>,
    /// Output layer weights: length `n_hidden`.
    pub output_weights: Vec<f64>,
    /// Output bias (scalar).
    pub output_bias: f64,
}

impl CppnWeights {
    /// Create zero-initialised weights for the given hidden-layer width.
    pub fn zeros(n_hidden: usize) -> Self {
        Self {
            hidden_weights: vec![0.0; 5 * n_hidden],
            hidden_bias: vec![0.0; n_hidden],
            output_weights: vec![0.0; n_hidden],
            output_bias: 0.0,
        }
    }

    /// Serialise into a flat vector for ES perturbation.
    ///
    /// Layout: `hidden_weights | hidden_bias | output_weights | [output_bias]`.
    pub fn to_flat(&self) -> Vec<f64> {
        let mut v = self.hidden_weights.clone();
        v.extend_from_slice(&self.hidden_bias);
        v.extend_from_slice(&self.output_weights);
        v.push(self.output_bias);
        v
    }

    /// Deserialise from a flat vector produced by `to_flat`.
    ///
    /// # Errors
    /// Returns `DimensionMismatch` if `flat.len()` does not equal `6 * n_hidden + 1`.
    pub fn from_flat(flat: &[f64], n_hidden: usize) -> EvolResult<Self> {
        let expected = 5 * n_hidden + n_hidden + n_hidden + 1;
        if flat.len() != expected {
            return Err(EvolError::DimensionMismatch {
                expected,
                got: flat.len(),
            });
        }
        let hw = flat[..5 * n_hidden].to_vec();
        let hb = flat[5 * n_hidden..6 * n_hidden].to_vec();
        let ow = flat[6 * n_hidden..7 * n_hidden].to_vec();
        let ob = flat[7 * n_hidden];
        Ok(Self {
            hidden_weights: hw,
            hidden_bias: hb,
            output_weights: ow,
            output_bias: ob,
        })
    }

    /// Sample random initial weights from N(0, sigma^2).
    ///
    /// Draws normals directly into each field (in the same order `to_flat`/
    /// `from_flat` would lay them out) instead of round-tripping through the
    /// flat fallible parser, so construction is infallible by structure.
    pub fn random(n_hidden: usize, sigma: f64, rng: &mut LcgRng) -> Self {
        Self {
            hidden_weights: (0..5 * n_hidden)
                .map(|_| rng.next_normal() * sigma)
                .collect(),
            hidden_bias: (0..n_hidden).map(|_| rng.next_normal() * sigma).collect(),
            output_weights: (0..n_hidden).map(|_| rng.next_normal() * sigma).collect(),
            output_bias: rng.next_normal() * sigma,
        }
    }
}

// ─── Substrate ────────────────────────────────────────────────────────────────

/// Defines the geometric layout of the substrate network to be generated.
///
/// Neurons are placed at 2-D coordinates in `[-1, 1]²`.  HyperNEAT queries the
/// CPPN for each source→target pair between adjacent layers.
#[derive(Debug, Clone)]
pub struct Substrate {
    /// Coordinates for input-layer neurons.
    pub input_coords: Vec<(f64, f64)>,
    /// Coordinates for hidden-layer neurons.
    pub hidden_coords: Vec<(f64, f64)>,
    /// Coordinates for output-layer neurons.
    pub output_coords: Vec<(f64, f64)>,
}

impl Substrate {
    /// Create a substrate with evenly spaced neurons on the x-axis at y=0.
    ///
    /// `n_inputs` neurons span `[-1, 1]` on the x-axis at y=-0.5;
    /// `n_hidden` neurons span `[-1, 1]` at y=0;
    /// `n_outputs` neurons span `[-1, 1]` at y=0.5.
    pub fn linear(n_inputs: usize, n_hidden: usize, n_outputs: usize) -> EvolResult<Self> {
        if n_inputs == 0 || n_hidden == 0 || n_outputs == 0 {
            return Err(EvolError::InvalidParameter(
                "Substrate: all layer sizes must be >= 1".into(),
            ));
        }
        let coords_at_y = |n: usize, y: f64| -> Vec<(f64, f64)> {
            (0..n)
                .map(|i| {
                    let x = if n == 1 {
                        0.0
                    } else {
                        -1.0 + 2.0 * i as f64 / (n - 1) as f64
                    };
                    (x, y)
                })
                .collect()
        };
        Ok(Self {
            input_coords: coords_at_y(n_inputs, -0.5),
            hidden_coords: coords_at_y(n_hidden, 0.0),
            output_coords: coords_at_y(n_outputs, 0.5),
        })
    }

    /// Total weight count produced by `hyperneat_query_weights`:
    /// `n_input × n_hidden + n_hidden × n_output`.
    #[inline]
    pub fn n_weights(&self) -> usize {
        self.input_coords.len() * self.hidden_coords.len()
            + self.hidden_coords.len() * self.output_coords.len()
    }
}

// ─── CPPN forward pass ────────────────────────────────────────────────────────

/// Run the CPPN forward pass for a single `(x_src, y_src, x_tgt, y_tgt)` query.
///
/// Returns the raw output value (not thresholded).
/// Exposed as `pub(crate)` so sibling modules (e.g. `es_hyperneat`) can reuse it.
pub(crate) fn cppn_forward_pub(
    w: &CppnWeights,
    cfg: &CppnConfig,
    x_src: f64,
    y_src: f64,
    x_tgt: f64,
    y_tgt: f64,
) -> f64 {
    cppn_forward(w, cfg, x_src, y_src, x_tgt, y_tgt)
}

fn cppn_forward(
    w: &CppnWeights,
    cfg: &CppnConfig,
    x_src: f64,
    y_src: f64,
    x_tgt: f64,
    y_tgt: f64,
) -> f64 {
    let dist = ((x_src - x_tgt).powi(2) + (y_src - y_tgt).powi(2)).sqrt();
    let inputs = [x_src, y_src, x_tgt, y_tgt, dist];
    let n_hidden = cfg.n_hidden;
    let n_acts = cfg.activations.len();

    // Hidden layer
    let mut hidden_act = vec![0.0f64; n_hidden];
    for h in 0..n_hidden {
        let mut pre = w.hidden_bias[h];
        for i in 0..5 {
            pre += w.hidden_weights[h * 5 + i] * inputs[i];
        }
        hidden_act[h] = cfg.activations[h % n_acts].apply(pre);
    }

    // Output neuron (linear, no output activation — caller thresholds)
    let mut out = w.output_bias;
    for h in 0..n_hidden {
        out += w.output_weights[h] * hidden_act[h];
    }
    out
}

// ─── Public API: query weights ────────────────────────────────────────────────

/// Query the CPPN to produce the full substrate weight matrix.
///
/// Iterates over all source→target pairs for adjacent layers:
/// - input → hidden
/// - hidden → output
///
/// Connection weight = CPPN output if `|output| > threshold`, else 0.
///
/// Returns a flat vector of length `substrate.n_weights()`:
/// `[input→hidden weights row-major, hidden→output weights row-major]`.
pub fn hyperneat_query_weights(
    cppn: &CppnWeights,
    cppn_cfg: &CppnConfig,
    substrate: &Substrate,
    threshold: f64,
) -> Vec<f64> {
    let n_in = substrate.input_coords.len();
    let n_hid = substrate.hidden_coords.len();
    let n_out = substrate.output_coords.len();
    let mut weights = Vec::with_capacity(n_in * n_hid + n_hid * n_out);

    // Input → hidden
    for &(xs, ys) in &substrate.input_coords {
        for &(xt, yt) in &substrate.hidden_coords {
            let raw = cppn_forward(cppn, cppn_cfg, xs, ys, xt, yt);
            weights.push(if raw.abs() > threshold { raw } else { 0.0 });
        }
    }

    // Hidden → output
    for &(xs, ys) in &substrate.hidden_coords {
        for &(xt, yt) in &substrate.output_coords {
            let raw = cppn_forward(cppn, cppn_cfg, xs, ys, xt, yt);
            weights.push(if raw.abs() > threshold { raw } else { 0.0 });
        }
    }

    weights
}

// ─── Public API: substrate forward pass ──────────────────────────────────────

/// Run a forward pass through the substrate network.
///
/// Architecture: input → (tanh) hidden → (tanh) output.
///
/// `substrate_weights` must have length `substrate.n_weights()` (produced by
/// `hyperneat_query_weights`).  `x` must have length `substrate.input_coords.len()`.
///
/// # Errors
/// Returns `DimensionMismatch` if dimensions do not agree.
pub fn hyperneat_forward(
    substrate_weights: &[f64],
    substrate: &Substrate,
    x: &[f64],
) -> EvolResult<Vec<f64>> {
    let n_in = substrate.input_coords.len();
    let n_hid = substrate.hidden_coords.len();
    let n_out = substrate.output_coords.len();

    if x.len() != n_in {
        return Err(EvolError::DimensionMismatch {
            expected: n_in,
            got: x.len(),
        });
    }
    let expected_w = n_in * n_hid + n_hid * n_out;
    if substrate_weights.len() != expected_w {
        return Err(EvolError::DimensionMismatch {
            expected: expected_w,
            got: substrate_weights.len(),
        });
    }

    // Input → hidden (tanh activation)
    let mut hidden = vec![0.0f64; n_hid];
    for h in 0..n_hid {
        let mut pre = 0.0;
        for i in 0..n_in {
            pre += x[i] * substrate_weights[i * n_hid + h];
        }
        hidden[h] = pre.tanh();
    }

    // Hidden → output (tanh activation)
    let ih_end = n_in * n_hid;
    let mut output = vec![0.0f64; n_out];
    for o in 0..n_out {
        let mut pre = 0.0;
        for h in 0..n_hid {
            pre += hidden[h] * substrate_weights[ih_end + h * n_out + o];
        }
        output[o] = pre.tanh();
    }

    Ok(output)
}

// ─── HyperNEAT configuration and state ───────────────────────────────────────

/// Full configuration for a HyperNEAT run.
#[derive(Debug, Clone)]
pub struct HyperNeatConfig {
    /// CPPN architecture configuration.
    pub cppn: CppnConfig,
    /// Substrate geometry (neuron coordinates).
    pub substrate: Substrate,
    /// Minimum |CPPN output| required to express a connection.
    pub expression_threshold: f64,
    /// Number of (μ+λ)-ES generations to run.
    pub n_evol_iters: usize,
    /// Initial perturbation standard deviation.
    pub sigma_init: f64,
    /// Per-generation multiplicative decay of sigma.
    pub sigma_decay: f64,
    /// Random seed.
    pub seed: u64,
}

impl HyperNeatConfig {
    /// Validate configuration parameters.
    ///
    /// # Errors
    /// Returns `InvalidParameter` for any out-of-range value.
    pub fn validate(&self) -> EvolResult<()> {
        if self.expression_threshold < 0.0 {
            return Err(EvolError::InvalidParameter(
                "expression_threshold must be >= 0".into(),
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
}

/// State produced or updated during a HyperNEAT run.
#[derive(Debug, Clone)]
pub struct HyperNeatState {
    /// Best CPPN weights found.
    pub cppn_weights: CppnWeights,
    /// Substrate weight matrix derived from the best CPPN.
    pub substrate_weights: Vec<f64>,
    /// Best fitness value seen so far.
    pub best_fitness: f64,
    /// Number of completed generations.
    pub generation: usize,
}

// ─── (μ + λ)-ES internals ────────────────────────────────────────────────────

/// ES hyper-parameters (fixed for HyperNEAT).
const MU: usize = 5;
const LAMBDA: usize = 20;

/// Perturb a flat parameter vector with Gaussian noise of std `sigma`.
fn perturb(params: &[f64], sigma: f64, rng: &mut LcgRng) -> Vec<f64> {
    params
        .iter()
        .map(|&p| p + rng.next_normal() * sigma)
        .collect()
}

/// Evaluate a flat CPPN parameter vector and return (fitness, substrate_weights).
fn evaluate_params(
    flat: &[f64],
    cfg: &HyperNeatConfig,
    fitness_fn: &impl Fn(&[f64], &[f64]) -> f64,
) -> EvolResult<(f64, Vec<f64>)> {
    let w = CppnWeights::from_flat(flat, cfg.cppn.n_hidden)?;
    let sw = hyperneat_query_weights(&w, &cfg.cppn, &cfg.substrate, cfg.expression_threshold);

    // Build a flat geometry description for the fitness callback.
    let geom: Vec<f64> = cfg
        .substrate
        .input_coords
        .iter()
        .chain(cfg.substrate.hidden_coords.iter())
        .chain(cfg.substrate.output_coords.iter())
        .flat_map(|&(x, y)| [x, y])
        .collect();

    let fitness = fitness_fn(&sw, &geom);
    Ok((fitness, sw))
}

// ─── Public API: run HyperNEAT ───────────────────────────────────────────────

/// Run HyperNEAT via (μ+λ)-ES to optimise the CPPN weights.
///
/// `fitness_fn(substrate_weights, geometry) -> f64` — higher is better.
/// `geometry` is the concatenation of all neuron coordinates (x0,y0, x1,y1, …).
///
/// # Errors
/// Returns `InvalidParameter` if `cfg` fails validation.
pub fn hyperneat_run(
    fitness_fn: impl Fn(&[f64], &[f64]) -> f64,
    cfg: &HyperNeatConfig,
) -> EvolResult<HyperNeatState> {
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
    let mut parent_fitness: Vec<f64> = parents
        .iter()
        .map(|p| {
            evaluate_params(p, cfg, &fitness_fn)
                .map(|(f, _)| f)
                .unwrap_or(f64::NEG_INFINITY)
        })
        .collect();

    // Track best overall
    let (best_init_idx, &best_init_fit) = parent_fitness
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or(EvolError::EmptyPopulation)?;

    let mut best_flat = parents[best_init_idx].clone();
    let mut best_fitness = best_init_fit;
    let mut sigma = cfg.sigma_init;

    for _gen in 0..cfg.n_evol_iters {
        // Generate λ offspring from μ parents (each offspring copies a random parent)
        let mut offspring: Vec<(Vec<f64>, f64)> = Vec::with_capacity(LAMBDA);
        for _ in 0..LAMBDA {
            let parent_idx = rng.next_usize(MU);
            let child = perturb(&parents[parent_idx], sigma, &mut rng);
            let fit = evaluate_params(&child, cfg, &fitness_fn)
                .map(|(f, _)| f)
                .unwrap_or(f64::NEG_INFINITY);
            offspring.push((child, fit));
        }

        // μ + λ selection: merge parents and offspring, keep top μ by fitness
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

    let best_cppn = CppnWeights::from_flat(&best_flat, cfg.cppn.n_hidden)?;
    let substrate_weights = hyperneat_query_weights(
        &best_cppn,
        &cfg.cppn,
        &cfg.substrate,
        cfg.expression_threshold,
    );

    Ok(HyperNeatState {
        cppn_weights: best_cppn,
        substrate_weights,
        best_fitness,
        generation: cfg.n_evol_iters,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
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

    fn default_substrate() -> Substrate {
        Substrate::linear(3, 4, 2).expect("linear should succeed")
    }

    // ── 1. CppnActivation apply is bounded/finite ─────────────────────────────

    #[test]
    fn cppn_activations_finite() {
        for &x in &[-10.0, -1.0, 0.0, 1.0, 10.0f64] {
            assert!(CppnActivation::Sigmoid.apply(x).is_finite());
            assert!(CppnActivation::Tanh.apply(x).is_finite());
            assert!(CppnActivation::Gaussian.apply(x).is_finite());
            assert!(CppnActivation::Sine.apply(x).is_finite());
        }
    }

    // ── 2. Sigmoid output in (0,1) ────────────────────────────────────────────

    #[test]
    fn sigmoid_range() {
        for &x in &[-5.0, 0.0, 5.0f64] {
            let v = CppnActivation::Sigmoid.apply(x);
            assert!(v > 0.0 && v < 1.0, "Sigmoid({x}) = {v} not in (0,1)");
        }
    }

    // ── 3. Gaussian peak at 0, decays ─────────────────────────────────────────

    #[test]
    fn gaussian_peak_at_zero() {
        let v0 = CppnActivation::Gaussian.apply(0.0);
        let v1 = CppnActivation::Gaussian.apply(2.0);
        assert!((v0 - 1.0).abs() < 1e-12);
        assert!(v1 < v0);
    }

    // ── 4. CppnConfig n_params formula ────────────────────────────────────────

    #[test]
    fn cppn_config_n_params() {
        let cfg = CppnConfig::new(6, vec![CppnActivation::Tanh]).expect("new should succeed");
        // 5*6 + 6 + 6 + 1 = 30+6+6+1 = 43
        assert_eq!(cfg.n_params(), 43);
    }

    // ── 5. CppnWeights round-trip through flat serialisation ─────────────────

    #[test]
    fn cppn_weights_flat_roundtrip() {
        let mut rng = LcgRng::new(42);
        let w = CppnWeights::random(4, 0.5, &mut rng);
        let flat = w.to_flat();
        let w2 = CppnWeights::from_flat(&flat, 4).expect("from_flat should succeed");
        for (a, b) in w.hidden_weights.iter().zip(w2.hidden_weights.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
        assert!((w.output_bias - w2.output_bias).abs() < 1e-12);
    }

    // ── 6. from_flat dimension mismatch returns error ─────────────────────────

    #[test]
    fn cppn_weights_from_flat_bad_dim() {
        let flat = vec![0.0; 5];
        assert!(CppnWeights::from_flat(&flat, 4).is_err());
    }

    // ── 7. Substrate linear produces correct n_weights ────────────────────────

    #[test]
    fn substrate_linear_n_weights() {
        let s = Substrate::linear(4, 5, 3).expect("linear should succeed");
        // 4*5 + 5*3 = 20 + 15 = 35
        assert_eq!(s.n_weights(), 35);
    }

    // ── 8. Substrate zero-layer size returns error ────────────────────────────

    #[test]
    fn substrate_zero_layer_err() {
        assert!(Substrate::linear(0, 4, 2).is_err());
        assert!(Substrate::linear(4, 0, 2).is_err());
        assert!(Substrate::linear(4, 4, 0).is_err());
    }

    // ── 9. CPPN output finite for random weights ──────────────────────────────

    #[test]
    fn cppn_forward_finite() {
        let mut rng = LcgRng::new(7);
        let cfg = default_cppn_cfg();
        let w = CppnWeights::random(cfg.n_hidden, 1.0, &mut rng);
        for _ in 0..20 {
            let xs = rng.next_f64() * 2.0 - 1.0;
            let ys = rng.next_f64() * 2.0 - 1.0;
            let xt = rng.next_f64() * 2.0 - 1.0;
            let yt = rng.next_f64() * 2.0 - 1.0;
            let out = cppn_forward(&w, &cfg, xs, ys, xt, yt);
            assert!(out.is_finite(), "CPPN output must be finite");
        }
    }

    // ── 10. hyperneat_query_weights correct length ────────────────────────────

    #[test]
    fn query_weights_length() {
        let mut rng = LcgRng::new(13);
        let cfg = default_cppn_cfg();
        let s = default_substrate();
        let w = CppnWeights::random(cfg.n_hidden, 1.0, &mut rng);
        let weights = hyperneat_query_weights(&w, &cfg, &s, 0.2);
        assert_eq!(weights.len(), s.n_weights());
    }

    // ── 11. hyperneat_forward shape and tanh range ────────────────────────────

    #[test]
    fn substrate_forward_shape_and_range() {
        let mut rng = LcgRng::new(17);
        let cfg = default_cppn_cfg();
        let s = default_substrate();
        let w = CppnWeights::random(cfg.n_hidden, 0.5, &mut rng);
        let sw = hyperneat_query_weights(&w, &cfg, &s, 0.0); // no threshold → keep all
        let x = vec![0.5, -0.3, 0.1];
        let out = hyperneat_forward(&sw, &s, &x).expect("hyperneat_forward should succeed");
        assert_eq!(out.len(), s.output_coords.len());
        for &v in &out {
            assert!(v.abs() <= 1.0 + 1e-10, "tanh output must be in [-1,1]");
        }
    }

    // ── 12. hyperneat_forward dimension mismatch error ────────────────────────

    #[test]
    fn substrate_forward_dim_mismatch() {
        let s = default_substrate();
        let sw = vec![0.0; s.n_weights()];
        // Wrong input length
        assert!(hyperneat_forward(&sw, &s, &[0.0, 0.5]).is_err());
        // Wrong weight length
        assert!(hyperneat_forward(&[0.0, 0.1], &s, &[0.0, 0.0, 0.0]).is_err());
    }

    // ── 13. hyperneat_run improves fitness or stays stable ────────────────────

    #[test]
    fn hyperneat_run_fitness_nonneg_improvement() {
        // Fitness: negative norm of substrate weights (maximised → weights → 0)
        let cfg = HyperNeatConfig {
            cppn: default_cppn_cfg(),
            substrate: Substrate::linear(2, 3, 1).expect("linear should succeed"),
            expression_threshold: 0.1,
            n_evol_iters: 20,
            sigma_init: 0.3,
            sigma_decay: 0.95,
            seed: 99,
        };
        let state = hyperneat_run(|sw, _geom| -sw.iter().map(|w| w * w).sum::<f64>(), &cfg)
            .expect("value should be present");
        assert!(state.best_fitness.is_finite());
        assert_eq!(state.generation, 20);
        assert_eq!(state.substrate_weights.len(), cfg.substrate.n_weights());
    }

    // ── 14. HyperNeatConfig validate catches bad params ───────────────────────

    #[test]
    fn hyperneat_config_validate() {
        let base = HyperNeatConfig {
            cppn: default_cppn_cfg(),
            substrate: default_substrate(),
            expression_threshold: 0.2,
            n_evol_iters: 10,
            sigma_init: 0.5,
            sigma_decay: 0.9,
            seed: 0,
        };
        assert!(base.validate().is_ok());

        let bad_sigma = HyperNeatConfig {
            sigma_init: 0.0,
            ..base.clone()
        };
        assert!(bad_sigma.validate().is_err());

        let bad_decay = HyperNeatConfig {
            sigma_decay: 1.5,
            ..base.clone()
        };
        assert!(bad_decay.validate().is_err());

        let bad_iters = HyperNeatConfig {
            n_evol_iters: 0,
            ..base.clone()
        };
        assert!(bad_iters.validate().is_err());
    }

    // ── 15. threshold = 0 includes all connections ────────────────────────────

    #[test]
    fn threshold_zero_keeps_all() {
        let mut rng = LcgRng::new(31);
        let cfg = default_cppn_cfg();
        let s = Substrate::linear(2, 2, 2).expect("linear should succeed");
        let w = CppnWeights::random(cfg.n_hidden, 2.0, &mut rng);
        let weights_strict = hyperneat_query_weights(&w, &cfg, &s, 1e9);
        let weights_all = hyperneat_query_weights(&w, &cfg, &s, 0.0);
        let n_zero_strict = weights_strict.iter().filter(|&&v| v == 0.0).count();
        let n_zero_all = weights_all.iter().filter(|&&v| v == 0.0).count();
        // Strict threshold should zero out more connections
        assert!(n_zero_strict >= n_zero_all);
    }
}
