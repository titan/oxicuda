//! Proposal network for Mip-NeRF 360-style importance sampling.
//!
//! Implements the learned density estimator from Barron et al. 2022
//! "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields".
//!
//! A small MLP predicts density along rays; the resulting histogram
//! is used to draw fine samples via inverse-CDF (importance sampling),
//! concentrating samples near high-density regions.

use crate::error::{NerfError, NerfResult};
use crate::handle::LcgRng;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the proposal MLP.
#[derive(Debug, Clone, Copy)]
pub struct ProposalMlpConfig {
    /// Hidden layer width (e.g. 256).
    pub n_hidden: usize,
    /// Number of hidden layers (e.g. 2).
    pub n_layers: usize,
    /// Number of proposal samples per ray.
    pub n_samples: usize,
    /// Positional encoding output dimension (e.g. 24 for 4 frequency levels).
    /// Must equal 3 + 6 * n_freqs for some integer n_freqs.
    pub pos_encoding_dim: usize,
}

// ─── Weights ─────────────────────────────────────────────────────────────────

/// Weight matrices and biases for the proposal MLP.
///
/// Layout: `layers[0]` is `n_hidden × in_dim`, `layers[1..n_layers]` are
/// `n_hidden × n_hidden`, `layers[n_layers]` is `1 × n_hidden`.
/// Biases match the row count of the corresponding weight matrix.
#[derive(Debug, Clone)]
pub struct ProposalMlpWeights {
    /// Weight matrices stored row-major; element `[i, j]` = `layers[l][i * in_dim + j]`.
    pub layers: Vec<Vec<f32>>,
    /// Bias vectors; `biases[l].len()` equals the output dimension of layer `l`.
    pub biases: Vec<Vec<f32>>,
}

// ─── ProposalNetwork ─────────────────────────────────────────────────────────

/// Small auxiliary MLP that predicts volumetric density for importance sampling.
///
/// Used in the Mip-NeRF 360 proposal sampling pipeline:
/// 1. Evaluate density at proposal sample midpoints.
/// 2. Build a normalised weight histogram.
/// 3. Draw fine samples via inverse-CDF.
/// 4. Compute the proposal loss to train the network.
#[derive(Debug, Clone)]
pub struct ProposalNetwork {
    /// Network hyperparameters.
    pub cfg: ProposalMlpConfig,
    /// Learned weight matrices and biases.
    pub weights: ProposalMlpWeights,
}

impl ProposalNetwork {
    /// Create a new `ProposalNetwork` with Kaiming-uniform–initialised weights.
    ///
    /// Kaiming uniform: U(−√(6 / fan_in), +√(6 / fan_in)).
    /// Biases are initialised to zero.
    ///
    /// # Errors
    ///
    /// Returns `InvalidSampleCount` if `n_hidden < 1` or `n_samples < 2`.
    /// Returns `InvalidFreqLevels` if `n_layers < 1`.
    /// Returns `InvalidFeatureDim` if `pos_encoding_dim < 3` or is inconsistent
    /// (not of the form 3 + 6 * k).
    pub fn new(cfg: ProposalMlpConfig, rng: &mut LcgRng) -> NerfResult<Self> {
        validate_config(&cfg)?;

        // Number of input features: positional encoding dimension.
        let in_dim = cfg.pos_encoding_dim;

        // Build layer shapes: [(in_dim → n_hidden), (n_hidden → n_hidden) × (n_layers-1), (n_hidden → 1)]
        let mut layer_weights: Vec<Vec<f32>> = Vec::with_capacity(cfg.n_layers + 1);
        let mut layer_biases: Vec<Vec<f32>> = Vec::with_capacity(cfg.n_layers + 1);

        // First hidden layer: in_dim → n_hidden
        let (w0, b0) = kaiming_layer(in_dim, cfg.n_hidden, rng);
        layer_weights.push(w0);
        layer_biases.push(b0);

        // Remaining hidden layers: n_hidden → n_hidden
        for _ in 1..cfg.n_layers {
            let (w, b) = kaiming_layer(cfg.n_hidden, cfg.n_hidden, rng);
            layer_weights.push(w);
            layer_biases.push(b);
        }

        // Output layer: n_hidden → 1 (raw density, before softplus)
        let (w_out, b_out) = kaiming_layer(cfg.n_hidden, 1, rng);
        layer_weights.push(w_out);
        layer_biases.push(b_out);

        Ok(Self {
            cfg,
            weights: ProposalMlpWeights {
                layers: layer_weights,
                biases: layer_biases,
            },
        })
    }

    /// Forward pass: given the positional encoding of a 3-D point, predict density.
    ///
    /// Applies ReLU hidden activations and softplus on the output to ensure σ > 0.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if `x_encoded.len() != cfg.pos_encoding_dim`.
    /// Returns `NanEncountered` if a NaN propagates through the network.
    pub fn predict_density(&self, x_encoded: &[f32]) -> NerfResult<f32> {
        if x_encoded.len() != self.cfg.pos_encoding_dim {
            return Err(NerfError::DimensionMismatch {
                expected: self.cfg.pos_encoding_dim,
                got: x_encoded.len(),
            });
        }

        let mut act: Vec<f32> = x_encoded.to_vec();

        // Hidden layers with ReLU
        for layer_idx in 0..self.cfg.n_layers {
            act = fc_relu(
                &act,
                &self.weights.layers[layer_idx],
                &self.weights.biases[layer_idx],
                self.cfg.n_hidden,
            );
        }

        // Output layer (no activation yet)
        let raw = fc_linear(
            &act,
            &self.weights.layers[self.cfg.n_layers],
            &self.weights.biases[self.cfg.n_layers],
            1,
        );

        let raw_val = raw[0];
        if raw_val.is_nan() {
            return Err(NerfError::NanEncountered {
                context: "proposal_network_output".into(),
            });
        }

        Ok(softplus(raw_val))
    }

    /// Compute proposal weights along a ray.
    ///
    /// For each bin midpoint, encodes the 3-D position, queries the MLP to get
    /// density σ, and computes alpha-composited weights with accumulated
    /// transmittance.
    ///
    /// # Arguments
    ///
    /// * `t_vals` — `n_samples + 1` bin edges (t-values along the ray).
    /// * `x_origin` — 3-D ray origin `[ox, oy, oz]`.
    /// * `x_dir` — 3-D ray direction (should be a unit vector) `[dx, dy, dz]`.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if `t_vals.len() != cfg.n_samples + 1`
    /// or if `x_origin`/`x_dir` are not length 3.
    pub fn ray_weights(
        &self,
        t_vals: &[f32],
        x_origin: &[f32],
        x_dir: &[f32],
    ) -> NerfResult<ProposalHistogram> {
        let expected_t = self.cfg.n_samples + 1;
        if t_vals.len() != expected_t {
            return Err(NerfError::DimensionMismatch {
                expected: expected_t,
                got: t_vals.len(),
            });
        }
        if x_origin.len() != 3 {
            return Err(NerfError::DimensionMismatch {
                expected: 3,
                got: x_origin.len(),
            });
        }
        if x_dir.len() != 3 {
            return Err(NerfError::DimensionMismatch {
                expected: 3,
                got: x_dir.len(),
            });
        }

        let n_samples = self.cfg.n_samples;
        let n_freqs = (self.cfg.pos_encoding_dim - 3) / 6;

        let mut densities = Vec::with_capacity(n_samples);
        let mut deltas = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let t_mid = 0.5 * (t_vals[i] + t_vals[i + 1]);
            let delta = (t_vals[i + 1] - t_vals[i]).max(0.0);

            // 3-D position at midpoint
            let pt = [
                x_origin[0] + t_mid * x_dir[0],
                x_origin[1] + t_mid * x_dir[1],
                x_origin[2] + t_mid * x_dir[2],
            ];

            // Encode and query density
            let encoded = Self::encode_position(&pt, n_freqs)?;
            let sigma = self.predict_density(&encoded)?;

            densities.push(sigma);
            deltas.push(delta);
        }

        // Compute transmittance-weighted alpha values.
        // alpha_i = 1 - exp(-sigma_i * delta_i)
        // T_i = prod_{j < i} exp(-sigma_j * delta_j)
        // w_i = T_i * alpha_i
        let mut weights = Vec::with_capacity(n_samples);
        let mut transmittance = 1.0_f32;
        for i in 0..n_samples {
            let neg_opt_depth = -(densities[i] * deltas[i]);
            let exp_term = neg_opt_depth.exp();
            let alpha = 1.0 - exp_term;
            let w = transmittance * alpha;
            weights.push(w.max(0.0));
            transmittance *= exp_term;
        }

        Ok(ProposalHistogram {
            bins: t_vals.to_vec(),
            weights,
        })
    }

    /// Draw `n_fine` importance samples by inverting the histogram CDF.
    ///
    /// Uniform samples u ~ U[0, 1] are mapped through the CDF of the proposal
    /// histogram.  The result is sorted in ascending order.
    ///
    /// # Errors
    ///
    /// Returns `InvalidSampleCount` if `n_fine == 0`.
    /// Returns `EmptyInput` if the histogram has no bins.
    pub fn importance_sample_from_histogram(
        hist: &ProposalHistogram,
        n_fine: usize,
        rng: &mut LcgRng,
    ) -> NerfResult<Vec<f32>> {
        if n_fine == 0 {
            return Err(NerfError::InvalidSampleCount { n: 0 });
        }
        if hist.weights.is_empty() {
            return Err(NerfError::EmptyInput);
        }

        let n = hist.weights.len();

        // Build CDF: cdf[0] = 0, cdf[i+1] = cdf[i] + w[i]
        let mut cdf = Vec::with_capacity(n + 1);
        cdf.push(0.0_f32);
        let mut running = 0.0_f32;
        for &w in &hist.weights {
            running += w.max(0.0);
            cdf.push(running);
        }

        // Normalise CDF (handle degenerate case of all-zero weights)
        let total = cdf[n];
        if total > 1e-10 {
            for v in cdf.iter_mut() {
                *v /= total;
            }
        } else {
            // Uniform fallback
            for (i, v) in cdf.iter_mut().enumerate() {
                *v = i as f32 / n as f32;
            }
        }
        // Ensure last element is exactly 1.0
        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }

        let mut t_samples = Vec::with_capacity(n_fine);
        for _ in 0..n_fine {
            let u = rng.next_f32();
            // Binary search for u in cdf (find the last index where cdf[idx] <= u)
            let idx = cdf
                .partition_point(|&c| c <= u)
                .saturating_sub(1)
                .min(n - 1);

            let c_lo = cdf[idx];
            let c_hi = cdf[idx + 1];
            let t_lo = hist.bins[idx];
            let t_hi = hist.bins[idx + 1];

            // Linear interpolation within bin
            let denom = c_hi - c_lo;
            let t = if denom < 1e-10 {
                t_lo
            } else {
                t_lo + (u - c_lo) / denom * (t_hi - t_lo)
            };
            t_samples.push(t);
        }

        t_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(t_samples)
    }

    /// Compute the proposal loss (Barron 2022 Eq. 14).
    ///
    /// Penalises the proposal histogram wherever it underestimates the true NeRF
    /// weight histogram.
    ///
    /// loss_i = max(0, w_nerf_i − w_proposal_i)² / (w_nerf_i + 1e-5)
    /// total   = mean over i
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if `proposal_hist.weights.len() != nerf_weights.len()`.
    /// Returns `EmptyInput` if the arrays are empty.
    pub fn proposal_loss(
        proposal_hist: &ProposalHistogram,
        nerf_weights: &[f32],
    ) -> NerfResult<f32> {
        let n = proposal_hist.weights.len();
        if n == 0 {
            return Err(NerfError::EmptyInput);
        }
        if nerf_weights.len() != n {
            return Err(NerfError::DimensionMismatch {
                expected: n,
                got: nerf_weights.len(),
            });
        }

        let loss: f32 = proposal_hist
            .weights
            .iter()
            .zip(nerf_weights.iter())
            .map(|(&w_p, &w_n)| {
                let excess = (w_n - w_p).max(0.0);
                excess * excess / (w_n + 1e-5)
            })
            .sum::<f32>()
            / n as f32;

        Ok(loss)
    }

    /// Linear positional encoding of a 3-D point.
    ///
    /// Output: `[x, y, z, sin(π x), cos(π x), sin(π y), cos(π y), sin(π z), cos(π z),
    ///           sin(2π x), cos(2π x), ..., sin(2^{n_freqs-1} π z), cos(2^{n_freqs-1} π z)]`
    ///
    /// Total length: 3 + 6 * n_freqs.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if `x.len() != 3`.
    pub fn encode_position(x: &[f32], n_freqs: usize) -> NerfResult<Vec<f32>> {
        if x.len() != 3 {
            return Err(NerfError::DimensionMismatch {
                expected: 3,
                got: x.len(),
            });
        }

        let out_dim = 3 + 6 * n_freqs;
        let mut out = Vec::with_capacity(out_dim);

        // Raw coordinates
        out.push(x[0]);
        out.push(x[1]);
        out.push(x[2]);

        // Frequency bands: sin/cos for each coord and each frequency level
        for k in 0..n_freqs {
            let freq = (1_u64 << k) as f32 * std::f32::consts::PI;
            for &coord in x.iter() {
                out.push((freq * coord).sin());
                out.push((freq * coord).cos());
            }
        }

        Ok(out)
    }

    /// Total number of trainable parameters in the network.
    #[must_use]
    pub fn n_params(&self) -> usize {
        let w_count: usize = self.weights.layers.iter().map(|w| w.len()).sum();
        let b_count: usize = self.weights.biases.iter().map(|b| b.len()).sum();
        w_count + b_count
    }
}

// ─── ProposalHistogram ───────────────────────────────────────────────────────

/// Discrete weight histogram along a ray, used as the proposal distribution.
#[derive(Debug, Clone)]
pub struct ProposalHistogram {
    /// `n_samples + 1` bin edges (t-values along the ray).
    pub bins: Vec<f32>,
    /// `n_samples` normalised weights (density × interval, transmittance-weighted).
    pub weights: Vec<f32>,
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Validate `ProposalMlpConfig` fields.
fn validate_config(cfg: &ProposalMlpConfig) -> NerfResult<()> {
    if cfg.n_hidden < 1 {
        return Err(NerfError::InvalidSampleCount { n: cfg.n_hidden });
    }
    if cfg.n_layers < 1 {
        return Err(NerfError::InvalidFreqLevels {
            levels: cfg.n_layers,
        });
    }
    if cfg.n_samples < 2 {
        return Err(NerfError::InvalidSampleCount { n: cfg.n_samples });
    }
    if cfg.pos_encoding_dim < 3 {
        return Err(NerfError::InvalidFeatureDim {
            dim: cfg.pos_encoding_dim,
        });
    }
    // Must satisfy pos_encoding_dim == 3 + 6 * k for some k >= 0
    let remainder = cfg.pos_encoding_dim.saturating_sub(3);
    if !remainder.is_multiple_of(6) {
        return Err(NerfError::InvalidFeatureDim {
            dim: cfg.pos_encoding_dim,
        });
    }
    Ok(())
}

/// Build a weight matrix (out_dim × in_dim) with Kaiming uniform initialisation.
/// Kaiming uniform: U(−√(6 / fan_in), +√(6 / fan_in)).
fn kaiming_layer(in_dim: usize, out_dim: usize, rng: &mut LcgRng) -> (Vec<f32>, Vec<f32>) {
    let bound = (6.0_f32 / in_dim as f32).sqrt();
    let w: Vec<f32> = (0..out_dim * in_dim)
        .map(|_| rng.next_f32_range(-bound, bound))
        .collect();
    let b = vec![0.0_f32; out_dim];
    (w, b)
}

/// Softplus activation: log(1 + exp(x)).  Numerically stable for large x.
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0_f32 + x.exp()).ln()
    }
}

/// Fully-connected layer with ReLU activation.
/// Weights stored row-major: `w[i * in_dim + j]`.
fn fc_relu(x: &[f32], w: &[f32], b: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    let mut out = vec![0.0_f32; out_dim];
    for (i, o) in out.iter_mut().enumerate() {
        let row_start = i * in_dim;
        let dot: f32 = w[row_start..row_start + in_dim]
            .iter()
            .zip(x.iter())
            .map(|(&wi, &xi)| wi * xi)
            .sum();
        *o = (dot + b[i]).max(0.0);
    }
    out
}

/// Fully-connected layer without activation.
/// Weights stored row-major: `w[i * in_dim + j]`.
fn fc_linear(x: &[f32], w: &[f32], b: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    let mut out = vec![0.0_f32; out_dim];
    for (i, o) in out.iter_mut().enumerate() {
        let row_start = i * in_dim;
        let dot: f32 = w[row_start..row_start + in_dim]
            .iter()
            .zip(x.iter())
            .map(|(&wi, &xi)| wi * xi)
            .sum();
        *o = dot + b[i];
    }
    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cfg() -> ProposalMlpConfig {
        ProposalMlpConfig {
            n_hidden: 16,
            n_layers: 2,
            n_samples: 8,
            pos_encoding_dim: 15, // 3 + 6*2
        }
    }

    fn make_net() -> ProposalNetwork {
        let mut rng = LcgRng::new(42);
        ProposalNetwork::new(test_cfg(), &mut rng).expect("value should be present")
    }

    // --- density prediction ---

    #[test]
    fn predict_density_positive() {
        let net = make_net();
        let enc = vec![0.0_f32; 15];
        let sigma = net
            .predict_density(&enc)
            .expect("predict_density should succeed");
        assert!(sigma > 0.0, "softplus output must be positive, got {sigma}");
    }

    #[test]
    fn predict_density_shape() {
        // Returns a single scalar (f32), not a vector — verified by type system.
        // We just test that it runs and is finite.
        let net = make_net();
        let enc = vec![0.1_f32; 15];
        let sigma = net
            .predict_density(&enc)
            .expect("predict_density should succeed");
        assert!(sigma.is_finite());
    }

    // --- ray_weights ---

    #[test]
    fn ray_weights_sums_to_at_most_1() {
        let net = make_net();
        let n = test_cfg().n_samples;
        let t_vals: Vec<f32> = (0..=n).map(|i| i as f32 * 0.1).collect();
        let origin = [0.0_f32, 0.0, 0.0];
        let dir = [0.0_f32, 0.0, 1.0];
        let hist = net
            .ray_weights(&t_vals, &origin, &dir)
            .expect("ray_weights should succeed");
        let total: f32 = hist.weights.iter().sum();
        assert!(total <= 1.0 + 1e-5, "total weight {total} > 1");
    }

    #[test]
    fn ray_weights_non_negative() {
        let net = make_net();
        let n = test_cfg().n_samples;
        let t_vals: Vec<f32> = (0..=n).map(|i| i as f32 * 0.1).collect();
        let origin = [0.0_f32, 0.0, 0.0];
        let dir = [1.0_f32, 0.0, 0.0];
        let hist = net
            .ray_weights(&t_vals, &origin, &dir)
            .expect("ray_weights should succeed");
        for &w in &hist.weights {
            assert!(w >= 0.0, "negative weight {w}");
        }
    }

    #[test]
    fn ray_weights_histogram_n_samples() {
        let net = make_net();
        let n = test_cfg().n_samples;
        let t_vals: Vec<f32> = (0..=n).map(|i| i as f32 * 0.1).collect();
        let origin = [0.0_f32, 0.0, 0.0];
        let dir = [0.0_f32, 1.0, 0.0];
        let hist = net
            .ray_weights(&t_vals, &origin, &dir)
            .expect("ray_weights should succeed");
        assert_eq!(hist.weights.len(), n);
        assert_eq!(hist.bins.len(), n + 1);
    }

    // --- importance sampling ---

    #[test]
    fn importance_sample_in_range() {
        let net = make_net();
        let n = test_cfg().n_samples;
        let t_vals: Vec<f32> = (0..=n).map(|i| i as f32 * 0.1).collect();
        let origin = [0.0_f32, 0.0, 0.0];
        let dir = [0.0_f32, 0.0, 1.0];
        let hist = net
            .ray_weights(&t_vals, &origin, &dir)
            .expect("ray_weights should succeed");
        let t_near = *t_vals.first().expect("first should succeed");
        let t_far = *t_vals.last().expect("last should succeed");

        let mut rng = LcgRng::new(7);
        let samples = ProposalNetwork::importance_sample_from_histogram(&hist, 16, &mut rng)
            .expect("importance_sample_from_histogram should succeed");
        for &t in &samples {
            assert!(
                t >= t_near - 1e-6 && t <= t_far + 1e-6,
                "t={t} outside [{t_near}, {t_far}]"
            );
        }
    }

    #[test]
    fn importance_sample_sorted() {
        let net = make_net();
        let n = test_cfg().n_samples;
        let t_vals: Vec<f32> = (0..=n).map(|i| i as f32 * 0.1).collect();
        let origin = [0.0_f32, 0.0, 0.0];
        let dir = [0.0_f32, 0.0, 1.0];
        let hist = net
            .ray_weights(&t_vals, &origin, &dir)
            .expect("ray_weights should succeed");
        let mut rng = LcgRng::new(13);
        let samples = ProposalNetwork::importance_sample_from_histogram(&hist, 12, &mut rng)
            .expect("importance_sample_from_histogram should succeed");
        assert!(
            samples.windows(2).all(|w| w[0] <= w[1]),
            "samples not sorted"
        );
    }

    #[test]
    fn importance_sample_count() {
        let net = make_net();
        let n = test_cfg().n_samples;
        let t_vals: Vec<f32> = (0..=n).map(|i| i as f32 * 0.2).collect();
        let origin = [0.0_f32, 0.0, 0.0];
        let dir = [0.0_f32, 0.0, 1.0];
        let hist = net
            .ray_weights(&t_vals, &origin, &dir)
            .expect("ray_weights should succeed");
        let mut rng = LcgRng::new(99);
        let n_fine = 32;
        let samples = ProposalNetwork::importance_sample_from_histogram(&hist, n_fine, &mut rng)
            .expect("importance_sample_from_histogram should succeed");
        assert_eq!(samples.len(), n_fine);
    }

    // --- proposal loss ---

    #[test]
    fn proposal_loss_non_negative() {
        let hist = ProposalHistogram {
            bins: vec![0.0, 0.25, 0.5, 0.75, 1.0],
            weights: vec![0.1, 0.5, 0.3, 0.1],
        };
        let nerf_weights = vec![0.2, 0.4, 0.3, 0.1];
        let loss = ProposalNetwork::proposal_loss(&hist, &nerf_weights)
            .expect("proposal_loss should succeed");
        assert!(loss >= 0.0, "loss should be non-negative, got {loss}");
    }

    #[test]
    fn proposal_loss_zero_when_proposal_ge_nerf() {
        let hist = ProposalHistogram {
            bins: vec![0.0, 0.25, 0.5, 0.75, 1.0],
            weights: vec![0.3, 0.5, 0.4, 0.2],
        };
        let nerf_weights = vec![0.2, 0.4, 0.3, 0.1];
        let loss = ProposalNetwork::proposal_loss(&hist, &nerf_weights)
            .expect("proposal_loss should succeed");
        assert!(
            loss.abs() < 1e-7,
            "loss should be 0 when proposal >= nerf everywhere, got {loss}"
        );
    }

    // --- positional encoding ---

    #[test]
    fn encode_position_output_len() {
        let n_freqs = 4;
        let x = [0.1_f32, 0.2, 0.3];
        let enc =
            ProposalNetwork::encode_position(&x, n_freqs).expect("encode_position should succeed");
        assert_eq!(enc.len(), 3 + 6 * n_freqs);
    }

    #[test]
    fn encode_position_contains_identity() {
        let x = [0.5_f32, -0.3, 0.1];
        let enc = ProposalNetwork::encode_position(&x, 2).expect("encode_position should succeed");
        assert!((enc[0] - x[0]).abs() < 1e-7);
        assert!((enc[1] - x[1]).abs() < 1e-7);
        assert!((enc[2] - x[2]).abs() < 1e-7);
    }

    // --- parameter count ---

    #[test]
    fn n_params_formula() {
        // 2-hidden-layer MLP: in→h, h→h, h→1
        // params = (in*h + h) + (h*h + h) + (h*1 + 1)
        let cfg = ProposalMlpConfig {
            n_hidden: 16,
            n_layers: 2,
            n_samples: 4,
            pos_encoding_dim: 15, // 3 + 6*2
        };
        let mut rng = LcgRng::new(1);
        let net = ProposalNetwork::new(cfg, &mut rng).expect("new should succeed");
        let in_dim = 15usize;
        let h = 16usize;
        let expected = (in_dim * h + h) + (h * h + h) + (h + 1);
        assert_eq!(net.n_params(), expected);
    }

    #[test]
    fn new_creates_correct_layer_count() {
        let cfg = ProposalMlpConfig {
            n_hidden: 8,
            n_layers: 3,
            n_samples: 4,
            pos_encoding_dim: 27, // 3 + 6*4
        };
        let mut rng = LcgRng::new(5);
        let net = ProposalNetwork::new(cfg, &mut rng).expect("new should succeed");
        // n_layers hidden + 1 output = n_layers + 1 total
        assert_eq!(net.weights.layers.len(), cfg.n_layers + 1);
    }

    // --- error cases ---

    #[test]
    fn err_n_hidden_zero() {
        let cfg = ProposalMlpConfig {
            n_hidden: 0,
            n_layers: 2,
            n_samples: 8,
            pos_encoding_dim: 15,
        };
        let mut rng = LcgRng::new(1);
        assert!(ProposalNetwork::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn err_n_samples_less_than_2() {
        let cfg = ProposalMlpConfig {
            n_hidden: 8,
            n_layers: 2,
            n_samples: 1,
            pos_encoding_dim: 15,
        };
        let mut rng = LcgRng::new(1);
        assert!(ProposalNetwork::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn err_wrong_encoding_dim() {
        // pos_encoding_dim = 10 → 10 - 3 = 7, 7 % 6 != 0 → error
        let cfg = ProposalMlpConfig {
            n_hidden: 8,
            n_layers: 2,
            n_samples: 4,
            pos_encoding_dim: 10,
        };
        let mut rng = LcgRng::new(1);
        assert!(ProposalNetwork::new(cfg, &mut rng).is_err());
    }
}
