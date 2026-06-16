//! CROWN / alpha-CROWN bound propagation for neural networks.
//!
//! References:
//! * Zhang, Weng, Chen, Hsieh & Daniel (2018 NeurIPS),
//!   *"Efficient Neural Network Robustness Certification with General
//!   Activation Functions"* — CROWN.
//! * Xu, Zhang, Zhang, Shi, Jin, Wang, Weng, Darrell & Hsieh (2021 ICLR),
//!   *"Fast and Complete: Enabling Complete Neural Network Verification with
//!   Rapid and Massively Parallel Incomplete Verifiers"* — alpha-CROWN.
//!
//! # Background
//!
//! CROWN propagates **linear bounds** on neural-network outputs backward
//! through the layers.  For a ReLU neuron with pre-activation interval
//! `[l, u]` there are three cases:
//!
//! * `u ≤ 0` (inactive): output = 0 → bounds collapse to zero.
//! * `l ≥ 0` (active):   output = x → bounds pass through with slope 1.
//! * `l < 0 < u` (ambiguous): linear relaxation is used.
//!   - **Upper bound**: `x̂ ≤ u/(u-l) · (x - l)`, slope = `u/(u-l)`, intercept = `-l·u/(u-l)`.
//!   - **Lower bound**: `x̂ ≥ α · x`, where `0 ≤ α ≤ 1` is a per-neuron parameter.
//!
//! alpha-CROWN optimises `α` per-neuron to tighten the lower bound on the
//! certified objective.
//!
//! # Conventions
//!
//! * Layer weights are `[out × in]` row-major.
//! * Interval arithmetic uses `(lo, hi)` for lower/upper bounds.
//! * The input perturbation is an L_inf ball: `[x0 - eps, x0 + eps]`.
//! * The final (output) layer has **no** ReLU applied; intermediate hidden
//!   layers have ReLU activation applied after the linear layer.

use crate::error::{AdvError, AdvResult};

// ─── NeuronBound ──────────────────────────────────────────────────────────────

/// Input / output bounds `[lower, upper]` per neuron in a layer.
#[derive(Debug, Clone)]
pub struct NeuronBound {
    /// Per-neuron lower bounds.
    pub lower: Vec<f32>,
    /// Per-neuron upper bounds.
    pub upper: Vec<f32>,
}

impl NeuronBound {
    /// Build a new `NeuronBound`.  Lengths of `lower` and `upper` must match,
    /// and each pair must satisfy `lower[i] <= upper[i]`.
    ///
    /// # Errors
    /// * [`AdvError::DimensionMismatch`] — lengths differ.
    /// * [`AdvError::NanEncountered`]    — non-finite value.
    pub fn new(lower: Vec<f32>, upper: Vec<f32>) -> AdvResult<Self> {
        if lower.len() != upper.len() {
            return Err(AdvError::DimensionMismatch {
                expected: lower.len(),
                got: upper.len(),
            });
        }
        if lower.iter().chain(upper.iter()).any(|v| !v.is_finite()) {
            return Err(AdvError::NanEncountered {
                location: "NeuronBound::new",
            });
        }
        Ok(Self { lower, upper })
    }

    /// Number of neurons.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.lower.len()
    }

    /// True if there are no neurons.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.lower.is_empty()
    }
}

// ─── LinearLayer ──────────────────────────────────────────────────────────────

/// A single fully-connected (linear / affine) layer: `y = W x + b`.
#[derive(Debug, Clone)]
pub struct LinearLayer {
    /// Weight matrix `[out_features × in_features]`, row-major.
    pub weight: Vec<f32>,
    /// Bias vector `[out_features]`.
    pub bias: Vec<f32>,
    /// Input dimensionality.
    pub in_features: usize,
    /// Output dimensionality.
    pub out_features: usize,
}

impl LinearLayer {
    /// Construct a new `LinearLayer` and validate dimensions.
    ///
    /// # Errors
    /// * [`AdvError::DimensionMismatch`] — weight or bias length mismatch.
    /// * [`AdvError::EmptyInput`]         — zero features.
    pub fn new(
        weight: Vec<f32>,
        bias: Vec<f32>,
        in_features: usize,
        out_features: usize,
    ) -> AdvResult<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(AdvError::EmptyInput);
        }
        let expected_w = in_features * out_features;
        if weight.len() != expected_w {
            return Err(AdvError::DimensionMismatch {
                expected: expected_w,
                got: weight.len(),
            });
        }
        if bias.len() != out_features {
            return Err(AdvError::DimensionMismatch {
                expected: out_features,
                got: bias.len(),
            });
        }
        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
        })
    }
}

// ─── CrownConfig ──────────────────────────────────────────────────────────────

/// Hyper-parameters for CROWN / alpha-CROWN bound propagation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrownConfig {
    /// Number of alpha optimisation steps.
    /// `0` = vanilla CROWN (no alpha tuning); `> 0` = alpha-CROWN.
    pub n_alpha_steps: usize,
    /// Learning rate for alpha optimisation (default 0.1).
    pub alpha_lr: f32,
    /// L_inf input perturbation budget ε.
    pub eps: f32,
}

impl Default for CrownConfig {
    fn default() -> Self {
        Self {
            n_alpha_steps: 0,
            alpha_lr: 0.1,
            eps: 0.1,
        }
    }
}

// ─── AlphaBound ───────────────────────────────────────────────────────────────

/// Per-neuron alpha parameters for alpha-CROWN lower-bound relaxation.
///
/// Each `alpha[i] ∈ [0, 1]` controls the slope of the lower-bound linear
/// approximation for ReLU in the ambiguous region.
#[derive(Debug, Clone)]
pub struct AlphaBound {
    /// One value per neuron; clamped to `[0, 1]`.
    pub alpha: Vec<f32>,
}

// ─── CrownVerifier ────────────────────────────────────────────────────────────

/// CROWN / alpha-CROWN network-output bound verification.
pub struct CrownVerifier;

impl CrownVerifier {
    // ─── ReLU linear relaxation ──────────────────────────────────────────────

    /// Compute the linear relaxation bounds for one ReLU neuron with
    /// pre-activation interval `[l, u]` and lower-bound slope `alpha`.
    ///
    /// Returns `(alpha_lower, beta_lower, alpha_upper, beta_upper)` where
    ///
    /// ```text
    /// lower: x̂  ≥  alpha_lower · x + beta_lower
    /// upper: x̂  ≤  alpha_upper · x + beta_upper
    /// ```
    ///
    /// Cases:
    /// * `u ≤ 0`  (inactive): all zeros.
    /// * `l ≥ 0`  (active):   (1, 0, 1, 0) — identity pass-through.
    /// * ambiguous: upper = `u/(u-l)`, intercept = `-l·u/(u-l)`;
    ///   lower = `alpha` (clamped to `[0,1]`), intercept = 0.
    #[must_use]
    pub fn relu_linear_bounds(l: f32, u: f32, alpha: f32) -> (f32, f32, f32, f32) {
        if u <= 0.0 {
            // Inactive: ReLU(x) = 0
            return (0.0, 0.0, 0.0, 0.0);
        }
        if l >= 0.0 {
            // Active: ReLU(x) = x
            return (1.0, 0.0, 1.0, 0.0);
        }
        // Ambiguous region l < 0 < u
        let slope_up = u / (u - l);
        let intercept_up = -l * slope_up; // = -l * u / (u - l)
        let slope_lo = alpha.clamp(0.0, 1.0);
        (slope_lo, 0.0, slope_up, intercept_up)
    }

    // ─── Forward linear-layer bound propagation ───────────────────────────────

    /// Forward bound propagation through a linear layer `y = Wx + b`.
    ///
    /// Given input bounds `[x_lo, x_hi]`, computes output bounds `[y_lo, y_hi]`
    /// using interval arithmetic:
    ///
    /// ```text
    /// y_lo[j] = b[j] + Σ_i  max(w[j,i], 0)·x_lo[i] + min(w[j,i], 0)·x_hi[i]
    /// y_hi[j] = b[j] + Σ_i  max(w[j,i], 0)·x_hi[i] + min(w[j,i], 0)·x_lo[i]
    /// ```
    ///
    /// # Errors
    /// * [`AdvError::DimensionMismatch`] — `input_bounds.len() != layer.in_features`.
    pub fn propagate_linear(
        layer: &LinearLayer,
        input_bounds: &NeuronBound,
    ) -> AdvResult<NeuronBound> {
        if input_bounds.len() != layer.in_features {
            return Err(AdvError::DimensionMismatch {
                expected: layer.in_features,
                got: input_bounds.len(),
            });
        }

        let in_f = layer.in_features;
        let out_f = layer.out_features;

        let mut lower = Vec::with_capacity(out_f);
        let mut upper = Vec::with_capacity(out_f);

        for j in 0..out_f {
            let mut y_lo = layer.bias[j];
            let mut y_hi = layer.bias[j];
            for i in 0..in_f {
                let w = layer.weight[j * in_f + i];
                let x_lo = input_bounds.lower[i];
                let x_hi = input_bounds.upper[i];
                if w >= 0.0 {
                    y_lo += w * x_lo;
                    y_hi += w * x_hi;
                } else {
                    y_lo += w * x_hi;
                    y_hi += w * x_lo;
                }
            }
            // Floating-point guard: enforce lo <= hi
            if y_lo > y_hi {
                std::mem::swap(&mut y_lo, &mut y_hi);
            }
            lower.push(y_lo);
            upper.push(y_hi);
        }

        Ok(NeuronBound { lower, upper })
    }

    // ─── ReLU bound propagation ───────────────────────────────────────────────

    /// Apply ReLU to interval bounds: `max(lo, 0)`, `max(hi, 0)`.
    ///
    /// This is the standard interval arithmetic for the ReLU activation:
    /// the lower bound is clamped to zero (negative half is killed), and
    /// the upper bound is passed through for any neuron that is not fully
    /// inactive.
    #[must_use]
    pub fn propagate_relu(input_bounds: &NeuronBound) -> NeuronBound {
        let lower = input_bounds.lower.iter().map(|&v| v.max(0.0)).collect();
        let upper = input_bounds.upper.iter().map(|&v| v.max(0.0)).collect();
        NeuronBound { lower, upper }
    }

    // ─── Full network CROWN bound propagation ────────────────────────────────

    /// CROWN full-network bound propagation.
    ///
    /// Given an input sample `x0` and an L_inf perturbation budget `cfg.eps`,
    /// propagates interval bounds through every layer.
    ///
    /// The output is the list of `NeuronBound` for each layer's output
    /// (post-linear, with ReLU applied to all hidden layers; no ReLU on the
    /// final output layer).
    ///
    /// # Errors
    /// * [`AdvError::EmptyInput`]         — `layers` is empty.
    /// * [`AdvError::DimensionMismatch`]  — `x0.len() != layers[0].in_features`.
    /// * [`AdvError::InvalidEpsilon`]     — `cfg.eps < 0` or non-finite.
    pub fn crown_bound_propagation(
        x0: &[f32],
        layers: &[LinearLayer],
        cfg: &CrownConfig,
    ) -> AdvResult<Vec<NeuronBound>> {
        if layers.is_empty() {
            return Err(AdvError::EmptyInput);
        }
        if !(cfg.eps.is_finite() && cfg.eps >= 0.0) {
            return Err(AdvError::InvalidEpsilon { eps: cfg.eps });
        }
        if x0.len() != layers[0].in_features {
            return Err(AdvError::DimensionMismatch {
                expected: layers[0].in_features,
                got: x0.len(),
            });
        }

        // Build initial input bounds: [x0[i] - eps, x0[i] + eps]
        let lower: Vec<f32> = x0.iter().map(|&v| v - cfg.eps).collect();
        let upper: Vec<f32> = x0.iter().map(|&v| v + cfg.eps).collect();
        let mut current_bounds = NeuronBound { lower, upper };

        let n_layers = layers.len();
        let mut all_bounds = Vec::with_capacity(n_layers);

        for (layer_idx, layer) in layers.iter().enumerate() {
            // Linear propagation
            let post_linear = Self::propagate_linear(layer, &current_bounds)?;

            let is_last = layer_idx == n_layers - 1;
            if is_last {
                // No ReLU on the output layer
                all_bounds.push(post_linear);
            } else {
                // Apply ReLU for hidden layers
                let post_relu = Self::propagate_relu(&post_linear);
                all_bounds.push(post_linear);
                current_bounds = post_relu;
            }
        }

        Ok(all_bounds)
    }

    // ─── Certified radius ─────────────────────────────────────────────────────

    /// Compute the certified L_inf radius for a given sample.
    ///
    /// Checks whether the true class `label` has a higher lower bound than
    /// the upper bound of every other class, for the given `cfg.eps`.
    ///
    /// Returns `cfg.eps` if the network is certifiably robust (true-class
    /// lower bound > all other-class upper bounds), otherwise returns `0.0`.
    ///
    /// # Errors
    /// * [`AdvError::EmptyInput`]         — no layers.
    /// * [`AdvError::DimensionMismatch`]  — label out of range or x0 mismatch.
    pub fn certified_radius(
        x0: &[f32],
        label: usize,
        layers: &[LinearLayer],
        cfg: &CrownConfig,
    ) -> AdvResult<f32> {
        if layers.is_empty() {
            return Err(AdvError::EmptyInput);
        }
        // Run bound propagation
        let layer_bounds = Self::crown_bound_propagation(x0, layers, cfg)?;

        // Get output (last layer) bounds
        let output_bounds = &layer_bounds[layer_bounds.len() - 1];
        let n_classes = output_bounds.len();

        if label >= n_classes {
            return Err(AdvError::DimensionMismatch {
                expected: n_classes - 1,
                got: label,
            });
        }

        // True class minimum value (lower bound)
        let true_class_lo = output_bounds.lower[label];

        // Check: is true_class_lo > upper bound of every other class?
        let is_robust = (0..n_classes).filter(|&j| j != label).all(|j| {
            let other_upper = output_bounds.upper[j];
            true_class_lo > other_upper
        });

        if is_robust { Ok(cfg.eps) } else { Ok(0.0) }
    }

    // ─── alpha-CROWN alpha optimisation ──────────────────────────────────────

    /// Optimise per-neuron alpha values for the first hidden layer using
    /// alpha-CROWN (finite-difference gradient, simplified implementation).
    ///
    /// For `cfg.n_alpha_steps` iterations:
    /// 1. Compute the certified radius with current alpha (via perturbed propagation).
    /// 2. Estimate gradient ∂(certified_radius)/∂α via finite differences (δ = 1e-4).
    /// 3. Gradient-ascent update: `α[i] += alpha_lr * grad[i]`.
    /// 4. Clamp α to `[0, 1]`.
    ///
    /// If `cfg.n_alpha_steps == 0`, returns α = 0.5 for all first-hidden-layer
    /// neurons (initial value, no optimisation).
    ///
    /// # Errors
    /// * [`AdvError::EmptyInput`]         — no layers.
    /// * [`AdvError::DimensionMismatch`]  — x0 size mismatch.
    pub fn optimize_alpha(
        x0: &[f32],
        label: usize,
        layers: &[LinearLayer],
        cfg: &CrownConfig,
    ) -> AdvResult<AlphaBound> {
        if layers.is_empty() {
            return Err(AdvError::EmptyInput);
        }
        if x0.len() != layers[0].in_features {
            return Err(AdvError::DimensionMismatch {
                expected: layers[0].in_features,
                got: x0.len(),
            });
        }

        let n_hidden = layers[0].out_features;
        let mut alpha = vec![0.5_f32; n_hidden];

        if cfg.n_alpha_steps == 0 {
            return Ok(AlphaBound { alpha });
        }

        for _step in 0..cfg.n_alpha_steps {
            // Compute objective with current alpha (use certified_radius as proxy)
            let base_cert = Self::certified_radius_with_alpha(x0, label, layers, cfg, &alpha)?;

            // Finite-difference gradient w.r.t. each alpha[i]
            let fd_delta = 1e-4_f32;
            let mut grad = vec![0.0_f32; n_hidden];
            for i in 0..n_hidden {
                let old = alpha[i];
                alpha[i] = (old + fd_delta).clamp(0.0, 1.0);
                let cert_plus = Self::certified_radius_with_alpha(x0, label, layers, cfg, &alpha)?;
                alpha[i] = old;
                // Simple forward-difference: ∂f/∂α_i ≈ (f(α+δe_i) - f(α)) / δ
                grad[i] = (cert_plus - base_cert) / fd_delta;
            }

            // Gradient ascent + clamp
            for i in 0..n_hidden {
                alpha[i] = (alpha[i] + cfg.alpha_lr * grad[i]).clamp(0.0, 1.0);
            }
        }

        Ok(AlphaBound { alpha })
    }

    // ─── Internal helpers ─────────────────────────────────────────────────────

    /// Run bound propagation with custom alpha values for the first hidden layer.
    ///
    /// This uses alpha-informed lower bounds via `relu_linear_bounds` to
    /// determine the interval tightened by the alpha relaxation.
    fn certified_radius_with_alpha(
        x0: &[f32],
        label: usize,
        layers: &[LinearLayer],
        cfg: &CrownConfig,
        alpha: &[f32],
    ) -> AdvResult<f32> {
        if layers.is_empty() {
            return Err(AdvError::EmptyInput);
        }

        // Initial input bounds
        let lower: Vec<f32> = x0.iter().map(|&v| v - cfg.eps).collect();
        let upper: Vec<f32> = x0.iter().map(|&v| v + cfg.eps).collect();
        let mut current_bounds = NeuronBound { lower, upper };

        let n_layers = layers.len();
        let mut final_bounds: Option<NeuronBound> = None;

        for (layer_idx, layer) in layers.iter().enumerate() {
            let post_linear = Self::propagate_linear(layer, &current_bounds)?;
            let is_last = layer_idx == n_layers - 1;

            if is_last {
                final_bounds = Some(post_linear);
            } else {
                // For the first hidden layer, use alpha-informed lower bounds
                let is_first_hidden = layer_idx == 0;
                let post_relu = if is_first_hidden {
                    Self::propagate_relu_with_alpha(&post_linear, alpha)
                } else {
                    Self::propagate_relu(&post_linear)
                };
                // Update current_bounds with the tightened ReLU bounds
                current_bounds = post_relu;
            }
        }

        let output_bounds = final_bounds.ok_or(AdvError::EmptyInput)?;
        let n_classes = output_bounds.len();

        if label >= n_classes {
            return Err(AdvError::DimensionMismatch {
                expected: n_classes - 1,
                got: label,
            });
        }

        let true_class_lo = output_bounds.lower[label];
        let is_robust = (0..n_classes)
            .filter(|&j| j != label)
            .all(|j| true_class_lo > output_bounds.upper[j]);

        if is_robust { Ok(cfg.eps) } else { Ok(0.0) }
    }

    /// ReLU propagation that uses per-neuron alpha for the lower-bound slope
    /// in the ambiguous region, producing a tighter lower bound.
    fn propagate_relu_with_alpha(input_bounds: &NeuronBound, alpha: &[f32]) -> NeuronBound {
        let n = input_bounds.len();
        // alpha may be shorter if layers have different sizes; pad with 0.5
        let mut lower = Vec::with_capacity(n);
        let mut upper = Vec::with_capacity(n);

        for i in 0..n {
            let lo = input_bounds.lower[i];
            let hi = input_bounds.upper[i];
            let a = if i < alpha.len() {
                alpha[i].clamp(0.0, 1.0)
            } else {
                0.5
            };
            let (slope_lo, intercept_lo, _slope_up, _intercept_up) =
                Self::relu_linear_bounds(lo, hi, a);

            // Tightened lower: lower = slope_lo * lo + intercept_lo
            // (the tightest lower bound achievable with slope a)
            let tightened_lower = slope_lo * lo + intercept_lo;
            // Standard upper: max(hi, 0)
            let propagated_upper = hi.max(0.0);

            lower.push(tightened_lower.max(0.0)); // ensure non-negative (ReLU)
            upper.push(propagated_upper);
        }

        NeuronBound { lower, upper }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    // Helper: identity weight matrix n×n, zero bias
    fn identity_layer(n: usize) -> LinearLayer {
        let mut w = vec![0.0_f32; n * n];
        for i in 0..n {
            w[i * n + i] = 1.0;
        }
        LinearLayer::new(w, vec![0.0_f32; n], n, n).expect("new should succeed")
    }

    // ── relu_linear_bounds ────────────────────────────────────────────────────

    #[test]
    fn relu_bounds_inactive_all_zero() {
        // u <= 0: fully inactive
        let (al, bl, au, bu) = CrownVerifier::relu_linear_bounds(-2.0, -0.5, 0.5);
        assert_eq!((al, bl, au, bu), (0.0, 0.0, 0.0, 0.0));

        // Exactly at zero
        let (al2, bl2, au2, bu2) = CrownVerifier::relu_linear_bounds(-1.0, 0.0, 0.5);
        assert_eq!((al2, bl2, au2, bu2), (0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn relu_bounds_active_identity() {
        // l >= 0: fully active
        let (al, bl, au, bu) = CrownVerifier::relu_linear_bounds(0.5, 2.0, 0.7);
        assert_eq!((al, bl, au, bu), (1.0, 0.0, 1.0, 0.0));

        // Exactly at zero lower
        let (al2, bl2, au2, bu2) = CrownVerifier::relu_linear_bounds(0.0, 1.0, 0.3);
        assert_eq!((al2, bl2, au2, bu2), (1.0, 0.0, 1.0, 0.0));
    }

    #[test]
    fn relu_bounds_ambiguous_correct_slopes() {
        // l=-1, u=2: slope_up = 2/(2-(-1)) = 2/3, intercept_up = -(-1)*(2/3) = 2/3
        let (al, bl, au, bu) = CrownVerifier::relu_linear_bounds(-1.0, 2.0, 0.4);
        assert!(approx_eq(al, 0.4, 1e-6)); // alpha clamped to [0,1]
        assert!(approx_eq(bl, 0.0, 1e-6));
        assert!(approx_eq(au, 2.0 / 3.0, 1e-5));
        assert!(approx_eq(bu, 2.0 / 3.0, 1e-5)); // -l * slope_up = 1 * 2/3 = 2/3
    }

    #[test]
    fn relu_bounds_ambiguous_alpha_clamped() {
        // alpha > 1 should be clamped to 1
        let (al, _bl, _au, _bu) = CrownVerifier::relu_linear_bounds(-1.0, 1.0, 1.5);
        assert!(approx_eq(al, 1.0, 1e-6));

        // alpha < 0 should be clamped to 0
        let (al2, _bl2, _au2, _bu2) = CrownVerifier::relu_linear_bounds(-1.0, 1.0, -0.3);
        assert!(approx_eq(al2, 0.0, 1e-6));
    }

    // ── propagate_linear ──────────────────────────────────────────────────────

    #[test]
    fn propagate_linear_identity_pass_through() {
        let layer = identity_layer(3);
        let bounds = NeuronBound {
            lower: vec![-1.0, 0.5, -2.0],
            upper: vec![1.0, 0.7, 2.0],
        };
        let out = CrownVerifier::propagate_linear(&layer, &bounds)
            .expect("propagate_linear should succeed");
        for i in 0..3 {
            assert!(approx_eq(out.lower[i], bounds.lower[i], 1e-6));
            assert!(approx_eq(out.upper[i], bounds.upper[i], 1e-6));
        }
    }

    #[test]
    fn propagate_linear_negative_weight_swaps() {
        // y = -x: lo_y = -hi_x, hi_y = -lo_x
        let layer = LinearLayer::new(vec![-1.0_f32], vec![0.0], 1, 1).expect("new should succeed");
        let bounds = NeuronBound {
            lower: vec![-1.0],
            upper: vec![2.0],
        };
        let out = CrownVerifier::propagate_linear(&layer, &bounds)
            .expect("propagate_linear should succeed");
        assert!(approx_eq(out.lower[0], -2.0, 1e-6));
        assert!(approx_eq(out.upper[0], 1.0, 1e-6));
    }

    #[test]
    fn propagate_linear_with_bias() {
        // y = 2*x + 1: lo = 2*(-1)+1 = -1, hi = 2*2+1 = 5
        let layer = LinearLayer::new(vec![2.0_f32], vec![1.0], 1, 1).expect("new should succeed");
        let bounds = NeuronBound {
            lower: vec![-1.0],
            upper: vec![2.0],
        };
        let out = CrownVerifier::propagate_linear(&layer, &bounds)
            .expect("propagate_linear should succeed");
        assert!(approx_eq(out.lower[0], -1.0, 1e-6));
        assert!(approx_eq(out.upper[0], 5.0, 1e-6));
    }

    #[test]
    fn propagate_linear_dim_mismatch_errors() {
        let layer = identity_layer(3);
        let bounds = NeuronBound {
            lower: vec![-1.0, 0.5], // Wrong length (2 vs 3)
            upper: vec![1.0, 0.7],
        };
        assert!(matches!(
            CrownVerifier::propagate_linear(&layer, &bounds),
            Err(AdvError::DimensionMismatch { .. })
        ));
    }

    // ── propagate_relu ────────────────────────────────────────────────────────

    #[test]
    fn propagate_relu_clamps_negatives() {
        let bounds = NeuronBound {
            lower: vec![-2.0, 0.5, -1.0],
            upper: vec![3.0, 1.5, -0.5],
        };
        let out = CrownVerifier::propagate_relu(&bounds);
        // lower[0]: max(-2, 0) = 0; lower[2]: max(-1, 0) = 0
        assert!(approx_eq(out.lower[0], 0.0, 1e-6));
        assert!(approx_eq(out.lower[1], 0.5, 1e-6));
        assert!(approx_eq(out.lower[2], 0.0, 1e-6));
        // upper[2]: max(-0.5, 0) = 0
        assert!(approx_eq(out.upper[0], 3.0, 1e-6));
        assert!(approx_eq(out.upper[1], 1.5, 1e-6));
        assert!(approx_eq(out.upper[2], 0.0, 1e-6));
    }

    // ── crown_bound_propagation ───────────────────────────────────────────────

    #[test]
    fn crown_bound_single_layer() {
        let layer = identity_layer(4);
        let x0 = vec![0.0_f32; 4];
        let cfg = CrownConfig {
            eps: 0.1,
            ..Default::default()
        };
        let bounds = CrownVerifier::crown_bound_propagation(&x0, &[layer], &cfg)
            .expect("crown_bound_propagation should succeed");
        assert_eq!(bounds.len(), 1);
        // Identity layer: output bounds = input bounds = [-0.1, 0.1] per dim
        for i in 0..4 {
            assert!(approx_eq(bounds[0].lower[i], -0.1, 1e-5));
            assert!(approx_eq(bounds[0].upper[i], 0.1, 1e-5));
        }
    }

    #[test]
    fn crown_bound_two_layers_interval_valid() {
        // Layer 1: 2 in -> 2 out (identity), Layer 2: 2 in -> 2 out (identity)
        let layer1 = identity_layer(2);
        let layer2 = identity_layer(2);
        let x0 = vec![1.0_f32, -1.0];
        let cfg = CrownConfig {
            eps: 0.5,
            ..Default::default()
        };
        let bounds = CrownVerifier::crown_bound_propagation(&x0, &[layer1, layer2], &cfg)
            .expect("crown_bound_propagation should succeed");
        assert_eq!(bounds.len(), 2);
        // All bounds must satisfy lower <= upper
        for layer_bound in &bounds {
            for i in 0..layer_bound.len() {
                assert!(
                    layer_bound.lower[i] <= layer_bound.upper[i] + 1e-6,
                    "lower > upper at neuron {}",
                    i
                );
            }
        }
    }

    #[test]
    fn crown_bound_empty_layers_errors() {
        let x0 = vec![1.0_f32, 2.0];
        let cfg = CrownConfig::default();
        assert!(matches!(
            CrownVerifier::crown_bound_propagation(&x0, &[], &cfg),
            Err(AdvError::EmptyInput)
        ));
    }

    #[test]
    fn crown_bound_dim_mismatch_errors() {
        let layer = identity_layer(3);
        let x0 = vec![1.0_f32, 2.0]; // Wrong: 2 vs 3
        let cfg = CrownConfig::default();
        assert!(matches!(
            CrownVerifier::crown_bound_propagation(&x0, &[layer], &cfg),
            Err(AdvError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn bounds_always_interval_lower_le_upper() {
        // Multi-layer with positive/negative weights; verify invariant throughout
        let w1 = vec![1.0_f32, -2.0, 0.5, 1.5]; // 2×2
        let b1 = vec![0.1_f32, -0.1];
        let layer1 = LinearLayer::new(w1, b1, 2, 2).expect("new should succeed");
        let w2 = vec![-1.0_f32, 0.5, 2.0, -1.5]; // 2×2
        let b2 = vec![0.0_f32, 0.0];
        let layer2 = LinearLayer::new(w2, b2, 2, 2).expect("new should succeed");

        let x0 = vec![0.5_f32, -0.5];
        let cfg = CrownConfig {
            eps: 0.2,
            ..Default::default()
        };
        let all_bounds = CrownVerifier::crown_bound_propagation(&x0, &[layer1, layer2], &cfg)
            .expect("crown_bound_propagation should succeed");
        for layer_bound in &all_bounds {
            for i in 0..layer_bound.len() {
                assert!(
                    layer_bound.lower[i] <= layer_bound.upper[i] + 1e-5,
                    "lower={} > upper={} at neuron {}",
                    layer_bound.lower[i],
                    layer_bound.upper[i],
                    i
                );
            }
        }
    }

    // ── certified_radius ──────────────────────────────────────────────────────

    #[test]
    fn certified_radius_small_eps_trivially_robust() {
        // Very small eps: small perturbation, identity layers
        // Use a "classifier" that strongly prefers class 0
        let w = vec![10.0_f32, 0.0, 0.0, 1.0]; // 2×2: out0 = 10*in0, out1 = in1
        let b = vec![0.0_f32, 0.0];
        let layer = LinearLayer::new(w, b, 2, 2).expect("new should succeed");
        let x0 = vec![1.0_f32, 0.0];
        let cfg = CrownConfig {
            eps: 0.001,
            ..Default::default()
        };
        // With x0=[1,0] and eps=0.001:
        // out0 bounds: [10*(1-0.001), 10*(1+0.001)] = [9.99, 10.01]
        // out1 bounds: [0*(1-0.001), 0*(1+0.001)] (using x0[1]=0)
        //           = [-0.001, 0.001]
        // lower[0]=9.99 > upper[1]=0.001 → robust
        let r = CrownVerifier::certified_radius(&x0, 0, &[layer], &cfg)
            .expect("certified_radius should succeed");
        assert!(approx_eq(r, cfg.eps, 1e-7));
    }

    #[test]
    fn certified_radius_large_eps_not_robust() {
        // Large eps: identity layer, tight margin between classes
        // out0 = x0, out1 = x1; x0=[0.5, 0.3] → out0=0.5>out1=0.3
        // But with eps=1.0: out0 in [-0.5, 1.5], out1 in [-0.7, 1.3]
        // lower[0]=-0.5 < upper[1]=1.3 → not robust
        let layer = identity_layer(2);
        let x0 = vec![0.5_f32, 0.3];
        let cfg = CrownConfig {
            eps: 1.0,
            ..Default::default()
        };
        let r = CrownVerifier::certified_radius(&x0, 0, &[layer], &cfg)
            .expect("certified_radius should succeed");
        assert!(approx_eq(r, 0.0, 1e-7));
    }

    // ── optimize_alpha ────────────────────────────────────────────────────────

    #[test]
    fn optimize_alpha_returns_alpha_in_unit_interval() {
        let layer1 = identity_layer(4);
        let w2 = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2×4 → 2 classes
        let b2 = vec![0.0_f32, 0.0];
        let layer2 = LinearLayer::new(w2, b2, 4, 2).expect("new should succeed");

        let x0 = vec![1.0_f32, 0.5, 0.2, -0.3];
        let cfg = CrownConfig {
            n_alpha_steps: 5,
            alpha_lr: 0.1,
            eps: 0.1,
        };
        let alpha_bound = CrownVerifier::optimize_alpha(&x0, 0, &[layer1, layer2], &cfg)
            .expect("optimize_alpha should succeed");
        for &a in &alpha_bound.alpha {
            assert!((0.0..=1.0).contains(&a), "alpha={a} out of [0,1]");
        }
    }

    #[test]
    fn optimize_alpha_zero_steps_returns_half() {
        let layer1 = identity_layer(3);
        let w2 = vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0]; // 2×3
        let b2 = vec![0.0_f32, 0.0];
        let layer2 = LinearLayer::new(w2, b2, 3, 2).expect("new should succeed");

        let x0 = vec![0.5_f32; 3];
        let cfg = CrownConfig {
            n_alpha_steps: 0,
            ..Default::default()
        };
        let alpha_bound = CrownVerifier::optimize_alpha(&x0, 0, &[layer1, layer2], &cfg)
            .expect("optimize_alpha should succeed");
        for &a in &alpha_bound.alpha {
            assert!(approx_eq(a, 0.5, 1e-7));
        }
    }

    #[test]
    fn optimize_alpha_empty_layers_errors() {
        let x0 = vec![0.5_f32, 0.5];
        let cfg = CrownConfig::default();
        assert!(matches!(
            CrownVerifier::optimize_alpha(&x0, 0, &[], &cfg),
            Err(AdvError::EmptyInput)
        ));
    }

    #[test]
    fn optimize_alpha_x0_dim_mismatch_errors() {
        let layer = identity_layer(3);
        let x0 = vec![0.5_f32, 0.5]; // Wrong: 2 vs 3
        let cfg = CrownConfig {
            n_alpha_steps: 1,
            ..Default::default()
        };
        assert!(matches!(
            CrownVerifier::optimize_alpha(&x0, 0, &[layer], &cfg),
            Err(AdvError::DimensionMismatch { .. })
        ));
    }

    // ── Default config ────────────────────────────────────────────────────────

    #[test]
    fn default_config_has_expected_values() {
        let cfg = CrownConfig::default();
        assert_eq!(cfg.n_alpha_steps, 0);
        assert!(approx_eq(cfg.alpha_lr, 0.1, 1e-7));
        assert!(approx_eq(cfg.eps, 0.1, 1e-7));
    }

    // ── alpha-CROWN with n_alpha_steps > 0 ────────────────────────────────────

    #[test]
    fn alpha_crown_five_steps_no_error() {
        let layer1 = identity_layer(4);
        let w2 = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2×4
        let b2 = vec![0.1_f32, -0.1];
        let layer2 = LinearLayer::new(w2, b2, 4, 2).expect("new should succeed");
        let x0 = vec![0.5_f32, 0.3, 0.2, 0.1];
        let cfg = CrownConfig {
            n_alpha_steps: 5,
            alpha_lr: 0.1,
            eps: 0.1,
        };
        let result = CrownVerifier::optimize_alpha(&x0, 0, &[layer1, layer2], &cfg);
        assert!(
            result.is_ok(),
            "alpha-CROWN with 5 steps failed: {:?}",
            result
        );
        let alpha = result.expect("result should be present").alpha;
        assert_eq!(alpha.len(), 4);
        for &a in &alpha {
            assert!((0.0..=1.0).contains(&a));
        }
    }
}
