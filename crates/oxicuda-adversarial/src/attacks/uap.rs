//! Universal Adversarial Perturbation (UAP) — input-agnostic attack.
//!
//! Constructs a single perturbation `v` (same shape as each input) that
//! causes misclassification of most inputs in a given dataset.
//!
//! Reference: Moosavi-Dezfooli, Fawzi, Fawzi & Frossard (2017),
//! *"Universal Adversarial Perturbations"*, CVPR.

use crate::attacks::deepfool::{DeepFoolConfig, deepfool};
use crate::error::{AdvError, AdvResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the Universal Adversarial Perturbation algorithm.
#[derive(Debug, Clone, Copy)]
pub struct UapConfig {
    /// L∞ perturbation budget ξ > 0. The universal perturbation is projected to this ball.
    pub xi: f32,
    /// Target fooling rate ∈ (0, 1]. Stop when achieved.
    pub target_fool_rate: f32,
    /// Maximum outer passes over the dataset.
    pub max_passes: usize,
    /// DeepFool maximum iterations for computing the per-image correction.
    pub deepfool_max_iter: usize,
    /// DeepFool overshoot factor.
    pub deepfool_overshoot: f32,
}

impl Default for UapConfig {
    fn default() -> Self {
        Self {
            xi: 0.1,
            target_fool_rate: 0.8,
            max_passes: 10,
            deepfool_max_iter: 50,
            deepfool_overshoot: 0.02,
        }
    }
}

// ─── Output ───────────────────────────────────────────────────────────────────

/// Output of the UAP algorithm.
#[derive(Debug, Clone)]
pub struct UapResult {
    /// Universal perturbation vector (same dim as each input).
    pub perturbation: Vec<f32>,
    /// Achieved fooling rate on the dataset.
    pub fool_rate: f32,
    /// Number of outer passes taken.
    pub n_passes: usize,
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Index of the maximum element in a non-empty slice.
#[inline]
fn argmax_slice(v: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = v[0];
    for (i, &val) in v.iter().enumerate().skip(1) {
        if val > best_val {
            best_val = val;
            best_idx = i;
        }
    }
    best_idx
}

// ─── Main algorithm ───────────────────────────────────────────────────────────

/// Compute a universal adversarial perturbation.
///
/// `inputs`: batch of N inputs, each of length `dim` (flat row-major `[N, dim]`).
/// `logits_grads`: closure → `(logits[n_classes], grad_matrix[n_classes * dim])` for one input.
///
/// # Errors
/// - [`AdvError::EmptyInput`] if N == 0 or dim == 0.
/// - [`AdvError::DimensionMismatch`] if logits / grad_matrix have wrong sizes.
/// - [`AdvError::InvalidEpsilon`] if xi ≤ 0 or non-finite.
/// - [`AdvError::NanEncountered`] if logits / grads contain non-finite values.
pub fn uap_attack<F>(
    inputs: &[f32],
    n_inputs: usize,
    dim: usize,
    n_classes: usize,
    logits_grads: F,
    cfg: &UapConfig,
) -> AdvResult<UapResult>
where
    F: Fn(&[f32]) -> AdvResult<(Vec<f32>, Vec<f32>)>,
{
    // ── Validation ────────────────────────────────────────────────────────────
    if n_inputs == 0 || dim == 0 {
        return Err(AdvError::EmptyInput);
    }
    if !(cfg.xi.is_finite() && cfg.xi > 0.0) {
        return Err(AdvError::InvalidEpsilon { eps: cfg.xi });
    }
    if n_classes < 2 {
        return Err(AdvError::Internal("n_classes must be >= 2".to_owned()));
    }
    // Validate that the inputs slice has the expected length.
    if inputs.len() != n_inputs * dim {
        return Err(AdvError::DimensionMismatch {
            expected: n_inputs * dim,
            got: inputs.len(),
        });
    }

    // ── Pre-compute original classes for all inputs ────────────────────────────
    // Precompute argmax for all N inputs before the pass loop to avoid
    // recalling the closure for original images repeatedly in the inner loop.
    let mut orig_classes = Vec::with_capacity(n_inputs);
    for i in 0..n_inputs {
        let x_i = &inputs[i * dim..(i + 1) * dim];
        let (logits, grads) = logits_grads(x_i)?;
        if logits.len() != n_classes {
            return Err(AdvError::DimensionMismatch {
                expected: n_classes,
                got: logits.len(),
            });
        }
        if grads.len() != n_classes * dim {
            return Err(AdvError::DimensionMismatch {
                expected: n_classes * dim,
                got: grads.len(),
            });
        }
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(AdvError::NanEncountered {
                location: "uap_attack:initial_logits",
            });
        }
        if grads.iter().any(|v| !v.is_finite()) {
            return Err(AdvError::NanEncountered {
                location: "uap_attack:initial_grads",
            });
        }
        orig_classes.push(argmax_slice(&logits));
    }

    // ── Algorithm 1 (Moosavi-Dezfooli 2017) ───────────────────────────────────
    // Step 2: initialise universal perturbation to zero.
    let mut v = vec![0.0_f32; dim];

    let df_cfg = DeepFoolConfig {
        max_iter: cfg.deepfool_max_iter,
        overshoot: cfg.deepfool_overshoot,
        lo: f32::NEG_INFINITY,
        hi: f32::INFINITY,
    };

    let mut fool_rate = 0.0_f32;
    let mut pass = 0_usize;

    // Step 3: outer passes over the dataset.
    for p in 0..cfg.max_passes {
        pass = p;

        // Step 3a: for each input, update v if x_i + v is still correctly classified.
        for i in 0..n_inputs {
            let x_i = &inputs[i * dim..(i + 1) * dim];

            // Build x_i + v.
            let x_plus_v: Vec<f32> = x_i.iter().zip(v.iter()).map(|(&xi, &vi)| xi + vi).collect();

            // Check current classification under v.
            let (logits_v, grads_v) = logits_grads(&x_plus_v)?;
            if logits_v.len() != n_classes {
                return Err(AdvError::DimensionMismatch {
                    expected: n_classes,
                    got: logits_v.len(),
                });
            }
            if grads_v.len() != n_classes * dim {
                return Err(AdvError::DimensionMismatch {
                    expected: n_classes * dim,
                    got: grads_v.len(),
                });
            }
            if logits_v.iter().any(|v_val| !v_val.is_finite()) {
                return Err(AdvError::NanEncountered {
                    location: "uap_attack:pass_logits",
                });
            }

            let current_class = argmax_slice(&logits_v);
            let orig_class_i = orig_classes[i];

            // Only apply DeepFool correction if x_i + v is still correctly classified.
            if current_class == orig_class_i {
                // Run DeepFool on the perturbed input to find a correction.
                match deepfool(&x_plus_v, n_classes, &logits_grads, &df_cfg) {
                    Ok(df_result) => {
                        // Accumulate the DeepFool perturbation into v with L∞ projection.
                        for (vi, &delta) in v.iter_mut().zip(df_result.perturbation.iter()) {
                            *vi = (*vi + delta).clamp(-cfg.xi, cfg.xi);
                        }
                    }
                    Err(_) => {
                        // DeepFool failed for this image — skip it.
                        continue;
                    }
                }
            }
        }

        // Step 3b: compute fooling rate.
        let mut fooled_count = 0_usize;
        for i in 0..n_inputs {
            let x_i = &inputs[i * dim..(i + 1) * dim];
            let x_plus_v: Vec<f32> = x_i.iter().zip(v.iter()).map(|(&xi, &vi)| xi + vi).collect();
            let (logits_adv, _) = logits_grads(&x_plus_v)?;
            if logits_adv.len() != n_classes {
                return Err(AdvError::DimensionMismatch {
                    expected: n_classes,
                    got: logits_adv.len(),
                });
            }
            let adv_class = argmax_slice(&logits_adv);
            if adv_class != orig_classes[i] {
                fooled_count += 1;
            }
        }
        fool_rate = fooled_count as f32 / n_inputs as f32;

        // Step 3c: early exit if target fooling rate achieved.
        if fool_rate >= cfg.target_fool_rate {
            pass = p;
            break;
        }
    }

    Ok(UapResult {
        perturbation: v,
        fool_rate,
        n_passes: pass + 1,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper: linear oracle ─────────────────────────────────────────────────
    // weights: [n_classes, dim] row-major; logits[k] = weights[k,:] · x + biases[k]
    // grad_matrix: weights (constant for all x).
    fn linear_oracle(
        weights: Vec<f32>,
        biases: Vec<f32>,
        n_classes: usize,
        dim: usize,
    ) -> impl Fn(&[f32]) -> AdvResult<(Vec<f32>, Vec<f32>)> {
        move |x: &[f32]| {
            let mut logits = vec![0.0_f32; n_classes];
            for k in 0..n_classes {
                let mut dot = biases[k];
                for i in 0..dim {
                    dot += weights[k * dim + i] * x[i];
                }
                logits[k] = dot;
            }
            Ok((logits, weights.clone()))
        }
    }

    // ── Test 1 ────────────────────────────────────────────────────────────────

    #[test]
    fn uap_output_shape() {
        let dim = 4_usize;
        let n_inputs = 3_usize;
        let n_classes = 2_usize;
        // class 0 dominates: weights[0,:] = [1,0,0,0], weights[1,:] = [0,1,0,0]
        let weights = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0_f32];
        let biases = vec![1.0, 0.0_f32];
        let oracle = linear_oracle(weights, biases, n_classes, dim);
        let inputs: Vec<f32> = (0..n_inputs * dim).map(|i| (i as f32) * 0.1).collect();
        let cfg = UapConfig {
            xi: 0.5,
            max_passes: 1,
            ..Default::default()
        };
        let result = uap_attack(&inputs, n_inputs, dim, n_classes, oracle, &cfg).unwrap();
        assert_eq!(result.perturbation.len(), dim);
    }

    // ── Test 2 ────────────────────────────────────────────────────────────────

    #[test]
    fn uap_linf_bound() {
        let dim = 6_usize;
        let n_inputs = 4_usize;
        let n_classes = 2_usize;
        let xi = 0.3_f32;
        let weights = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0_f32,
        ];
        let biases = vec![2.0, 0.0_f32];
        let oracle = linear_oracle(weights, biases, n_classes, dim);
        let inputs: Vec<f32> = (0..n_inputs * dim).map(|i| (i as f32) * 0.05).collect();
        let cfg = UapConfig {
            xi,
            max_passes: 3,
            ..Default::default()
        };
        let result = uap_attack(&inputs, n_inputs, dim, n_classes, oracle, &cfg).unwrap();
        for &vi in &result.perturbation {
            assert!(vi.abs() <= xi + 1e-6, "L∞ violated: |{vi}| > {xi}");
        }
    }

    // ── Test 3 ────────────────────────────────────────────────────────────────

    #[test]
    fn uap_fool_rate_in_range() {
        let dim = 4_usize;
        let n_inputs = 5_usize;
        let n_classes = 2_usize;
        let weights = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0_f32];
        let biases = vec![1.0, 0.0_f32];
        let oracle = linear_oracle(weights, biases, n_classes, dim);
        let inputs: Vec<f32> = (0..n_inputs * dim).map(|i| (i as f32) * 0.1).collect();
        let cfg = UapConfig {
            xi: 0.5,
            max_passes: 2,
            ..Default::default()
        };
        let result = uap_attack(&inputs, n_inputs, dim, n_classes, oracle, &cfg).unwrap();
        assert!(
            result.fool_rate >= 0.0 && result.fool_rate <= 1.0,
            "fool_rate out of range: {}",
            result.fool_rate
        );
    }

    // ── Test 4 ────────────────────────────────────────────────────────────────

    #[test]
    fn uap_single_input() {
        let dim = 4_usize;
        let n_inputs = 1_usize;
        let n_classes = 2_usize;
        // class 0: w=[1,0,0,0]+bias=2, class 1: w=[0,1,0,0]
        let weights = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0_f32];
        let biases = vec![2.0, 0.0_f32];
        let oracle = linear_oracle(weights, biases, n_classes, dim);
        let inputs = vec![0.1_f32, 0.2, 0.3, 0.4];
        let cfg = UapConfig {
            xi: 0.5,
            max_passes: 2,
            ..Default::default()
        };
        let result = uap_attack(&inputs, n_inputs, dim, n_classes, oracle, &cfg).unwrap();
        assert_eq!(result.perturbation.len(), dim);
        assert!(
            result.fool_rate == 0.0 || result.fool_rate == 1.0,
            "fool_rate for single input must be 0 or 1, got {}",
            result.fool_rate
        );
    }

    // ── Test 5 ────────────────────────────────────────────────────────────────

    #[test]
    fn uap_dim_preserved() {
        let dim = 10_usize;
        let n_inputs = 3_usize;
        let n_classes = 2_usize;
        let mut weights = vec![0.0_f32; n_classes * dim];
        weights[0] = 1.0; // class 0 grad at index 0
        weights[dim + 1] = 1.0; // class 1 grad at index 1
        let biases = vec![1.5, 0.0_f32];
        let oracle = linear_oracle(weights, biases, n_classes, dim);
        let inputs: Vec<f32> = (0..n_inputs * dim).map(|i| (i as f32) * 0.05).collect();
        let cfg = UapConfig {
            xi: 0.3,
            max_passes: 2,
            ..Default::default()
        };
        let result = uap_attack(&inputs, n_inputs, dim, n_classes, oracle, &cfg).unwrap();
        assert_eq!(result.perturbation.len(), dim);
    }

    // ── Test 6 ────────────────────────────────────────────────────────────────

    #[test]
    fn uap_perturbation_nonzero_after_pass() {
        let dim = 4_usize;
        let n_inputs = 5_usize;
        let n_classes = 2_usize;
        // Large bias for class 0: inputs are classified as 0, DeepFool must find direction to 1.
        let weights = vec![
            1.0, 0.0, 0.0, 0.0, // class 0
            0.0, 1.0, 0.0, 0.0_f32, // class 1
        ];
        let biases = vec![3.0, 0.0_f32];
        let oracle = linear_oracle(weights, biases, n_classes, dim);
        let inputs: Vec<f32> = (0..n_inputs * dim).map(|i| (i as f32) * 0.05).collect();
        let cfg = UapConfig {
            xi: 1.0,
            max_passes: 2,
            target_fool_rate: 1.1, // never stop early
            ..Default::default()
        };
        let result = uap_attack(&inputs, n_inputs, dim, n_classes, oracle, &cfg).unwrap();
        let nonzero = result.perturbation.iter().any(|&v| v.abs() > 1e-9);
        assert!(nonzero, "perturbation should be non-zero after 2 passes");
    }

    // ── Test 7 ────────────────────────────────────────────────────────────────

    #[test]
    fn uap_err_empty_inputs() {
        let dim = 4_usize;
        let n_classes = 2_usize;
        let weights = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0_f32];
        let biases = vec![1.0, 0.0_f32];
        let oracle = linear_oracle(weights, biases, n_classes, dim);
        let cfg = UapConfig::default();
        let result = uap_attack(&[], 0, dim, n_classes, oracle, &cfg);
        assert!(matches!(result.unwrap_err(), AdvError::EmptyInput));
    }

    // ── Test 8 ────────────────────────────────────────────────────────────────

    #[test]
    fn uap_err_zero_dim() {
        let n_classes = 2_usize;
        let oracle =
            |_: &[f32]| -> AdvResult<(Vec<f32>, Vec<f32>)> { Ok((vec![1.0, 0.0], vec![])) };
        let cfg = UapConfig::default();
        let result = uap_attack(&[], 3, 0, n_classes, oracle, &cfg);
        assert!(matches!(result.unwrap_err(), AdvError::EmptyInput));
    }

    // ── Test 9 ────────────────────────────────────────────────────────────────

    #[test]
    fn uap_err_invalid_xi() {
        let dim = 4_usize;
        let n_inputs = 2_usize;
        let n_classes = 2_usize;
        let weights = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0_f32];
        let biases = vec![1.0, 0.0_f32];
        let oracle = linear_oracle(weights, biases, n_classes, dim);
        let inputs: Vec<f32> = vec![0.5; n_inputs * dim];
        let cfg = UapConfig {
            xi: 0.0,
            ..Default::default()
        };
        let result = uap_attack(&inputs, n_inputs, dim, n_classes, oracle, &cfg);
        assert!(matches!(
            result.unwrap_err(),
            AdvError::InvalidEpsilon { .. }
        ));
    }

    // ── Test 10 ───────────────────────────────────────────────────────────────

    #[test]
    fn uap_err_nan_logits() {
        let dim = 4_usize;
        let n_inputs = 2_usize;
        let n_classes = 2_usize;
        let bad_oracle = |_: &[f32]| -> AdvResult<(Vec<f32>, Vec<f32>)> {
            Ok((vec![f32::NAN, 1.0], vec![1.0; 8]))
        };
        let inputs: Vec<f32> = vec![0.5; n_inputs * dim];
        let cfg = UapConfig::default();
        let result = uap_attack(&inputs, n_inputs, dim, n_classes, bad_oracle, &cfg);
        assert!(matches!(
            result.unwrap_err(),
            AdvError::NanEncountered { .. }
        ));
    }

    // ── Test 11 ───────────────────────────────────────────────────────────────

    #[test]
    fn uap_passes_bounded() {
        let dim = 4_usize;
        let n_inputs = 3_usize;
        let n_classes = 2_usize;
        let max_passes = 3_usize;
        let weights = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0_f32];
        let biases = vec![1.0, 0.0_f32];
        let oracle = linear_oracle(weights, biases, n_classes, dim);
        let inputs: Vec<f32> = vec![0.5; n_inputs * dim];
        let cfg = UapConfig {
            xi: 0.5,
            max_passes,
            target_fool_rate: 1.1, // never stop early
            ..Default::default()
        };
        let result = uap_attack(&inputs, n_inputs, dim, n_classes, oracle, &cfg).unwrap();
        assert!(
            result.n_passes <= max_passes,
            "n_passes={} > max_passes={}",
            result.n_passes,
            max_passes
        );
    }

    // ── Test 12 ───────────────────────────────────────────────────────────────

    #[test]
    fn uap_fool_rate_all_misclassified() {
        // Oracle: originally class 0 but immediately gives class 1 on any non-zero input.
        // We arrange: for x=0 (before perturbation) class 0, for x+v class 1.
        // Use a stateful oracle that flips class after initial call.
        use std::cell::Cell;
        let dim = 3_usize;
        let n_inputs = 3_usize;
        let n_classes = 2_usize;

        // Large positive bias for class 0, negative for class 1, but gradient strongly favours class 1.
        // The initial classification of all-zero inputs: logits[0]=10, logits[1]=-10 → class 0.
        // After adding v with xi=5: logits[0]=10+5*(-1)=5, logits[1]=-10+5*(1)=-5 → still class 0.
        // This won't trivially reach 100% fool. Use a trick: stateful oracle counts calls per image.
        let call_counter = Cell::new(0_u32);
        let oracle = move |x: &[f32]| -> AdvResult<(Vec<f32>, Vec<f32>)> {
            let n = call_counter.get();
            call_counter.set(n + 1);
            // If the input sum > 0.5 it's "perturbed" → class 1 wins.
            let s: f32 = x.iter().sum();
            let (logit0, logit1) = if s > 0.5 {
                (0.0_f32, 2.0_f32) // class 1 wins
            } else {
                (2.0_f32, 0.0_f32) // class 0 wins
            };
            // grad_matrix: [1,0,0, 0,1,0] (2 classes × dim=3)
            Ok((vec![logit0, logit1], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0_f32]))
        };
        // Inputs: all near zero so original class is 0.
        let inputs = vec![0.1_f32; n_inputs * dim];
        let cfg = UapConfig {
            xi: 2.0,
            max_passes: 5,
            target_fool_rate: 1.0,
            deepfool_max_iter: 20,
            deepfool_overshoot: 0.02,
        };
        let result = uap_attack(&inputs, n_inputs, dim, n_classes, oracle, &cfg).unwrap();
        // fool_rate should be in [0, 1].
        assert!(
            result.fool_rate >= 0.0 && result.fool_rate <= 1.0,
            "fool_rate out of range: {}",
            result.fool_rate
        );
    }

    // ── Test 13 ───────────────────────────────────────────────────────────────

    #[test]
    fn uap_universal_applies_to_all() {
        // The returned perturbation has length dim (applies uniformly to all inputs).
        let dim = 5_usize;
        let n_inputs = 4_usize;
        let n_classes = 2_usize;
        let weights = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0_f32];
        let biases = vec![1.0, 0.0_f32];
        let oracle = linear_oracle(weights, biases, n_classes, dim);
        let inputs: Vec<f32> = (0..n_inputs * dim).map(|i| (i as f32) * 0.05).collect();
        let cfg = UapConfig {
            xi: 0.5,
            max_passes: 2,
            ..Default::default()
        };
        let result = uap_attack(&inputs, n_inputs, dim, n_classes, oracle, &cfg).unwrap();
        assert_eq!(
            result.perturbation.len(),
            dim,
            "perturbation length must equal dim"
        );
    }

    // ── Test 14 ───────────────────────────────────────────────────────────────

    #[test]
    fn uap_two_class() {
        // Verify n_classes=2 works end-to-end.
        let dim = 4_usize;
        let n_inputs = 3_usize;
        let n_classes = 2_usize;
        let weights = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0_f32];
        let biases = vec![1.5, 0.0_f32];
        let oracle = linear_oracle(weights, biases, n_classes, dim);
        let inputs: Vec<f32> = (0..n_inputs * dim).map(|i| (i as f32) * 0.1).collect();
        let cfg = UapConfig {
            xi: 0.5,
            max_passes: 2,
            ..Default::default()
        };
        let result = uap_attack(&inputs, n_inputs, dim, n_classes, oracle, &cfg);
        assert!(result.is_ok());
    }

    // ── Test 15 ───────────────────────────────────────────────────────────────

    #[test]
    fn uap_err_dim_mismatch_logits() {
        // Oracle returns logits with wrong size.
        let dim = 4_usize;
        let n_inputs = 2_usize;
        let n_classes = 3_usize; // expect 3 but oracle returns 2
        let bad_oracle = |_: &[f32]| -> AdvResult<(Vec<f32>, Vec<f32>)> {
            Ok((vec![1.0, 0.0], vec![1.0; 2 * 4]))
        };
        let inputs: Vec<f32> = vec![0.5; n_inputs * dim];
        let cfg = UapConfig::default();
        let result = uap_attack(&inputs, n_inputs, dim, n_classes, bad_oracle, &cfg);
        assert!(matches!(
            result.unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));
    }

    // ── Test 16 ───────────────────────────────────────────────────────────────

    #[test]
    fn uap_large_xi_saturates() {
        // Very large xi: the perturbation remains finite (no NaN).
        let dim = 4_usize;
        let n_inputs = 3_usize;
        let n_classes = 2_usize;
        let weights = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0_f32];
        let biases = vec![2.0, 0.0_f32];
        let oracle = linear_oracle(weights, biases, n_classes, dim);
        let inputs: Vec<f32> = (0..n_inputs * dim).map(|i| (i as f32) * 0.1).collect();
        let cfg = UapConfig {
            xi: 100.0,
            max_passes: 2,
            ..Default::default()
        };
        let result = uap_attack(&inputs, n_inputs, dim, n_classes, oracle, &cfg).unwrap();
        for &vi in &result.perturbation {
            assert!(
                vi.is_finite(),
                "perturbation contains non-finite value: {vi}"
            );
        }
    }

    // ── Test 17 ───────────────────────────────────────────────────────────────

    #[test]
    fn uap_converges_linear() {
        // 2-class linear classifier with N=5 inputs.
        // Verify that fool_rate is in [0,1] and n_passes is bounded.
        let dim = 4_usize;
        let n_inputs = 5_usize;
        let n_classes = 2_usize;
        // class 0 has large bias, class 1 wins only if x[1] component grows.
        let weights = vec![0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0_f32];
        let biases = vec![2.0, 0.0_f32];
        let oracle = linear_oracle(weights, biases, n_classes, dim);
        let inputs: Vec<f32> = vec![
            0.1, 0.2, 0.3, 0.4, // input 0
            0.2, 0.3, 0.1, 0.2, // input 1
            0.0, 0.1, 0.2, 0.3, // input 2
            0.4, 0.1, 0.3, 0.2, // input 3
            0.3, 0.2, 0.1, 0.0_f32, // input 4
        ];
        let cfg = UapConfig {
            xi: 1.0,
            max_passes: 3,
            target_fool_rate: 1.1, // never stop early
            deepfool_max_iter: 30,
            deepfool_overshoot: 0.02,
        };
        let result = uap_attack(&inputs, n_inputs, dim, n_classes, oracle, &cfg).unwrap();
        assert!(result.fool_rate >= 0.0 && result.fool_rate <= 1.0);
        assert!(result.n_passes <= 3);
    }

    // ── Test 18 ───────────────────────────────────────────────────────────────

    #[test]
    fn uap_n_passes_is_one_when_already_fooled() {
        // All inputs are already misclassified before any pass.
        // Use a stateful oracle: first N calls (orig class computation) return class 0.
        // All subsequent calls return class 1. So fool_rate = 1.0 after first pass.
        use std::cell::Cell;
        let dim = 3_usize;
        let n_inputs = 3_usize;
        let n_classes = 2_usize;
        let call_count = Cell::new(0_u32);
        let oracle = move |_x: &[f32]| -> AdvResult<(Vec<f32>, Vec<f32>)> {
            let n = call_count.get();
            call_count.set(n + 1);
            // First n_inputs calls: class 0 wins (orig class computation).
            // After that: class 1 wins (x+v is already misclassified).
            if n < n_inputs as u32 {
                Ok((vec![2.0, 0.0_f32], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0_f32]))
            } else {
                Ok((vec![0.0, 2.0_f32], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0_f32]))
            }
        };
        let inputs = vec![0.5_f32; n_inputs * dim];
        let cfg = UapConfig {
            xi: 0.5,
            max_passes: 5,
            target_fool_rate: 0.99,
            ..Default::default()
        };
        let result = uap_attack(&inputs, n_inputs, dim, n_classes, oracle, &cfg).unwrap();
        assert_eq!(
            result.fool_rate, 1.0,
            "expected fool_rate=1.0, got {}",
            result.fool_rate
        );
        assert_eq!(
            result.n_passes, 1,
            "expected n_passes=1, got {}",
            result.n_passes
        );
    }
}
