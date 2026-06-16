//! ProxylessNAS binary architecture gates (Cai et al., 2019).
//!
//! Each position in the search space maintains a set of architecture weights.
//! During the search, a binary path is sampled: exactly one operation is
//! activated per forward pass according to a Bernoulli distribution whose
//! parameters are the softmax-normalised architecture weights (divided by
//! temperature).  The straight-through estimator (STE) is used to back-
//! propagate gradients through the discrete sampling step.

use crate::error::{NasError, NasResult};
use crate::handle::LcgRng;

/// Type alias so call-sites that only care about the RNG type do not need to
/// import `handle` directly.
pub type NasRng = LcgRng;

// ─── BinaryGateConfig ────────────────────────────────────────────────────────

/// Hyper-parameters controlling a [`BinaryGate`].
#[derive(Debug, Clone)]
pub struct BinaryGateConfig {
    /// Number of candidate operations at this position.
    pub n_ops: usize,
    /// Softmax temperature (> 0).  High temperature → uniform distribution;
    /// low temperature → peaked distribution.
    pub temperature: f32,
}

// ─── BinaryGate ──────────────────────────────────────────────────────────────

/// A single binary architecture gate in ProxylessNAS.
///
/// Maintains one architecture weight per candidate operation.  The gate
/// is parameterised by `arch_weights` (logits).  During the forward pass
/// one operation is sampled proportional to [`BinaryGate::arch_probs`] and
/// the rest are masked out, forming a binary activation pattern.
///
/// Gradient updates use the straight-through estimator (STE):
/// `arch_weights[i] -= lr * grad[i]`.
#[derive(Debug, Clone)]
pub struct BinaryGate {
    /// Raw logits, one per candidate op.  Initialised to zero.
    arch_weights: Vec<f32>,
    /// Configuration snapshot.
    config: BinaryGateConfig,
}

impl BinaryGate {
    /// Construct a new [`BinaryGate`] with zero-initialised weights.
    ///
    /// # Errors
    /// Returns [`NasError::InvalidNumOps`] when `config.n_ops == 0`.
    pub fn new(config: BinaryGateConfig) -> NasResult<Self> {
        if config.n_ops == 0 {
            return Err(NasError::InvalidNumOps);
        }
        let arch_weights = vec![0.0_f32; config.n_ops];
        Ok(Self {
            arch_weights,
            config,
        })
    }

    /// Number of candidate operations.
    #[must_use]
    #[inline]
    pub fn n_ops(&self) -> usize {
        self.config.n_ops
    }

    /// Compute the softmax probability of each candidate operation.
    ///
    /// Uses the temperature-scaled, numerically stable softmax:
    /// `p_i = exp((w_i - max_w) / T) / Σ_j exp((w_j - max_w) / T)`.
    #[must_use]
    pub fn arch_probs(&self) -> Vec<f32> {
        temperature_softmax(&self.arch_weights, self.config.temperature)
    }

    /// Sample a binary activation mask proportional to [`arch_probs`].
    ///
    /// Exactly one element is set to `true` (the sampled operation); all
    /// others are `false`.  The operation is drawn by comparing a uniform
    /// random variate to the cumulative distribution of `arch_probs`.
    ///
    /// [`arch_probs`]: Self::arch_probs
    pub fn sample_path(&self, rng: &mut NasRng) -> Vec<bool> {
        let probs = self.arch_probs();
        let n = probs.len();
        let u: f32 = rng.next_f32();
        let mut cumulative = 0.0_f32;
        let mut selected = n - 1; // fall-back: last op if rounding keeps u > sum
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if u < cumulative {
                selected = i;
                break;
            }
        }
        let mut mask = vec![false; n];
        mask[selected] = true;
        mask
    }

    /// Update architecture weights via the straight-through estimator (STE).
    ///
    /// Performs `arch_weights[i] -= lr * grad[i]` for every `i`.
    ///
    /// # Errors
    /// Returns [`NasError::DimensionMismatch`] if `grad.len() != n_ops`.
    pub fn update_arch_weights(&mut self, grad: &[f32], lr: f32) -> NasResult<()> {
        let n = self.config.n_ops;
        if grad.len() != n {
            return Err(NasError::DimensionMismatch {
                expected: n,
                got: grad.len(),
            });
        }
        for (w, &g) in self.arch_weights.iter_mut().zip(grad.iter()) {
            *w -= lr * g;
        }
        Ok(())
    }

    /// Return the index of the operation with the highest architecture weight
    /// (argmax of `arch_weights`).
    ///
    /// When multiple weights are tied at the maximum the lowest index wins.
    #[must_use]
    pub fn selected_op(&self) -> usize {
        self.arch_weights
            .iter()
            .enumerate()
            .fold(
                0usize,
                |best, (i, &w)| {
                    if w > self.arch_weights[best] { i } else { best }
                },
            )
    }
}

// ─── temperature_softmax helper ──────────────────────────────────────────────

/// Numerically stable softmax with temperature scaling.
///
/// Computes `p_i = exp((w_i - max_w) / temp) / Σ_j exp((w_j - max_w) / temp)`.
/// Falls back to a uniform distribution if the sum of exponents is zero.
fn temperature_softmax(weights: &[f32], temp: f32) -> Vec<f32> {
    if weights.is_empty() {
        return Vec::new();
    }
    let safe_temp = if temp <= 0.0 { 1e-6_f32 } else { temp };
    let max_w = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = weights
        .iter()
        .map(|&w| ((w - max_w) / safe_temp).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        vec![1.0 / weights.len() as f32; weights.len()]
    } else {
        exps.into_iter().map(|e| e / sum).collect()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gate(n_ops: usize, temp: f32) -> BinaryGate {
        BinaryGate::new(BinaryGateConfig {
            n_ops,
            temperature: temp,
        })
        .expect("valid config")
    }

    // 1. sample_path() returns a vec of length n_ops.
    #[test]
    fn sample_path_len() {
        let gate = make_gate(5, 1.0);
        let mut rng = NasRng::new(42);
        let mask = gate.sample_path(&mut rng);
        assert_eq!(mask.len(), 5);
    }

    // 2. arch_probs() sums to ≈ 1.0 (within 1e-5).
    #[test]
    fn arch_probs_sum_to_1() {
        let gate = make_gate(8, 1.0);
        let probs = gate.arch_probs();
        let s: f32 = probs.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "sum = {s}");
    }

    // 3. All arch_probs() >= 0.
    #[test]
    fn arch_probs_nonneg() {
        let gate = make_gate(6, 0.5);
        for &p in gate.arch_probs().iter() {
            assert!(p >= 0.0, "negative prob: {p}");
        }
    }

    // 4. selected_op() < n_ops.
    #[test]
    fn selected_op_in_range() {
        let gate = make_gate(7, 1.0);
        assert!(gate.selected_op() < 7);
    }

    // 5. After update_arch_weights at least one weight is different.
    #[test]
    fn update_changes_weights() {
        let mut gate = make_gate(4, 1.0);
        let before = gate.arch_weights.clone();
        let grad = vec![0.1_f32, 0.2, 0.3, 0.4];
        gate.update_arch_weights(&grad, 0.01).expect("update ok");
        assert_ne!(gate.arch_weights, before);
    }

    // 6. High temperature → more uniform probs (max prob is lower at high temp).
    #[test]
    fn temperature_affects_distribution() {
        // Create gates with different initial weights to make softmax non-trivial.
        let mut gate_low = make_gate(4, 0.1);
        let mut gate_high = make_gate(4, 100.0);
        // Give them the same non-uniform weights.
        let weights = [1.0_f32, 2.0, 3.0, 4.0];
        let grad_low: Vec<f32> = weights.iter().map(|&w| -w).collect(); // -= lr*grad ⇒ weight += w
        let grad_high = grad_low.clone();
        gate_low
            .update_arch_weights(&grad_low, 1.0)
            .expect("update low ok");
        gate_high
            .update_arch_weights(&grad_high, 1.0)
            .expect("update high ok");

        let probs_low = gate_low.arch_probs();
        let probs_high = gate_high.arch_probs();

        let max_low = probs_low.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let max_high = probs_high.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // High temperature should produce a more uniform distribution,
        // so its maximum probability must be smaller than the low-temperature max.
        assert!(
            max_high < max_low,
            "high-temp max={max_high} should be < low-temp max={max_low}"
        );
    }

    // 7. n_ops=1: selected_op() must always be 0.
    #[test]
    fn single_op_always_selected() {
        let gate = make_gate(1, 1.0);
        assert_eq!(gate.selected_op(), 0);
    }

    // 8. n_ops=0 returns Err(InvalidNumOps).
    #[test]
    fn n_ops_zero_error() {
        let result = BinaryGate::new(BinaryGateConfig {
            n_ops: 0,
            temperature: 1.0,
        });
        assert_eq!(result.unwrap_err(), NasError::InvalidNumOps);
    }

    // 9. Different RNG seeds produce different masks across multiple draws.
    #[test]
    fn sample_diverse_seeds() {
        let gate = make_gate(4, 1.0);
        // Collect the index of the selected op for several seeds.
        let chosen: Vec<usize> = (0_u64..20)
            .map(|seed| {
                let mut rng = NasRng::new(seed);
                let mask = gate.sample_path(&mut rng);
                mask.iter().position(|&b| b).expect("exactly one true")
            })
            .collect();
        // At least two distinct selections must appear.
        let first = chosen[0];
        let all_same = chosen.iter().all(|&c| c == first);
        assert!(!all_same, "expected diversity across seeds; got {chosen:?}");
    }
}
