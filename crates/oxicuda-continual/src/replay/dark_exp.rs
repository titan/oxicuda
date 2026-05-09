//! Dark Experience Replay (DER and DER++).
//!
//! Implements the method from:
//! Buzzega et al. "Dark Experience for General Continual Learning: a Strong,
//! Simple Baseline." NeurIPS 2020.
//!
//! DER++ stores not just the class labels but the full logits at training time,
//! allowing distillation of the model's predictions on replayed samples.
//!
//! Loss: `α·MSE(z_current, z_stored) + β·CE(z_current, y) + CE(z_new, y_new)`

use crate::error::{ContinualError, ContinualResult};
use crate::handle::LcgRng;

/// Configuration for DER++.
#[derive(Debug, Clone)]
pub struct DerConfig {
    /// Weight for logit distillation MSE term (α).
    pub alpha: f32,
    /// Weight for cross-entropy on replayed samples (β).
    pub beta: f32,
}

impl Default for DerConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            beta: 0.5,
        }
    }
}

impl DerConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> ContinualResult<()> {
        if !self.alpha.is_finite() || self.alpha < 0.0 {
            return Err(ContinualError::InvalidLambda { lambda: self.alpha });
        }
        if !self.beta.is_finite() || self.beta < 0.0 {
            return Err(ContinualError::InvalidLambda { lambda: self.beta });
        }
        Ok(())
    }
}

/// DER++ replay buffer storing features, stored logits, and labels.
#[derive(Debug, Clone)]
pub struct DerBuffer {
    /// Stored feature vectors.
    pub data: Vec<Vec<f32>>,
    /// Stored logits at the time of first training (used for distillation).
    pub logits: Vec<Vec<f32>>,
    /// Stored class labels.
    pub labels: Vec<u32>,
    /// Maximum buffer capacity.
    pub capacity: usize,
    /// Number of samples seen so far.
    pub n_seen: usize,
}

impl DerBuffer {
    /// Create a new empty DER buffer.
    pub fn new(capacity: usize) -> ContinualResult<Self> {
        if capacity == 0 {
            return Err(ContinualError::BufferCapacityTooSmall);
        }
        Ok(Self {
            data: Vec::with_capacity(capacity),
            logits: Vec::with_capacity(capacity),
            labels: Vec::with_capacity(capacity),
            capacity,
            n_seen: 0,
        })
    }

    /// Current number of samples in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True if buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Compute softmax of a logit vector in-place.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum < 1e-30 {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// Compute Mean Squared Error between two vectors.
fn mse(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f32>()
        / n
}

/// Compute cross-entropy loss for `current_logits` against hard label `label`.
///
/// `CE = -log(softmax(z)[label])`
fn cross_entropy(logits: &[f32], label: u32) -> ContinualResult<f32> {
    let n_classes = logits.len();
    let label_idx = label as usize;
    if label_idx >= n_classes {
        return Err(ContinualError::TaskIndexOutOfRange {
            index: label_idx,
            n_tasks: n_classes,
        });
    }
    let probs = softmax(logits);
    let p = probs[label_idx].max(1e-30);
    Ok(-p.ln())
}

/// Compute the DER++ loss for a single replayed sample.
///
/// `loss = α · MSE(z_current, z_stored) + β · CE(z_current, y)`
///
/// where `z_stored` are the logits stored at training time.
/// The caller accumulates the current-data CE separately.
pub fn der_loss(
    current_logits: &[f32],
    stored_logits: &[f32],
    label: u32,
    n_classes: usize,
    cfg: &DerConfig,
) -> ContinualResult<f32> {
    cfg.validate()?;
    if current_logits.len() != n_classes {
        return Err(ContinualError::DimensionMismatch {
            expected: n_classes,
            got: current_logits.len(),
        });
    }
    if stored_logits.len() != n_classes {
        return Err(ContinualError::DimensionMismatch {
            expected: n_classes,
            got: stored_logits.len(),
        });
    }
    let mse_term = mse(current_logits, stored_logits);
    let ce_term = cross_entropy(current_logits, label)?;
    let loss = cfg.alpha * mse_term + cfg.beta * ce_term;
    if !loss.is_finite() {
        return Err(ContinualError::NanEncountered {
            location: "der_loss",
        });
    }
    Ok(loss)
}

/// Add a sample (with its current logits) to the DER buffer using reservoir sampling.
pub fn der_add(
    buf: &mut DerBuffer,
    sample: Vec<f32>,
    logit: Vec<f32>,
    label: u32,
    rng: &mut LcgRng,
) {
    let n = buf.n_seen;
    if n < buf.capacity {
        buf.data.push(sample);
        buf.logits.push(logit);
        buf.labels.push(label);
    } else {
        let r = rng.next_usize(n + 1);
        if r < buf.capacity {
            buf.data[r] = sample;
            buf.logits[r] = logit;
            buf.labels[r] = label;
        }
    }
    buf.n_seen += 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn der_loss_finite() {
        let current = vec![1.0_f32, 0.5, -0.5];
        let stored = vec![0.8_f32, 0.6, -0.3];
        let cfg = DerConfig::default();
        let loss = der_loss(&current, &stored, 0, 3, &cfg).unwrap();
        assert!(loss.is_finite(), "DER loss should be finite, got {loss}");
        assert!(loss >= 0.0, "DER loss should be non-negative");
    }

    #[test]
    fn der_loss_decreases_when_logits_approach_stored() {
        let stored = vec![2.0_f32, -1.0, 0.5];
        let close = vec![1.9_f32, -0.9, 0.6]; // close to stored
        let far = vec![0.0_f32, 1.0, -1.0]; // far from stored
        let cfg = DerConfig {
            alpha: 1.0,
            beta: 0.0, // only MSE term
        };
        let loss_close = der_loss(&close, &stored, 0, 3, &cfg).unwrap();
        let loss_far = der_loss(&far, &stored, 0, 3, &cfg).unwrap();
        assert!(
            loss_close < loss_far,
            "MSE loss should decrease when logits approach stored (close={loss_close}, far={loss_far})"
        );
    }

    #[test]
    fn der_loss_zero_mse_when_identical() {
        let logits = vec![1.0_f32, 0.5, -0.5];
        let cfg = DerConfig {
            alpha: 1.0,
            beta: 0.0,
        };
        let loss = der_loss(&logits, &logits, 0, 3, &cfg).unwrap();
        // MSE = 0, so loss = alpha * 0 + beta * CE = 0
        assert!(loss.abs() < 1e-6, "MSE should be 0 for identical logits");
    }

    #[test]
    fn der_add_reservoir_bounded() {
        let mut rng = LcgRng::new(42);
        let mut buf = DerBuffer::new(10).unwrap();
        for i in 0..50_usize {
            der_add(
                &mut buf,
                vec![i as f32],
                vec![0.0, 1.0],
                (i % 2) as u32,
                &mut rng,
            );
        }
        assert_eq!(buf.len(), 10);
        assert_eq!(buf.n_seen, 50);
    }

    #[test]
    fn der_loss_dimension_mismatch_returns_err() {
        let current = vec![1.0_f32; 3];
        let stored = vec![1.0_f32; 4]; // wrong
        let cfg = DerConfig::default();
        assert!(der_loss(&current, &stored, 0, 3, &cfg).is_err());
    }

    #[test]
    fn der_loss_label_out_of_range_returns_err() {
        let current = vec![1.0_f32; 3];
        let stored = vec![1.0_f32; 3];
        let cfg = DerConfig::default();
        // label=5 but n_classes=3
        assert!(der_loss(&current, &stored, 5, 3, &cfg).is_err());
    }

    #[test]
    fn der_buffer_capacity_zero_returns_err() {
        assert!(DerBuffer::new(0).is_err());
    }

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![1.0_f32, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let s: f32 = probs.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "Softmax must sum to 1, got {s}");
    }

    #[test]
    fn der_config_invalid_alpha() {
        let cfg = DerConfig {
            alpha: -1.0,
            beta: 0.5,
        };
        assert!(cfg.validate().is_err());
    }
}
