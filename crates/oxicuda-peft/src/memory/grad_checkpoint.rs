//! Gradient checkpointing (activation checkpointing) for memory-efficient training.
//!
//! Gradient checkpointing trades compute for memory by discarding intermediate activations
//! during the forward pass, recomputing them on-demand during backpropagation.
//! This implementation models the scheduling and memory analysis of a checkpointing strategy.

use crate::error::{PeftError, PeftResult};

/// Configuration for a gradient checkpointing schedule.
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Total number of layers in the model.
    pub n_layers: usize,
    /// Save an activation checkpoint every this many layers.
    /// Lower values save more checkpoints (less recomputation, more memory).
    pub checkpoint_every_n: usize,
}

/// A computed schedule of which layers are checkpointed and which are recomputed.
///
/// Layer indices are zero-based (`0..n_layers`).
/// Layer 0 is always checkpointed; the last layer (`n_layers - 1`) is always checkpointed.
/// All layers at multiples of `checkpoint_every_n` are checkpointed.
#[derive(Debug, Clone)]
pub struct CheckpointSchedule {
    /// Layers whose activations are retained (sorted ascending).
    pub checkpointed_layers: Vec<usize>,
    /// Layers whose activations are discarded and must be recomputed (sorted ascending).
    pub recomputed_layers: Vec<usize>,
    config: CheckpointConfig,
}

impl CheckpointSchedule {
    /// Build the checkpoint schedule from a `CheckpointConfig`.
    ///
    /// # Errors
    /// - `PeftError::Internal` if `n_layers == 0`.
    /// - `PeftError::Internal` if `checkpoint_every_n == 0`.
    pub fn new(config: CheckpointConfig) -> PeftResult<Self> {
        if config.n_layers == 0 {
            return Err(PeftError::Internal {
                msg: "n_layers must be > 0".into(),
            });
        }
        if config.checkpoint_every_n == 0 {
            return Err(PeftError::Internal {
                msg: "checkpoint_every_n must be > 0".into(),
            });
        }

        let n = config.n_layers;
        let every = config.checkpoint_every_n;

        // Build the set of checkpointed layers:
        // - All i in 0..n where i % every == 0 (includes layer 0)
        // - Always include the last layer n-1
        let mut checkpointed = Vec::new();
        for i in 0..n {
            if i % every == 0 || i == n - 1 {
                checkpointed.push(i);
            }
        }
        // Deduplicate while preserving sort order (n-1 might already be included)
        checkpointed.dedup();

        // Recomputed = complement
        let checkpoint_set: std::collections::BTreeSet<usize> =
            checkpointed.iter().copied().collect();
        let recomputed: Vec<usize> = (0..n).filter(|i| !checkpoint_set.contains(i)).collect();

        Ok(Self {
            checkpointed_layers: checkpointed,
            recomputed_layers: recomputed,
            config,
        })
    }

    /// Fraction of layer activations that are discarded (memory saved).
    ///
    /// Returns `recomputed_layers.len() / n_layers`. With `checkpoint_every_n == 1`,
    /// all layers are checkpointed and this returns `0.0`. With large `checkpoint_every_n`,
    /// approaches `1.0`.
    #[must_use]
    pub fn memory_reduction_factor(&self) -> f32 {
        self.recomputed_layers.len() as f32 / self.config.n_layers as f32
    }

    /// Maximum number of consecutive recomputed layers in any segment between checkpoints.
    ///
    /// A "segment" is the run of recomputed layers between two consecutive checkpoints.
    /// Returns `0` if all layers are checkpointed.
    #[must_use]
    pub fn max_recompute_segment(&self) -> usize {
        if self.checkpointed_layers.len() <= 1 {
            // All layers between 0 and n_layers-1 are in a single segment
            return self.recomputed_layers.len();
        }
        let mut max_gap = 0usize;
        for window in self.checkpointed_layers.windows(2) {
            // Number of recomputed layers strictly between checkpoint[k] and checkpoint[k+1]
            let gap = window[1].saturating_sub(window[0]).saturating_sub(1);
            if gap > max_gap {
                max_gap = gap;
            }
        }
        max_gap
    }

    /// Number of checkpointed layers.
    #[must_use]
    pub fn n_checkpoints(&self) -> usize {
        self.checkpointed_layers.len()
    }

    /// Simulate the forward pass with checkpointing.
    ///
    /// Applies `layer_fn(activation, layer_idx)` sequentially for each layer in `0..n_layers`.
    /// Only the activations at checkpointed layers are retained; all others are discarded.
    ///
    /// `input` is the initial activation (fed into layer 0).
    /// Returns a `Vec` of length `n_checkpoints()`, one activation per checkpointed layer.
    ///
    /// # Errors
    /// - `PeftError::EmptyInput` if `input` is empty.
    pub fn forward_checkpoint(
        &self,
        input: &[f32],
        layer_fn: impl Fn(&[f32], usize) -> Vec<f32>,
    ) -> PeftResult<Vec<Vec<f32>>> {
        if input.is_empty() {
            return Err(PeftError::EmptyInput);
        }

        let n = self.config.n_layers;
        let checkpoint_set: std::collections::BTreeSet<usize> =
            self.checkpointed_layers.iter().copied().collect();

        let mut checkpointed_activations = Vec::with_capacity(self.checkpointed_layers.len());
        let mut current: Vec<f32> = input.to_vec();

        for layer_idx in 0..n {
            let next = layer_fn(&current, layer_idx);
            if checkpoint_set.contains(&layer_idx) {
                checkpointed_activations.push(next.clone());
            }
            current = next;
        }

        Ok(checkpointed_activations)
    }

    /// Find the checkpoint layer to start recomputing from to reach `target_layer`.
    ///
    /// Returns the largest checkpointed layer index `<= target_layer`.
    /// This is the layer whose retained activation can be used as a starting point
    /// to recompute forward to `target_layer`.
    ///
    /// # Errors
    /// - `PeftError::LayerOutOfRange` if `target_layer >= n_layers`.
    pub fn recompute_from(&self, target_layer: usize) -> PeftResult<usize> {
        if target_layer >= self.config.n_layers {
            return Err(PeftError::LayerOutOfRange {
                idx: target_layer,
                num_layers: self.config.n_layers,
            });
        }
        // Find the largest checkpoint <= target_layer
        // checkpointed_layers is sorted ascending, so scan from the right
        self.checkpointed_layers
            .iter()
            .rev()
            .find(|&&ck| ck <= target_layer)
            .copied()
            .ok_or_else(|| PeftError::Internal {
                msg: format!("no checkpoint found for target_layer {target_layer}"),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple identity layer function for testing.
    fn identity_layer(x: &[f32], _layer: usize) -> Vec<f32> {
        x.to_vec()
    }

    /// Layer function that scales by a small factor.
    fn scale_layer(x: &[f32], _layer: usize) -> Vec<f32> {
        x.iter().map(|&v| v * 1.001_f32).collect()
    }

    #[test]
    fn n_checkpoints_correct() {
        // n_layers=10, every_n=3: multiples of 3 are {0,3,6,9}, plus last layer=9 (already in set)
        let cfg = CheckpointConfig {
            n_layers: 10,
            checkpoint_every_n: 3,
        };
        let sched = CheckpointSchedule::new(cfg).expect("valid config");
        // Checkpointed: 0, 3, 6, 9
        assert_eq!(
            sched.n_checkpoints(),
            4,
            "expected 4 checkpoints for n=10, every_n=3, got {:?}",
            sched.checkpointed_layers
        );
    }

    #[test]
    fn memory_reduction_gt_0() {
        let cfg = CheckpointConfig {
            n_layers: 12,
            checkpoint_every_n: 4,
        };
        let sched = CheckpointSchedule::new(cfg).expect("valid config");
        let reduction = sched.memory_reduction_factor();
        assert!(
            reduction > 0.0,
            "memory reduction should be > 0 with every_n=4, got {reduction}"
        );
        assert!(
            reduction < 1.0,
            "memory reduction should be < 1.0 (some layers checkpointed), got {reduction}"
        );
    }

    #[test]
    fn max_recompute_bounded() {
        let every_n = 4usize;
        let cfg = CheckpointConfig {
            n_layers: 20,
            checkpoint_every_n: every_n,
        };
        let sched = CheckpointSchedule::new(cfg).expect("valid config");
        let max_seg = sched.max_recompute_segment();
        assert!(
            max_seg < every_n,
            "max_recompute_segment={max_seg} should be <= checkpoint_every_n - 1 = {}",
            every_n - 1
        );
    }

    #[test]
    fn forward_checkpoint_len() {
        let cfg = CheckpointConfig {
            n_layers: 8,
            checkpoint_every_n: 2,
        };
        let sched = CheckpointSchedule::new(cfg).expect("valid config");
        let input = vec![1.0_f32; 4];
        let result = sched
            .forward_checkpoint(&input, identity_layer)
            .expect("forward ok");
        assert_eq!(
            result.len(),
            sched.n_checkpoints(),
            "forward_checkpoint output length should match n_checkpoints"
        );
    }

    #[test]
    fn checkpoint_layer_0_always() {
        for n in [1usize, 2, 5, 10, 20] {
            let cfg = CheckpointConfig {
                n_layers: n,
                checkpoint_every_n: 3,
            };
            let sched = CheckpointSchedule::new(cfg).expect("valid config");
            assert!(
                sched.checkpointed_layers.contains(&0),
                "layer 0 must always be checkpointed (n={n})"
            );
        }
    }

    #[test]
    fn forward_output_finite() {
        let cfg = CheckpointConfig {
            n_layers: 6,
            checkpoint_every_n: 2,
        };
        let sched = CheckpointSchedule::new(cfg).expect("valid config");
        let input: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let result = sched
            .forward_checkpoint(&input, scale_layer)
            .expect("forward ok");
        for (ck_idx, act) in result.iter().enumerate() {
            for &v in act {
                assert!(
                    v.is_finite(),
                    "checkpoint activation at checkpoint {ck_idx} is not finite: {v}"
                );
            }
        }
    }

    #[test]
    fn n_layers_0_error() {
        let cfg = CheckpointConfig {
            n_layers: 0,
            checkpoint_every_n: 1,
        };
        let result = CheckpointSchedule::new(cfg);
        assert!(result.is_err(), "n_layers=0 should return an error");
    }

    #[test]
    fn checkpoint_every_n_gt_layers_ok() {
        // checkpoint_every_n > n_layers: only layer 0 and n-1 get checkpointed
        let cfg = CheckpointConfig {
            n_layers: 5,
            checkpoint_every_n: 100,
        };
        let sched = CheckpointSchedule::new(cfg).expect("large every_n should be ok");
        assert!(
            sched.checkpointed_layers.contains(&0),
            "layer 0 must be checkpointed"
        );
        assert!(
            sched.checkpointed_layers.contains(&4),
            "last layer must be checkpointed"
        );
    }

    #[test]
    fn recompute_from_valid() {
        let cfg = CheckpointConfig {
            n_layers: 12,
            checkpoint_every_n: 3,
        };
        let sched = CheckpointSchedule::new(cfg).expect("valid config");
        let ck = sched.recompute_from(5).expect("target layer 5 is valid");
        assert!(
            ck <= 5,
            "recompute_from(5) should return a checkpoint <= 5, got {ck}"
        );
        assert!(
            sched.checkpointed_layers.contains(&ck),
            "recompute_from(5)={ck} must be a checkpointed layer"
        );
    }

    #[test]
    fn last_layer_covered() {
        let cfg = CheckpointConfig {
            n_layers: 10,
            checkpoint_every_n: 4,
        };
        let sched = CheckpointSchedule::new(cfg).expect("valid config");
        let last = sched.config.n_layers - 1;
        // The last layer is always checkpointed
        assert!(
            sched.checkpointed_layers.contains(&last),
            "last layer {last} must be in checkpointed_layers: {:?}",
            sched.checkpointed_layers
        );
    }

    #[test]
    fn recompute_from_out_of_range_error() {
        let cfg = CheckpointConfig {
            n_layers: 10,
            checkpoint_every_n: 3,
        };
        let sched = CheckpointSchedule::new(cfg).expect("valid config");
        let result = sched.recompute_from(10); // exactly n_layers, out of range
        assert!(
            result.is_err(),
            "recompute_from(n_layers) should return an error"
        );
    }

    #[test]
    fn every_n_1_all_checkpointed() {
        // With checkpoint_every_n=1, every layer is checkpointed and none recomputed
        let cfg = CheckpointConfig {
            n_layers: 8,
            checkpoint_every_n: 1,
        };
        let sched = CheckpointSchedule::new(cfg).expect("valid config");
        assert_eq!(
            sched.checkpointed_layers.len(),
            8,
            "every_n=1: all layers should be checkpointed"
        );
        assert!(
            sched.recomputed_layers.is_empty(),
            "every_n=1: no layers should be recomputed"
        );
        assert!(
            (sched.memory_reduction_factor() - 0.0).abs() < 1e-7,
            "every_n=1: memory reduction should be 0.0"
        );
    }
}
