//! LayerDrop regularization — stochastic layer dropping for deep network distillation.
//!
//! During training each transformer/residual layer is independently kept or dropped
//! with probability `(1 - drop_rate)`.  Dropped layers pass through the previous
//! layer's output unchanged (residual passthrough), ensuring the network always
//! produces a valid output.

use crate::error::{DistillError, DistillResult};
use crate::handle::LcgRng;

/// Configuration for [`LayerDrop`].
pub struct LayerDropConfig {
    /// Total number of layers in the network.
    pub n_layers: usize,
    /// Probability of **dropping** each layer.  Must be in `[0, 1)`.
    pub drop_rate: f32,
}

/// Stochastic layer-dropping regularizer.
pub struct LayerDrop {
    config: LayerDropConfig,
}

impl LayerDrop {
    /// Construct a [`LayerDrop`] from `config`.
    ///
    /// # Errors
    /// - [`DistillError::InvalidConfig`] if `n_layers == 0`.
    /// - [`DistillError::InvalidConfig`] if `drop_rate` is not in `[0, 1)`.
    pub fn new(config: LayerDropConfig) -> DistillResult<Self> {
        if config.n_layers == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "n_layers must be > 0".to_string(),
            });
        }
        if !(0.0..1.0).contains(&config.drop_rate) {
            return Err(DistillError::InvalidConfig {
                msg: format!("drop_rate must be in [0, 1), got {}", config.drop_rate),
            });
        }
        Ok(Self { config })
    }

    /// Sample which layers should remain active during this forward pass.
    ///
    /// Returns a `Vec<bool>` of length `n_layers`.  Layer `i` is active when
    /// `rng.next_f32() >= drop_rate`.  If every layer would be dropped the
    /// method forces layer 0 to be active so that at least one layer always
    /// contributes to the output.
    pub fn sample_active_layers(&self, rng: &mut LcgRng) -> Vec<bool> {
        let mut active: Vec<bool> = (0..self.config.n_layers)
            .map(|_| rng.next_f32() >= self.config.drop_rate)
            .collect();

        // Guarantee at least one active layer.
        if !active.iter().any(|&a| a) {
            active[0] = true;
        }

        active
    }

    /// Apply the LayerDrop forward pass.
    ///
    /// `layers` must contain exactly `n_layers` slices, each of identical length
    /// (`seq_len × d_model`).  Dropped layers propagate the previous layer's
    /// output unchanged; the initial "previous" state is an all-zero vector.
    ///
    /// # Errors
    /// - [`DistillError::DimensionMismatch`] if `layers.len() != n_layers`.
    /// - [`DistillError::DimensionMismatch`] if any layer slice has a different
    ///   length from the first.
    pub fn forward(&self, layers: &[Vec<f32>], rng: &mut LcgRng) -> DistillResult<Vec<f32>> {
        if layers.len() != self.config.n_layers {
            return Err(DistillError::DimensionMismatch {
                expected: self.config.n_layers,
                got: layers.len(),
            });
        }

        // All layers must have the same length.
        let layer_len = layers[0].len();
        for (idx, layer) in layers.iter().enumerate() {
            if layer.len() != layer_len {
                return Err(DistillError::DimensionMismatch {
                    expected: layer_len,
                    got: layer.len(),
                });
            }
            let _ = idx; // suppress unused variable warning
        }

        let active = self.sample_active_layers(rng);

        // Residual passthrough: start from zeros, update when layer is active.
        let mut current = vec![0.0_f32; layer_len];
        for (i, layer) in layers.iter().enumerate() {
            if active[i] {
                current.clone_from(layer);
            }
            // If inactive, `current` retains the previous active layer's output.
        }

        Ok(current)
    }

    /// Expected number of active layers under the configured drop rate.
    #[must_use]
    #[inline]
    pub fn expected_active(&self) -> f32 {
        self.config.n_layers as f32 * (1.0 - self.config.drop_rate)
    }

    /// Total number of layers.
    #[must_use]
    #[inline]
    pub fn n_layers(&self) -> usize {
        self.config.n_layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ld(n_layers: usize, drop_rate: f32) -> LayerDrop {
        LayerDrop::new(LayerDropConfig {
            n_layers,
            drop_rate,
        })
        .expect("valid config")
    }

    // ── Test 1 ──────────────────────────────────────────────────────────
    /// `sample_active_layers` must return a Vec of exactly `n_layers` booleans.
    #[test]
    fn sample_len() {
        let ld = make_ld(8, 0.3);
        let mut rng = LcgRng::new(1);
        let active = ld.sample_active_layers(&mut rng);
        assert_eq!(active.len(), 8);
    }

    // ── Test 2 ──────────────────────────────────────────────────────────
    /// With a very high drop rate at least one layer must always remain active.
    #[test]
    fn at_least_one_active() {
        let ld = make_ld(6, 0.99);
        let mut rng = LcgRng::new(42);
        for _ in 0..500 {
            let active = ld.sample_active_layers(&mut rng);
            assert!(
                active.iter().any(|&a| a),
                "at least one layer must be active"
            );
        }
    }

    // ── Test 3 ──────────────────────────────────────────────────────────
    /// With `drop_rate = 0.0` every layer must be active on every call.
    #[test]
    fn drop_rate_0_all_active() {
        let ld = make_ld(5, 0.0);
        let mut rng = LcgRng::new(7);
        for _ in 0..20 {
            let active = ld.sample_active_layers(&mut rng);
            assert!(
                active.iter().all(|&a| a),
                "all layers must be active when drop_rate=0"
            );
        }
    }

    // ── Test 4 ──────────────────────────────────────────────────────────
    /// With a very high drop rate (≈1) at least one layer is always kept.
    #[test]
    fn high_drop_rate_at_least_one() {
        let ld = make_ld(10, 0.9999);
        let mut rng = LcgRng::new(99);
        for _ in 0..200 {
            let active = ld.sample_active_layers(&mut rng);
            let count = active.iter().filter(|&&a| a).count();
            assert!(count >= 1, "at least one must be active, got {count}");
        }
    }

    // ── Test 5 ──────────────────────────────────────────────────────────
    /// The `forward` output must have the same length as each input layer slice.
    #[test]
    fn forward_shape() {
        let ld = make_ld(4, 0.3);
        let mut rng = LcgRng::new(2);
        let layer_len = 12_usize;
        let layers: Vec<Vec<f32>> = (0..4)
            .map(|i| (0..layer_len).map(|j| (i * layer_len + j) as f32).collect())
            .collect();
        let out = ld.forward(&layers, &mut rng).expect("forward ok");
        assert_eq!(out.len(), layer_len);
    }

    // ── Test 6 ──────────────────────────────────────────────────────────
    /// Every value in the `forward` output must be finite.
    #[test]
    fn forward_finite() {
        let ld = make_ld(6, 0.4);
        let mut rng = LcgRng::new(3);
        let layers: Vec<Vec<f32>> = (0..6)
            .map(|i| vec![i as f32 * 0.5, 1.0 - i as f32 * 0.1, 2.0])
            .collect();
        let out = ld.forward(&layers, &mut rng).expect("forward ok");
        for &v in &out {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    // ── Test 7 ──────────────────────────────────────────────────────────
    /// `expected_active()` must equal `n_layers * (1 - drop_rate)`.
    #[test]
    fn expected_active_correct() {
        let n = 8_usize;
        let dr = 0.25_f32;
        let ld = make_ld(n, dr);
        let expected = n as f32 * (1.0 - dr);
        assert!(
            (ld.expected_active() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            ld.expected_active()
        );
    }

    // ── Test 8 ──────────────────────────────────────────────────────────
    /// Different RNG seeds must (with very high probability) produce different
    /// layer-drop masks when the drop rate is non-trivial.
    #[test]
    fn different_seeds_different_drops() {
        let ld = make_ld(10, 0.5);
        let mut rng_a = LcgRng::new(1000);
        let mut rng_b = LcgRng::new(2000);
        let a = ld.sample_active_layers(&mut rng_a);
        let b = ld.sample_active_layers(&mut rng_b);
        // With 10 layers and drop_rate=0.5 the probability both masks are
        // identical is (0.5)^10 + (0.5)^10 ≈ 0.002.  We just check they differ
        // in at least one position across multiple samples.
        let mut found_diff = a != b;
        if !found_diff {
            for _ in 0..20 {
                let a2 = ld.sample_active_layers(&mut rng_a);
                let b2 = ld.sample_active_layers(&mut rng_b);
                if a2 != b2 {
                    found_diff = true;
                    break;
                }
            }
        }
        assert!(found_diff, "different seeds must produce different masks");
    }

    // ── Test 9 ──────────────────────────────────────────────────────────
    /// Constructing a [`LayerDrop`] with `n_layers = 0` must return an error.
    #[test]
    fn n_layers_0_error() {
        let result = LayerDrop::new(LayerDropConfig {
            n_layers: 0,
            drop_rate: 0.3,
        });
        assert!(
            result.is_err(),
            "n_layers=0 must produce an InvalidConfig error"
        );
    }

    // ── Test 10 ─────────────────────────────────────────────────────────
    /// A `drop_rate >= 1.0` must be rejected.
    #[test]
    fn drop_rate_ge_1_error() {
        let result = LayerDrop::new(LayerDropConfig {
            n_layers: 4,
            drop_rate: 1.0,
        });
        assert!(result.is_err(), "drop_rate=1.0 must be rejected");
    }

    // ── Test 11 ─────────────────────────────────────────────────────────
    /// A wrong number of layer slices must produce [`DistillError::DimensionMismatch`].
    #[test]
    fn forward_wrong_n_layers_error() {
        let ld = make_ld(4, 0.3);
        let mut rng = LcgRng::new(5);
        let layers: Vec<Vec<f32>> = (0..3).map(|_| vec![1.0; 8]).collect();
        assert!(ld.forward(&layers, &mut rng).is_err());
    }
}
