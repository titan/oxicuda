use crate::error::{PeftError, PeftResult};
use crate::handle::LcgRng;

/// Configuration for a full IA³ transformer layer.
#[derive(Debug, Clone)]
pub struct Ia3Config {
    /// Hidden/model dimension: must be > 0.
    pub d_model: usize,
    /// Number of transformer layers this config describes (informational only).
    pub n_layers: usize,
}

/// IA³ scaling vectors for a single transformer layer.
///
/// Holds three element-wise scale vectors — one for keys, one for values, and one for
/// the feed-forward sub-layer — all initialised to `1.0` so the adapter begins as an
/// identity. Trainable parameter count: `3 × d_model`.
#[derive(Debug, Clone)]
pub struct Ia3Layer {
    /// Key-projection rescaling vector, shape `[d_model]`.
    pub l_k: Vec<f32>,
    /// Value-projection rescaling vector, shape `[d_model]`.
    pub l_v: Vec<f32>,
    /// Feed-forward rescaling vector, shape `[d_model]`.
    pub l_ff: Vec<f32>,
    /// Hidden dimension shared by all three vectors.
    pub d_model: usize,
}

impl Ia3Layer {
    /// Create a new `Ia3Layer` with all scale vectors initialised to `1.0`.
    ///
    /// # Errors
    ///
    /// Returns `PeftError::DimensionMismatch { expected: 1, got: 0 }` when
    /// `config.d_model == 0`.
    pub fn new(config: &Ia3Config, _rng: &mut LcgRng) -> PeftResult<Self> {
        if config.d_model == 0 {
            return Err(PeftError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        Ok(Self {
            l_k: vec![1.0_f32; config.d_model],
            l_v: vec![1.0_f32; config.d_model],
            l_ff: vec![1.0_f32; config.d_model],
            d_model: config.d_model,
        })
    }

    /// Element-wise multiply the key activations `k` by `l_k`.
    ///
    /// # Errors
    ///
    /// Returns `PeftError::DimensionMismatch` when `k.len() != d_model`.
    pub fn rescale_keys(&self, k: &[f32]) -> PeftResult<Vec<f32>> {
        if k.len() != self.d_model {
            return Err(PeftError::DimensionMismatch {
                expected: self.d_model,
                got: k.len(),
            });
        }
        Ok(k.iter()
            .zip(self.l_k.iter())
            .map(|(ki, li)| ki * li)
            .collect())
    }

    /// Element-wise multiply the value activations `v` by `l_v`.
    ///
    /// # Errors
    ///
    /// Returns `PeftError::DimensionMismatch` when `v.len() != d_model`.
    pub fn rescale_values(&self, v: &[f32]) -> PeftResult<Vec<f32>> {
        if v.len() != self.d_model {
            return Err(PeftError::DimensionMismatch {
                expected: self.d_model,
                got: v.len(),
            });
        }
        Ok(v.iter()
            .zip(self.l_v.iter())
            .map(|(vi, li)| vi * li)
            .collect())
    }

    /// Element-wise multiply the feed-forward activations `f` by `l_ff`.
    ///
    /// # Errors
    ///
    /// Returns `PeftError::DimensionMismatch` when `f.len() != d_model`.
    pub fn rescale_ffn(&self, f: &[f32]) -> PeftResult<Vec<f32>> {
        if f.len() != self.d_model {
            return Err(PeftError::DimensionMismatch {
                expected: self.d_model,
                got: f.len(),
            });
        }
        Ok(f.iter()
            .zip(self.l_ff.iter())
            .map(|(fi, li)| fi * li)
            .collect())
    }

    /// Total number of trainable parameters: `3 × d_model`.
    #[must_use]
    #[inline]
    pub fn n_params(&self) -> usize {
        3 * self.d_model
    }

    /// Reset all three scale vectors back to `1.0`.
    pub fn init_to_ones(&mut self) {
        for v in self.l_k.iter_mut() {
            *v = 1.0;
        }
        for v in self.l_v.iter_mut() {
            *v = 1.0;
        }
        for v in self.l_ff.iter_mut() {
            *v = 1.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_layer(d_model: usize) -> Ia3Layer {
        let cfg = Ia3Config {
            d_model,
            n_layers: 2,
        };
        let mut rng = LcgRng::new(42);
        Ia3Layer::new(&cfg, &mut rng).expect("valid config")
    }

    // 1. rescale_shape: output length equals input length
    #[test]
    fn rescale_shape() {
        let layer = make_layer(16);
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let out_k = layer.rescale_keys(&input).expect("rescale_keys");
        let out_v = layer.rescale_values(&input).expect("rescale_values");
        let out_ff = layer.rescale_ffn(&input).expect("rescale_ffn");
        assert_eq!(out_k.len(), 16);
        assert_eq!(out_v.len(), 16);
        assert_eq!(out_ff.len(), 16);
    }

    // 2. rescale_finite: all output values are finite
    #[test]
    fn rescale_finite() {
        let mut layer = make_layer(32);
        // Set some non-trivial scale values.
        for (i, v) in layer.l_k.iter_mut().enumerate() {
            *v = 0.5 + (i as f32) * 0.1;
        }
        let input: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 3.5).collect();
        let out = layer.rescale_keys(&input).expect("rescale");
        for &x in &out {
            assert!(x.is_finite(), "non-finite value in output: {x}");
        }
    }

    // 3. init_to_ones_effect: after modifying l_k and calling init_to_ones, l_k == 1.0 again
    #[test]
    fn init_to_ones_effect() {
        let mut layer = make_layer(8);
        for v in layer.l_k.iter_mut() {
            *v = 3.5;
        }
        layer.init_to_ones();
        assert!(layer.l_k.iter().all(|&v| (v - 1.0).abs() < 1e-7));
        assert!(layer.l_v.iter().all(|&v| (v - 1.0).abs() < 1e-7));
        assert!(layer.l_ff.iter().all(|&v| (v - 1.0).abs() < 1e-7));
    }

    // 4. n_params_correct: n_params() == 3 * d_model
    #[test]
    fn n_params_correct() {
        let layer = make_layer(64);
        assert_eq!(layer.n_params(), 3 * 64);
    }

    // 5. l_k_shape: l_k.len() == d_model after new()
    #[test]
    fn l_k_shape() {
        let layer = make_layer(24);
        assert_eq!(layer.l_k.len(), 24);
    }

    // 6. l_v_shape: l_v.len() == d_model after new()
    #[test]
    fn l_v_shape() {
        let layer = make_layer(24);
        assert_eq!(layer.l_v.len(), 24);
    }

    // 7. l_ff_shape: l_ff.len() == d_model after new()
    #[test]
    fn l_ff_shape() {
        let layer = make_layer(24);
        assert_eq!(layer.l_ff.len(), 24);
    }

    // 8. different_layers_independent: two Ia3Layers have independent vectors
    #[test]
    fn different_layers_independent() {
        let mut layer_a = make_layer(12);
        let layer_b = make_layer(12);
        for v in layer_a.l_k.iter_mut() {
            *v = 99.0;
        }
        // Mutating layer_a must not affect layer_b.
        assert!(layer_b.l_k.iter().all(|&v| (v - 1.0).abs() < 1e-7));
    }

    // 9. rescale_1_no_effect_after_init: rescale with 1.0 scale gives identity output
    #[test]
    fn rescale_1_no_effect_after_init() {
        let layer = make_layer(10);
        // All scales are 1.0 after new(); rescaling should be a no-op.
        let input: Vec<f32> = (0..10).map(|i| (i as f32) * 0.7 - 3.0).collect();
        let out = layer.rescale_keys(&input).expect("rescale");
        for (&inp, &out_v) in input.iter().zip(out.iter()) {
            assert!(
                (inp - out_v).abs() < 1e-7,
                "expected identity: {inp} != {out_v}"
            );
        }
    }

    // 10. d_model_0_error: Ia3Config{d_model:0} returns Err
    #[test]
    fn d_model_0_error() {
        let cfg = Ia3Config {
            d_model: 0,
            n_layers: 1,
        };
        let mut rng = LcgRng::new(0);
        let result = Ia3Layer::new(&cfg, &mut rng);
        match result {
            Err(PeftError::DimensionMismatch {
                expected: 1,
                got: 0,
            }) => {}
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }
}
