use crate::error::{PeftError, PeftResult};
use crate::handle::PeftHandle;
use crate::ia3::ia3::{Ia3Placement, Ia3Vector};
use crate::lora::lora::mat_vec_mul;

/// GELU activation: `0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))`.
///
/// Defined locally to avoid cross-module visibility coupling.
fn gelu(x: f32) -> f32 {
    const C0: f32 = 0.797_884_56;
    const C1: f32 = 0.044_715;
    let inner = C0 * (x + C1 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Configuration for a hybrid (IA)³ + bottleneck adapter layer.
///
/// When `bottleneck == 0` the layer is pure IA³ (element-wise scaling only).
/// When `bottleneck > 0` a residual bottleneck correction is added after scaling.
#[derive(Debug, Clone)]
pub struct Ia3AdapterConfig {
    /// Activation dimension; must be ≥ 1.
    pub size: usize,
    /// Bottleneck hidden dimension; `0` means pure IA³ with no adapter.
    pub bottleneck: usize,
    /// Which transformer position this adapter belongs to.
    pub placement: Ia3Placement,
    /// Initial value of each IA³ scale factor (`1.0` = identity start).
    pub init_scale: f32,
}

/// Hybrid (IA)³ + bottleneck adapter (He et al. 2022 "Towards a Unified View of PEFT").
///
/// Forward: `scaled = l ⊙ x`; if `bottleneck > 0`, add bottleneck residual; else return `scaled`.
/// The adapter is initialised so that the residual correction is exactly zero (up is zero-init),
/// making the full module start as a pure IA³ scaler.
#[derive(Debug, Clone)]
pub struct Ia3AdapterLayer {
    /// IA³ scaling vector.
    pub ia3_vec: Ia3Vector,
    /// Down-projection weight: `bottleneck × size`, Kaiming-uniform init; `None` if pure IA³.
    pub down: Option<Vec<f32>>,
    /// Down-projection bias: length `bottleneck`, zero init; `None` if pure IA³.
    pub down_bias: Option<Vec<f32>>,
    /// Up-projection weight: `size × bottleneck`, zero init; `None` if pure IA³.
    pub up: Option<Vec<f32>>,
    /// Up-projection bias: length `size`, zero init; `None` if pure IA³.
    pub up_bias: Option<Vec<f32>>,
    /// Configuration used to construct this layer.
    pub cfg: Ia3AdapterConfig,
}

impl Ia3AdapterLayer {
    /// Construct an `Ia3AdapterLayer`.
    ///
    /// # Errors
    /// Returns `Err` when `cfg.size == 0`.
    pub fn new(cfg: Ia3AdapterConfig, handle: &mut PeftHandle) -> PeftResult<Self> {
        if cfg.size == 0 {
            return Err(PeftError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }

        // Build IA³ vector, then overwrite scale with init_scale.
        let mut ia3_vec = Ia3Vector::new(cfg.size, cfg.placement.clone());
        for s in ia3_vec.scale.iter_mut() {
            *s = cfg.init_scale;
        }

        let (down, down_bias, up, up_bias) = if cfg.bottleneck > 0 {
            let kaiming_bound = (6.0_f32 / cfg.size as f32).sqrt();
            let down_w: Vec<f32> = (0..cfg.bottleneck * cfg.size)
                .map(|_| {
                    let u = handle.rng.next_f32();
                    (u * 2.0 - 1.0) * kaiming_bound
                })
                .collect();
            let down_b = vec![0.0_f32; cfg.bottleneck];
            let up_w = vec![0.0_f32; cfg.size * cfg.bottleneck];
            let up_b = vec![0.0_f32; cfg.size];
            (Some(down_w), Some(down_b), Some(up_w), Some(up_b))
        } else {
            (None, None, None, None)
        };

        Ok(Self {
            ia3_vec,
            down,
            down_bias,
            up,
            up_bias,
            cfg,
        })
    }

    /// Forward pass: IA³ scaling followed by optional bottleneck residual.
    ///
    /// `x` must have length `cfg.size`. Returns a vector of the same length.
    ///
    /// # Errors
    /// Returns `Err` when `x.len() != cfg.size`.
    pub fn forward(&self, x: &[f32]) -> PeftResult<Vec<f32>> {
        if x.len() != self.cfg.size {
            return Err(PeftError::DimensionMismatch {
                expected: self.cfg.size,
                got: x.len(),
            });
        }

        // IA³ element-wise scaling: scaled = l ⊙ x.
        let scaled = self.ia3_vec.apply(x);

        if self.cfg.bottleneck > 0 {
            // These unwraps are inside a test-free path but are only reached when
            // bottleneck > 0, which is the same condition under which the fields are Some.
            // Clippy/rustc will not flag these because we have no #[allow(...)].
            let down_w = self.down.as_deref().ok_or(PeftError::Internal {
                msg: "down weight missing".into(),
            })?;
            let down_b = self.down_bias.as_deref().ok_or(PeftError::Internal {
                msg: "down bias missing".into(),
            })?;
            let up_w = self.up.as_deref().ok_or(PeftError::Internal {
                msg: "up weight missing".into(),
            })?;
            let up_b = self.up_bias.as_deref().ok_or(PeftError::Internal {
                msg: "up bias missing".into(),
            })?;

            // Down projection + bias.
            let d_pre = mat_vec_mul(down_w, &scaled, self.cfg.bottleneck, self.cfg.size);
            let d_pre_biased: Vec<f32> = d_pre
                .iter()
                .zip(down_b.iter())
                .map(|(v, b)| v + b)
                .collect();

            // GELU activation.
            let d: Vec<f32> = d_pre_biased.iter().map(|&v| gelu(v)).collect();

            // Up projection + bias.
            let u_pre = mat_vec_mul(up_w, &d, self.cfg.size, self.cfg.bottleneck);
            let u: Vec<f32> = u_pre.iter().zip(up_b.iter()).map(|(v, b)| v + b).collect();

            // Residual: IA³ scaled + adapter correction.
            let output: Vec<f32> = scaled.iter().zip(u.iter()).map(|(s, c)| s + c).collect();
            Ok(output)
        } else {
            Ok(scaled)
        }
    }

    /// Total parameter count: IA³ + adapter weights and biases.
    #[must_use]
    pub fn total_params(&self) -> usize {
        self.ia3_params() + self.adapter_params()
    }

    /// IA³-only parameter count (equals `size`).
    #[must_use]
    pub fn ia3_params(&self) -> usize {
        self.cfg.size
    }

    /// Adapter (bottleneck) parameter count; `0` when `bottleneck == 0`.
    #[must_use]
    pub fn adapter_params(&self) -> usize {
        if self.cfg.bottleneck > 0 {
            2 * (self.cfg.size * self.cfg.bottleneck) + self.cfg.bottleneck + self.cfg.size
        } else {
            0
        }
    }

    /// Return `true` when the layer has no bottleneck adapter (pure IA³).
    #[must_use]
    pub fn is_pure_ia3(&self) -> bool {
        self.cfg.bottleneck == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::PeftHandle;
    use crate::ia3::ia3::Ia3Placement;

    fn make_handle(seed: u64) -> PeftHandle {
        PeftHandle::new(80, seed)
    }

    fn make_pure(size: usize, init_scale: f32, seed: u64) -> Ia3AdapterLayer {
        let cfg = Ia3AdapterConfig {
            size,
            bottleneck: 0,
            placement: Ia3Placement::FeedForward,
            init_scale,
        };
        let mut h = make_handle(seed);
        Ia3AdapterLayer::new(cfg, &mut h).unwrap()
    }

    fn make_hybrid(size: usize, bottleneck: usize, init_scale: f32, seed: u64) -> Ia3AdapterLayer {
        let cfg = Ia3AdapterConfig {
            size,
            bottleneck,
            placement: Ia3Placement::Key,
            init_scale,
        };
        let mut h = make_handle(seed);
        Ia3AdapterLayer::new(cfg, &mut h).unwrap()
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 1: bottleneck=0, init_scale=1.0 → forward(x) == x (identity).
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn pure_ia3_identity() {
        let layer = make_pure(4, 1.0, 1);
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let out = layer.forward(&x).unwrap();
        for (o, xi) in out.iter().zip(x.iter()) {
            assert!((o - xi).abs() < 1e-6, "{o} != {xi}");
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 2: bottleneck=0, init_scale=0.0 → forward(x) == zeros.
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn pure_ia3_scale_zero() {
        let layer = make_pure(4, 0.0, 2);
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let out = layer.forward(&x).unwrap();
        for &o in &out {
            assert!(o.abs() < 1e-6, "expected 0, got {o}");
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 3: bottleneck=0, init_scale=2.0 → forward(x) == 2*x.
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn pure_ia3_scale_two() {
        let layer = make_pure(4, 2.0, 3);
        let x = vec![0.5_f32, 1.5, -1.0, 3.0];
        let out = layer.forward(&x).unwrap();
        for (o, xi) in out.iter().zip(x.iter()) {
            assert!((o - 2.0 * xi).abs() < 1e-5, "{o} != {}", 2.0 * xi);
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 4: hybrid, up is zero-init → forward == IA³ scaled only.
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn hybrid_zero_up_is_scaled() {
        let layer = make_hybrid(4, 2, 1.0, 4);
        // up is zero-init so u_pre=0, output = scaled + 0 = scaled.
        let x = vec![1.0_f32, -1.0, 2.0, 0.5];
        let out = layer.forward(&x).unwrap();
        let scaled = layer.ia3_vec.apply(&x);
        for (o, s) in out.iter().zip(scaled.iter()) {
            assert!((o - s).abs() < 1e-5, "hybrid zero-up: {o} != {s}");
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 5: bottleneck=0 → is_pure_ia3() == true.
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn is_pure_ia3_true() {
        let layer = make_pure(4, 1.0, 5);
        assert!(layer.is_pure_ia3());
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 6: bottleneck=1 → is_pure_ia3() == false.
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn is_pure_ia3_false() {
        let layer = make_hybrid(4, 1, 1.0, 6);
        assert!(!layer.is_pure_ia3());
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 7: ia3_params() == size.
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn ia3_params_value() {
        let size = 8usize;
        let layer = make_pure(size, 1.0, 7);
        assert_eq!(layer.ia3_params(), size);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 8: bottleneck=4, size=8 → adapter_params() == 2*32+4+8 == 76.
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn adapter_params_formula() {
        let layer = make_hybrid(8, 4, 1.0, 8);
        assert_eq!(layer.adapter_params(), 2 * 32 + 4 + 8);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 9: bottleneck=0 → total_params() == size.
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn total_params_pure() {
        let size = 6usize;
        let layer = make_pure(size, 1.0, 9);
        assert_eq!(layer.total_params(), size);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 10: total_params() == ia3_params() + adapter_params().
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn total_params_hybrid() {
        let layer = make_hybrid(8, 4, 1.0, 10);
        assert_eq!(
            layer.total_params(),
            layer.ia3_params() + layer.adapter_params()
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 11: placement is preserved in ia3_vec.
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn placement_preserved() {
        let cfg = Ia3AdapterConfig {
            size: 4,
            bottleneck: 0,
            placement: Ia3Placement::Value,
            init_scale: 1.0,
        };
        let mut h = make_handle(11);
        let layer = Ia3AdapterLayer::new(cfg, &mut h).unwrap();
        assert_eq!(layer.ia3_vec.placement, Ia3Placement::Value);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 12: x.len() != size → Err.
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn dim_mismatch_err() {
        let layer = make_pure(4, 1.0, 12);
        let x = vec![1.0_f32; 5]; // wrong
        assert!(layer.forward(&x).is_err());
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 13: bottleneck=1, forward runs without error.
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn bottleneck_1_works() {
        let layer = make_hybrid(4, 1, 1.0, 13);
        let x = vec![0.1_f32, 0.2, -0.3, 0.4];
        let out = layer.forward(&x).unwrap();
        assert_eq!(out.len(), 4);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 14: ia3_vec.scale[0] == init_scale after construction.
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn init_scale_value() {
        let init_scale = 0.75_f32;
        let layer = make_pure(4, init_scale, 14);
        assert!((layer.ia3_vec.scale[0] - init_scale).abs() < 1e-7);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 15: same seed and init_scale → same forward output.
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn deterministic() {
        let x = vec![0.4_f32, -0.2, 0.8, -0.6];
        let layer_a = make_hybrid(4, 2, 1.0, 42);
        let layer_b = make_hybrid(4, 2, 1.0, 42);
        let out_a = layer_a.forward(&x).unwrap();
        let out_b = layer_b.forward(&x).unwrap();
        assert_eq!(out_a, out_b);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 16: down weights are in Kaiming uniform range.
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn down_kaiming_range() {
        let size = 8usize;
        let layer = make_hybrid(size, 4, 1.0, 16);
        let bound = (6.0_f32 / size as f32).sqrt() + 1e-5;
        let down = layer.down.as_ref().unwrap();
        for &v in down {
            assert!(
                v.abs() <= bound,
                "down weight {v} out of Kaiming range ±{bound}"
            );
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 17: gelu(0.0) ≈ 0.0.
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn gelu_at_zero() {
        let result = gelu(0.0);
        assert!(result.abs() < 1e-7, "gelu(0.0) = {result}");
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 18: bottleneck>0 → up field is all zeros.
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn up_zero_init() {
        let layer = make_hybrid(4, 2, 1.0, 18);
        let up = layer.up.as_ref().unwrap();
        for &v in up {
            assert!(v == 0.0, "up should be zero-init, got {v}");
        }
    }
}
