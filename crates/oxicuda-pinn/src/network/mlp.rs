//! Multi-layer perceptron (MLP) with SIREN initialization and tape-based AD.

use crate::autodiff::tape::{Tape, Var};
use crate::error::{PinnError, PinnResult};
use crate::handle::LcgRng;

/// Activation functions for MLP layers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    Tanh,
    Sin,
    Relu,
    Gelu,
}

/// MLP configuration.
pub struct MlpConfig {
    pub layer_widths: Vec<usize>,
    pub activation: Activation,
    /// Frequency scaling for Sin (SIREN) activation.
    pub omega_0: f32,
}

/// Multi-layer perceptron.
pub struct Mlp {
    config: MlpConfig,
    weights: Vec<Vec<f32>>,
    biases: Vec<Vec<f32>>,
}

fn apply_activation(x: f32, act: Activation, omega_0: f32) -> f32 {
    match act {
        Activation::Tanh => x.tanh(),
        Activation::Sin => (omega_0 * x).sin(),
        Activation::Relu => x.max(0.0),
        Activation::Gelu => {
            let c = (2.0_f32 / std::f32::consts::PI).sqrt();
            0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
        }
    }
}

impl Mlp {
    /// Construct a new MLP with Xavier-style initialization.
    pub fn new(config: MlpConfig, rng: &mut LcgRng) -> PinnResult<Self> {
        let widths = &config.layer_widths;
        if widths.len() < 2 {
            return Err(PinnError::InvalidNetworkDepth {
                depth: widths.len(),
            });
        }
        for &w in widths {
            if w == 0 {
                return Err(PinnError::InvalidLayerWidth);
            }
        }

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for win in widths.windows(2) {
            let d_in = win[0];
            let d_out = win[1];
            let scale = (2.0 / d_in as f32).sqrt();
            let w: Vec<f32> = (0..d_out * d_in)
                .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
                .collect();
            biases.push(vec![0.0_f32; d_out]);
            weights.push(w);
        }

        Ok(Self {
            config,
            weights,
            biases,
        })
    }

    /// SIREN initialization.
    ///
    /// First layer: `U(-1/d_in, 1/d_in)`.
    /// Hidden layers: `U(-sqrt(6/d_in)/ω₀, sqrt(6/d_in)/ω₀)`.
    pub fn siren_init(
        layer_widths: Vec<usize>,
        omega_0: f32,
        rng: &mut LcgRng,
    ) -> PinnResult<Self> {
        if layer_widths.len() < 2 {
            return Err(PinnError::InvalidNetworkDepth {
                depth: layer_widths.len(),
            });
        }
        for &w in &layer_widths {
            if w == 0 {
                return Err(PinnError::InvalidLayerWidth);
            }
        }

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for (layer_idx, win) in layer_widths.windows(2).enumerate() {
            let d_in = win[0];
            let d_out = win[1];
            let half_range = if layer_idx == 0 {
                1.0 / d_in as f32
            } else {
                (6.0_f32 / d_in as f32).sqrt() / omega_0
            };
            let w: Vec<f32> = (0..d_out * d_in)
                .map(|_| (rng.next_f32() * 2.0 - 1.0) * half_range)
                .collect();
            biases.push(vec![0.0_f32; d_out]);
            weights.push(w);
        }

        Ok(Self {
            config: MlpConfig {
                layer_widths,
                activation: Activation::Sin,
                omega_0,
            },
            weights,
            biases,
        })
    }

    /// Forward pass: input `[d_in]` → output `[d_out]`.
    pub fn forward(&self, x: &[f32]) -> PinnResult<Vec<f32>> {
        let widths = &self.config.layer_widths;
        if x.len() != widths[0] {
            return Err(PinnError::DimensionMismatch {
                expected: widths[0],
                got: x.len(),
            });
        }

        let act = self.config.activation;
        let omega_0 = self.config.omega_0;
        let n_layers = self.weights.len();
        let mut h = x.to_vec();

        for (l, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let d_in = h.len();
            let d_out = b.len();
            let pre: Vec<f32> = (0..d_out)
                .map(|i| {
                    let dot: f32 = (0..d_in).map(|j| w[i * d_in + j] * h[j]).sum();
                    dot + b[i]
                })
                .collect();

            // No activation on last layer
            h = if l < n_layers - 1 {
                pre.into_iter()
                    .map(|v| apply_activation(v, act, omega_0))
                    .collect()
            } else {
                pre
            };

            if h.iter().any(|v| !v.is_finite()) {
                return Err(PinnError::NanEncountered {
                    location: "mlp_forward",
                });
            }
        }
        Ok(h)
    }

    /// Forward pass through tape AD, returning `Var` output nodes.
    ///
    /// Input variables `x_vars` must already be registered in `tape`.
    pub fn grad_input(&self, tape: &mut Tape, x_vars: &[Var]) -> PinnResult<Vec<Var>> {
        let widths = &self.config.layer_widths;
        if x_vars.len() != widths[0] {
            return Err(PinnError::DimensionMismatch {
                expected: widths[0],
                got: x_vars.len(),
            });
        }

        let n_layers = self.weights.len();
        let mut h: Vec<Var> = x_vars.to_vec();

        for (l, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let d_in = h.len();
            let d_out = b.len();

            let mut next_h = Vec::with_capacity(d_out);
            for i in 0..d_out {
                // Compute dot product + bias through tape
                let mut acc = tape.constant(b[i]);
                for j in 0..d_in {
                    let wij = tape.constant(w[i * d_in + j]);
                    let prod = tape.mul(wij, h[j]);
                    acc = tape.add(acc, prod);
                }
                // Apply activation (last layer: linear)
                let activated = if l < n_layers - 1 {
                    match self.config.activation {
                        Activation::Tanh => tape.tanh(acc),
                        Activation::Sin => {
                            let scaled = tape.scale(acc, self.config.omega_0);
                            tape.sin(scaled)
                        }
                        Activation::Relu => {
                            let v = tape.value(acc);
                            if v > 0.0 { acc } else { tape.constant(0.0) }
                        }
                        Activation::Gelu => {
                            // approximate GeLU via tanh: 0.5x(1+tanh(sqrt(2/pi)*(x + 0.044715x^3)))
                            let c = (2.0_f32 / std::f32::consts::PI).sqrt();
                            let v = tape.value(acc);
                            let x3 = v * v * v;
                            let inner = c * (v + 0.044715 * x3);
                            let t_inner = tape.constant(inner.tanh());
                            let one = tape.constant(1.0);
                            let s = tape.add(one, t_inner);
                            let half = tape.constant(0.5);
                            let hs = tape.mul(half, s);
                            tape.mul(acc, hs)
                        }
                    }
                } else {
                    acc
                };
                next_h.push(activated);
            }
            h = next_h;
        }
        Ok(h)
    }

    /// Simple gradient-descent parameter update.
    pub fn step(&mut self, grad_w: &[Vec<f32>], grad_b: &[Vec<f32>], lr: f32) {
        for (layer_idx, (w, b)) in self
            .weights
            .iter_mut()
            .zip(self.biases.iter_mut())
            .enumerate()
        {
            if layer_idx < grad_w.len() {
                for (wi, gwi) in w.iter_mut().zip(grad_w[layer_idx].iter()) {
                    *wi -= lr * gwi;
                }
            }
            if layer_idx < grad_b.len() {
                for (bi, gbi) in b.iter_mut().zip(grad_b[layer_idx].iter()) {
                    *bi -= lr * gbi;
                }
            }
        }
    }

    /// Return reference to weights.
    pub fn weights(&self) -> &[Vec<f32>] {
        &self.weights
    }

    /// Return reference to biases.
    pub fn biases(&self) -> &[Vec<f32>] {
        &self.biases
    }

    /// Return the layer widths.
    pub fn layer_widths(&self) -> &[usize] {
        &self.config.layer_widths
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlp_construct_no_panic() {
        let mut rng = LcgRng::new(1);
        let cfg = MlpConfig {
            layer_widths: vec![2, 8, 8, 1],
            activation: Activation::Tanh,
            omega_0: 1.0,
        };
        let _mlp = Mlp::new(cfg, &mut rng).unwrap();
    }

    #[test]
    fn mlp_forward_shape() {
        let mut rng = LcgRng::new(2);
        let cfg = MlpConfig {
            layer_widths: vec![2, 8, 1],
            activation: Activation::Tanh,
            omega_0: 1.0,
        };
        let mlp = Mlp::new(cfg, &mut rng).unwrap();
        let out = mlp.forward(&[0.3, 0.7]).unwrap();
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn mlp_tanh_output_bounded() {
        // Tanh hidden → intermediate values bounded; final layer unbounded
        let mut rng = LcgRng::new(3);
        let cfg = MlpConfig {
            layer_widths: vec![1, 8, 8, 1],
            activation: Activation::Tanh,
            omega_0: 1.0,
        };
        let mlp = Mlp::new(cfg, &mut rng).unwrap();
        for i in 0..10 {
            let x = i as f32 * 0.1;
            let out = mlp.forward(&[x]).unwrap();
            assert!(out[0].is_finite(), "Tanh MLP output not finite at x={x}");
        }
    }

    #[test]
    fn mlp_no_nan() {
        let mut rng = LcgRng::new(4);
        let cfg = MlpConfig {
            layer_widths: vec![3, 16, 8, 2],
            activation: Activation::Relu,
            omega_0: 1.0,
        };
        let mlp = Mlp::new(cfg, &mut rng).unwrap();
        let out = mlp.forward(&[1.0, -1.0, 0.5]).unwrap();
        assert!(
            out.iter().all(|v| v.is_finite()),
            "ReLU MLP output not finite"
        );
    }

    #[test]
    fn mlp_gelu_forward() {
        let mut rng = LcgRng::new(5);
        let cfg = MlpConfig {
            layer_widths: vec![2, 8, 1],
            activation: Activation::Gelu,
            omega_0: 1.0,
        };
        let mlp = Mlp::new(cfg, &mut rng).unwrap();
        let out = mlp.forward(&[0.5, -0.5]).unwrap();
        assert!(out[0].is_finite());
    }

    #[test]
    fn siren_init_weights_in_range() {
        let mut rng = LcgRng::new(6);
        let mlp = Mlp::siren_init(vec![2, 32, 32, 1], 30.0, &mut rng).unwrap();
        let d_in = 2;
        // First layer weights in [-1/d_in, 1/d_in]
        for &w in &mlp.weights()[0] {
            assert!(
                w.abs() <= 1.0 / d_in as f32 + 1e-5,
                "SIREN first layer weight out of range: {w}"
            );
        }
    }

    #[test]
    fn siren_forward_finite() {
        let mut rng = LcgRng::new(7);
        let mlp = Mlp::siren_init(vec![1, 16, 1], 30.0, &mut rng).unwrap();
        for i in 0..10 {
            let x = i as f32 * 0.1;
            let out = mlp.forward(&[x]).unwrap();
            assert!(out[0].is_finite(), "SIREN output not finite at x={x}");
        }
    }

    #[test]
    fn mlp_sin_activation() {
        let mut rng = LcgRng::new(8);
        let cfg = MlpConfig {
            layer_widths: vec![1, 8, 1],
            activation: Activation::Sin,
            omega_0: 1.0,
        };
        let mlp = Mlp::new(cfg, &mut rng).unwrap();
        let out = mlp.forward(&[0.5]).unwrap();
        assert!(out[0].is_finite());
    }

    #[test]
    fn mlp_grad_input_tape_shape() {
        let mut rng = LcgRng::new(9);
        let cfg = MlpConfig {
            layer_widths: vec![2, 4, 1],
            activation: Activation::Tanh,
            omega_0: 1.0,
        };
        let mlp = Mlp::new(cfg, &mut rng).unwrap();
        let mut tape = Tape::new();
        let x0 = tape.variable(0.3);
        let x1 = tape.variable(0.7);
        let out_vars = mlp.grad_input(&mut tape, &[x0, x1]).unwrap();
        assert_eq!(out_vars.len(), 1);
    }

    #[test]
    fn mlp_step_changes_weights() {
        let mut rng = LcgRng::new(10);
        let cfg = MlpConfig {
            layer_widths: vec![2, 4, 1],
            activation: Activation::Tanh,
            omega_0: 1.0,
        };
        let mut mlp = Mlp::new(cfg, &mut rng).unwrap();
        let w_before = mlp.weights()[0][0];
        let grad_w: Vec<Vec<f32>> = mlp
            .weights()
            .iter()
            .map(|w| vec![1.0_f32; w.len()])
            .collect();
        let grad_b: Vec<Vec<f32>> = mlp
            .biases()
            .iter()
            .map(|b| vec![0.0_f32; b.len()])
            .collect();
        mlp.step(&grad_w, &grad_b, 0.1);
        let w_after = mlp.weights()[0][0];
        assert!(
            (w_before - w_after).abs() > 1e-6,
            "Step should change weights"
        );
    }
}
