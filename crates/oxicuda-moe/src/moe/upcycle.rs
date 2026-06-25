//! Sparse upcycling: initialise a sparse MoE FFN from a *dense* FFN checkpoint.
//!
//! Implements the warm-start procedure from:
//! Komatsuzaki et al. "Sparse Upcycling: Training Mixture-of-Experts from Dense
//! Checkpoints." ICLR 2023.
//!
//! A pre-trained dense transformer block has a single feed-forward network
//! `FFN(x) = W2 · act(W1 · x + b1) + b2`. Upcycling turns that one FFN into an
//! `n_experts`-way MoE layer by **copying the dense weights into every expert**
//! and attaching a freshly-initialised router. Immediately after upcycling each
//! expert is bit-identical to the source FFN, so — whichever expert a token is
//! routed to — its contribution is the same `FFN(x)` up to the router's scalar
//! gate weight `g_t`: a top-1 MoE outputs `g_t · FFN(x)`, a scalar multiple of
//! the dense forward pass. Training then specialises the experts and the router
//! learns to sharpen `g_t` toward `1`.
//!
//! To break the expert symmetry that would otherwise cause every expert to
//! receive identical gradients, a small amount of zero-mean Gaussian
//! perturbation may be added to the copied weights (`perturb_std > 0`). The
//! router is initialised from `N(0, router_init_std²)` exactly like
//! [`crate::routing::top_k::TopKRouter`].
//!
//! This module operates purely on CPU weight buffers; it is the data-movement
//! primitive a training harness invokes once when loading a dense checkpoint.

use crate::error::{MoeError, MoeResult};
use crate::expert::bank::ExpertBank;
use crate::expert::ffn::{ExpertActivation, ExpertFfn};
use crate::handle::LcgRng;
use crate::layer::moe_layer::{MoeLayer, MoeLayerConfig};

/// A dense FFN checkpoint to upcycle from.
///
/// Layout matches [`ExpertFfn`]: `w1` is row-major `[ffn_dim · input_dim]`,
/// `b1` is `[ffn_dim]`, `w2` is row-major `[input_dim · ffn_dim]`, and `b2` is
/// `[input_dim]`.
#[derive(Debug, Clone)]
pub struct DenseFfnCheckpoint {
    /// First-layer weights, row-major `[ffn_dim · input_dim]`.
    pub w1: Vec<f32>,
    /// First-layer biases, `[ffn_dim]`.
    pub b1: Vec<f32>,
    /// Second-layer weights, row-major `[input_dim · ffn_dim]`.
    pub w2: Vec<f32>,
    /// Second-layer biases, `[input_dim]`.
    pub b2: Vec<f32>,
    /// Input / output feature dimension.
    pub input_dim: usize,
    /// FFN hidden dimension.
    pub ffn_dim: usize,
    /// Activation function of the dense FFN.
    pub activation: ExpertActivation,
}

impl DenseFfnCheckpoint {
    /// Validate the checkpoint's buffer shapes.
    ///
    /// # Errors
    /// Returns [`MoeError`] for a zero dimension or a weight / bias length that
    /// disagrees with `input_dim` / `ffn_dim`.
    pub fn validate(&self) -> MoeResult<()> {
        if self.input_dim == 0 {
            return Err(MoeError::InvalidInputDim {
                dim: self.input_dim,
            });
        }
        if self.ffn_dim == 0 {
            return Err(MoeError::InvalidHiddenDim { dim: self.ffn_dim });
        }
        let w1_len = self.ffn_dim * self.input_dim;
        if self.w1.len() != w1_len {
            return Err(MoeError::DimensionMismatch {
                expected: w1_len,
                got: self.w1.len(),
            });
        }
        if self.b1.len() != self.ffn_dim {
            return Err(MoeError::DimensionMismatch {
                expected: self.ffn_dim,
                got: self.b1.len(),
            });
        }
        let w2_len = self.input_dim * self.ffn_dim;
        if self.w2.len() != w2_len {
            return Err(MoeError::DimensionMismatch {
                expected: w2_len,
                got: self.w2.len(),
            });
        }
        if self.b2.len() != self.input_dim {
            return Err(MoeError::DimensionMismatch {
                expected: self.input_dim,
                got: self.b2.len(),
            });
        }
        if self.w1.iter().chain(self.w2.iter()).any(|v| !v.is_finite()) {
            return Err(MoeError::NanEncountered {
                context: "dense checkpoint weights".to_string(),
            });
        }
        Ok(())
    }

    /// Reconstruct the single dense [`ExpertFfn`] this checkpoint represents.
    ///
    /// # Errors
    /// Propagates [`Self::validate`].
    pub fn to_expert(&self) -> MoeResult<ExpertFfn> {
        self.validate()?;
        Ok(ExpertFfn {
            w1: self.w1.clone(),
            b1: self.b1.clone(),
            w2: self.w2.clone(),
            b2: self.b2.clone(),
            input_dim: self.input_dim,
            ffn_dim: self.ffn_dim,
            activation: self.activation,
        })
    }
}

/// Settings controlling the upcycling operation.
#[derive(Debug, Clone)]
pub struct UpcycleConfig {
    /// Number of experts to replicate the dense FFN into (`> 0`).
    pub n_experts: usize,
    /// Symmetry-breaking perturbation std added to each expert's copied weights
    /// (`0.0` = exact copies, every expert identical).
    pub perturb_std: f32,
    /// Router weight init std (`N(0, router_init_std²)`).
    pub router_init_std: f32,
}

impl Default for UpcycleConfig {
    fn default() -> Self {
        Self {
            n_experts: 8,
            perturb_std: 0.0,
            router_init_std: 0.01,
        }
    }
}

impl UpcycleConfig {
    /// Validate the upcycling configuration.
    ///
    /// # Errors
    /// Returns [`MoeError`] for zero experts or a non-finite / negative std.
    pub fn validate(&self) -> MoeResult<()> {
        if self.n_experts == 0 {
            return Err(MoeError::InvalidExpertCount {
                n_experts: self.n_experts,
            });
        }
        if !self.perturb_std.is_finite() || self.perturb_std < 0.0 {
            return Err(MoeError::Internal {
                msg: format!("invalid perturb_std {}", self.perturb_std),
            });
        }
        if !self.router_init_std.is_finite() || self.router_init_std < 0.0 {
            return Err(MoeError::Internal {
                msg: format!("invalid router_init_std {}", self.router_init_std),
            });
        }
        Ok(())
    }
}

/// Build an [`ExpertBank`] whose every expert is a (optionally perturbed) copy
/// of `checkpoint`.
///
/// With `perturb_std == 0` all experts are bit-identical to the dense FFN.
///
/// # Errors
/// Propagates checkpoint / config validation.
pub fn upcycle_expert_bank(
    checkpoint: &DenseFfnCheckpoint,
    cfg: &UpcycleConfig,
    rng: &mut LcgRng,
) -> MoeResult<ExpertBank> {
    checkpoint.validate()?;
    cfg.validate()?;

    let mut experts: Vec<ExpertFfn> = Vec::with_capacity(cfg.n_experts);
    for _ in 0..cfg.n_experts {
        let mut w1 = checkpoint.w1.clone();
        let mut w2 = checkpoint.w2.clone();
        if cfg.perturb_std > 0.0 {
            add_noise(&mut w1, cfg.perturb_std, rng);
            add_noise(&mut w2, cfg.perturb_std, rng);
        }
        experts.push(ExpertFfn {
            w1,
            b1: checkpoint.b1.clone(),
            w2,
            b2: checkpoint.b2.clone(),
            input_dim: checkpoint.input_dim,
            ffn_dim: checkpoint.ffn_dim,
            activation: checkpoint.activation,
        });
    }

    ExpertBank::from_experts(experts)
}

/// Upcycle a dense checkpoint into a complete top-`top_k` [`MoeLayer`].
///
/// The returned layer's experts are copies of the dense FFN; its router is
/// freshly initialised. With `top_k == 1` and `perturb_std == 0` the layer's
/// per-token output equals the dense FFN output up to routing-induced token
/// drops (none occur while `capacity_factor ≥ 1`).
///
/// # Errors
/// Propagates checkpoint / config validation and [`MoeLayer::new`] errors.
pub fn upcycle_moe_layer(
    checkpoint: &DenseFfnCheckpoint,
    cfg: &UpcycleConfig,
    top_k: usize,
    capacity_factor: f32,
    rng: &mut LcgRng,
) -> MoeResult<MoeLayer> {
    checkpoint.validate()?;
    cfg.validate()?;

    // Build the layer (random experts + random router) then overwrite each
    // expert's weights with the (optionally perturbed) dense copy.
    let layer_cfg = MoeLayerConfig {
        input_dim: checkpoint.input_dim,
        ffn_dim: checkpoint.ffn_dim,
        n_experts: cfg.n_experts,
        top_k,
        capacity_factor,
        load_balance_coef: 0.01,
        router_z_loss_coef: 0.001,
        activation: checkpoint.activation,
    };
    let mut layer = MoeLayer::new(layer_cfg, rng)?;

    for expert in layer.experts_mut().experts_mut() {
        expert.w1.copy_from_slice(&checkpoint.w1);
        expert.b1.copy_from_slice(&checkpoint.b1);
        expert.w2.copy_from_slice(&checkpoint.w2);
        expert.b2.copy_from_slice(&checkpoint.b2);
        if cfg.perturb_std > 0.0 {
            add_noise(&mut expert.w1, cfg.perturb_std, rng);
            add_noise(&mut expert.w2, cfg.perturb_std, rng);
        }
    }

    Ok(layer)
}

/// Add zero-mean Gaussian noise of the given std in-place.
fn add_noise(buf: &mut [f32], std_dev: f32, rng: &mut LcgRng) {
    let mut noise = vec![0.0_f32; buf.len()];
    rng.fill_normal_scaled(&mut noise, std_dev);
    for (b, n) in buf.iter_mut().zip(noise.iter()) {
        *b += *n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_checkpoint(input_dim: usize, ffn_dim: usize, rng: &mut LcgRng) -> DenseFfnCheckpoint {
        let mut w1 = vec![0.0_f32; ffn_dim * input_dim];
        let mut w2 = vec![0.0_f32; input_dim * ffn_dim];
        let mut b1 = vec![0.0_f32; ffn_dim];
        let mut b2 = vec![0.0_f32; input_dim];
        rng.fill_normal_scaled(&mut w1, 0.1);
        rng.fill_normal_scaled(&mut w2, 0.1);
        rng.fill_normal_scaled(&mut b1, 0.05);
        rng.fill_normal_scaled(&mut b2, 0.05);
        DenseFfnCheckpoint {
            w1,
            b1,
            w2,
            b2,
            input_dim,
            ffn_dim,
            activation: ExpertActivation::Gelu,
        }
    }

    #[test]
    fn checkpoint_validates_shapes() {
        let mut rng = LcgRng::new(1);
        let ckpt = sample_checkpoint(8, 32, &mut rng);
        assert!(ckpt.validate().is_ok());

        let mut bad = ckpt.clone();
        bad.w1.pop();
        assert!(matches!(
            bad.validate(),
            Err(MoeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn exact_copies_are_identical() {
        let mut rng = LcgRng::new(2);
        let ckpt = sample_checkpoint(8, 24, &mut rng);
        let cfg = UpcycleConfig {
            n_experts: 4,
            perturb_std: 0.0,
            router_init_std: 0.01,
        };
        let bank = upcycle_expert_bank(&ckpt, &cfg, &mut rng).expect("upcycle should succeed");
        assert_eq!(bank.n_experts, 4);
        // Every expert reproduces the dense FFN on the same input.
        let dense = ckpt.to_expert().expect("to_expert");
        let x = vec![0.5_f32; 8];
        let dense_out = dense.forward(&x).expect("dense forward");
        for e in 0..4 {
            let e_out = bank
                .forward_expert(e, &x, 1)
                .expect("forward_expert should succeed");
            for (a, b) in dense_out.iter().zip(e_out.iter()) {
                assert!((a - b).abs() < 1e-6, "expert {e} diverged: {a} vs {b}");
            }
        }
    }

    #[test]
    fn perturbation_breaks_symmetry() {
        let mut rng = LcgRng::new(3);
        let ckpt = sample_checkpoint(8, 24, &mut rng);
        let cfg = UpcycleConfig {
            n_experts: 3,
            perturb_std: 0.05,
            router_init_std: 0.01,
        };
        let bank = upcycle_expert_bank(&ckpt, &cfg, &mut rng).expect("upcycle should succeed");
        let x = vec![0.7_f32; 8];
        let out0 = bank.forward_expert(0, &x, 1).expect("forward 0");
        let out1 = bank.forward_expert(1, &x, 1).expect("forward 1");
        // With perturbation the experts must differ.
        let max_diff = out0
            .iter()
            .zip(out1.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1e-5,
            "perturbed experts should differ, diff={max_diff}"
        );
    }

    #[test]
    fn upcycled_top1_layer_is_scalar_multiple_of_dense() {
        // After exact upcycling every expert equals the dense FFN, so a top-1
        // MoE output for a token is exactly `gate_weight · dense(x)` (the layer
        // weights the chosen expert by its softmax probability, not by 1). The
        // load-bearing upcycling property is therefore: the MoE output is a
        // *single positive scalar multiple* (in (0, 1]) of the dense output,
        // identical across every feature component.
        let mut rng = LcgRng::new(4);
        let input_dim = 16;
        let ckpt = sample_checkpoint(input_dim, 48, &mut rng);
        let cfg = UpcycleConfig {
            n_experts: 4,
            perturb_std: 0.0,
            router_init_std: 0.01,
        };
        let layer = upcycle_moe_layer(&ckpt, &cfg, 1, 2.0, &mut rng).expect("upcycle layer");
        let dense = ckpt.to_expert().expect("to_expert");

        let n_tokens = 6;
        let mut x = vec![0.0_f32; n_tokens * input_dim];
        rng.fill_normal_scaled(&mut x, 0.3);
        let out = layer.forward(&x, n_tokens).expect("forward should succeed");
        // With capacity_factor 2.0 no token is dropped.
        assert_eq!(out.n_overflows, 0);

        for tok in 0..n_tokens {
            let x_tok = &x[tok * input_dim..(tok + 1) * input_dim];
            let dense_out = dense.forward(x_tok).expect("dense forward");
            let moe_out = &out.hidden[tok * input_dim..(tok + 1) * input_dim];

            // Recover the per-token scalar from the component with the largest
            // dense magnitude (numerically safest), then verify every component
            // matches that single ratio.
            let (ref_i, &ref_d) = dense_out
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.abs()
                        .partial_cmp(&b.abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("dense output non-empty");
            assert!(ref_d.abs() > 1e-6, "degenerate dense output");
            let ratio = moe_out[ref_i] / ref_d;
            assert!(
                ratio > 0.0 && ratio <= 1.0 + 1e-4,
                "token {tok}: gate weight {ratio} not in (0, 1]"
            );
            for (a, b) in dense_out.iter().zip(moe_out.iter()) {
                assert!(
                    (ratio * a - b).abs() < 1e-4,
                    "token {tok}: moe {b} != gate {ratio} * dense {a}"
                );
            }
        }
    }

    #[test]
    fn zero_experts_rejected() {
        let mut rng = LcgRng::new(5);
        let ckpt = sample_checkpoint(8, 16, &mut rng);
        let cfg = UpcycleConfig {
            n_experts: 0,
            perturb_std: 0.0,
            router_init_std: 0.01,
        };
        assert!(matches!(
            upcycle_expert_bank(&ckpt, &cfg, &mut rng),
            Err(MoeError::InvalidExpertCount { .. })
        ));
    }
}
