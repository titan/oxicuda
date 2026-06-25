//! Spiking layers built on top of [`crate::neuron::lif`].
//!
//! Provides drop-in CPU implementations of the most common neural-network
//! building blocks adapted to the spiking setting: a fully-connected linear
//! layer, a 2-D convolutional layer, a transposed-convolution (up-sampling)
//! layer, a pooling layer, recurrent layers (single- and multi-time-constant),
//! and stochastic regularisers (spike dropout and stochastic depth). Each
//! layer composes a numerical synaptic integration (`W · x + b`,
//! sliding-window correlation, or scatter-add up-sampling) with a per-neuron
//! LIF step from the neuron module.

/// Multi-τ recurrent LIF layer with per-neuron multi-time-constant sub-states.
pub mod multi_tau_lif;
/// Recurrent spiking layer with self-connections.
pub mod recurrent;
/// Spikformer spike-driven self-attention layer.
pub mod spiking_attention;
/// Spiking 2D convolution layer.
pub mod spiking_conv;
/// Spiking 2D transposed convolution / deconvolution (up-sampling) layer.
pub mod spiking_deconv;
/// Spiking fully-connected layer (Linear + LIF).
pub mod spiking_linear;
/// Spiking max/avg pooling layer.
pub mod spiking_pool;
/// Spiking dropout and stochastic-depth regularisers.
pub mod spiking_regularization;
/// Spiking residual / skip-connection layer (spiking ResNet block).
pub mod spiking_residual;
/// Spiking transformer encoder block (Spikformer: SSA + spiking MLP + residual).
pub mod spiking_transformer;
/// Spiking variational autoencoder (FSVAE-style) building blocks.
pub mod spiking_vae;
/// Threshold-dependent batch normalisation (tdBN).
pub mod td_bn;
