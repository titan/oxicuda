//! Spiking layers built on top of [`crate::neuron::lif`].
//!
//! Provides drop-in CPU implementations of the most common neural-network
//! building blocks adapted to the spiking setting: a fully-connected linear
//! layer, a 2-D convolutional layer, a pooling layer, and a recurrent layer
//! with self-connections. Each layer composes a numerical synaptic
//! integration (`W · x + b` or sliding-window correlation) with a per-neuron
//! LIF step from the neuron module.

/// Recurrent spiking layer with self-connections.
pub mod recurrent;
/// Spiking 2D convolution layer.
pub mod spiking_conv;
/// Spiking fully-connected layer (Linear + LIF).
pub mod spiking_linear;
/// Spiking max/avg pooling layer.
pub mod spiking_pool;
