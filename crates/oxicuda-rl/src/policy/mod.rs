//! Policy distributions for discrete and continuous action spaces.
//!
//! * [`crate::policy::CategoricalPolicy`] — discrete actions via categorical / softmax distribution
//! * [`crate::policy::GaussianPolicy`] — continuous actions via diagonal Gaussian (reparameterised)
//! * [`crate::policy::DeterministicPolicy`] — deterministic policy for DDPG/TD3
//! * [`crate::policy::DecisionTransformer`] — sequence-based policy (Chen et al. 2021)
//! * [`crate::policy::EpsilonGreedy`] / [`crate::policy::Boltzmann`] — discrete
//!   action exploration strategies (Sutton & Barto 2018)
//! * [`crate::policy::NoisyLinear`] / [`crate::policy::dueling_q_values`] —
//!   Rainbow components: NoisyNet layers + dueling-Q aggregation (Hessel 2018)
//! * [`crate::policy::IcmReward`] / [`crate::policy::RndReward`] — intrinsic
//!   curiosity exploration bonuses (Pathak 2017 / Burda 2018)
//! * [`crate::policy::Plan2Explore`] — ensemble-disagreement exploration via
//!   self-supervised one-step world models (Sekar 2020)
//! * [`crate::policy::SimHashCount`] — static-hashing count-based exploration
//!   bonus (Tang 2017)

pub mod categorical;
pub mod curiosity;
pub mod decision_transformer;
pub mod deterministic;
pub mod exploration;
pub mod gaussian;
pub mod hash_count;
pub mod plan2explore;
pub mod rainbow;

pub use categorical::CategoricalPolicy;
pub use curiosity::{IcmConfig, IcmReward, RndReward, icm_intrinsic_reward, icm_inverse_loss};
pub use decision_transformer::{DecisionTransformer, DtConfig};
pub use deterministic::DeterministicPolicy;
pub use exploration::{Boltzmann, EpsilonGreedy};
pub use gaussian::GaussianPolicy;
pub use hash_count::SimHashCount;
pub use plan2explore::{Plan2Explore, Plan2ExploreConfig};
pub use rainbow::{NoisyLinear, dueling_q_values, dueling_q_values_batch};
