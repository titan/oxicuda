//! RL algorithm loss functions.
//!
//! * [`crate::loss::ppo_loss`]              — PPO clip + value + entropy loss
//! * [`crate::loss::dqn_loss`]              — DQN and Double-DQN Bellman error
//! * [`crate::loss::sac_critic_loss`]       — SAC soft Q + policy + temperature loss
//! * [`crate::loss::td3_critic_loss`]       — TD3 actor-critic losses
//! * [`crate::loss::ddpg_critic_loss`]      — DDPG single-critic + deterministic actor losses (Lillicrap 2016)
//! * [`crate::loss::c51_loss`]              — C51 categorical distributional Bellman loss
//! * [`crate::loss::qr_dqn_loss`]           — QR-DQN quantile Huber distributional loss
//! * [`crate::loss::iqn_loss`]              — IQN implicit quantile network loss (Dabney 2018)
//! * [`crate::loss::munchausen_dqn_loss`]   — Munchausen-DQN log-policy augmented loss (Vieillard 2020)
//! * [`crate::loss::DiscreteSacLoss`]       — Discrete SAC loss functions (Christodoulou 2019)
//! * [`crate::loss::SacOffPolicyLoss`]      — SAC structured off-policy loss object
//! * [`crate::loss::Td3PolicyLoss`]         — TD3 structured loss object with target noise
//! * [`crate::loss::cql_loss`]              — Conservative Q-Learning (Kumar 2020)
//! * [`crate::loss::iql_value_loss`]        — Implicit Q-Learning expectile + critic (Kostrikov 2021)
//! * [`crate::loss::awac_actor_loss`]       — Advantage-Weighted Actor-Critic (Nair 2020)
//! * [`crate::loss::bcq_target`]            — Batch-Constrained Q-Learning (Fujimoto 2019)

pub mod c51;
pub mod ddpg;
pub mod discrete_sac;
pub mod dqn;
pub mod iqn;
pub mod munchausen;
pub mod offline;
pub mod ppo;
pub mod qr_dqn;
pub mod sac;
pub mod sac_loss;
pub mod td3;
pub mod td3_loss;

pub use c51::{C51Config, C51Loss, c51_loss, c51_project, c51_support};
pub use ddpg::{DdpgConfig, DdpgCriticLoss, ddpg_actor_loss, ddpg_critic_loss, polyak_update};
pub use discrete_sac::{DiscreteSacConfig, DiscreteSacLoss};
pub use dqn::{DqnConfig, DqnLoss, double_dqn_loss, dqn_loss};
pub use iqn::{IqnConfig, IqnLoss, iqn_cosine_embedding, iqn_loss, iqn_targets, sample_taus};
pub use munchausen::{MunchausenConfig, MunchausenLoss, munchausen_dqn_loss, munchausen_target};
pub use offline::{
    AwacConfig, BcqConfig, CqlConfig, CqlLoss, IqlConfig, advantage_weighted_policy_loss,
    awac_actor_loss, bcq_critic_loss, bcq_target, bcq_vae_loss, cql_loss, expectile_weight,
    iql_critic_loss, iql_value_loss,
};
pub use ppo::{PpoConfig, PpoLoss, ppo_loss};
pub use qr_dqn::{QrDqnConfig, QrDqnLoss, qr_dqn_loss, qr_dqn_quantile_levels, qr_dqn_targets};
pub use sac::{SacConfig, SacLoss, sac_actor_loss, sac_critic_loss, sac_temperature_loss};
pub use sac_loss::{SacOffPolicyConfig, SacOffPolicyLoss};
pub use td3::{Td3Config, Td3Loss, td3_actor_loss, td3_critic_loss};
pub use td3_loss::{Td3PolicyConfig, Td3PolicyLoss};
