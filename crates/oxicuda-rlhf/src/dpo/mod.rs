pub mod bco;
#[allow(clippy::module_inception)]
pub mod cringe;
pub mod dpo;
pub mod dpop;
pub mod ipo;
pub mod kto;
pub mod length_dpo;
pub mod online_dpo;
pub mod rrhf;
pub mod sdpo;
pub mod slic;
pub mod step_dpo;
pub use bco::{
    BcoConfig, RewardShift, bco_loss, bco_loss_from_rewards, implicit_reward as bco_implicit_reward,
};
pub use cringe::{CringeBatch, CringeConfig, CringeLoss, CringeSample};
pub use dpop::{DpopConfig, dpop_log_ratio, dpop_loss, dpop_loss_per_pair, dpop_penalty};
pub use length_dpo::{LengthDpo, LengthDpoBatch, LengthDpoConfig, LengthPair};
pub use online_dpo::{
    OnlineDpoConfig, PairingMode, build_preference_pair, online_dpo_pairs, online_dpo_step,
};
pub use rrhf::{
    RrhfConfig, RrhfSample, ft_loss, length_normalized_scores, ranking_loss, rrhf_loss,
    rrhf_loss_batch,
};
pub use sdpo::{
    SdpoConfig, StagedDpo, sdpo_stage_loss, sdpo_stage_margin, sdpo_total_loss,
    sdpo_update_reference,
};
pub use slic::{
    SlicConfig, SlicPair, calibration_loss, regularization_loss, slic_loss, slic_loss_batch,
};
pub use step_dpo::{
    StepDpoConfig, StepDpoOutput, StepPair, StepReduceMode, StepWeightScheme, log_sigmoid,
    step_dpo_loss, step_dpo_loss_batch, step_implicit_reward,
};
