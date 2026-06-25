pub mod bco;
#[allow(clippy::module_inception)]
pub mod cringe;
pub mod dpo;
pub mod dpo_ipo_blend;
pub mod dpo_sft_mix;
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
    BcoConfig, BcoGrad, RewardShift, bco_grad, bco_loss, bco_loss_from_rewards,
    implicit_reward as bco_implicit_reward,
};
pub use cringe::{CringeBatch, CringeConfig, CringeLoss, CringeSample};
pub use dpo_ipo_blend::{
    BlendComponents, BlendGrad, DpoIpoBlendConfig, blend_components_per_pair, blend_grad_per_pair,
    dpo_ipo_blend_grad, dpo_ipo_blend_loss,
};
pub use dpo_sft_mix::{DpoSftMixConfig, MixGrad, MixLoss, dpo_sft_mix_grad, dpo_sft_mix_loss};
pub use dpop::{
    DpopConfig, DpopGrad, dpop_grad, dpop_grad_per_pair, dpop_log_ratio, dpop_loss,
    dpop_loss_per_pair, dpop_penalty,
};
pub use length_dpo::{LengthDpo, LengthDpoBatch, LengthDpoConfig, LengthDpoGrad, LengthPair};
pub use online_dpo::{
    OnlineDpoConfig, OnlineDpoGrad, PairingMode, build_preference_pair, online_dpo_grad,
    online_dpo_pairs, online_dpo_step,
};
pub use rrhf::{
    RrhfConfig, RrhfGrad, RrhfSample, ft_loss, length_normalized_scores, ranking_grad,
    ranking_loss, rrhf_grad, rrhf_loss, rrhf_loss_batch,
};
pub use sdpo::{
    SdpoConfig, SdpoStageGrad, StagedDpo, sdpo_stage_grad, sdpo_stage_loss, sdpo_stage_margin,
    sdpo_total_loss, sdpo_update_reference,
};
pub use slic::{
    SlicConfig, SlicGrad, SlicPair, calibration_loss, regularization_loss, slic_grad,
    slic_grad_batch, slic_loss, slic_loss_batch,
};
pub use step_dpo::{
    StepDpoConfig, StepDpoGrad, StepDpoOutput, StepPair, StepReduceMode, StepWeightScheme,
    log_sigmoid, step_dpo_grad, step_dpo_loss, step_dpo_loss_batch, step_implicit_reward,
};
