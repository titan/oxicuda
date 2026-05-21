#[allow(clippy::module_inception)]
pub mod cringe;
pub mod dpo;
pub mod ipo;
pub mod kto;
pub mod length_dpo;
pub mod online_dpo;
pub mod sdpo;
pub mod step_dpo;
pub use cringe::{CringeBatch, CringeConfig, CringeLoss, CringeSample};
pub use length_dpo::{LengthDpo, LengthDpoBatch, LengthDpoConfig, LengthPair};
pub use online_dpo::{
    OnlineDpoConfig, PairingMode, build_preference_pair, online_dpo_pairs, online_dpo_step,
};
pub use sdpo::{
    SdpoConfig, StagedDpo, sdpo_stage_loss, sdpo_stage_margin, sdpo_total_loss,
    sdpo_update_reference,
};
pub use step_dpo::{
    StepDpoConfig, StepDpoOutput, StepPair, StepReduceMode, StepWeightScheme, log_sigmoid,
    step_dpo_loss, step_dpo_loss_batch, step_implicit_reward,
};
