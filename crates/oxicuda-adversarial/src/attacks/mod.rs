//! Adversarial attack methods.
//!
//! All attacks accept a black-box loss-gradient closure
//! `loss_grad: impl Fn(&[f32]) -> AdvResult<Vec<f32>>` plus the original
//! input, an Lp norm budget ε, and step size α. They return the adversarial
//! example with the same shape as the input.
//!
//! Box constraints (e.g. `[0, 1]` for normalized images) are honoured via the
//! `lo` / `hi` clamp arguments.

pub mod auto_pgd;
pub mod autoattack;
pub mod cw;
pub mod deepfool;
pub mod fgsm;
pub mod jsma;
pub mod mim;
pub mod patch;
pub mod pgd;
pub mod square;
pub mod uap;

pub use autoattack::{AutoAttackConfig, autoattack, dlr_loss};
pub use deepfool::{DeepFoolConfig, DeepFoolResult, deepfool};
pub use jsma::{Jsma, JsmaConfig};
pub use patch::{PatchAttack, PatchConfig};
pub use square::{SquareAttackConfig, square_attack};
pub use uap::{UapConfig, UapResult, uap_attack};
