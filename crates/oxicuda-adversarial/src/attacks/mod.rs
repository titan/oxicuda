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
pub mod cw;
pub mod fgsm;
pub mod mim;
pub mod pgd;
