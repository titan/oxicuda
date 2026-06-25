//! Multi-modal fusion strategies.

pub mod attention_fusion;
pub mod bilinear_fusion;
pub mod concat_fusion;
pub mod film;
pub mod gmu;
pub mod lowrank_fusion;
pub mod mome;
pub mod tensor_fusion;

pub use mome::{FfnExpert, MoMeConfig, MoMeRouter, Modality};
