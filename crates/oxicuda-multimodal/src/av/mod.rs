//! Audio-visual self-supervised speech models.
//!
//! Currently provides [`av_hubert`], a pure-Rust AV-HuBERT (Shi et al., 2022)
//! masked-prediction model over fused audio + lip-ROI video streams.

pub mod av_hubert;

pub use av_hubert::{AvHubert, AvHubertConfig, AvHubertWeights, FusedFeatures, ModalityDrop};
