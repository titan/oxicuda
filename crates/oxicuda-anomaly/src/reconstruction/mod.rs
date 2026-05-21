//! Reconstruction-based anomaly detection (Autoencoder, VAE, PCA reconstruction, DAGMM,
//! Memory-Augmented Autoencoder, Self-Supervised, AnoGAN / f-AnoGAN, Diffusion).
pub mod anogan;
pub mod autoencoder;
pub mod dagmm;
pub mod diffusion_anomaly;
pub mod mem_ae;
pub mod pca_anomaly;
pub mod self_supervised;
pub mod vae_anomaly;

pub use anogan::{
    AnoganConfig, AnoganFit, anogan_fit, anogan_generate, anogan_predict, anogan_score,
};
pub use diffusion_anomaly::{
    DiffusionAnomalyConfig, DiffusionAnomalyFit, diffusion_anomaly_fit, diffusion_anomaly_predict,
    diffusion_anomaly_score,
};
