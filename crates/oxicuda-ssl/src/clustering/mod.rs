//! Clustering SSL losses: SwAV (Sinkhorn-Knopp normalised assignments),
//! DINO (centred + sharpened student-teacher cross-entropy),
//! iBOT (masked image modeling with online tokenizer), and
//! DeepCluster / DeeperCluster (k-means pseudo-label clustering).

pub mod deep_cluster;
pub mod dino;
pub mod dino_v2;
pub mod ibot;
pub mod swav;

pub use dino_v2::{DinoV2, DinoV2Config};
