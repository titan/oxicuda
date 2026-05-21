//! Contrastive SSL losses: SimCLR (NT-Xent) and MoCo (memory-bank InfoNCE).
//!
//! Both methods minimise an InfoNCE-style cross-entropy where the positive pair
//! is `(z_i, z_i^+)` (e.g. two augmentations of the same image) and the
//! negatives are either the in-batch other samples (SimCLR) or stored embeddings
//! from a momentum-updated memory bank (MoCo).

pub mod info_nce;
pub mod moco;
pub mod moco_v3;
pub mod simclr;

pub use moco_v3::{MocoV3Config, MocoV3State, moco_v3_loss, moco_v3_symmetric_loss};
