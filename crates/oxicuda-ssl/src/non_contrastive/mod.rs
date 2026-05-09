//! Non-contrastive SSL losses: BYOL (cosine target), Barlow Twins
//! (cross-correlation), VICReg (variance + invariance + covariance).
//!
//! These methods avoid explicit negatives and instead use either a target
//! network (BYOL) or a redundancy-reduction objective (Barlow / VICReg).

pub mod barlow;
pub mod byol;
pub mod vicreg;
