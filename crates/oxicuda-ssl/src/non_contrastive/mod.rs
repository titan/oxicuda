//! Non-contrastive SSL losses: BYOL (cosine target), Barlow Twins
//! (cross-correlation), VICReg (variance + invariance + covariance),
//! SimSiam (stop-gradient cosine), and MSN (masked siamese networks).
//!
//! These methods avoid explicit negatives and instead use either a target
//! network (BYOL, MSN) or a redundancy-reduction objective (Barlow / VICReg).

pub mod barlow;
pub mod byol;
pub mod dense_cl;
pub mod msn;
pub mod simsiam;
pub mod vicreg;
