//! Second-Order Cone Programming.

pub mod mehrotra_socp;
pub mod primal_dual_socp;

pub use mehrotra_socp::{MehrotraSocpConfig, MehrotraSocpResult, SocpStatus, mehrotra_socp};
pub use primal_dual_socp::{SocpResult, primal_dual_socp};
