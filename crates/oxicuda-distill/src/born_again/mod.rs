//! Born-again and iterative distillation methods.

pub mod ban;
pub mod ban_multigen;
pub mod progressive;
pub mod progressive_kd;
pub mod tas;

pub use ban_multigen::{BanMultiGen, GenerationMetric};
pub use progressive_kd::{ProgressiveKdSchedule, TeacherCheckpoint};
