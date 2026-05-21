//! Regularization-based continual learning methods.
//!
//! These methods prevent catastrophic forgetting by adding regularization
//! terms to the loss that penalize changes to important parameters.

pub mod clear_replay;
pub mod ewc;
pub mod gradient_compression;
pub mod lwf;
pub mod mas;
pub mod meta_learning;
pub mod mir;
pub mod online_ewc;
pub mod si;

// ─── Online EWC re-exports ───────────────────────────────────────────────────
pub use online_ewc::{
    OnlineEwcConfig, OnlineEwcState, online_ewc_fit_task, online_ewc_new, online_ewc_penalty,
    online_ewc_predict,
};

// ─── MIR re-exports ─────────────────────────────────────────────────────────
pub use mir::{
    MirBuffer, MirConfig, MirState, mir_buffer_size, mir_fit_task, mir_new, mir_predict,
    mir_retrieve,
};

// ─── CLEAR re-exports ────────────────────────────────────────────────────────
pub use clear_replay::{
    ClearConfig, ClearState, clear_buffer_size, clear_encode, clear_fit_task, clear_new,
    clear_predict,
};

// ─── Meta-Learning (OML / ANML) re-exports ───────────────────────────────────
pub use meta_learning::{
    MetaLearningConfig, MetaLearningState, TaskData, oml_adapt, oml_inner_step_count,
    oml_meta_train, oml_meta_train_with_lr, oml_new, oml_predict,
};

// ─── Gradient Compression re-exports ─────────────────────────────────────────
pub use gradient_compression::{
    GradCompConfig, GradCompState, GradMemory, grad_comp_fit_task, grad_comp_n_memories,
    grad_comp_new, grad_comp_predict,
};
