//! Architecture-based continual learning methods.
//!
//! These methods prevent catastrophic forgetting by modifying the network
//! architecture, either by pruning, masking, or expanding the network.

pub mod bic;
pub mod domain_incremental;
pub mod generative_replay;
pub mod hat;
pub mod icarl;
pub mod multihead;
pub mod packnet;
pub mod piggyback;
pub mod progressive;
pub mod sparse_mask_apply;
pub mod stochastic_binary;

// ─── iCaRL re-exports ────────────────────────────────────────────────────────
pub use icarl::{
    ExemplarSet, IcarlConfig, IcarlState, icarl_classify, icarl_construct_exemplar_set,
    icarl_encode, icarl_fit_task, icarl_new, icarl_update_representation,
};

// ─── HAT re-exports ──────────────────────────────────────────────────────────
pub use hat::{
    HatConfig, HatState, hat_classify, hat_fit_task, hat_forward, hat_new, hat_task_capacity,
};

// ─── Generative Replay (VAE) re-exports ──────────────────────────────────────
pub use generative_replay::{
    VaeReplayConfig, VaeReplayState, vae_replay_fit_task, vae_replay_fit_task_with_cfg,
    vae_replay_new, vae_replay_predict, vae_replay_reconstruct, vae_replay_sample,
};

// ─── Domain-Incremental re-exports ───────────────────────────────────────────
pub use domain_incremental::{
    DomainAdapter, DomainConfig, DomainState, domain_adapter_params, domain_fit_task,
    domain_forward, domain_new, domain_predict,
};

// ─── Multi-Head Class-Incremental re-exports ──────────────────────────────────
pub use multihead::{
    MultiHeadConfig, MultiHeadState, TaskHead, multihead_add_task, multihead_fit_task,
    multihead_n_classes_for_task, multihead_n_tasks, multihead_new, multihead_predict,
    multihead_predict_unknown_task,
};

// ─── Stochastic-Binary Piggyback (STE) re-exports ────────────────────────────
pub use stochastic_binary::{StochasticBinaryConfig, StochasticBinaryState, stable_sigmoid};

// ─── Sparse-Mask Fast Path re-exports ────────────────────────────────────────
pub use sparse_mask_apply::{
    SparseActiveMask, sparse_mask_apply, sparse_mask_backward, sparse_mask_compact,
    sparse_mask_scatter,
};
