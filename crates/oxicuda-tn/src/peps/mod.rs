//! Projected Entangled Pair States (PEPS) — 2D and 3D tensor networks.

pub mod contraction;
pub mod ctmrg;
pub mod full_update;
pub mod peps;
pub mod peps_3d;
pub mod simple_update;

pub use contraction::boundary_mps_contraction;
pub use ctmrg::{
    CtmrgConfig, CtmrgEnv, CtmrgResult, ctmrg_expectation, ctmrg_init, ctmrg_norm_per_site,
    ctmrg_run, ctmrg_step_down, ctmrg_step_right,
};
pub use full_update::{
    FullUpdateConfig, FullUpdateResult, full_update_energy, full_update_init, full_update_run,
    full_update_step_h, full_update_step_v,
};
pub use peps::{Peps, PepsTensor};
pub use peps_3d::{
    Peps3d, Peps3dTensor, Site3d, peps3d_bond_dimension, peps3d_entanglement_entropy_z,
    peps3d_local_expectation, peps3d_n_sites, peps3d_new, peps3d_norm_approx, peps3d_product_state,
    peps3d_random,
};
pub use simple_update::{
    PepsLambdas, PepsTensor as SuPepsTensor, SimpleUpdateConfig, SimpleUpdateResult,
    heisenberg_hamiltonian_2site, mat_exp_sym, simple_update_energy, simple_update_init,
    simple_update_run, simple_update_step_h, simple_update_step_v,
};
