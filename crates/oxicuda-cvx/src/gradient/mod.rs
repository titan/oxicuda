//! First-order gradient methods (projected GD, accelerated GD, heavy-ball, L-BFGS, Frank-Wolfe,
//! mirror descent, trust-region Newton with Steihaug-Toint CG, SVRG, SAGA, coordinate descent).

pub mod accelerated_gd;
pub mod averaged_subgradient;
pub mod block_coord_descent;
pub mod coord_descent;
pub mod dual_newton;
pub mod frank_wolfe;
pub mod lbfgs;
pub mod mirror_descent;
pub mod momentum_gd;
pub mod polyak;
pub mod projected_gradient;
pub mod spg;
pub mod svrg_saga;
pub mod trust_region;

pub use accelerated_gd::nesterov_accelerated;
pub use averaged_subgradient::{
    AveragedSubgradConfig, AveragedSubgradResult, SubgradStep, averaged_subgradient,
};
pub use block_coord_descent::{
    BcdConfig, BcdState, BcdSweep, InnerSolver, block_coord_descent, block_coord_descent_quadratic,
};
pub use coord_descent::{CdConfig, CdOrder, CdResult, CoordDescent};
pub use dual_newton::{DualNewtonConfig, DualNewtonState, StepKind, newton_on_dual};
pub use frank_wolfe::{FrankWolfeConfig, FrankWolfeResult, FwStepSize, frank_wolfe};
pub use lbfgs::{LbfgsConfig, LbfgsResult, lbfgs};
pub use mirror_descent::{
    MirrorDescentConfig, MirrorDescentResult, MirrorMap, StepSchedule, mirror_descent, p_norm,
    p_norm_dual_map, project_simplex, safe_log_vec, softmax,
};
pub use momentum_gd::heavy_ball;
pub use polyak::{PolyakConfig, PolyakResult, PolyakTarget, polyak_subgradient};
pub use projected_gradient::projected_gradient;
pub use spg::{Spg, SpgConfig, SpgResult};
pub use svrg_saga::{
    Saga, SagaConfig, Svrg, SvrgConfig, VrsgResult, prox_identity, prox_l1, prox_l2_sq,
};
pub use trust_region::{
    TrustRegionConfig, TrustRegionResult, fd_hess_vec, predicted_reduction, steihaug_cg,
    trust_region_newton,
};
