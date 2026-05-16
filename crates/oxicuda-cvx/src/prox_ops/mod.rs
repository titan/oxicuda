//! Closed-form proximal operators.
//!
//! For a convex function `g`, the proximal operator is
//! `prox_{λg}(v) = argmin_x { ½||x − v||² + λ g(x) }`.

pub mod elastic_net;
pub mod group_lasso;
pub mod indicator;
pub mod l1;
pub mod l2;
pub mod linf;
pub mod nuclear;
pub mod total_variation_1d;

pub use elastic_net::prox_elastic_net;
pub use group_lasso::prox_group_lasso;
pub use indicator::prox_indicator_box;
pub use l1::{prox_l1, soft_threshold};
pub use l2::prox_l2;
pub use linf::prox_linf;
pub use nuclear::prox_nuclear;
pub use total_variation_1d::prox_tv_1d;
