//! Traveling Salesperson approximations/heuristics.

pub mod christofides_approx;
pub mod nearest_neighbor;
pub mod two_opt;

pub use christofides_approx::christofides_tour;
pub use nearest_neighbor::nearest_neighbor_tour;
pub use two_opt::two_opt_improve;
