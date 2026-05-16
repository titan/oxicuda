//! Line search procedures for smooth optimisation.

pub mod armijo;
pub mod backtracking;
pub mod strong_wolfe;
pub mod wolfe;

pub use armijo::armijo_search;
pub use backtracking::backtracking_search;
pub use strong_wolfe::strong_wolfe_search;
pub use wolfe::wolfe_search;
