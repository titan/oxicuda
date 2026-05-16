//! Root-finding algorithms for scalar real (and complex polynomial) functions.

pub mod aberth_all_roots;
pub mod bisection;
pub mod brent;
pub mod halley;
pub mod newton;
pub mod secant;

pub use aberth_all_roots::{Complex64, aberth_all_roots};
pub use bisection::bisection;
pub use brent::brent;
pub use halley::halley;
pub use newton::newton;
pub use secant::secant;
