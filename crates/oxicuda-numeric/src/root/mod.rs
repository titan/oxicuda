//! Root-finding algorithms for scalar real (and complex polynomial) functions.

pub mod aberth_all_roots;
pub mod bisection;
pub mod brent;
pub mod complex_newton;
pub mod halley;
pub mod newton;
pub mod residual_refine;
pub mod secant;

pub use aberth_all_roots::{Complex64, aberth_all_roots};
pub use bisection::bisection;
pub use brent::brent;
pub use complex_newton::{
    ComplexRoot, ComplexRootConfig, Cplx, complex_halley, complex_newton, complex_poly_roots,
};
pub use halley::halley;
pub use newton::newton;
pub use residual_refine::{
    Double, RefineConfig, RefineResult, compensated_horner, refine_polynomial_root,
    refine_root_extended, two_prod, two_sum,
};
pub use secant::secant;
