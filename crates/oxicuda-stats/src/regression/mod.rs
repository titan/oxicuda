//! Linear, logistic, and ridge regression.

pub mod linear;
pub mod logistic;
pub mod ridge_lr;

pub use linear::{LinearModel, matrix_inverse_lu, matrix_mul, matrix_transpose, ols};
pub use logistic::{LogisticModel, logistic_fit_irls};
pub use ridge_lr::{RidgeModel, ridge_regression};
