//! Accelerated failure time (AFT) parametric models.

pub mod discrete_time;
pub mod exponential;
pub mod fit_aft;
pub mod generalized_gamma;
pub mod log_logistic;
pub mod log_normal;
pub mod royston_parmar;
pub mod weibull;

pub use discrete_time::{DiscreteTimeConfig, DiscreteTimeFit, DiscreteTimeLink, fit_discrete_time};
pub use exponential::{ExponentialFit, fit_exponential};
pub use fit_aft::{AftFamily, AftFit, fit_aft};
pub use generalized_gamma::{GeneralizedGammaFit, fit_generalized_gamma};
pub use log_logistic::{LogLogisticFit, fit_log_logistic};
pub use log_normal::{LogNormalFit, fit_log_normal};
pub use royston_parmar::{
    RoystonParmarConfig, RoystonParmarFit, fit_royston_parmar, linear_predictor, rcs_basis,
    rcs_deriv,
};
pub use weibull::{WeibullFit, fit_weibull};
