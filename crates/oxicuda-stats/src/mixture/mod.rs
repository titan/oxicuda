//! Probabilistic mixture models.
pub mod gmm;
pub use gmm::{
    GmmConfig, GmmCovariance, GmmModel, gmm_aic, gmm_bic, gmm_fit, gmm_predict, gmm_predict_proba,
    gmm_score,
};
