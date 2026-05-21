//! Goodness-of-fit tests: KS, Anderson-Darling, Shapiro-Wilk, Jarque-Bera,
//! Cramér-von Mises, Watson.

pub mod anderson_darling;
pub mod cvm_watson;
pub mod jarque_bera;
pub mod ks;
pub mod shapiro_wilk;

pub use anderson_darling::{AndersonDarlingResult, anderson_darling};
pub use cvm_watson::{
    CvmResult, WatsonResult, cvm_test_normal, cvm_test_uniform, cvm_two_sample, watson_test_uniform,
};
pub use jarque_bera::{JarqueBeraResult, jarque_bera};
pub use ks::{KsResult, ks_one_sample, ks_two_sample};
pub use shapiro_wilk::{ShapiroWilkResult, shapiro_wilk};
