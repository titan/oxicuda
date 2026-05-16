//! Goodness-of-fit tests: KS, Anderson-Darling, Shapiro-Wilk, Jarque-Bera.

pub mod anderson_darling;
pub mod jarque_bera;
pub mod ks;
pub mod shapiro_wilk;

pub use anderson_darling::{AndersonDarlingResult, anderson_darling};
pub use jarque_bera::{JarqueBeraResult, jarque_bera};
pub use ks::{KsResult, ks_one_sample, ks_two_sample};
pub use shapiro_wilk::{ShapiroWilkResult, shapiro_wilk};
