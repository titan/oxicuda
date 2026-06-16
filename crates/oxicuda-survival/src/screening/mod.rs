//! Variable screening methods for high-dimensional survival data.
//!
//! Provides Sure Independence Screening (SIS) for Cox PH models and CIF-SIS for
//! competing-risks cumulative-incidence screening.

pub mod cif_sis;
pub mod sis;

pub use cif_sis::{
    CifSisConfig, CifSisResult, cif_screened_dataset, cif_sure_independence_screening,
};
pub use sis::{SisConfig, SisResult, SisTieMethod, screened_dataset, sure_independence_screening};
