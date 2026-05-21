//! Evolutionary strategy implementations: CMA-ES, Differential Evolution, Memetic, Island Model, and Coevolution.

pub mod cmaes;
pub mod coevolution;
pub mod de;
pub mod island;
pub mod memetic;

pub use coevolution::{CoevolConfig, CoevolMode, CoevolResult, coevolve};
