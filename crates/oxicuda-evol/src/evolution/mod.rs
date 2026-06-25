//! Evolutionary strategy implementations: CMA-ES, Differential Evolution, Natural &
//! OpenAI Evolution Strategies, Simulated Annealing, Memetic, Island Model, and Coevolution.

pub mod cmaes;
pub mod coevolution;
pub mod de;
pub mod island;
pub mod memetic;
pub mod nes;
pub mod openai_es;
pub mod simulated_annealing;

pub use coevolution::{CoevolConfig, CoevolMode, CoevolResult, coevolve};
pub use nes::{NaturalEvolutionStrategies, NesConfig};
pub use openai_es::{EsOptimizer, OpenAiEs, OpenAiEsConfig};
pub use simulated_annealing::{CoolingSchedule, SaConfig, SaState, simulated_annealing};
