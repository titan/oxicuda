//! Fundamental `Individual` type: a real-valued genome + fitness score.

/// A single candidate solution: a vector of real-valued genes and an associated fitness.
///
/// Fitness is initialised to `f64::INFINITY` (worst possible for minimisation problems).
#[derive(Debug, Clone)]
pub struct Individual {
    /// Real-valued genome (decision variables).
    pub genome: Vec<f64>,
    /// Current fitness value (lower = better for minimisation).
    pub fitness: f64,
}

impl Individual {
    /// Create an unevaluated `Individual` with the given genome.
    pub fn new(genome: Vec<f64>) -> Self {
        Self {
            genome,
            fitness: f64::INFINITY,
        }
    }

    /// Evaluate this individual in-place using the provided objective function.
    pub fn evaluate<F: Fn(&[f64]) -> f64>(&mut self, f: F) {
        self.fitness = f(&self.genome);
    }
}
