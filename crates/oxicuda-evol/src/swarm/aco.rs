//! Ant Colony Optimization for the Travelling Salesman Problem (TSP).
//!
//! Uses the classic Ant System model (AS) with elitist pheromone updates.
//!
//! Reference: M. Dorigo et al., "Ant System: Optimization by a Colony of Cooperating Agents",
//! IEEE Trans. Systems, Man, and Cybernetics B 26(1):29-41, 1996.

#![allow(clippy::needless_range_loop)]

use crate::{EvolError, EvolResult, handle::LcgRng};

/// ACO hyper-parameters.
#[derive(Debug, Clone)]
pub struct AcoConfig {
    /// Number of cities in the TSP.
    pub n_cities: usize,
    /// Number of ants per iteration.
    pub n_ants: usize,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Pheromone trail importance α.
    pub alpha: f64,
    /// Heuristic information importance β.
    pub beta: f64,
    /// Pheromone evaporation rate ρ ∈ (0, 1).
    pub rho: f64,
    /// Pheromone deposit constant Q.
    pub q: f64,
}

impl AcoConfig {
    /// Build a sensible default config for `n_cities`-city TSP.
    pub fn new(n_cities: usize) -> EvolResult<Self> {
        if n_cities < 2 {
            return Err(EvolError::InvalidParameter(
                "TSP requires at least 2 cities".to_owned(),
            ));
        }
        Ok(Self {
            n_cities,
            n_ants: n_cities,
            max_iter: 100,
            alpha: 1.0,
            beta: 2.0,
            rho: 0.1,
            q: 1.0,
        })
    }
}

/// ACO pheromone and distance state.
pub struct AcoState {
    /// Pheromone matrix τ, n_cities × n_cities (row-major).
    pub pheromone: Vec<f64>,
    /// Distance matrix d, n_cities × n_cities (row-major).
    pub distances: Vec<f64>,
    /// Best tour found so far (indices).
    pub best_tour: Vec<usize>,
    /// Length of best tour.
    pub best_length: f64,
}

impl AcoState {
    /// Initialise from a distance matrix.
    ///
    /// Initial pheromone τ₀ = 1 / (n_cities × greedy_tour_length).
    pub fn new(distances: Vec<f64>, cfg: &AcoConfig, rng: &mut LcgRng) -> EvolResult<Self> {
        let n = cfg.n_cities;
        if distances.len() != n * n {
            return Err(EvolError::PheromoneDimensionMismatch);
        }
        let greedy_len = Self::greedy_tour_length(&distances, n, rng);
        let tau0 = 1.0 / (n as f64 * greedy_len.max(1e-10));
        let pheromone = vec![tau0; n * n];
        Ok(Self {
            pheromone,
            distances,
            best_tour: Vec::new(),
            best_length: f64::INFINITY,
        })
    }

    /// Nearest-neighbour greedy tour for pheromone initialisation.
    fn greedy_tour_length(distances: &[f64], n: usize, rng: &mut LcgRng) -> f64 {
        let start = rng.next_usize(n);
        let mut visited = vec![false; n];
        let mut current = start;
        visited[current] = true;
        let mut length = 0.0;
        for _ in 1..n {
            let mut nearest = usize::MAX;
            let mut min_d = f64::INFINITY;
            for j in 0..n {
                if !visited[j] {
                    let d = distances[current * n + j];
                    if d < min_d {
                        min_d = d;
                        nearest = j;
                    }
                }
            }
            if nearest == usize::MAX {
                break;
            }
            visited[nearest] = true;
            length += min_d;
            current = nearest;
        }
        length += distances[current * n + start];
        length
    }

    /// Construct a tour for one ant using roulette-wheel selection.
    fn construct_tour(&self, cfg: &AcoConfig, rng: &mut LcgRng) -> (Vec<usize>, f64) {
        let n = cfg.n_cities;
        let start = rng.next_usize(n);
        let mut tour = Vec::with_capacity(n);
        let mut visited = vec![false; n];
        let mut current = start;
        visited[current] = true;
        tour.push(current);

        for _ in 1..n {
            // Compute selection probabilities for unvisited cities
            let mut weights: Vec<f64> = (0..n)
                .map(|j| {
                    if visited[j] {
                        0.0
                    } else {
                        let tau = self.pheromone[current * n + j].max(1e-300);
                        let d = self.distances[current * n + j].max(1e-300);
                        let eta = 1.0 / d;
                        tau.powf(cfg.alpha) * eta.powf(cfg.beta)
                    }
                })
                .collect();

            let total: f64 = weights.iter().sum();
            if total <= 0.0 {
                // All cities visited or unreachable; pick first unvisited
                let next = (0..n).find(|&j| !visited[j]);
                if let Some(j) = next {
                    tour.push(j);
                    visited[j] = true;
                    current = j;
                }
                continue;
            }
            // Normalise (for numerical safety)
            for w in &mut weights {
                *w /= total;
            }

            // Roulette selection
            let r = rng.next_f64();
            let mut cumsum = 0.0;
            let mut next_city = n - 1; // fallback
            for (j, &w) in weights.iter().enumerate() {
                cumsum += w;
                if cumsum >= r {
                    next_city = j;
                    break;
                }
            }
            tour.push(next_city);
            visited[next_city] = true;
            current = next_city;
        }

        let length = self.tour_length(&tour, cfg);
        (tour, length)
    }

    /// Compute the total tour length (including return to start).
    fn tour_length(&self, tour: &[usize], cfg: &AcoConfig) -> f64 {
        let n = cfg.n_cities;
        if tour.len() != n {
            return f64::INFINITY;
        }
        let mut len = 0.0;
        for i in 0..n {
            let from = tour[i];
            let to = tour[(i + 1) % n];
            len += self.distances[from * n + to];
        }
        len
    }

    /// Execute one ACO iteration: all ants construct tours, update pheromones.
    ///
    /// Returns the best tour length seen in this iteration.
    pub fn step(&mut self, cfg: &AcoConfig, rng: &mut LcgRng) -> EvolResult<f64> {
        let n = cfg.n_cities;
        let mut iter_best_len = f64::INFINITY;
        let mut iter_best_tour = Vec::new();

        // All ants construct tours
        let mut ant_tours: Vec<(Vec<usize>, f64)> = Vec::with_capacity(cfg.n_ants);
        for _ in 0..cfg.n_ants {
            let (tour, len) = self.construct_tour(cfg, rng);
            if len < iter_best_len {
                iter_best_len = len;
                iter_best_tour = tour.clone();
            }
            ant_tours.push((tour, len));
        }

        // Pheromone evaporation
        for tau in &mut self.pheromone {
            *tau *= 1.0 - cfg.rho;
        }

        // Pheromone deposit: each ant deposits Q / L_k on its tour edges
        for (tour, length) in &ant_tours {
            if length.is_finite() && *length > 0.0 {
                let deposit = cfg.q / length;
                for i in 0..n {
                    let from = tour[i];
                    let to = tour[(i + 1) % n];
                    self.pheromone[from * n + to] += deposit;
                    self.pheromone[to * n + from] += deposit;
                }
            }
        }

        // Update global best
        if iter_best_len < self.best_length {
            self.best_length = iter_best_len;
            self.best_tour = iter_best_tour;
        }

        Ok(iter_best_len)
    }

    /// Run ACO for `cfg.max_iter` iterations.
    pub fn run(&mut self, cfg: &AcoConfig, rng: &mut LcgRng) -> EvolResult<(Vec<usize>, f64)> {
        for _ in 0..cfg.max_iter {
            self.step(cfg, rng)?;
        }
        if self.best_tour.len() != cfg.n_cities {
            return Err(EvolError::TourIncomplete(
                self.best_tour.len(),
                cfg.n_cities,
            ));
        }
        Ok((self.best_tour.clone(), self.best_length))
    }
}
