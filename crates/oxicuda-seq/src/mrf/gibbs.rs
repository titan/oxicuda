//! Gibbs sampling on the Ising MRF with optional simulated annealing.

use super::mrf::IsingModel;
use crate::error::{SeqError, SeqResult};
use crate::handle::LcgRng;

/// Gibbs sampling configuration.
#[derive(Debug, Clone, Copy)]
pub struct GibbsConfig {
    /// Number of full sweeps (each site visited once per sweep).
    pub n_sweeps: usize,
    /// Burn-in sweeps to discard before recording statistics.
    pub burn_in: usize,
    /// Optional inverse-temperature schedule: linearly anneal from
    /// `(beta_start, beta_end)` over the sweep budget.
    pub anneal: Option<(f64, f64)>,
}

impl Default for GibbsConfig {
    fn default() -> Self {
        Self {
            n_sweeps: 100,
            burn_in: 20,
            anneal: None,
        }
    }
}

/// Run Gibbs sampling on an Ising model.  Returns the final spin configuration
/// and the post-burn-in mean magnetisation.
pub fn ising_gibbs(
    model: &IsingModel,
    init: &[i32],
    cfg: &GibbsConfig,
    rng: &mut LcgRng,
) -> SeqResult<(Vec<i32>, f64)> {
    let n_sites = model.n_rows * model.n_cols;
    if init.len() != n_sites {
        return Err(SeqError::ShapeMismatch {
            expected: n_sites,
            got: init.len(),
        });
    }
    for &s in init {
        if s != 1 && s != -1 {
            return Err(SeqError::InvalidParameter {
                name: "spin".to_string(),
                value: s as f64,
            });
        }
    }
    let mut spins = init.to_vec();
    let mut mag_sum = 0.0;
    let mut sample_count = 0usize;

    for sweep in 0..cfg.n_sweeps {
        let beta = match cfg.anneal {
            Some((b0, b1)) => {
                let frac = sweep as f64 / cfg.n_sweeps.max(1) as f64;
                b0 + (b1 - b0) * frac
            }
            None => model.beta,
        };
        for r in 0..model.n_rows {
            for c in 0..model.n_cols {
                let mut nb_sum = 0i32;
                if r > 0 {
                    nb_sum += spins[(r - 1) * model.n_cols + c];
                }
                if r + 1 < model.n_rows {
                    nb_sum += spins[(r + 1) * model.n_cols + c];
                }
                if c > 0 {
                    nb_sum += spins[r * model.n_cols + (c - 1)];
                }
                if c + 1 < model.n_cols {
                    nb_sum += spins[r * model.n_cols + (c + 1)];
                }
                let local_field = model.coupling * nb_sum as f64 + model.field;
                let p_up = 1.0 / (1.0 + (-2.0 * beta * local_field).exp());
                let u = rng.next_f64();
                spins[r * model.n_cols + c] = if u < p_up { 1 } else { -1 };
            }
        }
        if sweep >= cfg.burn_in {
            mag_sum += model.magnetisation(&spins)?;
            sample_count += 1;
        }
    }
    let mean_mag = if sample_count == 0 {
        0.0
    } else {
        mag_sum / sample_count as f64
    };
    Ok((spins, mean_mag))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gibbs_low_temp_polarises() {
        // High β + strong coupling + positive field — magnetisation should be near +1.
        let m = IsingModel::new(5, 5, 0.1, 1.0, 2.0).expect("ok");
        let init = vec![1i32; 25];
        let cfg = GibbsConfig {
            n_sweeps: 200,
            burn_in: 50,
            anneal: None,
        };
        let mut rng = LcgRng::new(42);
        let (_, mag) = ising_gibbs(&m, &init, &cfg, &mut rng).expect("ok");
        assert!(mag > 0.5, "magnetisation {mag} should be > 0.5");
    }
}
