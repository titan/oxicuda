//! Mean-field variational inference for grid CRFs.

use super::grid_crf::GridCrf;
use crate::error::{SeqError, SeqResult};

/// Mean-field inference configuration.
#[derive(Debug, Clone, Copy)]
pub struct MeanFieldConfig {
    /// Number of mean-field sweeps.
    pub max_iter: usize,
    /// Convergence threshold on max change of q.
    pub tol: f64,
    /// Damping factor `q_new = (1−damp) q_old + damp q_update`.
    pub damping: f64,
}

impl Default for MeanFieldConfig {
    fn default() -> Self {
        Self {
            max_iter: 30,
            tol: 1e-4,
            damping: 0.5,
        }
    }
}

/// Run mean-field variational inference; returns the posterior `q` of shape
/// `(n_rows × n_cols × n_labels)`.
pub fn mean_field_inference(crf: &GridCrf, cfg: &MeanFieldConfig) -> SeqResult<Vec<f64>> {
    if cfg.max_iter == 0 {
        return Err(SeqError::InvalidConfiguration(
            "max_iter must be > 0".to_string(),
        ));
    }
    let r_max = crf.n_rows;
    let c_max = crf.n_cols;
    let l_max = crf.n_labels;
    let n_sites = r_max * c_max;

    // Initialise q with softmin of unary energies.
    let mut q = vec![0.0f64; n_sites * l_max];
    for r in 0..r_max {
        for c in 0..c_max {
            let base = (r * c_max + c) * l_max;
            let mut neg = vec![0.0f64; l_max];
            for l in 0..l_max {
                neg[l] = -crf.unary[base + l];
            }
            let m = neg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut s = 0.0;
            for l in 0..l_max {
                neg[l] = (neg[l] - m).exp();
                s += neg[l];
            }
            for l in 0..l_max {
                q[base + l] = if s > 0.0 {
                    neg[l] / s
                } else {
                    1.0 / l_max as f64
                };
            }
        }
    }

    let mut q_new = q.clone();
    for _it in 0..cfg.max_iter {
        let mut max_change = 0.0_f64;
        for r in 0..r_max {
            for c in 0..c_max {
                let base = (r * c_max + c) * l_max;
                // For each label l at site (r,c), compute log_q_new = -φ - Σ_nbrs ψ-expectation
                let mut log_q = vec![0.0f64; l_max];
                for l in 0..l_max {
                    let mut acc = -crf.unary[base + l];
                    if r > 0 {
                        let nbr = ((r - 1) * c_max + c) * l_max;
                        for lp in 0..l_max {
                            acc -= q[nbr + lp] * crf.pairwise[l * l_max + lp];
                        }
                    }
                    if r + 1 < r_max {
                        let nbr = ((r + 1) * c_max + c) * l_max;
                        for lp in 0..l_max {
                            acc -= q[nbr + lp] * crf.pairwise[l * l_max + lp];
                        }
                    }
                    if c > 0 {
                        let nbr = (r * c_max + (c - 1)) * l_max;
                        for lp in 0..l_max {
                            acc -= q[nbr + lp] * crf.pairwise[l * l_max + lp];
                        }
                    }
                    if c + 1 < c_max {
                        let nbr = (r * c_max + (c + 1)) * l_max;
                        for lp in 0..l_max {
                            acc -= q[nbr + lp] * crf.pairwise[l * l_max + lp];
                        }
                    }
                    log_q[l] = acc;
                }
                let m = log_q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0;
                for l in 0..l_max {
                    log_q[l] = (log_q[l] - m).exp();
                    sum += log_q[l];
                }
                for l in 0..l_max {
                    let val = if sum > 0.0 {
                        log_q[l] / sum
                    } else {
                        1.0 / l_max as f64
                    };
                    let damped = (1.0 - cfg.damping) * q[base + l] + cfg.damping * val;
                    let change = (damped - q[base + l]).abs();
                    if change > max_change {
                        max_change = change;
                    }
                    q_new[base + l] = damped;
                }
            }
        }
        q.copy_from_slice(&q_new);
        if max_change < cfg.tol {
            break;
        }
    }
    Ok(q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_field_runs_two_labels() {
        let n_rows = 3;
        let n_cols = 3;
        let n_labels = 2;
        // Unary prefers label 0 at corner (0,0) and label 1 at corner (2,2).
        let mut unary = vec![0.5f64; n_rows * n_cols * n_labels];
        unary[0] = 0.0; // label 0 at (0,0)
        unary[1] = 1.0; // label 1 at (0,0)
        let idx22 = (2 * n_cols + 2) * n_labels;
        unary[idx22] = 1.0;
        unary[idx22 + 1] = 0.0;
        // Pairwise smoothing: same-label = 0, different-label = 0.5
        let pairwise = vec![0.0, 0.5, 0.5, 0.0];
        let g = GridCrf::new(n_rows, n_cols, n_labels, unary, pairwise).expect("ok");
        let q = mean_field_inference(&g, &MeanFieldConfig::default()).expect("ok");
        // q rows sum to 1
        for r in 0..n_rows {
            for c in 0..n_cols {
                let s: f64 = q[(r * n_cols + c) * n_labels..(r * n_cols + c + 1) * n_labels]
                    .iter()
                    .sum();
                assert!((s - 1.0).abs() < 1e-6, "row sum {s}");
            }
        }
    }
}
