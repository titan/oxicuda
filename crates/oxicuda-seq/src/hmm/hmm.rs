//! HMM model definitions and shared helpers.

use crate::error::{SeqError, SeqResult};

/// Discrete-emission Hidden Markov Model.
///
/// * `pi[i]` — initial state probability for state `i`
/// * `a[i*n_states + j]` — transition probability from state `i` to state `j`
/// * `b[i*n_obs + k]` — emission probability of observation `k` from state `i`
#[derive(Debug, Clone)]
pub struct HmmDiscrete {
    pub n_states: usize,
    pub n_obs: usize,
    pub pi: Vec<f64>,
    pub a: Vec<f64>,
    pub b: Vec<f64>,
}

impl HmmDiscrete {
    /// Construct a discrete HMM, validating shapes and that each row of `pi`,
    /// `a`, and `b` sums to approximately 1.
    pub fn new(
        n_states: usize,
        n_obs: usize,
        pi: Vec<f64>,
        a: Vec<f64>,
        b: Vec<f64>,
    ) -> SeqResult<Self> {
        if n_states == 0 || n_obs == 0 {
            return Err(SeqError::InvalidConfiguration(
                "n_states and n_obs must be > 0".to_string(),
            ));
        }
        if pi.len() != n_states {
            return Err(SeqError::ShapeMismatch {
                expected: n_states,
                got: pi.len(),
            });
        }
        if a.len() != n_states * n_states {
            return Err(SeqError::ShapeMismatch {
                expected: n_states * n_states,
                got: a.len(),
            });
        }
        if b.len() != n_states * n_obs {
            return Err(SeqError::ShapeMismatch {
                expected: n_states * n_obs,
                got: b.len(),
            });
        }
        validate_distribution(&pi, "pi", 1)?;
        for i in 0..n_states {
            validate_distribution(&a[i * n_states..(i + 1) * n_states], "A row", i)?;
            validate_distribution(&b[i * n_obs..(i + 1) * n_obs], "B row", i)?;
        }
        Ok(Self {
            n_states,
            n_obs,
            pi,
            a,
            b,
        })
    }

    /// Log emission `log B_j(o_t)`.
    pub fn log_emission(&self, state: usize, obs: usize) -> SeqResult<f64> {
        if state >= self.n_states {
            return Err(SeqError::IndexOutOfBounds {
                index: state,
                len: self.n_states,
            });
        }
        if obs >= self.n_obs {
            return Err(SeqError::InvalidObservation(format!(
                "obs index {obs} >= n_obs {}",
                self.n_obs
            )));
        }
        Ok(log_safe(self.b[state * self.n_obs + obs]))
    }

    /// Log transition `log A_ij`.
    pub fn log_trans(&self, i: usize, j: usize) -> SeqResult<f64> {
        if i >= self.n_states || j >= self.n_states {
            return Err(SeqError::IndexOutOfBounds {
                index: i.max(j),
                len: self.n_states,
            });
        }
        Ok(log_safe(self.a[i * self.n_states + j]))
    }

    /// Log initial probability `log π_i`.
    pub fn log_init(&self, i: usize) -> SeqResult<f64> {
        if i >= self.n_states {
            return Err(SeqError::IndexOutOfBounds {
                index: i,
                len: self.n_states,
            });
        }
        Ok(log_safe(self.pi[i]))
    }
}

/// Gaussian-emission HMM with diagonal covariance per state.
///
/// * `means[i*dim + d]` — mean of dim `d` for state `i`
/// * `vars[i*dim + d]` — variance (diagonal) of dim `d` for state `i`
#[derive(Debug, Clone)]
pub struct HmmGaussian {
    pub n_states: usize,
    pub dim: usize,
    pub pi: Vec<f64>,
    pub a: Vec<f64>,
    pub means: Vec<f64>,
    pub vars: Vec<f64>,
}

impl HmmGaussian {
    /// Construct a Gaussian HMM, validating shapes and probabilistic invariants.
    pub fn new(
        n_states: usize,
        dim: usize,
        pi: Vec<f64>,
        a: Vec<f64>,
        means: Vec<f64>,
        vars: Vec<f64>,
    ) -> SeqResult<Self> {
        if n_states == 0 || dim == 0 {
            return Err(SeqError::InvalidConfiguration(
                "n_states and dim must be > 0".to_string(),
            ));
        }
        if pi.len() != n_states {
            return Err(SeqError::ShapeMismatch {
                expected: n_states,
                got: pi.len(),
            });
        }
        if a.len() != n_states * n_states {
            return Err(SeqError::ShapeMismatch {
                expected: n_states * n_states,
                got: a.len(),
            });
        }
        if means.len() != n_states * dim {
            return Err(SeqError::ShapeMismatch {
                expected: n_states * dim,
                got: means.len(),
            });
        }
        if vars.len() != n_states * dim {
            return Err(SeqError::ShapeMismatch {
                expected: n_states * dim,
                got: vars.len(),
            });
        }
        for &v in &vars {
            if v <= 0.0 || !v.is_finite() {
                return Err(SeqError::InvalidParameter {
                    name: "variance".to_string(),
                    value: v,
                });
            }
        }
        validate_distribution(&pi, "pi", 1)?;
        for i in 0..n_states {
            validate_distribution(&a[i * n_states..(i + 1) * n_states], "A row", i)?;
        }
        Ok(Self {
            n_states,
            dim,
            pi,
            a,
            means,
            vars,
        })
    }

    /// Log Gaussian-emission density `log N(x_t | μ_j, diag(σ²_j))`.
    pub fn log_emission(&self, state: usize, x: &[f64]) -> SeqResult<f64> {
        if state >= self.n_states {
            return Err(SeqError::IndexOutOfBounds {
                index: state,
                len: self.n_states,
            });
        }
        if x.len() != self.dim {
            return Err(SeqError::ShapeMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        let mut ll = 0.0;
        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        for d in 0..self.dim {
            let mu = self.means[state * self.dim + d];
            let var = self.vars[state * self.dim + d];
            let diff = x[d] - mu;
            ll += -0.5 * (log_2pi + var.ln() + diff * diff / var);
        }
        Ok(ll)
    }

    /// Log transition `log A_ij`.
    pub fn log_trans(&self, i: usize, j: usize) -> SeqResult<f64> {
        if i >= self.n_states || j >= self.n_states {
            return Err(SeqError::IndexOutOfBounds {
                index: i.max(j),
                len: self.n_states,
            });
        }
        Ok(log_safe(self.a[i * self.n_states + j]))
    }

    /// Log initial probability `log π_i`.
    pub fn log_init(&self, i: usize) -> SeqResult<f64> {
        if i >= self.n_states {
            return Err(SeqError::IndexOutOfBounds {
                index: i,
                len: self.n_states,
            });
        }
        Ok(log_safe(self.pi[i]))
    }
}

/// Compute `ln(x)` returning `-∞` for `x ≤ 0` instead of `NaN`.
#[inline]
pub fn log_safe(x: f64) -> f64 {
    if x <= 0.0 || !x.is_finite() {
        f64::NEG_INFINITY
    } else {
        x.ln()
    }
}

/// Validate that `p` is a probability vector summing to ~1 (tolerance 1e-6).
fn validate_distribution(p: &[f64], label: &str, idx: usize) -> SeqResult<()> {
    let mut s = 0.0;
    for &v in p {
        if !(0.0..=1.0 + 1e-9).contains(&v) || !v.is_finite() {
            return Err(SeqError::ProbabilityOutOfRange(v));
        }
        s += v;
    }
    if (s - 1.0).abs() > 1e-5 {
        return Err(SeqError::InvalidConfiguration(format!(
            "{label}[{idx}] sums to {s}, expected 1"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discrete_hmm_basic() {
        let hmm = HmmDiscrete::new(
            2,
            3,
            vec![0.6, 0.4],
            vec![0.7, 0.3, 0.4, 0.6],
            vec![0.1, 0.4, 0.5, 0.6, 0.3, 0.1],
        )
        .expect("ok");
        assert_eq!(hmm.n_states, 2);
        let le = hmm.log_emission(0, 2).expect("ok");
        assert!((le - 0.5_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn log_safe_negative_is_neg_inf() {
        assert!(log_safe(0.0).is_infinite());
        assert!(log_safe(-1.0).is_infinite());
        assert!(log_safe(f64::NAN).is_infinite());
        assert!((log_safe(2.0) - 2.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn gaussian_hmm_emission() {
        let hmm = HmmGaussian::new(
            2,
            1,
            vec![0.5, 0.5],
            vec![0.9, 0.1, 0.1, 0.9],
            vec![0.0, 5.0],
            vec![1.0, 1.0],
        )
        .expect("ok");
        let le0 = hmm.log_emission(0, &[0.0]).expect("ok");
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!((le0 - expected).abs() < 1e-12);
    }
}
