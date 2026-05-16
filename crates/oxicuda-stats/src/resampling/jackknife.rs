//! Jackknife (leave-one-out) variance estimator.

use crate::error::{StatsError, StatsResult};

/// Result of the jackknife procedure.
#[derive(Debug, Clone)]
pub struct JackknifeResult {
    pub theta_hat: f64,
    pub pseudovalues: Vec<f64>,
    pub bias: f64,
    pub std_error: f64,
}

/// Jackknife estimator of bias and standard error of a statistic.
pub fn jackknife(
    data: &[f64],
    statistic: impl Fn(&[f64]) -> StatsResult<f64>,
) -> StatsResult<JackknifeResult> {
    let n = data.len();
    if n < 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    let theta_hat = statistic(data)?;
    let mut pseudo = Vec::with_capacity(n);
    let mut buf = Vec::with_capacity(n - 1);
    for i in 0..n {
        buf.clear();
        for (j, &v) in data.iter().enumerate() {
            if j != i {
                buf.push(v);
            }
        }
        pseudo.push(statistic(&buf)?);
    }
    let mean_pseudo: f64 = pseudo.iter().sum::<f64>() / n as f64;
    let bias = (n as f64 - 1.0) * (mean_pseudo - theta_hat);
    let var: f64 = pseudo
        .iter()
        .map(|p| (p - mean_pseudo).powi(2))
        .sum::<f64>()
        * (n as f64 - 1.0)
        / n as f64;
    Ok(JackknifeResult {
        theta_hat,
        pseudovalues: pseudo,
        bias,
        std_error: var.sqrt(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptive::summary::mean;

    #[test]
    fn jackknife_mean_unbiased() {
        let data: Vec<f64> = (1..=10).map(|v| v as f64).collect();
        let r = jackknife(&data, mean).expect("ok");
        // For mean, jackknife bias is zero
        assert!(r.bias.abs() < 1e-10);
    }
}
