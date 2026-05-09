use crate::error::{CausalError, CausalResult};

fn clip_propensity(p: f32) -> f32 {
    p.clamp(0.05, 0.95)
}

/// Inverse Probability Weighting — Average Treatment Effect.
/// ATE = mean(Y*T/pi - Y*(1-T)/(1-pi))
pub fn ipw_ate(y: &[f32], t: &[f32], propensity: &[f32]) -> CausalResult<f32> {
    let n = y.len();
    if n == 0 {
        return Err(CausalError::EmptyInput);
    }
    if t.len() != n || propensity.len() != n {
        return Err(CausalError::DimensionMismatch {
            expected: n,
            got: t.len().min(propensity.len()),
        });
    }
    let sum: f32 = y
        .iter()
        .zip(t.iter())
        .zip(propensity.iter())
        .map(|((&yi, &ti), &pi)| {
            let pi = clip_propensity(pi);
            yi * ti / pi - yi * (1.0 - ti) / (1.0 - pi)
        })
        .sum();
    Ok(sum / n as f32)
}

/// Inverse Probability Weighting — Average Treatment Effect on the Treated.
/// ATT = mean(Y*(T - pi*(1-T)/(1-pi))) / mean(T)
pub fn ipw_att(y: &[f32], t: &[f32], propensity: &[f32]) -> CausalResult<f32> {
    let n = y.len();
    if n == 0 {
        return Err(CausalError::EmptyInput);
    }
    if t.len() != n || propensity.len() != n {
        return Err(CausalError::DimensionMismatch {
            expected: n,
            got: t.len().min(propensity.len()),
        });
    }
    let mean_t: f32 = t.iter().sum::<f32>() / n as f32;
    if mean_t < 1e-10 {
        return Err(CausalError::EmptyInput);
    }
    let sum: f32 = y
        .iter()
        .zip(t.iter())
        .zip(propensity.iter())
        .map(|((&yi, &ti), &pi)| {
            let pi = clip_propensity(pi);
            yi * (ti - pi * (1.0 - ti) / (1.0 - pi))
        })
        .sum();
    Ok(sum / (n as f32 * mean_t))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ipw_ate_basic() {
        let y = vec![1.0_f32, 0.0, 1.0, 0.0];
        let t = vec![1.0_f32, 1.0, 0.0, 0.0];
        let pi = vec![0.6_f32, 0.6, 0.4, 0.4];
        let ate = ipw_ate(&y, &t, &pi).unwrap();
        assert!(ate.is_finite());
    }
}
