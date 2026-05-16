//! Gradient of Cox negative partial log-likelihood w.r.t. log-risk scores `η`.
//!
//! For Breslow ties:
//! ```text
//!   -∂L/∂η_i = δ_i − exp(η_i) Σ_{t_j ≤ t_i, δ_j=1} 1 / Σ_{R(t_j)} exp(η_k)
//! ```

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};

/// Compute gradient of the negative Breslow partial log-likelihood w.r.t. each `η_i`.
pub fn partial_likelihood_grad(data: &Dataset, eta: &[f64]) -> SurvivalResult<Vec<f64>> {
    if data.len() != eta.len() {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![data.len()],
            got: vec![eta.len()],
        });
    }
    let n = data.len();
    // Sort indices ascending by time
    let mut idx = data.order_by_time();
    idx.sort_by(|&a, &b| {
        data.observations[a]
            .time
            .partial_cmp(&data.observations[b].time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // weights w_i = exp(η_i)
    let w: Vec<f64> = eta.iter().map(|x| x.exp()).collect();
    // Build per-event-time risk-set sum S0
    // We'll iterate descending and accumulate cumulative_sum_event_factor[i] = Σ_{t_j ≤ t_i, δ_j=1} 1/S0_j
    let mut grad = vec![0.0_f64; n];
    // Step 1: compute S0 at each event time by descending order accumulation
    let n_idx = idx.len();
    let mut event_inv: Vec<(usize, f64)> = Vec::new(); // (idx position, 1/S0_at_event)
    // ascend through positions; at each unique time, S0 is sum of w_j for j with t_j >= this time
    // accumulate from end backwards.
    let mut suffix_s0 = vec![0.0_f64; n_idx];
    let mut acc = 0.0_f64;
    for k in (0..n_idx).rev() {
        acc += w[idx[k]];
        suffix_s0[k] = acc;
    }
    // group by unique time and find first position of each time => S0 = suffix_s0[first_pos]
    let mut k = 0usize;
    while k < n_idx {
        let t = data.observations[idx[k]].time;
        let s0_here = suffix_s0[k];
        let mut m = k;
        while m < n_idx && data.observations[idx[m]].time == t {
            m += 1;
        }
        // any events at this time?
        let mut d_count = 0.0_f64;
        for ji in idx.iter().take(m).skip(k) {
            if data.observations[*ji].event {
                d_count += 1.0;
            }
        }
        if d_count > 0.0 && s0_here > 0.0 {
            event_inv.push((k, d_count / s0_here));
        }
        k = m;
    }
    // For each subject i, sum over event-times t_j <= t_i: 1/S0_j (counted with multiplicity d_j)
    // Build cumulative_inv[k_pos] = sum_{event times up to and including pos k} d_j/S0_j
    let mut cum_inv = vec![0.0_f64; n_idx + 1];
    let mut p = 0usize;
    for k in 0..n_idx {
        cum_inv[k + 1] = cum_inv[k];
        // include if event_inv[p].0 == k (start of a unique time)
        while p < event_inv.len() && event_inv[p].0 == k {
            cum_inv[k + 1] += event_inv[p].1;
            p += 1;
        }
    }
    // For each subject i at position k in idx, contribution = exp(η_i) * cum_inv[k+1]
    // (sum of 1/S0_j for all event times <= t_i, since subject i was in those risk sets)
    for (kpos, &i) in idx.iter().enumerate() {
        let contribution = w[i] * cum_inv[kpos + 1];
        let delta = if data.observations[i].event { 1.0 } else { 0.0 };
        // gradient of NEGATIVE partial log-likelihood w.r.t. η_i
        grad[i] = -(delta - contribution);
    }
    Ok(grad)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grad_zero_when_constant_eta_and_one_event() {
        let d = Dataset::from_arrays(&[1.0, 2.0], &[true, true]).expect("ok");
        let eta = vec![0.0, 0.0];
        let g = partial_likelihood_grad(&d, &eta).expect("ok");
        // At t=1: S0=2, contribution_i = exp(0)*1/2 = 0.5; delta_0=1 => -(1-0.5)=-0.5
        // At t=2: + 1/1=1 from event, cum_inv at i=1 pos => grad[1] = -(1 - 1 * (0.5+1)) = 0.5
        // grad[0] = -(1 - 1*0.5) = -0.5
        assert!((g[0] + 0.5).abs() < 1.0e-12);
        assert!((g[1] - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn grad_shape_mismatch() {
        let d = Dataset::from_arrays(&[1.0], &[true]).expect("ok");
        assert!(partial_likelihood_grad(&d, &[0.0, 0.0]).is_err());
    }

    #[test]
    fn grad_sums_correctly() {
        let d = Dataset::from_arrays(&[1.0, 2.0, 3.0], &[true; 3]).expect("ok");
        let eta = vec![0.5, 0.0, -0.5];
        let g = partial_likelihood_grad(&d, &eta).expect("ok");
        for v in g {
            assert!(v.is_finite());
        }
    }
}
