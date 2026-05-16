//! Gehan-Breslow generalised Wilcoxon test with weight `w(t) = n(t)`.

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};
use crate::linalg::inverse::gauss_jordan_inverse;
use crate::test::log_rank::{LogRankResult, chi_square_survival};

/// Gehan-Breslow test (weighted log-rank with weight equal to the number at risk).
///
/// Emphasises early differences (when more subjects are at risk).
pub fn gehan_breslow_test(data: &Dataset, groups: &[usize]) -> SurvivalResult<LogRankResult> {
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    if groups.len() != data.len() {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![data.len()],
            got: vec![groups.len()],
        });
    }
    let k = groups.iter().copied().max().map(|m| m + 1).unwrap_or(1);
    if k < 2 {
        return Err(SurvivalError::InvalidParameter(
            "need at least 2 groups".to_string(),
        ));
    }
    let mut order = data.order_by_time();
    order.sort_by(|&a, &b| {
        data.observations[a]
            .time
            .partial_cmp(&data.observations[b].time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut at_risk = vec![0.0_f64; k];
    for &i in &order {
        at_risk[groups[i]] += 1.0;
    }
    let mut total = data.len() as f64;
    let mut oe = vec![0.0_f64; k];
    let mut v = vec![0.0_f64; k * k];
    let mut i = 0usize;
    while i < order.len() {
        let t = data.observations[order[i]].time;
        let mut j = i;
        let mut d_total = 0.0_f64;
        let mut d_g = vec![0.0_f64; k];
        while j < order.len() && data.observations[order[j]].time == t {
            let g = groups[order[j]];
            if data.observations[order[j]].event {
                d_total += 1.0;
                d_g[g] += 1.0;
            }
            j += 1;
        }
        let n_total = total;
        let n_g = at_risk.clone();
        let w = n_total; // Gehan weight
        if n_total > 1.0 && d_total > 0.0 {
            for g in 0..k {
                let e = n_g[g] * d_total / n_total;
                oe[g] += w * (d_g[g] - e);
            }
            let common = d_total * (n_total - d_total) / (n_total - 1.0);
            for g in 0..k {
                for h in 0..k {
                    let term = if g == h {
                        n_g[g] * (n_total - n_g[g]) / (n_total * n_total)
                    } else {
                        -n_g[g] * n_g[h] / (n_total * n_total)
                    };
                    v[g * k + h] += w * w * common * term;
                }
            }
        }
        for jj in i..j {
            let g = groups[order[jj]];
            at_risk[g] -= 1.0;
            total -= 1.0;
        }
        i = j;
    }
    let km1 = k - 1;
    let mut v_red = vec![0.0_f64; km1 * km1];
    for r in 0..km1 {
        for c in 0..km1 {
            v_red[r * km1 + c] = v[r * k + c];
        }
    }
    let v_inv = match gauss_jordan_inverse(&v_red, km1) {
        Ok(m) => m,
        Err(_) => return Err(SurvivalError::SingularMatrix),
    };
    let mut chi = 0.0_f64;
    for r in 0..km1 {
        for c in 0..km1 {
            chi += oe[r] * v_inv[r * km1 + c] * oe[c];
        }
    }
    Ok(LogRankResult {
        observed_minus_expected: oe,
        chi_square: chi,
        df: km1,
        p_value: chi_square_survival(chi, km1),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gehan_zero_chi_identical() {
        let times = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let events = vec![true; 6];
        let groups = vec![0usize, 0, 0, 1, 1, 1];
        let d = Dataset::from_arrays(&times, &events).expect("ok");
        let r = gehan_breslow_test(&d, &groups).expect("ok");
        assert!(r.chi_square < 1.0e-9);
    }

    #[test]
    fn gehan_strong_diff() {
        let times = vec![1.0, 1.0, 1.0, 5.0, 5.0, 5.0];
        let events = vec![true; 6];
        let groups = vec![0usize, 0, 0, 1, 1, 1];
        let d = Dataset::from_arrays(&times, &events).expect("ok");
        let r = gehan_breslow_test(&d, &groups).expect("ok");
        assert!(r.chi_square > 1.0);
    }
}
