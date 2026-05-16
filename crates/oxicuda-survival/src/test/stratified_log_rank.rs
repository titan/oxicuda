//! Stratified log-rank test: aggregate O-E and V across strata.

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};
use crate::linalg::inverse::gauss_jordan_inverse;
use crate::test::log_rank::{LogRankResult, chi_square_survival};

/// Stratified log-rank test.
///
/// For each stratum, compute the per-time O-E and hypergeometric V contributions
/// (as in the regular log-rank), then sum across strata before forming the χ².
pub fn stratified_log_rank_test(
    data: &Dataset,
    groups: &[usize],
    strata: &[usize],
) -> SurvivalResult<LogRankResult> {
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    if groups.len() != data.len() {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![data.len()],
            got: vec![groups.len()],
        });
    }
    if strata.len() != data.len() {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![data.len()],
            got: vec![strata.len()],
        });
    }
    let k = groups.iter().copied().max().map(|m| m + 1).unwrap_or(1);
    if k < 2 {
        return Err(SurvivalError::InvalidParameter(
            "need at least 2 groups".to_string(),
        ));
    }
    let n_strata = strata.iter().copied().max().map(|m| m + 1).unwrap_or(1);
    let mut oe = vec![0.0_f64; k];
    let mut v = vec![0.0_f64; k * k];
    for stratum in 0..n_strata {
        // collect indices belonging to this stratum
        let mut idx: Vec<usize> = (0..data.len()).filter(|&i| strata[i] == stratum).collect();
        if idx.is_empty() {
            continue;
        }
        idx.sort_by(|&a, &b| {
            data.observations[a]
                .time
                .partial_cmp(&data.observations[b].time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut at_risk: Vec<f64> = vec![0.0_f64; k];
        for &i in &idx {
            at_risk[groups[i]] += 1.0;
        }
        let mut total_at_risk = idx.len() as f64;
        let mut p = 0usize;
        while p < idx.len() {
            let t = data.observations[idx[p]].time;
            let mut q = p;
            let mut d_total = 0.0_f64;
            let mut d_g: Vec<f64> = vec![0.0_f64; k];
            while q < idx.len() && data.observations[idx[q]].time == t {
                let g = groups[idx[q]];
                if data.observations[idx[q]].event {
                    d_total += 1.0;
                    d_g[g] += 1.0;
                }
                q += 1;
            }
            let n_g: Vec<f64> = at_risk.clone();
            let n_total = total_at_risk;
            if n_total > 1.0 && d_total > 0.0 {
                for g in 0..k {
                    let e = n_g[g] * d_total / n_total;
                    oe[g] += d_g[g] - e;
                }
                let common = d_total * (n_total - d_total) / (n_total - 1.0);
                for g in 0..k {
                    for h in 0..k {
                        let term = if g == h {
                            n_g[g] * (n_total - n_g[g]) / (n_total * n_total)
                        } else {
                            -n_g[g] * n_g[h] / (n_total * n_total)
                        };
                        v[g * k + h] += common * term;
                    }
                }
            }
            for jj in p..q {
                let g = groups[idx[jj]];
                at_risk[g] -= 1.0;
                total_at_risk -= 1.0;
            }
            p = q;
        }
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
    fn stratified_identical_groups_zero_chi() {
        let times = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let events = vec![true; 6];
        let groups = vec![0usize, 0, 0, 1, 1, 1];
        let strata = vec![0usize; 6];
        let d = Dataset::from_arrays(&times, &events).expect("ok");
        let r = stratified_log_rank_test(&d, &groups, &strata).expect("ok");
        assert!(r.chi_square < 1.0e-9);
    }

    #[test]
    fn stratified_two_strata() {
        let times = vec![1.0, 1.0, 5.0, 5.0, 2.0, 2.0, 8.0, 8.0];
        let events = vec![true; 8];
        let groups = vec![0usize, 1, 0, 1, 0, 1, 0, 1];
        let strata = vec![0usize, 0, 0, 0, 1, 1, 1, 1];
        let d = Dataset::from_arrays(&times, &events).expect("ok");
        let r = stratified_log_rank_test(&d, &groups, &strata).expect("ok");
        assert_eq!(r.df, 1);
        assert!(r.chi_square >= 0.0);
    }

    #[test]
    fn stratified_rejects_size_mismatch() {
        let d = Dataset::from_arrays(&[1.0, 2.0], &[true, true]).expect("ok");
        assert!(stratified_log_rank_test(&d, &[0, 1], &[0]).is_err());
    }
}
