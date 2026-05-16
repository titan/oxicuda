//! Permutation test for the difference of two-sample statistics.

use crate::error::{StatsError, StatsResult};
use crate::handle::LcgRng;

/// Result of a permutation test.
#[derive(Debug, Clone, Copy)]
pub struct PermutationResult {
    pub observed_statistic: f64,
    pub p_value_two_sided: f64,
    pub n_permutations: usize,
}

/// Permutation test: shuffles group labels `n_perm` times and computes p-value as
/// fraction of permutations whose |statistic| >= |observed|.
pub fn permutation_test(
    x1: &[f64],
    x2: &[f64],
    n_perm: usize,
    statistic: impl Fn(&[f64], &[f64]) -> StatsResult<f64>,
    rng: &mut LcgRng,
) -> StatsResult<PermutationResult> {
    if x1.is_empty() || x2.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    if n_perm == 0 {
        return Err(StatsError::InvalidParameter {
            name: "n_perm".into(),
            reason: "must be > 0".into(),
        });
    }
    let observed = statistic(x1, x2)?;
    let n1 = x1.len();
    let mut combined = Vec::with_capacity(x1.len() + x2.len());
    combined.extend_from_slice(x1);
    combined.extend_from_slice(x2);
    let n = combined.len();
    let mut a = vec![0.0; n1];
    let mut b = vec![0.0; n - n1];
    let mut at_least = 0usize;
    for _ in 0..n_perm {
        // Fisher-Yates partial shuffle
        let mut indices: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.next_usize(i + 1);
            indices.swap(i, j);
        }
        for i in 0..n1 {
            a[i] = combined[indices[i]];
        }
        for i in 0..(n - n1) {
            b[i] = combined[indices[n1 + i]];
        }
        let s = statistic(&a, &b)?;
        if s.abs() >= observed.abs() - 1e-12 {
            at_least += 1;
        }
    }
    let p = (at_least + 1) as f64 / (n_perm + 1) as f64;
    Ok(PermutationResult {
        observed_statistic: observed,
        p_value_two_sided: p,
        n_permutations: n_perm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptive::summary::mean;

    fn mean_diff(x1: &[f64], x2: &[f64]) -> StatsResult<f64> {
        Ok(mean(x1)? - mean(x2)?)
    }

    #[test]
    fn permutation_test_identical_distributions() {
        let mut rng = LcgRng::new(11);
        let x1 = [1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = [1.5, 2.5, 3.5, 4.5, 5.5];
        let r = permutation_test(&x1, &x2, 300, mean_diff, &mut rng).expect("ok");
        // Small but real shift; permutation p should be > 0
        assert!(r.p_value_two_sided > 0.0);
        assert!(r.p_value_two_sided <= 1.0);
    }

    #[test]
    fn permutation_test_clear_shift() {
        let mut rng = LcgRng::new(23);
        let x1 = [1.0, 2.0, 3.0];
        let x2 = [10.0, 11.0, 12.0];
        let r = permutation_test(&x1, &x2, 200, mean_diff, &mut rng).expect("ok");
        // Strong separation; p should be small (most permutations have smaller |diff|)
        assert!(r.p_value_two_sided < 0.2);
    }
}
