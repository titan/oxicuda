//! Non-parametric (rank-based) tests.

pub mod friedman;
pub mod kruskal_wallis;
pub mod mann_whitney;
pub mod wilcoxon;

pub use friedman::{FriedmanResult, friedman};
pub use kruskal_wallis::{KruskalWallisResult, kruskal_wallis};
pub use mann_whitney::{MannWhitneyResult, mann_whitney_u};
pub use wilcoxon::{WilcoxonResult, wilcoxon_signed_rank};

/// Internal helper: compute mid-ranks (average rank for ties) given a slice of values.
/// Returns a vector parallel to `x` containing ranks in [1, n].
pub(crate) fn rank_with_ties(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && (x[order[j + 1]] - x[order[i]]).abs() < 1e-15 {
            j += 1;
        }
        // mid-rank for positions [i, j] = ((i+1) + (j+1)) / 2
        let mid = (i + j + 2) as f64 / 2.0;
        for k in i..=j {
            ranks[order[k]] = mid;
        }
        i = j + 1;
    }
    ranks
}

/// Sum of tie-correction contributions: `sum_t (t^3 - t)` over all tie groups.
pub(crate) fn tie_correction_sum(ranks: &[f64]) -> f64 {
    let n = ranks.len();
    let mut sorted = ranks.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut total = 0.0;
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && (sorted[j + 1] - sorted[i]).abs() < 1e-12 {
            j += 1;
        }
        let t = (j - i + 1) as f64;
        if t > 1.0 {
            total += t * t * t - t;
        }
        i = j + 1;
    }
    total
}
