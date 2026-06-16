//! Non-metric (ordinal) Multidimensional Scaling (Kruskal 1964).
//!
//! Non-metric MDS seeks a low-dimensional configuration whose inter-point Euclidean
//! distances respect only the *rank order* of the supplied dissimilarities. It fits,
//! at every iteration, a monotone-increasing transform of the dissimilarities (the
//! "disparities") by isotonic regression, then moves the configuration toward those
//! disparities via the SMACOF Guttman transform.
//!
//! Algorithm:
//! 1. Initialise the configuration with classical (metric) MDS.
//! 2. Repeat:
//!    a. Compute the current Euclidean distances `d_ij`.
//!    b. Order the off-diagonal pairs `(i < j)` by ascending dissimilarity and run
//!    PAVA isotonic regression to fit disparities `dhat_ij` that are monotone
//!    non-decreasing in the dissimilarity and minimise `sum (d_ij - dhat_ij)^2`.
//!    c. Compute Kruskal's Stress-1 `sqrt( sum (d_ij - dhat_ij)^2 / sum d_ij^2 )`.
//!    d. Update the configuration with the Guttman transform toward the disparities:
//!    `X <- (1/n) B(dhat) X`.
//!    e. Stop when the stress change drops below `tol`.

use crate::error::{ManifoldError, ManifoldResult};
use crate::mds::classical_mds::classical_mds;

/// Result of a non-metric MDS fit.
pub struct NonmetricMdsResult {
    /// Embedded coordinates (`n x n_components`, row-major).
    pub embedding: Vec<f64>,
    /// Final Kruskal Stress-1.
    pub stress: f64,
    /// Number of iterations performed.
    pub n_iter: usize,
}

/// Pool-Adjacent-Violators Algorithm (PAVA) isotonic regression with unit weights.
///
/// Given values `y` already ordered by the independent variable, returns the
/// monotone non-decreasing fit `yhat` minimising `sum (y_i - yhat_i)^2`. The fit is
/// piecewise-constant: adjacent blocks that violate monotonicity are pooled and
/// replaced by their mean until the whole sequence is non-decreasing.
pub fn pava(y: &[f64]) -> Vec<f64> {
    let n = y.len();
    if n == 0 {
        return Vec::new();
    }
    // Each block stores (sum, count). The block value is sum / count.
    let mut block_sum: Vec<f64> = Vec::with_capacity(n);
    let mut block_cnt: Vec<usize> = Vec::with_capacity(n);
    for &v in y {
        let mut s = v;
        let mut c = 1usize;
        // Merge with the previous block while it would violate monotonicity.
        while let (Some(&ps), Some(&pc)) = (block_sum.last(), block_cnt.last()) {
            if ps / pc as f64 > s / c as f64 {
                s += ps;
                c += pc;
                block_sum.pop();
                block_cnt.pop();
            } else {
                break;
            }
        }
        block_sum.push(s);
        block_cnt.push(c);
    }
    // Expand blocks back to per-element values.
    let mut out = Vec::with_capacity(n);
    for (s, c) in block_sum.iter().zip(block_cnt.iter()) {
        let val = s / *c as f64;
        for _ in 0..*c {
            out.push(val);
        }
    }
    out
}

/// Fit non-metric MDS on an `n x n` symmetric dissimilarity matrix (row-major).
pub fn nonmetric_mds(
    dissimilarities: &[f64],
    n: usize,
    n_components: usize,
    max_iter: usize,
    tol: f64,
) -> ManifoldResult<NonmetricMdsResult> {
    if n == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if dissimilarities.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![dissimilarities.len()],
        });
    }
    if n_components == 0 || n_components >= n {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: format!("must be in 1..{n}, got {n_components}"),
        });
    }
    let dim = n_components;

    // Enumerate the off-diagonal pairs (i < j) and sort by ascending dissimilarity.
    // `pair_a`/`pair_b` give the endpoints; `order` is the permutation that sorts the
    // pairs by dissimilarity, used to apply PAVA in the correct order.
    let num_pairs = n * (n - 1) / 2;
    let mut pair_a = Vec::with_capacity(num_pairs);
    let mut pair_b = Vec::with_capacity(num_pairs);
    let mut delta = Vec::with_capacity(num_pairs);
    for i in 0..n {
        for j in (i + 1)..n {
            pair_a.push(i);
            pair_b.push(j);
            delta.push(dissimilarities[i * n + j]);
        }
    }
    let mut order: Vec<usize> = (0..num_pairs).collect();
    order.sort_by(|&p, &q| {
        delta[p]
            .partial_cmp(&delta[q])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Initialise the configuration with classical MDS on the dissimilarities.
    let mut x = match classical_mds(dissimilarities, n, dim) {
        Ok(r) if r.embedding.iter().all(|v| v.is_finite()) => r.embedding,
        _ => {
            // Degenerate dissimilarities: fall back to a deterministic spread.
            let mut x = vec![0.0; n * dim];
            for (idx, v) in x.iter_mut().enumerate() {
                // A simple reproducible pattern; configuration will be refined by SMACOF.
                let t = (idx as f64 * 0.6180339887498949).fract();
                *v = t - 0.5;
            }
            x
        }
    };

    // Scratch buffers reused across iterations.
    let mut dist = vec![0.0f64; num_pairs]; // current distances per pair
    let mut ordered_dist = vec![0.0f64; num_pairs]; // distances in dissimilarity order
    let mut dhat_full = vec![0.0f64; num_pairs]; // disparities per pair (original order)

    let mut prev_stress = f64::INFINITY;
    let mut stress = f64::INFINITY;
    let mut iter_count = 0;

    for it in 0..max_iter {
        iter_count = it + 1;

        // (a) Current Euclidean distances for every pair.
        for p in 0..num_pairs {
            let i = pair_a[p];
            let j = pair_b[p];
            let mut s = 0.0;
            for c in 0..dim {
                let v = x[i * dim + c] - x[j * dim + c];
                s += v * v;
            }
            dist[p] = s.sqrt();
        }

        // (b) Isotonic regression of the distances against the dissimilarity order.
        for (slot, &p) in order.iter().enumerate() {
            ordered_dist[slot] = dist[p];
        }
        // PAVA handles ties via the sorted order; tied dissimilarities form adjacent
        // entries and are pooled together where monotonicity demands.
        let dhat_ordered = pava(&ordered_dist);
        for (slot, &p) in order.iter().enumerate() {
            dhat_full[p] = dhat_ordered[slot];
        }

        // (c) Kruskal Stress-1.
        let mut num = 0.0;
        let mut den = 0.0;
        for p in 0..num_pairs {
            let diff = dist[p] - dhat_full[p];
            num += diff * diff;
            den += dist[p] * dist[p];
        }
        stress = if den > 1e-30 { (num / den).sqrt() } else { 0.0 };

        // (d) Guttman transform toward the disparities dhat.
        x = guttman_step_disparities(&x, &dhat_full, &pair_a, &pair_b, n, dim);

        // (e) Convergence on the stress change.
        if (prev_stress - stress).abs() < tol {
            break;
        }
        prev_stress = stress;
    }

    // Recompute the final stress for the returned configuration so that `stress`
    // matches `embedding` exactly (the last Guttman step moved the configuration).
    let final_stress = final_stress(&x, &dissimilarities_order(&order), &pair_a, &pair_b, n, dim);

    Ok(NonmetricMdsResult {
        embedding: x,
        stress: final_stress.unwrap_or(stress),
        n_iter: iter_count,
    })
}

/// Helper that returns the `order` permutation unchanged; kept for readability at the
/// call-site where the final stress is recomputed.
fn dissimilarities_order(order: &[usize]) -> Vec<usize> {
    order.to_vec()
}

/// Recompute Kruskal Stress-1 for a configuration by re-fitting disparities.
fn final_stress(
    x: &[f64],
    order: &[usize],
    pair_a: &[usize],
    pair_b: &[usize],
    n: usize,
    dim: usize,
) -> Option<f64> {
    let num_pairs = n * (n - 1) / 2;
    if order.len() != num_pairs {
        return None;
    }
    let mut dist = vec![0.0f64; num_pairs];
    for p in 0..num_pairs {
        let i = pair_a[p];
        let j = pair_b[p];
        let mut s = 0.0;
        for c in 0..dim {
            let v = x[i * dim + c] - x[j * dim + c];
            s += v * v;
        }
        dist[p] = s.sqrt();
    }
    let mut ordered_dist = vec![0.0f64; num_pairs];
    for (slot, &p) in order.iter().enumerate() {
        ordered_dist[slot] = dist[p];
    }
    let dhat_ordered = pava(&ordered_dist);
    let mut num = 0.0;
    let mut den = 0.0;
    for (slot, &p) in order.iter().enumerate() {
        let diff = dist[p] - dhat_ordered[slot];
        num += diff * diff;
        den += dist[p] * dist[p];
    }
    Some(if den > 1e-30 { (num / den).sqrt() } else { 0.0 })
}

/// SMACOF Guttman transform toward target disparities.
///
/// Builds the SMACOF B-matrix from the disparities `dhat` (one per off-diagonal pair)
/// and applies the update `X <- (1/n) B X`, then re-centres the configuration. This is
/// the same Guttman step used by metric SMACOF, with the raw dissimilarities replaced
/// by the isotonic disparities.
fn guttman_step_disparities(
    x: &[f64],
    dhat: &[f64],
    pair_a: &[usize],
    pair_b: &[usize],
    n: usize,
    dim: usize,
) -> Vec<f64> {
    // B is n x n: off-diagonal b_ij = -dhat_ij / d_ij (0 if d_ij ~ 0); diagonal is the
    // negated row sum so that B 1 = 0.
    let mut b = vec![0.0f64; n * n];
    for (p, &target) in dhat.iter().enumerate() {
        let i = pair_a[p];
        let j = pair_b[p];
        let mut dij = 0.0;
        for c in 0..dim {
            let v = x[i * dim + c] - x[j * dim + c];
            dij += v * v;
        }
        let dij = dij.sqrt();
        if dij > 1e-12 {
            let val = -target / dij;
            b[i * n + j] = val;
            b[j * n + i] = val;
        }
    }
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            if j != i {
                row_sum += b[i * n + j];
            }
        }
        b[i * n + i] = -row_sum;
    }
    // X_new = (1/n) B X.
    let mut x_new = vec![0.0f64; n * dim];
    for i in 0..n {
        for c in 0..dim {
            let mut acc = 0.0;
            for j in 0..n {
                acc += b[i * n + j] * x[j * dim + c];
            }
            x_new[i * dim + c] = acc / n as f64;
        }
    }
    // Centre each coordinate.
    for c in 0..dim {
        let mut mean = 0.0;
        for i in 0..n {
            mean += x_new[i * dim + c];
        }
        mean /= n as f64;
        for i in 0..n {
            x_new[i * dim + c] -= mean;
        }
    }
    x_new
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Spearman rank correlation between two equal-length slices.
    fn spearman(a: &[f64], b: &[f64]) -> f64 {
        fn ranks(v: &[f64]) -> Vec<f64> {
            let n = v.len();
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap_or(std::cmp::Ordering::Equal));
            let mut r = vec![0.0; n];
            let mut i = 0;
            while i < n {
                let mut j = i + 1;
                while j < n && (v[idx[j]] - v[idx[i]]).abs() < 1e-12 {
                    j += 1;
                }
                // Average rank for ties (1-based).
                let avg = ((i + 1 + j) as f64) / 2.0;
                for &k in &idx[i..j] {
                    r[k] = avg;
                }
                i = j;
            }
            r
        }
        let ra = ranks(a);
        let rb = ranks(b);
        let n = ra.len() as f64;
        let ma = ra.iter().sum::<f64>() / n;
        let mb = rb.iter().sum::<f64>() / n;
        let mut cov = 0.0;
        let mut va = 0.0;
        let mut vb = 0.0;
        for (x, y) in ra.iter().zip(&rb) {
            let dx = x - ma;
            let dy = y - mb;
            cov += dx * dy;
            va += dx * dx;
            vb += dy * dy;
        }
        if va <= 1e-30 || vb <= 1e-30 {
            return 0.0;
        }
        cov / (va.sqrt() * vb.sqrt())
    }

    fn euclidean_matrix(pts: &[f64], n: usize, dim: usize) -> Vec<f64> {
        let mut d = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for c in 0..dim {
                    let v = pts[i * dim + c] - pts[j * dim + c];
                    s += v * v;
                }
                d[i * n + j] = s.sqrt();
            }
        }
        d
    }

    /// (a) PAVA correctness on a known sequence and idempotency.
    #[test]
    fn pava_known_sequence() {
        let y = vec![1.0, 3.0, 2.0, 4.0];
        let fit = pava(&y);
        let expected = [1.0, 2.5, 2.5, 4.0];
        for (f, e) in fit.iter().zip(expected.iter()) {
            assert!((f - e).abs() < 1e-12, "got {f}, expected {e}");
        }
        // Monotone non-decreasing.
        for w in fit.windows(2) {
            assert!(w[1] >= w[0] - 1e-12);
        }
        // Idempotent on an already-monotone input.
        let mono = vec![0.0, 1.0, 1.0, 2.5, 4.0];
        let fit2 = pava(&mono);
        for (a, b) in fit2.iter().zip(mono.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
        // Idempotency of PAVA itself: pava(pava(y)) == pava(y).
        let twice = pava(&fit);
        for (a, b) in twice.iter().zip(fit.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    /// PAVA on a strictly decreasing input collapses to the global mean.
    #[test]
    fn pava_decreasing_collapses_to_mean() {
        let y = vec![4.0, 3.0, 2.0, 1.0];
        let fit = pava(&y);
        for f in &fit {
            assert!((f - 2.5).abs() < 1e-12);
        }
    }

    /// (b) Recovery: a monotone-nonlinear warp of true Euclidean distances is undone,
    /// so the recovered configuration's distances rank-correlate ~1 with the originals
    /// and the final stress is small.
    #[test]
    fn nonmetric_recovers_monotone_warp() {
        let mut rng = LcgRng::new(2024);
        let n = 30;
        let dim = 2;
        let mut pts = vec![0.0; n * dim];
        for v in &mut pts {
            *v = rng.next_range(-2.0, 2.0);
        }
        let true_d = euclidean_matrix(&pts, n, dim);
        // Monotone nonlinear transform Delta = d^1.7.
        let diss: Vec<f64> = true_d.iter().map(|d| d.powf(1.7)).collect();

        let res = nonmetric_mds(&diss, n, dim, 300, 1e-9).expect("nmds ok");
        assert!(res.embedding.iter().all(|v| v.is_finite()));

        // Pairwise distances of the recovered configuration vs true distances.
        let rec_d = euclidean_matrix(&res.embedding, n, dim);
        let mut a = Vec::new();
        let mut b = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                a.push(rec_d[i * n + j]);
                b.push(true_d[i * n + j]);
            }
        }
        let rho = spearman(&a, &b);
        assert!(rho > 0.97, "Spearman rank correlation too low: {rho}");
        assert!(res.stress < 0.05, "final stress too high: {}", res.stress);
    }

    /// (b') A different monotone warp (exp(d) - 1) is likewise undone.
    #[test]
    fn nonmetric_recovers_exp_warp() {
        let mut rng = LcgRng::new(77);
        let n = 28;
        let dim = 2;
        let mut pts = vec![0.0; n * dim];
        for v in &mut pts {
            *v = rng.next_range(-1.5, 1.5);
        }
        let true_d = euclidean_matrix(&pts, n, dim);
        let diss: Vec<f64> = true_d.iter().map(|d| d.exp() - 1.0).collect();
        let res = nonmetric_mds(&diss, n, dim, 300, 1e-9).expect("nmds ok");
        let rec_d = euclidean_matrix(&res.embedding, n, dim);
        let mut a = Vec::new();
        let mut b = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                a.push(rec_d[i * n + j]);
                b.push(true_d[i * n + j]);
            }
        }
        let rho = spearman(&a, &b);
        assert!(rho > 0.95, "Spearman rank correlation too low: {rho}");
    }

    /// (c) Stress is monotonically non-increasing across iterations.
    #[test]
    fn nonmetric_stress_monotone() {
        let mut rng = LcgRng::new(11);
        let n = 24;
        let dim = 2;
        let mut pts = vec![0.0; n * dim];
        for v in &mut pts {
            *v = rng.next_range(-2.0, 2.0);
        }
        let true_d = euclidean_matrix(&pts, n, dim);
        let diss: Vec<f64> = true_d.iter().map(|d| d.powf(1.5)).collect();

        // Run the loop ourselves to record the stress trajectory.
        let stresses = record_stress_trajectory(&diss, n, dim, 60);
        assert!(stresses.len() >= 2);
        for w in stresses.windows(2) {
            // Allow a tiny numerical slack.
            assert!(
                w[1] <= w[0] + 1e-6,
                "stress increased: {} -> {}",
                w[0],
                w[1]
            );
        }
    }

    /// (d) Stress ~ 0 when the dissimilarities are a monotone transform of a perfectly
    /// 2D-embeddable configuration.
    #[test]
    fn nonmetric_zero_stress_for_embeddable() {
        let mut rng = LcgRng::new(5);
        let n = 25;
        let dim = 2;
        let mut pts = vec![0.0; n * dim];
        for v in &mut pts {
            *v = rng.next_range(-1.0, 1.0);
        }
        let true_d = euclidean_matrix(&pts, n, dim);
        // A monotone transform that is still perfectly realisable in 2D after rescaling.
        let diss: Vec<f64> = true_d.iter().map(|d| 2.0 * d + 0.0).collect();
        let res = nonmetric_mds(&diss, n, dim, 400, 1e-12).expect("nmds ok");
        assert!(res.stress < 1e-3, "stress not near zero: {}", res.stress);
    }

    /// (e) Parameter-validation errors.
    #[test]
    fn nonmetric_param_errors() {
        // Non-square dissimilarity matrix.
        let bad = vec![0.0; 3 * 4];
        assert!(nonmetric_mds(&bad, 3, 1, 10, 1e-6).is_err());
        // n_components >= n.
        let d = vec![0.0; 4 * 4];
        assert!(nonmetric_mds(&d, 4, 4, 10, 1e-6).is_err());
        // Empty input.
        assert!(nonmetric_mds(&[], 0, 1, 10, 1e-6).is_err());
    }

    /// Run the non-metric MDS loop and return the per-iteration stress trajectory.
    fn record_stress_trajectory(
        dissimilarities: &[f64],
        n: usize,
        dim: usize,
        max_iter: usize,
    ) -> Vec<f64> {
        let num_pairs = n * (n - 1) / 2;
        let mut pair_a = Vec::with_capacity(num_pairs);
        let mut pair_b = Vec::with_capacity(num_pairs);
        let mut delta = Vec::with_capacity(num_pairs);
        for i in 0..n {
            for j in (i + 1)..n {
                pair_a.push(i);
                pair_b.push(j);
                delta.push(dissimilarities[i * n + j]);
            }
        }
        let mut order: Vec<usize> = (0..num_pairs).collect();
        order.sort_by(|&p, &q| {
            delta[p]
                .partial_cmp(&delta[q])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut x = classical_mds(dissimilarities, n, dim)
            .expect("cmds")
            .embedding;
        let mut dist = vec![0.0f64; num_pairs];
        let mut ordered_dist = vec![0.0f64; num_pairs];
        let mut dhat_full = vec![0.0f64; num_pairs];
        let mut out = Vec::new();
        for _ in 0..max_iter {
            for p in 0..num_pairs {
                let i = pair_a[p];
                let j = pair_b[p];
                let mut s = 0.0;
                for c in 0..dim {
                    let v = x[i * dim + c] - x[j * dim + c];
                    s += v * v;
                }
                dist[p] = s.sqrt();
            }
            for (slot, &p) in order.iter().enumerate() {
                ordered_dist[slot] = dist[p];
            }
            let dhat_ordered = pava(&ordered_dist);
            for (slot, &p) in order.iter().enumerate() {
                dhat_full[p] = dhat_ordered[slot];
            }
            let mut num = 0.0;
            let mut den = 0.0;
            for p in 0..num_pairs {
                let diff = dist[p] - dhat_full[p];
                num += diff * diff;
                den += dist[p] * dist[p];
            }
            out.push(if den > 1e-30 { (num / den).sqrt() } else { 0.0 });
            x = guttman_step_disparities(&x, &dhat_full, &pair_a, &pair_b, n, dim);
        }
        out
    }
}
