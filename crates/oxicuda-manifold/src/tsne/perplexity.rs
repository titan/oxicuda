//! Perplexity-based conditional probabilities for t-SNE.
//!
//! Given a row of squared distances `d_i = (||x_i - x_j||^2)_j`, find `beta_i = 1/(2 sigma_i^2)`
//! such that `H(P_{j|i}) = log(perplexity)` where `P_{j|i} = exp(-d_ij * beta_i) / Z`.

use crate::error::{ManifoldError, ManifoldResult};

/// Solve for one row's beta and return `p_{j|i}` for all j != i.
///
/// `dist_sq` has length `n`, `i` is the source index (its self-distance is ignored).
/// Returns `(p_row, beta)` of lengths `n` and 1 respectively. `p_row[i] = 0`.
pub fn p_row_from_distances(
    dist_sq: &[f64],
    i: usize,
    perplexity: f64,
    max_iter: usize,
    tol: f64,
) -> ManifoldResult<(Vec<f64>, f64)> {
    let n = dist_sq.len();
    if n == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if i >= n {
        return Err(ManifoldError::IndexOutOfBounds { index: i, len: n });
    }
    let log_perp = perplexity.ln();
    let mut beta = 1.0_f64;
    let mut beta_min = f64::NEG_INFINITY;
    let mut beta_max = f64::INFINITY;
    let mut p_row = vec![0.0; n];
    for _ in 0..max_iter {
        // Compute P_{j|i} ∝ exp(-d_ij * beta)
        let mut z = 0.0;
        let mut h_num = 0.0; // sum p * (-d * beta)
        for (j, &d) in dist_sq.iter().enumerate() {
            if j == i {
                p_row[j] = 0.0;
                continue;
            }
            let e = (-d * beta).exp();
            p_row[j] = e;
            z += e;
            h_num += e * (-d * beta);
        }
        if z < 1e-300 {
            beta /= 2.0;
            continue;
        }
        // Normalise
        for p in &mut p_row {
            *p /= z;
        }
        // Entropy H = log(Z) + beta * sum(d * p)
        let h = z.ln() - h_num / z;
        let diff = h - log_perp;
        if diff.abs() < tol {
            return Ok((p_row, beta));
        }
        if diff > 0.0 {
            // H too large => need larger beta
            beta_min = beta;
            beta = if beta_max == f64::INFINITY {
                beta * 2.0
            } else {
                (beta + beta_max) / 2.0
            };
        } else {
            beta_max = beta;
            beta = if beta_min == f64::NEG_INFINITY {
                beta / 2.0
            } else {
                (beta + beta_min) / 2.0
            };
        }
    }
    Ok((p_row, beta))
}

/// Build the symmetric joint distribution `P = (P_{j|i} + P_{i|j}) / (2n)`.
pub fn compute_perplexity_p_matrix(
    dist_sq: &[f64],
    n: usize,
    perplexity: f64,
    max_iter: usize,
    tol: f64,
) -> ManifoldResult<Vec<f64>> {
    if dist_sq.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![dist_sq.len()],
        });
    }
    let mut p_cond = vec![0.0; n * n];
    for i in 0..n {
        let row = &dist_sq[i * n..(i + 1) * n];
        let (p_row, _beta) = p_row_from_distances(row, i, perplexity, max_iter, tol)?;
        for (j, &v) in p_row.iter().enumerate() {
            p_cond[i * n + j] = v;
        }
    }
    // Symmetrise: P = (P_cond + P_cond^T) / (2n)
    let denom = 2.0 * n as f64;
    let mut p = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            p[i * n + j] = (p_cond[i * n + j] + p_cond[j * n + i]) / denom;
        }
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perplexity_matches_log() {
        let dist = vec![0.0, 1.0, 4.0, 9.0];
        let (p_row, _beta) = p_row_from_distances(&dist, 0, 2.0, 100, 1e-6).expect("ok");
        let mut sum = 0.0;
        let mut h_sum = 0.0;
        for (j, &p) in p_row.iter().enumerate() {
            if j != 0 {
                sum += p;
                if p > 1e-300 {
                    h_sum -= p * p.ln();
                }
            }
        }
        assert!((sum - 1.0).abs() < 1e-6);
        // H ~ log(2) target
        assert!((h_sum - 2.0_f64.ln()).abs() < 0.1);
    }

    #[test]
    fn p_matrix_symmetric() {
        let n = 4;
        let mut d = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                d[i * n + j] = ((i as f64 - j as f64).powi(2)) * 1.0;
            }
        }
        let p = compute_perplexity_p_matrix(&d, n, 2.0, 200, 1e-7).expect("ok");
        for i in 0..n {
            for j in 0..n {
                assert!((p[i * n + j] - p[j * n + i]).abs() < 1e-10);
            }
        }
    }
}
