//! SMACOF MDS via Guttman transform majorisation.
//!
//! Iteratively minimises raw stress `sigma(Y) = sum_{ij} w_ij (d_ij - ||y_i - y_j||)^2`
//! by majorising it with a quadratic surrogate. The Guttman transform writes
//! `Y_{t+1} = V^+ B(Y_t) Y_t` where `B` is the matrix of normalised inverse distances.

use crate::error::{ManifoldError, ManifoldResult};
use crate::handle::LcgRng;
use crate::mds::classical_mds::classical_mds;

/// SMACOF result.
pub struct SmacofResult {
    pub embedding: Vec<f64>,
    pub stress: f64,
    pub iterations: usize,
}

/// SMACOF metric MDS.
///
/// `distances` is an `n x n` row-major symmetric matrix of target distances.
/// All weights are taken as 1 (uniform).
pub fn smacof_mds(
    distances: &[f64],
    n: usize,
    n_components: usize,
    max_iter: usize,
    tol: f64,
    rng: &mut LcgRng,
) -> ManifoldResult<SmacofResult> {
    if n == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if distances.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![distances.len()],
        });
    }
    if n_components == 0 || n_components >= n {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: format!("must be in 1..{n}"),
        });
    }
    // Initialise via classical MDS, but if it returns NaN fall back to random
    let mut y = match classical_mds(distances, n, n_components) {
        Ok(r) => {
            if r.embedding.iter().any(|v| !v.is_finite()) {
                let mut y = vec![0.0; n * n_components];
                for v in &mut y {
                    *v = rng.next_normal() * 0.01;
                }
                y
            } else {
                r.embedding
            }
        }
        Err(_) => {
            let mut y = vec![0.0; n * n_components];
            for v in &mut y {
                *v = rng.next_normal() * 0.01;
            }
            y
        }
    };
    let mut prev_stress = compute_stress(&y, distances, n, n_components);
    let mut iter_count = 0;
    for it in 0..max_iter {
        iter_count = it + 1;
        let new_y = guttman_step(&y, distances, n, n_components);
        let new_stress = compute_stress(&new_y, distances, n, n_components);
        let rel = if prev_stress.abs() > 1e-14 {
            (prev_stress - new_stress).abs() / prev_stress.abs()
        } else {
            (prev_stress - new_stress).abs()
        };
        y = new_y;
        prev_stress = new_stress;
        if rel < tol {
            break;
        }
    }
    Ok(SmacofResult {
        embedding: y,
        stress: prev_stress,
        iterations: iter_count,
    })
}

fn compute_stress(y: &[f64], d: &[f64], n: usize, dim: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut dy = 0.0;
            for k in 0..dim {
                let v = y[i * dim + k] - y[j * dim + k];
                dy += v * v;
            }
            let dy = dy.sqrt();
            let target = d[i * n + j];
            let diff = target - dy;
            s += diff * diff;
        }
    }
    s
}

fn guttman_step(y: &[f64], d: &[f64], n: usize, dim: usize) -> Vec<f64> {
    let mut b = vec![0.0; n * n];
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            if i == j {
                continue;
            }
            let mut dy = 0.0;
            for k in 0..dim {
                let v = y[i * dim + k] - y[j * dim + k];
                dy += v * v;
            }
            let dy = dy.sqrt();
            let target = d[i * n + j];
            if dy < 1e-12 {
                b[i * n + j] = 0.0;
            } else {
                b[i * n + j] = -target / dy;
                row_sum += target / dy;
            }
        }
        b[i * n + i] = row_sum;
    }
    // Y_new = (1/n) * B * Y  (using pseudoinverse of V = (n I - 1 1^T) gives 1/n times the off-diagonal Y centred sum but the classic update is Y_new = (1/n) * B * Y)
    let mut y_new = vec![0.0; n * dim];
    for i in 0..n {
        for k in 0..dim {
            let mut acc = 0.0;
            for j in 0..n {
                acc += b[i * n + j] * y[j * dim + k];
            }
            y_new[i * dim + k] = acc / n as f64;
        }
    }
    // Centre the embedding
    for k in 0..dim {
        let mut m = 0.0;
        for i in 0..n {
            m += y_new[i * dim + k];
        }
        m /= n as f64;
        for i in 0..n {
            y_new[i * dim + k] -= m;
        }
    }
    y_new
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smacof_decreasing_stress() {
        let pts: [f64; 8] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let n = 4;
        let mut d = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let dx = pts[i * 2] - pts[j * 2];
                let dy = pts[i * 2 + 1] - pts[j * 2 + 1];
                d[i * n + j] = (dx * dx + dy * dy).sqrt();
            }
        }
        let mut rng = LcgRng::new(5);
        let r = smacof_mds(&d, n, 2, 200, 1e-9, &mut rng).expect("ok");
        // Reconstructed stress should be small for embeddable points
        assert!(r.stress.is_finite());
        assert!(r.stress < 1.0);
    }
}
