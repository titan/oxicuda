use crate::error::{CausalError, CausalResult};

fn compute_bic(residual_variance: f32, n: usize, n_parents: usize) -> f32 {
    let log_n = (n as f32).ln();
    // BIC = -n/2 * log(sigma^2) - k * log(n) / 2  (higher is better)
    if residual_variance <= 0.0 {
        return f32::NEG_INFINITY;
    }
    -0.5 * n as f32 * residual_variance.ln() - 0.5 * n_parents as f32 * log_n
}

fn linear_residual_variance(
    data: &[f32],
    target: usize,
    parents: &[usize],
    n: usize,
    d: usize,
) -> f32 {
    let y: Vec<f32> = (0..n).map(|i| data[i * d + target]).collect();
    if parents.is_empty() {
        let mean = y.iter().sum::<f32>() / n as f32;
        return y.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    }
    let p = parents.len();
    let mut x_mat = vec![0.0_f32; n * p];
    for (col, &par) in parents.iter().enumerate() {
        for row in 0..n {
            x_mat[row * p + col] = data[row * d + par];
        }
    }
    let mut xtx = vec![0.0_f32; p * p];
    let mut xty = vec![0.0_f32; p];
    for row in 0..n {
        for i in 0..p {
            for j in 0..p {
                xtx[i * p + j] += x_mat[row * p + i] * x_mat[row * p + j];
            }
            xty[i] += x_mat[row * p + i] * y[row];
        }
    }
    for i in 0..p {
        xtx[i * p + i] += 1e-6;
    }
    let inv = match super::notears::gauss_jordan_inv(&xtx, p, 0.0) {
        Ok(m) => m,
        Err(_) => return f32::MAX,
    };
    let beta: Vec<f32> = (0..p)
        .map(|i| (0..p).map(|j| inv[i * p + j] * xty[j]).sum())
        .collect();
    let ss_res: f32 = (0..n)
        .map(|i| {
            let pred: f32 = (0..p).map(|j| x_mat[i * p + j] * beta[j]).sum();
            (y[i] - pred).powi(2)
        })
        .sum();
    ss_res / n as f32
}

pub struct Ges {
    pub cpdag: Vec<(usize, usize, bool)>,
}

impl Ges {
    pub fn run(data: &[f32], n: usize, d: usize) -> CausalResult<Self> {
        if data.is_empty() || n < 4 || d < 2 {
            return Err(CausalError::EmptyInput);
        }
        if data.len() != n * d {
            return Err(CausalError::DimensionMismatch {
                expected: n * d,
                got: data.len(),
            });
        }

        // Start with empty graph: parents[i] = set of parents of i
        let mut parents: Vec<Vec<usize>> = vec![vec![]; d];

        // Compute baseline BIC scores
        let mut scores: Vec<f32> = (0..d)
            .map(|i| compute_bic(linear_residual_variance(data, i, &[], n, d), n, 0))
            .collect();

        // Forward phase: greedily add edges that most improve total BIC
        let mut changed = true;
        while changed {
            changed = false;
            let mut best_delta = 0.0_f32;
            let mut best_edge: Option<(usize, usize)> = None;

            for from in 0..d {
                for to in 0..d {
                    if from == to || parents[to].contains(&from) {
                        continue;
                    }
                    // Check for acyclicity: from should not be a descendant of to
                    if would_create_cycle(&parents, from, to, d) {
                        continue;
                    }
                    let mut new_parents = parents[to].clone();
                    new_parents.push(from);
                    let new_var = linear_residual_variance(data, to, &new_parents, n, d);
                    let new_score = compute_bic(new_var, n, new_parents.len());
                    let delta = new_score - scores[to];
                    if delta > best_delta {
                        best_delta = delta;
                        best_edge = Some((from, to));
                    }
                }
            }

            if let Some((from, to)) = best_edge {
                parents[to].push(from);
                let new_var = linear_residual_variance(data, to, &parents[to], n, d);
                scores[to] = compute_bic(new_var, n, parents[to].len());
                changed = true;
            }
        }

        // Backward phase: remove edges that improve BIC
        changed = true;
        while changed {
            changed = false;
            let mut best_delta = 0.0_f32;
            let mut best_remove: Option<(usize, usize)> = None;

            for to in 0..d {
                for (idx, &from) in parents[to].iter().enumerate() {
                    let mut new_parents = parents[to].clone();
                    new_parents.remove(idx);
                    let new_var = linear_residual_variance(data, to, &new_parents, n, d);
                    let new_score = compute_bic(new_var, n, new_parents.len());
                    let delta = new_score - scores[to];
                    if delta > best_delta {
                        best_delta = delta;
                        best_remove = Some((from, to));
                    }
                }
            }

            if let Some((from, to)) = best_remove {
                parents[to].retain(|&v| v != from);
                let new_var = linear_residual_variance(data, to, &parents[to], n, d);
                scores[to] = compute_bic(new_var, n, parents[to].len());
                changed = true;
            }
        }

        // Build CPDAG (represent as oriented edges)
        let mut cpdag = Vec::new();
        for (to, parent_set) in parents.iter().enumerate() {
            for &from in parent_set {
                cpdag.push((from, to, true));
            }
        }

        Ok(Self { cpdag })
    }
}

fn would_create_cycle(parents: &[Vec<usize>], from: usize, to: usize, d: usize) -> bool {
    // BFS from `to` following parent edges to check if we reach `from`
    let mut visited = vec![false; d];
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(to);
    visited[to] = true;
    while let Some(cur) = queue.pop_front() {
        if cur == from {
            return true;
        }
        for &p in &parents[cur] {
            if !visited[p] {
                visited[p] = true;
                queue.push_back(p);
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ges_runs_minimal() {
        let n = 40;
        let d = 3;
        let mut data = vec![0.0_f32; n * d];
        for i in 0..n {
            let x = i as f32 / n as f32;
            data[i * d] = x;
            data[i * d + 1] = 0.8 * x + 0.2;
            data[i * d + 2] = 0.5 * x + 0.5;
        }
        let result = Ges::run(&data, n, d);
        assert!(result.is_ok());
    }
}
