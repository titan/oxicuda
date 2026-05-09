use crate::error::{CausalError, CausalResult};
use std::collections::HashMap;

fn mean(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f32>() / v.len() as f32
}

fn pearson_corr(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    let mx = mean(&x[..n]);
    let my = mean(&y[..n]);
    let num: f32 = (0..n).map(|i| (x[i] - mx) * (y[i] - my)).sum();
    let sx: f32 = (0..n).map(|i| (x[i] - mx).powi(2)).sum::<f32>().sqrt();
    let sy: f32 = (0..n).map(|i| (y[i] - my).powi(2)).sum::<f32>().sqrt();
    if sx < 1e-10 || sy < 1e-10 {
        return 0.0;
    }
    (num / (sx * sy)).clamp(-1.0, 1.0)
}

/// Compute regression residuals of y on x_mat.
fn regress_residuals(x_mat: &[f32], y: &[f32], n: usize, d: usize) -> Vec<f32> {
    if d == 0 {
        return y.to_vec();
    }
    // Use closed-form OLS if d is small; fall back to simple mean subtraction
    let mut xtx = vec![0.0_f32; d * d];
    let mut xty = vec![0.0_f32; d];
    for row in 0..n {
        for i in 0..d {
            for j in 0..d {
                xtx[i * d + j] += x_mat[row * d + i] * x_mat[row * d + j];
            }
            xty[i] += x_mat[row * d + i] * y[row];
        }
    }
    // Add small ridge
    for i in 0..d {
        xtx[i * d + i] += 1e-6;
    }
    // Solve via Gauss-Jordan (reuse from notears)
    let inv = match super::notears::gauss_jordan_inv(&xtx, d, 0.0) {
        Ok(m) => m,
        Err(_) => return y.to_vec(),
    };
    let beta: Vec<f32> = (0..d)
        .map(|i| (0..d).map(|j| inv[i * d + j] * xty[j]).sum())
        .collect();
    let mut residuals = vec![0.0_f32; n];
    for row in 0..n {
        let pred: f32 = (0..d).map(|j| x_mat[row * d + j] * beta[j]).sum();
        residuals[row] = y[row] - pred;
    }
    residuals
}

/// Partial correlation of x and y conditioning on z columns.
pub fn partial_corr(x: &[f32], y: &[f32], z: &[Vec<f32>], n: usize) -> f32 {
    let dz = z.len();
    if dz == 0 {
        return pearson_corr(x, y);
    }
    // Build conditioning matrix (row-major)
    let mut z_mat = vec![0.0_f32; n * dz];
    for (col, zv) in z.iter().enumerate() {
        for row in 0..n.min(zv.len()) {
            z_mat[row * dz + col] = zv[row];
        }
    }
    let rx = regress_residuals(&z_mat, x, n, dz);
    let ry = regress_residuals(&z_mat, y, n, dz);
    pearson_corr(&rx, &ry)
}

/// Fisher Z-test for conditional independence.
/// Returns true if we reject independence (i.e., variables ARE dependent).
pub fn fisher_z_test(r: f32, n: usize, cond_set_size: usize, alpha: f32) -> bool {
    let r_clamped = r.clamp(-0.999, 0.999);
    let z = 0.5 * ((1.0 + r_clamped) / (1.0 - r_clamped)).ln();
    let df = (n as f32 - cond_set_size as f32 - 3.0).max(1.0);
    let stat = z.abs() * df.sqrt();
    // z_alpha for alpha=0.05 is 1.96
    let z_alpha = if alpha <= 0.01 {
        2.576
    } else if alpha <= 0.05 {
        1.96
    } else {
        1.645
    };
    stat > z_alpha
}

pub struct PcAlgorithm {
    pub skeleton: Vec<(usize, usize)>,
    pub cpdag: Vec<(usize, usize, bool)>,
    pub sep_sets: HashMap<(usize, usize), Vec<usize>>,
}

impl PcAlgorithm {
    pub fn run(data: &[f32], n: usize, d: usize, alpha: f32) -> CausalResult<Self> {
        if data.is_empty() || n < 4 || d < 2 {
            return Err(CausalError::EmptyInput);
        }
        if data.len() != n * d {
            return Err(CausalError::DimensionMismatch {
                expected: n * d,
                got: data.len(),
            });
        }

        // Extract columns
        let cols: Vec<Vec<f32>> = (0..d)
            .map(|j| (0..n).map(|i| data[i * d + j]).collect())
            .collect();

        // Start with complete undirected graph
        let mut adj: Vec<Vec<bool>> = vec![vec![true; d]; d];
        for (i, row) in adj.iter_mut().enumerate() {
            row[i] = false;
        }

        let mut sep_sets: HashMap<(usize, usize), Vec<usize>> = HashMap::new();

        // Skeleton phase: remove edges by conditional independence tests
        let max_cond_size = d.saturating_sub(2).min(3); // limit for tractability
        for cond_size in 0..=max_cond_size {
            let mut to_remove = Vec::new();
            for x in 0..d {
                for y in (x + 1)..d {
                    if !adj[x][y] {
                        continue;
                    }
                    // Collect neighbors of x (excluding y)
                    let neighbors_x: Vec<usize> =
                        (0..d).filter(|&v| v != x && v != y && adj[x][v]).collect();
                    if neighbors_x.len() < cond_size {
                        continue;
                    }
                    // Try all subsets of size cond_size
                    let subsets = subsets_of_size(&neighbors_x, cond_size);
                    for subset in subsets {
                        let z_vecs: Vec<Vec<f32>> =
                            subset.iter().map(|&k| cols[k].clone()).collect();
                        let r = partial_corr(&cols[x], &cols[y], &z_vecs, n);
                        let dependent = fisher_z_test(r, n, subset.len(), alpha);
                        if !dependent {
                            to_remove.push((x, y, subset.clone()));
                            break;
                        }
                    }
                }
            }
            for (x, y, sep) in to_remove {
                adj[x][y] = false;
                adj[y][x] = false;
                sep_sets.insert((x, y), sep.clone());
                sep_sets.insert((y, x), sep);
            }
        }

        // Build skeleton edges
        let mut skeleton = Vec::new();
        for (x, adj_x) in adj.iter().enumerate() {
            for (y, &edge) in adj_x.iter().enumerate().skip(x + 1) {
                if edge {
                    skeleton.push((x, y));
                }
            }
        }

        // V-structure orientation
        let mut oriented: Vec<Vec<Option<bool>>> = vec![vec![None; d]; d];
        for x in 0..d {
            for y in (x + 1)..d {
                if !adj[x][y] {
                    continue;
                }
                // Find common neighbors z where (x,z,y) form a potential v-structure
                for z in 0..d {
                    if z == x || z == y {
                        continue;
                    }
                    if !adj[x][z] || !adj[y][z] {
                        continue;
                    }
                    if adj[x][y] {
                        continue; // x and y are adjacent, not a v-structure
                    }
                    // x - z - y, x and y non-adjacent
                    let sep_xy = sep_sets.get(&(x, y)).cloned().unwrap_or_default();
                    if !sep_xy.contains(&z) {
                        // Orient x -> z <- y
                        oriented[x][z] = Some(true);
                        oriented[y][z] = Some(true);
                        oriented[z][x] = Some(false);
                        oriented[z][y] = Some(false);
                    }
                }
            }
        }

        // Apply Meek rules (simplified R1-R4)
        let mut changed = true;
        while changed {
            changed = false;
            // R1: If a->b-c and a not adj c, orient b->c
            for a in 0..d {
                for b in 0..d {
                    if oriented[a][b] != Some(true) {
                        continue;
                    }
                    for c in 0..d {
                        if c == a || !adj[b][c] || adj[a][c] {
                            continue;
                        }
                        if oriented[b][c].is_none() {
                            oriented[b][c] = Some(true);
                            oriented[c][b] = Some(false);
                            changed = true;
                        }
                    }
                }
            }
            // R2: If a->c and a-b->c, orient a->b (avoid cycle)
            for a in 0..d {
                for c in 0..d {
                    if oriented[a][c] != Some(true) {
                        continue;
                    }
                    for b in 0..d {
                        if b == a || b == c {
                            continue;
                        }
                        if adj[a][b] && oriented[a][b].is_none() && oriented[b][c] == Some(true) {
                            oriented[a][b] = Some(true);
                            oriented[b][a] = Some(false);
                            changed = true;
                        }
                    }
                }
            }
        }

        // Build CPDAG
        let mut cpdag = Vec::new();
        for x in 0..d {
            for y in (x + 1)..d {
                if !adj[x][y] {
                    continue;
                }
                match (oriented[x][y], oriented[y][x]) {
                    (Some(true), Some(false)) => cpdag.push((x, y, true)),
                    (Some(false), Some(true)) => cpdag.push((y, x, true)),
                    _ => {
                        cpdag.push((x, y, false));
                    }
                }
            }
        }

        Ok(Self {
            skeleton,
            cpdag,
            sep_sets,
        })
    }
}

fn subsets_of_size(items: &[usize], k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    if k > items.len() {
        return vec![];
    }
    if k == items.len() {
        return vec![items.to_vec()];
    }
    let mut result = Vec::new();
    for i in 0..items.len() {
        let rest = subsets_of_size(&items[i + 1..], k - 1);
        for mut sub in rest {
            sub.insert(0, items[i]);
            result.push(sub);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pc_runs_minimal() {
        // Simple 3-variable chain X -> Y -> Z
        let n = 30;
        let d = 3;
        let mut data = vec![0.0_f32; n * d];
        for i in 0..n {
            let x = (i as f32) / n as f32;
            data[i * d] = x;
            data[i * d + 1] = x + 0.1;
            data[i * d + 2] = x + 0.2;
        }
        let result = PcAlgorithm::run(&data, n, d, 0.05);
        assert!(result.is_ok());
    }
}
