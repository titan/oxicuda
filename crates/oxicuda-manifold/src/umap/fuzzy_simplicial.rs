//! Fuzzy simplicial set construction for UMAP.
//!
//! For kNN graph entries `(i, j)` with distance `d_ij`, compute membership
//! `mu_{ij} = exp(-max(0, d_ij - rho_i)/sigma_i)`. Symmetrise as `mu ∪ nu = mu + nu - mu * nu`.

use crate::error::{ManifoldError, ManifoldResult};

/// Build a sparse fuzzy graph as `(row, col, value)` triples (one direction per neighbour).
///
/// Returns a `(2*n*k)`-length triple of (rows, cols, vals).
pub fn fuzzy_simplicial_set(
    indices: &[usize],
    distances: &[f64],
    sigmas: &[f64],
    rhos: &[f64],
    n: usize,
    k: usize,
) -> ManifoldResult<(Vec<usize>, Vec<usize>, Vec<f64>)> {
    if indices.len() != n * k || distances.len() != n * k {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, k],
            got: vec![indices.len()],
        });
    }
    if sigmas.len() != n || rhos.len() != n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n],
            got: vec![sigmas.len()],
        });
    }
    let mut rows = Vec::with_capacity(n * k);
    let mut cols = Vec::with_capacity(n * k);
    let mut vals = Vec::with_capacity(n * k);
    for i in 0..n {
        for jj in 0..k {
            let j = indices[i * k + jj];
            let d = distances[i * k + jj];
            let arg = (d - rhos[i]).max(0.0);
            let mu = if sigmas[i] > 0.0 {
                (-arg / sigmas[i]).exp()
            } else {
                0.0
            };
            rows.push(i);
            cols.push(j);
            vals.push(mu);
        }
    }
    Ok((rows, cols, vals))
}

/// Symmetrise a sparse fuzzy graph via `mu ∪ nu = mu + nu - mu * nu`.
///
/// Returns deduplicated `(rows, cols, vals)` of merged edges.
pub fn symmetrise(
    rows: &[usize],
    cols: &[usize],
    vals: &[f64],
) -> ManifoldResult<(Vec<usize>, Vec<usize>, Vec<f64>)> {
    if rows.len() != cols.len() || cols.len() != vals.len() {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![rows.len()],
            got: vec![cols.len()],
        });
    }
    // Index map: (min(i,j), max(i,j)) -> (mu_ij, mu_ji)
    use std::collections::HashMap;
    let mut map: HashMap<(usize, usize), [f64; 2]> = HashMap::new();
    for k in 0..rows.len() {
        let i = rows[k];
        let j = cols[k];
        let v = vals[k];
        let key = (i.min(j), i.max(j));
        let entry = map.entry(key).or_insert([0.0; 2]);
        if i < j {
            entry[0] = v;
        } else {
            entry[1] = v;
        }
    }
    let mut out_rows = Vec::new();
    let mut out_cols = Vec::new();
    let mut out_vals = Vec::new();
    for ((i, j), v) in map {
        let merged = v[0] + v[1] - v[0] * v[1];
        if merged.abs() > 1e-12 {
            out_rows.push(i);
            out_cols.push(j);
            out_vals.push(merged);
        }
    }
    Ok((out_rows, out_cols, out_vals))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn membership_in_unit_interval() {
        let n = 3;
        let k = 2;
        let idx = vec![1, 2, 0, 2, 0, 1];
        let d = vec![0.1, 0.4, 0.2, 0.6, 0.3, 0.5];
        let sigma = vec![0.5, 0.7, 0.6];
        let rho = vec![0.05, 0.1, 0.2];
        let (_r, _c, v) = fuzzy_simplicial_set(&idx, &d, &sigma, &rho, n, k).expect("ok");
        for val in v {
            assert!((0.0..=1.0 + 1e-12).contains(&val));
        }
    }

    #[test]
    fn symmetrise_combines_pairs() {
        let rows = vec![0, 1];
        let cols = vec![1, 0];
        let vals = vec![0.5, 0.5];
        let (r, c, v) = symmetrise(&rows, &cols, &vals).expect("ok");
        assert_eq!(r.len(), 1);
        assert_eq!(c.len(), 1);
        // 0.5 + 0.5 - 0.25 = 0.75
        assert!((v[0] - 0.75).abs() < 1e-9);
    }
}
