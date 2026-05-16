//! Filtered simplicial complexes: Vietoris-Rips, sublevel-set filtrations.

use crate::complex::simplex::Simplex;
use crate::error::{TdaError, TdaResult};

/// A simplex together with its filtration value (the parameter at which it first appears).
#[derive(Debug, Clone)]
pub struct FilteredSimplex {
    pub simplex: Simplex,
    pub value: f64,
}

/// A filtration: a list of `FilteredSimplex` entries sorted by `value`.
///
/// Ties are broken by dimension (lower dimension comes first).
#[derive(Debug, Clone)]
pub struct Filtration {
    pub simplices: Vec<FilteredSimplex>,
}

impl Filtration {
    /// Build a filtration from a raw list of `FilteredSimplex` entries.
    ///
    /// Sorts by value (then dimension for ties) and validates that no `value` is NaN.
    pub fn new(mut simplices: Vec<FilteredSimplex>) -> TdaResult<Self> {
        for fs in &simplices {
            if fs.value.is_nan() {
                return Err(TdaError::NanFiltrationValue);
            }
        }
        simplices.sort_by(|a, b| {
            a.value
                .partial_cmp(&b.value)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.simplex.dim().cmp(&b.simplex.dim()))
                .then_with(|| a.simplex.cmp(&b.simplex))
        });
        Ok(Self { simplices })
    }

    /// Number of simplices in the filtration.
    pub fn n_simplices(&self) -> usize {
        self.simplices.len()
    }

    /// Build a Vietoris-Rips filtration up to dimension `max_dim` from a pairwise distance
    /// matrix (n × n, row-major, symmetric).
    ///
    /// - 0-simplices: all vertices at value 0.
    /// - d-simplices (d ≥ 1): all (d+1)-subsets of {0,…,n-1} whose max pairwise distance
    ///   is ≤ `max_radius`; filtration value = that max distance.
    pub fn vietoris_rips(
        dist: &[f64],
        n_pts: usize,
        max_radius: f64,
        max_dim: usize,
    ) -> TdaResult<Self> {
        if n_pts == 0 {
            return Err(TdaError::EmptyPointCloud);
        }
        if dist.len() != n_pts * n_pts {
            return Err(TdaError::DimensionMismatch {
                expected: n_pts * n_pts,
                got: dist.len(),
            });
        }
        if max_dim > 6 {
            return Err(TdaError::DimensionTooLarge(max_dim));
        }

        let mut simplices: Vec<FilteredSimplex> = Vec::new();

        // 0-simplices: all vertices at value 0
        for i in 0..n_pts {
            simplices.push(FilteredSimplex {
                simplex: Simplex { vertices: vec![i] },
                value: 0.0,
            });
        }

        // Higher-dimensional simplices via subset enumeration
        for d in 1..=max_dim {
            // Enumerate all (d+1)-subsets of {0..n_pts}
            let size = d + 1;
            let mut indices: Vec<usize> = (0..size).collect();
            loop {
                // Compute filtration value = max pairwise distance in subset
                let mut max_dist = 0.0_f64;
                'outer: for ii in 0..size {
                    for jj in (ii + 1)..size {
                        let a = indices[ii];
                        let b = indices[jj];
                        let dab = dist[a * n_pts + b];
                        if dab.is_nan() {
                            return Err(TdaError::NanFiltrationValue);
                        }
                        if dab > max_dist {
                            max_dist = dab;
                        }
                        if max_dist > max_radius {
                            break 'outer;
                        }
                    }
                }
                if max_dist <= max_radius {
                    simplices.push(FilteredSimplex {
                        simplex: Simplex {
                            vertices: indices.clone(),
                        },
                        value: max_dist,
                    });
                }
                // Advance to the next (d+1)-subset in lex order
                if !next_combination(&mut indices, n_pts) {
                    break;
                }
            }
        }

        Self::new(simplices)
    }

    /// Build Vietoris-Rips from a raw point cloud (n_pts × n_dims, row-major).
    pub fn vietoris_rips_from_points(
        points: &[f64],
        n_dims: usize,
        max_radius: f64,
        max_dim: usize,
    ) -> TdaResult<Self> {
        if n_dims == 0 {
            return Err(TdaError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if points.is_empty() {
            return Err(TdaError::EmptyPointCloud);
        }
        let n_pts = points.len() / n_dims;
        if n_pts == 0 || !points.len().is_multiple_of(n_dims) {
            return Err(TdaError::DimensionMismatch {
                expected: n_pts * n_dims,
                got: points.len(),
            });
        }
        let dist = crate::distance::pairwise::pairwise_euclidean(points, n_dims)?;
        Self::vietoris_rips(&dist, n_pts, max_radius, max_dim)
    }

    /// Sublevel-set filtration from scalar function values on vertices + an edge list.
    ///
    /// Vertex i appears at value `values[i]`.
    /// Edge (i,j) appears at `max(values[i], values[j])`.
    pub fn sublevel_set(values: &[f64], edges: &[(usize, usize)]) -> TdaResult<Self> {
        if values.is_empty() {
            return Err(TdaError::EmptyPointCloud);
        }
        for &v in values {
            if v.is_nan() {
                return Err(TdaError::NanFiltrationValue);
            }
        }
        let n = values.len();
        let mut simplices: Vec<FilteredSimplex> = Vec::new();

        // Vertices
        for (i, &val) in values.iter().enumerate() {
            simplices.push(FilteredSimplex {
                simplex: Simplex { vertices: vec![i] },
                value: val,
            });
        }

        // Edges
        for &(a, b) in edges {
            if a >= n || b >= n {
                return Err(TdaError::InvalidSimplex(format!(
                    "edge ({a},{b}) out of range for {n} vertices"
                )));
            }
            let val = values[a].max(values[b]);
            let mut verts = vec![a, b];
            verts.sort_unstable();
            simplices.push(FilteredSimplex {
                simplex: Simplex { vertices: verts },
                value: val,
            });
        }

        Self::new(simplices)
    }
}

/// Advance `indices` to the next k-combination of {0, …, n-1} in lex order.
/// Returns `false` when `indices` is already the last combination.
fn next_combination(indices: &mut [usize], n: usize) -> bool {
    let k = indices.len();
    if k == 0 {
        return false;
    }
    // Find rightmost index that can be incremented.
    let mut i = k;
    loop {
        if i == 0 {
            return false; // exhausted all combinations
        }
        i -= 1;
        if indices[i] < n - (k - i) {
            indices[i] += 1;
            for j in (i + 1)..k {
                indices[j] = indices[j - 1] + 1;
            }
            return true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vr_r0_vertices_only() {
        // With radius=0, only 0-simplices should appear.
        let pts = vec![0.0f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5];
        let filt = Filtration::vietoris_rips_from_points(&pts, 2, 0.0, 2).expect("ok");
        assert!(filt.simplices.iter().all(|fs| fs.simplex.dim() == 0));
    }

    #[test]
    fn vr_sorted_by_value() {
        let pts = vec![0.0f64, 0.0, 1.0, 0.0, 0.0, 1.0];
        let filt = Filtration::vietoris_rips_from_points(&pts, 2, 10.0, 2).expect("ok");
        for w in filt.simplices.windows(2) {
            assert!(w[0].value <= w[1].value, "filtration not sorted");
        }
    }
}
