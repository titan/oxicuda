//! CSR-style sparse spike encoding.
//!
//! A dense spike train is a `[T × N]` row-major binary matrix where row `t`
//! holds the spikes emitted at time step `t` (the layout produced by every
//! encoder in this module). In sparse spiking regimes the vast majority of
//! those entries are zero, so storing the full matrix wastes both memory and
//! bandwidth.
//!
//! [`SparseSpikes`] stores the same information in Compressed-Sparse-Row form:
//! one row per time step, with a `row_ptr` offset array (length `T + 1`), a
//! flat `col_idx` array of the active neuron indices (ascending within each
//! row), and an optional parallel `values` array for graded / weighted spikes
//! (binary spikes use the implicit value `1.0`). Only the non-zero spike
//! events are materialised, so the memory footprint scales with the total
//! number of spikes `nnz` rather than `T · N`.
//!
//! The round-trip `dense → sparse → dense` is exact for any binary spike
//! matrix, and [`SparseSpikes::forward`] computes the membrane current
//! `current = S · Wᵀ` (with `S` the `[T × N]` spike matrix and `W` an
//! `[out_dim × in_dim]` row-major weight matrix) touching **only** the active
//! spikes of each time step, matching the dense matmul bit-for-bit in exact
//! arithmetic.

use crate::error::{SnnError, SnnResult};

/// Sparse spike train in Compressed-Sparse-Row (per-time-step) layout.
///
/// Invariants (all upheld by the constructors / encoders in this module):
/// * `row_ptr.len() == t_steps + 1`, `row_ptr[0] == 0`,
///   `row_ptr[t_steps] == col_idx.len()`, and `row_ptr` is non-decreasing.
/// * `col_idx[row_ptr[t]..row_ptr[t + 1]]` is strictly ascending and every
///   index is `< n` (one entry per active neuron at time `t`).
/// * `values`, when present, has the same length as `col_idx`.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseSpikes {
    /// Number of time steps (rows).
    pub t_steps: usize,
    /// Number of neurons (dense column count).
    pub n: usize,
    /// Row offsets into `col_idx` / `values`, length `t_steps + 1`.
    pub row_ptr: Vec<usize>,
    /// Active neuron indices, ascending within each row.
    pub col_idx: Vec<usize>,
    /// Optional per-event spike magnitudes; `None` means every event is `1.0`.
    pub values: Option<Vec<f32>>,
}

impl SparseSpikes {
    /// Total number of stored spike events (non-zeros).
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.col_idx.len()
    }

    /// Number of active spikes at time step `t` (`0` if `t` is out of range).
    #[must_use]
    pub fn row_nnz(&self, t: usize) -> usize {
        if t >= self.t_steps {
            return 0;
        }
        self.row_ptr[t + 1] - self.row_ptr[t]
    }

    /// Fraction of the dense `[T × N]` matrix that is non-zero, in `[0, 1]`.
    ///
    /// Returns `0.0` for a degenerate (zero-sized) train.
    #[must_use]
    pub fn density(&self) -> f32 {
        let total = self.t_steps.saturating_mul(self.n);
        if total == 0 {
            0.0
        } else {
            self.nnz() as f32 / total as f32
        }
    }

    /// Borrow the `(col_idx, values?)` slices of the active spikes at time `t`.
    ///
    /// Returns an empty slice (and `None` values) when `t` is out of range.
    #[must_use]
    pub fn row(&self, t: usize) -> (&[usize], Option<&[f32]>) {
        if t >= self.t_steps {
            return (&[], None);
        }
        let lo = self.row_ptr[t];
        let hi = self.row_ptr[t + 1];
        let cols = &self.col_idx[lo..hi];
        let vals = self.values.as_ref().map(|v| &v[lo..hi]);
        (cols, vals)
    }

    /// Validate the structural invariants of a hand-constructed instance.
    ///
    /// The encoders in this module always produce valid instances; this guards
    /// `SparseSpikes` values assembled directly by a caller before they are
    /// fed to [`SparseSpikes::to_dense`] or [`SparseSpikes::forward`].
    pub fn validate(&self) -> SnnResult<()> {
        if self.row_ptr.len() != self.t_steps + 1 {
            return Err(SnnError::BadShape {
                expected: self.t_steps + 1,
                got: self.row_ptr.len(),
            });
        }
        if self.row_ptr[0] != 0 {
            return Err(SnnError::Internal {
                msg: format!("row_ptr[0] = {} (must be 0)", self.row_ptr[0]),
            });
        }
        if *self.row_ptr.last().unwrap_or(&0) != self.col_idx.len() {
            return Err(SnnError::BadShape {
                expected: self.col_idx.len(),
                got: *self.row_ptr.last().unwrap_or(&0),
            });
        }
        if let Some(vals) = &self.values
            && vals.len() != self.col_idx.len()
        {
            return Err(SnnError::IncompatibleLength {
                a: vals.len(),
                b: self.col_idx.len(),
            });
        }
        for t in 0..self.t_steps {
            let lo = self.row_ptr[t];
            let hi = self.row_ptr[t + 1];
            if hi < lo {
                return Err(SnnError::Internal {
                    msg: format!("row_ptr not monotone at t={t}: {lo} > {hi}"),
                });
            }
            let mut prev: Option<usize> = None;
            for &c in &self.col_idx[lo..hi] {
                if c >= self.n {
                    return Err(SnnError::OutOfRange {
                        name: "col_idx".into(),
                        val: c as f32,
                    });
                }
                if let Some(p) = prev
                    && c <= p
                {
                    return Err(SnnError::Internal {
                        msg: format!("col_idx not strictly ascending in row {t}"),
                    });
                }
                prev = Some(c);
            }
        }
        Ok(())
    }
}

/// Encode a dense `[t_steps × n]` row-major spike matrix into CSR form.
///
/// Any entry that is non-zero and finite is treated as an active spike. When
/// `keep_values` is `false` the magnitudes are dropped (binary spikes), saving
/// the `values` array; when `true`, the original entry values are preserved
/// (graded spikes).
///
/// Errors: [`SnnError::BadTimesteps`] / [`SnnError::BadDim`] for zero-sized
/// axes, [`SnnError::BadShape`] when `dense.len() != t_steps * n`.
pub fn encode_dense_to_sparse(
    dense: &[f32],
    t_steps: usize,
    n: usize,
    keep_values: bool,
) -> SnnResult<SparseSpikes> {
    if t_steps == 0 {
        return Err(SnnError::BadTimesteps { got: t_steps });
    }
    if n == 0 {
        return Err(SnnError::BadDim { got: n });
    }
    if dense.len() != t_steps * n {
        return Err(SnnError::BadShape {
            expected: t_steps * n,
            got: dense.len(),
        });
    }
    let mut row_ptr = Vec::with_capacity(t_steps + 1);
    let mut col_idx = Vec::new();
    let mut values: Vec<f32> = Vec::new();
    row_ptr.push(0_usize);
    for t in 0..t_steps {
        let row = &dense[t * n..(t + 1) * n];
        for (j, &s) in row.iter().enumerate() {
            if s != 0.0 && s.is_finite() {
                col_idx.push(j);
                if keep_values {
                    values.push(s);
                }
            }
        }
        row_ptr.push(col_idx.len());
    }
    Ok(SparseSpikes {
        t_steps,
        n,
        row_ptr,
        col_idx,
        values: if keep_values { Some(values) } else { None },
    })
}

impl SparseSpikes {
    /// Reconstruct the dense `[t_steps × n]` row-major spike matrix.
    ///
    /// Binary trains (`values == None`) decode every event back to `1.0`; graded
    /// trains restore the stored magnitudes. The result is an exact inverse of
    /// [`encode_dense_to_sparse`] for any matrix whose only zeros are genuine
    /// absences of a spike.
    #[must_use]
    pub fn to_dense(&self) -> Vec<f32> {
        let mut dense = vec![0.0_f32; self.t_steps * self.n];
        for t in 0..self.t_steps {
            let lo = self.row_ptr[t];
            let hi = self.row_ptr[t + 1];
            let base = t * self.n;
            for k in lo..hi {
                let j = self.col_idx[k];
                let v = match &self.values {
                    Some(vals) => vals[k],
                    None => 1.0_f32,
                };
                dense[base + j] = v;
            }
        }
        dense
    }

    /// Sparse spike × weight forward pass: `current = S · Wᵀ`.
    ///
    /// `weight` is the `[out_dim × in_dim]` row-major matrix (the same layout
    /// as [`crate::layer::spiking_linear::SpikingLinear`]), where `in_dim`
    /// equals `self.n`. For each time step only the active spikes are visited,
    /// so the cost scales with `nnz · out_dim` rather than `t_steps · n ·
    /// out_dim`. The returned buffer is the dense `[t_steps × out_dim]`
    /// membrane current.
    ///
    /// Errors: [`SnnError::BadDim`] for `out_dim == 0`,
    /// [`SnnError::BadShape`] when `weight.len() != out_dim * self.n`.
    pub fn forward(&self, weight: &[f32], out_dim: usize) -> SnnResult<Vec<f32>> {
        if out_dim == 0 {
            return Err(SnnError::BadDim { got: out_dim });
        }
        if weight.len() != out_dim * self.n {
            return Err(SnnError::BadShape {
                expected: out_dim * self.n,
                got: weight.len(),
            });
        }
        let mut current = vec![0.0_f32; self.t_steps * out_dim];
        for t in 0..self.t_steps {
            let lo = self.row_ptr[t];
            let hi = self.row_ptr[t + 1];
            if lo == hi {
                continue; // no spikes this step → current row stays zero
            }
            let out_base = t * out_dim;
            for o in 0..out_dim {
                let row_off = o * self.n;
                let mut acc = 0.0_f32;
                for k in lo..hi {
                    let j = self.col_idx[k];
                    let s = match &self.values {
                        Some(vals) => vals[k],
                        None => 1.0_f32,
                    };
                    acc += weight[row_off + j] * s;
                }
                current[out_base + o] = acc;
            }
        }
        Ok(current)
    }
}

/// Dense reference forward pass `current = S · Wᵀ` for a dense `[t_steps × n]`
/// spike matrix; used to validate [`SparseSpikes::forward`].
///
/// Errors mirror [`SparseSpikes::forward`] plus shape checks on `dense`.
pub fn dense_forward(
    dense: &[f32],
    t_steps: usize,
    n: usize,
    weight: &[f32],
    out_dim: usize,
) -> SnnResult<Vec<f32>> {
    if t_steps == 0 {
        return Err(SnnError::BadTimesteps { got: t_steps });
    }
    if n == 0 || out_dim == 0 {
        return Err(SnnError::BadDim {
            got: n.min(out_dim),
        });
    }
    if dense.len() != t_steps * n {
        return Err(SnnError::BadShape {
            expected: t_steps * n,
            got: dense.len(),
        });
    }
    if weight.len() != out_dim * n {
        return Err(SnnError::BadShape {
            expected: out_dim * n,
            got: weight.len(),
        });
    }
    let mut current = vec![0.0_f32; t_steps * out_dim];
    for t in 0..t_steps {
        let row = &dense[t * n..(t + 1) * n];
        let out_base = t * out_dim;
        for o in 0..out_dim {
            let row_off = o * n;
            let mut acc = 0.0_f32;
            for (j, &s) in row.iter().enumerate() {
                acc += weight[row_off + j] * s;
            }
            current[out_base + o] = acc;
        }
    }
    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// A small known spike pattern: 4 time steps, 5 neurons.
    fn known_pattern() -> (Vec<f32>, usize, usize) {
        let t_steps = 4;
        let n = 5;
        // rows:
        // t0: neurons 0, 3
        // t1: (silent)
        // t2: neuron 4
        // t3: neurons 1, 2, 4
        let mut dense = vec![0.0_f32; t_steps * n];
        let mut set = |t: usize, j: usize| dense[t * n + j] = 1.0;
        set(0, 0);
        set(0, 3);
        set(2, 4);
        set(3, 1);
        set(3, 2);
        set(3, 4);
        (dense, t_steps, n)
    }

    #[test]
    fn round_trip_exact_binary() {
        let (dense, t, n) = known_pattern();
        let sparse = encode_dense_to_sparse(&dense, t, n, false).expect("encode");
        sparse.validate().expect("valid csr");
        let back = sparse.to_dense();
        assert_eq!(back, dense, "binary round-trip must be bit-exact");
    }

    #[test]
    fn round_trip_exact_graded() {
        // Graded (weighted) spikes round-trip with values preserved.
        let t = 3;
        let n = 4;
        let mut dense = vec![0.0_f32; t * n];
        let mut set = |t: usize, j: usize, v: f32| dense[t * n + j] = v;
        set(0, 1, 0.25);
        set(1, 0, -1.5);
        set(1, 3, 2.0);
        set(2, 2, 0.75);
        let sparse = encode_dense_to_sparse(&dense, t, n, true).expect("encode");
        sparse.validate().expect("valid csr");
        assert!(sparse.values.is_some());
        assert_eq!(sparse.to_dense(), dense);
    }

    #[test]
    fn nnz_and_row_structure_tracked() {
        let (dense, t, n) = known_pattern();
        let sparse = encode_dense_to_sparse(&dense, t, n, false).expect("encode");
        assert_eq!(sparse.nnz(), 6);
        assert_eq!(sparse.row_nnz(0), 2);
        assert_eq!(sparse.row_nnz(1), 0);
        assert_eq!(sparse.row_nnz(2), 1);
        assert_eq!(sparse.row_nnz(3), 3);
        assert_eq!(sparse.row_ptr, vec![0, 2, 2, 3, 6]);
        // ascending columns inside each row
        let (cols0, _) = sparse.row(0);
        assert_eq!(cols0, &[0, 3]);
        let (cols3, _) = sparse.row(3);
        assert_eq!(cols3, &[1, 2, 4]);
        // density = 6 / (4*5) = 0.3
        assert!((sparse.density() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn sparse_forward_equals_dense_forward_binary() {
        let (dense, t, n) = known_pattern();
        let out_dim = 3;
        // deterministic weight matrix [out_dim x n]
        let mut rng = LcgRng::new(123);
        let mut weight = vec![0.0_f32; out_dim * n];
        rng.fill_normal(&mut weight);

        let sparse = encode_dense_to_sparse(&dense, t, n, false).expect("encode");
        let cur_sparse = sparse.forward(&weight, out_dim).expect("sparse fwd");
        let cur_dense = dense_forward(&dense, t, n, &weight, out_dim).expect("dense fwd");
        assert_eq!(cur_sparse.len(), t * out_dim);
        for (a, b) in cur_sparse.iter().zip(cur_dense.iter()) {
            // identical FP32 accumulation order ⇒ exact equality.
            assert_eq!(a, b, "sparse forward must match dense forward exactly");
        }
        // silent step t1 produces an all-zero current row.
        let t1_base = out_dim; // step index 1
        for o in 0..out_dim {
            assert_eq!(cur_sparse[t1_base + o], 0.0);
        }
    }

    #[test]
    fn sparse_forward_equals_dense_forward_graded() {
        let t = 3;
        let n = 4;
        let out_dim = 2;
        let mut dense = vec![0.0_f32; t * n];
        let mut set = |t: usize, j: usize, v: f32| dense[t * n + j] = v;
        set(0, 1, 0.5);
        set(1, 3, -2.0);
        set(2, 0, 1.25);
        set(2, 2, 0.5);
        let mut rng = LcgRng::new(7);
        let mut weight = vec![0.0_f32; out_dim * n];
        rng.fill_normal(&mut weight);

        let sparse = encode_dense_to_sparse(&dense, t, n, true).expect("encode");
        let cur_sparse = sparse.forward(&weight, out_dim).expect("sparse fwd");
        let cur_dense = dense_forward(&dense, t, n, &weight, out_dim).expect("dense fwd");
        for (a, b) in cur_sparse.iter().zip(cur_dense.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn forward_on_random_sparse_train_matches_dense() {
        // Larger Bernoulli-sparse train: confirm equivalence at scale.
        let t = 32;
        let n = 64;
        let out_dim = 16;
        let p = 0.05_f32; // ~5% spiking
        let mut rng = LcgRng::new(99);
        let mut dense = vec![0.0_f32; t * n];
        for s in dense.iter_mut() {
            if rng.next_f32() < p {
                *s = 1.0;
            }
        }
        let mut weight = vec![0.0_f32; out_dim * n];
        rng.fill_normal(&mut weight);

        let sparse = encode_dense_to_sparse(&dense, t, n, false).expect("encode");
        // sparsity should be far below dense
        assert!(sparse.nnz() < t * n / 4, "train should be sparse");
        let cur_sparse = sparse.forward(&weight, out_dim).expect("sparse fwd");
        let cur_dense = dense_forward(&dense, t, n, &weight, out_dim).expect("dense fwd");
        for (a, b) in cur_sparse.iter().zip(cur_dense.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn rejects_invalid_arguments() {
        assert!(matches!(
            encode_dense_to_sparse(&[], 0, 4, false),
            Err(SnnError::BadTimesteps { .. })
        ));
        assert!(matches!(
            encode_dense_to_sparse(&[0.0; 4], 4, 0, false),
            Err(SnnError::BadDim { .. })
        ));
        assert!(matches!(
            encode_dense_to_sparse(&[0.0; 3], 4, 5, false),
            Err(SnnError::BadShape { .. })
        ));
        let (dense, t, n) = known_pattern();
        let sparse = encode_dense_to_sparse(&dense, t, n, false).expect("encode");
        assert!(matches!(
            sparse.forward(&[0.0; 3], 0),
            Err(SnnError::BadDim { .. })
        ));
        assert!(matches!(
            sparse.forward(&[0.0; 3], 3),
            Err(SnnError::BadShape { .. })
        ));
    }

    #[test]
    fn validate_catches_corruption() {
        let (dense, t, n) = known_pattern();
        let mut sparse = encode_dense_to_sparse(&dense, t, n, false).expect("encode");
        // break ascending-column invariant
        sparse.col_idx[0] = 3;
        sparse.col_idx[1] = 3;
        assert!(sparse.validate().is_err());
    }
}
