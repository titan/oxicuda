//! Binary einsum-style contraction by label matching.
//!
//! Given two labelled tensors `A[i, j, k]` (labels `[i, j, k]`) and `B[j, l]` (labels
//! `[j, l]`), the contraction over shared labels (here `j`) returns `C[i, k, l]` with
//! `C[i, k, l] = sum_j A[i, j, k] * B[j, l]`.
//!
//! The implementation forbids repeated labels within a single tensor (no internal traces).

use crate::{TnError, TnResult};

/// A row-major flat tensor with a label for each axis.
#[derive(Debug, Clone)]
pub struct LabelledTensor {
    pub data: Vec<f64>,
    pub dims: Vec<usize>,
    pub labels: Vec<char>,
}

impl LabelledTensor {
    /// Construct a labelled tensor. Validates shape and that labels are distinct.
    pub fn new(data: Vec<f64>, dims: Vec<usize>, labels: Vec<char>) -> TnResult<Self> {
        if dims.len() != labels.len() {
            return Err(TnError::ShapeMismatch {
                expected: vec![labels.len()],
                got: vec![dims.len()],
            });
        }
        let total: usize = dims.iter().product();
        if data.len() != total {
            return Err(TnError::ShapeMismatch {
                expected: dims.clone(),
                got: vec![data.len()],
            });
        }
        let mut seen = std::collections::HashSet::new();
        for &l in &labels {
            if !seen.insert(l) {
                return Err(TnError::InvalidConfiguration(format!(
                    "duplicate label {l} in single tensor"
                )));
            }
        }
        Ok(Self { data, dims, labels })
    }
}

/// Contract two labelled tensors over their shared labels. The output's label order is
/// `a.labels \ shared ++ b.labels \ shared`.
pub fn einsum_binary(a: &LabelledTensor, b: &LabelledTensor) -> TnResult<LabelledTensor> {
    // Find shared labels (and their positions in a and b).
    let mut shared: Vec<(usize, usize)> = Vec::new();
    for (i, &la) in a.labels.iter().enumerate() {
        for (j, &lb) in b.labels.iter().enumerate() {
            if la == lb {
                shared.push((i, j));
            }
        }
    }
    // Verify that shared dims match
    for &(i, j) in &shared {
        if a.dims[i] != b.dims[j] {
            return Err(TnError::DimensionMismatch {
                a: a.dims[i],
                b: b.dims[j],
            });
        }
    }
    let shared_a: Vec<usize> = shared.iter().map(|(i, _)| *i).collect();
    let shared_b: Vec<usize> = shared.iter().map(|(_, j)| *j).collect();
    let kept_a: Vec<usize> = (0..a.labels.len())
        .filter(|i| !shared_a.contains(i))
        .collect();
    let kept_b: Vec<usize> = (0..b.labels.len())
        .filter(|j| !shared_b.contains(j))
        .collect();
    // Output dims and labels
    let mut out_dims: Vec<usize> = kept_a.iter().map(|&i| a.dims[i]).collect();
    out_dims.extend(kept_b.iter().map(|&j| b.dims[j]));
    let mut out_labels: Vec<char> = kept_a.iter().map(|&i| a.labels[i]).collect();
    out_labels.extend(kept_b.iter().map(|&j| b.labels[j]));
    let out_total: usize = out_dims.iter().product::<usize>().max(1);
    let mut out = vec![0.0; out_total];

    // Multi-index helper: given linear index in a tensor with strides, return per-axis indices.
    let a_strides = strides_from_dims(&a.dims);
    let b_strides = strides_from_dims(&b.dims);
    let out_strides = strides_from_dims(&out_dims);

    // Enumerate kept indices on a and b
    let n_kept_a: usize = kept_a.iter().map(|&i| a.dims[i]).product::<usize>().max(1);
    let n_kept_b: usize = kept_b.iter().map(|&j| b.dims[j]).product::<usize>().max(1);
    let n_shared: usize = shared_a
        .iter()
        .map(|&i| a.dims[i])
        .product::<usize>()
        .max(1);
    let kept_a_dims: Vec<usize> = kept_a.iter().map(|&i| a.dims[i]).collect();
    let kept_b_dims: Vec<usize> = kept_b.iter().map(|&j| b.dims[j]).collect();
    let shared_dims: Vec<usize> = shared_a.iter().map(|&i| a.dims[i]).collect();

    // Build expansion templates: arrays of axis-indices to apply to per-axis positions
    for ka in 0..n_kept_a {
        let ka_idx = unravel(ka, &kept_a_dims);
        for kb in 0..n_kept_b {
            let kb_idx = unravel(kb, &kept_b_dims);
            // Build output index
            let mut out_idx = 0usize;
            for (axis, &val) in ka_idx.iter().enumerate() {
                out_idx += val * out_strides[axis];
            }
            for (axis_off, &val) in kb_idx.iter().enumerate() {
                let axis = kept_a.len() + axis_off;
                out_idx += val * out_strides[axis];
            }
            let mut acc = 0.0;
            for s_idx in 0..n_shared {
                let s_pos = unravel(s_idx, &shared_dims);
                // Compute a_idx
                let mut a_idx = 0usize;
                for (k, &i) in kept_a.iter().enumerate() {
                    a_idx += ka_idx[k] * a_strides[i];
                }
                for (k, &i) in shared_a.iter().enumerate() {
                    a_idx += s_pos[k] * a_strides[i];
                }
                let mut b_idx = 0usize;
                for (k, &j) in kept_b.iter().enumerate() {
                    b_idx += kb_idx[k] * b_strides[j];
                }
                for (k, &j) in shared_b.iter().enumerate() {
                    b_idx += s_pos[k] * b_strides[j];
                }
                acc += a.data[a_idx] * b.data[b_idx];
            }
            out[out_idx] = acc;
        }
    }
    LabelledTensor::new(out, out_dims, out_labels)
}

fn strides_from_dims(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len().max(1)];
    if dims.is_empty() {
        return strides;
    }
    for i in (0..dims.len() - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

fn unravel(flat: usize, dims: &[usize]) -> Vec<usize> {
    let mut out = vec![0usize; dims.len()];
    let mut rem = flat;
    for i in (0..dims.len()).rev() {
        out[i] = rem % dims[i];
        rem /= dims[i];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn einsum_matmul_2x3_3x2() {
        let a = LabelledTensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            vec!['i', 'j'],
        )
        .expect("ok");
        let b = LabelledTensor::new(
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![3, 2],
            vec!['j', 'k'],
        )
        .expect("ok");
        let c = einsum_binary(&a, &b).expect("ok");
        assert_eq!(c.dims, vec![2, 2]);
        // [[1*1+2*0+3*1, 1*0+2*1+3*1], [4+0+6, 0+5+6]] = [[4, 5], [10, 11]]
        assert!((c.data[0] - 4.0).abs() < 1e-12);
        assert!((c.data[1] - 5.0).abs() < 1e-12);
        assert!((c.data[2] - 10.0).abs() < 1e-12);
        assert!((c.data[3] - 11.0).abs() < 1e-12);
    }

    #[test]
    fn einsum_dot_product() {
        let a = LabelledTensor::new(vec![1.0, 2.0, 3.0], vec![3], vec!['i']).expect("ok");
        let b = LabelledTensor::new(vec![4.0, 5.0, 6.0], vec![3], vec!['i']).expect("ok");
        let c = einsum_binary(&a, &b).expect("ok");
        assert_eq!(c.dims, vec![]);
        assert!((c.data[0] - 32.0).abs() < 1e-12);
    }

    #[test]
    fn duplicate_label_rejected() {
        let bad = LabelledTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], vec!['i', 'i']);
        assert!(bad.is_err());
    }
}
