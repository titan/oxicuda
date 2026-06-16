//! MLB and MFB compact bilinear pooling fusion.
//!
//! Two variants of bilinear modality fusion:
//! - **MLB** (Multi-modal Low-rank Bilinear): `h = tanh(W_v·v) ⊙ tanh(W_q·q)` then project.
//! - **MFB** (Multi-modal Factorized Bilinear): expand features to k×d, Hadamard,
//!   sum-pool over k pairs, then tanh.

use crate::error::{MmResult, MultiModalError};

// ─── MLB ─────────────────────────────────────────────────────────────────────

/// Multi-modal Low-rank Bilinear (MLB) fusion.
///
/// `out = W_out · (tanh(W_v·v) ⊙ tanh(W_q·q))`
/// where `⊙` is element-wise product.
#[derive(Debug, Clone)]
pub struct MlbFusion {
    /// W_v: `[d_v × d_joint]`
    pub w_v: Vec<f32>,
    /// W_q: `[d_q × d_joint]`
    pub w_q: Vec<f32>,
    /// W_out: `[d_joint × d_out]`
    pub w_out: Vec<f32>,
    /// b_out: `[d_out]`
    pub b_out: Vec<f32>,
    pub d_v: usize,
    pub d_q: usize,
    pub d_joint: usize,
    pub d_out: usize,
}

impl MlbFusion {
    /// Create MLB with zero weights.
    #[must_use]
    pub fn zeros(d_v: usize, d_q: usize, d_joint: usize, d_out: usize) -> Self {
        Self {
            w_v: vec![0.0_f32; d_v * d_joint],
            w_q: vec![0.0_f32; d_q * d_joint],
            w_out: vec![0.0_f32; d_joint * d_out],
            b_out: vec![0.0_f32; d_out],
            d_v,
            d_q,
            d_joint,
            d_out,
        }
    }

    /// Forward pass on a single pair `(v [d_v], q [d_q])` → `[d_out]`.
    pub fn forward_single(&self, v: &[f32], q: &[f32]) -> MmResult<Vec<f32>> {
        if v.len() != self.d_v {
            return Err(MultiModalError::DimensionMismatch {
                expected: self.d_v,
                got: v.len(),
            });
        }
        if q.len() != self.d_q {
            return Err(MultiModalError::DimensionMismatch {
                expected: self.d_q,
                got: q.len(),
            });
        }

        // h_v = tanh(W_v · v)  [d_joint]
        let mut h_v = vec![0.0_f32; self.d_joint];
        for j in 0..self.d_joint {
            let mut acc = 0.0_f32;
            for i in 0..self.d_v {
                acc += v[i] * self.w_v[i * self.d_joint + j];
            }
            h_v[j] = acc.tanh();
        }

        // h_q = tanh(W_q · q)  [d_joint]
        let mut h_q = vec![0.0_f32; self.d_joint];
        for j in 0..self.d_joint {
            let mut acc = 0.0_f32;
            for i in 0..self.d_q {
                acc += q[i] * self.w_q[i * self.d_joint + j];
            }
            h_q[j] = acc.tanh();
        }

        // h = h_v ⊙ h_q  [d_joint]
        let h: Vec<f32> = h_v.iter().zip(h_q.iter()).map(|(a, b)| a * b).collect();

        // out = W_out · h + b_out  [d_out]
        let mut out = self.b_out.clone();
        for o in 0..self.d_out {
            for j in 0..self.d_joint {
                out[o] += h[j] * self.w_out[j * self.d_out + o];
            }
        }
        Ok(out)
    }

    /// Batched forward: `v [batch × d_v]`, `q [batch × d_q]` → `[batch × d_out]`.
    pub fn forward(&self, v: &[f32], q: &[f32], batch: usize) -> MmResult<Vec<f32>> {
        if v.len() != batch * self.d_v {
            return Err(MultiModalError::DimensionMismatch {
                expected: batch * self.d_v,
                got: v.len(),
            });
        }
        if q.len() != batch * self.d_q {
            return Err(MultiModalError::DimensionMismatch {
                expected: batch * self.d_q,
                got: q.len(),
            });
        }
        let mut out = Vec::with_capacity(batch * self.d_out);
        for bi in 0..batch {
            let v_i = &v[bi * self.d_v..(bi + 1) * self.d_v];
            let q_i = &q[bi * self.d_q..(bi + 1) * self.d_q];
            let o_i = self.forward_single(v_i, q_i)?;
            out.extend_from_slice(&o_i);
        }
        Ok(out)
    }
}

// ─── MFB ─────────────────────────────────────────────────────────────────────

/// Multi-modal Factorized Bilinear (MFB) fusion.
///
/// Each modality is projected to `[k_factor × d_out]`, Hadamard-multiplied,
/// then sum-pooled over the `k_factor` dimension, followed by `tanh`.
///
/// `h_v = (W_v · v).reshape(k, d_out)`
/// `h_q = (W_q · q).reshape(k, d_out)`
/// `out = tanh( sum_k( h_v[k] ⊙ h_q[k] ) )`
#[derive(Debug, Clone)]
pub struct MfbFusion {
    /// W_v: `[d_v × (k_factor * d_out)]`
    pub w_v: Vec<f32>,
    /// W_q: `[d_q × (k_factor * d_out)]`
    pub w_q: Vec<f32>,
    pub d_v: usize,
    pub d_q: usize,
    pub k_factor: usize,
    pub d_out: usize,
}

impl MfbFusion {
    /// Create MFB with zero weights.
    pub fn zeros(d_v: usize, d_q: usize, k_factor: usize, d_out: usize) -> MmResult<Self> {
        if k_factor == 0 {
            return Err(MultiModalError::InvalidKFactor { k_factor });
        }
        let inner = k_factor * d_out;
        Ok(Self {
            w_v: vec![0.0_f32; d_v * inner],
            w_q: vec![0.0_f32; d_q * inner],
            d_v,
            d_q,
            k_factor,
            d_out,
        })
    }

    /// Forward on a single pair `(v [d_v], q [d_q])` → `[d_out]`.
    pub fn forward_single(&self, v: &[f32], q: &[f32]) -> MmResult<Vec<f32>> {
        if v.len() != self.d_v {
            return Err(MultiModalError::DimensionMismatch {
                expected: self.d_v,
                got: v.len(),
            });
        }
        if q.len() != self.d_q {
            return Err(MultiModalError::DimensionMismatch {
                expected: self.d_q,
                got: q.len(),
            });
        }

        let inner = self.k_factor * self.d_out;

        // h_v = W_v · v  [k_factor * d_out]
        let mut h_v = vec![0.0_f32; inner];
        for i in 0..inner {
            for di in 0..self.d_v {
                h_v[i] += v[di] * self.w_v[di * inner + i];
            }
        }

        // h_q = W_q · q  [k_factor * d_out]
        let mut h_q = vec![0.0_f32; inner];
        for i in 0..inner {
            for di in 0..self.d_q {
                h_q[i] += q[di] * self.w_q[di * inner + i];
            }
        }

        // sum-pool over k dimension: out[d] = sum_k h_v[k,d] * h_q[k,d]
        let mut out = vec![0.0_f32; self.d_out];
        for k in 0..self.k_factor {
            for d in 0..self.d_out {
                out[d] += h_v[k * self.d_out + d] * h_q[k * self.d_out + d];
            }
        }

        // tanh activation
        for v in out.iter_mut() {
            *v = v.tanh();
        }

        Ok(out)
    }

    /// Batched forward: `v [batch × d_v]`, `q [batch × d_q]` → `[batch × d_out]`.
    pub fn forward(&self, v: &[f32], q: &[f32], batch: usize) -> MmResult<Vec<f32>> {
        if v.len() != batch * self.d_v {
            return Err(MultiModalError::DimensionMismatch {
                expected: batch * self.d_v,
                got: v.len(),
            });
        }
        if q.len() != batch * self.d_q {
            return Err(MultiModalError::DimensionMismatch {
                expected: batch * self.d_q,
                got: q.len(),
            });
        }
        let mut out = Vec::with_capacity(batch * self.d_out);
        for bi in 0..batch {
            let v_i = &v[bi * self.d_v..(bi + 1) * self.d_v];
            let q_i = &q[bi * self.d_q..(bi + 1) * self.d_q];
            let o_i = self.forward_single(v_i, q_i)?;
            out.extend_from_slice(&o_i);
        }
        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── MLB ──────────────────────────────────────────────────────────────────

    #[test]
    fn mlb_output_shape() {
        let f = MlbFusion::zeros(16, 16, 32, 8);
        let v = vec![0.5_f32; 4 * 16];
        let q = vec![0.3_f32; 4 * 16];
        let out = f.forward(&v, &q, 4).expect("forward should succeed");
        assert_eq!(out.len(), 4 * 8);
    }

    #[test]
    fn mlb_output_finite() {
        let mut f = MlbFusion::zeros(8, 8, 16, 4);
        // Give non-zero weights
        for (i, w) in f.w_v.iter_mut().enumerate() {
            *w = (i as f32 * 0.1).sin() * 0.1;
        }
        for (i, w) in f.w_q.iter_mut().enumerate() {
            *w = (i as f32 * 0.13).cos() * 0.1;
        }
        for (i, w) in f.w_out.iter_mut().enumerate() {
            *w = (i as f32 * 0.07).sin() * 0.1;
        }
        let v = vec![1.0_f32; 2 * 8];
        let q = vec![1.0_f32; 2 * 8];
        let out = f.forward(&v, &q, 2).expect("forward should succeed");
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn mlb_nonzero_input_nonzero_output() {
        let mut f = MlbFusion::zeros(4, 4, 8, 4);
        // identity-like w_v and w_q
        for i in 0..4 {
            for j in 0..8 {
                f.w_v[i * 8 + j] = if i == j % 4 { 1.0 } else { 0.0 };
                f.w_q[i * 8 + j] = if i == j % 4 { 1.0 } else { 0.0 };
            }
        }
        // w_out: sum all joints into each output
        for w in f.w_out.iter_mut() {
            *w = 0.1;
        }
        let v = vec![1.0_f32; 4];
        let q = vec![1.0_f32; 4];
        let out = f
            .forward_single(&v, &q)
            .expect("forward_single should succeed");
        // tanh(1)^2 * 0.1 * 8 joints = non-zero
        let total: f32 = out.iter().sum();
        assert!(total.abs() > 1e-6, "expected non-zero output: {total}");
    }

    #[test]
    fn mlb_dimension_mismatch() {
        let f = MlbFusion::zeros(4, 8, 16, 4);
        let v = vec![0.0_f32; 5]; // wrong
        let q = vec![0.0_f32; 8];
        let err = f.forward_single(&v, &q).unwrap_err();
        assert!(matches!(err, MultiModalError::DimensionMismatch { .. }));
    }

    // ── MFB ──────────────────────────────────────────────────────────────────

    #[test]
    fn mfb_output_shape() {
        let f = MfbFusion::zeros(16, 16, 4, 8).expect("zeros should succeed");
        let v = vec![0.5_f32; 3 * 16];
        let q = vec![0.3_f32; 3 * 16];
        let out = f.forward(&v, &q, 3).expect("forward should succeed");
        assert_eq!(out.len(), 3 * 8);
    }

    #[test]
    fn mfb_output_finite() {
        let mut f = MfbFusion::zeros(8, 8, 3, 4).expect("zeros should succeed");
        let inner = 3 * 4;
        for (i, w) in f.w_v.iter_mut().enumerate() {
            *w = (i as f32 * 0.07).sin() * 0.1;
        }
        for (i, w) in f.w_q.iter_mut().enumerate() {
            *w = (i as f32 * 0.13).cos() * 0.1;
        }
        let _ = inner; // suppress warning
        let v = vec![1.0_f32; 2 * 8];
        let q = vec![1.0_f32; 2 * 8];
        let out = f.forward(&v, &q, 2).expect("forward should succeed");
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn mfb_invalid_k_factor() {
        let err = MfbFusion::zeros(4, 4, 0, 8).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidKFactor { .. }));
    }

    #[test]
    fn mfb_tanh_bounds() {
        let f = MfbFusion::zeros(4, 4, 2, 4).expect("zeros should succeed");
        let v = vec![100.0_f32; 4]; // large values
        let q = vec![100.0_f32; 4];
        let out = f
            .forward_single(&v, &q)
            .expect("forward_single should succeed");
        // tanh of any value is in (-1, 1)
        for &x in &out {
            assert!(x.abs() <= 1.0 + 1e-6, "tanh output out of range: {x}");
        }
    }

    #[test]
    fn mfb_zero_weights_zero_output() {
        let f = MfbFusion::zeros(4, 4, 2, 4).expect("zeros should succeed");
        let v = vec![1.0_f32; 4];
        let q = vec![1.0_f32; 4];
        let out = f
            .forward_single(&v, &q)
            .expect("forward_single should succeed");
        // tanh(0*...) = 0
        for &x in &out {
            assert!(x.abs() < 1e-6);
        }
    }
}
