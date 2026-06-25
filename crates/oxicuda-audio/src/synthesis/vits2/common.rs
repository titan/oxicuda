//! Shared numeric primitives for the VITS2 submodule.
//!
//! These mirror the private helpers used by [`crate::synthesis::fastspeech2`]
//! (`matmul`, `transpose_2d`, `linear`, `conv1d_same`, `make_normal_vec`) so
//! that the flow / posterior / duration components share a single, audited set
//! of dense-linear and convolution kernels. Everything is flat row-major
//! `[rows, cols]` and `f32`, matching the rest of `oxicuda-audio`.

use crate::handle::LcgRng;

/// Dense matrix multiply `C = A · B` where `A` is `[m, k]`, `B` is `[k, n]`.
///
/// Inputs and output are flat row-major.
pub(crate) fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            let b_row = &b[p * n..p * n + n];
            let c_row = &mut c[i * n..i * n + n];
            for (cj, &bj) in c_row.iter_mut().zip(b_row.iter()) {
                *cj += a_ip * bj;
            }
        }
    }
    c
}

/// Transpose a `[rows, cols]` flat matrix to `[cols, rows]`.
pub(crate) fn transpose_2d(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = a[r * cols + c];
        }
    }
    out
}

/// Apply a linear projection `y = x · wᵀ + b` where `w` is `[d_out, d_in]`.
///
/// `x` is `[t, d_in]`; the result is `[t, d_out]`.
pub(crate) fn linear(
    x: &[f32],
    t: usize,
    d_in: usize,
    d_out: usize,
    w: &[f32],
    b: &[f32],
) -> Vec<f32> {
    let w_t = transpose_2d(w, d_out, d_in);
    let mut out = matmul(x, &w_t, t, d_in, d_out);
    for ti in 0..t {
        let row = &mut out[ti * d_out..ti * d_out + d_out];
        for (o, &bias) in row.iter_mut().zip(b.iter()) {
            *o += bias;
        }
    }
    out
}

/// Allocate a weight buffer of `sz` elements drawn from `N(0, 1)` scaled by
/// `scale` (deterministically via `rng`).
pub(crate) fn make_normal_vec(sz: usize, scale: f32, rng: &mut LcgRng) -> Vec<f32> {
    let mut w = vec![0.0_f32; sz];
    rng.fill_normal(&mut w);
    for v in w.iter_mut() {
        *v *= scale;
    }
    w
}

/// Rectified-linear activation, applied in place.
pub(crate) fn relu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// 1-D convolution with `same` padding over a `[t, channels_in]` sequence,
/// producing `[t, channels_out]`.
///
/// `weight` is laid out `[channels_out, channels_in, kernel]` (row-major) and
/// `bias` is `[channels_out]`. The kernel is centred (`padding = kernel / 2`);
/// out-of-range taps are treated as zero (zero padding). This is the
/// non-causal convolution used by the VITS posterior encoder.
pub(crate) fn conv1d_same(
    x: &[f32],
    t: usize,
    channels_in: usize,
    channels_out: usize,
    kernel: usize,
    weight: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let pad = kernel / 2;
    let mut out = vec![0.0_f32; t * channels_out];
    for ti in 0..t {
        for co in 0..channels_out {
            let mut acc = bias[co];
            for kk in 0..kernel {
                let src = ti as isize + kk as isize - pad as isize;
                if src < 0 || src >= t as isize {
                    continue;
                }
                let src = src as usize;
                let w_base = (co * channels_in) * kernel + kk;
                for ci in 0..channels_in {
                    acc += x[src * channels_in + ci] * weight[w_base + ci * kernel];
                }
            }
            out[ti * channels_out + co] = acc;
        }
    }
    out
}

/// A position-wise dense (fully-connected) layer `y = x · wᵀ + b`.
///
/// `w` is stored `[d_out, d_in]` row-major and `b` is `[d_out]`. Applying it to
/// a `[t, d_in]` sequence yields `[t, d_out]` (the same projection is shared
/// across every time step, i.e. a 1×1 convolution).
#[derive(Debug, Clone)]
pub(crate) struct DenseLayer {
    /// Weight matrix `[d_out, d_in]`.
    pub w: Vec<f32>,
    /// Bias vector `[d_out]`.
    pub b: Vec<f32>,
    /// Input feature dimension.
    pub d_in: usize,
    /// Output feature dimension.
    pub d_out: usize,
}

impl DenseLayer {
    /// Construct a dense layer with `N(0, scale²)` weights and zero bias.
    pub(crate) fn new(d_in: usize, d_out: usize, scale: f32, rng: &mut LcgRng) -> Self {
        Self {
            w: make_normal_vec(d_out * d_in, scale, rng),
            b: vec![0.0_f32; d_out],
            d_in,
            d_out,
        }
    }

    /// Apply the projection to `x` of shape `[t, d_in]`, returning `[t, d_out]`.
    pub(crate) fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        linear(x, t, self.d_in, self.d_out, &self.w, &self.b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_identity() {
        // [[1,2],[3,4]] · I = itself.
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let eye = vec![1.0_f32, 0.0, 0.0, 1.0];
        let c = matmul(&a, &eye, 2, 2, 2);
        assert_eq!(c, a);
    }

    #[test]
    fn transpose_roundtrip() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
        let at = transpose_2d(&a, 2, 3); // [3,2]
        let att = transpose_2d(&at, 3, 2); // back to [2,3]
        assert_eq!(att, a);
    }

    #[test]
    fn dense_layer_shape() {
        let mut rng = LcgRng::new(1);
        let layer = DenseLayer::new(4, 7, 0.1, &mut rng);
        let t = 5usize;
        let mut x = vec![0.0_f32; t * 4];
        rng.fill_normal(&mut x);
        let y = layer.forward(&x, t);
        assert_eq!(y.len(), t * 7);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn relu_inplace_clamps() {
        let mut v = vec![-2.0_f32, -0.1, 0.0, 0.5, 3.0];
        relu_inplace(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 0.0, 0.5, 3.0]);
    }

    #[test]
    fn conv1d_centre_tap_identity() {
        let t = 4;
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let weight = vec![0.0_f32, 1.0, 0.0]; // centre tap
        let bias = vec![0.0_f32];
        let out = conv1d_same(&x, t, 1, 1, 3, &weight, &bias);
        assert_eq!(out, x);
    }
}
