//! Internal neural-network primitives shared by the generative / feature-selection
//! tabular models (CTGAN, TVAE, STG).
//!
//! These are deliberately small, allocation-friendly CPU helpers: dense layers
//! with row-major weights, ReLU / leaky-ReLU activations, numerically-stable
//! softmax / log-softmax, and Glorot-uniform initialisation driven by the
//! crate's [`LcgRng`].  They are `pub(crate)` and not part of the public API.

use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;

/// Fill `w` with Glorot/Xavier-uniform samples for a layer of the given fan-in
/// and fan-out.  The bound is `sqrt(6 / (fan_in + fan_out))`.
///
/// Uses [`LcgRng::next_f32`], which yields a genuine uniform on `[0, 1)`, mapped
/// onto the symmetric interval `[-limit, limit)`.
pub(crate) fn glorot_uniform(w: &mut [f32], fan_in: usize, fan_out: usize, rng: &mut LcgRng) {
    let denom = (fan_in + fan_out).max(1) as f32;
    let limit = (6.0_f32 / denom).sqrt();
    for v in w.iter_mut() {
        let u = rng.next_f32(); // uniform on [0, 1)
        *v = 2.0 * limit * u - limit;
    }
}

/// Rectified linear unit.
#[inline]
pub(crate) fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Leaky rectified linear unit with the given negative slope.
#[inline]
pub(crate) fn leaky_relu(x: f32, slope: f32) -> f32 {
    if x >= 0.0 { x } else { slope * x }
}

/// Numerically-stable softmax over a logit vector.
///
/// Returns a vector of the same length whose entries are non-negative and sum to
/// (approximately) one.  An empty input yields an empty output.
pub(crate) fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for e in &mut exps {
        *e *= inv;
    }
    exps
}

/// Numerically-stable log-softmax over a logit vector.
pub(crate) fn log_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    let log_sum = max + sum.max(1e-30).ln();
    logits.iter().map(|&l| l - log_sum).collect()
}

/// A dense (fully-connected) layer with row-major weights `[out_dim × in_dim]`
/// and a bias of length `out_dim`.
#[derive(Debug, Clone)]
pub(crate) struct Dense {
    /// Flattened weight matrix, row-major `[out_dim × in_dim]`.
    pub(crate) w: Vec<f32>,
    /// Bias vector, length `out_dim`.
    pub(crate) b: Vec<f32>,
    /// Number of input units.
    pub(crate) in_dim: usize,
    /// Number of output units.
    pub(crate) out_dim: usize,
}

impl Dense {
    /// Construct a dense layer with Glorot-uniform weights and zero bias.
    pub(crate) fn new(in_dim: usize, out_dim: usize, rng: &mut LcgRng) -> Self {
        let mut w = vec![0.0_f32; in_dim * out_dim];
        glorot_uniform(&mut w, in_dim, out_dim, rng);
        Self {
            w,
            b: vec![0.0_f32; out_dim],
            in_dim,
            out_dim,
        }
    }

    /// Affine forward pass `y = W·x + b`.
    ///
    /// Missing input entries (if `x` is shorter than `in_dim`) are treated as
    /// zero, so the call never panics on a short slice.
    pub(crate) fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0_f32; self.out_dim];
        for (o, oy) in out.iter_mut().enumerate() {
            let base = o * self.in_dim;
            let mut acc = self.b[o];
            for i in 0..self.in_dim {
                acc += self.w[base + i] * x.get(i).copied().unwrap_or(0.0);
            }
            *oy = acc;
        }
        out
    }
}

/// A simple multilayer perceptron: a stack of [`Dense`] layers with ReLU applied
/// to every hidden layer and a linear final layer.
#[derive(Debug, Clone)]
pub(crate) struct Mlp {
    layers: Vec<Dense>,
}

impl Mlp {
    /// Build an MLP from a list of layer dimensions.  `dims = [in, h1, …, out]`
    /// produces `dims.len() - 1` dense layers.
    ///
    /// # Errors
    /// Returns [`TabularError::InvalidParameter`] if fewer than two dimensions
    /// are supplied (a layer requires both an input and an output size).
    pub(crate) fn new(dims: &[usize], rng: &mut LcgRng) -> TabularResult<Self> {
        if dims.len() < 2 {
            return Err(TabularError::InvalidParameter {
                name: "dims".into(),
                msg: "an MLP needs at least an input and an output dimension".into(),
            });
        }
        let mut layers = Vec::with_capacity(dims.len() - 1);
        for pair in dims.windows(2) {
            layers.push(Dense::new(pair[0], pair[1], rng));
        }
        Ok(Self { layers })
    }

    /// Forward pass with ReLU on the hidden layers and a linear output layer.
    pub(crate) fn forward(&self, x: &[f32]) -> Vec<f32> {
        let n = self.layers.len();
        let mut cur = x.to_vec();
        for (idx, layer) in self.layers.iter().enumerate() {
            let mut z = layer.forward(&cur);
            if idx + 1 < n {
                for v in &mut z {
                    *v = relu(*v);
                }
            }
            cur = z;
        }
        cur
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glorot_uniform_in_bounds() {
        let mut rng = LcgRng::new(1);
        let mut w = vec![0.0_f32; 200];
        glorot_uniform(&mut w, 10, 10, &mut rng);
        let limit = (6.0_f32 / 20.0).sqrt();
        assert!(w.iter().all(|&v| v >= -limit && v < limit + 1e-6));
        // Not all identical.
        assert!(w.windows(2).any(|p| (p[0] - p[1]).abs() > 1e-6));
    }

    #[test]
    fn relu_and_leaky() {
        assert_eq!(relu(2.0), 2.0);
        assert_eq!(relu(-2.0), 0.0);
        assert_eq!(leaky_relu(2.0, 0.2), 2.0);
        assert!((leaky_relu(-2.0, 0.2) - (-0.4)).abs() < 1e-6);
    }

    #[test]
    fn softmax_sums_to_one() {
        let s = softmax(&[1.0, 2.0, 3.0, -1.0]);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(s.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn softmax_empty() {
        assert!(softmax(&[]).is_empty());
    }

    #[test]
    fn log_softmax_matches_log_of_softmax() {
        let logits = [0.3_f32, -1.2, 2.0];
        let sm = softmax(&logits);
        let lsm = log_softmax(&logits);
        for (p, lp) in sm.iter().zip(lsm.iter()) {
            assert!((p.ln() - lp).abs() < 1e-5);
        }
    }

    #[test]
    fn dense_forward_shape_and_value() {
        let mut rng = LcgRng::new(7);
        let mut layer = Dense::new(3, 2, &mut rng);
        layer.w = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        layer.b = vec![0.5, -0.5];
        let y = layer.forward(&[2.0, 3.0, 4.0]);
        assert_eq!(y.len(), 2);
        assert!((y[0] - 2.5).abs() < 1e-6);
        assert!((y[1] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn mlp_forward_shape() {
        let mut rng = LcgRng::new(11);
        let mlp = Mlp::new(&[4, 8, 3], &mut rng).expect("new should succeed");
        let y = mlp.forward(&[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(y.len(), 3);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mlp_requires_two_dims() {
        let mut rng = LcgRng::new(1);
        assert!(Mlp::new(&[4], &mut rng).is_err());
    }
}
