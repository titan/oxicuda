//! Random Feedback Alignment (FA) and Direct Feedback Alignment (DFA).
//!
//! Standard backpropagation transports the error signal through the **transpose**
//! of the forward weight matrix, `δ_in = Wᵀ · δ_out`. This "weight transport"
//! requires the backward pass to read the exact forward weights, which is
//! biologically implausible and couples the two passes. Feedback Alignment
//! (Lillicrap, Cownden, Tweed & Akerman, *Nature Communications* 2016) shows
//! that learning still succeeds when `Wᵀ` is replaced by a **fixed random**
//! matrix `B` that is drawn once and never trained:
//!
//! ```text
//! forward:   a = W · x                       (W is [out × in], row-major)
//! backward:  δ_in = B · δ_out                (B is [in × out], frozen random)
//! update:    ΔW = −lr · (δ_out ⊗ x)          (outer product, standard rule)
//! ```
//!
//! Crucially `B` is **not** the transpose of `W`: it is initialised from the
//! RNG at construction time and then frozen. Over training the forward weights
//! self-organise so that `W` partially aligns with `Bᵀ`, making `B·δ` a useful
//! descent direction even though it is random.
//!
//! Direct Feedback Alignment (Nøkland, *NeurIPS* 2016) goes further: instead of
//! propagating the error layer-by-layer, a single **global** error vector `e`
//! (typically the output-layer error) is projected directly into every hidden
//! layer through that layer's own fixed random matrix:
//!
//! ```text
//! δ_layer = B · e.
//! ```
//!
//! Here `B` has shape `[in × out]` where `out` is the dimensionality of the
//! global error, so [`crate::training::feedback_alignment::FeedbackAlignment::dfa_project`] reuses the same frozen
//! matrix.

#![allow(clippy::needless_range_loop)]

use crate::error::{SnnError, SnnResult};
use crate::handle::LcgRng;

/// A single feedback-aligned linear layer.
///
/// Holds the trainable forward weights `W` (`[out_dim × in_dim]`, row-major) and
/// the frozen random feedback matrix `B` (`[in_dim × out_dim]`, row-major). The
/// feedback matrix is drawn once in [`FeedbackAlignment::new`] and is never
/// modified by any method on this type.
#[derive(Debug, Clone)]
pub struct FeedbackAlignment {
    /// Number of input features.
    in_dim: usize,
    /// Number of output features.
    out_dim: usize,
    /// Forward weights `W`, shape `[out_dim × in_dim]`, row-major.
    weights: Vec<f32>,
    /// Frozen random feedback matrix `B`, shape `[in_dim × out_dim]`, row-major.
    feedback: Vec<f32>,
}

impl FeedbackAlignment {
    /// Construct a layer with random forward weights and a frozen random
    /// feedback matrix, both drawn from `rng`.
    ///
    /// Forward weights are scaled by `1/√in_dim` (LeCun-style fan-in) and the
    /// feedback matrix by `1/√out_dim`, keeping the propagated signals at unit
    /// scale. The two matrices are statistically independent draws — `B` is
    /// **not** the transpose of `W`.
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::BadDim`] if either dimension is zero.
    pub fn new(in_dim: usize, out_dim: usize, rng: &mut LcgRng) -> SnnResult<Self> {
        if in_dim == 0 {
            return Err(SnnError::BadDim { got: in_dim });
        }
        if out_dim == 0 {
            return Err(SnnError::BadDim { got: out_dim });
        }
        let w_scale = 1.0 / (in_dim as f32).sqrt();
        let b_scale = 1.0 / (out_dim as f32).sqrt();
        let weights: Vec<f32> = (0..in_dim * out_dim)
            .map(|_| rng.next_normal() * w_scale)
            .collect();
        let feedback: Vec<f32> = (0..in_dim * out_dim)
            .map(|_| rng.next_normal() * b_scale)
            .collect();
        Ok(Self {
            in_dim,
            out_dim,
            weights,
            feedback,
        })
    }

    /// Number of input features.
    #[must_use]
    #[inline]
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// Number of output features.
    #[must_use]
    #[inline]
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// Immutable view of the forward weights `W` (`[out_dim × in_dim]`).
    #[must_use]
    #[inline]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Immutable view of the frozen feedback matrix `B` (`[in_dim × out_dim]`).
    #[must_use]
    #[inline]
    pub fn feedback(&self) -> &[f32] {
        &self.feedback
    }

    /// Forward pass `a = W · x`, returning the pre-activation vector.
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::EmptyInput`] if `x` is empty, or
    /// [`SnnError::IncompatibleLength`] if `x.len() != in_dim`.
    pub fn forward(&self, x: &[f32]) -> SnnResult<Vec<f32>> {
        if x.is_empty() {
            return Err(SnnError::EmptyInput);
        }
        if x.len() != self.in_dim {
            return Err(SnnError::IncompatibleLength {
                a: x.len(),
                b: self.in_dim,
            });
        }
        let mut out = vec![0.0_f32; self.out_dim];
        for (i, o) in out.iter_mut().enumerate() {
            let row = &self.weights[i * self.in_dim..(i + 1) * self.in_dim];
            *o = row.iter().zip(x.iter()).map(|(&w, &xi)| w * xi).sum();
        }
        Ok(out)
    }

    /// Feedback-aligned backward pass `δ_in = B · δ_out`.
    ///
    /// Replaces the standard `Wᵀ · δ_out` with the frozen random feedback
    /// matrix. The result has length `in_dim`.
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::EmptyInput`] if `delta_out` is empty, or
    /// [`SnnError::IncompatibleLength`] if `delta_out.len() != out_dim`.
    pub fn backward_error(&self, delta_out: &[f32]) -> SnnResult<Vec<f32>> {
        if delta_out.is_empty() {
            return Err(SnnError::EmptyInput);
        }
        if delta_out.len() != self.out_dim {
            return Err(SnnError::IncompatibleLength {
                a: delta_out.len(),
                b: self.out_dim,
            });
        }
        // B is [in_dim × out_dim]; row k dotted with δ_out gives δ_in[k].
        let mut delta_in = vec![0.0_f32; self.in_dim];
        for (k, d) in delta_in.iter_mut().enumerate() {
            let row = &self.feedback[k * self.out_dim..(k + 1) * self.out_dim];
            *d = row.iter().zip(delta_out.iter()).map(|(&b, &e)| b * e).sum();
        }
        Ok(delta_in)
    }

    /// Direct Feedback Alignment: project a global error directly into this
    /// layer's delta via the frozen random matrix, `δ_layer = B · e`.
    ///
    /// Functionally identical linear algebra to [`Self::backward_error`], but the
    /// distinct name documents DFA intent: `global_error` is the network's single
    /// output-layer error rather than the next layer's back-propagated delta.
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::EmptyInput`] if `global_error` is empty, or
    /// [`SnnError::IncompatibleLength`] if `global_error.len() != out_dim`.
    pub fn dfa_project(&self, global_error: &[f32]) -> SnnResult<Vec<f32>> {
        self.backward_error(global_error)
    }

    /// Apply the standard outer-product weight update `ΔW = −lr · (δ_out ⊗ x)`.
    ///
    /// Only the forward weights `W` are modified; the feedback matrix `B` is left
    /// untouched (it is frozen for the lifetime of the layer).
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::IncompatibleLength`] if `x.len() != in_dim` or
    /// `delta_out.len() != out_dim`, or [`SnnError::OutOfRange`] if `lr` is
    /// non-finite.
    pub fn weight_update(&mut self, x: &[f32], delta_out: &[f32], lr: f32) -> SnnResult<()> {
        if x.len() != self.in_dim {
            return Err(SnnError::IncompatibleLength {
                a: x.len(),
                b: self.in_dim,
            });
        }
        if delta_out.len() != self.out_dim {
            return Err(SnnError::IncompatibleLength {
                a: delta_out.len(),
                b: self.out_dim,
            });
        }
        if !lr.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "lr".into(),
                val: lr,
            });
        }
        for (i, &d) in delta_out.iter().enumerate() {
            let row = &mut self.weights[i * self.in_dim..(i + 1) * self.in_dim];
            for (w, &xi) in row.iter_mut().zip(x.iter()) {
                *w -= lr * d * xi;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_rejects_zero_dims() {
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            FeedbackAlignment::new(0, 4, &mut rng),
            Err(SnnError::BadDim { .. })
        ));
        assert!(matches!(
            FeedbackAlignment::new(4, 0, &mut rng),
            Err(SnnError::BadDim { .. })
        ));
    }

    #[test]
    fn shapes_are_correct() {
        let mut rng = LcgRng::new(7);
        let fa = FeedbackAlignment::new(5, 3, &mut rng).expect("ok");
        assert_eq!(fa.weights().len(), 5 * 3);
        assert_eq!(fa.feedback().len(), 5 * 3);
        assert_eq!(fa.in_dim(), 5);
        assert_eq!(fa.out_dim(), 3);
    }

    #[test]
    fn feedback_is_not_weight_transpose() {
        // B is an independent random draw, so it must not equal Wᵀ.
        let mut rng = LcgRng::new(123);
        let fa = FeedbackAlignment::new(4, 3, &mut rng).expect("ok");
        let mut equal = true;
        for k in 0..fa.in_dim() {
            for j in 0..fa.out_dim() {
                let b_kj = fa.feedback()[k * fa.out_dim() + j];
                let w_jk = fa.weights()[j * fa.in_dim() + k];
                if (b_kj - w_jk).abs() > 1e-9 {
                    equal = false;
                }
            }
        }
        assert!(!equal, "B must not be the transpose of W");
    }

    #[test]
    fn forward_matches_manual_matmul() {
        let mut rng = LcgRng::new(9);
        let fa = FeedbackAlignment::new(3, 2, &mut rng).expect("ok");
        let x = vec![1.0_f32, -2.0, 0.5];
        let y = fa.forward(&x).expect("ok");
        for i in 0..fa.out_dim() {
            let mut acc = 0.0_f32;
            for j in 0..fa.in_dim() {
                acc += fa.weights()[i * fa.in_dim() + j] * x[j];
            }
            assert!((y[i] - acc).abs() < 1e-6, "{} vs {acc}", y[i]);
        }
    }

    #[test]
    fn backward_error_uses_feedback_matrix() {
        let mut rng = LcgRng::new(55);
        let fa = FeedbackAlignment::new(3, 2, &mut rng).expect("ok");
        let delta_out = vec![0.5_f32, -1.0];
        let delta_in = fa.backward_error(&delta_out).expect("ok");
        assert_eq!(delta_in.len(), 3);
        for k in 0..fa.in_dim() {
            let mut acc = 0.0_f32;
            for j in 0..fa.out_dim() {
                acc += fa.feedback()[k * fa.out_dim() + j] * delta_out[j];
            }
            assert!((delta_in[k] - acc).abs() < 1e-6, "{} vs {acc}", delta_in[k]);
        }
    }

    #[test]
    fn dfa_project_equals_backward_error() {
        let mut rng = LcgRng::new(77);
        let fa = FeedbackAlignment::new(4, 3, &mut rng).expect("ok");
        let e = vec![0.2_f32, -0.4, 0.1];
        let a = fa.backward_error(&e).expect("ok");
        let b = fa.dfa_project(&e).expect("ok");
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    #[test]
    fn feedback_frozen_after_updates() {
        // B must be byte-identical after a sequence of weight updates.
        let mut rng = LcgRng::new(2024);
        let mut fa = FeedbackAlignment::new(4, 3, &mut rng).expect("ok");
        let b_before = fa.feedback().to_vec();
        let x = vec![0.3_f32, -0.2, 0.7, 0.1];
        let delta = vec![0.5_f32, -0.3, 0.2];
        for _ in 0..20 {
            fa.weight_update(&x, &delta, 0.05).expect("ok");
        }
        for (a, b) in fa.feedback().iter().zip(b_before.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "feedback matrix changed");
        }
    }

    #[test]
    fn weight_update_changes_weights() {
        let mut rng = LcgRng::new(33);
        let mut fa = FeedbackAlignment::new(2, 2, &mut rng).expect("ok");
        let w_before = fa.weights().to_vec();
        let x = vec![1.0_f32, 1.0];
        let delta = vec![1.0_f32, 1.0];
        fa.weight_update(&x, &delta, 0.1).expect("ok");
        let mut changed = false;
        for (a, b) in fa.weights().iter().zip(w_before.iter()) {
            if (a - b).abs() > 1e-9 {
                changed = true;
            }
        }
        assert!(changed, "weights should change after update");
    }

    #[test]
    fn two_layer_fa_reduces_linear_mse() {
        // A 2-layer linear network trained with Feedback Alignment must reduce
        // MSE on a fixed linear target map. Seeded → fully deterministic.
        //
        // net:  h = W1 · x   (hidden, identity activation)
        //       o = W2 · h   (output)
        // target: y* = T · x for a fixed random T.
        let mut rng = LcgRng::new(20260620);
        let in_dim = 4_usize;
        let hid_dim = 6_usize;
        let out_dim = 3_usize;

        let mut l1 = FeedbackAlignment::new(in_dim, hid_dim, &mut rng).expect("ok");
        let mut l2 = FeedbackAlignment::new(hid_dim, out_dim, &mut rng).expect("ok");

        // Fixed linear teacher map T : R^in → R^out.
        let teacher: Vec<f32> = (0..out_dim * in_dim).map(|_| rng.next_normal()).collect();

        // Fixed training batch.
        let batch: Vec<Vec<f32>> = (0..16)
            .map(|_| (0..in_dim).map(|_| rng.next_normal()).collect::<Vec<f32>>())
            .collect();

        let teacher_apply = |x: &[f32]| -> Vec<f32> {
            (0..out_dim)
                .map(|i| {
                    (0..in_dim)
                        .map(|j| teacher[i * in_dim + j] * x[j])
                        .sum::<f32>()
                })
                .collect()
        };

        let epoch_mse = |l1: &FeedbackAlignment, l2: &FeedbackAlignment| -> f32 {
            let mut acc = 0.0_f32;
            let mut count = 0usize;
            for x in &batch {
                let h = l1.forward(x).expect("ok");
                let o = l2.forward(&h).expect("ok");
                let y = teacher_apply(x);
                for (oi, yi) in o.iter().zip(y.iter()) {
                    let d = oi - yi;
                    acc += d * d;
                    count += 1;
                }
            }
            acc / count as f32
        };

        let mse_start = epoch_mse(&l1, &l2);
        let lr = 0.02_f32;
        for _ in 0..400 {
            for x in &batch {
                let h = l1.forward(x).expect("ok");
                let o = l2.forward(&h).expect("ok");
                let y = teacher_apply(x);
                // Output error δ2 = (o − y*) (MSE gradient, identity output).
                let delta2: Vec<f32> = o.iter().zip(y.iter()).map(|(&oi, &yi)| oi - yi).collect();
                // Feedback-aligned hidden error δ1 = B2 · δ2 (NOT W2ᵀ).
                let delta1 = l2.backward_error(&delta2).expect("ok");
                // Updates (identity activations → no derivative gating).
                l2.weight_update(&h, &delta2, lr).expect("ok");
                l1.weight_update(x, &delta1, lr).expect("ok");
            }
        }
        let mse_end = epoch_mse(&l1, &l2);
        assert!(
            mse_end < mse_start,
            "FA failed to reduce MSE: start={mse_start} end={mse_end}"
        );
        // Demand a substantial reduction, not just any decrease.
        assert!(
            mse_end < 0.5 * mse_start,
            "FA reduction too small: start={mse_start} end={mse_end}"
        );
    }

    #[test]
    fn forward_rejects_bad_length() {
        let mut rng = LcgRng::new(3);
        let fa = FeedbackAlignment::new(3, 2, &mut rng).expect("ok");
        assert!(matches!(fa.forward(&[]), Err(SnnError::EmptyInput)));
        assert!(matches!(
            fa.forward(&[1.0_f32, 2.0]),
            Err(SnnError::IncompatibleLength { .. })
        ));
    }

    #[test]
    fn weight_update_rejects_bad_lengths() {
        let mut rng = LcgRng::new(4);
        let mut fa = FeedbackAlignment::new(3, 2, &mut rng).expect("ok");
        let bad_x = vec![1.0_f32, 2.0];
        let delta = vec![1.0_f32, 1.0];
        assert!(matches!(
            fa.weight_update(&bad_x, &delta, 0.1),
            Err(SnnError::IncompatibleLength { .. })
        ));
    }

    #[test]
    fn construction_is_deterministic() {
        let mut a = LcgRng::new(999);
        let mut b = LcgRng::new(999);
        let fa = FeedbackAlignment::new(4, 3, &mut a).expect("ok");
        let fb = FeedbackAlignment::new(4, 3, &mut b).expect("ok");
        for (x, y) in fa.weights().iter().zip(fb.weights().iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        for (x, y) in fa.feedback().iter().zip(fb.feedback().iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }
}
