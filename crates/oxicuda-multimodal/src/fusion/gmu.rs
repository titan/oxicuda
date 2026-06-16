//! Gated Multimodal Unit (Arevalo et al. 2017).
//!
//! Implements the fusion cell from:
//! Arevalo et al. "Gated Multimodal Units for Information Fusion." ICLR 2017
//! (Workshop).
//!
//! For two modality inputs `x_a ∈ R^{d_a}` and `x_b ∈ R^{d_b}` the GMU computes
//! per-modality `tanh` feature transforms and a sigmoid gate `z` that decides, on
//! a per-feature basis, how much each modality contributes to the fused output:
//!
//! ```text
//! h_a = tanh(W_a · x_a)            (∈ R^{d_h})
//! h_b = tanh(W_b · x_b)            (∈ R^{d_h})
//! z   = σ( W_z · [x_a ; x_b] )      (∈ R^{d_h}, gate)
//! h   = z ⊙ h_a + (1 − z) ⊙ h_b
//! ```
//!
//! The gate is data-dependent and learned, so the network can route information
//! adaptively (e.g. trust the image for some features, the text for others).

use crate::error::{MmResult, MultiModalError};

/// Numerically-stable logistic sigmoid.
#[must_use]
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Gated Multimodal Unit fusing two modalities into a shared hidden space.
#[derive(Debug, Clone)]
pub struct GatedMultimodalUnit {
    /// `W_a`: `[d_a × d_h]` row-major.
    pub w_a: Vec<f32>,
    /// `W_b`: `[d_b × d_h]` row-major.
    pub w_b: Vec<f32>,
    /// `W_z`: `[(d_a + d_b) × d_h]` row-major (gate projection).
    pub w_z: Vec<f32>,
    /// Modality-A input dimension.
    pub d_a: usize,
    /// Modality-B input dimension.
    pub d_b: usize,
    /// Fused hidden dimension.
    pub d_h: usize,
}

impl GatedMultimodalUnit {
    /// Create a GMU with all-zero weights.
    #[must_use]
    pub fn zeros(d_a: usize, d_b: usize, d_h: usize) -> Self {
        Self {
            w_a: vec![0.0_f32; d_a * d_h],
            w_b: vec![0.0_f32; d_b * d_h],
            w_z: vec![0.0_f32; (d_a + d_b) * d_h],
            d_a,
            d_b,
            d_h,
        }
    }

    /// Forward pass on a single `(x_a [d_a], x_b [d_b])` pair → `(h [d_h], z [d_h])`.
    ///
    /// Returns the fused hidden vector and the gate activations.
    ///
    /// # Errors
    /// Returns [`MultiModalError::DimensionMismatch`] when an input length does
    /// not match its declared dimension.
    pub fn forward_single(&self, x_a: &[f32], x_b: &[f32]) -> MmResult<(Vec<f32>, Vec<f32>)> {
        if x_a.len() != self.d_a {
            return Err(MultiModalError::DimensionMismatch {
                expected: self.d_a,
                got: x_a.len(),
            });
        }
        if x_b.len() != self.d_b {
            return Err(MultiModalError::DimensionMismatch {
                expected: self.d_b,
                got: x_b.len(),
            });
        }

        // h_a = tanh(W_a · x_a)
        let mut h_a = vec![0.0_f32; self.d_h];
        for (j, slot) in h_a.iter_mut().enumerate() {
            let mut acc = 0.0_f32;
            for i in 0..self.d_a {
                acc += x_a[i] * self.w_a[i * self.d_h + j];
            }
            *slot = acc.tanh();
        }

        // h_b = tanh(W_b · x_b)
        let mut h_b = vec![0.0_f32; self.d_h];
        for (j, slot) in h_b.iter_mut().enumerate() {
            let mut acc = 0.0_f32;
            for i in 0..self.d_b {
                acc += x_b[i] * self.w_b[i * self.d_h + j];
            }
            *slot = acc.tanh();
        }

        // z = σ(W_z · [x_a ; x_b])  (gate over the concatenated modalities)
        let mut z = vec![0.0_f32; self.d_h];
        for (j, slot) in z.iter_mut().enumerate() {
            let mut acc = 0.0_f32;
            for i in 0..self.d_a {
                acc += x_a[i] * self.w_z[i * self.d_h + j];
            }
            for i in 0..self.d_b {
                acc += x_b[i] * self.w_z[(self.d_a + i) * self.d_h + j];
            }
            *slot = sigmoid(acc);
        }

        // h = z ⊙ h_a + (1 − z) ⊙ h_b
        let mut h = vec![0.0_f32; self.d_h];
        for j in 0..self.d_h {
            h[j] = z[j] * h_a[j] + (1.0 - z[j]) * h_b[j];
        }
        Ok((h, z))
    }

    /// Batched forward: `x_a [batch × d_a]`, `x_b [batch × d_b]` → `[batch × d_h]`.
    ///
    /// Returns only the fused hidden states (gates discarded).
    ///
    /// # Errors
    /// Returns [`MultiModalError`] on a batch/shape mismatch or empty batch.
    pub fn forward(&self, x_a: &[f32], x_b: &[f32], batch: usize) -> MmResult<Vec<f32>> {
        if x_a.len() != batch * self.d_a {
            return Err(MultiModalError::DimensionMismatch {
                expected: batch * self.d_a,
                got: x_a.len(),
            });
        }
        if x_b.len() != batch * self.d_b {
            return Err(MultiModalError::DimensionMismatch {
                expected: batch * self.d_b,
                got: x_b.len(),
            });
        }
        if batch == 0 {
            return Err(MultiModalError::InvalidBatchSize);
        }
        let mut out = Vec::with_capacity(batch * self.d_h);
        for bi in 0..batch {
            let a_i = &x_a[bi * self.d_a..(bi + 1) * self.d_a];
            let b_i = &x_b[bi * self.d_b..(bi + 1) * self.d_b];
            let (h, _z) = self.forward_single(a_i, b_i)?;
            out.extend_from_slice(&h);
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(MultiModalError::NanEncountered { location: "gmu" });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_range_and_midpoint() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(50.0) > 0.999);
        assert!(sigmoid(-50.0) < 0.001);
        assert!(sigmoid(1e9).is_finite());
        assert!(sigmoid(-1e9).is_finite());
    }

    #[test]
    fn output_shape() {
        let gmu = GatedMultimodalUnit::zeros(4, 6, 8);
        let xa = vec![0.5_f32; 3 * 4];
        let xb = vec![0.3_f32; 3 * 6];
        let out = gmu.forward(&xa, &xb, 3).expect("forward should succeed");
        assert_eq!(out.len(), 3 * 8);
    }

    #[test]
    fn zero_weights_gate_is_half_output_zero() {
        // W=0 → h_a=h_b=0 (tanh(0)), z=σ(0)=0.5 → h = 0.5*0 + 0.5*0 = 0.
        let gmu = GatedMultimodalUnit::zeros(4, 4, 4);
        let xa = vec![1.0_f32; 4];
        let xb = vec![2.0_f32; 4];
        let (h, z) = gmu
            .forward_single(&xa, &xb)
            .expect("forward_single should succeed");
        for &zi in &z {
            assert!((zi - 0.5).abs() < 1e-6, "gate should be 0.5, got {zi}");
        }
        assert!(h.iter().all(|&v| v.abs() < 1e-9));
    }

    #[test]
    fn gate_open_selects_modality_a() {
        // Force z ≈ 1 by a large positive W_z on x_a; W_a identity-ish.
        let d = 2;
        let mut gmu = GatedMultimodalUnit::zeros(d, d, d);
        // W_a = identity so h_a = tanh(x_a)
        for k in 0..d {
            gmu.w_a[k * d + k] = 1.0;
        }
        // W_z huge positive on x_a channels → z ≈ 1
        for k in 0..d {
            gmu.w_z[k * d + k] = 100.0;
        }
        let xa = vec![0.5_f32, 0.5];
        let xb = vec![9.0_f32, 9.0];
        let (h, z) = gmu
            .forward_single(&xa, &xb)
            .expect("forward_single should succeed");
        assert!(z.iter().all(|&zi| zi > 0.99), "gate should be open");
        // h ≈ h_a = tanh(0.5)
        let expected = 0.5_f32.tanh();
        for &hi in &h {
            assert!((hi - expected).abs() < 1e-3, "h={hi}, expected {expected}");
        }
    }

    #[test]
    fn gate_closed_selects_modality_b() {
        let d = 2;
        let mut gmu = GatedMultimodalUnit::zeros(d, d, d);
        // W_b = identity so h_b = tanh(x_b)
        for k in 0..d {
            gmu.w_b[k * d + k] = 1.0;
        }
        // W_z huge negative on x_a → z ≈ 0
        for k in 0..d {
            gmu.w_z[k * d + k] = -100.0;
        }
        let xa = vec![1.0_f32, 1.0];
        let xb = vec![0.3_f32, 0.3];
        let (h, z) = gmu
            .forward_single(&xa, &xb)
            .expect("forward_single should succeed");
        assert!(z.iter().all(|&zi| zi < 0.01), "gate should be closed");
        let expected = 0.3_f32.tanh();
        for &hi in &h {
            assert!((hi - expected).abs() < 1e-3, "h={hi}, expected {expected}");
        }
    }

    #[test]
    fn output_bounded_by_tanh() {
        // h is a convex combination of two tanh outputs → |h| < 1.
        let mut gmu = GatedMultimodalUnit::zeros(3, 3, 3);
        for v in gmu.w_a.iter_mut() {
            *v = 5.0;
        }
        for v in gmu.w_b.iter_mut() {
            *v = -5.0;
        }
        let xa = vec![10.0_f32; 3];
        let xb = vec![10.0_f32; 3];
        let out = gmu.forward(&xa, &xb, 1).expect("forward should succeed");
        assert!(out.iter().all(|&v| v.abs() <= 1.0 + 1e-6));
    }

    #[test]
    fn dim_mismatch_a_errors() {
        let gmu = GatedMultimodalUnit::zeros(4, 6, 8);
        let xa = vec![0.0_f32; 3]; // wrong
        let xb = vec![0.0_f32; 6];
        assert!(matches!(
            gmu.forward_single(&xa, &xb),
            Err(MultiModalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn dim_mismatch_b_errors() {
        let gmu = GatedMultimodalUnit::zeros(4, 6, 8);
        let xa = vec![0.0_f32; 4];
        let xb = vec![0.0_f32; 5]; // wrong
        assert!(matches!(
            gmu.forward_single(&xa, &xb),
            Err(MultiModalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn batch_zero_errors() {
        let gmu = GatedMultimodalUnit::zeros(4, 6, 8);
        let xa: Vec<f32> = vec![];
        let xb: Vec<f32> = vec![];
        assert!(matches!(
            gmu.forward(&xa, &xb, 0),
            Err(MultiModalError::InvalidBatchSize)
        ));
    }

    #[test]
    fn gate_in_unit_interval() {
        let mut gmu = GatedMultimodalUnit::zeros(3, 3, 4);
        for (i, v) in gmu.w_z.iter_mut().enumerate() {
            *v = (i as f32 * 0.3).sin() * 4.0;
        }
        let xa = vec![0.7_f32, -0.2, 0.5];
        let xb = vec![0.1_f32, 0.9, -0.3];
        let (_h, z) = gmu
            .forward_single(&xa, &xb)
            .expect("forward_single should succeed");
        assert!(z.iter().all(|&zi| (0.0..=1.0).contains(&zi)));
    }

    #[test]
    fn output_finite_for_large_inputs() {
        let mut gmu = GatedMultimodalUnit::zeros(4, 4, 4);
        for v in gmu.w_z.iter_mut() {
            *v = 2.0;
        }
        let xa = vec![1e6_f32; 4];
        let xb = vec![-1e6_f32; 4];
        let out = gmu.forward(&xa, &xb, 1).expect("forward should succeed");
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
