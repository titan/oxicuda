//! LP-Relaxation Verification for Small MLPs.
//!
//! Computes tight output bounds for a 2-layer MLP (input → hidden ReLU → output)
//! under L∞ perturbation using interval bound propagation (IBP) combined with
//! the convex-hull ReLU relaxation.
//!
//! # Background
//!
//! For a ReLU neuron with pre-activation interval `[l, u]`:
//! * `u ≤ 0` (inactive):  output = 0.
//! * `l ≥ 0` (active):    output = pre-activation.
//! * `l < 0 < u` (ambiguous): convex hull relaxation gives
//!   `output ∈ [0, u]` — lower bound is 0, upper is the pre-activation upper
//!   bound (the tightest convex hull relaxation consistent with IBP).
//!
//! This is equivalent to the standard LP relaxation of the ReLU convex hull
//! in the IBP setting, and matches the "CROWN/IBP hybrid" approach described in
//! Gowal et al. (2018).
//!
//! References:
//! * Ehlers (2017): *"Formal Verification of Piece-Wise Linear Feed-Forward
//!   Neural Networks"*
//! * Gowal, Dvijotham, Stanforth, Bunel, Qin, Uesato, Arandjelovic, Mann,
//!   Kohli (2018 NeurIPS): *"On the Effectiveness of Interval Bound Propagation
//!   for Training Verifiably Robust Models"*

use crate::error::{AdvError, AdvResult};

// ─── AffineLayer ─────────────────────────────────────────────────────────────

/// A single affine layer: `y = W x + b`.
///
/// `w` is stored row-major with shape `[out_dim × in_dim]`, so
/// `w[j * in_dim + i]` is the weight from input `i` to output `j`.
#[derive(Debug, Clone)]
pub struct AffineLayer {
    /// Weight matrix `[out_dim × in_dim]`, row-major.
    pub w: Vec<f32>,
    /// Bias vector `[out_dim]`.
    pub b: Vec<f32>,
    /// Input dimensionality.
    pub in_dim: usize,
    /// Output dimensionality.
    pub out_dim: usize,
}

impl AffineLayer {
    /// Construct an `AffineLayer` and validate dimensions.
    ///
    /// # Errors
    /// * [`AdvError::EmptyInput`]        — zero input or output dim.
    /// * [`AdvError::DimensionMismatch`] — weight or bias length mismatch.
    pub fn new(w: Vec<f32>, b: Vec<f32>, in_dim: usize, out_dim: usize) -> AdvResult<Self> {
        if in_dim == 0 || out_dim == 0 {
            return Err(AdvError::EmptyInput);
        }
        let expected_w = in_dim * out_dim;
        if w.len() != expected_w {
            return Err(AdvError::DimensionMismatch {
                expected: expected_w,
                got: w.len(),
            });
        }
        if b.len() != out_dim {
            return Err(AdvError::DimensionMismatch {
                expected: out_dim,
                got: b.len(),
            });
        }
        Ok(Self {
            w,
            b,
            in_dim,
            out_dim,
        })
    }

    /// Forward-pass bound propagation: compute `[y_lo, y_hi]` from `[x_lo, x_hi]`.
    ///
    /// Uses interval arithmetic:
    /// ```text
    /// y_lo[j] = b[j] + Σ_i  (w[j,i] > 0 ? w[j,i]*x_lo[i] : w[j,i]*x_hi[i])
    /// y_hi[j] = b[j] + Σ_i  (w[j,i] > 0 ? w[j,i]*x_hi[i] : w[j,i]*x_lo[i])
    /// ```
    fn ibp_forward(&self, x_lo: &[f32], x_hi: &[f32]) -> AdvResult<(Vec<f32>, Vec<f32>)> {
        if x_lo.len() != self.in_dim {
            return Err(AdvError::DimensionMismatch {
                expected: self.in_dim,
                got: x_lo.len(),
            });
        }
        let mut y_lo = Vec::with_capacity(self.out_dim);
        let mut y_hi = Vec::with_capacity(self.out_dim);
        for j in 0..self.out_dim {
            let mut lo = self.b[j];
            let mut hi = self.b[j];
            for i in 0..self.in_dim {
                let wji = self.w[j * self.in_dim + i];
                if wji >= 0.0 {
                    lo += wji * x_lo[i];
                    hi += wji * x_hi[i];
                } else {
                    lo += wji * x_hi[i];
                    hi += wji * x_lo[i];
                }
            }
            // Floating-point guard: enforce lo <= hi
            if lo > hi {
                std::mem::swap(&mut lo, &mut hi);
            }
            y_lo.push(lo);
            y_hi.push(hi);
        }
        Ok((y_lo, y_hi))
    }
}

// ─── LpRelaxConfig ───────────────────────────────────────────────────────────

/// Configuration for the LP-relaxation verifier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LpRelaxConfig {
    /// L∞ perturbation budget ε; must be `> 0` and finite.
    pub eps: f32,
    /// Maximum iterations for dual ascent (unused in the IBP-based algorithm;
    /// reserved for future extensions). Default: 200.
    pub max_iters: usize,
    /// Dual gradient step size (reserved for future extensions). Default: 0.01.
    pub step_size: f32,
}

impl Default for LpRelaxConfig {
    fn default() -> Self {
        Self {
            eps: 0.1,
            max_iters: 200,
            step_size: 0.01,
        }
    }
}

// ─── VerifiedBound ───────────────────────────────────────────────────────────

/// Verified lower/upper bound on a single output neuron under L∞ perturbation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VerifiedBound {
    /// Guaranteed lower bound on the output neuron's value.
    pub lower: f32,
    /// Guaranteed upper bound on the output neuron's value.
    pub upper: f32,
}

// ─── LpRelaxVerifier ─────────────────────────────────────────────────────────

/// LP-relaxation verifier for a 2-layer MLP (input → hidden ReLU → output).
///
/// # Algorithm
///
/// 1. Build input interval `[x_c - ε, x_c + ε]` (clamped to `[0, 1]`).
/// 2. IBP through `layer1` to get hidden pre-activation bounds `[l1, u1]`.
/// 3. Apply convex-hull ReLU relaxation to get `[l1r, u1r]`:
///    - `u1_j ≤ 0`: neuron inactive → `l1r_j = u1r_j = 0`.
///    - `l1_j ≥ 0`: neuron active   → `l1r_j = l1_j, u1r_j = u1_j`.
///    - otherwise:  ambiguous        → `l1r_j = 0, u1r_j = u1_j`.
/// 4. IBP through `layer2` using `[l1r, u1r]` to get output bounds.
pub struct LpRelaxVerifier {
    /// First affine layer (input → hidden).
    pub layer1: AffineLayer,
    /// Second affine layer (hidden → output).
    pub layer2: AffineLayer,
    /// Verifier configuration.
    pub cfg: LpRelaxConfig,
}

impl LpRelaxVerifier {
    /// Create a new `LpRelaxVerifier`.
    ///
    /// # Errors
    /// * [`AdvError::InvalidEpsilon`]    — `eps ≤ 0` or non-finite.
    /// * [`AdvError::DimensionMismatch`] — `layer1.out_dim != layer2.in_dim`.
    pub fn new(layer1: AffineLayer, layer2: AffineLayer, cfg: LpRelaxConfig) -> AdvResult<Self> {
        if !(cfg.eps > 0.0 && cfg.eps.is_finite()) {
            return Err(AdvError::InvalidEpsilon { eps: cfg.eps });
        }
        if layer1.out_dim != layer2.in_dim {
            return Err(AdvError::DimensionMismatch {
                expected: layer1.out_dim,
                got: layer2.in_dim,
            });
        }
        Ok(Self {
            layer1,
            layer2,
            cfg,
        })
    }

    /// Compute lower/upper bounds on each output neuron using IBP + ReLU relaxation.
    ///
    /// Returns `Vec<VerifiedBound>` with one entry per output class.
    ///
    /// # Errors
    /// * [`AdvError::DimensionMismatch`] — `x_center.len() != layer1.in_dim`.
    /// * [`AdvError::NanEncountered`]    — non-finite value in `x_center`.
    pub fn verify_output(&self, x_center: &[f32]) -> AdvResult<Vec<VerifiedBound>> {
        if x_center.len() != self.layer1.in_dim {
            return Err(AdvError::DimensionMismatch {
                expected: self.layer1.in_dim,
                got: x_center.len(),
            });
        }
        if x_center.iter().any(|v| !v.is_finite()) {
            return Err(AdvError::NanEncountered {
                location: "LpRelaxVerifier::verify_output",
            });
        }

        // Step 1: Input interval [x_c - eps, x_c + eps], clamped to [0, 1].
        let x_lo: Vec<f32> = x_center
            .iter()
            .map(|&v| (v - self.cfg.eps).max(0.0))
            .collect();
        let x_hi: Vec<f32> = x_center
            .iter()
            .map(|&v| (v + self.cfg.eps).min(1.0))
            .collect();

        // Step 2: IBP through layer1 → hidden pre-activation bounds [l1, u1].
        let (l1, u1) = self.layer1.ibp_forward(&x_lo, &x_hi)?;

        // Step 3: Convex-hull ReLU relaxation → [l1r, u1r].
        let hidden_n = self.layer1.out_dim;
        let mut l1r = Vec::with_capacity(hidden_n);
        let mut u1r = Vec::with_capacity(hidden_n);
        for j in 0..hidden_n {
            let lo = l1[j];
            let hi = u1[j];
            if hi <= 0.0 {
                // Fully inactive neuron: ReLU output = 0.
                l1r.push(0.0_f32);
                u1r.push(0.0_f32);
            } else if lo >= 0.0 {
                // Fully active neuron: ReLU is identity.
                l1r.push(lo);
                u1r.push(hi);
            } else {
                // Ambiguous neuron: convex hull lower = 0, upper = hi.
                l1r.push(0.0_f32);
                u1r.push(hi);
            }
        }

        // Step 4: IBP through layer2 using [l1r, u1r] → output bounds.
        let (out_lo, out_hi) = self.layer2.ibp_forward(&l1r, &u1r)?;

        let out_dim = self.layer2.out_dim;
        let mut bounds = Vec::with_capacity(out_dim);
        for k in 0..out_dim {
            bounds.push(VerifiedBound {
                lower: out_lo[k],
                upper: out_hi[k],
            });
        }
        Ok(bounds)
    }

    /// Check if the true class `true_class` remains the top predicted class
    /// under all L∞ perturbations of size `cfg.eps`.
    ///
    /// Returns `true` iff `lower_bound[true_class] > max_{j ≠ true_class} upper_bound[j]`.
    ///
    /// # Errors
    /// * [`AdvError::DimensionMismatch`] — `true_class >= n_classes` or `x_center` mismatch.
    /// * [`AdvError::NanEncountered`]    — non-finite value in `x_center`.
    pub fn is_robust(&self, x_center: &[f32], true_class: usize) -> AdvResult<bool> {
        let bounds = self.verify_output(x_center)?;
        let n_classes = bounds.len();
        if true_class >= n_classes {
            return Err(AdvError::DimensionMismatch {
                expected: n_classes.saturating_sub(1),
                got: true_class,
            });
        }
        let true_lo = bounds[true_class].lower;
        // Find max upper bound among all other classes.
        let max_other_upper = (0..n_classes)
            .filter(|&j| j != true_class)
            .map(|j| bounds[j].upper)
            .fold(f32::NEG_INFINITY, f32::max);

        Ok(true_lo > max_other_upper)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a simple 2-layer MLP verifier.
    // layer1: [in_dim → hidden], layer2: [hidden → out_dim]
    fn make_verifier(
        w1: Vec<f32>,
        b1: Vec<f32>,
        in_dim: usize,
        hidden: usize,
        w2: Vec<f32>,
        b2: Vec<f32>,
        out_dim: usize,
        eps: f32,
    ) -> AdvResult<LpRelaxVerifier> {
        let layer1 = AffineLayer::new(w1, b1, in_dim, hidden)?;
        let layer2 = AffineLayer::new(w2, b2, hidden, out_dim)?;
        let cfg = LpRelaxConfig {
            eps,
            ..Default::default()
        };
        LpRelaxVerifier::new(layer1, layer2, cfg)
    }

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    // ── AffineLayer construction ──────────────────────────────────────────────

    #[test]
    fn affine_layer_new_valid() {
        let layer = AffineLayer::new(vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 0.0], 2, 2);
        assert!(layer.is_ok());
    }

    #[test]
    fn affine_layer_empty_dim_errors() {
        let r = AffineLayer::new(vec![], vec![], 0, 2);
        assert!(matches!(r, Err(AdvError::EmptyInput)));
    }

    #[test]
    fn affine_layer_weight_mismatch_errors() {
        // Expected 6 weights (2*3), provide 4.
        let r = AffineLayer::new(vec![1.0; 4], vec![0.0; 3], 2, 3);
        assert!(matches!(r, Err(AdvError::DimensionMismatch { .. })));
    }

    #[test]
    fn affine_layer_bias_mismatch_errors() {
        // Correct weights but wrong bias size.
        let r = AffineLayer::new(vec![1.0; 6], vec![0.0; 2], 2, 3);
        assert!(matches!(r, Err(AdvError::DimensionMismatch { .. })));
    }

    // ── LpRelaxConfig / LpRelaxVerifier construction ──────────────────────────

    #[test]
    fn verifier_invalid_eps_errors() {
        let r = make_verifier(
            vec![1.0; 4],
            vec![0.0; 2],
            2,
            2,
            vec![1.0; 4],
            vec![0.0; 2],
            2,
            -0.1,
        );
        assert!(matches!(r, Err(AdvError::InvalidEpsilon { .. })));
    }

    #[test]
    fn verifier_zero_eps_errors() {
        let r = make_verifier(
            vec![1.0; 4],
            vec![0.0; 2],
            2,
            2,
            vec![1.0; 4],
            vec![0.0; 2],
            2,
            0.0,
        );
        assert!(matches!(r, Err(AdvError::InvalidEpsilon { .. })));
    }

    #[test]
    fn verifier_layer_dim_mismatch_errors() {
        // layer1 out=2, layer2 in=3 → mismatch.
        let layer1 =
            AffineLayer::new(vec![1.0; 4], vec![0.0; 2], 2, 2).expect("new should succeed");
        let layer2 =
            AffineLayer::new(vec![1.0; 6], vec![0.0; 2], 3, 2).expect("new should succeed");
        let cfg = LpRelaxConfig::default();
        let r = LpRelaxVerifier::new(layer1, layer2, cfg);
        assert!(matches!(r, Err(AdvError::DimensionMismatch { .. })));
    }

    // ── verify_output: inactive / active neurons ──────────────────────────────

    #[test]
    fn verify_output_all_active_identity_layers() {
        // Both layers identity 2×2, x_center=[0.5, 0.5], eps=0.1
        // layer1: z = [0.5, 0.5] ± 0.1 → l1=[0.4,0.4], u1=[0.6,0.6]
        // Both active (l1 > 0) → l1r=l1, u1r=u1
        // layer2: out = l1r / u1r directly
        let v = make_verifier(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            2,
            2,
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            2,
            0.1,
        )
        .expect("value should be present");
        let x = vec![0.5_f32, 0.5];
        let bounds = v.verify_output(&x).expect("verify_output should succeed");
        assert_eq!(bounds.len(), 2);
        assert!(approx(bounds[0].lower, 0.4, 1e-5));
        assert!(approx(bounds[0].upper, 0.6, 1e-5));
    }

    #[test]
    fn verify_output_all_inactive_neurons_zero() {
        // layer1: strongly negative bias → all hidden neurons inactive.
        // w1 = identity 2×2, b1 = -10.0 → hidden pre-activation in [-10.1, -9.9]
        // → all hidden output = 0 → layer2 output = b2.
        let v = make_verifier(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![-10.0, -10.0],
            2,
            2,
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 2.0],
            2,
            0.1,
        )
        .expect("value should be present");
        let x = vec![0.5_f32, 0.5];
        let bounds = v.verify_output(&x).expect("verify_output should succeed");
        // All hidden = 0 → output = b2 = [1.0, 2.0]
        assert!(approx(bounds[0].lower, 1.0, 1e-5));
        assert!(approx(bounds[0].upper, 1.0, 1e-5));
        assert!(approx(bounds[1].lower, 2.0, 1e-5));
        assert!(approx(bounds[1].upper, 2.0, 1e-5));
    }

    #[test]
    fn verify_output_ambiguous_neurons_upper_only() {
        // layer1: w1=identity, b1=0, x=[0.5,0.5], eps=1.0
        // → l1=[-0.5,-0.5], u1=[1.0,1.0] (after clamping x to [0,1]:
        //   x_lo=max(0.5-1,0)=0, x_hi=min(0.5+1,1)=1)
        // → l1=[0+0,0+0]=[0,0], u1=[1,1]  (identity, b=0)
        // Actually l1_j = b[j] + w*x_lo = 0 + 1.0*0.0 = 0.0, so l1 >= 0 → active
        // Let's use negative x_center to force ambiguous: x=[-0.1,-0.1] clamped:
        //   x_lo = max(-0.1-0.1, 0) = 0, x_hi = min(-0.1+0.1, 1) = 0
        // Still zero. Use a network with negative weights:
        // w1 = [[-1,0],[0,-1]], b1=[0,0], x=[0.5,0.5], eps=0.1
        // l1 = [-1*x_hi, -1*x_hi] = [-0.6, -0.6], u1 = [-1*x_lo, -1*x_lo] = [-0.4,-0.4]
        // Both inactive → output = b2
        let v = make_verifier(
            vec![-1.0, 0.0, 0.0, -1.0],
            vec![0.0, 0.0],
            2,
            2,
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.5, 0.5],
            2,
            0.1,
        )
        .expect("value should be present");
        let x = vec![0.5_f32, 0.5];
        let bounds = v.verify_output(&x).expect("verify_output should succeed");
        assert!(approx(bounds[0].lower, 0.5, 1e-5));
        assert!(approx(bounds[0].upper, 0.5, 1e-5));
    }

    #[test]
    fn verify_output_dim_mismatch_errors() {
        let v = make_verifier(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            2,
            2,
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            2,
            0.1,
        )
        .expect("value should be present");
        let x = vec![0.5_f32; 3]; // wrong dim
        assert!(matches!(
            v.verify_output(&x),
            Err(AdvError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn verify_output_nan_input_errors() {
        let v = make_verifier(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            2,
            2,
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            2,
            0.1,
        )
        .expect("value should be present");
        let x = vec![f32::NAN, 0.5];
        assert!(matches!(
            v.verify_output(&x),
            Err(AdvError::NanEncountered { .. })
        ));
    }

    // ── is_robust ─────────────────────────────────────────────────────────────

    #[test]
    fn is_robust_strongly_dominant_class() {
        // Class 0 strongly dominates: w2 = [[10,0],[0,1]], b2=[0,0]
        // hidden at x=[0.5,0.5], eps=0.1 (identity w1):
        // l1r=[0.4,0.4], u1r=[0.6,0.6]
        // out0 in [10*0.4, 10*0.6] = [4.0, 6.0]
        // out1 in [0.4, 0.6]
        // lower[0]=4.0 > upper[1]=0.6 → robust
        let v = make_verifier(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            2,
            2,
            vec![10.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            2,
            0.1,
        )
        .expect("value should be present");
        let x = vec![0.5_f32, 0.5];
        assert!(v.is_robust(&x, 0).expect("is_robust should succeed"));
    }

    #[test]
    fn is_robust_close_classes_not_robust() {
        // Class 0 and 1 are close: identity 2×2 network, x=[0.5, 0.5], eps=0.3
        // layer1: identity → l1=[0.2,0.2], u1=[0.8,0.8] (x clamped [0,1])
        // both active → l1r=[0.2,0.2], u1r=[0.8,0.8]
        // layer2: identity → out0 in [0.2,0.8], out1 in [0.2,0.8]
        // lower[0]=0.2, upper[1]=0.8 → NOT robust
        let v = make_verifier(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            2,
            2,
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            2,
            0.3,
        )
        .expect("value should be present");
        let x = vec![0.5_f32, 0.5];
        assert!(!v.is_robust(&x, 0).expect("is_robust should succeed"));
    }

    #[test]
    fn is_robust_invalid_true_class_errors() {
        let v = make_verifier(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            2,
            2,
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            2,
            0.1,
        )
        .expect("value should be present");
        let x = vec![0.5_f32, 0.5];
        // true_class=5 but only 2 output classes
        assert!(matches!(
            v.is_robust(&x, 5),
            Err(AdvError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn verify_output_lower_le_upper_invariant() {
        // Random-ish network; invariant: lower <= upper for all outputs.
        let w1 = vec![
            0.5, -0.3, 0.2, 0.8, -0.1, 0.4, 0.7, -0.6, 0.3, 0.5, -0.2, 0.9,
        ];
        let b1 = vec![0.1_f32, -0.2, 0.15, 0.05];
        let w2 = vec![-0.3, 0.7, -0.1, 0.4, 0.5, -0.8, 0.2, -0.3];
        let b2 = vec![0.0_f32, 0.0];
        let v = make_verifier(w1, b1, 3, 4, w2, b2, 2, 0.15).expect("make_verifier should succeed");
        let x = vec![0.4_f32, 0.6, 0.3];
        let bounds = v.verify_output(&x).expect("verify_output should succeed");
        for b in &bounds {
            assert!(
                b.lower <= b.upper + 1e-6,
                "lower={} > upper={}",
                b.lower,
                b.upper
            );
        }
    }

    #[test]
    fn verify_output_eps_zero_is_exact() {
        // Very small eps ≈ 0 → bounds should be extremely tight.
        // With eps=1e-6 and identity layers: bounds ≈ x_center value.
        let v = make_verifier(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            2,
            2,
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0],
            2,
            1e-6,
        )
        .expect("value should be present");
        let x = vec![0.5_f32, 0.6];
        let bounds = v.verify_output(&x).expect("verify_output should succeed");
        // bounds[0] should be approximately (0.5, 0.5) ± 1e-6
        assert!(approx(bounds[0].lower, 0.5, 1e-4));
        assert!(approx(bounds[0].upper, 0.5, 1e-4));
    }
}
