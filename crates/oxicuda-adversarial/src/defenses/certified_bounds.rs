//! Certified bounds — Interval Bound Propagation (IBP) and Lipschitz radius.
//!
//! References:
//! * Gowal, Dvijotham, Stanforth, Bunel, Qin, Uesato, Arandjelovic, Mann &
//!   Kohli (2018), *"On the Effectiveness of Interval Bound Propagation for
//!   Training Verifiably Robust Models"*, arXiv:1810.12715.
//! * Tsuzuku, Sato & Sugiyama (2018),
//!   *"Lipschitz-Margin Training: Scalable Certification of Perturbation
//!   Invariance for Deep Neural Networks"*, NeurIPS.
//!
//! IBP propagates per-coordinate intervals `[lo, hi]` through the network.
//! For an affine layer `y = Wx + b` we split `W = W^+ + W^−` with
//! `W^+ = max(W, 0)` and `W^− = min(W, 0)`, giving the **tight**
//! per-output-coordinate interval
//!
//! ```text
//! lo_y = W^+ · lo_x + W^− · hi_x + b
//! hi_y = W^+ · hi_x + W^− · lo_x + b
//! ```
//!
//! This is exact under any element-wise rectifier composed with the affine
//! map (e.g. ReLU clamps the negative side to 0).
//!
//! The Lipschitz-margin certificate uses the L2 spectral-norm product of the
//! network as Lipschitz constant `L`; for a sample with predicted-class
//! margin `m` (i.e. logit difference between the top class and the runner-up)
//! the certified L2 radius is
//!
//! ```text
//! r = m / (L · √2).
//! ```
//!
//! See Tsuzuku et al. 2018, Theorem 1 — the `√2` factor comes from the
//! cross-Lipschitz constant between two output components.

use crate::error::{AdvError, AdvResult};

// ─── IntervalBound ───────────────────────────────────────────────────────────

/// Per-element interval `[lo, hi]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntervalBound {
    /// Lower endpoint.
    pub lo: f32,
    /// Upper endpoint.
    pub hi: f32,
}

impl IntervalBound {
    /// Build a new interval. Both endpoints must be finite and `lo <= hi`.
    ///
    /// # Errors
    /// * [`AdvError::NanEncountered`]    — non-finite endpoint.
    /// * [`AdvError::InvalidLossWeight`] — `lo > hi`.
    pub fn new(lo: f32, hi: f32) -> AdvResult<Self> {
        if !(lo.is_finite() && hi.is_finite()) {
            return Err(AdvError::NanEncountered {
                location: "IntervalBound::new",
            });
        }
        if lo > hi {
            return Err(AdvError::InvalidLossWeight { weight: hi - lo });
        }
        Ok(Self { lo, hi })
    }

    /// Width of the interval, `hi - lo`.
    #[must_use]
    #[inline]
    pub fn width(self) -> f32 {
        self.hi - self.lo
    }

    /// True iff `lo <= v <= hi`.
    #[must_use]
    #[inline]
    pub fn contains(self, v: f32) -> bool {
        self.lo <= v && v <= self.hi
    }

    /// Minkowski sum `[a + c, b + d]` (assumes both intervals are valid).
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Self) -> Self {
        Self {
            lo: self.lo + other.lo,
            hi: self.hi + other.hi,
        }
    }

    /// Multiply by a scalar. Negative scalars flip the endpoints.
    #[must_use]
    pub fn mul_scalar(self, s: f32) -> Self {
        if s >= 0.0 {
            Self {
                lo: self.lo * s,
                hi: self.hi * s,
            }
        } else {
            Self {
                lo: self.hi * s,
                hi: self.lo * s,
            }
        }
    }

    /// ReLU: clamps the negative side to zero.
    #[must_use]
    pub fn relu(self) -> Self {
        Self {
            lo: self.lo.max(0.0),
            hi: self.hi.max(0.0),
        }
    }
}

// ─── IBP propagation ─────────────────────────────────────────────────────────

/// Propagate interval bounds through a single affine layer `y = Wx + b`.
///
/// `W` is `[out_dim × in_dim]` row-major; `b` is `[out_dim]`. Per the IBP
/// recipe, we split `W = W^+ + W^−` and compute
///
/// ```text
/// lo_y[i] = Σ_j  max(W[i,j], 0) · lo_x[j] + min(W[i,j], 0) · hi_x[j]  + b[i]
/// hi_y[i] = Σ_j  max(W[i,j], 0) · hi_x[j] + min(W[i,j], 0) · lo_x[j]  + b[i]
/// ```
///
/// # Errors
/// * [`AdvError::EmptyInput`]        — zero-sized layer.
/// * [`AdvError::DimensionMismatch`] — `bounds_in.len() != in_dim`,
///   `w.len() != out_dim*in_dim`, or `b.len() != out_dim`.
/// * [`AdvError::NanEncountered`]    — non-finite weight or bias entry.
pub fn ibp_propagate(
    bounds_in: &[IntervalBound],
    w: &[f32],
    b: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> AdvResult<Vec<IntervalBound>> {
    if in_dim == 0 || out_dim == 0 {
        return Err(AdvError::EmptyInput);
    }
    if bounds_in.len() != in_dim {
        return Err(AdvError::DimensionMismatch {
            expected: in_dim,
            got: bounds_in.len(),
        });
    }
    if w.len() != in_dim * out_dim {
        return Err(AdvError::DimensionMismatch {
            expected: in_dim * out_dim,
            got: w.len(),
        });
    }
    if b.len() != out_dim {
        return Err(AdvError::DimensionMismatch {
            expected: out_dim,
            got: b.len(),
        });
    }
    if w.iter().any(|v| !v.is_finite()) {
        return Err(AdvError::NanEncountered {
            location: "ibp_propagate:w",
        });
    }
    if b.iter().any(|v| !v.is_finite()) {
        return Err(AdvError::NanEncountered {
            location: "ibp_propagate:b",
        });
    }

    let mut out = Vec::with_capacity(out_dim);
    for i in 0..out_dim {
        let row = &w[i * in_dim..(i + 1) * in_dim];
        let mut lo_acc = b[i];
        let mut hi_acc = b[i];
        for (j, &wij) in row.iter().enumerate() {
            let bnd = bounds_in[j];
            if wij >= 0.0 {
                lo_acc += wij * bnd.lo;
                hi_acc += wij * bnd.hi;
            } else {
                lo_acc += wij * bnd.hi;
                hi_acc += wij * bnd.lo;
            }
        }
        // Floating round-off can produce lo slightly > hi when the row is
        // numerically degenerate; clamp to maintain the invariant.
        if lo_acc > hi_acc {
            std::mem::swap(&mut lo_acc, &mut hi_acc);
        }
        out.push(IntervalBound {
            lo: lo_acc,
            hi: hi_acc,
        });
    }
    Ok(out)
}

// ─── Lipschitz-based certified L2 radius ────────────────────────────────────

/// Lipschitz-margin certified L2 radius.
///
/// Given a network with global L2-Lipschitz constant `L` and a sample whose
/// predicted-class **margin** is `m` (logit gap between the top class and the
/// runner-up), the certified L2 radius is `r = m / (L · √2)`
/// (Tsuzuku et al. 2018, Theorem 1).
///
/// # Errors
/// * [`AdvError::InvalidLossWeight`]  — `margin` is non-finite or negative.
/// * [`AdvError::InvalidEpsilon`]     — `lipschitz_constant` is non-finite or
///   `<= 0`.
pub fn lipschitz_certified_radius(margin: f32, lipschitz_constant: f32) -> AdvResult<f32> {
    if !(margin.is_finite() && margin >= 0.0) {
        return Err(AdvError::InvalidLossWeight { weight: margin });
    }
    if !(lipschitz_constant.is_finite() && lipschitz_constant > 0.0) {
        return Err(AdvError::InvalidEpsilon {
            eps: lipschitz_constant,
        });
    }
    Ok(margin / (lipschitz_constant * std::f32::consts::SQRT_2))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn interval_constructor_and_predicates() {
        let i = IntervalBound::new(-1.0, 2.0).expect("ok");
        assert!(approx_eq(i.width(), 3.0, 1e-6));
        assert!(i.contains(0.5));
        assert!(i.contains(-1.0));
        assert!(i.contains(2.0));
        assert!(!i.contains(2.1));
        assert!(!i.contains(-1.1));
        assert!(IntervalBound::new(2.0, 1.0).is_err());
        assert!(IntervalBound::new(f32::NAN, 1.0).is_err());
        assert!(IntervalBound::new(0.0, f32::INFINITY).is_err());
    }

    #[test]
    fn interval_add_and_mul_scalar() {
        let a = IntervalBound::new(-1.0, 2.0).expect("a");
        let b = IntervalBound::new(0.5, 1.5).expect("b");
        let c = a.add(b);
        assert!(approx_eq(c.lo, -0.5, 1e-6));
        assert!(approx_eq(c.hi, 3.5, 1e-6));

        let d = a.mul_scalar(2.0);
        assert!(approx_eq(d.lo, -2.0, 1e-6));
        assert!(approx_eq(d.hi, 4.0, 1e-6));

        // Negative scalar swaps endpoints.
        let e = a.mul_scalar(-1.0);
        assert!(approx_eq(e.lo, -2.0, 1e-6));
        assert!(approx_eq(e.hi, 1.0, 1e-6));
        assert!(e.lo <= e.hi);
    }

    #[test]
    fn relu_clips_negative_side() {
        let i = IntervalBound::new(-2.0, 3.0).expect("i");
        let r = i.relu();
        assert!(approx_eq(r.lo, 0.0, 1e-6));
        assert!(approx_eq(r.hi, 3.0, 1e-6));

        // Fully-negative interval collapses to [0, 0].
        let neg = IntervalBound::new(-5.0, -1.0).expect("neg");
        let rn = neg.relu();
        assert!(approx_eq(rn.lo, 0.0, 1e-6));
        assert!(approx_eq(rn.hi, 0.0, 1e-6));

        // Fully-positive interval is unchanged.
        let pos = IntervalBound::new(1.0, 4.0).expect("pos");
        let rp = pos.relu();
        assert!(approx_eq(rp.lo, 1.0, 1e-6));
        assert!(approx_eq(rp.hi, 4.0, 1e-6));
    }

    #[test]
    fn ibp_identity_weights_preserve_bounds() {
        // I_3 weight, zero bias → output bounds = input bounds.
        let bounds = vec![
            IntervalBound::new(-1.0, 1.0).expect("b0"),
            IntervalBound::new(0.5, 0.7).expect("b1"),
            IntervalBound::new(-2.0, -0.5).expect("b2"),
        ];
        let w = vec![
            1.0_f32, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0,
        ];
        let b = vec![0.0_f32; 3];
        let out = ibp_propagate(&bounds, &w, &b, 3, 3).expect("ibp");
        for (i, o) in out.iter().enumerate() {
            assert!(approx_eq(o.lo, bounds[i].lo, 1e-5));
            assert!(approx_eq(o.hi, bounds[i].hi, 1e-5));
        }
    }

    #[test]
    fn ibp_negative_weight_swaps_bounds() {
        // y = -x  →  output_lo = -hi_x, output_hi = -lo_x.
        let bounds = vec![IntervalBound::new(-1.0, 2.0).expect("b")];
        let w = vec![-1.0_f32];
        let b = vec![0.0_f32];
        let out = ibp_propagate(&bounds, &w, &b, 1, 1).expect("ibp");
        assert!(approx_eq(out[0].lo, -2.0, 1e-6));
        assert!(approx_eq(out[0].hi, 1.0, 1e-6));
    }

    #[test]
    fn ibp_with_bias() {
        // Single output: y = 2*x_0 + 3*x_1 + 0.5
        let bounds = vec![
            IntervalBound::new(-1.0, 1.0).expect("b0"),
            IntervalBound::new(0.0, 2.0).expect("b1"),
        ];
        let w = vec![2.0_f32, 3.0];
        let b = vec![0.5_f32];
        let out = ibp_propagate(&bounds, &w, &b, 2, 1).expect("ibp");
        // lo = 2*(-1) + 3*0 + 0.5 = -1.5
        // hi = 2*1 + 3*2 + 0.5 = 8.5
        assert!(approx_eq(out[0].lo, -1.5, 1e-6));
        assert!(approx_eq(out[0].hi, 8.5, 1e-6));
    }

    #[test]
    fn ibp_dim_mismatch_errors() {
        let bounds = vec![IntervalBound::new(0.0, 1.0).expect("b"); 3];
        let w_bad = vec![0.0_f32; 5]; // Should be 6 = 2*3.
        let b = vec![0.0_f32, 0.0];
        assert!(matches!(
            ibp_propagate(&bounds, &w_bad, &b, 3, 2).unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));

        let w_ok = vec![0.0_f32; 6];
        let b_bad = vec![0.0_f32; 1];
        assert!(matches!(
            ibp_propagate(&bounds, &w_ok, &b_bad, 3, 2).unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));

        let bounds_bad = vec![IntervalBound::new(0.0, 1.0).expect("b"); 2];
        assert!(matches!(
            ibp_propagate(&bounds_bad, &w_ok, &b, 3, 2).unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn ibp_zero_dim_or_nan_rejected() {
        let bounds = vec![IntervalBound::new(0.0, 1.0).expect("b")];
        let w = vec![1.0_f32];
        let b = vec![0.0_f32];
        assert_eq!(
            ibp_propagate(&bounds, &w, &b, 0, 1).unwrap_err(),
            AdvError::EmptyInput
        );
        let nan_w = vec![f32::NAN];
        assert!(matches!(
            ibp_propagate(&bounds, &nan_w, &b, 1, 1).unwrap_err(),
            AdvError::NanEncountered { .. }
        ));
        let nan_b = vec![f32::NAN];
        assert!(matches!(
            ibp_propagate(&bounds, &w, &nan_b, 1, 1).unwrap_err(),
            AdvError::NanEncountered { .. }
        ));
    }

    #[test]
    fn lipschitz_radius_formula() {
        // r = m / (L * √2)
        let r = lipschitz_certified_radius(1.0, 1.0).expect("ok");
        assert!(approx_eq(r, 1.0 / std::f32::consts::SQRT_2, 1e-6));

        let r2 = lipschitz_certified_radius(4.0, 2.0).expect("ok");
        assert!(approx_eq(r2, 4.0 / (2.0 * std::f32::consts::SQRT_2), 1e-6));

        // Zero margin → zero radius.
        let r0 = lipschitz_certified_radius(0.0, 5.0).expect("ok");
        assert!(approx_eq(r0, 0.0, 1e-6));
    }

    #[test]
    fn lipschitz_radius_invalid_inputs() {
        assert!(lipschitz_certified_radius(-0.1, 1.0).is_err());
        assert!(lipschitz_certified_radius(f32::NAN, 1.0).is_err());
        assert!(lipschitz_certified_radius(1.0, 0.0).is_err());
        assert!(lipschitz_certified_radius(1.0, -1.0).is_err());
        assert!(lipschitz_certified_radius(1.0, f32::INFINITY).is_err());
    }
}
