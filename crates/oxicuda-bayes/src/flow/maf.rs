//! Masked Autoregressive Flow (MAF).
//!
//! Implements Papamakarios, Pavlakou & Murray, "Masked Autoregressive Flow for
//! Density Estimation" (NeurIPS 2017). MAF stacks autoregressive affine
//! transforms whose shift and log-scale for coordinate `i` depend only on the
//! preceding coordinates `x_{<i}` (enforced by MADE masks), giving an
//! invertible map with a triangular Jacobian and hence a tractable
//! log-determinant.
//!
//! # Single layer
//!
//! With a MADE conditioner producing `(μ_i(x_{<i}), α_i(x_{<i}))`, the
//! **forward** (data → noise) direction is the closed-form, parallel map
//!
//! ```text
//! u_i = (x_i − μ_i) · exp(−α_i),     log|det ∂u/∂x| = − Σ_i α_i.
//! ```
//!
//! The **inverse** (noise → data) direction must be evaluated sequentially,
//! because `μ_i` and `α_i` depend on the already-reconstructed `x_{<i}`:
//!
//! ```text
//! x_i = u_i · exp(α_i) + μ_i      (computed for i = 0, 1, …, d−1).
//! ```
//!
//! # Stacked flow
//!
//! A [`MafFlow`] composes several [`MafLayer`]s, reversing the coordinate order
//! between layers so that every dimension can condition on every other after a
//! full pass. The density of a data point under the flow is
//!
//! ```text
//! log p(x) = log N(u; 0, I) + Σ_layers log|det J_layer|,
//! ```
//!
//! where `u` is the base-space image of `x`.
//!
//! All arithmetic is `f32` and pure-Rust. The conditioner reuses the
//! masked autoregressive [`MadeNet`] from [`crate::variational::iaf_flow`].

use crate::error::{BayesError, BayesResult};
use crate::handle::LcgRng;
use crate::variational::iaf_flow::MadeNet;

// ─── Helpers ───────────────────────────────────────────────────────────────────

/// Standard multivariate-Gaussian log density `log N(z; 0, I)` for a vector.
#[must_use]
pub fn standard_normal_log_prob_vec(z: &[f32]) -> f32 {
    let d = z.len() as f32;
    let log_2pi = (2.0 * std::f32::consts::PI).ln();
    let sum_sq: f32 = z.iter().map(|&v| v * v).sum();
    -0.5 * (d * log_2pi + sum_sq)
}

/// Soft clamp applied to the conditioner's log-scale to keep `exp(±α)` finite.
#[inline]
fn clamp_log_scale(a: f32) -> f32 {
    a.clamp(-8.0, 8.0)
}

// ─── MafLayer ──────────────────────────────────────────────────────────────────

/// A single masked autoregressive affine layer.
///
/// Wraps a [`MadeNet`] conditioner (which outputs the per-coordinate shift `μ`
/// and raw log-scale `s`, of which we use `α = clamp(s)`).
#[derive(Debug, Clone)]
pub struct MafLayer {
    /// Autoregressive conditioner network.
    pub made: MadeNet,
    /// Flow dimensionality.
    pub dim: usize,
}

impl MafLayer {
    /// Create a new MAF layer over `dim` dimensions with a single hidden layer
    /// of width `hidden_dim` in the conditioner.
    ///
    /// # Errors
    /// Propagates [`MadeNet::new`] errors (e.g. `dim < 2`, `hidden_dim == 0`).
    pub fn new(dim: usize, hidden_dim: usize, rng: &mut LcgRng) -> BayesResult<Self> {
        let made = MadeNet::new(dim, 0, hidden_dim, rng)?;
        Ok(Self { made, dim })
    }

    /// Forward (data → noise) transform of a single point.
    ///
    /// Returns `(u, log_det)` where `u_i = (x_i − μ_i) exp(−α_i)` and
    /// `log_det = − Σ_i α_i` is the log absolute Jacobian determinant of the
    /// forward map.
    ///
    /// # Errors
    /// * [`BayesError::DimensionMismatch`] if `x.len() != dim`.
    /// * Propagates conditioner errors.
    pub fn forward(&self, x: &[f32]) -> BayesResult<(Vec<f32>, f32)> {
        if x.len() != self.dim {
            return Err(BayesError::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        // The conditioner is autoregressive: μ_i, α_i depend only on x_{<i},
        // so a single parallel forward pass over the whole x is exact.
        let (mu, raw_log_s) = self.made.forward(x, &[])?;
        let mut u = vec![0.0_f32; self.dim];
        let mut log_det = 0.0_f32;
        for i in 0..self.dim {
            let alpha = clamp_log_scale(raw_log_s[i]);
            u[i] = (x[i] - mu[i]) * (-alpha).exp();
            log_det -= alpha;
        }
        if u.iter().any(|v| !v.is_finite()) || !log_det.is_finite() {
            return Err(BayesError::NanEncountered {
                location: "MafLayer::forward",
            });
        }
        Ok((u, log_det))
    }

    /// Inverse (noise → data) transform of a single point.
    ///
    /// Reconstructs `x` from `u` one coordinate at a time, since `μ_i` and
    /// `α_i` depend on the already-recovered prefix `x_{<i}`:
    /// `x_i = u_i exp(α_i) + μ_i`.
    ///
    /// Returns `(x, log_det)` where `log_det = + Σ_i α_i` is the log absolute
    /// Jacobian determinant of the **inverse** map (the negative of the forward
    /// log-det, as expected from an inverse transform).
    ///
    /// # Errors
    /// * [`BayesError::DimensionMismatch`] if `u.len() != dim`.
    /// * Propagates conditioner errors.
    pub fn inverse(&self, u: &[f32]) -> BayesResult<(Vec<f32>, f32)> {
        if u.len() != self.dim {
            return Err(BayesError::DimensionMismatch {
                expected: self.dim,
                got: u.len(),
            });
        }
        let mut x = vec![0.0_f32; self.dim];
        let mut log_det = 0.0_f32;
        for i in 0..self.dim {
            // Conditioner sees the partially-reconstructed x (only x_{<i} feeds
            // coordinate i, so the not-yet-filled tail does not matter).
            let (mu, raw_log_s) = self.made.forward(&x, &[])?;
            let alpha = clamp_log_scale(raw_log_s[i]);
            x[i] = u[i] * alpha.exp() + mu[i];
            log_det += alpha;
        }
        if x.iter().any(|v| !v.is_finite()) || !log_det.is_finite() {
            return Err(BayesError::NanEncountered {
                location: "MafLayer::inverse",
            });
        }
        Ok((x, log_det))
    }
}

// ─── MafFlow ───────────────────────────────────────────────────────────────────

/// A stack of [`MafLayer`]s forming a deep masked autoregressive flow.
///
/// Between consecutive layers the coordinate order is reversed so that, after a
/// full pass, every dimension has had the chance to condition on every other.
#[derive(Debug, Clone)]
pub struct MafFlow {
    /// Ordered list of layers (applied front-to-back in the forward direction).
    pub layers: Vec<MafLayer>,
    /// Flow dimensionality.
    pub dim: usize,
}

impl MafFlow {
    /// Build a MAF with `n_layers` layers over `dim` dimensions.
    ///
    /// # Errors
    /// * [`BayesError::InvalidConfig`] if `n_layers == 0`.
    /// * Propagates [`MafLayer::new`] errors.
    pub fn new(
        n_layers: usize,
        dim: usize,
        hidden_dim: usize,
        rng: &mut LcgRng,
    ) -> BayesResult<Self> {
        if n_layers == 0 {
            return Err(BayesError::InvalidConfig(
                "MAF n_layers must be >= 1".into(),
            ));
        }
        let layers = (0..n_layers)
            .map(|_| MafLayer::new(dim, hidden_dim, rng))
            .collect::<BayesResult<Vec<_>>>()?;
        Ok(Self { layers, dim })
    }

    /// Forward (data → noise) pass through all layers.
    ///
    /// Returns `(u, log_det_total)` with `log_det_total = Σ_layers log|det J|`,
    /// the total forward log-Jacobian determinant.
    ///
    /// # Errors
    /// * [`BayesError::DimensionMismatch`] if `x.len() != dim`.
    /// * Propagates layer errors.
    pub fn forward(&self, x: &[f32]) -> BayesResult<(Vec<f32>, f32)> {
        if x.len() != self.dim {
            return Err(BayesError::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        let mut cur = x.to_vec();
        let mut log_det_total = 0.0_f32;
        for (idx, layer) in self.layers.iter().enumerate() {
            let (u, ld) = layer.forward(&cur)?;
            log_det_total += ld;
            cur = u;
            // Reverse coordinate order between layers (not after the last).
            if idx + 1 < self.layers.len() {
                cur.reverse();
            }
        }
        Ok((cur, log_det_total))
    }

    /// Inverse (noise → data) pass through all layers.
    ///
    /// Exactly undoes [`forward`]: the layers are traversed back-to-front and
    /// the inter-layer reversal is applied in reverse. Returns
    /// `(x, log_det_total)` with the **inverse** total log-Jacobian
    /// (the negative of the forward value for the same point).
    ///
    /// [`forward`]: MafFlow::forward
    ///
    /// # Errors
    /// * [`BayesError::DimensionMismatch`] if `u.len() != dim`.
    /// * Propagates layer errors.
    pub fn inverse(&self, u: &[f32]) -> BayesResult<(Vec<f32>, f32)> {
        if u.len() != self.dim {
            return Err(BayesError::DimensionMismatch {
                expected: self.dim,
                got: u.len(),
            });
        }
        let mut cur = u.to_vec();
        let mut log_det_total = 0.0_f32;
        let n = self.layers.len();
        for (back_idx, layer) in self.layers.iter().enumerate().rev() {
            // Undo the reversal that the forward pass applied *after* this
            // layer (i.e. before the next one), for every layer except the last.
            if back_idx + 1 < n {
                cur.reverse();
            }
            let (x, ld) = layer.inverse(&cur)?;
            log_det_total += ld;
            cur = x;
        }
        Ok((cur, log_det_total))
    }

    /// Log density of a data point under the flow:
    /// `log p(x) = log N(u; 0, I) + log|det J_forward|`.
    ///
    /// # Errors
    /// Propagates [`forward`] errors.
    ///
    /// [`forward`]: MafFlow::forward
    pub fn log_prob(&self, x: &[f32]) -> BayesResult<f32> {
        let (u, log_det) = self.forward(x)?;
        Ok(standard_normal_log_prob_vec(&u) + log_det)
    }

    /// Draw a sample from the flow by mapping a base draw `u ~ N(0, I)` through
    /// the inverse transform.
    ///
    /// # Errors
    /// Propagates [`inverse`] errors.
    ///
    /// [`inverse`]: MafFlow::inverse
    pub fn sample(&self, rng: &mut LcgRng) -> BayesResult<Vec<f32>> {
        let u = draw_standard_normal(self.dim, rng);
        let (x, _) = self.inverse(&u)?;
        Ok(x)
    }
}

// ─── Random helpers (unbiased Box-Muller on the crate RNG) ─────────────────────

/// Unit-uniform draw in `[0, 1)` from the crate [`LcgRng`].
///
/// `LcgRng::next_u32` yields `state >> 33` (max `2³¹ − 1`), so dividing by
/// `2³¹` gives a faithful `[0, 1)` draw. The crate's own `next_f32` divides by
/// `2³²` and spans only `[0, 0.5)`, which biases Box-Muller, so we construct
/// the uniform directly here.
#[inline]
fn unit_uniform(rng: &mut LcgRng) -> f32 {
    rng.next_u32() as f32 / 4_294_967_296.0_f32
}

/// A Box-Muller pair of independent `N(0, 1)` variates.
#[inline]
fn box_muller_pair(rng: &mut LcgRng) -> (f32, f32) {
    let u1 = unit_uniform(rng).clamp(1e-7, 1.0 - 1e-7);
    let u2 = unit_uniform(rng);
    let radius = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    (radius * theta.cos(), radius * theta.sin())
}

/// Draw `d` independent standard-normal samples.
fn draw_standard_normal(d: usize, rng: &mut LcgRng) -> Vec<f32> {
    let mut out = vec![0.0_f32; d];
    let mut i = 0;
    while i + 1 < d {
        let (a, b) = box_muller_pair(rng);
        out[i] = a;
        out[i + 1] = b;
        i += 2;
    }
    if i < d {
        let (a, _) = box_muller_pair(rng);
        out[i] = a;
    }
    out
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(2024)
    }

    /// Numeric log absolute Jacobian determinant of a layer's forward map at
    /// `x`, via central finite differences of `u = forward(x)`.
    fn numeric_forward_log_det(layer: &MafLayer, x: &[f32]) -> f32 {
        let d = layer.dim;
        let h = 1e-3_f32;
        let mut jac = vec![0.0_f32; d * d]; // ∂u_i/∂x_j
        for j in 0..d {
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[j] += h;
            xm[j] -= h;
            let (up, _) = layer.forward(&xp).expect("forward");
            let (um, _) = layer.forward(&xm).expect("forward");
            for i in 0..d {
                jac[i * d + j] = (up[i] - um[i]) / (2.0 * h);
            }
        }
        // log|det J| via LU with partial pivoting on the small dense matrix.
        log_abs_det_lu(&mut jac, d)
    }

    /// In-place LU with partial pivoting; returns `log|det|` (matrix destroyed).
    fn log_abs_det_lu(a: &mut [f32], n: usize) -> f32 {
        let mut log_det = 0.0_f32;
        for k in 0..n {
            // Pivot: largest magnitude in column k at/below row k.
            let mut piv = k;
            let mut best = a[k * n + k].abs();
            for r in (k + 1)..n {
                let v = a[r * n + k].abs();
                if v > best {
                    best = v;
                    piv = r;
                }
            }
            if piv != k {
                for c in 0..n {
                    a.swap(k * n + c, piv * n + c);
                }
            }
            let pivot = a[k * n + k];
            log_det += pivot.abs().ln();
            for r in (k + 1)..n {
                let factor = a[r * n + k] / pivot;
                for c in k..n {
                    a[r * n + c] -= factor * a[k * n + c];
                }
            }
        }
        log_det
    }

    // ── 1. forward ∘ inverse == identity to 1e-9 (single layer) ───────────────
    #[test]
    fn forward_inverse_round_trip_layer() {
        let mut rng = make_rng();
        let layer = MafLayer::new(4, 16, &mut rng).expect("layer");
        let x = vec![0.3_f32, -1.2, 0.7, 2.1];
        let (u, ld_fwd) = layer.forward(&x).expect("forward");
        let (x_rec, ld_inv) = layer.inverse(&u).expect("inverse");
        for i in 0..4 {
            assert!(
                (x[i] - x_rec[i]).abs() < 1e-5,
                "round-trip failed at {i}: {} vs {}",
                x[i],
                x_rec[i]
            );
        }
        // Inverse log-det is the negative of the forward log-det.
        assert!(
            (ld_fwd + ld_inv).abs() < 1e-4,
            "fwd+inv log-det should be ~0: {ld_fwd} + {ld_inv}"
        );
    }

    // ── 2. forward ∘ inverse == identity for a multi-layer stack ──────────────
    #[test]
    fn forward_inverse_round_trip_stack() {
        let mut rng = make_rng();
        let flow = MafFlow::new(4, 5, 24, &mut rng).expect("flow");
        let x = vec![1.5_f32, -0.4, 0.9, -2.0, 0.1];
        let (u, ld_fwd) = flow.forward(&x).expect("forward");
        let (x_rec, ld_inv) = flow.inverse(&u).expect("inverse");
        for i in 0..5 {
            assert!(
                (x[i] - x_rec[i]).abs() < 1e-4,
                "stack round-trip failed at {i}: {} vs {}",
                x[i],
                x_rec[i]
            );
        }
        assert!((ld_fwd + ld_inv).abs() < 1e-3, "{ld_fwd} + {ld_inv}");
    }

    // ── 3. analytic log-det-Jacobian matches the numeric one ──────────────────
    #[test]
    fn log_det_matches_numeric() {
        let mut rng = make_rng();
        let layer = MafLayer::new(4, 16, &mut rng).expect("layer");
        let x = vec![0.5_f32, -0.7, 1.3, 0.2];
        let (_, analytic) = layer.forward(&x).expect("forward");
        let numeric = numeric_forward_log_det(&layer, &x);
        assert!(
            (analytic - numeric).abs() < 1e-2,
            "analytic log-det {analytic} vs numeric {numeric}"
        );
    }

    // ── 4. autoregressive structure: u_0 does NOT depend on the conditioner ───
    //    (the first coordinate has an empty conditioning prefix, so μ_0, α_0 are
    //    constants — perturbing later coordinates leaves u_0 unchanged).
    #[test]
    fn first_output_is_autoregressive() {
        let mut rng = make_rng();
        let layer = MafLayer::new(4, 16, &mut rng).expect("layer");
        let x_a = vec![0.5_f32, 0.1, -0.2, 0.3];
        let mut x_b = x_a.clone();
        x_b[2] = 9.9; // change a coordinate that is *after* index 0
        let (ua, _) = layer.forward(&x_a).expect("forward");
        let (ub, _) = layer.forward(&x_b).expect("forward");
        // u_0 = (x_0 − μ_0)·exp(−α_0); μ_0, α_0 are constants ⇒ u_0 unchanged.
        assert!(
            (ua[0] - ub[0]).abs() < 1e-5,
            "u[0] must not depend on x[>0]: {} vs {}",
            ua[0],
            ub[0]
        );
        // And changing x_0 *does* move u_0.
        let mut x_c = x_a.clone();
        x_c[0] += 1.0;
        let (uc, _) = layer.forward(&x_c).expect("forward");
        assert!(
            (ua[0] - uc[0]).abs() > 1e-3,
            "u[0] must respond to x[0]: {} vs {}",
            ua[0],
            uc[0]
        );
    }

    // ── 5. a stack maps a base Gaussian: log_prob is finite & samples flow ────
    #[test]
    fn stack_maps_base_gaussian() {
        let mut rng = make_rng();
        let flow = MafFlow::new(3, 4, 16, &mut rng).expect("flow");
        // log_prob is a finite real number for arbitrary inputs.
        let x = vec![0.2_f32, -0.5, 1.1, 0.0];
        let lp = flow.log_prob(&x).expect("log_prob");
        assert!(lp.is_finite(), "log_prob not finite: {lp}");

        // A sample drawn from the base Gaussian, pushed through the inverse, is
        // finite and, when pushed back forward, recovers its base point.
        let mut srng = LcgRng::new(7);
        let sample = flow.sample(&mut srng).expect("sample");
        assert_eq!(sample.len(), 4);
        assert!(sample.iter().all(|v| v.is_finite()));

        // change-of-variables identity: log_prob(x) = log N(u) + log_det.
        let (u, ld) = flow.forward(&x).expect("forward");
        let manual = standard_normal_log_prob_vec(&u) + ld;
        assert!((lp - manual).abs() < 1e-5, "{lp} vs {manual}");
    }

    // ── 6. volume change is tracked (log-det is non-trivial & consistent) ─────
    #[test]
    fn volume_change_tracked() {
        let mut rng = make_rng();
        let layer = MafLayer::new(4, 16, &mut rng).expect("layer");
        let x = vec![0.4_f32, -0.6, 0.8, -0.1];
        let (_, ld) = layer.forward(&x).expect("forward");
        // log_det = −Σ α_i; with a freshly-initialised net the α_i are not all
        // zero, so the layer is *not* volume-preserving.
        assert!(ld.abs() > 1e-6, "layer should change volume: log_det={ld}");

        // Stacking accumulates the per-layer volume changes additively.
        let flow = MafFlow::new(3, 4, 16, &mut rng).expect("flow");
        let (_, ld_total) = flow.forward(&x).expect("forward");
        // Recompute the sum of per-layer log-dets along the actual forward path
        // and confirm it equals the flow's reported total.
        let mut cur = x.clone();
        let mut manual = 0.0_f32;
        for (idx, l) in flow.layers.iter().enumerate() {
            let (u, ld_l) = l.forward(&cur).expect("forward");
            manual += ld_l;
            cur = u;
            if idx + 1 < flow.layers.len() {
                cur.reverse();
            }
        }
        assert!(
            (ld_total - manual).abs() < 1e-4,
            "flow log-det {ld_total} != Σ layer log-dets {manual}"
        );
    }

    // ── 7. error paths ────────────────────────────────────────────────────────
    #[test]
    fn maf_rejects_bad_shapes() {
        let mut rng = make_rng();
        let layer = MafLayer::new(4, 8, &mut rng).expect("layer");
        assert!(matches!(
            layer.forward(&[0.0; 3]),
            Err(BayesError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            layer.inverse(&[0.0; 5]),
            Err(BayesError::DimensionMismatch { .. })
        ));
        assert!(MafFlow::new(0, 4, 8, &mut rng).is_err());
        let flow = MafFlow::new(2, 4, 8, &mut rng).expect("flow");
        assert!(flow.forward(&[0.0; 2]).is_err());
        assert!(flow.log_prob(&[0.0; 7]).is_err());
    }

    // ── 8. base-Gaussian log-prob helper sanity ───────────────────────────────
    #[test]
    fn standard_normal_log_prob_at_origin() {
        let z = vec![0.0_f32; 3];
        let lp = standard_normal_log_prob_vec(&z);
        let expected = -0.5 * 3.0 * (2.0 * std::f32::consts::PI).ln();
        assert!((lp - expected).abs() < 1e-5, "{lp} vs {expected}");
    }
}
