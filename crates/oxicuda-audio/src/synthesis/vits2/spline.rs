//! Monotonic rational-quadratic (RQ) neural spline flows
//! ([Durkan et al. 2019](https://arxiv.org/abs/1906.04032), *Neural Spline
//! Flows*).
//!
//! A rational-quadratic spline is the most expressive **invertible, monotone,
//! analytically differentiable** scalar transform commonly used as a flow
//! building block. On a bounded interval `[-B, B]` it is a piecewise function of
//! `K` *bins*, each a ratio of two quadratics, glued at `K + 1` *knots* so that
//! the result is `C¹` (value **and** first derivative continuous). Outside
//! `[-B, B]` it is the identity (slope-`1`) tail, so the whole map is a strictly
//! increasing bijection `ℝ → ℝ` — the *unconstrained / linear-tails* variant.
//!
//! ## Parameterisation
//!
//! Each spline is built from three unnormalised parameter vectors:
//!
//! * `K` widths `θ_w` → `softmax` → bin widths summing to the interval length
//!   `2B` (so the cumulative knots run `x₀ = −B … x_K = +B`);
//! * `K` heights `θ_h` → `softmax` → bin heights summing to `2B`
//!   (`y₀ = −B … y_K = +B`);
//! * `K − 1` **internal** derivatives `θ_δ` → `softplus` → strictly positive
//!   knot slopes `δ₁ … δ_{K−1}`. The two **boundary** derivatives are pinned to
//!   `δ₀ = δ_K = 1` so the spline matches the slope-`1` identity tail exactly,
//!   guaranteeing `C¹` continuity at `±B`.
//!
//! Small floors (`min` bin width/height and `min` derivative) keep every bin
//! non-degenerate and the inverse numerically robust, exactly as in Durkan's
//! reference implementation.
//!
//! ## Transform (Durkan eq. 4–5)
//!
//! For `x` in bin `k` (`x_k ≤ x < x_{k+1}`) write `ξ = (x − x_k)/w_k` with bin
//! width `w_k`, height `h_k`, slope `s_k = h_k / w_k`, and knot derivatives
//! `δ_k, δ_{k+1}`:
//!
//! ```text
//!           h_k · ( s_k ξ² + δ_k ξ(1−ξ) )
//! y = y_k + ─────────────────────────────────────
//!            s_k + (δ_{k+1} + δ_k − 2 s_k) ξ(1−ξ)
//!
//!          s_k² ( δ_{k+1} ξ² + 2 s_k ξ(1−ξ) + δ_k (1−ξ)² )
//! dy/dx = ─────────────────────────────────────────────────
//!          ( s_k + (δ_{k+1} + δ_k − 2 s_k) ξ(1−ξ) )²
//! ```
//!
//! The denominator is strictly positive and the numerator of `dy/dx` is a
//! positive combination, so `dy/dx > 0` everywhere — the map is **strictly
//! increasing** and its log-determinant `log(dy/dx)` is always finite.
//!
//! The inverse solves the per-bin quadratic `a ξ² + b ξ + c = 0` (Durkan
//! appendix C) for the unique root `ξ ∈ [0, 1]`, then `x = x_k + ξ w_k`.
//!
//! ## Use in VITS2
//!
//! [`RqSplineCoupling`] wraps the scalar spline into a Durkan **coupling layer**:
//! the channel axis is split into an identity half and a transformed half; a
//! conditioner reads the identity half (plus an external condition) and emits the
//! per-element `(3K − 1)` spline parameters of the transformed half. Because the
//! conditioner never sees the half it transforms, the layer inverts exactly and
//! its Jacobian is triangular with log-determinant `Σ log(dy/dx)`.
//!
//! In the stochastic duration predictor this is the *monotone spline
//! dequantiser*: it maps the auxiliary base noise `u ~ N(0, 1)` to the
//! dequantisation variable `e = T(u)` conditioned on `log d` and the text — the
//! real rational-quadratic replacement for the previous fixed-`N(0, 1)`
//! auxiliary path (Kim et al. 2021, the SDP posterior spline flow).

use crate::error::{AudioError, AudioResult};
use crate::handle::LcgRng;
use crate::synthesis::vits2::common::DenseLayer;

/// Minimum normalised bin width (fraction of the interval).
const MIN_BIN_WIDTH: f32 = 1e-3;
/// Minimum normalised bin height (fraction of the interval).
const MIN_BIN_HEIGHT: f32 = 1e-3;
/// Minimum knot derivative (added after `softplus`).
const MIN_DERIVATIVE: f32 = 1e-3;

/// Numerically stable softplus `log(1 + exp(x))`.
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        // exp(x) overflows f32 well before this; the asymptote is exact here.
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Numerically stable softmax over a slice (subtracts the max before `exp`).
fn softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = logits.iter().map(|&v| (v - m).exp()).collect();
    let sum: f32 = out.iter().sum();
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for v in out.iter_mut() {
        *v *= inv;
    }
    out
}

/// Solve `a ξ² + b ξ + c = 0` for the unique root in `[0, 1]`.
///
/// The forward spline restricted to a bin is a strictly monotone bijection
/// `[0, 1] → [y_k, y_{k+1}]`, so for any target inside that range exactly one of
/// the two quadratic roots lies in `[0, 1]`; this routine returns it, clamped to
/// guard against round-off, and stays finite for every input (the `a ≈ 0` linear
/// case and a clamped discriminant are handled explicitly).
fn solve_quadratic_unit(a: f32, b: f32, c: f32) -> f32 {
    const EPS: f32 = 1e-12;
    if a.abs() < EPS {
        // Degenerate to the linear equation `b ξ + c = 0`.
        if b.abs() < EPS {
            return 0.0;
        }
        return (-c / b).clamp(0.0, 1.0);
    }
    let disc = (b * b - 4.0 * a * c).max(0.0);
    let sq = disc.sqrt();
    let two_a = 2.0 * a;
    let root_minus = (-b - sq) / two_a;
    let root_plus = (-b + sq) / two_a;
    // A small slack absorbs f32 round-off right at the bin edges.
    let in_minus = (-1e-4..=1.0 + 1e-4).contains(&root_minus);
    let in_plus = (-1e-4..=1.0 + 1e-4).contains(&root_plus);
    let root = if in_minus && !in_plus {
        root_minus
    } else if in_plus && !in_minus {
        root_plus
    } else if in_minus && in_plus {
        // Both nominally valid (a near-linear bin): either is correct.
        root_minus
    } else {
        // Neither strictly inside; keep whichever sits closest to the interval.
        let d_minus = (root_minus - root_minus.clamp(0.0, 1.0)).abs();
        let d_plus = (root_plus - root_plus.clamp(0.0, 1.0)).abs();
        if d_minus <= d_plus {
            root_minus
        } else {
            root_plus
        }
    };
    root.clamp(0.0, 1.0)
}

/// Index of the bin `k` with `knots[k] <= v < knots[k+1]`, clamped to the last
/// bin for `v` at or beyond the final knot. `knots` holds `K + 1` increasing
/// entries; the returned `k` is in `0..K`.
fn bin_index(knots: &[f32], v: f32) -> usize {
    let n = knots.len(); // K + 1
    let mut k = 0usize;
    // Stop once `k` reaches the last bin (`K - 1`, i.e. `k + 2 == n`).
    while k + 2 < n && v >= knots[k + 1] {
        k += 1;
    }
    k
}

// ─── Scalar rational-quadratic spline ────────────────────────────────────────

/// A monotone rational-quadratic spline on `[-bound, bound]` with identity tails.
///
/// Construct it from unnormalised width/height/derivative parameters via
/// [`RationalQuadraticSpline::new`]; evaluate the exact bijection with
/// [`RationalQuadraticSpline::forward`] (returning `(y, log dy/dx)`) and
/// [`RationalQuadraticSpline::inverse`].
#[derive(Debug, Clone)]
pub struct RationalQuadraticSpline {
    /// Interval half-width `B` (the spline acts on `[-B, B]`).
    bound: f32,
    /// `K + 1` cumulative input knots, `x₀ = −B … x_K = +B`.
    x_knots: Vec<f32>,
    /// `K + 1` cumulative output knots, `y₀ = −B … y_K = +B`.
    y_knots: Vec<f32>,
    /// `K + 1` knot derivatives, with `δ₀ = δ_K = 1` (tail-matching).
    derivs: Vec<f32>,
}

impl RationalQuadraticSpline {
    /// Build a spline of `K = widths.len()` bins on `[-bound, bound]`.
    ///
    /// `widths` and `heights` are unnormalised logits (mapped through `softmax`);
    /// `derivatives` are the `K − 1` unnormalised **internal** knot slopes
    /// (mapped through `softplus`). The boundary slopes are fixed to `1`.
    ///
    /// # Errors
    ///
    /// - [`AudioError::Internal`] when `K == 0` or `bound` is not finite/positive,
    ///   or when the bin floor would over-subscribe the interval
    ///   (`K · MIN_BIN_WIDTH >= 1`).
    /// - [`AudioError::ShapeMismatch`] when `heights.len() != K` or
    ///   `derivatives.len() != K − 1`.
    pub fn new(
        widths: &[f32],
        heights: &[f32],
        derivatives: &[f32],
        bound: f32,
    ) -> AudioResult<Self> {
        let k = widths.len();
        if k == 0 {
            return Err(AudioError::Internal("RQ spline: K == 0".into()));
        }
        if !bound.is_finite() || bound <= 0.0 {
            return Err(AudioError::Internal(format!(
                "RQ spline: bad bound {bound}"
            )));
        }
        if heights.len() != k {
            return Err(AudioError::ShapeMismatch {
                msg: format!("RQ spline: heights.len()={} != K={k}", heights.len()),
            });
        }
        if derivatives.len() + 1 != k {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "RQ spline: derivatives.len()={} != K-1={}",
                    derivatives.len(),
                    k - 1
                ),
            });
        }
        if MIN_BIN_WIDTH * k as f32 >= 1.0 || MIN_BIN_HEIGHT * k as f32 >= 1.0 {
            return Err(AudioError::Internal(format!(
                "RQ spline: too many bins ({k}) for the bin floor"
            )));
        }

        let span = 2.0 * bound;
        let x_knots = Self::cumulative_knots(widths, MIN_BIN_WIDTH, span, bound, k);
        let y_knots = Self::cumulative_knots(heights, MIN_BIN_HEIGHT, span, bound, k);

        let mut derivs = Vec::with_capacity(k + 1);
        derivs.push(1.0); // δ₀ — matches the lower identity tail.
        for &d in derivatives {
            derivs.push(softplus(d) + MIN_DERIVATIVE);
        }
        derivs.push(1.0); // δ_K — matches the upper identity tail.

        Ok(Self {
            bound,
            x_knots,
            y_knots,
            derivs,
        })
    }

    /// Build `K + 1` cumulative knots from unnormalised `logits`, applying the
    /// `min_frac` floor and scaling the normalised widths to `span`. The first
    /// and last knots are pinned to `-bound` and `+bound` exactly.
    fn cumulative_knots(
        logits: &[f32],
        min_frac: f32,
        span: f32,
        bound: f32,
        k: usize,
    ) -> Vec<f32> {
        let mut frac = softmax(logits);
        let scale = 1.0 - min_frac * k as f32;
        for f in frac.iter_mut() {
            *f = min_frac + scale * *f;
        }
        let mut knots = Vec::with_capacity(k + 1);
        let mut acc = -bound;
        knots.push(acc);
        for &f in &frac {
            acc += f * span;
            knots.push(acc);
        }
        // Pin the endpoints exactly so the identity tails meet C¹.
        knots[0] = -bound;
        knots[k] = bound;
        knots
    }

    /// Interval half-width `B` (the spline transforms `[-B, B]`).
    #[must_use]
    pub fn bound(&self) -> f32 {
        self.bound
    }

    /// Number of bins `K`.
    #[must_use]
    pub fn num_bins(&self) -> usize {
        self.x_knots.len() - 1
    }

    /// Evaluate bin `k` at local coordinate `ξ ∈ [0, 1]`, returning the
    /// rational-quadratic value `y` and its derivative `dy/dx` (Durkan eq. 4–5).
    fn eval_bin(&self, k: usize, xi: f32) -> (f32, f32) {
        let xk = self.x_knots[k];
        let xk1 = self.x_knots[k + 1];
        let yk = self.y_knots[k];
        let yk1 = self.y_knots[k + 1];
        let dk = self.derivs[k];
        let dk1 = self.derivs[k + 1];

        let w = xk1 - xk;
        let h = yk1 - yk;
        let s = h / w;
        let one_m = 1.0 - xi;
        let xi_xi = xi * one_m; // ξ(1−ξ)
        let denom = s + (dk1 + dk - 2.0 * s) * xi_xi;

        let y = yk + h * (s * xi * xi + dk * xi_xi) / denom;
        let deriv_num = s * s * (dk1 * xi * xi + 2.0 * s * xi_xi + dk * one_m * one_m);
        let dydx = deriv_num / (denom * denom);
        (y, dydx)
    }

    /// Forward transform `x ↦ (y, log dy/dx)`.
    ///
    /// Inside `[-B, B]` this is the rational-quadratic map; outside it is the
    /// identity (`y = x`, `log dy/dx = 0`). The returned log-derivative is the
    /// exact analytic `log(dy/dx)` used as the flow log-determinant.
    #[must_use]
    pub fn forward(&self, x: f32) -> (f32, f32) {
        if x <= -self.bound || x >= self.bound {
            return (x, 0.0); // identity tail, slope 1 → log-derivative 0.
        }
        let k = bin_index(&self.x_knots, x);
        let w = self.x_knots[k + 1] - self.x_knots[k];
        let xi = (x - self.x_knots[k]) / w;
        let (y, dydx) = self.eval_bin(k, xi);
        (y, dydx.ln())
    }

    /// Inverse transform `y ↦ x`, the exact inverse of
    /// [`RationalQuadraticSpline::forward`].
    ///
    /// The per-bin quadratic (Durkan appendix C) gives the root analytically; one
    /// or two Newton steps on the exact forward map then polish away the f32
    /// cancellation error so the round trip is accurate to machine precision.
    #[must_use]
    pub fn inverse(&self, y: f32) -> f32 {
        if y <= -self.bound || y >= self.bound {
            return y; // identity tail.
        }
        let k = bin_index(&self.y_knots, y);
        let xk = self.x_knots[k];
        let w = self.x_knots[k + 1] - xk;
        let h = self.y_knots[k + 1] - self.y_knots[k];
        let s = h / w;
        let dy = y - self.y_knots[k];
        let delta = self.derivs[k + 1] + self.derivs[k] - 2.0 * s;

        // a ξ² + b ξ + c = 0  (Durkan appendix C) → analytic initial guess.
        let a = h * (s - self.derivs[k]) + dy * delta;
        let b = h * self.derivs[k] - dy * delta;
        let c = -s * dy;
        let mut xi = solve_quadratic_unit(a, b, c);

        // Newton polish: ξ ← ξ − (y(ξ) − y) / (w · dy/dx).
        for _ in 0..2 {
            let (y_xi, dydx) = self.eval_bin(k, xi);
            let dydxi = dydx * w;
            if dydxi.abs() > 1e-20 {
                xi = (xi - (y_xi - y) / dydxi).clamp(0.0, 1.0);
            }
        }
        xk + xi * w
    }
}

// ─── Rational-quadratic spline coupling ──────────────────────────────────────

/// A Durkan rational-quadratic **spline coupling** over a `[t, dim]` sequence.
///
/// The channels are split into an identity half `[0, dim_a)` and a transformed
/// half `[dim_a, dim)`. A two-layer `tanh`-MLP conditioner reads the identity
/// half together with an external per-step condition `g` and emits, for every
/// transformed channel, the `3K − 1` unnormalised parameters of a
/// [`RationalQuadraticSpline`]. The transformed channels are pushed through their
/// per-element splines; the identity half is copied through unchanged. Because
/// the conditioner never reads the transformed half, the layer inverts exactly
/// and its Jacobian is triangular with log-determinant `Σ log(dy/dx)`.
#[derive(Debug, Clone)]
pub struct RqSplineCoupling {
    /// Conditioner first layer `(dim_a + cond_dim) → hidden`.
    fc1: DenseLayer,
    /// Conditioner second layer `hidden → dim_b · (3K − 1)`.
    fc2: DenseLayer,
    /// Identity-half channel count.
    dim_a: usize,
    /// Transformed-half channel count.
    dim_b: usize,
    /// Total channel count (`dim_a + dim_b`).
    dim: usize,
    /// External condition feature width.
    cond_dim: usize,
    /// Bins per spline `K`.
    num_bins: usize,
    /// Spline interval half-width `B`.
    bound: f32,
}

impl RqSplineCoupling {
    /// Construct a spline coupling over `dim` (`>= 2`) channels conditioned on a
    /// `cond_dim`-wide external signal, with `num_bins` (`>= 1`) bins per spline
    /// on `[-bound, bound]`.
    ///
    /// # Errors
    ///
    /// - [`AudioError::InvalidEmbedDim`] when `dim < 2`.
    /// - [`AudioError::Internal`] when `cond_dim`, `hidden`, or `num_bins` is `0`,
    ///   or `bound` is not finite/positive.
    pub fn new(
        dim: usize,
        cond_dim: usize,
        hidden: usize,
        num_bins: usize,
        bound: f32,
        rng: &mut LcgRng,
    ) -> AudioResult<Self> {
        if dim < 2 {
            return Err(AudioError::InvalidEmbedDim(dim));
        }
        if cond_dim == 0 {
            return Err(AudioError::Internal("RQ coupling: cond_dim == 0".into()));
        }
        if hidden == 0 {
            return Err(AudioError::Internal("RQ coupling: hidden == 0".into()));
        }
        if num_bins == 0 {
            return Err(AudioError::Internal("RQ coupling: num_bins == 0".into()));
        }
        if !bound.is_finite() || bound <= 0.0 {
            return Err(AudioError::Internal(format!(
                "RQ coupling: bad bound {bound}"
            )));
        }
        let dim_a = dim / 2;
        let dim_b = dim - dim_a;
        let params_per_elem = 3 * num_bins - 1;
        let s1 = (2.0 / (dim_a + cond_dim) as f32).sqrt();
        // A small second-layer scale keeps the initial splines gentle (near the
        // uniform-bin map) so the coupling is well-conditioned from the start.
        let s2 = 0.3 / (hidden as f32).sqrt();
        Ok(Self {
            fc1: DenseLayer::new(dim_a + cond_dim, hidden, s1, rng),
            fc2: DenseLayer::new(hidden, dim_b * params_per_elem, s2, rng),
            dim_a,
            dim_b,
            dim,
            cond_dim,
            num_bins,
            bound,
        })
    }

    /// Spline interval half-width `B`.
    #[must_use]
    pub fn bound(&self) -> f32 {
        self.bound
    }

    /// Bins per spline `K`.
    #[must_use]
    pub fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Total channel count this coupling operates on.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Unnormalised parameters per transformed element (`3K − 1`).
    fn params_per_elem(&self) -> usize {
        3 * self.num_bins - 1
    }

    /// Run the conditioner over the identity half `x_a` and condition `g`,
    /// returning the `[t, dim_b · (3K − 1)]` spline-parameter tensor.
    fn conditioner(&self, x: &[f32], g: &[f32], t: usize) -> Vec<f32> {
        let in_dim = self.dim_a + self.cond_dim;
        let mut inp = vec![0.0_f32; t * in_dim];
        for ti in 0..t {
            let row = &mut inp[ti * in_dim..(ti + 1) * in_dim];
            let src = &x[ti * self.dim..ti * self.dim + self.dim_a];
            row[..self.dim_a].copy_from_slice(src);
            row[self.dim_a..].copy_from_slice(&g[ti * self.cond_dim..(ti + 1) * self.cond_dim]);
        }
        let mut h = self.fc1.forward(&inp, t);
        for v in h.iter_mut() {
            *v = v.tanh();
        }
        self.fc2.forward(&h, t)
    }

    /// Build the spline for transformed channel `j` from a per-step parameter row.
    fn spline_for(&self, params_row: &[f32], j: usize) -> AudioResult<RationalQuadraticSpline> {
        let k = self.num_bins;
        let base = j * self.params_per_elem();
        let widths = &params_row[base..base + k];
        let heights = &params_row[base + k..base + 2 * k];
        let derivs = &params_row[base + 2 * k..base + 3 * k - 1];
        RationalQuadraticSpline::new(widths, heights, derivs, self.bound)
    }

    /// Validate the `[t, dim]` input and `[t, cond_dim]` condition shapes.
    fn check(&self, x: &[f32], g: &[f32], t: usize) -> AudioResult<()> {
        if t == 0 {
            return Err(AudioError::EmptyInput {
                msg: "RQ coupling: t == 0".into(),
            });
        }
        if x.len() != t * self.dim {
            return Err(AudioError::ShapeMismatch {
                msg: format!("RQ coupling: x.len()={} != t*dim={}", x.len(), t * self.dim),
            });
        }
        if g.len() != t * self.cond_dim {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "RQ coupling: g.len()={} != t*cond_dim={}",
                    g.len(),
                    t * self.cond_dim
                ),
            });
        }
        Ok(())
    }

    /// Forward transform `x [t, dim] → (y [t, dim], logdet)` given condition `g`.
    ///
    /// The identity half is copied through; each transformed channel is mapped by
    /// its per-element rational-quadratic spline. `logdet` is the exact
    /// `Σ log(dy/dx)` over the transformed half (the identity half contributes 0).
    ///
    /// # Errors
    ///
    /// [`AudioError::EmptyInput`] / [`AudioError::ShapeMismatch`] for bad shapes;
    /// propagates spline-construction errors.
    pub fn forward(&self, x: &[f32], g: &[f32], t: usize) -> AudioResult<(Vec<f32>, f32)> {
        self.check(x, g, t)?;
        let params = self.conditioner(x, g, t);
        let ppe = self.params_per_elem();
        let mut y = x.to_vec();
        let mut logdet = 0.0_f32;
        for ti in 0..t {
            let row = &params[ti * self.dim_b * ppe..(ti + 1) * self.dim_b * ppe];
            for j in 0..self.dim_b {
                let spline = self.spline_for(row, j)?;
                let ch = self.dim_a + j;
                let (yv, ld) = spline.forward(x[ti * self.dim + ch]);
                y[ti * self.dim + ch] = yv;
                logdet += ld;
            }
        }
        Ok((y, logdet))
    }

    /// Inverse transform `y [t, dim] → x [t, dim]`, the exact inverse of
    /// [`RqSplineCoupling::forward`].
    ///
    /// # Errors
    ///
    /// [`AudioError::EmptyInput`] / [`AudioError::ShapeMismatch`] for bad shapes;
    /// propagates spline-construction errors.
    pub fn inverse(&self, y: &[f32], g: &[f32], t: usize) -> AudioResult<Vec<f32>> {
        self.check(y, g, t)?;
        // The conditioner reads only the identity half, which forward left
        // untouched (`y_a == x_a`), so it reproduces the exact same parameters.
        let params = self.conditioner(y, g, t);
        let ppe = self.params_per_elem();
        let mut x = y.to_vec();
        for ti in 0..t {
            let row = &params[ti * self.dim_b * ppe..(ti + 1) * self.dim_b * ppe];
            for j in 0..self.dim_b {
                let spline = self.spline_for(row, j)?;
                let ch = self.dim_a + j;
                x[ti * self.dim + ch] = spline.inverse(y[ti * self.dim + ch]);
            }
        }
        Ok(x)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a deterministic `K`-bin spline on `[-bound, bound]` from a seed.
    ///
    /// `logit_scale` controls how non-uniform the bins are: a larger value gives
    /// a more strongly volume-changing spline (a harder bijection stress test);
    /// a smaller value keeps the per-bin derivatives well-conditioned (so a
    /// finite-difference derivative check is not swamped by f32 round-off).
    fn make_spline(k: usize, bound: f32, seed: u64, logit_scale: f32) -> RationalQuadraticSpline {
        let mut rng = LcgRng::new(seed);
        let mut widths = vec![0.0_f32; k];
        let mut heights = vec![0.0_f32; k];
        let mut derivs = vec![0.0_f32; k - 1];
        rng.fill_normal(&mut widths);
        rng.fill_normal(&mut heights);
        rng.fill_normal(&mut derivs);
        for v in widths.iter_mut() {
            *v *= logit_scale;
        }
        for v in heights.iter_mut() {
            *v *= logit_scale;
        }
        RationalQuadraticSpline::new(&widths, &heights, &derivs, bound).expect("spline")
    }

    /// `log|det J|` of a square `[n, n]` matrix via Gaussian elimination with
    /// partial pivoting (test-only reference for the finite-difference check).
    fn log_abs_det(mut m: Vec<f32>, n: usize) -> f32 {
        let mut log_det = 0.0_f32;
        for col in 0..n {
            let mut pivot = col;
            let mut best = m[col * n + col].abs();
            for r in (col + 1)..n {
                let v = m[r * n + col].abs();
                if v > best {
                    best = v;
                    pivot = r;
                }
            }
            if pivot != col {
                for c in 0..n {
                    m.swap(col * n + c, pivot * n + c);
                }
            }
            let diag = m[col * n + col];
            log_det += diag.abs().ln();
            for r in (col + 1)..n {
                let factor = m[r * n + col] / diag;
                for c in col..n {
                    m[r * n + c] -= factor * m[col * n + c];
                }
            }
        }
        log_det
    }

    #[test]
    fn spline_is_a_bijection_inside_and_in_tails() {
        // TEST 1: inverse(forward(x)) ≈ x and forward(inverse(y)) ≈ y to <= 1e-4,
        // sampling inside [-B, B] and in both tails.
        let bound = 4.0_f32;
        let spline = make_spline(10, bound, 1, 1.5);
        let mut rng = LcgRng::new(123);
        let mut max_fwd_err = 0.0_f32;
        let mut max_inv_err = 0.0_f32;
        for _ in 0..2000 {
            // Range [-1.5B, 1.5B] so ~1/3 of the draws land in the identity tails.
            let x = (rng.next_f32() * 2.0 - 1.0) * 1.5 * bound;
            let (y, _ld) = spline.forward(x);
            let back = spline.inverse(y);
            max_fwd_err = max_fwd_err.max((back - x).abs());
            // And the other direction from an independent point.
            let yv = (rng.next_f32() * 2.0 - 1.0) * 1.5 * bound;
            let xv = spline.inverse(yv);
            let (yy, _) = spline.forward(xv);
            max_inv_err = max_inv_err.max((yy - yv).abs());
        }
        assert!(max_fwd_err < 1e-4, "inverse∘forward err {max_fwd_err}");
        assert!(max_inv_err < 1e-4, "forward∘inverse err {max_inv_err}");
    }

    #[test]
    fn spline_is_strictly_increasing() {
        // TEST 1 (monotonicity): forward strictly increasing and dy/dx > 0.
        let bound = 5.0_f32;
        let spline = make_spline(12, bound, 7, 1.5);
        let mut prev_y = f32::NEG_INFINITY;
        let steps = 4000;
        for i in 0..=steps {
            // Grid spanning the tails as well as the interval.
            let x = -1.3 * bound + (2.6 * bound) * (i as f32 / steps as f32);
            let (y, ld) = spline.forward(x);
            assert!(y.is_finite() && ld.is_finite());
            assert!(y > prev_y, "not increasing at x={x}: {y} <= {prev_y}");
            assert!(ld.exp() > 0.0, "non-positive derivative at x={x}");
            prev_y = y;
        }
    }

    #[test]
    fn spline_logdet_matches_finite_difference() {
        // TEST 2 (scalar): analytic log(dy/dx) ≈ log of the central difference,
        // to <= 1e-3, across a grid of x inside [-B, B].
        let bound = 4.0_f32;
        // A well-conditioned (gentler) spline so the f32 finite difference is not
        // swamped by round-off amplified through `log`; it is still genuinely
        // non-uniform (a real rational-quadratic, not the identity).
        let spline = make_spline(8, bound, 11, 0.7);
        // Step near the f32 optimum (truncation ∝ h², round-off ∝ ε/h).
        let h = 2e-3_f32;
        let steps = 400;
        let mut max_err = 0.0_f32;
        let mut checked = 0usize;
        for i in 1..steps {
            let x = -bound + 2.0 * bound * (i as f32 / steps as f32);
            // Stay inside the interval, away from the tail seams.
            if (x.abs() - bound).abs() < 0.05 {
                continue;
            }
            // A central difference only approximates the derivative where the
            // spline is locally C²; skip stencils [x−2h, x+2h] straddling a C¹
            // knot seam, where the (correct) analytic slope is discontinuous in
            // its own derivative and the finite difference is a poor estimator.
            if spline.x_knots.iter().any(|&xk| (xk - x).abs() < 2.0 * h) {
                continue;
            }
            let (_y, analytic) = spline.forward(x);
            let (yp, _) = spline.forward(x + h);
            let (ym, _) = spline.forward(x - h);
            let numeric = ((yp - ym) / (2.0 * h)).ln();
            max_err = max_err.max((analytic - numeric).abs());
            checked += 1;
        }
        assert!(checked > 50, "too few interior points checked: {checked}");
        assert!(max_err < 1e-3, "scalar logdet vs finite-diff err {max_err}");
    }

    #[test]
    fn spline_tails_are_c1_continuous() {
        // TEST 3: at ±B the spline matches the identity-linear tail in value and
        // derivative.
        let bound = 3.0_f32;
        let spline = make_spline(9, bound, 5, 1.5);

        // Structural: the boundary knots and slopes are pinned to the tail.
        let k = spline.num_bins();
        assert_eq!(spline.x_knots[0], -bound);
        assert_eq!(spline.x_knots[k], bound);
        assert_eq!(spline.y_knots[0], -bound);
        assert_eq!(spline.y_knots[k], bound);
        assert_eq!(spline.derivs[0], 1.0);
        assert_eq!(spline.derivs[k], 1.0);

        // Numeric: approaching ±B from inside, value → ±B and dy/dx → 1, matching
        // the identity tail just outside.
        let eps = 1e-4_f32;
        for &edge in &[bound, -bound] {
            let inside = edge - edge.signum() * eps;
            let (y_in, ld_in) = spline.forward(inside);
            assert!((y_in - inside).abs() < 1e-3, "value jump at {edge}: {y_in}");
            assert!((ld_in.exp() - 1.0).abs() < 1e-2, "slope ≠ 1 at {edge}");
            // Just outside is exactly the identity.
            let outside = edge + edge.signum() * eps;
            let (y_out, ld_out) = spline.forward(outside);
            assert_eq!(y_out, outside);
            assert_eq!(ld_out, 0.0);
        }
    }

    #[test]
    fn spline_is_deterministic_and_finite() {
        // TEST 4: same parameters → identical outputs; outputs always finite.
        let bound = 4.0_f32;
        let a = make_spline(8, bound, 42, 1.5);
        let b = make_spline(8, bound, 42, 1.5);
        let mut rng = LcgRng::new(999);
        for _ in 0..500 {
            let x = (rng.next_f32() * 2.0 - 1.0) * 1.4 * bound;
            let (ya, lda) = a.forward(x);
            let (yb, ldb) = b.forward(x);
            assert_eq!(ya, yb);
            assert_eq!(lda, ldb);
            assert!(ya.is_finite() && lda.is_finite());
            assert!(a.inverse(ya).is_finite());
        }
    }

    #[test]
    fn spline_rejects_bad_parameter_shapes() {
        // K = 3 needs heights.len() == 3 and derivatives.len() == 2.
        assert!(RationalQuadraticSpline::new(&[0.0; 3], &[0.0; 2], &[0.0; 2], 4.0).is_err());
        assert!(RationalQuadraticSpline::new(&[0.0; 3], &[0.0; 3], &[0.0; 3], 4.0).is_err());
        assert!(RationalQuadraticSpline::new(&[], &[], &[], 4.0).is_err());
        assert!(RationalQuadraticSpline::new(&[0.0; 3], &[0.0; 3], &[0.0; 2], 0.0).is_err());
        assert!(RationalQuadraticSpline::new(&[0.0; 3], &[0.0; 3], &[0.0; 2], -1.0).is_err());
    }

    #[test]
    fn coupling_is_invertible() {
        // TEST 1 (coupling): inverse(forward(x)) ≈ x to <= 1e-4.
        let mut rng = LcgRng::new(3);
        let coupling = RqSplineCoupling::new(4, 5, 16, 8, 5.0, &mut rng).expect("coupling");
        let t = 6usize;
        let mut x = vec![0.0_f32; t * 4];
        let mut g = vec![0.0_f32; t * 5];
        let mut data = LcgRng::new(33);
        data.fill_normal(&mut x);
        data.fill_normal(&mut g);
        let (y, logdet) = coupling.forward(&x, &g, t).expect("forward");
        assert!(logdet.is_finite());
        assert!(y.iter().all(|v| v.is_finite()));
        let back = coupling.inverse(&y, &g, t).expect("inverse");
        let err = x
            .iter()
            .zip(back.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(err < 1e-4, "coupling round-trip err {err}");
    }

    #[test]
    fn coupling_logdet_matches_numerical_jacobian() {
        // TEST 2 (multi-dim coupling): analytic logdet ≈ log|det J| from the
        // numerical Jacobian on a small dimension, to <= 1e-2.
        let mut rng = LcgRng::new(8);
        let dim = 2usize;
        let t = 2usize; // n = t * dim = 4 → 4×4 Jacobian.
        let coupling = RqSplineCoupling::new(dim, 4, 12, 8, 6.0, &mut rng).expect("coupling");
        let n = t * dim;
        let mut x = vec![0.0_f32; n];
        let mut g = vec![0.0_f32; t * 4];
        let mut data = LcgRng::new(80);
        data.fill_normal(&mut x);
        data.fill_normal(&mut g);

        let (_y, analytic) = coupling.forward(&x, &g, t).expect("forward");

        let h = 1e-3_f32;
        let mut jac = vec![0.0_f32; n * n];
        for j in 0..n {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[j] += h;
            xm[j] -= h;
            let (yp, _) = coupling.forward(&xp, &g, t).expect("fwd+");
            let (ym, _) = coupling.forward(&xm, &g, t).expect("fwd-");
            for i in 0..n {
                jac[i * n + j] = (yp[i] - ym[i]) / (2.0 * h);
            }
        }
        let numeric = log_abs_det(jac, n);
        let err = (analytic - numeric).abs();
        assert!(
            err < 1e-2,
            "coupling logdet analytic={analytic} numeric={numeric}"
        );
    }

    #[test]
    fn coupling_is_deterministic_under_seed() {
        // TEST 4 (coupling): identical seeds → identical transform.
        let coupling_a = RqSplineCoupling::new(4, 5, 16, 8, 5.0, &mut LcgRng::new(7)).expect("a");
        let coupling_b = RqSplineCoupling::new(4, 5, 16, 8, 5.0, &mut LcgRng::new(7)).expect("b");
        let t = 5usize;
        let mut x = vec![0.0_f32; t * 4];
        let mut g = vec![0.0_f32; t * 5];
        LcgRng::new(70).fill_normal(&mut x);
        LcgRng::new(71).fill_normal(&mut g);
        let (ya, lda) = coupling_a.forward(&x, &g, t).expect("a");
        let (yb, ldb) = coupling_b.forward(&x, &g, t).expect("b");
        assert_eq!(ya, yb);
        assert_eq!(lda, ldb);
    }

    #[test]
    fn coupling_rejects_bad_config_and_shapes() {
        let mut rng = LcgRng::new(1);
        assert!(RqSplineCoupling::new(1, 4, 8, 8, 5.0, &mut rng).is_err()); // dim < 2
        assert!(RqSplineCoupling::new(4, 0, 8, 8, 5.0, &mut rng).is_err()); // cond_dim 0
        assert!(RqSplineCoupling::new(4, 4, 0, 8, 5.0, &mut rng).is_err()); // hidden 0
        assert!(RqSplineCoupling::new(4, 4, 8, 0, 5.0, &mut rng).is_err()); // num_bins 0
        assert!(RqSplineCoupling::new(4, 4, 8, 8, 0.0, &mut rng).is_err()); // bound 0

        let coupling = RqSplineCoupling::new(4, 5, 8, 8, 5.0, &mut rng).expect("coupling");
        assert!(coupling.forward(&[0.0; 6], &[0.0; 10], 2).is_err()); // x len bad
        assert!(coupling.forward(&[0.0; 8], &[0.0; 7], 2).is_err()); // g len bad
        assert!(coupling.forward(&[], &[], 0).is_err()); // t == 0
    }
}
