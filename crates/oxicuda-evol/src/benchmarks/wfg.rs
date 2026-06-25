//! The WFG (Walking Fish Group) scalable multi-objective test problem toolkit.
//!
//! Faithful pure-Rust implementation of the nine WFG problems and the underlying transformation
//! toolkit from:
//!
//! - S. Huband, P. Hingston, L. Barone & L. While, "A Review of Multiobjective Test Problems and a
//!   Scalable Test Problem Toolkit", *IEEE Transactions on Evolutionary Computation*, 10(5),
//!   477–506, 2006.
//!
//! # Problem construction
//!
//! Every WFG problem is built from the same recipe. A decision vector `z` of `n = k + l` variables
//! (with `zᵢ ∈ [0, 2i]`, `k` position-related and `l` distance-related) is normalised to
//! `yᵢ = zᵢ/(2i) ∈ [0, 1]`, run through a problem-specific chain of *transition* functions that
//! collapse `y` into an `M`-vector `t`, mapped through the degeneracy constants `Aᵢ` into the
//! shape arguments `x`, and finally combined with a *shape* function `h` and scaling constants
//! `Sₘ = 2m`, `D = 1`:
//!
//! ```text
//! xᵢ = max(t_M, Aᵢ)·(tᵢ − 0.5) + 0.5     (i = 1 … M−1)
//! x_M = t_M
//! fₘ  = D·x_M + Sₘ·hₘ(x₁ … x_{M−1})
//! ```
//!
//! The transition toolkit comprises **bias** (`b_poly`, `b_flat`, `b_param`), **shift**
//! (`s_linear`, `s_decept`, `s_multi`) and **reduction** (`r_sum`, `r_nonsep`) functions; the
//! shapes are `linear`, `convex`, `concave`, `mixed` and `disconnected`. These are composed
//! exactly as specified by Huband et al. (Table II / Section V) — no transformation is simplified.
//!
//! For the concave problems (WFG4–WFG9) the Pareto front is a scaled hypersphere: at the distance
//! optimum (`x_M = 0`) the objectives satisfy `Σ (fₘ/Sₘ)² = 1`. WFG1 has a convex+mixed front,
//! WFG2 a convex+disconnected front, and WFG3 a degenerate (linear) front.

use std::f64::consts::{FRAC_PI_2, PI};

// ─────────────────────────────────────────────────────────────────────────────
// Parameters
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration of a WFG problem instance.
///
/// - `k` — number of position-related parameters (must be divisible by `M − 1`)
/// - `l` — number of distance-related parameters (must be even for WFG2 / WFG3, which apply a
///   pairwise non-separable reduction; `≥ 1` otherwise)
/// - `m` — number of objectives (`≥ 2`)
///
/// The decision vector length is `n = k + l`, with `zᵢ ∈ [0, 2·i]` for the `i`-th (1-based)
/// variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WfgParams {
    /// Position-related parameter count (divisible by `M − 1`).
    pub k: usize,
    /// Distance-related parameter count.
    pub l: usize,
    /// Objective count (`≥ 2`).
    pub m: usize,
}

impl WfgParams {
    /// Construct a parameter set with `k` position, `l` distance variables and `m` objectives.
    pub fn new(k: usize, l: usize, m: usize) -> Self {
        Self { k, l, m }
    }

    /// Total number of decision variables `n = k + l`.
    #[inline]
    pub fn n(&self) -> usize {
        self.k + self.l
    }

    /// Structural validity check for a candidate decision-vector length.
    fn valid(&self, z_len: usize) -> bool {
        self.m >= 2
            && self.k >= 1
            && self.l >= 1
            && self.k.is_multiple_of(self.m - 1)
            && z_len == self.n()
    }
}

#[inline]
fn clamp01(v: f64) -> f64 {
    v.clamp(0.0, 1.0)
}

/// Normalise decision variables: `yᵢ = zᵢ / (2·i)` (1-based `i`), clamped to `[0, 1]`.
fn normalize(z: &[f64]) -> Vec<f64> {
    z.iter()
        .enumerate()
        .map(|(i, &zi)| clamp01(zi / (2.0 * (i + 1) as f64)))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Transition toolkit (Huband et al. 2006, Section V-A)
// ─────────────────────────────────────────────────────────────────────────────

/// Polynomial bias `b_poly(y, α) = y^α` (`α > 0`, `α ≠ 1`).
#[inline]
fn b_poly(y: f64, alpha: f64) -> f64 {
    y.max(0.0).powf(alpha)
}

/// Flat-region bias `b_flat`:
/// `A + min(0, ⌊y−B⌋)·A(B−y)/B − min(0, ⌊C−y⌋)·(1−A)(y−C)/(1−C)`.
fn b_flat(y: f64, a: f64, b: f64, c: f64) -> f64 {
    let t1 = 0.0_f64.min((y - b).floor()) * (a * (b - y) / b);
    let t2 = 0.0_f64.min((c - y).floor()) * ((1.0 - a) * (y - c) / (1.0 - c));
    clamp01(a + t1 - t2)
}

/// Parameter-dependent bias exponent `B + (C−B)·(A − (1−2u)·|⌊0.5−u⌋ + A|)` with the canonical
/// WFG7/8/9 constants `A = 0.98/49.98`, `B = 0.02`, `C = 50`.
fn b_param_exponent(u: f64) -> f64 {
    const A: f64 = 0.98 / 49.98;
    const B: f64 = 0.02;
    const C: f64 = 50.0;
    let v = A - (1.0 - 2.0 * u) * ((0.5 - u).floor() + A).abs();
    B + (C - B) * v
}

/// Parameter-dependent bias `b_param(y, u) = y^{exponent(u)}` (constants fixed per WFG7/8/9).
#[inline]
fn b_param(y: f64, u: f64) -> f64 {
    y.max(0.0).powf(b_param_exponent(u))
}

/// Linear shift `s_linear(y, A) = |y − A| / |⌊A − y⌋ + A|`.
fn s_linear(y: f64, a: f64) -> f64 {
    ((y - a).abs() / ((a - y).floor() + a).abs()).clamp(0.0, 1.0)
}

/// Deceptive shift `s_decept(y; A, B, C)` (single global optimum at `y = A`, with deceptive basins).
fn s_decept(y: f64, a: f64, b: f64, c: f64) -> f64 {
    let lead = (y - a).abs() - b;
    let left = (y - a + b).floor() * (1.0 - c + (a - b) / b) / (a - b);
    let right = (a + b - y).floor() * (1.0 - c + (1.0 - a - b) / b) / (1.0 - a - b);
    clamp01(1.0 + lead * (left + right + 1.0 / b))
}

/// Multimodal shift `s_multi(y; A, B, C)` with `A` minima; global optimum at `y = C`.
fn s_multi(y: f64, a: f64, b: f64, c: f64) -> f64 {
    let denom = 2.0 * ((c - y).floor() + c);
    let ratio = (y - c).abs() / denom;
    let cos_arg = (4.0 * a + 2.0) * PI * (0.5 - ratio);
    clamp01((1.0 + cos_arg.cos() + 4.0 * b * ratio * ratio) / (b + 2.0))
}

/// Weighted-sum reduction `r_sum(y, w) = Σ wᵢyᵢ / Σ wᵢ`.
fn r_sum(y: &[f64], w: &[f64]) -> f64 {
    let num: f64 = y.iter().zip(w).map(|(yi, wi)| wi * yi).sum();
    let den: f64 = w.iter().sum();
    num / den
}

/// Non-separable reduction `r_nonsep(y, A)` (Huband et al. 2006, eq. for `r_nonsep`).
fn r_nonsep(y: &[f64], a: usize) -> f64 {
    let n = y.len();
    let mut num = 0.0;
    for (j, &yj) in y.iter().enumerate() {
        num += yj;
        for k in 0..a.saturating_sub(1) {
            num += (yj - y[(j + k + 1) % n]).abs();
        }
    }
    let a_f = a as f64;
    let ceil_a2 = (a_f / 2.0).ceil();
    let den = (n as f64 / a_f) * ceil_a2 * (1.0 + 2.0 * a_f - 2.0 * ceil_a2);
    num / den
}

// ─────────────────────────────────────────────────────────────────────────────
// Final reductions producing the M-vector `t`
// ─────────────────────────────────────────────────────────────────────────────

/// Reduce `y` to `M` values by weighted sum: the first `k` entries split into `M−1` contiguous
/// position groups, the trailing entries form the single distance group.
fn reduce_sum(y: &[f64], k: usize, m: usize, w: &[f64]) -> Vec<f64> {
    let groups = m - 1;
    let gsize = k / groups;
    let mut t = Vec::with_capacity(m);
    for g in 0..groups {
        let lo = g * gsize;
        let hi = lo + gsize;
        t.push(r_sum(&y[lo..hi], &w[lo..hi]));
    }
    t.push(r_sum(&y[k..], &w[k..]));
    t
}

/// Reduce `y` to `M` values by weighted sum with uniform weights.
fn reduce_sum_uniform(y: &[f64], k: usize, m: usize) -> Vec<f64> {
    let w = vec![1.0_f64; y.len()];
    reduce_sum(y, k, m, &w)
}

/// Reduce `y` to `M` values by non-separable reduction: each position group uses `A = group size`,
/// the distance group uses `A = (number of distance entries)`.
fn reduce_nonsep(y: &[f64], k: usize, m: usize) -> Vec<f64> {
    let groups = m - 1;
    let gsize = k / groups;
    let mut t = Vec::with_capacity(m);
    for g in 0..groups {
        let lo = g * gsize;
        let hi = lo + gsize;
        t.push(r_nonsep(&y[lo..hi], gsize));
    }
    let dist = &y[k..];
    t.push(r_nonsep(dist, dist.len()));
    t
}

// ─────────────────────────────────────────────────────────────────────────────
// Shape functions (Huband et al. 2006, Section V-B)
// ─────────────────────────────────────────────────────────────────────────────

/// Linear shape (hyperplane front). `x` has length `M − 1`; returns `M` values.
fn shape_linear(x: &[f64]) -> Vec<f64> {
    let m = x.len() + 1;
    let mut h = Vec::with_capacity(m);
    for j in 1..=m {
        let mut v = 1.0;
        for &xi in &x[..m - j] {
            v *= xi;
        }
        if j > 1 {
            v *= 1.0 - x[m - j];
        }
        h.push(v);
    }
    h
}

/// Convex shape. `x` has length `M − 1`; returns `M` values.
fn shape_convex(x: &[f64]) -> Vec<f64> {
    let m = x.len() + 1;
    let mut h = Vec::with_capacity(m);
    for j in 1..=m {
        let mut v = 1.0;
        for &xi in &x[..m - j] {
            v *= 1.0 - (xi * FRAC_PI_2).cos();
        }
        if j > 1 {
            v *= 1.0 - (x[m - j] * FRAC_PI_2).sin();
        }
        h.push(v);
    }
    h
}

/// Concave (spherical) shape. `x` has length `M − 1`; returns `M` values satisfying `Σ hₘ² = 1`.
fn shape_concave(x: &[f64]) -> Vec<f64> {
    let m = x.len() + 1;
    let mut h = Vec::with_capacity(m);
    for j in 1..=m {
        let mut v = 1.0;
        for &xi in &x[..m - j] {
            v *= (xi * FRAC_PI_2).sin();
        }
        if j > 1 {
            v *= (x[m - j] * FRAC_PI_2).cos();
        }
        h.push(v);
    }
    h
}

/// Mixed convex/concave shape for the final objective: `(1 − x₁ − cos(2Aπx₁ + π/2)/(2Aπ))^α`.
fn shape_mixed(x1: f64, alpha: f64, a: f64) -> f64 {
    let two_a_pi = 2.0 * a * PI;
    let base = 1.0 - x1 - (two_a_pi * x1 + FRAC_PI_2).cos() / two_a_pi;
    clamp01(base.max(0.0).powf(alpha))
}

/// Disconnected shape for the final objective: `1 − x₁^α·cos²(A·x₁^β·π)`.
fn shape_disconnected(x1: f64, alpha: f64, beta: f64, a: f64) -> f64 {
    let c = (a * x1.max(0.0).powf(beta) * PI).cos();
    clamp01(1.0 - x1.max(0.0).powf(alpha) * c * c)
}

// ─────────────────────────────────────────────────────────────────────────────
// Degeneracy mapping and final objective assembly
// ─────────────────────────────────────────────────────────────────────────────

/// Degeneracy constants `Aᵢ`: all `1` except the degenerate WFG3, where `A₁ = 1`, `Aᵢ = 0` (i ≥ 2).
fn a_constants(m: usize, degenerate: bool) -> Vec<f64> {
    let mut a = vec![1.0_f64; m - 1];
    if degenerate {
        for v in a.iter_mut().skip(1) {
            *v = 0.0;
        }
    }
    a
}

/// Map the transition vector `t` (length `M`) into the shape arguments `x` (length `M`):
/// `xᵢ = max(t_M, Aᵢ)·(tᵢ − 0.5) + 0.5` for `i < M`, and `x_M = t_M`.
fn degeneracy(t: &[f64], a: &[f64]) -> Vec<f64> {
    let m = t.len();
    let t_m = t[m - 1];
    let mut x = Vec::with_capacity(m);
    for (ai, ti) in a.iter().zip(t.iter().take(m - 1)) {
        x.push(t_m.max(*ai) * (*ti - 0.5) + 0.5);
    }
    x.push(t_m);
    x
}

/// Assemble objectives `fₘ = D·x_M + Sₘ·hₘ` with `D = 1`, `Sₘ = 2m`.
fn objectives(x: &[f64], h: &[f64]) -> Vec<f64> {
    let m = x.len();
    let x_m = x[m - 1];
    h.iter()
        .enumerate()
        .map(|(i, &hi)| x_m + 2.0 * (i + 1) as f64 * hi)
        .collect()
}

/// `2i` weight vector used by WFG1's final weighted-sum reduction.
fn weights_2i(n: usize) -> Vec<f64> {
    (0..n).map(|i| 2.0 * (i + 1) as f64).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-problem transition pipelines (produce the M-vector `t`)
// ─────────────────────────────────────────────────────────────────────────────

fn wfg1_transitions(y: &[f64], p: &WfgParams) -> Vec<f64> {
    let (k, n) = (p.k, p.n());
    let mut yy = y.to_vec();
    // t1: linear shift on distance parameters.
    for v in yy[k..n].iter_mut() {
        *v = s_linear(*v, 0.35);
    }
    // t2: flat-region bias on distance parameters.
    for v in yy[k..n].iter_mut() {
        *v = b_flat(*v, 0.8, 0.75, 0.85);
    }
    // t3: polynomial bias on every parameter.
    for v in yy.iter_mut() {
        *v = b_poly(*v, 0.02);
    }
    // t4: weighted-sum reduction with weights 2i.
    let w = weights_2i(n);
    reduce_sum(&yy, k, p.m, &w)
}

fn wfg2_transitions(y: &[f64], p: &WfgParams) -> Vec<f64> {
    let (k, l, n) = (p.k, p.l, p.n());
    let mut yy = y.to_vec();
    // t1: linear shift on distance parameters.
    for v in yy[k..n].iter_mut() {
        *v = s_linear(*v, 0.35);
    }
    // t2: pairwise non-separable reduction of the l distance parameters into l/2.
    let mut reduced = yy[..k].to_vec();
    for pair in 0..l / 2 {
        let a = yy[k + 2 * pair];
        let b = yy[k + 2 * pair + 1];
        reduced.push(r_nonsep(&[a, b], 2));
    }
    // t3: uniform weighted-sum reduction.
    reduce_sum_uniform(&reduced, k, p.m)
}

fn wfg4_transitions(y: &[f64], p: &WfgParams) -> Vec<f64> {
    let mut yy = y.to_vec();
    // t1: multimodal shift on every parameter.
    for v in yy.iter_mut() {
        *v = s_multi(*v, 30.0, 10.0, 0.35);
    }
    reduce_sum_uniform(&yy, p.k, p.m)
}

fn wfg5_transitions(y: &[f64], p: &WfgParams) -> Vec<f64> {
    let mut yy = y.to_vec();
    // t1: deceptive shift on every parameter.
    for v in yy.iter_mut() {
        *v = s_decept(*v, 0.35, 0.001, 0.05);
    }
    reduce_sum_uniform(&yy, p.k, p.m)
}

fn wfg6_transitions(y: &[f64], p: &WfgParams) -> Vec<f64> {
    let (k, n) = (p.k, p.n());
    let mut yy = y.to_vec();
    // t1: linear shift on distance parameters.
    for v in yy[k..n].iter_mut() {
        *v = s_linear(*v, 0.35);
    }
    // t2: non-separable reduction.
    reduce_nonsep(&yy, k, p.m)
}

fn wfg7_transitions(y: &[f64], p: &WfgParams) -> Vec<f64> {
    let (k, n) = (p.k, p.n());
    let orig = y.to_vec();
    let mut yy = y.to_vec();
    // t1: parameter-dependent bias on position parameters, dependent on the sum of *following*
    // (original) parameters.
    let ones = vec![1.0_f64; n];
    for (i, slot) in yy.iter_mut().enumerate().take(k) {
        let u = r_sum(&orig[i + 1..n], &ones[i + 1..n]);
        *slot = b_param(orig[i], u);
    }
    // t2: linear shift on distance parameters.
    for v in yy[k..n].iter_mut() {
        *v = s_linear(*v, 0.35);
    }
    reduce_sum_uniform(&yy, k, p.m)
}

fn wfg8_transitions(y: &[f64], p: &WfgParams) -> Vec<f64> {
    let (k, n) = (p.k, p.n());
    let orig = y.to_vec();
    let mut yy = y.to_vec();
    // t1: parameter-dependent bias on distance parameters, dependent on the sum of *preceding*
    // (original) parameters.
    let ones = vec![1.0_f64; n];
    for (i, slot) in yy.iter_mut().enumerate().skip(k) {
        let u = r_sum(&orig[..i], &ones[..i]);
        *slot = b_param(orig[i], u);
    }
    // t2: linear shift on the (already biased) distance parameters.
    for v in yy[k..n].iter_mut() {
        *v = s_linear(*v, 0.35);
    }
    reduce_sum_uniform(&yy, k, p.m)
}

fn wfg9_transitions(y: &[f64], p: &WfgParams) -> Vec<f64> {
    let (k, n) = (p.k, p.n());
    let orig = y.to_vec();
    let mut yy = y.to_vec();
    // t1: parameter-dependent bias on parameters 1..n−1, dependent on the sum of *following*
    // (original) parameters; the last parameter is unchanged.
    let ones = vec![1.0_f64; n];
    for (i, slot) in yy.iter_mut().enumerate().take(n - 1) {
        let u = r_sum(&orig[i + 1..n], &ones[i + 1..n]);
        *slot = b_param(orig[i], u);
    }
    // t2: deceptive shift on positions, multimodal shift on distances.
    for v in yy[..k].iter_mut() {
        *v = s_decept(*v, 0.35, 0.001, 0.05);
    }
    for v in yy[k..n].iter_mut() {
        *v = s_multi(*v, 30.0, 95.0, 0.35);
    }
    // t3: non-separable reduction.
    reduce_nonsep(&yy, k, p.m)
}

// ─────────────────────────────────────────────────────────────────────────────
// The nine WFG objective functions
// ─────────────────────────────────────────────────────────────────────────────

/// WFG1: convex front with a flat-bias / polynomial-bias distance landscape and a *mixed*
/// (partly convex, partly concave) final objective. Decision vector `z` of length `k + l`.
pub fn wfg1(z: &[f64], p: &WfgParams) -> Vec<f64> {
    if !p.valid(z.len()) {
        return vec![0.0; p.m.max(1)];
    }
    let t = wfg1_transitions(&normalize(z), p);
    let x = degeneracy(&t, &a_constants(p.m, false));
    let mut h = shape_convex(&x[..p.m - 1]);
    h[p.m - 1] = shape_mixed(x[0], 1.0, 5.0);
    objectives(&x, &h)
}

/// WFG2: convex shape on the leading objectives and a *disconnected* final objective, with a
/// non-separable pairwise distance reduction. Requires `l` even.
pub fn wfg2(z: &[f64], p: &WfgParams) -> Vec<f64> {
    if !p.valid(z.len()) || !p.l.is_multiple_of(2) {
        return vec![0.0; p.m.max(1)];
    }
    let t = wfg2_transitions(&normalize(z), p);
    let x = degeneracy(&t, &a_constants(p.m, false));
    let mut h = shape_convex(&x[..p.m - 1]);
    h[p.m - 1] = shape_disconnected(x[0], 1.0, 1.0, 5.0);
    objectives(&x, &h)
}

/// WFG3: *degenerate* linear front (a connected line in objective space), sharing WFG2's
/// non-separable distance reduction. Requires `l` even.
pub fn wfg3(z: &[f64], p: &WfgParams) -> Vec<f64> {
    if !p.valid(z.len()) || !p.l.is_multiple_of(2) {
        return vec![0.0; p.m.max(1)];
    }
    let t = wfg2_transitions(&normalize(z), p);
    let x = degeneracy(&t, &a_constants(p.m, true));
    let h = shape_linear(&x[..p.m - 1]);
    objectives(&x, &h)
}

/// WFG4: concave (spherical) front with a strongly *multimodal* distance landscape (`s_multi`).
pub fn wfg4(z: &[f64], p: &WfgParams) -> Vec<f64> {
    if !p.valid(z.len()) {
        return vec![0.0; p.m.max(1)];
    }
    let t = wfg4_transitions(&normalize(z), p);
    let x = degeneracy(&t, &a_constants(p.m, false));
    let h = shape_concave(&x[..p.m - 1]);
    objectives(&x, &h)
}

/// WFG5: concave front with a *deceptive* distance landscape (`s_decept`).
pub fn wfg5(z: &[f64], p: &WfgParams) -> Vec<f64> {
    if !p.valid(z.len()) {
        return vec![0.0; p.m.max(1)];
    }
    let t = wfg5_transitions(&normalize(z), p);
    let x = degeneracy(&t, &a_constants(p.m, false));
    let h = shape_concave(&x[..p.m - 1]);
    objectives(&x, &h)
}

/// WFG6: concave front with a *non-separable* distance reduction (`r_nonsep`).
pub fn wfg6(z: &[f64], p: &WfgParams) -> Vec<f64> {
    if !p.valid(z.len()) {
        return vec![0.0; p.m.max(1)];
    }
    let t = wfg6_transitions(&normalize(z), p);
    let x = degeneracy(&t, &a_constants(p.m, false));
    let h = shape_concave(&x[..p.m - 1]);
    objectives(&x, &h)
}

/// WFG7: concave front whose *position* parameters carry a parameter-dependent bias driven by the
/// distance parameters (`b_param` on the sum of following variables).
pub fn wfg7(z: &[f64], p: &WfgParams) -> Vec<f64> {
    if !p.valid(z.len()) {
        return vec![0.0; p.m.max(1)];
    }
    let t = wfg7_transitions(&normalize(z), p);
    let x = degeneracy(&t, &a_constants(p.m, false));
    let h = shape_concave(&x[..p.m - 1]);
    objectives(&x, &h)
}

/// WFG8: concave front whose *distance* parameters carry a parameter-dependent bias driven by the
/// preceding (position) variables (`b_param` on the sum of preceding variables).
pub fn wfg8(z: &[f64], p: &WfgParams) -> Vec<f64> {
    if !p.valid(z.len()) {
        return vec![0.0; p.m.max(1)];
    }
    let t = wfg8_transitions(&normalize(z), p);
    let x = degeneracy(&t, &a_constants(p.m, false));
    let h = shape_concave(&x[..p.m - 1]);
    objectives(&x, &h)
}

/// WFG9: concave front combining a parameter-dependent bias with both deceptive (position) and
/// multimodal (distance) shifts and a non-separable reduction — the hardest of the family.
pub fn wfg9(z: &[f64], p: &WfgParams) -> Vec<f64> {
    if !p.valid(z.len()) {
        return vec![0.0; p.m.max(1)];
    }
    let t = wfg9_transitions(&normalize(z), p);
    let x = degeneracy(&t, &a_constants(p.m, false));
    let h = shape_concave(&x[..p.m - 1]);
    objectives(&x, &h)
}

// ─────────────────────────────────────────────────────────────────────────────
// Analytic Pareto-optimal objective vectors
// ─────────────────────────────────────────────────────────────────────────────

/// Distance-parameter values (normalised `y ∈ [0, 1]`) that place a solution on the WFG Pareto
/// front for `problem ∈ 1..=9`, given the normalised position parameters.
///
/// For WFG1–WFG7 the per-problem distance shift is minimised at `y = 0.35`. WFG8 (bias driven by
/// preceding variables) and WFG9 (bias driven by following variables) require inverting the
/// `b_param` exponent so that the post-bias value reaches the shift optimum; this is solved
/// sequentially (forward for WFG8, backward for WFG9).
fn optimum_distance_y(problem: usize, p: &WfgParams, position_y: &[f64]) -> Vec<f64> {
    let (k, l, n) = (p.k, p.l, p.n());
    match problem {
        8 => {
            // Forward: distance i needs b_param(y_i, mean(preceding)) = 0.35 ⇒ y_i = 0.35^{1/exp}.
            let mut all = position_y.to_vec();
            let mut dist = Vec::with_capacity(l);
            for _ in k..n {
                let ones = vec![1.0_f64; all.len()];
                let u = r_sum(&all, &ones);
                let yi = 0.35_f64.powf(1.0 / b_param_exponent(u));
                dist.push(yi);
                all.push(yi);
            }
            dist
        }
        9 => {
            // Backward: the last distance variable is unbiased (optimum 0.35); earlier ones need
            // b_param(y_i, mean(following)) = 0.35.
            let mut dist = vec![0.35_f64; l];
            for idx in (0..l.saturating_sub(1)).rev() {
                let following = &dist[idx + 1..];
                let ones = vec![1.0_f64; following.len()];
                let u = r_sum(following, &ones);
                dist[idx] = 0.35_f64.powf(1.0 / b_param_exponent(u));
            }
            dist
        }
        _ => vec![0.35_f64; l],
    }
}

/// Objective vector at a WFG Pareto-optimal decision vector whose `k` position parameters are
/// given (normalised, in `[0, 1]`) and whose distance parameters sit at the problem's front
/// optimum. `problem ∈ 1..=9`.
///
/// For the concave problems (WFG4–WFG9) the returned vector satisfies `Σ (fₘ/2m)² = 1` (it lies on
/// the scaled unit hypersphere). For WFG3 it lies on the degenerate line. This is the analytic
/// counterpart of the `zdt*_pareto_front_f2` helpers.
pub fn wfg_optimum_objectives(problem: usize, p: &WfgParams, position: &[f64]) -> Vec<f64> {
    let (k, n) = (p.k, p.n());
    if p.m < 2 || !k.is_multiple_of(p.m - 1) {
        return vec![0.0; p.m.max(1)];
    }
    // Normalised position parameters (default 0.5 when not supplied).
    let position_y: Vec<f64> = (0..k)
        .map(|i| clamp01(position.get(i).copied().unwrap_or(0.5)))
        .collect();
    let dist_y = optimum_distance_y(problem, p, &position_y);

    // Reconstruct the raw decision vector `z` (zᵢ = 2·i·yᵢ, 1-based i).
    let mut z = vec![0.0_f64; n];
    for (i, zi) in z.iter_mut().enumerate().take(k) {
        *zi = 2.0 * (i + 1) as f64 * position_y[i];
    }
    for (j, &yj) in dist_y.iter().enumerate() {
        let i = k + j;
        z[i] = 2.0 * (i + 1) as f64 * yj;
    }

    match problem {
        1 => wfg1(&z, p),
        2 => wfg2(&z, p),
        3 => wfg3(&z, p),
        4 => wfg4(&z, p),
        5 => wfg5(&z, p),
        6 => wfg6(&z, p),
        7 => wfg7(&z, p),
        8 => wfg8(&z, p),
        _ => wfg9(&z, p),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests — mathematically-provable properties only (no optimiser-convergence claims)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// All nine problem objective functions, dispatched by index for table-driven tests.
    fn eval(problem: usize, z: &[f64], p: &WfgParams) -> Vec<f64> {
        match problem {
            1 => wfg1(z, p),
            2 => wfg2(z, p),
            3 => wfg3(z, p),
            4 => wfg4(z, p),
            5 => wfg5(z, p),
            6 => wfg6(z, p),
            7 => wfg7(z, p),
            8 => wfg8(z, p),
            _ => wfg9(z, p),
        }
    }

    /// Build a decision vector from normalised values: `zᵢ = 2·i·valᵢ`.
    fn z_from_y(yvals: &[f64]) -> Vec<f64> {
        yvals
            .iter()
            .enumerate()
            .map(|(i, &v)| 2.0 * (i + 1) as f64 * v)
            .collect()
    }

    /// Scaled-sphere radius `√Σ(fₘ/2m)²` — equals 1 on the concave (WFG4–9) Pareto front.
    fn scaled_radius(f: &[f64]) -> f64 {
        f.iter()
            .enumerate()
            .map(|(i, &fm)| {
                let v = fm / (2.0 * (i + 1) as f64);
                v * v
            })
            .sum::<f64>()
            .sqrt()
    }

    // ── Universal structural properties ───────────────────────────────────────

    #[test]
    fn wfg_all_return_m_objectives() {
        for &m in &[2usize, 3] {
            let k = 4; // divisible by m-1 ∈ {1,2}
            let l = 4; // even (WFG2/3) and ≥ 1
            let p = WfgParams::new(k, l, m);
            let y = vec![0.5_f64; k + l];
            let z = z_from_y(&y);
            for problem in 1..=9 {
                let f = eval(problem, &z, &p);
                assert_eq!(
                    f.len(),
                    m,
                    "WFG{problem} (M={m}) returned {} objectives",
                    f.len()
                );
            }
        }
    }

    #[test]
    fn wfg_all_deterministic() {
        let p = WfgParams::new(4, 4, 3);
        let z = z_from_y(&[0.1, 0.7, 0.3, 0.9, 0.2, 0.6, 0.4, 0.8]);
        for problem in 1..=9 {
            assert_eq!(
                eval(problem, &z, &p),
                eval(problem, &z, &p),
                "WFG{problem} nondeterministic"
            );
        }
    }

    #[test]
    fn wfg_objectives_nonnegative() {
        // fₘ = x_M + 2m·hₘ with x_M ≥ 0 and hₘ ∈ [0, 1] ⇒ fₘ ≥ 0 for every decision vector.
        let p = WfgParams::new(4, 4, 3);
        for &probe in &[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.2, 0.8, 0.5, 0.1, 0.9, 0.35, 0.6, 0.25],
        ] {
            let z = z_from_y(&probe);
            for problem in 1..=9 {
                for (m, &fm) in eval(problem, &z, &p).iter().enumerate() {
                    assert!(fm >= -1e-12, "WFG{problem} f{} = {fm} negative", m + 1);
                }
            }
        }
    }

    // ── WFG4–9 concave front: scaled unit sphere ──────────────────────────────

    #[test]
    fn wfg_concave_optimum_on_scaled_unit_sphere() {
        // At the constructed distance optimum (x_M = 0) the concave shape gives Σ(fₘ/2m)² = 1
        // for ANY position parameters (spherical identity). This holds exactly for WFG4–WFG9,
        // including the bias-coupled WFG7/8/9 whose optima we obtain by inverting b_param.
        let positions: [&[f64]; 4] = [
            &[0.5, 0.5, 0.5, 0.5],
            &[0.1, 0.9, 0.3, 0.7],
            &[0.0, 0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0, 1.0],
        ];
        for &m in &[2usize, 3] {
            let p = WfgParams::new(4, 4, m);
            for problem in 4..=9 {
                for pos in &positions {
                    let f = wfg_optimum_objectives(problem, &p, pos);
                    let r = scaled_radius(&f);
                    assert!(
                        (r - 1.0).abs() < 1e-9,
                        "WFG{problem} M={m} pos={pos:?}: scaled radius {r} (want 1)"
                    );
                }
            }
        }
    }

    #[test]
    fn wfg_concave_distance_perturbation_leaves_sphere_outward() {
        // Σ(fₘ/2m)² = Σ(hₘ + x_M/2m)² = 1 + (positive in x_M) ≥ 1, with equality only at x_M = 0.
        // Any distance perturbation lifts the radius strictly above 1 (away from the front).
        let p = WfgParams::new(4, 4, 3);
        for problem in 4..=9 {
            let on = wfg_optimum_objectives(problem, &p, &[0.5, 0.5, 0.5, 0.5]);
            assert!(
                (scaled_radius(&on) - 1.0).abs() < 1e-9,
                "WFG{problem} optimum off sphere"
            );
            // Perturb distance variables well away from their optimum.
            let mut z = z_from_y(&[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
            for (i, zi) in z.iter_mut().enumerate().skip(p.k) {
                *zi = 2.0 * (i + 1) as f64 * 0.8;
            }
            let off = eval(problem, &z, &p);
            assert!(
                scaled_radius(&off) > 1.0 + 1e-6,
                "WFG{problem} perturbed radius {} not > 1",
                scaled_radius(&off)
            );
        }
    }

    // ── WFG1 / WFG2: shared additive distance penalty ─────────────────────────

    #[test]
    fn wfg1_wfg2_additive_distance_penalty() {
        // For WFG1/WFG2 the position-shape parameters are independent of the distance parameters,
        // so fₘ = x_M + 2m·hₘ(positions): changing the distance variables shifts every objective by
        // the SAME amount x_M. We verify the inter-objective differences are identical and that the
        // front optimum (x_M = 0) is the component-wise minimum.
        let p = WfgParams::new(4, 4, 3);
        for problem in [1usize, 2] {
            let pos = [0.3, 0.6, 0.4, 0.7];
            let on = wfg_optimum_objectives(problem, &p, &pos);
            let mut z = z_from_y(&[pos[0], pos[1], pos[2], pos[3], 0.6, 0.6, 0.6, 0.6]);
            for (i, zi) in z.iter_mut().enumerate().skip(p.k) {
                *zi = 2.0 * (i + 1) as f64 * 0.6;
            }
            let off = eval(problem, &z, &p);
            let diffs: Vec<f64> = off.iter().zip(&on).map(|(o, n)| o - n).collect();
            let dmax = diffs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let dmin = diffs.iter().cloned().fold(f64::INFINITY, f64::min);
            assert!(
                (dmax - dmin).abs() < 1e-9,
                "WFG{problem} distance shift not uniform across objectives: diffs={diffs:?}"
            );
            assert!(
                dmin > 0.0,
                "WFG{problem} off-front should exceed on-front, diffs={diffs:?}"
            );
        }
    }

    // ── WFG2: disconnected front ──────────────────────────────────────────────

    #[test]
    fn wfg2_front_is_disconnected() {
        // With M=2 and distance at optimum, sweeping the position parameter traces a curve whose
        // non-dominated (Pareto) portion splits into several disconnected pieces (the `disconnected`
        // shape with A=5). We count maximal runs of running-minimum points along f1.
        let p = WfgParams::new(2, 4, 2);
        let n_samples = 6000usize;
        let mut curve: Vec<(f64, f64)> = Vec::with_capacity(n_samples + 1);
        for s in 0..=n_samples {
            let y1 = s as f64 / n_samples as f64;
            // Position parameter sweeps; distances fixed at their optimum 0.35.
            let mut yv = vec![0.35_f64; p.n()];
            yv[0] = y1;
            yv[1] = y1; // both position vars (k=2) move together so x1 sweeps [0,1]
            let f = wfg2(&z_from_y(&yv), &p);
            curve.push((f[0], f[1]));
        }
        // Pareto front of a parametric curve with strictly increasing f1: a point is non-dominated
        // iff its f2 is a strict running minimum. Count contiguous on-front segments.
        let mut running_min = f64::INFINITY;
        let mut segments = 0usize;
        let mut prev_on = false;
        for &(_, f2) in &curve {
            let on = f2 < running_min - 1e-12;
            if on {
                running_min = f2;
                if !prev_on {
                    segments += 1;
                }
            }
            prev_on = on;
        }
        assert!(
            segments >= 2,
            "WFG2 front should be disconnected, found {segments} segment(s)"
        );
    }

    // ── WFG3: degenerate (linear) front ───────────────────────────────────────

    #[test]
    fn wfg3_front_is_degenerate_line() {
        // M=3 degenerate front: at the optimum A=[1,0] forces x₂ = 0.5, giving the linear relations
        // f₂ = 2·f₁ and f₃ = 6 − 6·f₁ (a one-dimensional line embedded in 3-objective space).
        let p = WfgParams::new(4, 4, 3);
        for &pos in &[
            [0.1, 0.4, 0.6, 0.9],
            [0.5, 0.2, 0.8, 0.3],
            [0.9, 0.5, 0.1, 0.7],
        ] {
            let f = wfg_optimum_objectives(3, &p, &pos);
            assert!(
                (f[1] - 2.0 * f[0]).abs() < 1e-9,
                "WFG3 f2={} not 2·f1={}",
                f[1],
                2.0 * f[0]
            );
            assert!(
                (f[2] - (6.0 - 6.0 * f[0])).abs() < 1e-9,
                "WFG3 f3={} not 6−6·f1={}",
                f[2],
                6.0 - 6.0 * f[0]
            );
        }
    }

    // ── Invalid configuration guards ──────────────────────────────────────────

    #[test]
    fn wfg_invalid_params_return_zero_vector() {
        let p = WfgParams::new(3, 4, 3); // k=3 not divisible by m-1=2
        let z = vec![0.5_f64; p.n()];
        assert_eq!(wfg4(&z, &p), vec![0.0, 0.0, 0.0]);
        // Wrong decision-vector length.
        let p2 = WfgParams::new(4, 4, 3);
        assert_eq!(wfg4(&[0.1, 0.2], &p2), vec![0.0, 0.0, 0.0]);
    }

    // ── Transition-toolkit unit checks (exact arithmetic) ─────────────────────

    #[test]
    fn shift_functions_zero_at_their_optima() {
        // s_linear, s_decept, s_multi all reach 0 at their documented optimum — the mechanism by
        // which the distance reduction collapses to x_M = 0 on the front.
        assert!(s_linear(0.35, 0.35).abs() < 1e-12);
        assert!(s_decept(0.35, 0.35, 0.001, 0.05).abs() < 1e-12);
        assert!(s_multi(0.35, 30.0, 10.0, 0.35).abs() < 1e-12);
        assert!(s_multi(0.35, 30.0, 95.0, 0.35).abs() < 1e-12);
    }

    #[test]
    fn concave_shape_is_unit_sphere_for_any_x() {
        // Σ hₘ² = 1 for arbitrary x (spherical identity) — the backbone of the WFG4–9 front test.
        for x in [
            vec![0.2, 0.7],
            vec![0.0, 1.0],
            vec![0.5, 0.5],
            vec![0.9, 0.1],
        ] {
            let h = shape_concave(&x);
            let s: f64 = h.iter().map(|v| v * v).sum();
            assert!((s - 1.0).abs() < 1e-12, "Σh² = {s} for x={x:?}");
        }
    }

    #[test]
    fn r_nonsep_pair_matches_closed_form() {
        // r_nonsep([a,b], 2) = (a + b + 2|a−b|)/3; zero at a=b=0 (the front-collapse condition).
        let v = r_nonsep(&[0.2, 0.5], 2);
        assert!(
            (v - (0.2 + 0.5 + 2.0 * 0.3) / 3.0).abs() < 1e-12,
            "r_nonsep = {v}"
        );
        assert!(r_nonsep(&[0.0, 0.0], 2).abs() < 1e-12);
    }
}
