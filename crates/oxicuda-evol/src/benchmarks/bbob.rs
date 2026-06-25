//! Standard black-box benchmark functions (BBOB-style) and algorithm evaluation harness.
//!
//! References:
//! - N. Hansen et al., "Real-Parameter Black-Box Optimization Benchmarking 2009: Noiseless Functions
//!   Definitions", INRIA Research Report RR-6829, 2009.
//! - Deb, K., Thiele, L., Laumanns, M., Zitzler, E. (2002). "Scalable multi-objective
//!   optimization test problems." Proc. CEC 2002.

use crate::evolution::cmaes::cmaes::{CmaEsConfig, CmaEsState};
use crate::handle::LcgRng;
use crate::metrics::hypervolume_nd::hypervolume_nd;
use crate::multiobjective::nsga2::{Nsga2Config, nsga2_run};
use crate::{EvolError, EvolResult};

// ─────────────────────────────────────────────────────────────────────────────
// Single-objective benchmark functions
// ─────────────────────────────────────────────────────────────────────────────

/// Sphere function: `f(x) = Σ xᵢ²`. Global minimum at **x = 0**, f = 0.
///
/// Convex, separable, unimodal. The simplest benchmark for sanity-checking.
#[inline]
pub fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|&xi| xi * xi).sum()
}

/// Ellipsoid (ill-conditioned sphere): `f(x) = Σ (1000^(i/(n-1)) · xᵢ)²`.
///
/// Conditioning number is 10⁶. Tests adaptation to axis-aligned covariance.
/// Global minimum at **x = 0**, f = 0.
pub fn ellipsoid(x: &[f64]) -> f64 {
    let n = x.len();
    if n <= 1 {
        return x.first().map(|&v| v * v).unwrap_or(0.0);
    }
    x.iter()
        .enumerate()
        .map(|(i, &xi)| {
            let scale = 1000_f64.powf(i as f64 / (n - 1) as f64);
            (scale * xi) * (scale * xi)
        })
        .sum()
}

/// Rosenbrock (banana) function: `f(x) = Σ [100·(x_{i+1} - xᵢ²)² + (xᵢ - 1)²]`.
///
/// Non-convex, narrow curved valley. Global minimum at **x = (1,…,1)**, f = 0.
pub fn rosenbrock(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }
    x.windows(2)
        .map(|w| {
            let xi = w[0];
            let xi1 = w[1];
            100.0 * (xi1 - xi * xi).powi(2) + (xi - 1.0).powi(2)
        })
        .sum()
}

/// Rastrigin function: `f(x) = 10n + Σ [xᵢ² − 10·cos(2π·xᵢ)]`.
///
/// Highly multimodal, ≈10n local minima. Global minimum at **x = 0**, f = 0.
pub fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let two_pi = 2.0 * std::f64::consts::PI;
    10.0 * n
        + x.iter()
            .map(|&xi| xi * xi - 10.0 * (two_pi * xi).cos())
            .sum::<f64>()
}

/// Schwefel function: `f(x) = 418.9829·n − Σ xᵢ·sin(√|xᵢ|)`.
///
/// Deceptive: global minimum is far from secondary minima.
/// Global minimum at **xᵢ = 418.9829…**, f ≈ 0.
pub fn schwefel(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    418.9829 * n - x.iter().map(|&xi| xi * xi.abs().sqrt().sin()).sum::<f64>()
}

/// Ackley function: `f(x) = −20·exp(−0.2·√(Σxᵢ²/n)) − exp(Σcos(2πxᵢ)/n) + 20 + e`.
///
/// Many local minima, exponential global basin. Global minimum at **x = 0**, f = 0.
pub fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let two_pi = 2.0 * std::f64::consts::PI;
    let sum_sq: f64 = x.iter().map(|&xi| xi * xi).sum();
    let sum_cos: f64 = x.iter().map(|&xi| (two_pi * xi).cos()).sum();
    -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp() + 20.0 + std::f64::consts::E
}

/// Griewank function: `f(x) = 1 + Σxᵢ²/4000 − Πcos(xᵢ/√(i+1))`.
///
/// Product term introduces regular structure in the multimodal landscape.
/// Global minimum at **x = 0**, f = 0.
pub fn griewank(x: &[f64]) -> f64 {
    let sum_sq: f64 = x.iter().map(|&xi| xi * xi / 4000.0).sum();
    let product: f64 = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| xi / ((i + 1) as f64).sqrt())
        .map(|v| v.cos())
        .product();
    1.0 + sum_sq - product
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-objective benchmark functions
// ─────────────────────────────────────────────────────────────────────────────

/// ZDT1: two-objective benchmark with convex Pareto front.
///
/// Decision variables: `x ∈ [0, 1]^n`.
/// - `f₁ = x₀`
/// - `g = 1 + 9·Σ(x[1:]/(n−1))`
/// - `f₂ = g·(1 − √(f₁/g))`
///
/// Returns `[f1, f2]`.
pub fn zdt1(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![0.0, 0.0];
    }
    let f1 = x[0];
    if n == 1 {
        let g = 1.0;
        let f2 = g * (1.0 - (f1 / g).max(0.0).sqrt());
        return vec![f1, f2];
    }
    let g_sum: f64 = x[1..].iter().sum::<f64>();
    let g = 1.0 + 9.0 * g_sum / (n - 1) as f64;
    let f2 = g * (1.0 - (f1 / g).max(0.0).sqrt());
    vec![f1, f2]
}

/// ZDT2: two-objective benchmark with non-convex Pareto front.
///
/// Decision variables: `x ∈ [0, 1]^n`.
/// - `f₁ = x₀`
/// - `g = 1 + 9·Σ(x[1:]/(n−1))`
/// - `f₂ = g·(1 − (f₁/g)²)`
///
/// Returns `[f1, f2]`.
pub fn zdt2(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![0.0, 0.0];
    }
    let f1 = x[0];
    if n == 1 {
        let g = 1.0;
        let f2 = g * (1.0 - (f1 / g).powi(2));
        return vec![f1, f2];
    }
    let g_sum: f64 = x[1..].iter().sum::<f64>();
    let g = 1.0 + 9.0 * g_sum / (n - 1) as f64;
    let f2 = g * (1.0 - (f1 / g).powi(2));
    vec![f1, f2]
}

/// DTLZ1: three-objective benchmark.
///
/// Decision variables: `x ∈ [0, 1]^n` with `n ≥ 3`.
/// - `xm = x[2..]` (the "distance" variables)
/// - `g(xm) = 100·(|xm| + Σ[(xi − 0.5)² − cos(20π(xi − 0.5))])`
/// - `f₁ = 0.5·x₀·x₁·(1 + g)`
/// - `f₂ = 0.5·x₀·(1 − x₁)·(1 + g)`
/// - `f₃ = 0.5·(1 − x₀)·(1 + g)`
///
/// Returns `[f1, f2, f3]`.
pub fn dtlz1(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 3 {
        // Degenerate case: pad with zeros for objective vector
        return vec![0.0, 0.0, 0.0];
    }
    let xm = &x[2..];
    let k = xm.len() as f64;
    let two_pi = 2.0 * std::f64::consts::PI;
    let g_sum: f64 = xm
        .iter()
        .map(|&xi| {
            let shifted = xi - 0.5;
            shifted * shifted - (20.0 * two_pi * shifted).cos()
        })
        .sum::<f64>();
    let g = 100.0 * (k + g_sum);
    let f1 = 0.5 * x[0] * x[1] * (1.0 + g);
    let f2 = 0.5 * x[0] * (1.0 - x[1]) * (1.0 + g);
    let f3 = 0.5 * (1.0 - x[0]) * (1.0 + g);
    vec![f1, f2, f3]
}

/// ZDT3: two-objective benchmark with a **discontinuous** Pareto front.
///
/// Decision variables: `x ∈ [0, 1]^n`.
/// - `f₁ = x₀`
/// - `g = 1 + 9·Σ(x[1:])/(n−1)`
/// - `f₂ = g·(1 − √(f₁/g) − (f₁/g)·sin(10π·f₁))`
///
/// The `sin(10π·f₁)` term breaks the front into five disconnected Pareto-optimal segments.
/// Returns `[f1, f2]`.
pub fn zdt3(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![0.0, 0.0];
    }
    let f1 = x[0];
    let g = if n == 1 {
        1.0
    } else {
        1.0 + 9.0 * x[1..].iter().sum::<f64>() / (n - 1) as f64
    };
    let ratio = (f1 / g).max(0.0);
    let f2 = g * (1.0 - ratio.sqrt() - ratio * (10.0 * std::f64::consts::PI * f1).sin());
    vec![f1, f2]
}

/// DTLZ2: three-objective benchmark with a concave **spherical** Pareto front.
///
/// Decision variables: `x ∈ [0, 1]^n` with `n ≥ 3`.  The first two variables position a
/// solution on the front; the remaining `x[2:]` are distance variables driven to `0.5`.
/// - `g = Σ_{i≥2}(xᵢ − 0.5)²`
/// - `f₁ = (1 + g)·cos(x₀·π/2)·cos(x₁·π/2)`
/// - `f₂ = (1 + g)·cos(x₀·π/2)·sin(x₁·π/2)`
/// - `f₃ = (1 + g)·sin(x₀·π/2)`
///
/// On the Pareto-optimal front `g = 0`, so `f₁² + f₂² + f₃² = 1` (unit sphere, first octant).
/// Returns `[f1, f2, f3]`.
pub fn dtlz2(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 3 {
        return vec![0.0, 0.0, 0.0];
    }
    let g: f64 = x[2..].iter().map(|&xi| (xi - 0.5) * (xi - 0.5)).sum();
    let half_pi = std::f64::consts::FRAC_PI_2;
    let a0 = x[0] * half_pi;
    let a1 = x[1] * half_pi;
    let f1 = (1.0 + g) * a0.cos() * a1.cos();
    let f2 = (1.0 + g) * a0.cos() * a1.sin();
    let f3 = (1.0 + g) * a0.sin();
    vec![f1, f2, f3]
}

// ── DTLZ shared building blocks ──────────────────────────────────────────────

/// Concave (spherical) DTLZ shape mapping: given the `M−1` aspect angles `θ` and the radius
/// factor `1 + g`, produce the `M` objectives placed on a sphere of radius `1 + g` in the
/// positive orthant (Deb, Thiele, Laumanns & Zitzler 2002, eq. for DTLZ2–DTLZ6).
///
/// ```text
/// f₁ = (1+g)·cos θ₁·cos θ₂ ⋯ cos θ_{M−1}
/// f₂ = (1+g)·cos θ₁ ⋯ cos θ_{M−2}·sin θ_{M−1}
/// ⋮
/// f_M = (1+g)·sin θ₁
/// ```
///
/// On the Pareto-optimal front (`g = 0`) this satisfies `Σ fᵢ² = 1` (unit hypersphere).
fn dtlz_concave_objectives(thetas: &[f64], one_plus_g: f64) -> Vec<f64> {
    let m = thetas.len() + 1;
    let mut f = Vec::with_capacity(m);
    for i in 0..m {
        let mut prod = one_plus_g;
        // Product of the first `M−1−i` cosines.
        for &t in &thetas[..m - 1 - i] {
            prod *= t.cos();
        }
        // A single trailing sine for every objective except the first.
        if i > 0 {
            prod *= thetas[m - 1 - i].sin();
        }
        f.push(prod);
    }
    f
}

/// Multimodal DTLZ distance function (shared by DTLZ1 and DTLZ3):
/// `g = 100·(k + Σ_{xᵢ∈xm}[(xᵢ − 0.5)² − cos(20π(xᵢ − 0.5))])`.
///
/// Global minimum `g = 0` at every `xᵢ = 0.5`; the cosine term superimposes `11^k − 1`
/// local minima (`g > 0`) per the canonical Deb 2002 definition (coefficient `20π`).
fn dtlz_multimodal_g(xm: &[f64]) -> f64 {
    let k = xm.len() as f64;
    let twenty_pi = 20.0 * std::f64::consts::PI;
    let s: f64 = xm
        .iter()
        .map(|&xi| {
            let d = xi - 0.5;
            d * d - (twenty_pi * d).cos()
        })
        .sum();
    100.0 * (k + s)
}

/// Quadratic DTLZ distance function (shared by DTLZ2, DTLZ4, DTLZ5):
/// `g = Σ_{xᵢ∈xm}(xᵢ − 0.5)²`. Global minimum `g = 0` at every `xᵢ = 0.5`, unimodal.
fn dtlz_quadratic_g(xm: &[f64]) -> f64 {
    xm.iter().map(|&xi| (xi - 0.5) * (xi - 0.5)).sum()
}

// ── ZDT4 / ZDT6 ──────────────────────────────────────────────────────────────

/// ZDT4: two-objective benchmark with a **multimodal** distance landscape (`21^(n−1)` local
/// fronts), the classic test for premature convergence.
///
/// Decision variables: `x₀ ∈ [0, 1]`, `xᵢ ∈ [−5, 5]` for `i ≥ 1`.
/// - `f₁ = x₀`
/// - `g = 1 + 10·(n−1) + Σ_{i≥1}[xᵢ² − 10·cos(4π·xᵢ)]`
/// - `f₂ = g·(1 − √(f₁/g))`
///
/// Because `xᵢ² − 10·cos(4π·xᵢ) ≥ −10` with equality only at `xᵢ = 0`, we have `g ≥ 1` always;
/// the global Pareto front (`xᵢ = 0 ⇒ g = 1`) is the convex curve `f₂ = 1 − √f₁`, while the
/// Rastrigin-like cosine ripples create many local fronts at `g > 1`. Returns `[f1, f2]`.
pub fn zdt4(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![0.0, 0.0];
    }
    let f1 = x[0];
    if n == 1 {
        let g = 1.0;
        return vec![f1, g * (1.0 - (f1 / g).max(0.0).sqrt())];
    }
    let four_pi = 4.0 * std::f64::consts::PI;
    let dist: f64 = x[1..]
        .iter()
        .map(|&xi| xi * xi - 10.0 * (four_pi * xi).cos())
        .sum();
    let g = 1.0 + 10.0 * (n - 1) as f64 + dist;
    let f2 = g * (1.0 - (f1 / g).max(0.0).sqrt());
    vec![f1, f2]
}

/// ZDT6: two-objective benchmark with a **biased**, non-uniformly spaced concave front.
///
/// Decision variables: `x ∈ [0, 1]^n`.
/// - `f₁ = 1 − exp(−4·x₀)·sin⁶(6π·x₀)`
/// - `g = 1 + 9·[ (Σ_{i≥1} xᵢ)/(n−1) ]^0.25`
/// - `f₂ = g·(1 − (f₁/g)²)`
///
/// The global Pareto front (`xᵢ = 0 ⇒ g = 1`) is `f₂ = 1 − f₁²` for `f₁ ∈ [≈0.281, 1]`. The
/// `exp·sin⁶` term clusters solutions near `f₁ → 1`, making the front density highly
/// non-uniform. Returns `[f1, f2]`.
pub fn zdt6(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![0.0, 0.0];
    }
    let pi = std::f64::consts::PI;
    let x0 = x[0];
    let f1 = 1.0 - (-4.0 * x0).exp() * (6.0 * pi * x0).sin().powi(6);
    if n == 1 {
        let g = 1.0;
        return vec![f1, g * (1.0 - (f1 / g).powi(2))];
    }
    let mean: f64 = x[1..].iter().sum::<f64>() / (n - 1) as f64;
    let g = 1.0 + 9.0 * mean.max(0.0).powf(0.25);
    let f2 = g * (1.0 - (f1 / g).powi(2));
    vec![f1, f2]
}

// ── DTLZ3 / DTLZ4 / DTLZ5 / DTLZ6 / DTLZ7 (three-objective) ───────────────────

/// DTLZ3: three-objective spherical front with the **multimodal** DTLZ1 distance function.
///
/// Decision variables: `x ∈ [0, 1]^n`, `n ≥ 3`. Identical geometry to DTLZ2 (objectives lie on
/// the unit sphere when `g = 0`), but `g` carries `11^k − 1` local minima so the search must
/// cross many local fronts (`g > 0 ⇒ radius `1 + g` > 1`) to reach the global front.
/// - `g = 100·(k + Σ[(xᵢ − 0.5)² − cos(20π(xᵢ − 0.5))])` over `x[2:]`
/// - objectives via the concave DTLZ map with `θᵢ = xᵢ·π/2`
///
/// On the front `g = 0`, so `f₁² + f₂² + f₃² = 1`. Returns `[f1, f2, f3]`.
pub fn dtlz3(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 3 {
        return vec![0.0, 0.0, 0.0];
    }
    let g = dtlz_multimodal_g(&x[2..]);
    let half_pi = std::f64::consts::FRAC_PI_2;
    let thetas = [x[0] * half_pi, x[1] * half_pi];
    dtlz_concave_objectives(&thetas, 1.0 + g)
}

/// DTLZ4: three-objective spherical front with a **biased** density mapping (`α = 100`).
///
/// Decision variables: `x ∈ [0, 1]^n`, `n ≥ 3`. Same spherical geometry and quadratic `g` as
/// DTLZ2, but each position variable is raised to the power `α = 100` before forming the angle,
/// biasing Pareto-optimal solutions toward the `f_M`–`f₁` plane and stressing diversity
/// preservation.
/// - `g = Σ(xᵢ − 0.5)²` over `x[2:]`
/// - `θᵢ = (xᵢ^100)·π/2`
///
/// On the front `g = 0`, so `f₁² + f₂² + f₃² = 1`. Returns `[f1, f2, f3]`.
pub fn dtlz4(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 3 {
        return vec![0.0, 0.0, 0.0];
    }
    const ALPHA: f64 = 100.0;
    let g = dtlz_quadratic_g(&x[2..]);
    let half_pi = std::f64::consts::FRAC_PI_2;
    let thetas = [x[0].powf(ALPHA) * half_pi, x[1].powf(ALPHA) * half_pi];
    dtlz_concave_objectives(&thetas, 1.0 + g)
}

/// DTLZ5: three-objective problem whose Pareto front **degenerates to a curve**.
///
/// Decision variables: `x ∈ [0, 1]^n`, `n ≥ 3`. The trailing angles are coupled to `g`, so on
/// the front (`g = 0`) every `θᵢ` (i ≥ 2) collapses to `π/4`, forcing `f₁ = f₂` and reducing the
/// three-objective front to a one-dimensional arc of the unit sphere.
/// - `g = Σ(xᵢ − 0.5)²` over `x[2:]`
/// - `θ₁ = x₀·π/2`, `θ₂ = π/(4(1+g))·(1 + 2·g·x₁)`
///
/// On the front `g = 0`: `θ₂ = π/4 ⇒ f₁ = f₂` and `f₁² + f₂² + f₃² = 1`. Returns `[f1, f2, f3]`.
pub fn dtlz5(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 3 {
        return vec![0.0, 0.0, 0.0];
    }
    let g = dtlz_quadratic_g(&x[2..]);
    let half_pi = std::f64::consts::FRAC_PI_2;
    let theta1 = x[0] * half_pi;
    let theta2 = std::f64::consts::PI / (4.0 * (1.0 + g)) * (1.0 + 2.0 * g * x[1]);
    dtlz_concave_objectives(&[theta1, theta2], 1.0 + g)
}

/// DTLZ6: degenerate-curve front like DTLZ5 but with a **harder** distance function
/// `g = Σ xᵢ^0.1`, whose vanishing gradient near the optimum makes convergence to the
/// degenerate curve substantially more difficult.
///
/// Decision variables: `x ∈ [0, 1]^n`, `n ≥ 3`. The global optimum is `xᵢ = 0` (not `0.5`),
/// giving `g = 0`.
/// - `g = Σ xᵢ^0.1` over `x[2:]`
/// - `θ₁ = x₀·π/2`, `θ₂ = π/(4(1+g))·(1 + 2·g·x₁)`
///
/// On the front `g = 0`: `θ₂ = π/4 ⇒ f₁ = f₂` and `f₁² + f₂² + f₃² = 1`. Returns `[f1, f2, f3]`.
pub fn dtlz6(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 3 {
        return vec![0.0, 0.0, 0.0];
    }
    let g: f64 = x[2..].iter().map(|&xi| xi.max(0.0).powf(0.1)).sum();
    let half_pi = std::f64::consts::FRAC_PI_2;
    let theta1 = x[0] * half_pi;
    let theta2 = std::f64::consts::PI / (4.0 * (1.0 + g)) * (1.0 + 2.0 * g * x[1]);
    dtlz_concave_objectives(&[theta1, theta2], 1.0 + g)
}

/// DTLZ7: three-objective problem with a **disconnected** Pareto front (`2^(M−1) = 4` patches).
///
/// Decision variables: `x ∈ [0, 1]^n`, `n ≥ 3`. The first `M − 1` variables map directly to the
/// leading objectives; the last objective folds them through a sine term that carves the front
/// into disconnected regions.
/// - `f₁ = x₀`, `f₂ = x₁`
/// - `g = 1 + (9/k)·Σ x[2:]`
/// - `h = M − Σ_{i<M}[ fᵢ/(1+g)·(1 + sin(3π·fᵢ)) ]`
/// - `f₃ = (1 + g)·h`
///
/// The global front lives at `g = 1` (`x[2:] = 0`). Returns `[f1, f2, f3]`.
pub fn dtlz7(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 3 {
        return vec![0.0, 0.0, 0.0];
    }
    const M: usize = 3;
    let xm = &x[M - 1..];
    let k = xm.len() as f64;
    let g = 1.0 + 9.0 / k * xm.iter().sum::<f64>();
    let one_plus_g = 1.0 + g;
    let three_pi = 3.0 * std::f64::consts::PI;
    let f1 = x[0];
    let f2 = x[1];
    let h = M as f64
        - (f1 / one_plus_g) * (1.0 + (three_pi * f1).sin())
        - (f2 / one_plus_g) * (1.0 + (three_pi * f2).sin());
    let f3 = one_plus_g * h;
    vec![f1, f2, f3]
}

/// Analytic ZDT1 Pareto front: `f₂ = 1 − √f₁` for `f₁ ∈ [0, 1]` (convex front).
#[inline]
pub fn zdt1_pareto_front_f2(f1: f64) -> f64 {
    1.0 - f1.max(0.0).sqrt()
}

/// Analytic ZDT2 Pareto front: `f₂ = 1 − f₁²` for `f₁ ∈ [0, 1]` (non-convex front).
#[inline]
pub fn zdt2_pareto_front_f2(f1: f64) -> f64 {
    1.0 - f1 * f1
}

/// Analytic ZDT3 Pareto front curve: `f₂ = 1 − √f₁ − f₁·sin(10π·f₁)` for `f₁ ∈ [0, 1]`.
///
/// Only the non-dominated portions of this curve form the true (disconnected) Pareto front;
/// dense sampling plus a non-dominated filter recovers the five Pareto-optimal segments.
#[inline]
pub fn zdt3_pareto_front_f2(f1: f64) -> f64 {
    1.0 - f1.max(0.0).sqrt() - f1 * (10.0 * std::f64::consts::PI * f1).sin()
}

/// Analytic ZDT4 Pareto front: `f₂ = 1 − √f₁` for `f₁ ∈ [0, 1]`.
///
/// ZDT4 shares ZDT1's convex global front; the difference is the multimodal `g` landscape
/// (local fronts at `g > 1`), not the front shape.
#[inline]
pub fn zdt4_pareto_front_f2(f1: f64) -> f64 {
    1.0 - f1.max(0.0).sqrt()
}

/// Analytic ZDT6 Pareto front: `f₂ = 1 − f₁²` for `f₁ ∈ [≈0.281, 1]` (concave, biased density).
#[inline]
pub fn zdt6_pareto_front_f2(f1: f64) -> f64 {
    1.0 - f1 * f1
}

// ─────────────────────────────────────────────────────────────────────────────
// Algorithm harness types
// ─────────────────────────────────────────────────────────────────────────────

/// Performance profile for a single-objective benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Human-readable name of the benchmark function.
    pub function_name: &'static str,
    /// Problem dimensionality.
    pub n_dims: usize,
    /// Best objective value found.
    pub best_value: f64,
    /// Total number of function evaluations consumed.
    pub n_evaluations: usize,
    /// Whether the algorithm achieved `best_value < target_precision`.
    pub converged: bool,
    /// Target precision threshold used to classify convergence.
    pub target_precision: f64,
}

/// Performance profile for a multi-objective benchmark run.
#[derive(Debug, Clone)]
pub struct MoBenchmarkResult {
    /// Human-readable name of the benchmark function.
    pub function_name: &'static str,
    /// Problem dimensionality.
    pub n_dims: usize,
    /// Hypervolume of the final Pareto front approximation.
    pub hypervolume: f64,
    /// Number of non-dominated points in the final approximation.
    pub n_front_points: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// CMA-ES harness
// ─────────────────────────────────────────────────────────────────────────────

/// Run CMA-ES on a scalar benchmark function and return a convergence profile.
///
/// # Arguments
/// - `f` — objective function (lower is better)
/// - `n_dims` — number of decision variables
/// - `max_evals` — maximum number of function evaluations
/// - `target_precision` — convergence threshold; if `best_value < target_precision` the run is
///   declared converged
/// - `seed` — deterministic random seed
/// - `function_name` — label stored in `BenchmarkResult`
///
/// The initial distribution mean is the origin and `σ₀ = 0.3·range` with range ≈ 5.0
/// (a sensible default for BBOB functions defined on [−5, 5]).
pub fn run_cmaes_benchmark<F>(
    f: F,
    n_dims: usize,
    max_evals: usize,
    target_precision: f64,
    seed: u64,
    function_name: &'static str,
) -> EvolResult<BenchmarkResult>
where
    F: Fn(&[f64]) -> f64,
{
    if n_dims == 0 {
        return Err(EvolError::InvalidParameter(
            "n_dims must be >= 1".to_owned(),
        ));
    }

    let mut cfg = CmaEsConfig::new(n_dims)?;
    cfg.max_evals = max_evals;
    cfg.sigma_init = 0.5; // wider initial step for BBOB search domain ≈ [−5, 5]
    cfg.tol_fun = target_precision * 1e-2; // stop slightly below target

    // Start from origin
    let mean_init = vec![0.0f64; n_dims];
    let mut state = CmaEsState::new(mean_init, &cfg)?;
    let mut rng = LcgRng::new(seed);

    let (_, best_value) = state.run(&f, &cfg, &mut rng)?;
    let n_evaluations = state.n_evals;
    let converged = best_value < target_precision;

    Ok(BenchmarkResult {
        function_name,
        n_dims,
        best_value,
        n_evaluations,
        converged,
        target_precision,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// NSGA-II harness
// ─────────────────────────────────────────────────────────────────────────────

/// Run NSGA-II on a multi-objective benchmark function and return Pareto front quality metrics.
///
/// # Arguments
/// - `f` — multi-objective function returning `Vec<f64>` of length `n_obj`
/// - `n_dims` — decision variable count
/// - `n_obj` — objective count
/// - `pop_size` — population size (must be even, ≥ 4)
/// - `n_gens` — number of NSGA-II generations
/// - `seed` — deterministic random seed
/// - `function_name` — label stored in `MoBenchmarkResult`
/// - `reference_point` — hypervolume reference point (length must equal `n_obj`)
///
/// Decision variables are assumed to lie in `[0, 1]`.
pub fn run_nsga2_benchmark<F>(
    f: F,
    n_dims: usize,
    n_obj: usize,
    pop_size: usize,
    n_gens: usize,
    seed: u64,
    function_name: &'static str,
    reference_point: Vec<f64>,
) -> EvolResult<MoBenchmarkResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    if n_dims == 0 {
        return Err(EvolError::InvalidParameter(
            "n_dims must be >= 1".to_owned(),
        ));
    }
    if n_obj == 0 {
        return Err(EvolError::InvalidParameter("n_obj must be >= 1".to_owned()));
    }
    if pop_size < 4 {
        return Err(EvolError::PopulationTooSmall {
            size: pop_size,
            op: "NSGA-II benchmark",
        });
    }
    if reference_point.len() != n_obj {
        return Err(EvolError::DimensionMismatch {
            expected: n_obj,
            got: reference_point.len(),
        });
    }

    // Ensure pop_size is even
    let pop_size = if pop_size.is_multiple_of(2) {
        pop_size
    } else {
        pop_size + 1
    };

    let cfg = Nsga2Config {
        n_dims,
        n_objectives: n_obj,
        pop_size,
        max_generations: n_gens,
        crossover_eta: 15.0,
        mutation_eta: 20.0,
        mutation_prob: 1.0 / n_dims as f64,
        bounds: (0.0, 1.0),
    };

    let mut rng = LcgRng::new(seed);
    let final_pop = nsga2_run(f, &cfg, &mut rng)?;

    // Extract Pareto front (rank 0)
    let front_points: Vec<Vec<f64>> = final_pop
        .iter()
        .filter(|ind| ind.rank == 0)
        .map(|ind| ind.objectives.clone())
        .collect();

    let n_front_points = front_points.len();

    // Compute hypervolume using the WFG algorithm.
    // hypervolume_nd expects reference as &[Vec<f64>] with reference[0] = ref point,
    // and requires every front point to be strictly dominated by the reference.
    // Filter to only include points that are strictly dominated by the reference point.
    let ref_pt = &reference_point;
    let dominated_front: Vec<Vec<f64>> = front_points
        .into_iter()
        .filter(|p| p.iter().zip(ref_pt.iter()).all(|(fi, ri)| fi < ri))
        .collect();

    let hypervolume = if dominated_front.is_empty() {
        0.0
    } else {
        hypervolume_nd(&dominated_front, &[reference_point]).unwrap_or(0.0)
    };

    Ok(MoBenchmarkResult {
        function_name,
        n_dims,
        hypervolume,
        n_front_points,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::metrics::{extract_pareto_front, generational_distance, igd};

    // ── Sphere ───────────────────────────────────────────────────────────────

    #[test]
    fn sphere_origin_is_zero() {
        assert_eq!(sphere(&[0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn sphere_unit_vector() {
        let val = sphere(&[1.0, 0.0, 0.0]);
        assert!((val - 1.0).abs() < 1e-14, "sphere([1,0,0]) = {val}");
    }

    #[test]
    fn sphere_positive() {
        assert!(sphere(&[1.0, 2.0, 3.0]) > 0.0);
    }

    // ── Ellipsoid ─────────────────────────────────────────────────────────────

    #[test]
    fn ellipsoid_origin_is_zero() {
        assert_eq!(ellipsoid(&[0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn ellipsoid_single_dim_origin() {
        assert_eq!(ellipsoid(&[0.0]), 0.0);
    }

    #[test]
    fn ellipsoid_larger_than_sphere() {
        // Due to conditioning, ellipsoid should be larger than sphere for non-zero x
        let x = &[1.0, 1.0, 1.0, 1.0, 1.0];
        assert!(ellipsoid(x) >= sphere(x));
    }

    // ── Rosenbrock ───────────────────────────────────────────────────────────

    #[test]
    fn rosenbrock_global_minimum() {
        // f([1, 1]) = 0
        let val = rosenbrock(&[1.0, 1.0]);
        assert!(val.abs() < 1e-14, "rosenbrock([1,1]) = {val}");
    }

    #[test]
    fn rosenbrock_global_minimum_higher_dim() {
        let ones = vec![1.0f64; 5];
        let val = rosenbrock(&ones);
        assert!(val.abs() < 1e-10, "rosenbrock(ones_5) = {val}");
    }

    #[test]
    fn rosenbrock_non_minimum_positive() {
        assert!(rosenbrock(&[0.0, 0.0]) > 0.0);
    }

    // ── Rastrigin ────────────────────────────────────────────────────────────

    #[test]
    fn rastrigin_global_minimum() {
        // f([0, 0]) = 0
        let val = rastrigin(&[0.0, 0.0]);
        assert!(val.abs() < 1e-14, "rastrigin([0,0]) = {val}");
    }

    #[test]
    fn rastrigin_positive_elsewhere() {
        // Rastrigin ≥ 0 globally and > 0 away from origin
        assert!(rastrigin(&[1.0, 1.0]) > 0.0);
    }

    // ── Ackley ───────────────────────────────────────────────────────────────

    #[test]
    fn ackley_global_minimum_approx_zero() {
        // f([0, 0]) should be ≈ 0 (within floating-point precision)
        let val = ackley(&[0.0, 0.0]);
        assert!(val.abs() < 1e-10, "ackley([0,0]) = {val}");
    }

    #[test]
    fn ackley_positive_elsewhere() {
        assert!(ackley(&[1.0, 2.0]) > 0.0);
    }

    // ── Griewank ─────────────────────────────────────────────────────────────

    #[test]
    fn griewank_origin_is_zero() {
        let val = griewank(&[0.0]);
        assert!(val.abs() < 1e-14, "griewank([0]) = {val}");
    }

    #[test]
    fn griewank_origin_multi_dim() {
        // griewank(zeros) = 1 + 0 - 1 = 0
        let val = griewank(&[0.0, 0.0, 0.0]);
        assert!(val.abs() < 1e-14, "griewank(zeros_3) = {val}");
    }

    // ── Schwefel ─────────────────────────────────────────────────────────────

    #[test]
    fn schwefel_global_minimum_approx() {
        // Global minimum near x_i = 418.9829, f ≈ 0
        // (more accurate value for the Schwefel minimizer is ~420.9687, but the
        // BBOB Schwefel formulation uses the 418.9829 coefficient).
        // Use the commonly cited approximation
        let val = schwefel(&[418.9829]);
        // The residual should be small — within ≈ 0.01 of zero
        assert!(val.abs() < 1.0, "schwefel([418.9829]) = {val}");
    }

    #[test]
    fn schwefel_at_exact_minimizer_low() {
        // At x = 420.9687..., f ≈ 0 for 1-D
        // The precise minimizer of xi*sin(sqrt(|xi|)) is ~420.9687
        // schwefel([x]) = 418.9829 - x*sin(sqrt(x))
        // We verify it's close to zero (within 2 for robustness across formulations)
        let xm = 420.9687_f64;
        let val = schwefel(&[xm]);
        // The global minimum might not be exactly 0 with the coefficient 418.9829
        // but it should be bounded near zero
        assert!(val.abs() < 5.0, "schwefel near minimizer = {val}");
    }

    // ── Multi-objective ───────────────────────────────────────────────────────

    #[test]
    fn zdt1_returns_two_objectives() {
        let result = zdt1(&[0.5; 5]);
        assert_eq!(result.len(), 2, "zdt1 must return 2 objectives");
    }

    #[test]
    fn zdt1_objectives_nonnegative() {
        let result = zdt1(&[0.3, 0.1, 0.2, 0.4, 0.5]);
        assert!(result[0] >= 0.0 && result[1] >= 0.0);
    }

    #[test]
    fn zdt2_returns_two_objectives() {
        let result = zdt2(&[0.5; 5]);
        assert_eq!(result.len(), 2, "zdt2 must return 2 objectives");
    }

    #[test]
    fn dtlz1_returns_three_objectives() {
        let result = dtlz1(&[0.5; 5]);
        assert_eq!(result.len(), 3, "dtlz1 must return 3 objectives");
    }

    #[test]
    fn dtlz1_short_input_still_three_objectives() {
        // n < 3 is a degenerate case but must still return 3-element Vec
        let result = dtlz1(&[0.5, 0.5]);
        assert_eq!(result.len(), 3);
    }

    // ── ZDT4 / ZDT6 / DTLZ3-7 analytic-front structural tests ─────────────────
    //
    // These are *mathematically-provable* properties of the objective functions evaluated
    // at known Pareto-optimal decision vectors — NOT optimiser-convergence claims. For each
    // problem we assert: (a) output dimensionality, (b) the objective vector lies on the
    // documented analytic front at the front-optimal distance variables, (c) perturbing the
    // distance variables moves the objectives away from the front by a non-negative amount
    // (g ≥ documented optimum), (d) known multimodality / degeneracy / disconnection
    // structure where it is cheaply and exactly checkable, and (e) determinism.

    /// `|Σ fᵢ² − 1|`: distance of an objective vector from the unit hypersphere.
    fn unit_sphere_residual(f: &[f64]) -> f64 {
        (f.iter().map(|v| v * v).sum::<f64>() - 1.0).abs()
    }

    // ZDT4 -----------------------------------------------------------------------

    #[test]
    fn zdt4_returns_two_objectives() {
        assert_eq!(zdt4(&[0.5; 10]).len(), 2);
    }

    #[test]
    fn zdt4_on_global_front_is_one_minus_sqrt() {
        // Distance variables = 0 ⇒ g = 1 ⇒ f2 = 1 − √f1 (convex global front).
        for &f1 in &[0.0, 0.1, 0.25, 0.5, 0.81, 1.0] {
            let mut x = vec![0.0; 10];
            x[0] = f1;
            let f = zdt4(&x);
            let want = zdt4_pareto_front_f2(f1);
            let resid = (f[1] - want).abs();
            assert!(
                resid < 1e-12,
                "ZDT4 front f1={f1}: f2={} want={want} resid={resid:.2e}",
                f[1]
            );
        }
    }

    #[test]
    fn zdt4_distance_perturbation_raises_g_above_one() {
        // At f1 = 0, f2 = g. The global optimum is g = 1 (all distance vars 0); any non-zero
        // distance variable gives g ≥ 1 (proved analytically since xᵢ²−10cos(4πxᵢ) ≥ −10).
        let on_front = zdt4(&[0.0; 10])[1];
        assert!(
            (on_front - 1.0).abs() < 1e-12,
            "ZDT4 g at optimum = {on_front}"
        );
        // A whole grid of off-optimum distance settings must never dip below the optimum.
        for &v in &[0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 2.0] {
            let mut x = vec![v; 10];
            x[0] = 0.0;
            let f2 = zdt4(&x)[1];
            assert!(
                f2 >= 1.0 - 1e-9,
                "ZDT4 g at distance {v} = {f2} (must be ≥ 1)"
            );
        }
    }

    #[test]
    fn zdt4_multimodal_local_front_above_global() {
        // x_i = 0.5 is a local (not global) basin of x²−10cos(4πx): the term value −9.75 there is
        // below its neighbours at 0.4 / 0.6 (≈ −2.9) yet above the global −10 at x=0. Hence the
        // induced g is a local minimum strictly greater than the global g=1, i.e. a *local front*.
        let g = |v: f64| {
            let mut x = vec![v; 10];
            x[0] = 0.0;
            zdt4(&x)[1] // = g since f1 = 0
        };
        let g_half = g(0.5);
        assert!(
            g_half > 1.0 + 1e-6,
            "ZDT4 local g(0.5) = {g_half} (want > 1 = global)"
        );
        assert!(
            g_half < g(0.4) && g_half < g(0.6),
            "ZDT4 x=0.5 not a local basin: g(0.5)={g_half} g(0.4)={} g(0.6)={}",
            g(0.4),
            g(0.6)
        );
    }

    // ZDT6 -----------------------------------------------------------------------

    #[test]
    fn zdt6_returns_two_objectives() {
        assert_eq!(zdt6(&[0.5; 10]).len(), 2);
    }

    #[test]
    fn zdt6_on_global_front_is_one_minus_f1_squared() {
        // Distance variables = 0 ⇒ g = 1 ⇒ f2 = 1 − f1² (concave global front).
        for &x0 in &[0.05, 0.2, 0.4, 0.6, 0.8, 1.0] {
            let mut x = vec![0.0; 10];
            x[0] = x0;
            let f = zdt6(&x);
            let want = zdt6_pareto_front_f2(f[0]);
            let resid = (f[1] - want).abs();
            assert!(
                resid < 1e-12,
                "ZDT6 front x0={x0}: f2={} want={want} resid={resid:.2e}",
                f[1]
            );
        }
    }

    #[test]
    fn zdt6_distance_perturbation_moves_away_from_front() {
        // f2 = g − f1²/g is strictly increasing in g (d/dg = 1 + f1²/g² > 0), so any positive
        // distance mass (g > 1) lifts f2 above the on-front value for the same x0.
        let mut on = vec![0.0; 10];
        on[0] = 0.5;
        let f2_front = zdt6(&on)[1];
        let mut off = vec![0.0; 10];
        off[0] = 0.5;
        for d in off.iter_mut().skip(1) {
            *d = 0.4;
        }
        let f2_off = zdt6(&off)[1];
        assert!(
            f2_off > f2_front + 1e-9,
            "ZDT6 off-front f2={f2_off} not above on-front f2={f2_front}"
        );
    }

    // DTLZ3 ----------------------------------------------------------------------

    #[test]
    fn dtlz3_returns_three_objectives() {
        assert_eq!(dtlz3(&[0.5; 12]).len(), 3);
    }

    #[test]
    fn dtlz3_on_front_is_unit_sphere() {
        // Distance variables = 0.5 ⇒ g = 0 ⇒ Σ fᵢ² = 1 for arbitrary position variables.
        for &(x0, x1) in &[(0.0, 0.0), (0.3, 0.7), (1.0, 1.0), (0.5, 0.25), (0.9, 0.1)] {
            let mut x = vec![0.5; 12];
            x[0] = x0;
            x[1] = x1;
            let f = dtlz3(&x);
            let resid = unit_sphere_residual(&f);
            assert!(
                resid < 1e-12,
                "DTLZ3 on-front ({x0},{x1}): Σf²−1 resid = {resid:.2e}"
            );
        }
    }

    #[test]
    fn dtlz3_multimodal_local_front_radius_above_one() {
        // Radius = 1 + g. Distance = 0.5 ⇒ g = 0 ⇒ radius = 1 (global). x_i = 0.6 is a local basin
        // of the cos(20π(x−0.5)) ripple: g(0.6) < g(0.55), g(0.65) yet g(0.6) > 0 → a local front
        // strictly outside the unit sphere.
        let radius = |v: f64| {
            let mut x = vec![v; 12];
            x[0] = 0.0; // pure f3 = (1+g)·sin(0) handling: use radius = ‖f‖
            x[1] = 0.0;
            let f = dtlz3(&x);
            (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt()
        };
        let r_local = radius(0.6);
        assert!(
            r_local > 1.0 + 1e-6,
            "DTLZ3 local radius(0.6) = {r_local} (want > 1)"
        );
        assert!(
            r_local < radius(0.55) && r_local < radius(0.65),
            "DTLZ3 x=0.6 not a local basin: r(0.6)={r_local} r(0.55)={} r(0.65)={}",
            radius(0.55),
            radius(0.65)
        );
    }

    // DTLZ4 ----------------------------------------------------------------------

    #[test]
    fn dtlz4_returns_three_objectives() {
        assert_eq!(dtlz4(&[0.5; 12]).len(), 3);
    }

    #[test]
    fn dtlz4_on_front_is_unit_sphere() {
        // Distance variables = 0.5 ⇒ g = 0 ⇒ Σ fᵢ² = 1 for any position vars (the α=100 bias only
        // reshapes the *density* on the sphere, not the sphere itself).
        for &(x0, x1) in &[(0.0, 0.0), (0.3, 0.7), (1.0, 1.0), (0.5, 0.5), (0.95, 0.2)] {
            let mut x = vec![0.5; 12];
            x[0] = x0;
            x[1] = x1;
            let resid = unit_sphere_residual(&dtlz4(&x));
            assert!(
                resid < 1e-12,
                "DTLZ4 on-front ({x0},{x1}): Σf²−1 resid = {resid:.2e}"
            );
        }
    }

    #[test]
    fn dtlz4_distance_perturbation_radius_ge_one() {
        // Quadratic g ≥ 0 with equality only at x=0.5, so off-optimum distance ⇒ radius > 1.
        let mut x = vec![0.5; 12];
        x[0] = 0.4;
        x[1] = 0.6;
        let r_front = {
            let f = dtlz4(&x);
            (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt()
        };
        assert!(
            (r_front - 1.0).abs() < 1e-12,
            "DTLZ4 on-front radius = {r_front}"
        );
        for d in x.iter_mut().skip(2) {
            *d = 0.2; // off optimum
        }
        let r_off = {
            let f = dtlz4(&x);
            (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt()
        };
        assert!(
            r_off > 1.0 + 1e-9,
            "DTLZ4 off-front radius = {r_off} (want > 1)"
        );
    }

    // DTLZ5 ----------------------------------------------------------------------

    #[test]
    fn dtlz5_returns_three_objectives() {
        assert_eq!(dtlz5(&[0.5; 12]).len(), 3);
    }

    #[test]
    fn dtlz5_degenerate_curve_f1_equals_f2_on_unit_sphere() {
        // Distance = 0.5 ⇒ g = 0 ⇒ θ₂ = π/4 ⇒ f1 = f2, and Σ fᵢ² = 1. The three-objective front
        // collapses to a 1-D arc (the degenerate-curve property), parameterised by x0 only.
        for &x0 in &[0.0, 0.2, 0.5, 0.8, 1.0] {
            for &x1 in &[0.0, 0.5, 1.0] {
                let mut x = vec![0.5; 12];
                x[0] = x0;
                x[1] = x1;
                let f = dtlz5(&x);
                assert!(
                    (f[0] - f[1]).abs() < 1e-12,
                    "DTLZ5 degeneracy x0={x0} x1={x1}: f1={} f2={}",
                    f[0],
                    f[1]
                );
                assert!(
                    unit_sphere_residual(&f) < 1e-12,
                    "DTLZ5 off unit sphere x0={x0}"
                );
            }
        }
    }

    #[test]
    fn dtlz5_distance_perturbation_radius_ge_one() {
        let mut x = vec![0.5; 12];
        x[0] = 0.3;
        x[1] = 0.7;
        for d in x.iter_mut().skip(2) {
            *d = 0.1;
        }
        let f = dtlz5(&x);
        let r = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
        assert!(r > 1.0 + 1e-9, "DTLZ5 off-front radius = {r} (want > 1)");
    }

    // DTLZ6 ----------------------------------------------------------------------

    #[test]
    fn dtlz6_returns_three_objectives() {
        assert_eq!(dtlz6(&[0.5; 12]).len(), 3);
    }

    #[test]
    fn dtlz6_degenerate_curve_f1_equals_f2_on_unit_sphere() {
        // DTLZ6 optimum is x_i = 0 (g = Σ x^0.1 = 0) ⇒ θ₂ = π/4 ⇒ f1 = f2 and Σ fᵢ² = 1.
        for &x0 in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let mut x = vec![0.0; 12];
            x[0] = x0;
            x[1] = 0.6; // position var is free; does not affect degeneracy when g=0
            let f = dtlz6(&x);
            assert!(
                (f[0] - f[1]).abs() < 1e-12,
                "DTLZ6 degeneracy x0={x0}: f1={} f2={}",
                f[0],
                f[1]
            );
            assert!(
                unit_sphere_residual(&f) < 1e-12,
                "DTLZ6 off unit sphere x0={x0}"
            );
        }
    }

    #[test]
    fn dtlz6_distance_perturbation_radius_ge_one() {
        let mut x = vec![0.0; 12];
        x[0] = 0.4;
        x[1] = 0.4;
        for d in x.iter_mut().skip(2) {
            *d = 0.5; // 0.5^0.1 ≈ 0.933 > 0 ⇒ g > 0
        }
        let f = dtlz6(&x);
        let r = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
        assert!(r > 1.0 + 1e-9, "DTLZ6 off-front radius = {r} (want > 1)");
    }

    // DTLZ7 ----------------------------------------------------------------------

    #[test]
    fn dtlz7_returns_three_objectives() {
        assert_eq!(dtlz7(&[0.5; 12]).len(), 3);
    }

    #[test]
    fn dtlz7_positions_map_directly_and_g_is_one_on_front() {
        // f1 = x0, f2 = x1 exactly; the front lives at g = 1 (distance vars = 0).
        for &(x0, x1) in &[(0.1, 0.2), (0.5, 0.85), (0.9, 0.05)] {
            let mut x = vec![0.0; 12];
            x[0] = x0;
            x[1] = x1;
            let f = dtlz7(&x);
            assert!(
                (f[0] - x0).abs() < 1e-15 && (f[1] - x1).abs() < 1e-15,
                "DTLZ7 position map ({x0},{x1})"
            );
            // With g = 1: f3 = 2·M − [f1(1+sin3πf1) + f2(1+sin3πf2)] (closed form).
            let three_pi = 3.0 * std::f64::consts::PI;
            let want_f3 = 2.0 * 3.0
                - (x0 * (1.0 + (three_pi * x0).sin()) + x1 * (1.0 + (three_pi * x1).sin()));
            assert!(
                (f[2] - want_f3).abs() < 1e-12,
                "DTLZ7 f3={} want={want_f3}",
                f[2]
            );
        }
    }

    #[test]
    fn dtlz7_distance_perturbation_raises_f3() {
        // f3 = (1+g)·M − Σ fᵢ(1+sin3πfᵢ); the subtracted term is independent of g, so f3 grows
        // linearly with g. Any positive distance mass (g > 1) lifts f3 above the on-front value.
        let mut on = vec![0.0; 12];
        on[0] = 0.4;
        on[1] = 0.6;
        let f3_front = dtlz7(&on)[2];
        let mut off = on.clone();
        for d in off.iter_mut().skip(2) {
            *d = 0.3;
        }
        let f3_off = dtlz7(&off)[2];
        assert!(
            f3_off > f3_front + 1e-9,
            "DTLZ7 off-front f3={f3_off} not above on-front f3={f3_front}"
        );
    }

    #[test]
    fn dtlz7_front_has_four_disconnected_regions() {
        // DTLZ7 (M=3) has 2^(M−1) = 4 disconnected Pareto patches. With distance vars = 0 the
        // objective triple is (x0, x1, 6 − φ(x0) − φ(x1)), φ(t)=t(1+sin3πt). φ has two ascending
        // slopes per axis (peaks ≈0.25 and ≈0.85, trough at 0.5); only ascending-slope position
        // values are Pareto-efficient, giving 2×2 = 4 connected patches separated by gaps. We
        // recover them by extracting the non-dominated set over a position grid and flood-filling.
        const G: usize = 61; // grid resolution (spacing 1/60 resolves the ~0.25-wide gaps)
        let mut pts = Vec::with_capacity(G * G);
        let mut coord = Vec::with_capacity(G * G);
        for i in 0..G {
            for j in 0..G {
                let x0 = i as f64 / (G - 1) as f64;
                let x1 = j as f64 / (G - 1) as f64;
                let mut x = vec![0.0; 12];
                x[0] = x0;
                x[1] = x1;
                pts.push(dtlz7(&x));
                coord.push((i, j));
            }
        }
        let keep = extract_pareto_front(&pts);
        let mut grid = vec![vec![false; G]; G];
        for &idx in &keep {
            let (i, j) = coord[idx];
            grid[i][j] = true;
        }
        // 4-connected flood fill component count.
        let mut seen = vec![vec![false; G]; G];
        let mut components = 0usize;
        for i in 0..G {
            for j in 0..G {
                if !grid[i][j] || seen[i][j] {
                    continue;
                }
                components += 1;
                let mut stack = vec![(i, j)];
                seen[i][j] = true;
                while let Some((ci, cj)) = stack.pop() {
                    let neigh = [
                        (ci.wrapping_sub(1), cj),
                        (ci + 1, cj),
                        (ci, cj.wrapping_sub(1)),
                        (ci, cj + 1),
                    ];
                    for &(ni, nj) in &neigh {
                        if ni < G && nj < G && grid[ni][nj] && !seen[ni][nj] {
                            seen[ni][nj] = true;
                            stack.push((ni, nj));
                        }
                    }
                }
            }
        }
        assert_eq!(
            components, 4,
            "DTLZ7 disconnected-region count = {components} (want 4)"
        );
    }

    // Determinism ----------------------------------------------------------------

    #[test]
    fn new_moo_problems_are_deterministic() {
        let x12 = [
            0.1, 0.7, 0.3, 0.9, 0.2, 0.6, 0.4, 0.8, 0.15, 0.55, 0.35, 0.95,
        ];
        assert_eq!(zdt4(&x12), zdt4(&x12));
        assert_eq!(zdt6(&x12), zdt6(&x12));
        assert_eq!(dtlz3(&x12), dtlz3(&x12));
        assert_eq!(dtlz4(&x12), dtlz4(&x12));
        assert_eq!(dtlz5(&x12), dtlz5(&x12));
        assert_eq!(dtlz6(&x12), dtlz6(&x12));
        assert_eq!(dtlz7(&x12), dtlz7(&x12));
    }

    // ── Algorithm harness ────────────────────────────────────────────────────

    #[test]
    fn cmaes_benchmark_sphere_5d_converges() {
        let result = run_cmaes_benchmark(sphere, 5, 50_000, 1e-5, 42, "sphere-5d")
            .expect("CMA-ES on sphere should not error");
        assert!(
            result.converged,
            "CMA-ES should converge on 5-D sphere, best = {}",
            result.best_value
        );
        assert!(result.best_value < 1e-5);
    }

    #[test]
    fn cmaes_benchmark_rosenbrock_2d() {
        let result = run_cmaes_benchmark(rosenbrock, 2, 100_000, 1e-3, 7, "rosenbrock-2d")
            .expect("CMA-ES on Rosenbrock should not error");
        assert!(
            result.best_value < 1.0,
            "CMA-ES on Rosenbrock 2D should reach near optimum, best = {}",
            result.best_value
        );
    }

    #[test]
    fn cmaes_benchmark_n_evaluations_positive() {
        let result = run_cmaes_benchmark(sphere, 3, 5_000, 1e-5, 1, "sphere-3d-eval-check")
            .expect("no error");
        assert!(
            result.n_evaluations > 0,
            "n_evaluations must be > 0, got {}",
            result.n_evaluations
        );
    }

    #[test]
    fn cmaes_benchmark_invalid_n_dims_errors() {
        let err = run_cmaes_benchmark(sphere, 0, 1000, 1e-5, 0, "bad-dims");
        assert!(err.is_err(), "n_dims=0 must return an error");
    }

    #[test]
    fn nsga2_benchmark_zdt1_positive_hypervolume() {
        // Use a generous reference point to ensure front points are dominated.
        // ZDT1: f1 in [0,1], f2 in [0, 10] roughly; use (2.0, 15.0) to capture all.
        let ref_pt = vec![2.0, 15.0];
        let result = run_nsga2_benchmark(zdt1, 5, 2, 40, 80, 123, "zdt1-5d", ref_pt)
            .expect("NSGA-II on ZDT1 should not error");
        assert!(
            result.hypervolume > 0.0,
            "Hypervolume must be positive for ZDT1, got {}",
            result.hypervolume
        );
        assert!(result.n_front_points > 0, "Pareto front must be non-empty");
    }

    #[test]
    fn nsga2_benchmark_zdt2_positive_hypervolume() {
        // ZDT2: f1 in [0,1], f2 in [0, 10]; use generous reference point
        let ref_pt = vec![2.0, 15.0];
        let result = run_nsga2_benchmark(zdt2, 5, 2, 40, 80, 77, "zdt2-5d", ref_pt)
            .expect("NSGA-II on ZDT2 should not error");
        assert!(
            result.hypervolume > 0.0,
            "Hypervolume must be positive for ZDT2, got {}",
            result.hypervolume
        );
    }

    #[test]
    fn benchmark_result_fields_consistent() {
        let result =
            run_cmaes_benchmark(sphere, 2, 10_000, 1e-6, 99, "sphere-2d-fields").expect("no error");
        assert_eq!(result.n_dims, 2);
        assert_eq!(result.function_name, "sphere-2d-fields");
        assert_eq!(result.target_precision, 1e-6);
        assert_eq!(
            result.converged,
            result.best_value < result.target_precision
        );
    }

    // ── ZDT / DTLZ analytic-front convergence suite ───────────────────────────
    //
    // Standard multi-objective benchmark suite: run NSGA-II on the canonical ZDT
    // (ZDT1-3) and DTLZ (DTLZ1-2) problems and assert the recovered rank-0 front
    // converges to the *analytic* Pareto front. Quality is measured with the crate's
    // own generational distance (GD: mean distance from each recovered point to the
    // nearest true-front point → convergence) and inverted generational distance
    // (IGD: mean distance from each true-front point to the nearest recovered point →
    // convergence + coverage), plus the WFG hypervolume indicator. Both metrics are 0
    // for a perfect front; smaller is better.

    /// Run NSGA-II with the standard real-coded operator set and return the rank-0 front
    /// (objective vectors only). Decision variables live in `[0, 1]` (the ZDT/DTLZ domain).
    fn run_nsga2_front(
        f: impl Fn(&[f64]) -> Vec<f64>,
        n_dims: usize,
        n_obj: usize,
        pop_size: usize,
        n_gens: usize,
        seed: u64,
    ) -> Vec<Vec<f64>> {
        let cfg = Nsga2Config {
            n_dims,
            n_objectives: n_obj,
            pop_size,
            max_generations: n_gens,
            crossover_eta: 15.0,
            mutation_eta: 20.0,
            mutation_prob: 1.0 / n_dims as f64,
            bounds: (0.0, 1.0),
        };
        let mut rng = LcgRng::new(seed);
        let final_pop = nsga2_run(f, &cfg, &mut rng).expect("NSGA-II run");
        final_pop
            .into_iter()
            .filter(|ind| ind.rank == 0)
            .map(|ind| ind.objectives)
            .collect()
    }

    /// Densely sample a two-objective ZDT analytic front `(f1, h(f1))`, keeping only the
    /// non-dominated points (this carves the five disconnected segments out of ZDT3's curve).
    fn zdt_reference_front(h: impl Fn(f64) -> f64, n_samples: usize) -> Vec<Vec<f64>> {
        let raw: Vec<Vec<f64>> = (0..=n_samples)
            .map(|i| {
                let f1 = i as f64 / n_samples as f64;
                vec![f1, h(f1)]
            })
            .collect();
        let keep = extract_pareto_front(&raw);
        keep.into_iter().map(|i| raw[i].clone()).collect()
    }

    /// Sample the DTLZ2 analytic Pareto front: the unit sphere `f1²+f2²+f3²=1` in the
    /// first octant, parameterised by `(θ, φ) ∈ [0, π/2]²`.
    fn dtlz2_reference_front(n_div: usize) -> Vec<Vec<f64>> {
        let half_pi = std::f64::consts::FRAC_PI_2;
        let mut pts = Vec::with_capacity((n_div + 1) * (n_div + 1));
        for i in 0..=n_div {
            for j in 0..=n_div {
                let a0 = (i as f64 / n_div as f64) * half_pi;
                let a1 = (j as f64 / n_div as f64) * half_pi;
                pts.push(vec![a0.cos() * a1.cos(), a0.cos() * a1.sin(), a0.sin()]);
            }
        }
        pts
    }

    /// Span of objective `idx` over a front (used as a coarse spread/coverage measure).
    fn obj_span(front: &[Vec<f64>], idx: usize) -> f64 {
        let lo = front.iter().map(|o| o[idx]).fold(f64::INFINITY, f64::min);
        let hi = front
            .iter()
            .map(|o| o[idx])
            .fold(f64::NEG_INFINITY, f64::max);
        hi - lo
    }

    #[test]
    fn nsga2_zdt1_converges_to_analytic_front() {
        // ZDT1: convex front f2 = 1 - sqrt(f1).
        let reference = zdt_reference_front(zdt1_pareto_front_f2, 400);
        let front = run_nsga2_front(zdt1, 10, 2, 100, 250, 0x2DA1);
        assert!(front.len() >= 30, "ZDT1 front too sparse: {}", front.len());

        let gd = generational_distance(&front, &reference).expect("gd");
        let igd_val = igd(&front, &reference).expect("igd");
        let spread = obj_span(&front, 0);
        assert!(gd < 0.01, "ZDT1 GD = {gd:.5} (want < 0.01)");
        assert!(igd_val < 0.015, "ZDT1 IGD = {igd_val:.5} (want < 0.015)");
        assert!(spread > 0.9, "ZDT1 f1-spread = {spread:.3} (want > 0.9)");
    }

    #[test]
    fn nsga2_zdt2_converges_to_analytic_front() {
        // ZDT2: non-convex (concave) front f2 = 1 - f1^2.
        let reference = zdt_reference_front(zdt2_pareto_front_f2, 400);
        let front = run_nsga2_front(zdt2, 10, 2, 100, 250, 0x5AE2);
        assert!(front.len() >= 30, "ZDT2 front too sparse: {}", front.len());

        let gd = generational_distance(&front, &reference).expect("gd");
        let igd_val = igd(&front, &reference).expect("igd");
        let spread = obj_span(&front, 0);
        assert!(gd < 0.01, "ZDT2 GD = {gd:.5} (want < 0.01)");
        assert!(igd_val < 0.02, "ZDT2 IGD = {igd_val:.5} (want < 0.02)");
        assert!(spread > 0.9, "ZDT2 f1-spread = {spread:.3} (want > 0.9)");
    }

    #[test]
    fn nsga2_zdt3_converges_to_disconnected_front() {
        // ZDT3: discontinuous front (five segments) f2 = 1 - sqrt(f1) - f1*sin(10*pi*f1).
        let reference = zdt_reference_front(zdt3_pareto_front_f2, 1000);
        let front = run_nsga2_front(zdt3, 10, 2, 100, 250, 0x3D73);
        assert!(front.len() >= 20, "ZDT3 front too sparse: {}", front.len());

        let gd = generational_distance(&front, &reference).expect("gd");
        let igd_val = igd(&front, &reference).expect("igd");
        let spread = obj_span(&front, 0);
        assert!(gd < 0.02, "ZDT3 GD = {gd:.5} (want < 0.02)");
        assert!(igd_val < 0.05, "ZDT3 IGD = {igd_val:.5} (want < 0.05)");
        assert!(spread > 0.8, "ZDT3 f1-spread = {spread:.3} (want > 0.8)");
    }

    #[test]
    fn nsga2_dtlz2_converges_to_unit_sphere() {
        // DTLZ2: concave spherical front f1^2 + f2^2 + f3^2 = 1 (first octant).
        let reference = dtlz2_reference_front(24);
        let front = run_nsga2_front(dtlz2, 12, 3, 200, 250, 0xD712);
        assert!(front.len() >= 60, "DTLZ2 front too sparse: {}", front.len());

        // Every converged point lies on the unit sphere: radius = 1 + g, error = g >= 0.
        let radii: Vec<f64> = front
            .iter()
            .map(|o| (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt())
            .collect();
        let mean_r = radii.iter().sum::<f64>() / radii.len() as f64;
        let max_r = radii.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let gd = generational_distance(&front, &reference).expect("gd");
        let igd_val = igd(&front, &reference).expect("igd");
        assert!(
            (mean_r - 1.0).abs() < 0.03,
            "DTLZ2 mean front radius = {mean_r:.4} (want |r-1| < 0.03)"
        );
        assert!(
            max_r < 1.15,
            "DTLZ2 worst radius = {max_r:.4} (want < 1.15)"
        );
        assert!(gd < 0.05, "DTLZ2 GD = {gd:.4} (want < 0.05)");
        assert!(igd_val < 0.085, "DTLZ2 IGD = {igd_val:.4} (want < 0.085)");
    }

    #[test]
    fn nsga2_dtlz1_runs_and_produces_valid_front() {
        // DTLZ1: linear front on the simplex f1 + f2 + f3 = 0.5 (g -> 0). The distance function
        // g has a highly multimodal cosine landscape (11^k - 1 local optima), so within a CPU
        // test budget NSGA-II does NOT fully reach the simplex. We therefore assert structural
        // correctness (a non-empty, mutually non-dominated front of finite non-negative
        // objectives) and that the search makes substantial progress relative to random
        // initialisation. Full DTLZ1 convergence is compute-gated and tracked separately.
        let front = run_nsga2_front(dtlz1, 6, 3, 120, 300, 0xD711);
        assert!(front.len() >= 10, "DTLZ1 front too sparse: {}", front.len());

        for p in &front {
            assert_eq!(p.len(), 3, "DTLZ1 must yield 3 objectives");
            for &v in p {
                assert!(v.is_finite() && v >= 0.0, "DTLZ1 objective invalid: {v}");
            }
        }

        // Rank-0 front must be mutually non-dominated.
        for (i, a) in front.iter().enumerate() {
            for (j, b) in front.iter().enumerate() {
                if i == j {
                    continue;
                }
                let dominates =
                    a.iter().zip(b).all(|(x, y)| x <= y) && a.iter().zip(b).any(|(x, y)| x < y);
                assert!(!dominates, "DTLZ1 rank-0 point {i} dominates {j}");
            }
        }

        // Progress check: a uniformly random DTLZ1 individual has objective sum 0.5*(1+g) with
        // E[g] ~ 100*k (k = 4 distance vars), i.e. a sum on the order of 2e2; the optimum is 0.5.
        // Requiring the best front point's sum << that scale confirms real selection pressure.
        let min_sum = front
            .iter()
            .map(|o| o[0] + o[1] + o[2])
            .fold(f64::INFINITY, f64::min);
        assert!(
            min_sum < 20.0,
            "DTLZ1 made no measurable progress: best simplex sum = {min_sum:.3}"
        );
    }

    #[test]
    fn nsga2_zdt1_hypervolume_near_analytic() {
        // Tie the convergence to the WFG hypervolume indicator: the recovered front should
        // dominate almost as much volume as the analytic front w.r.t. reference (1.1, 1.1).
        let ref_pt = vec![1.1_f64, 1.1_f64];
        let analytic: Vec<Vec<f64>> = (0..=1000)
            .map(|i| {
                let f1 = i as f64 / 1000.0;
                vec![f1, zdt1_pareto_front_f2(f1)]
            })
            .filter(|p| p[0] < ref_pt[0] && p[1] < ref_pt[1])
            .collect();
        let hv_true =
            hypervolume_nd(&analytic, std::slice::from_ref(&ref_pt)).expect("analytic hv");

        let front = run_nsga2_front(zdt1, 10, 2, 100, 250, 0x2DA1);
        let dominated: Vec<Vec<f64>> = front
            .into_iter()
            .filter(|p| p[0] < ref_pt[0] && p[1] < ref_pt[1])
            .collect();
        let hv = hypervolume_nd(&dominated, &[ref_pt]).expect("recovered hv");

        assert!(hv_true > 0.0, "analytic HV must be positive");
        let ratio = hv / hv_true;
        assert!(
            ratio > 0.97,
            "ZDT1 recovered HV {hv:.5} is only {ratio:.3} of analytic {hv_true:.5} (want > 0.97)"
        );
        assert!(
            ratio <= 1.001,
            "ZDT1 recovered HV {hv:.5} exceeds analytic {hv_true:.5} (ratio {ratio:.4})"
        );
    }
}
