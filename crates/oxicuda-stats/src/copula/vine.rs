//! Vine copulas (pair-copula constructions) — C-vines and D-vines.
//!
//! A *regular vine* (R-vine) factorises a `d`-variate copula density into a
//! cascade of `d(d−1)/2` bivariate (conditional) **pair-copulas** arranged on a
//! nested sequence of trees. This module implements the two canonical special
//! cases, the **C-vine** (canonical vine, one root variable per tree) and the
//! **D-vine** (drawable vine, a path through the variables), following the
//! pair-copula construction (PCC) of Aas et al. (2009) built on the regular-vine
//! framework of Bedford & Cooke (2001, 2002).
//!
//! # Pair-copula construction
//!
//! Any joint density `f(x₁,…,x_d)` of continuous variables with uniform marginals
//! (a copula density `c(u₁,…,u_d)`) can be written as a product of bivariate
//! conditional copula densities. For a D-vine on variables `1,…,d`:
//!
//! ```text
//! c(u₁,…,u_d) = ∏_{j=1}^{d-1} ∏_{i=1}^{d-j}
//!               c_{i, i+j | i+1,…,i+j-1}( F(u_i | ·) , F(u_{i+j} | ·) )
//! ```
//!
//! and for a C-vine rooted at `1,2,…`:
//!
//! ```text
//! c(u₁,…,u_d) = ∏_{j=1}^{d-1} ∏_{i=1}^{d-j}
//!               c_{j, j+i | 1,…,j-1}( F(u_j|·) , F(u_{j+i}|·) ).
//! ```
//!
//! The conditional CDFs `F(u|·)` are obtained recursively through the **h-function**
//! of each pair-copula, `h(u | v; θ) = ∂C(u, v; θ) / ∂v`, which is exactly the
//! conditional distribution of `U` given `V = v`.
//!
//! # Sequential estimation
//!
//! Parameters are estimated tree-by-tree (Aas et al. 2009, §6):
//! 1. Fit the tree-1 pair-copulas directly on the pseudo-observations.
//! 2. Apply each fitted pair-copula's h-function to obtain the tree-2 arguments
//!    (conditional CDFs).
//! 3. Fit the tree-2 pair-copulas on the transformed data; repeat to tree `d−1`.
//!
//! # References
//! - Aas, K., Czado, C., Frigessi, A. & Bakken, H. (2009). *Pair-copula
//!   constructions of multiple dependence.* Insurance Math. Econ. 44(2):182-198.
//! - Bedford, T. & Cooke, R.M. (2002). *Vines — a new graphical model for
//!   dependent random variables.* Ann. Statist. 30(4):1031-1068.
//! - Bedford, T. & Cooke, R.M. (2001). *Probability density decomposition for
//!   conditionally dependent random variables modeled by vines.* Ann. Math.
//!   Artif. Intell. 32:245-268.
//! - Czado, C. (2019). *Analyzing Dependent Data with Vine Copulas.* Springer.

use crate::copula::copulas::{CopulaFamily, CopulaFit, copula_fit};
use crate::error::{StatsError, StatsResult};
use std::f64::consts::{PI, SQRT_2};

// ---------------------------------------------------------------------------
// Local standard-normal helpers (Φ, Φ⁻¹) — used by the Gaussian h-function.
// Kept module-private so the vine code is self-contained and does not rely on
// the (private) helpers inside `copulas.rs`.
// ---------------------------------------------------------------------------

/// Abramowitz-Stegun 7.1.26 erf approximation (max abs error ~1.5e-7).
fn erf_approx(x: f64) -> f64 {
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    const P: f64 = 0.327_591_1;
    if x == 0.0 {
        return 0.0;
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + P * ax);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-ax * ax).exp();
    sign * y
}

/// Standard normal CDF Φ(x).
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / SQRT_2))
}

/// Standard normal PDF φ(x).
fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Peter Acklam's rational approximation seed for Φ⁻¹(p).
fn acklam_seed(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_374_269e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Standard normal quantile Φ⁻¹(p) (Acklam seed + a few Newton steps).
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let mut x = acklam_seed(p);
    for _ in 0..50 {
        let fx = normal_cdf(x) - p;
        let fpx = normal_pdf(x);
        if fpx.abs() < 1e-300 {
            break;
        }
        let dx = fx / fpx;
        x -= dx;
        if dx.abs() < 1e-13 {
            break;
        }
    }
    x
}

// ---------------------------------------------------------------------------
// Vine structure
// ---------------------------------------------------------------------------

/// Vine-structure flavour controlling the tree topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VineType {
    /// Canonical vine: each tree has one root connected to every other node.
    /// Suited to a setting where one variable governs the dependence.
    CVine,
    /// Drawable vine: every tree is a path. Suited to ordered / serial data.
    DVine,
}

/// A single fitted pair-copula sitting at a position in a vine tree.
#[derive(Debug, Clone)]
pub struct PairCopula {
    /// Underlying bivariate family + parameter (reuses [`CopulaFit`]).
    pub fit: CopulaFit,
}

impl PairCopula {
    /// Construct from a family and parameter without re-fitting (e.g. to build a
    /// *known* vine for simulation/testing).
    ///
    /// # Errors
    /// Returns [`StatsError::InvalidParameter`] if `theta` is out of range for
    /// the chosen family (Clayton `θ>0`, Gumbel `θ≥1`, Gaussian `|ρ|<1`).
    pub fn from_param(family: CopulaFamily, theta: f64) -> StatsResult<Self> {
        validate_theta(family, theta)?;
        Ok(Self {
            fit: CopulaFit {
                family,
                theta,
                kendall_tau: f64::NAN,
                n_samples: 0,
            },
        })
    }

    /// The independence pair-copula `C(u,v)=uv` (`c≡1`, `h(u|v)=u`).
    #[must_use]
    pub fn independence() -> Self {
        Self {
            fit: CopulaFit {
                family: CopulaFamily::Frank,
                theta: 0.0,
                kendall_tau: 0.0,
                n_samples: 0,
            },
        }
    }

    /// Whether this pair-copula is (numerically) the independence copula.
    #[must_use]
    pub fn is_independence(&self) -> bool {
        matches!(self.fit.family, CopulaFamily::Frank) && self.fit.theta.abs() < 1e-10
    }

    /// Bivariate density `c(u, v)` of this pair-copula on the strict interior.
    ///
    /// # Errors
    /// Returns [`StatsError::InvalidParameter`] if `u` or `v` ∉ (0, 1).
    pub fn density(&self, u: f64, v: f64) -> StatsResult<f64> {
        pair_density(self.fit.family, self.fit.theta, u, v)
    }

    /// h-function `h(u | v; θ) = ∂C(u, v; θ)/∂v` — the conditional CDF of `U`
    /// given `V = v`. The returned value lies in `[0, 1]`.
    ///
    /// # Errors
    /// Returns [`StatsError::InvalidParameter`] if `u` or `v` ∉ (0, 1).
    pub fn h_func(&self, u: f64, v: f64) -> StatsResult<f64> {
        h_function(self.fit.family, self.fit.theta, u, v)
    }
}

/// A fitted vine copula: the structure plus the lower-triangular array of
/// pair-copulas. `trees[t]` holds the pair-copulas of tree `t` (0-indexed);
/// tree `t` has `d − 1 − t` pair-copulas.
#[derive(Debug, Clone)]
pub struct VineCopula {
    /// Dimension `d` (number of variables).
    pub dim: usize,
    /// C-vine or D-vine topology.
    pub vine_type: VineType,
    /// `trees[t][i]` = pair-copula at edge `i` of tree `t`.
    pub trees: Vec<Vec<PairCopula>>,
}

// ---------------------------------------------------------------------------
// Parameter validation and per-family h-functions / densities
// ---------------------------------------------------------------------------

fn validate_theta(family: CopulaFamily, theta: f64) -> StatsResult<()> {
    if !theta.is_finite() {
        return Err(StatsError::InvalidParameter {
            name: "theta".to_owned(),
            reason: format!("must be finite, got {theta}"),
        });
    }
    match family {
        CopulaFamily::Gaussian => {
            if !(theta > -1.0 && theta < 1.0) {
                return Err(StatsError::InvalidParameter {
                    name: "theta".to_owned(),
                    reason: format!("Gaussian ρ must be in (-1, 1), got {theta}"),
                });
            }
        }
        CopulaFamily::Clayton => {
            if theta <= 0.0 {
                return Err(StatsError::InvalidParameter {
                    name: "theta".to_owned(),
                    reason: format!("Clayton θ must be > 0, got {theta}"),
                });
            }
        }
        CopulaFamily::Gumbel => {
            if theta < 1.0 {
                return Err(StatsError::InvalidParameter {
                    name: "theta".to_owned(),
                    reason: format!("Gumbel θ must be ≥ 1, got {theta}"),
                });
            }
        }
        CopulaFamily::Frank => { /* any finite θ (θ=0 ⇒ independence) */ }
    }
    Ok(())
}

fn check_interior(u: f64, v: f64) -> StatsResult<()> {
    if !(u > 0.0 && u < 1.0) {
        return Err(StatsError::InvalidParameter {
            name: "u".to_owned(),
            reason: format!("must be in (0, 1), got {u}"),
        });
    }
    if !(v > 0.0 && v < 1.0) {
        return Err(StatsError::InvalidParameter {
            name: "v".to_owned(),
            reason: format!("must be in (0, 1), got {v}"),
        });
    }
    Ok(())
}

/// h-function `h(u|v;θ) = ∂C(u,v;θ)/∂v`, the conditional CDF `P(U≤u | V=v)`.
///
/// Closed forms (Aas et al. 2009, Appendix):
/// - **Gaussian**: `h = Φ( (Φ⁻¹(u) − ρ Φ⁻¹(v)) / √(1−ρ²) )`.
/// - **Clayton**: `h = v^{−θ−1} (u^{−θ} + v^{−θ} − 1)^{−1−1/θ}`.
/// - **Frank**: `h = (e^{−θv}(e^{−θu}−1)) / ((e^{−θ}−1)+(e^{−θu}−1)(e^{−θv}−1))`.
/// - **Gumbel**: `h = C(u,v) · (−ln v)^{θ−1} · A^{1/θ−1} / v`,
///   with `A=(−ln u)^θ+(−ln v)^θ`.
fn h_function(family: CopulaFamily, theta: f64, u: f64, v: f64) -> StatsResult<f64> {
    check_interior(u, v)?;
    let h = match family {
        CopulaFamily::Frank if theta.abs() < 1e-12 => u, // independence
        CopulaFamily::Gaussian => {
            let qu = normal_quantile(u);
            let qv = normal_quantile(v);
            let denom = (1.0 - theta * theta).max(1e-15).sqrt();
            normal_cdf((qu - theta * qv) / denom)
        }
        CopulaFamily::Clayton => {
            let inner = u.powf(-theta) + v.powf(-theta) - 1.0;
            if inner <= 0.0 {
                0.0
            } else {
                v.powf(-theta - 1.0) * inner.powf(-1.0 - 1.0 / theta)
            }
        }
        CopulaFamily::Frank => {
            let em1 = (-theta).exp() - 1.0;
            let eu = (-theta * u).exp() - 1.0;
            let ev = (-theta * v).exp() - 1.0;
            let denom = em1 + eu * ev;
            if denom.abs() < 1e-300 {
                u
            } else {
                let num = (-theta * v).exp() * eu;
                num / denom
            }
        }
        CopulaFamily::Gumbel => {
            let neg_ln_u = -u.ln();
            let neg_ln_v = -v.ln();
            let a_u = neg_ln_u.powf(theta);
            let a_v = neg_ln_v.powf(theta);
            let big_a = a_u + a_v;
            if big_a < 1e-300 {
                0.0
            } else {
                let s = big_a.powf(1.0 / theta);
                let c_val = (-s).exp();
                c_val * neg_ln_v.powf(theta - 1.0) * big_a.powf(1.0 / theta - 1.0) / v
            }
        }
    };
    Ok(h.clamp(0.0, 1.0))
}

/// Bivariate copula density `c(u, v; θ)` for the supported families.
fn pair_density(family: CopulaFamily, theta: f64, u: f64, v: f64) -> StatsResult<f64> {
    check_interior(u, v)?;
    let d = match family {
        CopulaFamily::Frank if theta.abs() < 1e-12 => 1.0, // independence
        CopulaFamily::Gaussian => {
            let x = normal_quantile(u);
            let y = normal_quantile(v);
            let rho2 = theta * theta;
            let one_minus_rho2 = (1.0 - rho2).max(1e-15);
            let exponent = (2.0 * theta * x * y - rho2 * (x * x + y * y)) / (2.0 * one_minus_rho2);
            exponent.exp() / one_minus_rho2.sqrt()
        }
        CopulaFamily::Clayton => {
            let inner = u.powf(-theta) + v.powf(-theta) - 1.0;
            if inner <= 0.0 {
                return Ok(0.0);
            }
            (theta + 1.0) * (u * v).powf(-(theta + 1.0)) * inner.powf(-(2.0 + 1.0 / theta))
        }
        CopulaFamily::Frank => {
            let em1 = (-theta).exp() - 1.0;
            if em1.abs() < 1e-14 {
                return Ok(1.0);
            }
            let eu = (-theta * u).exp() - 1.0;
            let ev = (-theta * v).exp() - 1.0;
            let denom = em1 + eu * ev;
            if denom.abs() < 1e-300 {
                return Ok(0.0);
            }
            let num = -theta * em1 * ((-theta * (u + v)).exp());
            (num / (denom * denom)).abs()
        }
        CopulaFamily::Gumbel => {
            // Standard Gumbel density (Joe 1997; Nelsen 2006):
            // c = C · (1/(u v)) · ((−ln u)(−ln v))^{θ−1} · A^{1/θ−2} · (A^{1/θ} + θ − 1),
            // with A = (−ln u)^θ + (−ln v)^θ and s = A^{1/θ}. This is the exact
            // mixed partial ∂²C/∂u∂v consistent with the h-function below.
            let neg_ln_u = -u.ln();
            let neg_ln_v = -v.ln();
            let a_u = neg_ln_u.powf(theta);
            let a_v = neg_ln_v.powf(theta);
            let big_a = a_u + a_v;
            if big_a < 1e-300 {
                return Ok(0.0);
            }
            let s = big_a.powf(1.0 / theta);
            let c_val = (-s).exp();
            let factor1 = 1.0 / (u * v);
            let factor2 = big_a.powf(1.0 / theta - 2.0);
            let factor3 = (neg_ln_u * neg_ln_v).powf(theta - 1.0);
            let factor4 = s + theta - 1.0;
            c_val * factor1 * factor2 * factor3 * factor4
        }
    };
    Ok(d.max(0.0))
}

// ---------------------------------------------------------------------------
// Construction of a *known* vine (for simulation / density evaluation)
// ---------------------------------------------------------------------------

impl VineCopula {
    /// Build a vine from a triangular list of pair-copulas you already know.
    ///
    /// `trees[t]` must have exactly `d − 1 − t` entries (tree `t` is 0-indexed).
    ///
    /// # Errors
    /// Returns [`StatsError::InvalidParameter`] if `dim < 2` or any tree has the
    /// wrong number of pair-copulas.
    pub fn from_pair_copulas(
        dim: usize,
        vine_type: VineType,
        trees: Vec<Vec<PairCopula>>,
    ) -> StatsResult<Self> {
        if dim < 2 {
            return Err(StatsError::InsufficientSampleSize { got: dim, need: 2 });
        }
        if trees.len() != dim - 1 {
            return Err(StatsError::InvalidParameter {
                name: "trees".to_owned(),
                reason: format!("expected {} trees, got {}", dim - 1, trees.len()),
            });
        }
        for (t, tree) in trees.iter().enumerate() {
            let expect = dim - 1 - t;
            if tree.len() != expect {
                return Err(StatsError::InvalidParameter {
                    name: format!("trees[{t}]"),
                    reason: format!("expected {expect} pair-copulas, got {}", tree.len()),
                });
            }
        }
        Ok(Self {
            dim,
            vine_type,
            trees,
        })
    }

    /// Build a `d`-dimensional vine where every pair-copula is the independence
    /// copula (joint density ≡ 1 for uniform marginals).
    ///
    /// # Errors
    /// Returns [`StatsError::InsufficientSampleSize`] if `dim < 2`.
    pub fn independence(dim: usize, vine_type: VineType) -> StatsResult<Self> {
        if dim < 2 {
            return Err(StatsError::InsufficientSampleSize { got: dim, need: 2 });
        }
        let trees: Vec<Vec<PairCopula>> = (0..dim - 1)
            .map(|t| {
                (0..dim - 1 - t)
                    .map(|_| PairCopula::independence())
                    .collect()
            })
            .collect();
        Ok(Self {
            dim,
            vine_type,
            trees,
        })
    }

    /// Total number of pair-copulas `d(d−1)/2`.
    #[must_use]
    pub fn n_pairs(&self) -> usize {
        self.dim * (self.dim - 1) / 2
    }

    /// Evaluate the vine copula density `c(u₁,…,u_d)` at one point.
    ///
    /// Walks the trees bottom-up, applying h-functions to produce each
    /// successive tree's conditional arguments and accumulating the product of
    /// pair-copula densities.
    ///
    /// # Errors
    /// - [`StatsError::DimensionMismatch`] if `u.len() != dim`.
    /// - [`StatsError::InvalidParameter`] if any `u[i] ∉ (0, 1)`.
    pub fn density(&self, u: &[f64]) -> StatsResult<f64> {
        if u.len() != self.dim {
            return Err(StatsError::DimensionMismatch {
                a: u.len(),
                b: self.dim,
            });
        }
        for (i, &ui) in u.iter().enumerate() {
            if !(ui > 0.0 && ui < 1.0) {
                return Err(StatsError::InvalidParameter {
                    name: format!("u[{i}]"),
                    reason: format!("must be in (0, 1), got {ui}"),
                });
            }
        }
        match self.vine_type {
            VineType::CVine => self.density_cvine(u),
            VineType::DVine => self.density_dvine(u),
        }
    }

    /// Log-density `ln c(u₁,…,u_d)`; `−∞` (≈`f64::MIN`) if the density underflows.
    ///
    /// # Errors
    /// Same as [`VineCopula::density`].
    pub fn log_density(&self, u: &[f64]) -> StatsResult<f64> {
        let d = self.density(u)?;
        Ok(if d <= 0.0 { -1e300 } else { d.ln() })
    }

    /// Total log-likelihood over a stacked observation matrix.
    ///
    /// `data` is row-major: observation `r` occupies `data[r*dim .. r*dim+dim]`.
    ///
    /// # Errors
    /// - [`StatsError::DimensionMismatch`] if `data.len() != n_obs * dim`.
    /// - propagates errors from [`VineCopula::density`].
    pub fn log_likelihood(&self, data: &[f64], n_obs: usize) -> StatsResult<f64> {
        if data.len() != n_obs * self.dim {
            return Err(StatsError::DimensionMismatch {
                a: data.len(),
                b: n_obs * self.dim,
            });
        }
        let mut ll = 0.0;
        for r in 0..n_obs {
            let row = &data[r * self.dim..(r + 1) * self.dim];
            ll += self.log_density(row)?;
        }
        Ok(ll)
    }

    // -- C-vine density -----------------------------------------------------

    /// C-vine density (Aas et al. 2009, Algorithm 1).
    ///
    /// Maintains a rolling array `v` whose element 0 is the *recursively
    /// transformed* root `F(u_{t} | u_0,…,u_{t-1})` of the current tree and whose
    /// remaining elements are the partners' conditional CDFs. At tree `t`, edge
    /// `i` contributes the factor `c( v[0], v[i+1] )` and the next array is
    /// `v'[i] = h( v[i+1] | v[0] )`.
    fn density_cvine(&self, u: &[f64]) -> StatsResult<f64> {
        let d = self.dim;
        let mut dens = 1.0_f64;
        // v holds the conditional arguments relevant to the current tree; at the
        // start of tree t it has length d - t with v[0] the (transformed) root.
        let mut v: Vec<f64> = u.iter().map(|&x| clamp01(x)).collect();
        for t in 0..d - 1 {
            let root = v[0];
            let m = d - 1 - t; // partners / edges in this tree
            let mut next: Vec<f64> = Vec::with_capacity(m);
            for i in 0..m {
                let pc = &self.trees[t][i];
                let partner = v[i + 1];
                dens *= pc.density(root, partner)?;
                // Conditional CDF of the partner given the root → next root/partners.
                next.push(clamp01(pc.h_func(partner, root)?));
            }
            v = next;
        }
        Ok(dens)
    }

    // -- D-vine density -----------------------------------------------------

    /// D-vine density (Aas et al. 2009, Algorithm 2).
    ///
    /// Uses a per-tree array `v` of conditional CDFs. For tree `j ≥ 2` edge `i`
    /// reads the pair `(v_prev[2i-1], v_prev[2i])` (1-indexed); the recurrence
    /// builds the next tree's `v` from forward/backward h-functions of the
    /// current edges. Tree indices here are 0-based (`t = j − 1`).
    fn density_dvine(&self, u: &[f64]) -> StatsResult<f64> {
        let d = self.dim;
        let mut dens = 1.0_f64;
        let clamped: Vec<f64> = u.iter().map(|&x| clamp01(x)).collect();

        // We store each tree's `v` as a 1-indexed vector (slot 0 unused) to match
        // the published indexing exactly.
        // Tree 1 (t = 0): densities on consecutive raw variables.
        for i in 0..d - 1 {
            let pc = &self.trees[0][i];
            dens *= pc.density(clamped[i], clamped[i + 1])?;
        }
        if d == 2 {
            return Ok(dens);
        }

        // Build v for tree 1 (length 2(d-1)-1 + slot0). v is 1-indexed.
        let vlen = 2 * (d - 1) - 1; // number of meaningful entries in tree-1 v
        let mut v = vec![0.0_f64; vlen + 1];
        // v[1] = h(x_1 | x_2)  (1-based vars; clamped[0]=x_1, clamped[1]=x_2)
        v[1] = clamp01(self.trees[0][0].h_func(clamped[0], clamped[1])?);
        // for k = 1..=d-3: v[2k] = h(x_{k+2} | x_{k+1}); v[2k+1] = h(x_{k+1} | x_{k+2})
        if d >= 4 {
            for k in 1..=d - 3 {
                let pc = &self.trees[0][k]; // edge (k+1,k+2) 1-based → 0-based index k
                // x_{k+1}=clamped[k], x_{k+2}=clamped[k+1]
                v[2 * k] = clamp01(pc.h_func(clamped[k + 1], clamped[k])?); // h(x_{k+2}|x_{k+1})
                v[2 * k + 1] = clamp01(pc.h_func(clamped[k], clamped[k + 1])?); // h(x_{k+1}|x_{k+2})
            }
        }
        // v[2(d-1)-2] = h(x_d | x_{d-1})
        {
            let pc = &self.trees[0][d - 2]; // last edge (d-1,d)
            v[2 * (d - 1) - 2] = clamp01(pc.h_func(clamped[d - 1], clamped[d - 2])?);
        }

        // Trees j = 2..=d-1  (0-based t = j-1 = 1..=d-2).
        for j in 2..=d - 1 {
            let t = j - 1;
            // densities for this tree
            for i in 1..=d - j {
                let pc = &self.trees[t][i - 1];
                dens *= pc.density(v[2 * i - 1], v[2 * i])?;
            }
            if j == d - 1 {
                break;
            }
            // Build next v.
            let next_len = 2 * (d - j) - 1;
            let mut vn = vec![0.0_f64; next_len + 1];
            // vn[1] = h(v[1] | v[2]) via edge i=1 of this tree
            {
                let pc = &self.trees[t][0];
                vn[1] = clamp01(pc.h_func(v[1], v[2])?);
            }
            if d - j - 1 > 1 {
                for i in 1..=d - j - 1 {
                    let pc = &self.trees[t][i]; // edge i+1 (1-based) of tree j
                    vn[2 * i] = clamp01(pc.h_func(v[2 * i + 2], v[2 * i + 1])?);
                    vn[2 * i + 1] = clamp01(pc.h_func(v[2 * i + 1], v[2 * i + 2])?);
                }
            }
            // vn[2(d-j)-2] = h(v[2(d-j)] | v[2(d-j)-1]) via the last edge of tree j
            {
                let last_edge = d - j - 1; // 0-based index of edge (d-j) in tree j
                let pc = &self.trees[t][last_edge];
                vn[2 * (d - j) - 2] = clamp01(pc.h_func(v[2 * (d - j)], v[2 * (d - j) - 1])?);
            }
            v = vn;
        }
        Ok(dens)
    }
}

// ---------------------------------------------------------------------------
// Sequential estimation
// ---------------------------------------------------------------------------

/// Configuration for sequential (tree-by-tree) vine estimation.
#[derive(Debug, Clone)]
pub struct VineFitConfig {
    /// Vine topology to fit.
    pub vine_type: VineType,
    /// Pair-copula family used at every edge.
    pub family: CopulaFamily,
}

impl Default for VineFitConfig {
    fn default() -> Self {
        Self {
            vine_type: VineType::CVine,
            family: CopulaFamily::Frank,
        }
    }
}

fn clamp01(x: f64) -> f64 {
    x.clamp(1e-12, 1.0 - 1e-12)
}

/// Sequentially estimate a vine copula from copula-scale data (pseudo-obs in
/// `(0,1)`), one tree at a time (Aas et al. 2009, §6).
///
/// `data` is row-major: observation `r` is `data[r*dim .. r*dim+dim]`.
///
/// The same `family` is fitted at every edge; the h-functions of the fitted
/// pair-copulas transform the data for the next tree.
///
/// # Errors
/// - [`StatsError::InsufficientSampleSize`] if `dim < 2` or `n_obs < 2`.
/// - [`StatsError::DimensionMismatch`] if `data.len() != n_obs * dim`.
/// - [`StatsError::InvalidParameter`] if any datum ∉ (0, 1).
/// - propagates the underlying [`copula_fit`] errors.
pub fn vine_fit(
    data: &[f64],
    n_obs: usize,
    dim: usize,
    config: &VineFitConfig,
) -> StatsResult<VineCopula> {
    if dim < 2 {
        return Err(StatsError::InsufficientSampleSize { got: dim, need: 2 });
    }
    if n_obs < 2 {
        return Err(StatsError::InsufficientSampleSize {
            got: n_obs,
            need: 2,
        });
    }
    if data.len() != n_obs * dim {
        return Err(StatsError::DimensionMismatch {
            a: data.len(),
            b: n_obs * dim,
        });
    }
    for (i, &x) in data.iter().enumerate() {
        if !(x > 0.0 && x < 1.0) {
            return Err(StatsError::InvalidParameter {
                name: format!("data[{i}]"),
                reason: format!("must be in (0, 1), got {x}"),
            });
        }
    }
    match config.vine_type {
        VineType::CVine => fit_cvine(data, n_obs, dim, config.family),
        VineType::DVine => fit_dvine(data, n_obs, dim, config.family),
    }
}

/// Fit one pair-copula of the chosen family on `(u, v)` columns, falling back to
/// the independence copula if the family is undefined for the sample's τ.
fn fit_pair(u: &[f64], v: &[f64], n: usize, family: CopulaFamily) -> PairCopula {
    match copula_fit(u, v, n, family) {
        Ok(fit) => PairCopula { fit },
        Err(_) => PairCopula::independence(),
    }
}

/// Sequential C-vine estimation.
fn fit_cvine(
    data: &[f64],
    n_obs: usize,
    dim: usize,
    family: CopulaFamily,
) -> StatsResult<VineCopula> {
    // `cols[j]` is the current (transformed) column for variable index j.
    let mut cols: Vec<Vec<f64>> = (0..dim)
        .map(|j| (0..n_obs).map(|r| clamp01(data[r * dim + j])).collect())
        .collect();
    let mut trees: Vec<Vec<PairCopula>> = Vec::with_capacity(dim - 1);

    for t in 0..dim - 1 {
        let root = cols[t].clone();
        let mut tree: Vec<PairCopula> = Vec::with_capacity(dim - 1 - t);
        let mut new_cols: Vec<Vec<f64>> = Vec::with_capacity(dim - 1 - t);
        for i in 0..dim - 1 - t {
            let partner = &cols[t + 1 + i];
            let pc = fit_pair(partner, &root, n_obs, family);
            // Transform partner → h(partner | root) for the next tree.
            let mut transformed = Vec::with_capacity(n_obs);
            for r in 0..n_obs {
                transformed.push(clamp01(pc.h_func(partner[r], root[r])?));
            }
            tree.push(pc);
            new_cols.push(transformed);
        }
        // Overwrite the (t+1..) columns with their transforms.
        for (i, col) in new_cols.into_iter().enumerate() {
            cols[t + 1 + i] = col;
        }
        trees.push(tree);
    }

    Ok(VineCopula {
        dim,
        vine_type: VineType::CVine,
        trees,
    })
}

/// Sequential D-vine estimation (Aas et al. 2009, Algorithm 4).
fn fit_dvine(
    data: &[f64],
    n_obs: usize,
    dim: usize,
    family: CopulaFamily,
) -> StatsResult<VineCopula> {
    // Raw columns.
    let cols: Vec<Vec<f64>> = (0..dim)
        .map(|j| (0..n_obs).map(|r| clamp01(data[r * dim + j])).collect())
        .collect();
    let mut trees: Vec<Vec<PairCopula>> = Vec::with_capacity(dim - 1);

    // Tree 0.
    // a[i] = h(col_i | col_{i+1}), b[i] = h(col_{i+1} | col_i).
    let mut a: Vec<Vec<f64>> = Vec::with_capacity(dim - 1);
    let mut b: Vec<Vec<f64>> = Vec::with_capacity(dim - 1);
    let mut tree0: Vec<PairCopula> = Vec::with_capacity(dim - 1);
    for i in 0..dim - 1 {
        let left = &cols[i];
        let right = &cols[i + 1];
        let pc = fit_pair(left, right, n_obs, family);
        let mut ai = Vec::with_capacity(n_obs);
        let mut bi = Vec::with_capacity(n_obs);
        for r in 0..n_obs {
            ai.push(clamp01(pc.h_func(left[r], right[r])?));
            bi.push(clamp01(pc.h_func(right[r], left[r])?));
        }
        tree0.push(pc);
        a.push(ai);
        b.push(bi);
    }
    trees.push(tree0);

    // Trees 1..dim-1.
    for t in 1..dim - 1 {
        let m = dim - 1 - t;
        let mut tree: Vec<PairCopula> = Vec::with_capacity(m);
        let mut new_a: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut new_b: Vec<Vec<f64>> = Vec::with_capacity(m);
        for i in 0..m {
            // Left = b_prev[i] (forward), Right = a_prev[i+1] (backward).
            let x = &b[i];
            let y = &a[i + 1];
            let pc = fit_pair(x, y, n_obs, family);
            let mut ai = Vec::with_capacity(n_obs);
            let mut bi = Vec::with_capacity(n_obs);
            for r in 0..n_obs {
                ai.push(clamp01(pc.h_func(x[r], y[r])?));
                bi.push(clamp01(pc.h_func(y[r], x[r])?));
            }
            tree.push(pc);
            new_a.push(ai);
            new_b.push(bi);
        }
        a = new_a;
        b = new_b;
        trees.push(tree);
    }

    Ok(VineCopula {
        dim,
        vine_type: VineType::DVine,
        trees,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    const EPS: f64 = 1e-9;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ---- (a) independence vine → density ≡ 1 -----------------------------
    #[test]
    fn independence_vine_density_one_d3() {
        let vine =
            VineCopula::independence(3, VineType::CVine).expect("independence should succeed");
        for &p in &[
            [0.2, 0.5, 0.8],
            [0.1, 0.9, 0.3],
            [0.5, 0.5, 0.5],
            [0.05, 0.95, 0.45],
        ] {
            let d = vine.density(&p).expect("density should succeed");
            assert!(approx(d, 1.0, 1e-12), "indep density {d} ≠ 1 at {p:?}");
        }
    }

    #[test]
    fn independence_vine_density_one_dvine_d4() {
        let vine =
            VineCopula::independence(4, VineType::DVine).expect("independence should succeed");
        let d = vine
            .density(&[0.2, 0.4, 0.6, 0.8])
            .expect("density should succeed");
        assert!(approx(d, 1.0, 1e-12), "D-vine indep density {d} ≠ 1");
    }

    // ---- (b) d=3 vine density integrates to ≈1 over [0,1]³ ----------------
    #[test]
    fn cvine_density_integrates_to_one() {
        // A non-trivial C-vine: tree-0 Clayton(θ=2) & Frank(θ=3), tree-1 Gumbel(θ=1.5).
        let trees = vec![
            vec![
                PairCopula::from_param(CopulaFamily::Clayton, 2.0)
                    .expect("from_param should succeed"),
                PairCopula::from_param(CopulaFamily::Frank, 3.0)
                    .expect("from_param should succeed"),
            ],
            vec![
                PairCopula::from_param(CopulaFamily::Gumbel, 1.5)
                    .expect("from_param should succeed"),
            ],
        ];
        let vine = VineCopula::from_pair_copulas(3, VineType::CVine, trees)
            .expect("from_pair_copulas should succeed");
        // Midpoint rule over a coarse grid (interior points only).
        let n = 24usize;
        let h = 1.0 / n as f64;
        let mut integral = 0.0;
        for i in 0..n {
            let u1 = (i as f64 + 0.5) * h;
            for j in 0..n {
                let u2 = (j as f64 + 0.5) * h;
                for k in 0..n {
                    let u3 = (k as f64 + 0.5) * h;
                    integral += vine.density(&[u1, u2, u3]).expect("density should succeed");
                }
            }
        }
        integral *= h * h * h;
        assert!(
            approx(integral, 1.0, 0.05),
            "C-vine density should integrate to ≈1, got {integral}"
        );
    }

    #[test]
    fn dvine_density_integrates_to_one() {
        let trees = vec![
            vec![
                PairCopula::from_param(CopulaFamily::Frank, 4.0)
                    .expect("from_param should succeed"),
                PairCopula::from_param(CopulaFamily::Clayton, 1.5)
                    .expect("from_param should succeed"),
            ],
            vec![
                PairCopula::from_param(CopulaFamily::Frank, 2.0)
                    .expect("from_param should succeed"),
            ],
        ];
        let vine = VineCopula::from_pair_copulas(3, VineType::DVine, trees)
            .expect("from_pair_copulas should succeed");
        let n = 24usize;
        let h = 1.0 / n as f64;
        let mut integral = 0.0;
        for i in 0..n {
            let u1 = (i as f64 + 0.5) * h;
            for j in 0..n {
                let u2 = (j as f64 + 0.5) * h;
                for k in 0..n {
                    let u3 = (k as f64 + 0.5) * h;
                    integral += vine.density(&[u1, u2, u3]).expect("density should succeed");
                }
            }
        }
        integral *= h * h * h;
        assert!(
            approx(integral, 1.0, 0.05),
            "D-vine density should integrate to ≈1, got {integral}"
        );
    }

    // ---- (c) sequential fit recovers known pair-copula parameters ---------
    #[test]
    fn cvine_sequential_fit_recovers_tree0_params() {
        // Simulate a C-vine with strong tree-0 dependence and (near) independent
        // tree-1, then check tree-0 Frank parameters are recovered.
        let mut rng = LcgRng::new(2024);
        let n = 4000usize;
        let dim = 3usize;
        // True tree-0 pair-copulas: variable 0 ↔ 1 and 0 ↔ 2, Frank θ=5.
        let pc01 =
            PairCopula::from_param(CopulaFamily::Frank, 5.0).expect("from_param should succeed");
        let pc02 =
            PairCopula::from_param(CopulaFamily::Frank, 5.0).expect("from_param should succeed");
        // Generate data by inverse-h sampling on the C-vine (tree-1 independent).
        let mut data = Vec::with_capacity(n * dim);
        for _ in 0..n {
            let w0 = rng.next_f64().clamp(EPS, 1.0 - EPS);
            let w1 = rng.next_f64().clamp(EPS, 1.0 - EPS);
            let w2 = rng.next_f64().clamp(EPS, 1.0 - EPS);
            // u0 = w0; u1 from h(·|u0)=w1; u2 from h(·|u0)=w2 (tree-1 indep).
            let u0 = w0;
            let u1 = invert_h(&pc01, w1, u0);
            let u2 = invert_h(&pc02, w2, u0);
            data.push(u0);
            data.push(u1);
            data.push(u2);
        }
        let cfg = VineFitConfig {
            vine_type: VineType::CVine,
            family: CopulaFamily::Frank,
        };
        let fitted = vine_fit(&data, n, dim, &cfg).expect("vine_fit should succeed");
        let th01 = fitted.trees[0][0].fit.theta;
        let th02 = fitted.trees[0][1].fit.theta;
        assert!(
            (th01 - 5.0).abs() < 0.8,
            "tree-0 θ(0,1) recovered {th01} (true 5)"
        );
        assert!(
            (th02 - 5.0).abs() < 0.8,
            "tree-0 θ(0,2) recovered {th02} (true 5)"
        );
    }

    // ---- (d) d=2 C-vine == D-vine == single bivariate copula --------------
    #[test]
    fn d2_cvine_equals_dvine_equals_bivariate() {
        let pc =
            PairCopula::from_param(CopulaFamily::Clayton, 2.5).expect("from_param should succeed");
        let cvine = VineCopula::from_pair_copulas(2, VineType::CVine, vec![vec![pc.clone()]])
            .expect("value should be present");
        let dvine = VineCopula::from_pair_copulas(2, VineType::DVine, vec![vec![pc.clone()]])
            .expect("value should be present");
        for &p in &[[0.2, 0.7], [0.4, 0.4], [0.9, 0.1], [0.55, 0.33]] {
            let dc = cvine.density(&p).expect("density should succeed");
            let dd = dvine.density(&p).expect("density should succeed");
            let db = pc.density(p[0], p[1]).expect("density should succeed");
            assert!(approx(dc, db, 1e-12), "C-vine d=2 {dc} ≠ bivariate {db}");
            assert!(approx(dd, db, 1e-12), "D-vine d=2 {dd} ≠ bivariate {db}");
            assert!(approx(dc, dd, 1e-12), "C-vine {dc} ≠ D-vine {dd}");
        }
    }

    // ---- (e) h-functions are valid conditional CDFs ∈ [0,1] ---------------
    #[test]
    fn h_functions_are_valid_cdfs() {
        let families = [
            (CopulaFamily::Gaussian, 0.6_f64),
            (CopulaFamily::Clayton, 2.0),
            (CopulaFamily::Frank, 4.0),
            (CopulaFamily::Gumbel, 1.8),
        ];
        for &(fam, th) in &families {
            let pc = PairCopula::from_param(fam, th).expect("from_param should succeed");
            // Range [0,1].
            for &u in &[0.05, 0.25, 0.5, 0.75, 0.95] {
                for &v in &[0.05, 0.25, 0.5, 0.75, 0.95] {
                    let h = pc.h_func(u, v).expect("h_func should succeed");
                    assert!(
                        (0.0..=1.0).contains(&h),
                        "h({u},{v})={h} ∉ [0,1] for {fam:?}"
                    );
                }
            }
            // Monotone non-decreasing in u for fixed v (it is a CDF in u).
            let v = 0.4;
            let mut prev = -1.0;
            for k in 1..20 {
                let u = k as f64 / 20.0;
                let h = pc.h_func(u, v).expect("h_func should succeed");
                assert!(h + 1e-9 >= prev, "h not monotone in u for {fam:?}");
                prev = h;
            }
        }
    }

    // ---- (f) log-likelihood finite and improves over independence ---------
    #[test]
    fn loglik_finite_and_beats_independence() {
        // Dependent 3-variate data simulated from a C-vine with Frank tree-0.
        let mut rng = LcgRng::new(31337);
        let n = 1500usize;
        let dim = 3usize;
        let pc01 =
            PairCopula::from_param(CopulaFamily::Frank, 6.0).expect("from_param should succeed");
        let pc02 =
            PairCopula::from_param(CopulaFamily::Frank, 6.0).expect("from_param should succeed");
        let mut data = Vec::with_capacity(n * dim);
        for _ in 0..n {
            let w0 = rng.next_f64().clamp(EPS, 1.0 - EPS);
            let w1 = rng.next_f64().clamp(EPS, 1.0 - EPS);
            let w2 = rng.next_f64().clamp(EPS, 1.0 - EPS);
            let u0 = w0;
            let u1 = invert_h(&pc01, w1, u0);
            let u2 = invert_h(&pc02, w2, u0);
            data.push(u0);
            data.push(u1);
            data.push(u2);
        }
        let cfg = VineFitConfig {
            vine_type: VineType::CVine,
            family: CopulaFamily::Frank,
        };
        let fitted = vine_fit(&data, n, dim, &cfg).expect("vine_fit should succeed");
        let ll_fit = fitted
            .log_likelihood(&data, n)
            .expect("log_likelihood should succeed");
        assert!(ll_fit.is_finite(), "fitted ll not finite: {ll_fit}");

        let indep =
            VineCopula::independence(dim, VineType::CVine).expect("independence should succeed");
        let ll_indep = indep
            .log_likelihood(&data, n)
            .expect("log_likelihood should succeed");
        assert!(
            approx(ll_indep, 0.0, 1e-9),
            "indep ll should be 0, got {ll_indep}"
        );
        assert!(
            ll_fit > ll_indep,
            "fitted vine ll {ll_fit} should beat independence ll {ll_indep}"
        );
    }

    // ---- extra: from_param parameter validation ---------------------------
    #[test]
    fn from_param_rejects_bad_theta() {
        assert!(PairCopula::from_param(CopulaFamily::Clayton, -1.0).is_err());
        assert!(PairCopula::from_param(CopulaFamily::Gumbel, 0.5).is_err());
        assert!(PairCopula::from_param(CopulaFamily::Gaussian, 1.5).is_err());
        assert!(PairCopula::from_param(CopulaFamily::Frank, 0.0).is_ok());
    }

    #[test]
    fn from_pair_copulas_wrong_tree_count_errors() {
        let trees = vec![vec![PairCopula::independence()]]; // only 1 tree for d=3
        assert!(VineCopula::from_pair_copulas(3, VineType::CVine, trees).is_err());
    }

    #[test]
    fn density_dimension_mismatch_errors() {
        let vine =
            VineCopula::independence(3, VineType::CVine).expect("independence should succeed");
        assert!(matches!(
            vine.density(&[0.5, 0.5]),
            Err(StatsError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn n_pairs_is_triangular() {
        let vine =
            VineCopula::independence(5, VineType::DVine).expect("independence should succeed");
        assert_eq!(vine.n_pairs(), 10);
    }

    // Helper: invert the h-function h(·|v)=t for the partner argument by
    // bisection — used to simulate from a known pair-copula in the tests.
    fn invert_h(pc: &PairCopula, t: f64, v: f64) -> f64 {
        let mut lo = 1e-9_f64;
        let mut hi = 1.0 - 1e-9_f64;
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            let h = pc.h_func(mid, v).unwrap_or(0.5);
            if h < t {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (0.5 * (lo + hi)).clamp(1e-9, 1.0 - 1e-9)
    }
}
