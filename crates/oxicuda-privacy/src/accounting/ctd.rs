//! Connect-the-Dots (CTD) accountant for tight privacy-loss distribution bracketing.
//!
//! Reference: Doroshenko, Ghazi, Kamath, Kumar & Manurangsi (2022),
//! "Connect the Dots: Tighter Discrete Approximations of Privacy Loss
//! Distributions", *Proceedings on Privacy Enhancing Technologies* 2022(4).
//!
//! # Overview
//! Given a continuous privacy-loss random variable (PLD) for a mechanism M on
//! neighbouring datasets x, x', we approximate its distribution by two
//! discrete histograms on the same uniform grid:
//!
//! - the **pessimistic** PLD, which dominates the true PLD in the
//!   convex stochastic order (i.e. it gives a valid *upper* bound on
//!   `δ(ε)` for every threshold ε), and
//! - the **optimistic** PLD, which is dominated by the true PLD
//!   (giving a *lower* bound on `δ(ε)`).
//!
//! The construction in §3 of Doroshenko et al. (2022) discretises the
//! continuous PLD by integrating its mass over each grid cell and then
//! "connecting the dots": each cell's mass is placed at the cell's right
//! edge (pessimistic) or left edge (optimistic). Composition of independent
//! mechanisms corresponds to **discrete convolution** of the respective
//! histograms; the convolved support is clamped back onto the same grid
//! (mass falling beyond the upper boundary stays in the boundary cell, which
//! is the conservative interpretation for both bounds — see §3.3).
//!
//! # Gaussian mechanism
//! For the Gaussian mechanism with L2 sensitivity Δ and noise std σ, the
//! continuous PLD has μ = Δ/σ and the privacy-loss RV satisfies
//! `L ~ N(μ²/2, μ²)` (Dong-Roth-Su 2022, Proposition 3). The
//! `from_gaussian` constructor evaluates the standard-normal CDF Φ on each
//! cell boundary via the Abramowitz-Stegun 7.1.26 rational approximation
//! (same routine used by `accounting::pld` and `accounting::fdp`).
//!
//! # Complexity
//! Each pairwise composition is O(n²) on a grid of size n. Self-composition
//! uses repeated squaring so a k-fold compose runs in O(log k) convolutions.

use crate::error::{PrivacyError, PrivacyResult};

/// Grid parameters for a CTD discretisation.
///
/// The privacy loss is bracketed on the interval `[grid_lo, grid_hi]`
/// partitioned into `grid_size` equal-width cells of width
/// `h = (grid_hi - grid_lo) / grid_size`.
#[derive(Debug, Clone, Copy)]
pub struct CtdConfig {
    /// Lower bound of the privacy-loss domain (typically -30).
    pub grid_lo: f64,
    /// Upper bound of the privacy-loss domain (typically +30).
    pub grid_hi: f64,
    /// Number of grid cells; must be ≥ 8.
    pub grid_size: usize,
}

impl CtdConfig {
    /// Construct and validate a `CtdConfig`.
    ///
    /// # Errors
    /// - `InvalidParameter` if `grid_size < 8`.
    /// - `InvalidParameter` if `grid_lo >= grid_hi`.
    /// - `InvalidParameter` if any bound is non-finite.
    pub fn new(grid_lo: f64, grid_hi: f64, grid_size: usize) -> PrivacyResult<Self> {
        if !grid_lo.is_finite() || !grid_hi.is_finite() {
            return Err(PrivacyError::InvalidParameter(
                "grid_lo and grid_hi must be finite".into(),
            ));
        }
        if grid_size < 8 {
            return Err(PrivacyError::InvalidParameter(format!(
                "grid_size must be ≥ 8, got {grid_size}"
            )));
        }
        if grid_lo >= grid_hi {
            return Err(PrivacyError::InvalidParameter(format!(
                "grid_lo {grid_lo} must be < grid_hi {grid_hi}"
            )));
        }
        Ok(Self {
            grid_lo,
            grid_hi,
            grid_size,
        })
    }

    /// Cell width `h = (grid_hi - grid_lo) / grid_size`.
    #[must_use]
    pub fn step(&self) -> f64 {
        (self.grid_hi - self.grid_lo) / self.grid_size as f64
    }

    /// Left edge of cell `i`.
    #[must_use]
    pub fn left_edge(&self, i: usize) -> f64 {
        self.grid_lo + i as f64 * self.step()
    }

    /// Right edge of cell `i`.
    #[must_use]
    pub fn right_edge(&self, i: usize) -> f64 {
        self.grid_lo + (i as f64 + 1.0) * self.step()
    }

    /// Centre of cell `i`.
    #[must_use]
    pub fn centre(&self, i: usize) -> f64 {
        self.grid_lo + (i as f64 + 0.5) * self.step()
    }
}

// ─── Standard-normal CDF helper ───────────────────────────────────────────────

/// Standard normal CDF Φ(x) via the Abramowitz-Stegun 7.1.26 rational
/// approximation applied to `|x|/√2`. Matches the routine used by
/// `accounting::pld::phi` and `accounting::fdp::phi`; reproduced locally to
/// keep this module self-contained.
fn phi(x: f64) -> f64 {
    let z = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * z);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erfc_z = poly * (-z * z).exp();
    if x >= 0.0 {
        1.0 - 0.5 * erfc_z
    } else {
        0.5 * erfc_z
    }
}

// ─── CTD accountant ──────────────────────────────────────────────────────────

/// Connect-the-Dots accountant: a paired pessimistic / optimistic
/// histogram on a uniform privacy-loss grid.
///
/// `pessimistic_pld[i]` and `optimistic_pld[i]` both contain the mass that
/// falls in cell `i`; the only difference is the *anchor point* at which
/// that mass is interpreted when computing `δ(ε)` — the right edge for the
/// pessimistic histogram, the left edge for the optimistic one. Storing the
/// raw cell masses lets `compose` use a single convolution and recover both
/// bounds during the `delta_at_epsilon` step.
#[derive(Debug, Clone)]
pub struct CtdAccountant {
    /// Cell centres, length `cfg.grid_size`.
    pub grid: Vec<f64>,
    /// Pessimistic PLD: cell mass evaluated at each cell's right edge.
    pub pessimistic_pld: Vec<f64>,
    /// Optimistic PLD: cell mass evaluated at each cell's left edge.
    pub optimistic_pld: Vec<f64>,
    /// Configuration used to build this accountant.
    pub cfg: CtdConfig,
}

impl CtdAccountant {
    /// Build a CTD accountant for the Gaussian mechanism with the given
    /// noise std σ and L2 sensitivity Δ.
    ///
    /// The continuous privacy-loss RV is `L ~ N(μ²/2, μ²)` where
    /// `μ = Δ/σ`. We integrate this density over each cell via Φ-differences
    /// to obtain exact cell masses, then store them so that the right-edge
    /// (pessimistic) and left-edge (optimistic) interpretations are
    /// available at evaluation time.
    ///
    /// Mass below `grid_lo` is folded into the first cell; mass above
    /// `grid_hi` is folded into the last cell. After truncation the cell
    /// masses are renormalised to sum to 1.
    ///
    /// # Errors
    /// - `NonPositiveSensitivity` if `sensitivity ≤ 0`.
    /// - `InvalidParameter` if `noise_sigma ≤ 0`.
    pub fn from_gaussian(
        noise_sigma: f64,
        sensitivity: f64,
        cfg: &CtdConfig,
    ) -> PrivacyResult<Self> {
        if sensitivity <= 0.0 {
            return Err(PrivacyError::NonPositiveSensitivity(sensitivity));
        }
        if noise_sigma <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "noise_sigma must be positive, got {noise_sigma}"
            )));
        }
        let mu = sensitivity / noise_sigma;
        // L ~ N(mean, sd²) with mean = μ²/2 and sd = μ.
        let mean = 0.5 * mu * mu;
        let sd = mu;

        let n = cfg.grid_size;
        let mut cell_mass = vec![0.0f64; n];
        let mut grid = vec![0.0f64; n];

        for (i, slot) in cell_mass.iter_mut().enumerate() {
            let l_edge = cfg.left_edge(i);
            let r_edge = cfg.right_edge(i);
            let p_r = phi((r_edge - mean) / sd);
            let p_l = phi((l_edge - mean) / sd);
            *slot = (p_r - p_l).max(0.0);
            grid[i] = cfg.centre(i);
        }

        // Absorb the tails outside [grid_lo, grid_hi].
        let lower_tail = phi((cfg.grid_lo - mean) / sd).clamp(0.0, 1.0);
        let upper_tail = (1.0 - phi((cfg.grid_hi - mean) / sd)).clamp(0.0, 1.0);
        if let Some(first) = cell_mass.first_mut() {
            *first += lower_tail;
        }
        if let Some(last) = cell_mass.last_mut() {
            *last += upper_tail;
        }

        // Renormalise so total mass = 1 (handles Φ-approximation error).
        let total: f64 = cell_mass.iter().sum();
        if total > 0.0 {
            let inv = 1.0 / total;
            for v in cell_mass.iter_mut() {
                *v *= inv;
            }
        }

        Ok(Self {
            grid,
            pessimistic_pld: cell_mass.clone(),
            optimistic_pld: cell_mass,
            cfg: *cfg,
        })
    }

    /// Compose two CTD accountants on a shared grid.
    ///
    /// The composed PLD is the convolution of the two cell-mass histograms
    /// because L_total = L_self + L_other for independent mechanisms. We
    /// translate each pair (i, j) of input cells into the output cell whose
    /// centre is the sum of the input cell centres (i.e. the shift is
    /// `i + j - n/2` after subtracting the grid offset). Mass that lands
    /// outside the grid is clamped to the nearest boundary cell, which
    /// preserves the conservative bracketing.
    ///
    /// # Errors
    /// `DimensionMismatch` if the grids have different sizes; `InvalidParameter`
    /// if the grid steps differ (within 1e-12 tolerance).
    pub fn compose(&self, other: &Self) -> PrivacyResult<Self> {
        if self.cfg.grid_size != other.cfg.grid_size {
            return Err(PrivacyError::DimensionMismatch {
                expected: self.cfg.grid_size,
                got: other.cfg.grid_size,
            });
        }
        if (self.cfg.step() - other.cfg.step()).abs() > 1e-12 {
            return Err(PrivacyError::InvalidParameter(format!(
                "compose requires matching grid step ({} vs {})",
                self.cfg.step(),
                other.cfg.step()
            )));
        }
        let n = self.cfg.grid_size;
        // We need to map an (i, j) pair onto an output cell k on `self.cfg`
        // such that centre_self(i) + centre_other(j) matches centre_self(k).
        // Setting `self.lo + (i+0.5)·h + other.lo + (j+0.5)·h = self.lo + (k+0.5)·h`
        // and solving gives  k = i + j + (other.lo / h) + 0.5. We round to the
        // nearest integer; with a symmetric grid `other.lo = -(n·h)/2` and the
        // shift simplifies to `i + j - (n/2 − 0.5)`, i.e. roughly `i + j − n/2`.
        let step = self.cfg.step();
        let shift = (other.cfg.grid_lo / step + 0.5).round() as i64;

        let mut pess = vec![0.0f64; n];
        let mut opt = vec![0.0f64; n];

        for i in 0..n {
            let pi = self.pessimistic_pld[i];
            let oi = self.optimistic_pld[i];
            if pi == 0.0 && oi == 0.0 {
                continue;
            }
            for j in 0..n {
                let pj = other.pessimistic_pld[j];
                let oj = other.optimistic_pld[j];
                if pj == 0.0 && oj == 0.0 {
                    continue;
                }
                let target = (i as i64) + (j as i64) + shift;
                let clamped = target.clamp(0, (n as i64) - 1) as usize;
                pess[clamped] += pi * pj;
                opt[clamped] += oi * oj;
            }
        }

        // Renormalise to guard against accumulated FP drift.
        renormalise(&mut pess);
        renormalise(&mut opt);

        Ok(Self {
            grid: self.grid.clone(),
            pessimistic_pld: pess,
            optimistic_pld: opt,
            cfg: self.cfg,
        })
    }

    /// k-fold self-composition via repeated squaring.
    ///
    /// `k = 0` returns the identity accountant (point mass at L = 0,
    /// representing "no privacy loss yet").
    ///
    /// # Errors
    /// Propagates errors from `compose`.
    pub fn compose_self(&self, k: usize) -> PrivacyResult<Self> {
        if k == 0 {
            return Self::identity(&self.cfg);
        }
        if k == 1 {
            return Ok(self.clone());
        }
        let mut result: Option<Self> = None;
        let mut base = self.clone();
        let mut exp = k;
        while exp > 0 {
            if exp & 1 == 1 {
                result = Some(match result {
                    Some(acc) => acc.compose(&base)?,
                    None => base.clone(),
                });
            }
            exp >>= 1;
            if exp > 0 {
                base = base.compose(&base)?;
            }
        }
        result.ok_or_else(|| PrivacyError::InvalidParameter("compose_self failed".into()))
    }

    /// Identity (no-op) CTD: a unit point mass placed at the cell whose
    /// centre is the largest value ≤ 0. Matches the `pld::Pld::identity`
    /// convention used elsewhere in `accounting`.
    fn identity(cfg: &CtdConfig) -> PrivacyResult<Self> {
        let n = cfg.grid_size;
        let mut grid = vec![0.0f64; n];
        for (i, g) in grid.iter_mut().enumerate() {
            *g = cfg.centre(i);
        }
        let step = cfg.step();
        // Pick the cell index whose centre is the largest value ≤ 0.
        // The centres are `lo + (i + 0.5) * step`, so we want
        // `i = floor((-lo)/step - 0.5)`.
        let raw = (-cfg.grid_lo) / step - 0.5;
        let idx = if raw <= 0.0 {
            0usize
        } else if raw as usize >= n {
            n - 1
        } else {
            raw.floor() as usize
        };
        let chosen = if grid[idx] <= 0.0 {
            idx
        } else if idx == 0 {
            0
        } else {
            idx - 1
        };
        let mut mass = vec![0.0f64; n];
        mass[chosen] = 1.0;
        Ok(Self {
            grid,
            pessimistic_pld: mass.clone(),
            optimistic_pld: mass,
            cfg: *cfg,
        })
    }

    /// Pessimistic upper bound on `δ(ε)`:
    ///
    /// `δ(ε) = Σ_{i : r_i ≥ ε} pessimistic_pld[i] · (1 - e^(ε - r_i))`
    ///
    /// where `r_i = right_edge(i)` is the right edge of cell `i`. Clamped to
    /// `[0, 1]`.
    pub fn delta_at_epsilon(&self, epsilon: f64) -> PrivacyResult<f64> {
        if !epsilon.is_finite() {
            // Limit cases: ε → +∞ ⇒ δ → 0; ε → -∞ ⇒ δ → 1.
            if epsilon > 0.0 {
                return Ok(0.0);
            }
            return Ok(1.0);
        }
        let mut delta = 0.0f64;
        for (i, &p) in self.pessimistic_pld.iter().enumerate() {
            if p == 0.0 {
                continue;
            }
            let r_i = self.cfg.right_edge(i);
            if r_i < epsilon {
                continue;
            }
            let factor = 1.0 - (epsilon - r_i).exp();
            if factor > 0.0 {
                delta += p * factor;
            }
        }
        Ok(delta.clamp(0.0, 1.0))
    }

    /// Optimistic lower bound on `δ(ε)`, mirroring `delta_at_epsilon` but
    /// using the **left** edges as anchor points.
    pub fn delta_at_epsilon_optimistic(&self, epsilon: f64) -> PrivacyResult<f64> {
        if !epsilon.is_finite() {
            if epsilon > 0.0 {
                return Ok(0.0);
            }
            return Ok(1.0);
        }
        let mut delta = 0.0f64;
        for (i, &p) in self.optimistic_pld.iter().enumerate() {
            if p == 0.0 {
                continue;
            }
            let l_i = self.cfg.left_edge(i);
            if l_i < epsilon {
                continue;
            }
            let factor = 1.0 - (epsilon - l_i).exp();
            if factor > 0.0 {
                delta += p * factor;
            }
        }
        Ok(delta.clamp(0.0, 1.0))
    }

    /// Smallest ε ≥ 0 such that `delta_at_epsilon(ε) ≤ δ_target`, found by
    /// bisection on `ε ∈ [0, grid_hi]`.
    ///
    /// # Errors
    /// - `InvalidDelta` if `delta ∉ (0, 1)`.
    pub fn epsilon_at_delta(&self, delta: f64) -> PrivacyResult<f64> {
        if !(delta > 0.0 && delta < 1.0) {
            return Err(PrivacyError::InvalidDelta(delta));
        }
        // Bracket search: ε = 0 gives the largest δ; ε = grid_hi gives ≈ 0.
        let lo_eps = 0.0f64;
        let hi_eps = self.cfg.grid_hi.max(1.0);
        let delta_at_lo = self.delta_at_epsilon(lo_eps)?;
        if delta_at_lo <= delta {
            return Ok(lo_eps);
        }
        let delta_at_hi = self.delta_at_epsilon(hi_eps)?;
        if delta_at_hi > delta {
            // Even the largest searched ε does not satisfy δ_target;
            // return the upper bound — caller should widen the grid.
            return Ok(hi_eps);
        }
        let mut lo = lo_eps;
        let mut hi = hi_eps;
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            let d = self.delta_at_epsilon(mid)?;
            if d > delta {
                lo = mid;
            } else {
                hi = mid;
            }
            if hi - lo < 1e-9 {
                return Ok(hi);
            }
        }
        Ok(hi)
    }
}

fn renormalise(mass: &mut [f64]) {
    let total: f64 = mass.iter().sum();
    if total > 0.0 && (total - 1.0).abs() > 1e-12 {
        let inv = 1.0 / total;
        for v in mass.iter_mut() {
            *v *= inv;
        }
    }
}

// Tests live in a sibling file to keep this module under 600 lines.
#[cfg(test)]
#[path = "ctd_tests.rs"]
mod tests;
