//! Parks-McClellan / Remez exchange equiripple FIR filter design.
//!
//! Implements the McClellan-Parks-Rabiner (1973) Remez exchange algorithm for
//! **type-I** linear-phase FIR filters (odd number of taps `N`, symmetric
//! impulse response).  The resulting filter is *equiripple*: it minimises the
//! maximum weighted approximation error `max_f W(f)·|D(f) − A(f)|` over the
//! union of the specified frequency bands, producing equal-height ripples in
//! every band (Chebyshev / minimax optimal).
//!
//! ## Theory
//!
//! For a type-I filter with `N = 2L + 1` taps the zero-phase frequency
//! response is a cosine sum
//! ```text
//! A(ω) = Σ_{k=0}^{L} a_k · cos(k·ω),   ω = 2π f,  f ∈ [0, 0.5].
//! ```
//! With the substitution `x = cos(ω)` this is an ordinary polynomial of
//! degree `L` in `x`, so the best Chebyshev approximation is characterised by
//! the alternation theorem: the optimal error `E(f) = W(f)·(D(f) − A(f))`
//! attains its maximum magnitude `δ` with alternating sign at (at least)
//! `r = L + 2` *extremal frequencies*.  The Remez exchange iteratively
//! relocates those extrema until the alternation is achieved.
//!
//! Each iteration uses the barycentric form of Lagrange interpolation, which
//! both yields a closed-form expression for the deviation `δ` and lets us
//! evaluate `A(f)` cheaply on a dense grid.
//!
//! References:
//!   T. W. Parks & J. H. McClellan (1972), "Chebyshev Approximation for
//!     Nonrecursive Digital Filters with Linear Phase", IEEE Trans. Circuit
//!     Theory 19(2):189-194.
//!   J. H. McClellan, T. W. Parks & L. R. Rabiner (1973), "A Computer Program
//!     for Designing Optimum FIR Linear Phase Digital Filters", IEEE Trans.
//!     Audio Electroacoust. 21(6):506-526.

use std::f64::consts::PI;

use crate::error::{SignalError, SignalResult};

/// A single frequency band specification for [`remez`].
///
/// Band edges are in *normalised* frequency `f ∈ [0, 0.5]` where `0.5` is the
/// Nyquist frequency (so `f = frequency / sampling_rate`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RemezBand {
    /// Lower band edge, normalised to `[0, 0.5]`.
    pub low: f64,
    /// Upper band edge, normalised to `[0, 0.5]` and `> low`.
    pub high: f64,
    /// Desired (target) magnitude response in this band.
    pub desired: f64,
    /// Relative error weight in this band (larger ⇒ smaller ripple here).
    pub weight: f64,
}

impl RemezBand {
    /// Construct a band, validating that `0 ≤ low < high ≤ 0.5` and
    /// `weight > 0`.
    ///
    /// # Errors
    /// Returns [`SignalError::InvalidParameter`] on any violated constraint.
    pub fn new(low: f64, high: f64, desired: f64, weight: f64) -> SignalResult<Self> {
        if !(low.is_finite() && high.is_finite() && desired.is_finite() && weight.is_finite()) {
            return Err(SignalError::InvalidParameter(
                "remez band parameters must be finite".to_owned(),
            ));
        }
        if !(0.0..=0.5).contains(&low) || !(0.0..=0.5).contains(&high) {
            return Err(SignalError::InvalidParameter(format!(
                "remez band edges [{low}, {high}] must lie in [0, 0.5]"
            )));
        }
        if low >= high {
            return Err(SignalError::InvalidParameter(format!(
                "remez band low ({low}) must be < high ({high})"
            )));
        }
        if weight <= 0.0 {
            return Err(SignalError::InvalidParameter(format!(
                "remez band weight ({weight}) must be > 0"
            )));
        }
        Ok(Self {
            low,
            high,
            desired,
            weight,
        })
    }
}

/// Grid oversampling density: number of dense-grid frequencies per cosine
/// basis term.  The classic MPR program uses 16; denser grids improve
/// extremum localisation at higher cost.
const GRID_DENSITY: usize = 16;

/// Maximum number of Remez exchange iterations before giving up.
const MAX_ITERATIONS: usize = 64;

/// Relative convergence tolerance on the extremal-error spread.
const CONVERGENCE_TOL: f64 = 1e-6;

/// A point on the dense frequency grid together with its desired response and
/// weight (frequencies that fall in "don't care" transition regions are simply
/// never added to the grid).
struct GridPoint {
    /// Normalised frequency `f ∈ [0, 0.5]`.
    freq: f64,
    /// `x = cos(2π f)` — the polynomial abscissa.
    cos_w: f64,
    /// Desired response `D(f)`.
    desired: f64,
    /// Weight `W(f)`.
    weight: f64,
}

/// Build the dense frequency grid spanning all bands.
///
/// Each band receives a share of the grid proportional to its width, with a
/// minimum so that even razor-thin bands are represented.  Band edges are
/// always included as candidate extrema.
fn build_grid(bands: &[RemezBand], n_basis: usize) -> Vec<GridPoint> {
    let total_target = (GRID_DENSITY * n_basis).max(bands.len() * 4);
    let total_width: f64 = bands.iter().map(|b| b.high - b.low).sum();
    let mut grid = Vec::with_capacity(total_target + bands.len());

    for band in bands {
        let width = band.high - band.low;
        // Proportional allocation, at least 2 interior steps per band.
        let share = if total_width > 0.0 {
            ((total_target as f64) * width / total_width).round() as usize
        } else {
            total_target / bands.len().max(1)
        };
        let n_pts = share.max(3);
        for i in 0..n_pts {
            let frac = i as f64 / (n_pts - 1) as f64;
            let freq = band.low + frac * width;
            grid.push(GridPoint {
                freq,
                cos_w: (2.0 * PI * freq).cos(),
                desired: band.desired,
                weight: band.weight,
            });
        }
    }

    // The grid is naturally sorted by frequency because bands are processed in
    // order and `RemezBand::new` forbids overlaps only loosely; sort to be safe
    // and de-duplicate coincident abscissae that would break the barycentric
    // weights (Π (x_i − x_j) must be non-zero).
    grid.sort_by(|a, b| {
        a.freq
            .partial_cmp(&b.freq)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    grid.dedup_by(|a, b| (a.freq - b.freq).abs() < 1e-12);
    grid
}

/// Compute barycentric Lagrange weights `γ_i = 1 / Π_{j≠i}(x_i − x_j)` for the
/// abscissae `x`.
fn barycentric_weights(x: &[f64]) -> Vec<f64> {
    let r = x.len();
    let mut gamma = vec![1.0_f64; r];
    for i in 0..r {
        let mut prod = 1.0_f64;
        for (j, &xj) in x.iter().enumerate() {
            if i != j {
                prod *= x[i] - xj;
            }
        }
        gamma[i] = 1.0 / prod;
    }
    gamma
}

/// One Remez exchange iteration result: the achieved deviation and the cosine
/// response sampled on the dense grid.
struct ExchangeStep {
    /// Closed-form deviation `δ` at the current extrema.
    delta: f64,
    /// `A(f)` evaluated on every dense-grid point.
    response: Vec<f64>,
}

/// Evaluate the deviation `δ` and the interpolated response `A(f)` over the
/// whole grid given the current extremal frequencies.
///
/// `ext` holds indices into `grid` of the current `r = L + 2` extrema.  The
/// barycentric interpolation uses the first `r − 1` extrema as nodes; the last
/// one only participates in the alternation (it pins `δ`).
fn exchange_step(grid: &[GridPoint], ext: &[usize]) -> ExchangeStep {
    let r = ext.len();
    // Abscissae and per-extremum data.
    let x_ext: Vec<f64> = ext.iter().map(|&i| grid[i].cos_w).collect();
    let gamma = barycentric_weights(&x_ext);

    // Closed-form deviation:
    //   δ = (Σ γ_i D_i) / (Σ (−1)^i γ_i / W_i).
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for (i, &idx) in ext.iter().enumerate() {
        let g = gamma[i];
        let d = grid[idx].desired;
        let w = grid[idx].weight;
        num += g * d;
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        den += sign * g / w;
    }
    let delta = if den.abs() > f64::MIN_POSITIVE {
        num / den
    } else {
        0.0
    };

    // Interpolation values at the extrema: C_i = D_i − (−1)^i δ / W_i.
    let c: Vec<f64> = ext
        .iter()
        .enumerate()
        .map(|(i, &idx)| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            grid[idx].desired - sign * delta / grid[idx].weight
        })
        .collect();

    // Barycentric interpolation of A(f) through the first (r − 1) nodes.  Using
    // r − 1 nodes (degree L polynomial through L + 1 points) is the standard
    // MPR trick: the final extremum is redundant and serves only to fix δ.
    let n_nodes = r - 1;
    let nodes_x = &x_ext[..n_nodes];
    let nodes_w = barycentric_weights(nodes_x);
    let nodes_c = &c[..n_nodes];

    let mut response = vec![0.0_f64; grid.len()];
    for (gi, gp) in grid.iter().enumerate() {
        let x = gp.cos_w;
        // Check for coincidence with a node (avoid 0/0 in barycentric form).
        let mut exact: Option<f64> = None;
        for k in 0..n_nodes {
            if (x - nodes_x[k]).abs() < 1e-14 {
                exact = Some(nodes_c[k]);
                break;
            }
        }
        response[gi] = if let Some(v) = exact {
            v
        } else {
            let mut numr = 0.0_f64;
            let mut denr = 0.0_f64;
            for k in 0..n_nodes {
                let t = nodes_w[k] / (x - nodes_x[k]);
                numr += t * nodes_c[k];
                denr += t;
            }
            numr / denr
        };
    }

    ExchangeStep { delta, response }
}

/// Find the new set of `r` extremal frequencies: the largest-magnitude local
/// extrema of the weighted error `E(f)` over the dense grid, including band
/// edges as candidates, retaining an alternating-sign subset of size `r`.
fn select_extrema(grid: &[GridPoint], err: &[f64], r: usize) -> Vec<usize> {
    let n = grid.len();
    // Collect candidate local maxima of |E| (interior sign-change-free peaks)
    // plus the two global endpoints (band-union edges).
    let mut cands: Vec<usize> = Vec::new();
    if n > 0 {
        cands.push(0);
    }
    for i in 1..n.saturating_sub(1) {
        let a = err[i - 1];
        let b = err[i];
        let c = err[i + 1];
        // Local extremum of the signed error (peak or trough).
        let is_peak = b >= a && b >= c;
        let is_trough = b <= a && b <= c;
        if is_peak || is_trough {
            cands.push(i);
        }
    }
    if n > 1 {
        cands.push(n - 1);
    }
    cands.dedup();

    if cands.len() <= r {
        return cands;
    }

    // Reduce the candidate set to exactly `r` alternating extrema.  Strategy
    // (per MPR): scan adjacent candidates; when two consecutive candidates have
    // the *same* error sign, keep only the one with larger magnitude.  Then, if
    // still too many, repeatedly drop the smallest-magnitude extremum from the
    // shorter end until exactly `r` remain.
    let mut kept: Vec<usize> = Vec::with_capacity(cands.len());
    let mut iter = cands.into_iter();
    if let Some(first) = iter.next() {
        kept.push(first);
        for idx in iter {
            let prev = *kept.last().unwrap_or(&idx);
            if err[idx].signum() == err[prev].signum() {
                // Same sign run — keep the larger magnitude.
                if err[idx].abs() > err[prev].abs() {
                    let last = kept.len() - 1;
                    kept[last] = idx;
                }
            } else {
                kept.push(idx);
            }
        }
    }

    // Now `kept` strictly alternates in sign.  Trim to `r` by discarding the
    // smallest-magnitude endpoints (preserving alternation, which removing an
    // end always does).
    while kept.len() > r {
        let first_mag = err[kept[0]].abs();
        let last_mag = err[kept[kept.len() - 1]].abs();
        if first_mag <= last_mag {
            kept.remove(0);
        } else {
            kept.pop();
        }
    }
    kept
}

/// Design a type-I (odd-length, symmetric) linear-phase equiripple FIR filter
/// via the Remez exchange algorithm.
///
/// # Parameters
/// - `num_taps` — total number of taps `N`; **must be odd** (type I).
/// - `bands` — non-overlapping frequency bands with desired gain and weight;
///   frequencies between bands are "don't care" transition regions.
///
/// Returns the symmetric impulse response `h[0..num_taps]` (so
/// `h[n] == h[N−1−n]`), which guarantees exact linear phase.
///
/// # Errors
/// Returns [`SignalError::InvalidParameter`] if `num_taps` is even or `< 3`,
/// if `bands` is empty, or if any band is malformed (see [`RemezBand::new`]).
pub fn remez(num_taps: usize, bands: &[RemezBand]) -> SignalResult<Vec<f32>> {
    if bands.is_empty() {
        return Err(SignalError::InvalidParameter(
            "remez requires at least one frequency band".to_owned(),
        ));
    }
    if num_taps < 3 {
        return Err(SignalError::InvalidParameter(format!(
            "remez num_taps ({num_taps}) must be ≥ 3"
        )));
    }
    if num_taps % 2 == 0 {
        return Err(SignalError::InvalidParameter(format!(
            "remez supports only type-I (odd) filters; num_taps = {num_taps} is even"
        )));
    }
    // Validate band edges (re-run the constructor checks; bands may have been
    // built by hand rather than via `RemezBand::new`).
    for b in bands {
        RemezBand::new(b.low, b.high, b.desired, b.weight)?;
    }
    // Guard against overlapping bands which would corrupt the grid ordering.
    let mut sorted = bands.to_vec();
    sorted.sort_by(|a, b| {
        a.low
            .partial_cmp(&b.low)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for w in sorted.windows(2) {
        if w[1].low < w[0].high - 1e-12 {
            return Err(SignalError::InvalidParameter(format!(
                "remez bands overlap: [{}, {}] and [{}, {}]",
                w[0].low, w[0].high, w[1].low, w[1].high
            )));
        }
    }

    let l = (num_taps - 1) / 2; // half length
    let n_basis = l + 1; // cosine terms a_0..a_L
    let r = l + 2; // number of extremal frequencies

    let grid = build_grid(&sorted, n_basis);
    if grid.len() < r {
        return Err(SignalError::InvalidParameter(format!(
            "remez grid too small ({} points) for {r} extrema; widen the bands",
            grid.len()
        )));
    }

    // Initial extrema: uniformly spaced indices across the grid.
    let mut ext: Vec<usize> = (0..r)
        .map(|i| {
            // Spread r indices across [0, grid.len()-1].
            ((i * (grid.len() - 1)) as f64 / (r - 1) as f64).round() as usize
        })
        .collect();
    ext.dedup();
    // Ensure exactly r distinct indices (pathological tiny grids).
    let mut probe = 0usize;
    while ext.len() < r && probe < grid.len() {
        if !ext.contains(&probe) {
            ext.push(probe);
        }
        probe += 1;
    }
    ext.sort_unstable();
    ext.truncate(r);

    let mut last_response = vec![0.0_f64; grid.len()];
    let mut converged_delta = 0.0_f64;

    for _iter in 0..MAX_ITERATIONS {
        let step = exchange_step(&grid, &ext);
        converged_delta = step.delta;
        last_response.clone_from(&step.response);

        // Weighted error on the dense grid.
        let err: Vec<f64> = grid
            .iter()
            .zip(step.response.iter())
            .map(|(gp, &a)| gp.weight * (gp.desired - a))
            .collect();

        // New extrema.
        let new_ext = select_extrema(&grid, &err, r);
        if new_ext.len() < r {
            // Not enough extrema located — keep current set and stop refining.
            break;
        }

        // Convergence: extremal error magnitudes equal within tolerance.
        let ext_errs: Vec<f64> = new_ext.iter().map(|&i| err[i].abs()).collect();
        let max_e = ext_errs.iter().cloned().fold(0.0_f64, f64::max);
        let min_e = ext_errs.iter().cloned().fold(f64::INFINITY, f64::min);
        let spread = if max_e > 0.0 {
            (max_e - min_e) / max_e
        } else {
            0.0
        };

        let unchanged = new_ext == ext;
        ext = new_ext;
        if unchanged || spread < CONVERGENCE_TOL {
            break;
        }
    }

    // Recover the impulse response.  We have A(f) on the dense grid; to obtain
    // the cosine coefficients a_k we sample A at the L + 1 equally-spaced
    // frequencies f_m = m / (2L), m = 0..L, then apply the inverse discrete
    // cosine transform of the half-band response.
    //
    // Build A at the sample frequencies by re-running the barycentric
    // interpolation that produced `last_response`; simplest is to interpolate
    // from the current extrema once more at the desired abscissae.
    let a_samples = sample_response_at_uniform(&grid, &ext, l);

    // Inverse cosine transform (orthogonal DCT-I).  We have
    //   A(f_m) = Σ_{k=0}^{L} a_k cos(2π k f_m),   f_m = m/(2L)  ⇒  2π f_m = π m / L,
    // i.e. A_m = Σ_k a_k cos(π k m / L).  Inverting this DCT-I (size L+1) gives
    //   a_k = (2/L) · ε_k · Σ_{m=0}^{L} ε_m · A_m · cos(π k m / L),
    // with the endpoint half-weights ε_j = 1/2 for j ∈ {0, L} and 1 otherwise.
    let mut a_coef = vec![0.0_f64; l + 1];
    let l_f = l.max(1) as f64;
    let eps = |j: usize| -> f64 { if j == 0 || j == l { 0.5 } else { 1.0 } };
    for (k, ak) in a_coef.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for (m, &am) in a_samples.iter().enumerate() {
            acc += eps(m) * am * (PI * k as f64 * m as f64 / l_f).cos();
        }
        *ak = (2.0 / l_f) * eps(k) * acc;
    }

    // Assemble the symmetric impulse response from the cosine coefficients:
    //   h[L]     = a_0
    //   h[L ± k] = a_k / 2,   k = 1..L.
    let mut h = vec![0.0_f64; num_taps];
    h[l] = a_coef[0];
    for k in 1..=l {
        let v = a_coef[k] / 2.0;
        h[l + k] = v;
        h[l - k] = v;
    }

    // Force exact symmetry (kills tiny asymmetries from finite-precision IDCT).
    for k in 0..l {
        let avg = 0.5 * (h[k] + h[num_taps - 1 - k]);
        h[k] = avg;
        h[num_taps - 1 - k] = avg;
    }

    let _ = converged_delta; // δ is an internal diagnostic, not returned.
    Ok(h.into_iter().map(|v| v as f32).collect())
}

/// Sample the equiripple half-band response `A(f)` at the `L + 1` uniformly
/// spaced frequencies `f_m = m / (2L)`, using the final extrema as the
/// barycentric interpolation nodes.
fn sample_response_at_uniform(grid: &[GridPoint], ext: &[usize], l: usize) -> Vec<f64> {
    let r = ext.len();
    let x_ext: Vec<f64> = ext.iter().map(|&i| grid[i].cos_w).collect();
    let n_nodes = r - 1;
    let nodes_x = &x_ext[..n_nodes];
    let nodes_w = barycentric_weights(nodes_x);

    // Reconstruct C_i at the nodes from the converged extrema.  We re-derive δ
    // the same way as in `exchange_step` for self-consistency.
    let gamma = barycentric_weights(&x_ext);
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for (i, &idx) in ext.iter().enumerate() {
        let g = gamma[i];
        num += g * grid[idx].desired;
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        den += sign * g / grid[idx].weight;
    }
    let delta = if den.abs() > f64::MIN_POSITIVE {
        num / den
    } else {
        0.0
    };
    let c: Vec<f64> = ext
        .iter()
        .enumerate()
        .map(|(i, &idx)| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            grid[idx].desired - sign * delta / grid[idx].weight
        })
        .collect();
    let nodes_c = &c[..n_nodes];

    let mut out = vec![0.0_f64; l + 1];
    for (m, om) in out.iter_mut().enumerate() {
        let freq = if l == 0 {
            0.0
        } else {
            m as f64 / (2.0 * l as f64)
        };
        let x = (2.0 * PI * freq).cos();
        let mut exact: Option<f64> = None;
        for k in 0..n_nodes {
            if (x - nodes_x[k]).abs() < 1e-14 {
                exact = Some(nodes_c[k]);
                break;
            }
        }
        *om = if let Some(v) = exact {
            v
        } else {
            let mut numr = 0.0_f64;
            let mut denr = 0.0_f64;
            for k in 0..n_nodes {
                let t = nodes_w[k] / (x - nodes_x[k]);
                numr += t * nodes_c[k];
                denr += t;
            }
            numr / denr
        };
    }
    out
}

// --------------------------------------------------------------------------- //
//  Convenience constructors
// --------------------------------------------------------------------------- //

/// Design an equiripple lowpass filter.
///
/// # Parameters
/// - `num_taps` — odd number of taps.
/// - `pass_edge` — passband upper edge in `[0, 0.5]`.
/// - `stop_edge` — stopband lower edge in `[0, 0.5]`, `> pass_edge`.
/// - `pass_weight`, `stop_weight` — relative error weights.
///
/// # Errors
/// See [`remez`].
pub fn remez_lowpass(
    num_taps: usize,
    pass_edge: f64,
    stop_edge: f64,
    pass_weight: f64,
    stop_weight: f64,
) -> SignalResult<Vec<f32>> {
    let bands = [
        RemezBand::new(0.0, pass_edge, 1.0, pass_weight)?,
        RemezBand::new(stop_edge, 0.5, 0.0, stop_weight)?,
    ];
    remez(num_taps, &bands)
}

/// Design an equiripple highpass filter.
///
/// # Errors
/// See [`remez`].
pub fn remez_highpass(
    num_taps: usize,
    stop_edge: f64,
    pass_edge: f64,
    stop_weight: f64,
    pass_weight: f64,
) -> SignalResult<Vec<f32>> {
    let bands = [
        RemezBand::new(0.0, stop_edge, 0.0, stop_weight)?,
        RemezBand::new(pass_edge, 0.5, 1.0, pass_weight)?,
    ];
    remez(num_taps, &bands)
}

/// Design an equiripple bandpass filter.
///
/// Band layout: stop `[0, stop_lo]`, pass `[pass_lo, pass_hi]`, stop
/// `[stop_hi, 0.5]`.
///
/// # Errors
/// See [`remez`].
pub fn remez_bandpass(
    num_taps: usize,
    stop_lo: f64,
    pass_lo: f64,
    pass_hi: f64,
    stop_hi: f64,
    stop_weight: f64,
    pass_weight: f64,
) -> SignalResult<Vec<f32>> {
    let bands = [
        RemezBand::new(0.0, stop_lo, 0.0, stop_weight)?,
        RemezBand::new(pass_lo, pass_hi, 1.0, pass_weight)?,
        RemezBand::new(stop_hi, 0.5, 0.0, stop_weight)?,
    ];
    remez(num_taps, &bands)
}

/// Design an equiripple bandstop (band-reject) filter.
///
/// Band layout: pass `[0, pass_lo]`, stop `[stop_lo, stop_hi]`, pass
/// `[pass_hi, 0.5]`.
///
/// # Errors
/// See [`remez`].
pub fn remez_bandstop(
    num_taps: usize,
    pass_lo: f64,
    stop_lo: f64,
    stop_hi: f64,
    pass_hi: f64,
    pass_weight: f64,
    stop_weight: f64,
) -> SignalResult<Vec<f32>> {
    let bands = [
        RemezBand::new(0.0, pass_lo, 1.0, pass_weight)?,
        RemezBand::new(stop_lo, stop_hi, 0.0, stop_weight)?,
        RemezBand::new(pass_hi, 0.5, 1.0, pass_weight)?,
    ];
    remez(num_taps, &bands)
}

/// Evaluate the magnitude response `|H(e^{jω})|` of a real FIR filter at the
/// normalised frequency `f ∈ [0, 0.5]`.  Helper for tests and band analysis.
#[must_use]
pub fn magnitude_at(h: &[f32], freq: f64) -> f64 {
    let omega = 2.0 * PI * freq;
    let (mut re, mut im) = (0.0_f64, 0.0_f64);
    for (n, &hn) in h.iter().enumerate() {
        let angle = -omega * n as f64;
        re += hn as f64 * angle.cos();
        im += hn as f64 * angle.sin();
    }
    (re * re + im * im).sqrt()
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    /// Largest deviation in a band evaluated on a fine sub-grid.
    fn band_max_dev(h: &[f32], lo: f64, hi: f64, desired: f64, n: usize) -> f64 {
        let mut worst = 0.0_f64;
        for i in 0..n {
            let f = lo + (hi - lo) * i as f64 / (n - 1) as f64;
            let dev = (magnitude_at(h, f) - desired).abs();
            worst = worst.max(dev);
        }
        worst
    }

    #[test]
    fn test_remez_lowpass_symmetric() {
        let h = remez_lowpass(21, 0.2, 0.3, 1.0, 1.0).expect("valid lowpass");
        assert_eq!(h.len(), 21);
        for n in 0..h.len() {
            assert!(
                (h[n] - h[h.len() - 1 - n]).abs() < 1e-6,
                "asymmetry at {n}: {} vs {}",
                h[n],
                h[h.len() - 1 - n]
            );
        }
    }

    #[test]
    fn test_remez_lowpass_dc_gain() {
        let h = remez_lowpass(31, 0.2, 0.3, 1.0, 1.0).expect("valid lowpass");
        let dc = magnitude_at(&h, 0.0);
        assert!((dc - 1.0).abs() < 0.05, "DC gain = {dc}");
    }

    #[test]
    fn test_remez_lowpass_ripple_bounds() {
        // Achieved ripple of an N=31 design with these edges is modest; assert
        // passband and stopband stay within a generous equiripple bound.
        let h = remez_lowpass(31, 0.2, 0.3, 1.0, 1.0).expect("valid lowpass");
        let pass_dev = band_max_dev(&h, 0.0, 0.2, 1.0, 64);
        let stop_dev = band_max_dev(&h, 0.3, 0.5, 0.0, 64);
        // The two ripples should both be small and comparable (equiripple).
        assert!(pass_dev < 0.08, "passband ripple too large: {pass_dev}");
        assert!(stop_dev < 0.08, "stopband ripple too large: {stop_dev}");
        // Equiripple ⇒ similar magnitude (unit weights).
        let ratio = pass_dev.max(stop_dev) / pass_dev.min(stop_dev).max(1e-12);
        assert!(
            ratio < 3.0,
            "ripples not balanced: pass={pass_dev} stop={stop_dev}"
        );
    }

    #[test]
    fn test_remez_weight_trades_ripple() {
        // Weighting the stopband 10× should shrink stopband ripple relative to
        // passband ripple compared with equal weights.
        let h_eq = remez_lowpass(31, 0.2, 0.3, 1.0, 1.0).expect("valid");
        let h_w = remez_lowpass(31, 0.2, 0.3, 1.0, 10.0).expect("valid");
        let stop_eq = band_max_dev(&h_eq, 0.32, 0.5, 0.0, 64);
        let stop_w = band_max_dev(&h_w, 0.32, 0.5, 0.0, 64);
        assert!(
            stop_w < stop_eq,
            "weighted stopband ({stop_w}) should beat equal-weight ({stop_eq})"
        );
    }

    #[test]
    fn test_remez_highpass_dc_reject() {
        let h = remez_highpass(31, 0.2, 0.3, 1.0, 1.0).expect("valid highpass");
        let dc = magnitude_at(&h, 0.0);
        let nyq = magnitude_at(&h, 0.5);
        assert!(dc < 0.1, "HP DC leakage = {dc}");
        assert!((nyq - 1.0).abs() < 0.1, "HP Nyquist gain = {nyq}");
    }

    #[test]
    fn test_remez_bandpass_shape() {
        let h = remez_bandpass(41, 0.1, 0.15, 0.3, 0.35, 1.0, 1.0).expect("valid bandpass");
        let mid = magnitude_at(&h, 0.225);
        let lo = magnitude_at(&h, 0.05);
        let hi = magnitude_at(&h, 0.45);
        assert!((mid - 1.0).abs() < 0.15, "passband centre gain = {mid}");
        assert!(lo < 0.15, "lower stopband leak = {lo}");
        assert!(hi < 0.15, "upper stopband leak = {hi}");
    }

    #[test]
    fn test_remez_bandstop_notch() {
        let h = remez_bandstop(41, 0.1, 0.18, 0.32, 0.4, 1.0, 1.0).expect("valid bandstop");
        let notch = magnitude_at(&h, 0.25);
        let dc = magnitude_at(&h, 0.02);
        let hi = magnitude_at(&h, 0.48);
        assert!(notch < 0.15, "notch depth insufficient: {notch}");
        assert!((dc - 1.0).abs() < 0.15, "DC passband gain = {dc}");
        assert!((hi - 1.0).abs() < 0.15, "high passband gain = {hi}");
    }

    /// Collect signed-error ripple peaks strictly inside a single band.
    fn band_ripple_peaks(h: &[f32], lo: f64, hi: f64, desired: f64, n: usize) -> Vec<f64> {
        let errs: Vec<f64> = (0..n)
            .map(|i| {
                let f = lo + (hi - lo) * i as f64 / (n - 1) as f64;
                magnitude_at(h, f) - desired
            })
            .collect();
        let mut peaks = Vec::new();
        for w in errs.windows(3) {
            let b = w[1];
            if (b >= w[0] && b >= w[2]) || (b <= w[0] && b <= w[2]) {
                peaks.push(b);
            }
        }
        peaks
    }

    #[test]
    fn test_remez_equiripple_alternation() {
        // Equiripple optimality (the Chebyshev minimax property).  Note that
        // `magnitude_at` returns |H(e^{jω})| = |A(f)|, which folds the negative
        // lobes of the zero-phase response A(f) upward.  Around the passband
        // target D = 1, A stays positive so |H| − 1 oscillates above/below zero
        // and we can verify sign alternation there.  In the stopband, |A| folds
        // every lobe positive, so we only assert equal-ripple magnitude.
        let h = remez_lowpass(25, 0.2, 0.3, 1.0, 1.0).expect("valid");
        let pass_peaks = band_ripple_peaks(&h, 0.0, 0.2, 1.0, 256);
        let stop_peaks = band_ripple_peaks(&h, 0.3, 0.5, 0.0, 256);

        let global_max = pass_peaks
            .iter()
            .chain(stop_peaks.iter())
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        assert!(global_max > 0.0, "no ripple peaks found");

        // Passband: dominant peaks alternate above/below the target.
        let pass_sig: Vec<f64> = pass_peaks
            .iter()
            .copied()
            .filter(|v| v.abs() > 0.4 * global_max)
            .collect();
        for pair in pass_sig.windows(2) {
            assert!(
                pair[0].signum() != pair[1].signum(),
                "passband ripple peaks must alternate in sign: {pair:?}"
            );
        }

        // Both bands: dominant ripple peaks have equal magnitude.
        for (name, peaks) in [("passband", &pass_peaks), ("stopband", &stop_peaks)] {
            for m in peaks
                .iter()
                .map(|v| v.abs())
                .filter(|&m| m > 0.6 * global_max)
            {
                assert!(
                    (m - global_max).abs() / global_max < 0.15,
                    "{name} dominant peak {m} deviates from max {global_max}"
                );
            }
        }

        // Equal weights ⇒ passband and stopband peak ripples agree.
        let pass_max = pass_peaks.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let stop_max = stop_peaks.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let ratio = pass_max.max(stop_max) / pass_max.min(stop_max).max(1e-12);
        assert!(
            ratio < 1.5,
            "pass {pass_max} vs stop {stop_max} ripple imbalance"
        );
    }

    #[test]
    fn test_remez_even_taps_error() {
        assert!(remez_lowpass(20, 0.2, 0.3, 1.0, 1.0).is_err());
    }

    #[test]
    fn test_remez_empty_bands_error() {
        assert!(remez(21, &[]).is_err());
    }

    #[test]
    fn test_remez_edge_out_of_range_error() {
        assert!(RemezBand::new(0.0, 0.7, 1.0, 1.0).is_err());
        assert!(RemezBand::new(-0.1, 0.3, 1.0, 1.0).is_err());
    }

    #[test]
    fn test_remez_band_low_ge_high_error() {
        assert!(RemezBand::new(0.3, 0.2, 1.0, 1.0).is_err());
    }

    #[test]
    fn test_remez_overlapping_bands_error() {
        let bands = [
            RemezBand {
                low: 0.0,
                high: 0.3,
                desired: 1.0,
                weight: 1.0,
            },
            RemezBand {
                low: 0.2,
                high: 0.5,
                desired: 0.0,
                weight: 1.0,
            },
        ];
        assert!(remez(21, &bands).is_err());
    }

    #[test]
    fn test_remez_too_few_taps_error() {
        assert!(remez_lowpass(1, 0.2, 0.3, 1.0, 1.0).is_err());
    }
}
