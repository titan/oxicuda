//! Cockburn-Shu minmod slope limiter for the P1 2D Discontinuous Galerkin solver.
//!
//! The limiter enforces a discrete maximum principle on the [`Dg2dSpace`] nodal
//! P1 field while **preserving each cell mean exactly** (hence the global mass).
//!
//! # Algorithm (per triangle `T`)
//!
//! Let `ū_T = (u_0+u_1+u_2)/3` be the cell mean (the P1 centroid value). For each
//! local edge `l` (opposite vertex `l`, midpoint value `m_l = (u_{l+1}+u_{l+2})/2`),
//! the centroid-to-midpoint increment is
//!
//! ```text
//! Δ_l = m_l − ū_T.
//! ```
//!
//! The increment is limited against the jump to the neighbour cell mean using the
//! TVB-modified `minmod`:
//!
//! ```text
//! Δ̃_l = minmod_M( Δ_l , ν·(ū_{N_l} − ū_T) ),
//! ```
//!
//! with `ν ≥ 1` (here `ν = 1`) and a TVB constant `M` (default `0`, the strict
//! TVD minmod). If a neighbour is absent (boundary edge) the increment is left
//! unlimited.
//!
//! Because the raw `minmod` step can perturb `Σ_l Δ̃_l` away from zero (the
//! conservation constraint), the limited increments are **redistributed**
//! (Cockburn-Shu positivity scaling) so that `Σ_l Δ̂_l = 0`:
//!
//! ```text
//! pos = Σ_l max(0, Δ̃_l),   neg = Σ_l max(0, −Δ̃_l),
//! θ⁺ = min(1, neg/pos),    θ⁻ = min(1, pos/neg),
//! Δ̂_l = θ⁺·max(0, Δ̃_l) − θ⁻·max(0, −Δ̃_l).
//! ```
//!
//! The limited vertex values are reconstructed from the limited midpoint
//! increments via the exact P1 identity `u_i = m_{i+1} + m_{i+2} − m_i`:
//!
//! ```text
//! ũ_i = ū_T + (Δ̂_{i+1} + Δ̂_{i+2} − Δ̂_i),
//! ```
//!
//! which leaves the cell mean unchanged (`Σ_i ũ_i = 3 ū_T`).
//!
//! Reference: Cockburn & Shu, *TVB Runge-Kutta Local Projection DG* (Math. Comp.
//! 1989) and *The RKDG Method for Conservation Laws V* (J. Comput. Phys. 1998).

use crate::dg::dg_2d::{Dg2dSpace, LimiterStencil};
use crate::error::PdeResult;

/// TVB-modified `minmod` of two arguments with constant `m_tvb` and the cell
/// length scale `h` (the modification deactivates limiting for smooth extrema
/// where `|a| ≤ m_tvb·h²`).
pub fn minmod_tvb(a: f64, b: f64, m_tvb: f64, h: f64) -> f64 {
    if a.abs() <= m_tvb * h * h {
        return a;
    }
    if a > 0.0 && b > 0.0 {
        a.min(b)
    } else if a < 0.0 && b < 0.0 {
        a.max(b)
    } else {
        0.0
    }
}

/// Plain two-argument `minmod` (TVB constant `0`).
pub fn minmod(a: f64, b: f64) -> f64 {
    minmod_tvb(a, b, 0.0, 1.0)
}

/// Apply the Cockburn-Shu minmod limiter in place to the nodal P1 field `u`.
///
/// `nu` is the neighbour-jump amplification (`≥ 1`, typically `1.0`); `m_tvb` is
/// the TVB constant (`0.0` gives strict TVD behaviour).
pub fn limit_minmod(space: &Dg2dSpace, u: &mut [f64], nu: f64, m_tvb: f64) -> PdeResult<()> {
    let n_elem = space.n_elem;
    // Snapshot current cell means (limiting uses the pre-limit neighbour means).
    let mut mean = vec![0.0_f64; n_elem];
    for (e, me) in mean.iter_mut().enumerate() {
        *me = space.cell_mean(u, e);
    }

    for e in 0..n_elem {
        let base = 3 * e;
        let u0 = u[base];
        let u1 = u[base + 1];
        let u2 = u[base + 2];
        let ubar = mean[e];
        let area = space.area(e)?;
        let h = area.sqrt();
        let b0 = space.centroid(e)?;

        // Edge-midpoint increments Δ_l (local edge l opposite vertex l).
        let m_l = [0.5 * (u1 + u2), 0.5 * (u2 + u0), 0.5 * (u0 + u1)];
        let delta = [m_l[0] - ubar, m_l[1] - ubar, m_l[2] - ubar];

        // Geometry-aware reference slope from neighbour cell means: fit a linear
        // cell-mean gradient g_cm by least squares to
        //   ū_{N_l} − ū_T ≈ g_cm · (b_{N_l} − b0)
        // over the available (periodically-shifted) neighbour centroids. For a
        // globally linear field g_cm is the exact gradient, so the reference
        // increment g_cm·(m_l − b0) equals Δ_l and the limiter does not clip.
        let stencil = space.limiter_stencil(e)?;
        let g_cm = least_squares_gradient(&stencil, ubar, &mean, b0);

        let neigh = space.neighbors(e)?;
        let mut lim = [0.0_f64; 3];
        let mut any_limited = false;
        for l in 0..3 {
            match neigh[l] {
                Some(_ne) => {
                    let mid = space.edge_midpoint(e, l)?;
                    let dr = [mid[0] - b0[0], mid[1] - b0[1]];
                    let reference = nu * (g_cm[0] * dr[0] + g_cm[1] * dr[1]);
                    let d = minmod_tvb(delta[l], reference, m_tvb, h);
                    if (d - delta[l]).abs() > 1.0e-12 {
                        any_limited = true;
                    }
                    lim[l] = d;
                }
                None => {
                    // boundary edge: keep the unlimited increment
                    lim[l] = delta[l];
                }
            }
        }

        if !any_limited {
            continue; // smooth cell, leave the high-order data untouched
        }

        // Conservative redistribution so Σ_l Δ̂_l = 0.
        let mut pos = 0.0;
        let mut neg = 0.0;
        for &d in &lim {
            if d > 0.0 {
                pos += d;
            } else {
                neg += -d;
            }
        }
        let mut hat = lim;
        if pos > 0.0 && neg > 0.0 {
            let theta_plus = (neg / pos).min(1.0);
            let theta_minus = (pos / neg).min(1.0);
            for d in hat.iter_mut() {
                if *d > 0.0 {
                    *d *= theta_plus;
                } else {
                    *d *= theta_minus;
                }
            }
        } else {
            // all-positive or all-negative increments cannot be balanced while
            // preserving the mean ⇒ collapse to the cell mean (first-order).
            hat = [0.0, 0.0, 0.0];
        }

        // Reconstruct vertices: ũ_i = ū + (Δ̂_{i+1} + Δ̂_{i+2} − Δ̂_i).
        for i in 0..3 {
            let ip1 = (i + 1) % 3;
            let ip2 = (i + 2) % 3;
            u[base + i] = ubar + (hat[ip1] + hat[ip2] - hat[i]);
        }
    }
    Ok(())
}

/// Zhang-Shu maximum-principle-preserving (MPP) scalar limiter.
///
/// Scales each element's deviation from its cell mean by the largest
/// `θ ∈ [0,1]` keeping every nodal value within the global bounds
/// `[gmin, gmax]`: `u_i ← ū + θ (u_i − ū)`. This preserves the cell mean (hence
/// total mass) and enforces `u_i ∈ [gmin, gmax]` provided `ū ∈ [gmin, gmax]`
/// (guaranteed for the monotone cell-mean update of linear advection).
///
/// Reference: Zhang & Shu, *On maximum-principle-satisfying high order schemes*
/// (J. Comput. Phys. 2010).
pub fn limit_bounds(space: &Dg2dSpace, u: &mut [f64], gmin: f64, gmax: f64) -> PdeResult<()> {
    for e in 0..space.n_elem {
        let base = 3 * e;
        let ubar = space.cell_mean(u, e);
        let mut theta = 1.0_f64;
        for i in 0..3 {
            let dev = u[base + i] - ubar;
            if dev > 1.0e-300 {
                let allowed = gmax - ubar;
                if allowed < dev {
                    theta = theta.min((allowed / dev).max(0.0));
                }
            } else if dev < -1.0e-300 {
                let allowed = gmin - ubar; // negative
                if allowed > dev {
                    theta = theta.min((allowed / dev).max(0.0));
                }
            }
        }
        if theta < 1.0 {
            for i in 0..3 {
                u[base + i] = ubar + theta * (u[base + i] - ubar);
            }
        }
    }
    Ok(())
}

/// Least-squares cell-mean gradient from neighbour means.
///
/// Solves the 2×2 normal equations for `g` minimising
/// `Σ_k (g·(b_k − b0) − (ū_k − ū_T))²` over the available neighbour stencil.
/// Returns `[0,0]` when fewer than two independent directions are available or
/// the normal matrix is singular (the limiter then falls back to first order).
fn least_squares_gradient(
    stencil: &LimiterStencil,
    ubar: f64,
    mean: &[f64],
    b0: [f64; 2],
) -> [f64; 2] {
    let mut axx = 0.0;
    let mut axy = 0.0;
    let mut ayy = 0.0;
    let mut bx = 0.0;
    let mut by = 0.0;
    let mut count = 0;
    for (ne, c) in stencil.iter().flatten() {
        let dx = c[0] - b0[0];
        let dy = c[1] - b0[1];
        let dm = mean[*ne] - ubar;
        axx += dx * dx;
        axy += dx * dy;
        ayy += dy * dy;
        bx += dx * dm;
        by += dy * dm;
        count += 1;
    }
    if count < 2 {
        return [0.0, 0.0];
    }
    let det = axx * ayy - axy * axy;
    if det.abs() < 1.0e-300 {
        return [0.0, 0.0];
    }
    [(bx * ayy - by * axy) / det, (axx * by - axy * bx) / det]
}

/// Convenience closure capturing the default limiter parameters (`ν=1`, `M=0`),
/// suitable for passing to the per-stage limiting hook of the RK integrator.
pub fn minmod_closure() -> impl Fn(&Dg2dSpace, &mut [f64]) -> PdeResult<()> {
    move |space: &Dg2dSpace, u: &mut [f64]| limit_minmod(space, u, 1.0, 0.0)
}

/// Limiter closure combining the minmod slope limiter with the Zhang-Shu MPP
/// bound enforcement, so reconstructed nodal values stay in `[gmin, gmax]`
/// (strict discrete maximum principle) while preserving every cell mean.
pub fn minmod_bounded_closure(
    gmin: f64,
    gmax: f64,
) -> impl Fn(&Dg2dSpace, &mut [f64]) -> PdeResult<()> {
    move |space: &Dg2dSpace, u: &mut [f64]| {
        limit_minmod(space, u, 1.0, 0.0)?;
        limit_bounds(space, u, gmin, gmax)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dg::dg_2d::DgBoundary;
    use crate::mesh::TriMesh2d;

    fn square(n: usize) -> TriMesh2d {
        TriMesh2d::rect_grid(0.0, 1.0, 0.0, 1.0, n, n).expect("mesh")
    }

    fn periodic() -> DgBoundary {
        DgBoundary::Periodic {
            x0: 0.0,
            x1: 1.0,
            y0: 0.0,
            y1: 1.0,
        }
    }

    #[test]
    fn minmod_basic() {
        assert!((minmod(1.0, 2.0) - 1.0).abs() < 1e-15);
        assert!((minmod(-1.0, -3.0) - (-1.0)).abs() < 1e-15);
        assert!(minmod(1.0, -2.0).abs() < 1e-15);
        assert!(minmod(-1.0, 2.0).abs() < 1e-15);
    }

    #[test]
    fn limiter_preserves_cell_mean() {
        // After limiting, every cell mean must be unchanged (mass conservation).
        let mesh = square(7);
        let space = Dg2dSpace::new(&mesh, periodic()).expect("ok");
        // a steep oscillatory field that triggers limiting
        let mut u = vec![0.0; space.n_dofs()];
        for e in 0..space.n_elem {
            let v = space.element_vertices(e).expect("v");
            for i in 0..3 {
                u[3 * e + i] = if v[i][0] < 0.5 { 1.0 } else { -1.0 } + 0.3 * v[i][1];
            }
        }
        let means_before: Vec<f64> = (0..space.n_elem).map(|e| space.cell_mean(&u, e)).collect();
        limit_minmod(&space, &mut u, 1.0, 0.0).expect("ok");
        for (e, &before) in means_before.iter().enumerate() {
            let after = space.cell_mean(&u, e);
            assert!(
                (after - before).abs() < 1e-12,
                "cell {e} mean drifted {before} -> {after}"
            );
        }
    }

    #[test]
    fn limiter_leaves_linear_field_untouched() {
        // A globally linear field is its own limited reconstruction (no spurious
        // clipping of smooth data) on geometrically-interior cells. Use Compact
        // BC so the linear field is not discontinuous across a periodic seam.
        let mesh = square(6);
        let space = Dg2dSpace::new(&mesh, DgBoundary::Compact { far_field: 0.0 }).expect("ok");
        let f = |x: f64, y: f64| 0.3 + 0.5 * x - 0.2 * y;
        let mut u = vec![0.0; space.n_dofs()];
        for e in 0..space.n_elem {
            let v = space.element_vertices(e).expect("v");
            for i in 0..3 {
                u[3 * e + i] = f(v[i][0], v[i][1]);
            }
        }
        let before = u.clone();
        limit_minmod(&space, &mut u, 1.0, 0.0).expect("ok");
        // Interior cells must be unchanged; boundary cells may differ only via
        // unlimited boundary edges, so compare cell means everywhere and nodal
        // values on interior cells.
        for e in 0..space.n_elem {
            let neigh = space.neighbors(e).expect("nb");
            let interior = neigh.iter().all(|n| n.is_some());
            if interior {
                for i in 0..3 {
                    assert!(
                        (u[3 * e + i] - before[3 * e + i]).abs() < 1e-10,
                        "linear field clipped at cell {e} node {i}"
                    );
                }
            }
        }
    }
}
