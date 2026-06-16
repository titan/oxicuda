//! Periodic boundary condition losses and hard periodic feature embedding.
//!
//! Many PDEs (advection, Burgers on a ring, Kuramoto-Sivashinsky, spectral
//! benchmarks) are posed on a periodic domain `x ∈ [a, b]` with `u(a) = u(b)`.
//! There are two standard ways to impose periodicity in a PINN:
//!
//! 1. **Soft (penalty) enforcement** — add boundary loss terms that match the
//!    solution and (optionally) its first derivative on the two periodic faces:
//!    ```text
//!    L_periodic = (1/M) Σ_m [ (u(a, t_m) − u(b, t_m))²
//!                            + (u_x(a, t_m) − u_x(b, t_m))² ] .
//!    ```
//!    Matching the derivative as well as the value gives `C¹` periodicity, which
//!    is what high-order PDEs require.
//!
//! 2. **Hard (exact) enforcement** — replace the raw coordinate `x` by a periodic
//!    Fourier feature map so the network is periodic *by construction* and no
//!    boundary loss is needed (Dong & Ni 2021; the "periodic-embedding" trick):
//!    ```text
//!    x  ↦  [ cos(2π k (x−a)/L), sin(2π k (x−a)/L) ]_{k=1..K},   L = b − a .
//!    ```
//!    Every component is `L`-periodic, so any downstream MLP of these features is
//!    automatically periodic with period `L`.
//!
//! This module provides both: [`periodic_bc_loss`] / [`periodic_bc_loss_value`]
//! for the soft penalty, and [`PeriodicEmbedding`] for the hard embedding.

use crate::error::{PinnError, PinnResult};

/// Soft periodic boundary loss matching **value and first derivative** on the two
/// periodic faces.
///
/// All slices have length `M` (number of paired boundary samples):
/// - `u_left[m]`, `u_right[m]`     : `u` at `x = a` and `x = b` for sample `m`,
/// - `ux_left[m]`, `ux_right[m]`   : `∂u/∂x` at `x = a` and `x = b`.
///
/// Returns `(1/M) Σ_m (u_L − u_R)² + (u_x,L − u_x,R)²`.
///
/// # Errors
/// - [`PinnError::EmptyCollocationSet`] if `u_left` is empty.
/// - [`PinnError::DimensionMismatch`] if any slice length differs from `u_left`.
/// - [`PinnError::NanEncountered`] on non-finite input / result.
pub fn periodic_bc_loss(
    u_left: &[f32],
    u_right: &[f32],
    ux_left: &[f32],
    ux_right: &[f32],
) -> PinnResult<f32> {
    let m = u_left.len();
    if m == 0 {
        return Err(PinnError::EmptyCollocationSet);
    }
    for slice in [u_right, ux_left, ux_right] {
        if slice.len() != m {
            return Err(PinnError::DimensionMismatch {
                expected: m,
                got: slice.len(),
            });
        }
    }
    if [u_left, u_right, ux_left, ux_right]
        .iter()
        .any(|s| s.iter().any(|v| !v.is_finite()))
    {
        return Err(PinnError::NanEncountered {
            location: "periodic_bc_loss(input)",
        });
    }

    let mut acc = 0.0_f32;
    for m_i in 0..m {
        let dv = u_left[m_i] - u_right[m_i];
        let dd = ux_left[m_i] - ux_right[m_i];
        acc += dv * dv + dd * dd;
    }
    let loss = acc / m as f32;
    if !loss.is_finite() {
        return Err(PinnError::NanEncountered {
            location: "periodic_bc_loss(output)",
        });
    }
    Ok(loss)
}

/// Soft periodic boundary loss matching **value only** (`C⁰` periodicity):
/// `(1/M) Σ_m (u_L − u_R)²`.
///
/// # Errors
/// - [`PinnError::EmptyCollocationSet`] if `u_left` is empty.
/// - [`PinnError::DimensionMismatch`] if `u_right.len() != u_left.len()`.
/// - [`PinnError::NanEncountered`] on non-finite input / result.
pub fn periodic_bc_loss_value(u_left: &[f32], u_right: &[f32]) -> PinnResult<f32> {
    let m = u_left.len();
    if m == 0 {
        return Err(PinnError::EmptyCollocationSet);
    }
    if u_right.len() != m {
        return Err(PinnError::DimensionMismatch {
            expected: m,
            got: u_right.len(),
        });
    }
    if u_left.iter().chain(u_right).any(|v| !v.is_finite()) {
        return Err(PinnError::NanEncountered {
            location: "periodic_bc_loss_value(input)",
        });
    }
    let loss = u_left
        .iter()
        .zip(u_right.iter())
        .map(|(&l, &r)| (l - r) * (l - r))
        .sum::<f32>()
        / m as f32;
    if !loss.is_finite() {
        return Err(PinnError::NanEncountered {
            location: "periodic_bc_loss_value(output)",
        });
    }
    Ok(loss)
}

/// Hard periodic Fourier-feature embedding for exact periodicity by construction.
///
/// Maps a scalar coordinate `x ∈ [a, b]` (period `L = b − a`) to
/// `[cos(θ₁), sin(θ₁), …, cos(θ_K), sin(θ_K)]` where `θ_k = 2π k (x − a)/L`.
/// The output dimension is `2·K`. Every feature is `L`-periodic, so an MLP
/// applied to the embedding is automatically periodic.
#[derive(Debug, Clone)]
pub struct PeriodicEmbedding {
    /// Lower bound `a` of the periodic domain.
    domain_lo: f32,
    /// Period `L = b − a`.
    period: f32,
    /// Number of Fourier modes `K` (output dim is `2K`).
    n_modes: usize,
}

impl PeriodicEmbedding {
    /// Create a periodic embedding for `[domain_lo, domain_hi]` with `n_modes`
    /// Fourier modes.
    ///
    /// # Errors
    /// - [`PinnError::InvalidTimeInterval`] if `domain_hi <= domain_lo`.
    /// - [`PinnError::InvalidLayerWidth`] if `n_modes == 0`.
    pub fn new(domain_lo: f32, domain_hi: f32, n_modes: usize) -> PinnResult<Self> {
        if !domain_lo.is_finite() || !domain_hi.is_finite() || domain_hi <= domain_lo {
            return Err(PinnError::InvalidTimeInterval {
                t0: domain_lo,
                t1: domain_hi,
            });
        }
        if n_modes == 0 {
            return Err(PinnError::InvalidLayerWidth);
        }
        Ok(Self {
            domain_lo,
            period: domain_hi - domain_lo,
            n_modes,
        })
    }

    /// Output dimension `2·n_modes`.
    #[must_use]
    pub fn output_dim(&self) -> usize {
        2 * self.n_modes
    }

    /// Embed a single coordinate `x` into `[cos θ_k, sin θ_k]_{k=1..K}`.
    #[must_use]
    pub fn embed(&self, x: f32) -> Vec<f32> {
        let base = 2.0 * std::f32::consts::PI * (x - self.domain_lo) / self.period;
        let mut out = Vec::with_capacity(self.output_dim());
        for k in 1..=self.n_modes {
            let theta = base * k as f32;
            out.push(theta.cos());
            out.push(theta.sin());
        }
        out
    }

    /// Embed a batch of `n` coordinates, returning an `n * output_dim` row-major
    /// matrix.
    ///
    /// # Errors
    /// - [`PinnError::EmptyInput`] if `xs` is empty.
    pub fn embed_batch(&self, xs: &[f32]) -> PinnResult<Vec<f32>> {
        if xs.is_empty() {
            return Err(PinnError::EmptyInput);
        }
        let d = self.output_dim();
        let mut out = Vec::with_capacity(xs.len() * d);
        for &x in xs {
            out.extend(self.embed(x));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn periodic_loss_zero_when_matched() {
        let u_l = vec![0.1, 0.2, 0.3];
        let u_r = vec![0.1, 0.2, 0.3];
        let ux_l = vec![1.0, 1.0, 1.0];
        let ux_r = vec![1.0, 1.0, 1.0];
        let loss = periodic_bc_loss(&u_l, &u_r, &ux_l, &ux_r)
            .expect("periodic boundary condition computation should succeed for matched inputs");
        assert!(loss < 1e-9, "matched periodicity → zero loss, got {loss}");
    }

    #[test]
    fn periodic_loss_value_formula() {
        // (1−0)² + (2−0)² over M=2 → (1+4)/2 = 2.5
        let loss = periodic_bc_loss_value(&[1.0, 2.0], &[0.0, 0.0])
            .expect("periodic boundary condition value loss should succeed for valid inputs");
        assert!(approx(loss, 2.5, 1e-6), "got {loss}");
    }

    #[test]
    fn periodic_loss_includes_derivative() {
        // value matched, derivative mismatched: (2−0)² averaged over M=1 = 4
        let loss = periodic_bc_loss(&[0.5], &[0.5], &[2.0], &[0.0]).expect("periodic boundary condition computation should succeed for value-matched derivative-mismatched inputs");
        assert!(approx(loss, 4.0, 1e-6), "got {loss}");
    }

    #[test]
    fn periodic_loss_nonnegative() {
        let loss = periodic_bc_loss(&[-1.0, 3.0], &[2.0, -1.0], &[0.0, 1.0], &[1.0, 0.0]).expect(
            "periodic boundary condition computation should succeed for valid finite inputs",
        );
        assert!(loss >= 0.0);
    }

    #[test]
    fn periodic_loss_empty_errors() {
        assert!(matches!(
            periodic_bc_loss(&[], &[], &[], &[]),
            Err(PinnError::EmptyCollocationSet)
        ));
        assert!(matches!(
            periodic_bc_loss_value(&[], &[]),
            Err(PinnError::EmptyCollocationSet)
        ));
    }

    #[test]
    fn periodic_loss_dim_mismatch_errors() {
        assert!(matches!(
            periodic_bc_loss(&[1.0, 2.0], &[1.0], &[0.0, 0.0], &[0.0, 0.0]),
            Err(PinnError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            periodic_bc_loss_value(&[1.0, 2.0], &[1.0]),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn periodic_loss_nan_errors() {
        assert!(matches!(
            periodic_bc_loss(&[f32::NAN], &[0.0], &[0.0], &[0.0]),
            Err(PinnError::NanEncountered { .. })
        ));
    }

    #[test]
    fn embedding_output_dim() {
        let emb = PeriodicEmbedding::new(0.0, 1.0, 4)
            .expect("PeriodicEmbedding construction with valid params should succeed");
        assert_eq!(emb.output_dim(), 8);
        assert_eq!(emb.embed(0.3).len(), 8);
    }

    #[test]
    fn embedding_is_periodic() {
        // Feature at x and x + L must be identical (exact periodicity).
        let emb = PeriodicEmbedding::new(0.0, 2.0, 3)
            .expect("PeriodicEmbedding construction with valid params should succeed");
        let a = emb.embed(0.4);
        let b = emb.embed(0.4 + 2.0); // + one period
        for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
            assert!(approx(va, vb, 1e-4), "feature {i}: {va} vs {vb}");
        }
    }

    #[test]
    fn embedding_endpoints_match() {
        // u(a) features == u(b) features since b = a + L.
        let emb = PeriodicEmbedding::new(-1.0, 1.0, 5)
            .expect("PeriodicEmbedding construction with valid params should succeed");
        let left = emb.embed(-1.0);
        let right = emb.embed(1.0);
        for (&l, &r) in left.iter().zip(right.iter()) {
            assert!(approx(l, r, 1e-4));
        }
    }

    #[test]
    fn embedding_values_bounded() {
        // cos/sin → all features in [-1, 1].
        let emb = PeriodicEmbedding::new(0.0, 1.0, 6)
            .expect("PeriodicEmbedding construction with valid params should succeed");
        for i in 0..20 {
            let x = i as f32 * 0.05;
            for &v in &emb.embed(x) {
                assert!((-1.0..=1.0).contains(&v));
            }
        }
    }

    #[test]
    fn embedding_batch_shape() {
        let emb = PeriodicEmbedding::new(0.0, 1.0, 3)
            .expect("PeriodicEmbedding construction with valid params should succeed");
        let xs = vec![0.0, 0.25, 0.5, 0.75];
        let out = emb
            .embed_batch(&xs)
            .expect("embed_batch should succeed for non-empty input");
        assert_eq!(out.len(), 4 * emb.output_dim());
    }

    #[test]
    fn embedding_invalid_config_errors() {
        assert!(PeriodicEmbedding::new(1.0, 1.0, 3).is_err());
        assert!(PeriodicEmbedding::new(0.0, 1.0, 0).is_err());
        assert!(matches!(
            PeriodicEmbedding::new(0.0, 1.0, 2)
                .expect("PeriodicEmbedding construction with valid params should succeed")
                .embed_batch(&[]),
            Err(PinnError::EmptyInput)
        ));
    }
}
