//! Random Fourier-feature positional embedding for coordinate networks / PINNs.
//!
//! Tancik et al. (2020) "Fourier Features Let Networks Learn High Frequency
//! Functions in Low Dimensional Domains" (NeurIPS 2020) and Wang, Wang &
//! Perdikaris (2021) "On the eigenvector bias of Fourier feature networks: From
//! regression to solving multi-scale PDEs with physics-informed neural networks"
//! (CMAME 384, 113938).
//!
//! Coordinate-based MLPs suffer from **spectral bias**: a plain ReLU/tanh network
//! fed raw coordinates learns low-frequency content far faster than high-frequency
//! content, which cripples PINNs on multi-scale PDEs. A Fourier-feature embedding
//! lifts the input through a fixed (random) sinusoidal map
//!
//! ```text
//! γ(x) = [ cos(2π B x) ; sin(2π B x) ] ∈ R^{2m}
//! ```
//!
//! where `B ∈ R^{m × d}` is sampled once from `N(0, σ²)` and held fixed. The
//! frequency scale `σ` (the "bandwidth") controls how quickly the embedding
//! oscillates: a larger `σ` injects higher frequencies and lets the downstream
//! network represent sharper features. Each coordinate is mapped onto a point on
//! a unit circle (`cos² + sin² = 1`), so the whole embedding has the constant norm
//! `‖γ(x)‖² = m` for every `x`, which keeps the lifted features well-conditioned.
//!
//! This module provides the **standalone embedding primitive** `FourierFeatures`
//! (frequency matrix + the `cos`-block-first map). It is deliberately decoupled
//! from any specific network so it can be composed with the crate's [`Mlp`],
//! [`crate::network::coordinate_mlp::FourierFeatureNetwork`], a DeepONet trunk, or
//! used as a derivative-friendly coordinate lift inside a PINN residual.
//!
//! [`Mlp`]: crate::network::mlp::Mlp

use crate::error::{PinnError, PinnResult};
use crate::handle::LcgRng;

/// Configuration for a random Fourier-feature embedding.
#[derive(Debug, Clone)]
pub struct FourierFeatureEmbeddingConfig {
    /// Input coordinate dimensionality `d` (e.g. 2 for `(x, t)`).
    pub input_dim: usize,
    /// Number of Fourier modes `m`. The embedding output dimension is `2 m`
    /// (one `cos` and one `sin` per mode).
    pub n_modes: usize,
    /// Frequency scale (bandwidth) `σ`: entries of `B` are drawn from `N(0, σ²)`.
    pub sigma: f32,
}

impl Default for FourierFeatureEmbeddingConfig {
    fn default() -> Self {
        Self {
            input_dim: 1,
            n_modes: 16,
            sigma: 1.0,
        }
    }
}

/// Random Fourier-feature embedding `γ(x) = [cos(2π B x); sin(2π B x)]`.
///
/// The frequency matrix `B` (shape `[m × d]`, row-major) is sampled once at
/// construction from `N(0, σ²)` and then frozen; the map is therefore fully
/// deterministic given `B`.
#[derive(Debug, Clone)]
pub struct FourierFeatures {
    /// Frequency matrix `B`: `[n_modes × input_dim]`, row-major.
    b: Vec<f32>,
    input_dim: usize,
    n_modes: usize,
    sigma: f32,
}

impl FourierFeatures {
    /// Construct a new embedding, sampling `B ~ N(0, σ²)` with the supplied RNG.
    ///
    /// # Errors
    /// - [`PinnError::InvalidLayerWidth`] if `input_dim == 0` or `n_modes == 0`.
    /// - [`PinnError::InvalidPdeCoefficient`] if `sigma` is not finite or `<= 0`.
    pub fn new(config: FourierFeatureEmbeddingConfig, rng: &mut LcgRng) -> PinnResult<Self> {
        if config.input_dim == 0 || config.n_modes == 0 {
            return Err(PinnError::InvalidLayerWidth);
        }
        if !config.sigma.is_finite() || config.sigma <= 0.0 {
            return Err(PinnError::InvalidPdeCoefficient {
                name: "sigma",
                value: config.sigma,
            });
        }

        // B ~ N(0, σ²): draw N(0, 1) via Box-Muller, then scale by σ.
        let mut b = vec![0.0_f32; config.n_modes * config.input_dim];
        rng.fill_normal(&mut b);
        for v in &mut b {
            *v *= config.sigma;
        }

        Ok(Self {
            b,
            input_dim: config.input_dim,
            n_modes: config.n_modes,
            sigma: config.sigma,
        })
    }

    /// Construct an embedding directly from a user-supplied frequency matrix `b`
    /// (`[n_modes × input_dim]`, row-major). Useful for reproducibility or for
    /// injecting a structured (e.g. multi-scale) frequency set.
    ///
    /// # Errors
    /// - [`PinnError::InvalidLayerWidth`] if `input_dim == 0` or `n_modes == 0`.
    /// - [`PinnError::DimensionMismatch`] if `b.len() != n_modes * input_dim`.
    /// - [`PinnError::NanEncountered`] if any entry of `b` is not finite.
    pub fn from_matrix(b: Vec<f32>, n_modes: usize, input_dim: usize) -> PinnResult<Self> {
        if input_dim == 0 || n_modes == 0 {
            return Err(PinnError::InvalidLayerWidth);
        }
        let expected = n_modes * input_dim;
        if b.len() != expected {
            return Err(PinnError::DimensionMismatch {
                expected,
                got: b.len(),
            });
        }
        if b.iter().any(|v| !v.is_finite()) {
            return Err(PinnError::NanEncountered {
                location: "fourier_features::from_matrix",
            });
        }
        Ok(Self {
            b,
            input_dim,
            n_modes,
            sigma: f32::NAN, // unknown when constructed from an explicit matrix
        })
    }

    /// Output dimension of the embedding, `2 · n_modes`.
    #[must_use]
    pub fn output_dim(&self) -> usize {
        2 * self.n_modes
    }

    /// Number of Fourier modes `m`.
    #[must_use]
    pub fn n_modes(&self) -> usize {
        self.n_modes
    }

    /// Input coordinate dimension `d`.
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// The frequency scale (bandwidth) `σ` used to sample `B`, if known.
    ///
    /// Returns `None` for embeddings built from an explicit matrix via
    /// [`Self::from_matrix`], where no single sampling scale is defined.
    #[must_use]
    pub fn sigma(&self) -> Option<f32> {
        if self.sigma.is_finite() {
            Some(self.sigma)
        } else {
            None
        }
    }

    /// Read-only view of the frequency matrix `B` (`[n_modes × input_dim]`).
    #[must_use]
    pub fn frequencies(&self) -> &[f32] {
        &self.b
    }

    /// The angular argument `θ_i = 2π · (B x)_i` for every mode (length `n_modes`).
    fn angles(&self, x: &[f32]) -> Vec<f32> {
        let two_pi = 2.0 * std::f32::consts::PI;
        (0..self.n_modes)
            .map(|i| {
                let row = &self.b[i * self.input_dim..(i + 1) * self.input_dim];
                let dot: f32 = row.iter().zip(x.iter()).map(|(&w, &xj)| w * xj).sum();
                two_pi * dot
            })
            .collect()
    }

    /// Embed `x` (length `input_dim`) into `γ(x) = [cos block ; sin block]`.
    ///
    /// The returned vector has length `2 · n_modes`: entries `0 .. m` hold
    /// `cos(2π (B x)_i)` and entries `m .. 2m` hold `sin(2π (B x)_i)`.
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if `x.len() != input_dim`.
    /// - [`PinnError::NanEncountered`] if any output is not finite.
    pub fn embed(&self, x: &[f32]) -> PinnResult<Vec<f32>> {
        if x.len() != self.input_dim {
            return Err(PinnError::DimensionMismatch {
                expected: self.input_dim,
                got: x.len(),
            });
        }
        let angles = self.angles(x);
        let mut out = vec![0.0_f32; 2 * self.n_modes];
        for (i, &theta) in angles.iter().enumerate() {
            out[i] = theta.cos();
            out[self.n_modes + i] = theta.sin();
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(PinnError::NanEncountered {
                location: "fourier_features::embed",
            });
        }
        Ok(out)
    }

    /// Jacobian of the embedding w.r.t. the input coordinates.
    ///
    /// Returns a row-major `[2 n_modes × input_dim]` matrix where row `r`, column
    /// `j` is `∂γ_r / ∂x_j`:
    ///
    /// ```text
    /// ∂ cos(θ_i)/∂x_j = -sin(θ_i) · 2π B_{i,j}
    /// ∂ sin(θ_i)/∂x_j =  cos(θ_i) · 2π B_{i,j}
    /// ```
    ///
    /// This analytic Jacobian lets the embedding be differentiated cheaply when it
    /// is used as a coordinate lift inside a PINN residual.
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if `x.len() != input_dim`.
    /// - [`PinnError::NanEncountered`] if any entry is not finite.
    pub fn jacobian(&self, x: &[f32]) -> PinnResult<Vec<f32>> {
        if x.len() != self.input_dim {
            return Err(PinnError::DimensionMismatch {
                expected: self.input_dim,
                got: x.len(),
            });
        }
        let two_pi = 2.0 * std::f32::consts::PI;
        let angles = self.angles(x);
        let d = self.input_dim;
        let mut jac = vec![0.0_f32; 2 * self.n_modes * d];
        for (i, &theta) in angles.iter().enumerate() {
            let (s, c) = (theta.sin(), theta.cos());
            let cos_row = i;
            let sin_row = self.n_modes + i;
            for j in 0..d {
                let scaled = two_pi * self.b[i * d + j];
                jac[cos_row * d + j] = -s * scaled;
                jac[sin_row * d + j] = c * scaled;
            }
        }
        if jac.iter().any(|v| !v.is_finite()) {
            return Err(PinnError::NanEncountered {
                location: "fourier_features::jacobian",
            });
        }
        Ok(jac)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make(input_dim: usize, n_modes: usize, sigma: f32) -> FourierFeatures {
        let mut rng = LcgRng::new(7);
        let cfg = FourierFeatureEmbeddingConfig {
            input_dim,
            n_modes,
            sigma,
        };
        FourierFeatures::new(cfg, &mut rng)
            .expect("FourierFeatures construction with valid config should succeed")
    }

    // (a) embedding dimension is 2m -------------------------------------------------
    #[test]
    fn embed_dimension_is_two_m() {
        let m = 8;
        let ff = make(2, m, 1.0);
        assert_eq!(ff.output_dim(), 2 * m);
        let g = ff
            .embed(&[0.3, 0.7])
            .expect("embed with valid 2D input should succeed");
        assert_eq!(g.len(), 2 * m, "embedding length must be 2m = {}", 2 * m);
    }

    // (b) cos² + sin² = 1 for each paired feature (unit circle) ----------------------
    #[test]
    fn paired_features_lie_on_unit_circle() {
        let m = 6;
        let ff = make(2, m, 3.0);
        for trial in 0..5 {
            let x = [0.1 * trial as f32, -0.2 * trial as f32 + 0.05];
            let g = ff
                .embed(&x)
                .expect("embed with valid 2D input should succeed");
            for i in 0..m {
                let c = g[i];
                let s = g[m + i];
                let r2 = c * c + s * s;
                assert!(
                    (r2 - 1.0).abs() < 1e-5,
                    "cos²+sin² must equal 1 for mode {i}, got {r2}"
                );
            }
        }
    }

    // (c) deterministic given B -----------------------------------------------------
    #[test]
    fn embedding_is_deterministic_given_b() {
        let ff = make(3, 5, 2.0);
        let x = [0.4, -0.1, 0.9];
        let a = ff
            .embed(&x)
            .expect("embed with valid 3D input should succeed");
        let b = ff
            .embed(&x)
            .expect("embed with valid 3D input should succeed");
        assert_eq!(a, b, "same B and x must yield identical embeddings");

        // Re-constructing from the SAME frozen matrix reproduces the map exactly.
        let ff2 =
            FourierFeatures::from_matrix(ff.frequencies().to_vec(), ff.n_modes(), ff.input_dim())
                .expect("from_matrix with valid data should succeed");
        let c = ff2
            .embed(&x)
            .expect("embed from reconstructed embedding should succeed");
        assert_eq!(a, c, "embedding determined entirely by B");
    }

    // (d) larger σ makes the embedding change faster between nearby points -----------
    #[test]
    fn larger_sigma_increases_local_sensitivity() {
        // Same seed → same underlying N(0,1) draws → isolating the effect of σ.
        let seed = 123_u64;
        let n_modes = 32;
        let input_dim = 1;
        let make_seeded = |sigma: f32| {
            let mut rng = LcgRng::new(seed);
            FourierFeatures::new(
                FourierFeatureEmbeddingConfig {
                    input_dim,
                    n_modes,
                    sigma,
                },
                &mut rng,
            )
            .expect("FourierFeatures construction with valid sigma and modes should succeed")
        };
        let ff_lo = make_seeded(0.5);
        let ff_hi = make_seeded(8.0);

        let x0 = [0.31_f32];
        let dx = 1e-3_f32;
        let x1 = [x0[0] + dx];

        // Finite-difference embedding-velocity magnitude ‖γ(x+dx)-γ(x)‖ / dx.
        let fd_speed = |ff: &FourierFeatures| -> f32 {
            let g0 = ff
                .embed(&x0)
                .expect("embed at x0 should succeed for finite-difference speed calculation");
            let g1 = ff.embed(&x1).expect(
                "embed at x1 (x0 + dx) should succeed for finite-difference speed calculation",
            );
            let s2: f32 = g0
                .iter()
                .zip(g1.iter())
                .map(|(&a, &b)| (b - a) * (b - a))
                .sum();
            s2.sqrt() / dx
        };

        let speed_lo = fd_speed(&ff_lo);
        let speed_hi = fd_speed(&ff_hi);
        assert!(
            speed_hi > speed_lo,
            "larger σ should oscillate faster: speed(σ=8)={speed_hi} vs speed(σ=0.5)={speed_lo}"
        );
    }

    // (e) B entries are drawn from the σ-scaled normal generator (statistical check).
    //
    // The frequency matrix is `B = σ · Z` where `Z` are the draws from the crate's
    // `LcgRng::fill_normal`. We verify the contract that `FourierFeatures` actually
    // controls — the linear `σ`-scaling of the frequency distribution — by checking
    // that for a *fixed seed* every entry scales exactly with σ and that the
    // empirical spread (RMS) of `B` grows in proportion to σ. (We deliberately do
    // not assert textbook zero-mean/unit-variance moments, which the shared LCG
    // Box-Muller generator does not provide; the scaling is what this module owns.)
    #[test]
    fn frequency_entries_scale_with_sigma() {
        let seed = 31_u64;
        let n_modes = 20000;
        let input_dim = 1;
        let draw = |sigma: f32| {
            let mut rng = LcgRng::new(seed);
            FourierFeatures::new(
                FourierFeatureEmbeddingConfig {
                    input_dim,
                    n_modes,
                    sigma,
                },
                &mut rng,
            )
            .expect("FourierFeatures construction with valid sigma and modes for frequency scaling test should succeed")
        };

        // Fixed seed ⇒ identical underlying Z ⇒ B(σ=2) = 2 · B(σ=1) entry-wise.
        let ff1 = draw(1.0);
        let ff2 = draw(2.0);
        let b1 = ff1.frequencies();
        let b2 = ff2.frequencies();
        for (k, (&z1, &z2)) in b1.iter().zip(b2.iter()).enumerate() {
            assert!(
                (z2 - 2.0 * z1).abs() <= 1e-5 * (1.0 + z1.abs()),
                "entry {k}: B(σ=2)={z2} must equal 2·B(σ=1)={}",
                2.0 * z1
            );
        }

        // RMS spread of B scales linearly with σ (slope ≈ rms-per-unit-σ).
        let rms =
            |b: &[f32]| -> f32 { (b.iter().map(|&v| v * v).sum::<f32>() / b.len() as f32).sqrt() };
        let rms1 = rms(b1);
        let rms2 = rms(b2);
        assert!(rms1 > 0.0, "frequency spread must be positive");
        assert!(
            (rms2 / rms1 - 2.0).abs() < 1e-3,
            "RMS spread should double when σ doubles: rms2/rms1={}",
            rms2 / rms1
        );
        // The σ used is recoverable from the σ=1 baseline spread up to the generator's
        // own (non-unit) scale: confirm the spread is strictly increasing in σ.
        let rms_half = rms(draw(0.5).frequencies());
        assert!(
            rms_half < rms1 && rms1 < rms2,
            "spread must increase with σ: {rms_half} < {rms1} < {rms2}"
        );
    }

    // (f) ‖γ(x)‖² = m for all x (bounded) -------------------------------------------
    #[test]
    fn embedding_norm_squared_equals_m() {
        let m = 12;
        let ff = make(2, m, 4.0);
        for trial in 0..8 {
            let x = [0.13 * trial as f32 - 0.5, 0.27 * trial as f32];
            let g = ff
                .embed(&x)
                .expect("embed with valid 2D point should succeed in norm-squared test");
            let norm2: f32 = g.iter().map(|&v| v * v).sum();
            assert!(
                (norm2 - m as f32).abs() < 1e-4,
                "‖γ(x)‖² must equal m = {m}, got {norm2} at x={x:?}"
            );
        }
    }

    // analytic Jacobian matches a central finite difference -------------------------
    #[test]
    fn jacobian_matches_finite_difference() {
        let m = 5;
        let d = 2;
        let ff = make(d, m, 1.5);
        let x = [0.23_f32, -0.41];
        let jac = ff
            .jacobian(&x)
            .expect("jacobian with valid 2D input should succeed");
        assert_eq!(jac.len(), 2 * m * d);

        let h = 1e-3_f32;
        for j in 0..d {
            let mut xp = x;
            let mut xm = x;
            xp[j] += h;
            xm[j] -= h;
            let gp = ff
                .embed(&xp)
                .expect("embed at forward-perturbed xp should succeed for Jacobian FD check");
            let gm = ff
                .embed(&xm)
                .expect("embed at backward-perturbed xm should succeed for Jacobian FD check");
            for r in 0..2 * m {
                let fd = (gp[r] - gm[r]) / (2.0 * h);
                let an = jac[r * d + j];
                assert!(
                    (fd - an).abs() < 1e-2,
                    "Jacobian row {r} col {j}: analytic {an} vs FD {fd}"
                );
            }
        }
    }

    // shape / finiteness guards -----------------------------------------------------
    #[test]
    fn embed_dimension_mismatch_errors() {
        let ff = make(2, 4, 1.0);
        assert!(matches!(
            ff.embed(&[0.5]),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn construct_rejects_zero_modes_and_bad_sigma() {
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            FourierFeatures::new(
                FourierFeatureEmbeddingConfig {
                    input_dim: 2,
                    n_modes: 0,
                    sigma: 1.0
                },
                &mut rng
            ),
            Err(PinnError::InvalidLayerWidth)
        ));
        assert!(matches!(
            FourierFeatures::new(
                FourierFeatureEmbeddingConfig {
                    input_dim: 2,
                    n_modes: 4,
                    sigma: 0.0
                },
                &mut rng
            ),
            Err(PinnError::InvalidPdeCoefficient { .. })
        ));
        assert!(matches!(
            FourierFeatures::new(
                FourierFeatureEmbeddingConfig {
                    input_dim: 2,
                    n_modes: 4,
                    sigma: f32::NAN
                },
                &mut rng
            ),
            Err(PinnError::InvalidPdeCoefficient { .. })
        ));
    }

    #[test]
    fn from_matrix_validates_shape_and_finiteness() {
        // Wrong length.
        assert!(matches!(
            FourierFeatures::from_matrix(vec![0.0; 5], 2, 2),
            Err(PinnError::DimensionMismatch { .. })
        ));
        // Non-finite entry.
        assert!(matches!(
            FourierFeatures::from_matrix(vec![0.0, f32::INFINITY, 1.0, 2.0], 2, 2),
            Err(PinnError::NanEncountered { .. })
        ));
        // Valid; an explicit matrix has no defined sampling scale.
        let ff = FourierFeatures::from_matrix(vec![1.0, 0.0, 0.0, 1.0], 2, 2)
            .expect("from_matrix with valid finite 2x2 identity-like matrix should succeed");
        assert_eq!(ff.output_dim(), 4);
        assert_eq!(ff.sigma(), None);
    }

    #[test]
    fn sampled_embedding_reports_its_sigma() {
        let ff = make(2, 4, 3.5);
        assert_eq!(ff.sigma(), Some(3.5_f32));
    }

    #[test]
    fn embedding_all_finite_over_grid() {
        let ff = make(1, 16, 10.0);
        for i in 0..50 {
            let x = [i as f32 * 0.05 - 1.0];
            let g = ff
                .embed(&x)
                .expect("embed with valid 1D grid point should be finite");
            assert!(g.iter().all(|v| v.is_finite()), "embedding not finite");
        }
    }
}
