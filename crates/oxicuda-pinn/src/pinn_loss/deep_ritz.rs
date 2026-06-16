//! Deep Ritz Method: solve PDEs by minimising an energy functional.
//!
//! E & Yu (2018) "The Deep Ritz Method: A Deep Learning-Based Numerical Algorithm
//! for Solving Variational Problems", Communications in Mathematics and Statistics.
//!
//! Instead of minimising the strong-form PDE residual, the Deep Ritz method
//! minimises the **energy functional** associated with the variational form of
//! the problem. For the Poisson problem `−Δu = f` on a domain `Ω` with Dirichlet
//! boundary condition `u = g` on `∂Ω`, the energy is
//!
//! ```text
//! E[u] = ∫_Ω ( ½ |∇u(x)|² − f(x)·u(x) ) dx  +  β ∫_∂Ω ( u(x) − g(x) )² ds .
//! ```
//!
//! The volume and surface integrals are estimated by Monte-Carlo integration over
//! uniformly sampled interior / boundary points. The Euler-Lagrange equation of
//! this functional (in the limit `β → ∞`) recovers the strong-form Poisson PDE,
//! so a minimiser of `E` is a weak solution of the PDE.
//!
//! ## Architecture
//!
//! The paper's distinctive architectural contribution is a **residual block** with a
//! skip connection:
//!
//! ```text
//! y = s + act( W₂ · act( W₁·s + b₁ ) + b₂ ) ,        in_dim == out_dim (skip)
//! ```
//!
//! Here we use `tanh` as the activation (smooth, with analytic derivative
//! `1 − tanh²`) so that the gradient `∇_x u` needed for the Dirichlet energy term
//! can be computed analytically via the chain rule through every block.
//!
//! The full network lifts the `dim`-dimensional input to a hidden `width`, applies
//! `n_blocks` residual blocks, then projects to a scalar with a linear output layer.

use crate::error::{PinnError, PinnResult};
use crate::handle::LcgRng;
use core::cmp::Ordering;

// ────────────────────────────── activation ───────────────────────────────────

/// `tanh` activation value.
#[inline]
fn act(x: f32) -> f32 {
    x.tanh()
}

/// Derivative of `tanh`: `1 − tanh²(x)`.
#[inline]
fn act_prime(x: f32) -> f32 {
    let t = x.tanh();
    1.0 - t * t
}

// ────────────────────────────── config ───────────────────────────────────────

/// Configuration for the Deep Ritz energy functional.
#[derive(Debug, Clone)]
pub struct DeepRitzConfig {
    /// Spatial dimension of the domain `Ω`.
    pub dim: usize,
    /// Boundary penalty coefficient `β` weighting the Dirichlet surface term.
    pub boundary_penalty: f32,
    /// Number of Monte-Carlo samples drawn in the interior `Ω`.
    pub n_interior: usize,
    /// Number of Monte-Carlo samples drawn on the boundary `∂Ω`.
    pub n_boundary: usize,
}

/// Decomposition of the Deep Ritz energy into its constituent terms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeepRitzEnergy {
    /// Total energy `E[u] = dirichlet_term − source_term + boundary_term`.
    pub total: f32,
    /// Dirichlet (kinetic) term: MC estimate of `∫_Ω ½ |∇u|² dx`.
    pub dirichlet_term: f32,
    /// Source term: MC estimate of `∫_Ω f·u dx`.
    pub source_term: f32,
    /// Boundary penalty term: `β · ` MC estimate of `∫_∂Ω (u − g)² ds`.
    pub boundary_term: f32,
}

// ────────────────────────────── residual block ───────────────────────────────

/// A Deep Ritz residual block with a skip connection.
///
/// Computes `y = s + act( W₂ · act( W₁·s + b₁ ) + b₂ )` with `in_dim == out_dim ==
/// width` so that the identity skip connection is well defined. All weight matrices
/// are stored row-major (`width × width`).
#[derive(Debug, Clone)]
pub struct DeepRitzBlock {
    /// First weight matrix `W₁`, row-major `width × width`.
    w1: Vec<f32>,
    /// First bias `b₁`, length `width`.
    b1: Vec<f32>,
    /// Second weight matrix `W₂`, row-major `width × width`.
    w2: Vec<f32>,
    /// Second bias `b₂`, length `width`.
    b2: Vec<f32>,
    /// Block width (== input and output dimension).
    width: usize,
}

impl DeepRitzBlock {
    /// Create a new residual block of the given `width` with Xavier-style init.
    ///
    /// # Errors
    /// - [`PinnError::InvalidLayerWidth`] if `width == 0`.
    pub fn new(width: usize, rng: &mut LcgRng) -> PinnResult<Self> {
        if width == 0 {
            return Err(PinnError::InvalidLayerWidth);
        }
        let scale = (2.0 / width as f32).sqrt();
        let mut draw = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
                .collect()
        };
        let w1 = draw(width * width);
        let w2 = draw(width * width);
        Ok(Self {
            w1,
            b1: vec![0.0_f32; width],
            w2,
            b2: vec![0.0_f32; width],
            width,
        })
    }

    /// Block width.
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Matrix-vector product `m · v` for a row-major `width × width` matrix `m`.
    fn matvec(&self, m: &[f32], v: &[f32]) -> Vec<f32> {
        let w = self.width;
        (0..w)
            .map(|i| {
                let row = &m[i * w..i * w + w];
                row.iter().zip(v.iter()).map(|(a, b)| a * b).sum()
            })
            .collect()
    }

    /// Forward value of the block: `y = s + act(W₂·act(W₁·s + b₁) + b₂)`.
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if `s.len() != width`.
    pub fn forward(&self, s: &[f32]) -> PinnResult<Vec<f32>> {
        if s.len() != self.width {
            return Err(PinnError::DimensionMismatch {
                expected: self.width,
                got: s.len(),
            });
        }
        let w1s = self.matvec(&self.w1, s);
        let a1: Vec<f32> = w1s
            .iter()
            .zip(self.b1.iter())
            .map(|(z, b)| act(z + b))
            .collect();
        let w2a1 = self.matvec(&self.w2, &a1);
        let out: Vec<f32> = (0..self.width)
            .map(|i| s[i] + act(w2a1[i] + self.b2[i]))
            .collect();
        Ok(out)
    }

    /// Analytic Jacobian `d(out)/d(in)`, row-major `width × width`.
    ///
    /// With pre-activations `a₁ = W₁·s + b₁` and `a₂ = W₂·act(a₁) + b₂`, the skip
    /// connection contributes the identity term:
    ///
    /// ```text
    /// J_block = I + diag(act'(a₂)) · W₂ · diag(act'(a₁)) · W₁ .
    /// ```
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if `s.len() != width`.
    pub fn jacobian(&self, s: &[f32]) -> PinnResult<Vec<f32>> {
        if s.len() != self.width {
            return Err(PinnError::DimensionMismatch {
                expected: self.width,
                got: s.len(),
            });
        }
        let w = self.width;
        // Pre-activations.
        let a1: Vec<f32> = self
            .matvec(&self.w1, s)
            .iter()
            .zip(self.b1.iter())
            .map(|(z, b)| z + b)
            .collect();
        let h1: Vec<f32> = a1.iter().map(|z| act(*z)).collect();
        let a2: Vec<f32> = self
            .matvec(&self.w2, &h1)
            .iter()
            .zip(self.b2.iter())
            .map(|(z, b)| z + b)
            .collect();
        let d1: Vec<f32> = a1.iter().map(|z| act_prime(*z)).collect();
        let d2: Vec<f32> = a2.iter().map(|z| act_prime(*z)).collect();

        // M = diag(d2) · W2 · diag(d1) · W1, then J = I + M.
        // (W2 · diag(d1))[i][k] = w2[i][k] * d1[k]
        // M[i][j] = d2[i] * Σ_k w2[i][k] * d1[k] * w1[k][j]
        let mut jac = vec![0.0_f32; w * w];
        for i in 0..w {
            for j in 0..w {
                let mut acc = 0.0_f32;
                for (k, &d1k) in d1.iter().enumerate() {
                    acc += self.w2[i * w + k] * d1k * self.w1[k * w + j];
                }
                let mut val = d2[i] * acc;
                if i == j {
                    val += 1.0;
                }
                jac[i * w + j] = val;
            }
        }
        Ok(jac)
    }
}

// ────────────────────────────── network ──────────────────────────────────────

/// Deep Ritz network: linear lift → residual blocks → linear scalar output.
#[derive(Debug, Clone)]
pub struct DeepRitzNet {
    /// Lift weight `width × dim`, row-major.
    lift_w: Vec<f32>,
    /// Lift bias, length `width`.
    lift_b: Vec<f32>,
    /// Residual blocks.
    blocks: Vec<DeepRitzBlock>,
    /// Output weight `1 × width`.
    out_w: Vec<f32>,
    /// Output scalar bias.
    out_b: f32,
    /// Hidden width.
    width: usize,
    /// Input dimension.
    dim: usize,
}

impl DeepRitzNet {
    /// Construct a new Deep Ritz network.
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if `dim == 0` (expected `≥ 1`).
    /// - [`PinnError::InvalidLayerWidth`] if `width == 0`.
    /// - [`PinnError::InvalidNetworkDepth`] if `n_blocks == 0`.
    pub fn new(dim: usize, width: usize, n_blocks: usize, rng: &mut LcgRng) -> PinnResult<Self> {
        if dim == 0 {
            return Err(PinnError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if width == 0 {
            return Err(PinnError::InvalidLayerWidth);
        }
        if n_blocks == 0 {
            return Err(PinnError::InvalidNetworkDepth { depth: n_blocks });
        }
        let lift_scale = (2.0 / dim as f32).sqrt();
        let lift_w: Vec<f32> = (0..width * dim)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * lift_scale)
            .collect();
        let lift_b = vec![0.0_f32; width];

        let mut blocks = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
            blocks.push(DeepRitzBlock::new(width, rng)?);
        }

        let out_scale = (2.0 / width as f32).sqrt();
        let out_w: Vec<f32> = (0..width)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * out_scale)
            .collect();

        Ok(Self {
            lift_w,
            lift_b,
            blocks,
            out_w,
            out_b: 0.0,
            width,
            dim,
        })
    }

    /// Input dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Hidden width.
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Number of residual blocks.
    #[inline]
    pub fn n_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Lift the `dim`-dimensional input to the hidden `width` and apply `tanh`.
    fn lift(&self, x: &[f32]) -> Vec<f32> {
        let w = self.width;
        let d = self.dim;
        (0..w)
            .map(|i| {
                let row = &self.lift_w[i * d..i * d + d];
                let dot: f32 = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                act(dot + self.lift_b[i])
            })
            .collect()
    }

    /// Scalar output `u(x)`.
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if `x.len() != dim`.
    /// - [`PinnError::NanEncountered`] if a non-finite value is produced.
    pub fn forward(&self, x: &[f32]) -> PinnResult<f32> {
        if x.len() != self.dim {
            return Err(PinnError::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        let mut h = self.lift(x);
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        let out: f32 = self
            .out_w
            .iter()
            .zip(h.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>()
            + self.out_b;
        if !out.is_finite() {
            return Err(PinnError::NanEncountered {
                location: "deep_ritz_forward",
            });
        }
        Ok(out)
    }

    /// Spatial gradient `∇_x u(x)`, length `dim`, via the analytic chain rule.
    ///
    /// We forward through `lift → blocks → output` while accumulating the Jacobian
    /// of the current hidden state w.r.t. the input `x`. The lift layer Jacobian is
    /// `diag(act'(z)) · W_lift` where `z = W_lift·x + b_lift`; each block contributes
    /// its [`DeepRitzBlock::jacobian`]; the output layer maps the final Jacobian by
    /// `W_out · J_total`, giving the `1 × dim` gradient.
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if `x.len() != dim`.
    /// - [`PinnError::NanEncountered`] if a non-finite value is produced.
    pub fn grad_x(&self, x: &[f32]) -> PinnResult<Vec<f32>> {
        if x.len() != self.dim {
            return Err(PinnError::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        let w = self.width;
        let d = self.dim;

        // Lift pre-activation and Jacobian J_lift = diag(act'(z)) · W_lift  (width × dim).
        let z: Vec<f32> = (0..w)
            .map(|i| {
                let row = &self.lift_w[i * d..i * d + d];
                let dot: f32 = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                dot + self.lift_b[i]
            })
            .collect();
        let mut h: Vec<f32> = z.iter().map(|v| act(*v)).collect();
        let dlift: Vec<f32> = z.iter().map(|v| act_prime(*v)).collect();
        // jac: current hidden Jacobian, row-major width × dim.
        let mut jac = vec![0.0_f32; w * d];
        for i in 0..w {
            for j in 0..d {
                jac[i * d + j] = dlift[i] * self.lift_w[i * d + j];
            }
        }

        // Propagate through residual blocks: jac ← J_block · jac, h ← block(h).
        for block in &self.blocks {
            let jb = block.jacobian(&h)?; // width × width
            let mut new_jac = vec![0.0_f32; w * d];
            for i in 0..w {
                for j in 0..d {
                    let mut acc = 0.0_f32;
                    for k in 0..w {
                        acc += jb[i * w + k] * jac[k * d + j];
                    }
                    new_jac[i * d + j] = acc;
                }
            }
            jac = new_jac;
            h = block.forward(&h)?;
        }

        // Output layer: grad = W_out · jac  (1 × dim).
        let mut grad = vec![0.0_f32; d];
        for j in 0..d {
            let mut acc = 0.0_f32;
            for k in 0..w {
                acc += self.out_w[k] * jac[k * d + j];
            }
            grad[j] = acc;
        }
        if grad.iter().any(|v| !v.is_finite()) {
            return Err(PinnError::NanEncountered {
                location: "deep_ritz_grad_x",
            });
        }
        Ok(grad)
    }

    /// Total number of trainable parameters.
    pub fn n_params(&self) -> usize {
        let lift = self.lift_w.len() + self.lift_b.len();
        let blocks: usize = self
            .blocks
            .iter()
            .map(|b| b.w1.len() + b.b1.len() + b.w2.len() + b.b2.len())
            .sum();
        let out = self.out_w.len() + 1;
        lift + blocks + out
    }

    /// Flatten all parameters into a single vector (lift, blocks, output).
    pub fn get_params(&self) -> Vec<f32> {
        let mut p = Vec::with_capacity(self.n_params());
        p.extend_from_slice(&self.lift_w);
        p.extend_from_slice(&self.lift_b);
        for b in &self.blocks {
            p.extend_from_slice(&b.w1);
            p.extend_from_slice(&b.b1);
            p.extend_from_slice(&b.w2);
            p.extend_from_slice(&b.b2);
        }
        p.extend_from_slice(&self.out_w);
        p.push(self.out_b);
        p
    }

    /// Set all parameters from a flat vector in the same order as [`Self::get_params`].
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if `p.len() != n_params()`.
    pub fn set_params(&mut self, p: &[f32]) -> PinnResult<()> {
        let expected = self.n_params();
        if p.len() != expected {
            return Err(PinnError::DimensionMismatch {
                expected,
                got: p.len(),
            });
        }
        let width = self.width;
        let lift_w_len = self.lift_w.len();
        let lift_b_len = self.lift_b.len();
        let out_w_len = self.out_w.len();
        let mut off = 0usize;

        self.lift_w.copy_from_slice(&p[off..off + lift_w_len]);
        off += lift_w_len;
        self.lift_b.copy_from_slice(&p[off..off + lift_b_len]);
        off += lift_b_len;
        for b in &mut self.blocks {
            b.w1.copy_from_slice(&p[off..off + width * width]);
            off += width * width;
            b.b1.copy_from_slice(&p[off..off + width]);
            off += width;
            b.w2.copy_from_slice(&p[off..off + width * width]);
            off += width * width;
            b.b2.copy_from_slice(&p[off..off + width]);
            off += width;
        }
        self.out_w.copy_from_slice(&p[off..off + out_w_len]);
        off += out_w_len;
        self.out_b = p[off];
        Ok(())
    }
}

// ────────────────────────────── Deep Ritz driver ─────────────────────────────

/// Deep Ritz energy assembly, Monte-Carlo sampling, and finite-difference training.
pub struct DeepRitz;

impl DeepRitz {
    /// Dirichlet (kinetic) energy density `½ Σ_d grad_d²`.
    pub fn dirichlet_density(grad: &[f32]) -> f32 {
        0.5 * grad.iter().map(|g| g * g).sum::<f32>()
    }

    /// Validate a sampling box `[lo, hi]` against the configured dimension.
    fn validate_box(cfg: &DeepRitzConfig, lo: &[f32], hi: &[f32]) -> PinnResult<()> {
        if cfg.dim == 0 {
            return Err(PinnError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if lo.len() != cfg.dim {
            return Err(PinnError::DimensionMismatch {
                expected: cfg.dim,
                got: lo.len(),
            });
        }
        if hi.len() != cfg.dim {
            return Err(PinnError::DimensionMismatch {
                expected: cfg.dim,
                got: hi.len(),
            });
        }
        for d in 0..cfg.dim {
            if lo[d].partial_cmp(&hi[d]) != Some(Ordering::Less) {
                return Err(PinnError::InvalidTimeInterval {
                    t0: lo[d],
                    t1: hi[d],
                });
            }
        }
        Ok(())
    }

    /// Sample `n_interior` uniform interior points in the box `[lo, hi]`.
    ///
    /// Returns a flat `n_interior × dim` row-major buffer.
    ///
    /// # Errors
    /// - [`PinnError::EmptyInput`] if `n_interior == 0`.
    /// - dimension / interval errors from box validation.
    pub fn sample_interior(
        cfg: &DeepRitzConfig,
        lo: &[f32],
        hi: &[f32],
        rng: &mut LcgRng,
    ) -> PinnResult<Vec<f32>> {
        Self::validate_box(cfg, lo, hi)?;
        if cfg.n_interior == 0 {
            return Err(PinnError::EmptyInput);
        }
        let d = cfg.dim;
        let mut pts = Vec::with_capacity(cfg.n_interior * d);
        for _ in 0..cfg.n_interior {
            for dd in 0..d {
                let u = rng.next_f32();
                pts.push(lo[dd] + u * (hi[dd] - lo[dd]));
            }
        }
        Ok(pts)
    }

    /// Sample `n_boundary` uniform points on the faces of the box `[lo, hi]`.
    ///
    /// Each point is drawn uniformly in the interior, then one coordinate is pinned
    /// to either `lo` or `hi`, placing the point on a face of `∂Ω`.
    ///
    /// Returns a flat `n_boundary × dim` row-major buffer (empty if `n_boundary == 0`).
    ///
    /// # Errors
    /// - dimension / interval errors from box validation.
    pub fn sample_boundary(
        cfg: &DeepRitzConfig,
        lo: &[f32],
        hi: &[f32],
        rng: &mut LcgRng,
    ) -> PinnResult<Vec<f32>> {
        Self::validate_box(cfg, lo, hi)?;
        let d = cfg.dim;
        let mut pts = Vec::with_capacity(cfg.n_boundary * d);
        for _ in 0..cfg.n_boundary {
            // Base interior coordinates.
            let mut p: Vec<f32> = (0..d)
                .map(|dd| lo[dd] + rng.next_f32() * (hi[dd] - lo[dd]))
                .collect();
            // Choose a face: pin one coordinate to lo or hi.
            let face_dim = rng.next_usize(d);
            let to_hi = rng.next_f32() >= 0.5;
            p[face_dim] = if to_hi { hi[face_dim] } else { lo[face_dim] };
            pts.extend_from_slice(&p);
        }
        Ok(pts)
    }

    /// Monte-Carlo estimate of the Deep Ritz energy on pre-sampled points.
    ///
    /// `interior` is a flat `n_interior × dim` buffer, `boundary` a flat
    /// `n_boundary × dim` buffer. `source` evaluates `f(x)` and `g` the boundary
    /// target `g(x)`.
    ///
    /// The energy terms are MC averages (mean over samples), giving an unbiased
    /// estimate of the corresponding integrals up to the constant domain / surface
    /// measure (which is irrelevant for the minimisation and is dropped, matching the
    /// common Deep Ritz practice of averaging over batch samples).
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if buffer lengths are not multiples of `dim`.
    /// - [`PinnError::NanEncountered`] if a non-finite term is produced.
    pub fn energy<S, G>(
        net: &DeepRitzNet,
        cfg: &DeepRitzConfig,
        interior: &[f32],
        boundary: &[f32],
        source: S,
        g: G,
    ) -> PinnResult<DeepRitzEnergy>
    where
        S: Fn(&[f32]) -> f32,
        G: Fn(&[f32]) -> f32,
    {
        let d = cfg.dim;
        if d == 0 {
            return Err(PinnError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if interior.len() % d != 0 {
            return Err(PinnError::DimensionMismatch {
                expected: d,
                got: interior.len() % d,
            });
        }
        if boundary.len() % d != 0 {
            return Err(PinnError::DimensionMismatch {
                expected: d,
                got: boundary.len() % d,
            });
        }
        let n_int = interior.len() / d;
        let n_bnd = boundary.len() / d;

        // Volume terms: ½|∇u|² and f·u, averaged over interior samples.
        let mut dirichlet = 0.0_f32;
        let mut src = 0.0_f32;
        for i in 0..n_int {
            let x = &interior[i * d..i * d + d];
            let grad = net.grad_x(x)?;
            dirichlet += Self::dirichlet_density(&grad);
            let u = net.forward(x)?;
            src += source(x) * u;
        }
        if n_int > 0 {
            let inv = 1.0 / n_int as f32;
            dirichlet *= inv;
            src *= inv;
        }

        // Boundary term: β · mean (u − g)².
        let mut bnd = 0.0_f32;
        for i in 0..n_bnd {
            let x = &boundary[i * d..i * d + d];
            let u = net.forward(x)?;
            let diff = u - g(x);
            bnd += diff * diff;
        }
        if n_bnd > 0 {
            bnd *= cfg.boundary_penalty / n_bnd as f32;
        } else {
            bnd = 0.0;
        }

        let total = dirichlet - src + bnd;
        let energy = DeepRitzEnergy {
            total,
            dirichlet_term: dirichlet,
            source_term: src,
            boundary_term: bnd,
        };
        if !total.is_finite() || !dirichlet.is_finite() || !src.is_finite() || !bnd.is_finite() {
            return Err(PinnError::NanEncountered {
                location: "deep_ritz_energy",
            });
        }
        Ok(energy)
    }

    /// One finite-difference energy-descent step. Returns the energy **before** the step.
    ///
    /// The interior / boundary points are sampled **once** at the start of the step and
    /// reused for every parameter perturbation, so the finite-difference gradient is a
    /// consistent estimate of `∇_θ E` on a fixed Monte-Carlo batch. For each parameter
    /// `k` the centred difference `g_k = (E⁺ − E⁻) / (2·fd_eps)` is formed, then the
    /// parameters are updated by `θ ← θ − lr·g`.
    ///
    /// This is `O(n_params)` energy evaluations per step, so keep test networks small
    /// (`width ≤ 8`, `n_blocks ≤ 2`).
    ///
    /// # Errors
    /// - [`PinnError::InvalidStepSize`] if `fd_eps ≤ 0` or not finite.
    /// - sampling / energy errors propagated from the helpers.
    #[allow(clippy::too_many_arguments)]
    pub fn train_step<S, G>(
        net: &mut DeepRitzNet,
        cfg: &DeepRitzConfig,
        lo: &[f32],
        hi: &[f32],
        source: &S,
        g: &G,
        lr: f32,
        fd_eps: f32,
        rng: &mut LcgRng,
    ) -> PinnResult<f32>
    where
        S: Fn(&[f32]) -> f32,
        G: Fn(&[f32]) -> f32,
    {
        if !fd_eps.is_finite() || fd_eps <= 0.0 {
            return Err(PinnError::InvalidStepSize { h: fd_eps });
        }
        // Sample the MC batch ONCE for the whole step.
        let interior = Self::sample_interior(cfg, lo, hi, rng)?;
        let boundary = Self::sample_boundary(cfg, lo, hi, rng)?;

        let base = Self::energy(net, cfg, &interior, &boundary, source, g)?;

        let theta = net.get_params();
        let n = theta.len();
        let mut grad = vec![0.0_f32; n];
        let mut perturbed = theta.clone();
        for k in 0..n {
            let original = theta[k];

            perturbed[k] = original + fd_eps;
            net.set_params(&perturbed)?;
            let e_plus = Self::energy(net, cfg, &interior, &boundary, source, g)?.total;

            perturbed[k] = original - fd_eps;
            net.set_params(&perturbed)?;
            let e_minus = Self::energy(net, cfg, &interior, &boundary, source, g)?.total;

            perturbed[k] = original; // restore
            grad[k] = (e_plus - e_minus) / (2.0 * fd_eps);
        }

        // Gradient-descent update from the original parameters.
        let updated: Vec<f32> = theta
            .iter()
            .zip(grad.iter())
            .map(|(t, gk)| t - lr * gk)
            .collect();
        net.set_params(&updated)?;
        Ok(base.total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(dim: usize) -> DeepRitzConfig {
        DeepRitzConfig {
            dim,
            boundary_penalty: 10.0,
            n_interior: 8,
            n_boundary: 4,
        }
    }

    #[test]
    fn dirichlet_density_formula() {
        let grad = [3.0_f32, 4.0];
        // ½(9 + 16) = 12.5
        assert!((DeepRitz::dirichlet_density(&grad) - 12.5).abs() < 1e-6);
        assert_eq!(DeepRitz::dirichlet_density(&[]), 0.0);
    }

    #[test]
    fn sample_interior_length_and_bounds() {
        let mut rng = LcgRng::new(1);
        let c = cfg(2);
        let lo = [0.0_f32, -1.0];
        let hi = [1.0_f32, 2.0];
        let pts = DeepRitz::sample_interior(&c, &lo, &hi, &mut rng)
            .expect("interior sampling should succeed");
        assert_eq!(pts.len(), c.n_interior * c.dim);
        for chunk in pts.chunks(2) {
            assert!(chunk[0] >= lo[0] && chunk[0] <= hi[0]);
            assert!(chunk[1] >= lo[1] && chunk[1] <= hi[1]);
        }
    }

    #[test]
    fn sample_boundary_on_face() {
        let mut rng = LcgRng::new(2);
        let c = cfg(2);
        let lo = [0.0_f32, 0.0];
        let hi = [1.0_f32, 1.0];
        let pts = DeepRitz::sample_boundary(&c, &lo, &hi, &mut rng)
            .expect("boundary sampling should succeed");
        assert_eq!(pts.len(), c.n_boundary * c.dim);
        for chunk in pts.chunks(2) {
            let on_face =
                (0..2).any(|d| (chunk[d] - lo[d]).abs() < 1e-6 || (chunk[d] - hi[d]).abs() < 1e-6);
            assert!(on_face, "boundary point not on a face: {chunk:?}");
        }
    }

    #[test]
    fn block_forward_length() {
        let mut rng = LcgRng::new(3);
        let block = DeepRitzBlock::new(5, &mut rng)
            .expect("DeepRitzBlock construction with valid params should succeed");
        let out = block
            .forward(&[0.1, 0.2, 0.3, 0.4, 0.5])
            .expect("forward pass should succeed for valid input");
        assert_eq!(out.len(), 5);
    }

    #[test]
    fn block_jacobian_dims() {
        let mut rng = LcgRng::new(4);
        let block = DeepRitzBlock::new(4, &mut rng)
            .expect("DeepRitzBlock construction with valid params should succeed");
        let jac = block
            .jacobian(&[0.1, -0.2, 0.3, 0.0])
            .expect("jacobian computation should succeed");
        assert_eq!(jac.len(), 16);
    }

    #[test]
    fn block_jacobian_matches_finite_difference() {
        let mut rng = LcgRng::new(40);
        let width = 4;
        let block = DeepRitzBlock::new(width, &mut rng)
            .expect("DeepRitzBlock construction with valid params should succeed");
        let s = [0.2_f32, -0.3, 0.1, 0.4];
        let jac = block
            .jacobian(&s)
            .expect("jacobian computation should succeed");
        let eps = 1e-3_f32;
        for j in 0..width {
            let mut sp = s;
            let mut sm = s;
            sp[j] += eps;
            sm[j] -= eps;
            let yp = block
                .forward(&sp)
                .expect("forward pass should succeed for valid input");
            let ym = block
                .forward(&sm)
                .expect("forward pass should succeed for valid input");
            for i in 0..width {
                let fd = (yp[i] - ym[i]) / (2.0 * eps);
                assert!(
                    (jac[i * width + j] - fd).abs() < 1e-2,
                    "block jac[{i}][{j}]={} fd={fd}",
                    jac[i * width + j]
                );
            }
        }
    }

    #[test]
    fn net_forward_scalar() {
        let mut rng = LcgRng::new(5);
        let net = DeepRitzNet::new(2, 6, 2, &mut rng)
            .expect("DeepRitzNet construction with valid params should succeed");
        let u = net
            .forward(&[0.3, 0.7])
            .expect("forward pass should succeed for valid input");
        assert!(u.is_finite());
    }

    #[test]
    fn grad_x_length() {
        let mut rng = LcgRng::new(6);
        let net = DeepRitzNet::new(3, 5, 1, &mut rng)
            .expect("DeepRitzNet construction with valid params should succeed");
        let grad = net
            .grad_x(&[0.1, 0.2, 0.3])
            .expect("gradient computation should succeed");
        assert_eq!(grad.len(), 3);
    }

    #[test]
    fn grad_x_matches_finite_difference() {
        let mut rng = LcgRng::new(7);
        let net = DeepRitzNet::new(2, 6, 2, &mut rng)
            .expect("DeepRitzNet construction with valid params should succeed");
        let x = [0.35_f32, -0.2];
        let grad = net.grad_x(&x).expect("gradient computation should succeed");
        let eps = 1e-3_f32;
        for j in 0..2 {
            let mut xp = x;
            let mut xm = x;
            xp[j] += eps;
            xm[j] -= eps;
            let up = net
                .forward(&xp)
                .expect("forward pass should succeed for valid input");
            let um = net
                .forward(&xm)
                .expect("forward pass should succeed for valid input");
            let fd = (up - um) / (2.0 * eps);
            assert!(
                (grad[j] - fd).abs() < 1e-2,
                "grad_x[{j}]={} finite-diff={fd}",
                grad[j]
            );
        }
    }

    #[test]
    fn n_params_matches_get_params_len() {
        let mut rng = LcgRng::new(8);
        let net = DeepRitzNet::new(2, 7, 3, &mut rng)
            .expect("DeepRitzNet construction with valid params should succeed");
        assert_eq!(net.n_params(), net.get_params().len());
    }

    #[test]
    fn set_params_round_trip() {
        let mut rng = LcgRng::new(9);
        let mut net = DeepRitzNet::new(2, 5, 2, &mut rng)
            .expect("DeepRitzNet construction with valid params should succeed");
        let p0 = net.get_params();
        // Map to new values, set, and read back.
        let p1: Vec<f32> = p0
            .iter()
            .enumerate()
            .map(|(i, _)| 0.01 * i as f32)
            .collect();
        net.set_params(&p1)
            .expect("set_params should succeed for valid param vector");
        let p2 = net.get_params();
        assert_eq!(p1.len(), p2.len());
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn energy_terms_finite() {
        let mut rng = LcgRng::new(10);
        let c = cfg(1);
        let net = DeepRitzNet::new(1, 6, 2, &mut rng)
            .expect("DeepRitzNet construction with valid params should succeed");
        let lo = [0.0_f32];
        let hi = [1.0_f32];
        let interior = DeepRitz::sample_interior(&c, &lo, &hi, &mut rng)
            .expect("interior sampling should succeed");
        let boundary = DeepRitz::sample_boundary(&c, &lo, &hi, &mut rng)
            .expect("boundary sampling should succeed");
        let f = |x: &[f32]| {
            let pi = std::f32::consts::PI;
            pi * pi * (pi * x[0]).sin()
        };
        let g = |_x: &[f32]| 0.0_f32;
        let e = DeepRitz::energy(&net, &c, &interior, &boundary, f, g)
            .expect("energy computation should succeed");
        assert!(e.total.is_finite());
        assert!(e.dirichlet_term.is_finite());
        assert!(e.source_term.is_finite());
        assert!(e.boundary_term.is_finite());
    }

    #[test]
    fn energy_assembly_sign() {
        // Tiny hand setup: total == dirichlet − source + boundary.
        let mut rng = LcgRng::new(11);
        let c = cfg(1);
        let net = DeepRitzNet::new(1, 4, 1, &mut rng)
            .expect("DeepRitzNet construction with valid params should succeed");
        let lo = [0.0_f32];
        let hi = [1.0_f32];
        let interior = DeepRitz::sample_interior(&c, &lo, &hi, &mut rng)
            .expect("interior sampling should succeed");
        let boundary = DeepRitz::sample_boundary(&c, &lo, &hi, &mut rng)
            .expect("boundary sampling should succeed");
        let f = |_x: &[f32]| 1.0_f32;
        let g = |_x: &[f32]| 0.5_f32;
        let e = DeepRitz::energy(&net, &c, &interior, &boundary, f, g)
            .expect("energy computation should succeed");
        let reassembled = e.dirichlet_term - e.source_term + e.boundary_term;
        assert!((e.total - reassembled).abs() < 1e-5);
    }

    #[test]
    fn energy_dirichlet_nonneg() {
        let mut rng = LcgRng::new(12);
        let c = cfg(2);
        let net = DeepRitzNet::new(2, 5, 2, &mut rng)
            .expect("DeepRitzNet construction with valid params should succeed");
        let lo = [0.0_f32, 0.0];
        let hi = [1.0_f32, 1.0];
        let interior = DeepRitz::sample_interior(&c, &lo, &hi, &mut rng)
            .expect("interior sampling should succeed");
        let boundary = DeepRitz::sample_boundary(&c, &lo, &hi, &mut rng)
            .expect("boundary sampling should succeed");
        let e = DeepRitz::energy(&net, &c, &interior, &boundary, |_| 0.0, |_| 0.0)
            .expect("energy computation should succeed");
        // ½|∇u|² averaged is non-negative; boundary penalty (u−g)² non-negative.
        assert!(e.dirichlet_term >= 0.0);
        assert!(e.boundary_term >= 0.0);
    }

    #[test]
    fn train_step_decreases_energy_1d_poisson() {
        // 1D Poisson −u'' = f, f(x) = π² sin(πx), g = 0 on [0,1]; true u = sin(πx).
        let mut rng = LcgRng::new(2024);
        let c = DeepRitzConfig {
            dim: 1,
            boundary_penalty: 20.0,
            n_interior: 16,
            n_boundary: 2,
        };
        let mut net = DeepRitzNet::new(1, 8, 2, &mut rng)
            .expect("DeepRitzNet construction with valid params should succeed");
        let lo = [0.0_f32];
        let hi = [1.0_f32];
        let f = |x: &[f32]| {
            let pi = std::f32::consts::PI;
            pi * pi * (pi * x[0]).sin()
        };
        let g = |_x: &[f32]| 0.0_f32;

        let mut first_energy = None;
        let mut last_energy = 0.0_f32;
        for _ in 0..30 {
            let e = DeepRitz::train_step(&mut net, &c, &lo, &hi, &f, &g, 2e-3, 1e-3, &mut rng)
                .expect("train_step should succeed");
            if first_energy.is_none() {
                first_energy = Some(e);
            }
            last_energy = e;
        }
        let e0 = first_energy.expect("first_energy should be set after at least one train step");
        assert!(
            last_energy < e0,
            "energy did not decrease: start={e0} end={last_energy}"
        );
    }

    #[test]
    fn err_dim_mismatch_forward() {
        let mut rng = LcgRng::new(13);
        let net = DeepRitzNet::new(2, 4, 1, &mut rng)
            .expect("DeepRitzNet construction with valid params should succeed");
        assert!(matches!(
            net.forward(&[0.1]),
            Err(PinnError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            net.grad_x(&[0.1, 0.2, 0.3]),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_lo_ge_hi() {
        let mut rng = LcgRng::new(14);
        let c = cfg(1);
        let lo = [1.0_f32];
        let hi = [1.0_f32]; // not strictly greater
        assert!(matches!(
            DeepRitz::sample_interior(&c, &lo, &hi, &mut rng),
            Err(PinnError::InvalidTimeInterval { .. })
        ));
    }

    #[test]
    fn err_width_zero_and_blocks_zero() {
        let mut rng = LcgRng::new(15);
        assert!(matches!(
            DeepRitzBlock::new(0, &mut rng),
            Err(PinnError::InvalidLayerWidth)
        ));
        assert!(matches!(
            DeepRitzNet::new(2, 0, 1, &mut rng),
            Err(PinnError::InvalidLayerWidth)
        ));
        assert!(matches!(
            DeepRitzNet::new(2, 4, 0, &mut rng),
            Err(PinnError::InvalidNetworkDepth { .. })
        ));
        assert!(matches!(
            DeepRitzNet::new(0, 4, 1, &mut rng),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_set_params_wrong_len() {
        let mut rng = LcgRng::new(16);
        let mut net = DeepRitzNet::new(2, 4, 1, &mut rng)
            .expect("DeepRitzNet construction with valid params should succeed");
        let bad = vec![0.0_f32; net.n_params() + 1];
        assert!(matches!(
            net.set_params(&bad),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_sample_interior_zero() {
        let mut rng = LcgRng::new(17);
        let c = DeepRitzConfig {
            dim: 1,
            boundary_penalty: 1.0,
            n_interior: 0,
            n_boundary: 1,
        };
        assert!(matches!(
            DeepRitz::sample_interior(&c, &[0.0], &[1.0], &mut rng),
            Err(PinnError::EmptyInput)
        ));
    }

    #[test]
    fn err_train_step_bad_eps() {
        let mut rng = LcgRng::new(18);
        let c = cfg(1);
        let mut net = DeepRitzNet::new(1, 4, 1, &mut rng)
            .expect("DeepRitzNet construction with valid params should succeed");
        let f = |_x: &[f32]| 1.0_f32;
        let g = |_x: &[f32]| 0.0_f32;
        assert!(matches!(
            DeepRitz::train_step(&mut net, &c, &[0.0], &[1.0], &f, &g, 1e-2, 0.0, &mut rng),
            Err(PinnError::InvalidStepSize { .. })
        ));
    }

    #[test]
    fn deterministic_given_seed() {
        let mut rng_a = LcgRng::new(123);
        let mut rng_b = LcgRng::new(123);
        let net_a = DeepRitzNet::new(2, 5, 2, &mut rng_a)
            .expect("DeepRitzNet construction with valid params should succeed");
        let net_b = DeepRitzNet::new(2, 5, 2, &mut rng_b)
            .expect("DeepRitzNet construction with valid params should succeed");
        let pa = net_a.get_params();
        let pb = net_b.get_params();
        assert_eq!(pa.len(), pb.len());
        for (a, b) in pa.iter().zip(pb.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
        // Same forward output.
        let ua = net_a
            .forward(&[0.2, 0.4])
            .expect("forward pass should succeed for valid input");
        let ub = net_b
            .forward(&[0.2, 0.4])
            .expect("forward pass should succeed for valid input");
        assert!((ua - ub).abs() < 1e-9);
    }

    #[test]
    fn boundary_zero_samples_term_zero() {
        let mut rng = LcgRng::new(19);
        let c = DeepRitzConfig {
            dim: 1,
            boundary_penalty: 100.0,
            n_interior: 4,
            n_boundary: 0,
        };
        let net = DeepRitzNet::new(1, 4, 1, &mut rng)
            .expect("DeepRitzNet construction with valid params should succeed");
        let lo = [0.0_f32];
        let hi = [1.0_f32];
        let interior = DeepRitz::sample_interior(&c, &lo, &hi, &mut rng)
            .expect("interior sampling should succeed");
        let boundary = DeepRitz::sample_boundary(&c, &lo, &hi, &mut rng)
            .expect("boundary sampling should succeed");
        assert!(boundary.is_empty());
        let e = DeepRitz::energy(&net, &c, &interior, &boundary, |_| 1.0, |_| 0.0)
            .expect("energy computation should succeed");
        assert_eq!(e.boundary_term, 0.0);
    }
}
