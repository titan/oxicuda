//! Deformable 3D Gaussians for dynamic scenes.
//!
//! Yang, Gao, Zhou, Jiao, Zhang & Jin (2023), "Deformable 3D Gaussians for
//! High-Fidelity Monocular Dynamic Scene Reconstruction".
//!
//! A set of *canonical* 3D Gaussians (a static template) is animated by a small
//! deformation MLP `Φ`. Conditioned on a time embedding `γ(t)` and the canonical
//! position `x`, `Φ` predicts per-Gaussian offsets `(δμ, δs, δr)` that are
//! applied before rasterisation:
//!
//! ```text
//! μ'(t) = μ + δμ(x, t)
//! s'(t) = s + δs(x, t)        (clamped strictly positive)
//! q'(t) = normalize(q + δr(x, t))
//! ```
//!
//! The deformed Gaussians are then splatted with the static
//! [`crate::rendering::gaussian_splat_3d`] rasteriser (reused verbatim).
//!
//! **Canonical anchoring.** A *trained* network learns `Φ(x, t_c) ≈ 0` at the
//! canonical time `t_c`. Because this CPU core is randomly initialised
//! (untrained), we anchor the deformation analytically by returning the residual
//!
//! ```text
//! δ(x, t) = scale · ( Φ(x, t) − Φ(x, t_c) ),
//! ```
//!
//! which is *exactly* zero at `t = t_c` while remaining a smooth, time-varying
//! field elsewhere — the behaviour a converged network approximates. The time
//! conditioning reuses the crate's Fourier [`positional_encode`], so nearby
//! times map to nearby embeddings and the deformation is Lipschitz-smooth in `t`.

use crate::encoding::positional::{PosEncConfig, positional_encode};
use crate::error::{NerfError, NerfResult};
use crate::handle::LcgRng;
use crate::rendering::gaussian_splat_3d::{Gaussian3d, SplatCamera, SplatImage, rasterize};

/// Number of scalar outputs predicted per Gaussian: `δμ(3) + δs(3) + δr(4)`.
const DEFORM_OUTPUTS: usize = 10;

/// Lower bound enforced on deformed scale components (keeps them positive).
const MIN_SCALE: f32 = 1.0e-4;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the deformation field.
#[derive(Debug, Clone)]
pub struct DeformationConfig {
    /// Fourier frequency levels for the canonical position embedding (3-D).
    pub pos_freq: usize,
    /// Fourier frequency levels for the scalar time embedding (1-D).
    pub time_freq: usize,
    /// Hidden width of the deformation MLP.
    pub hidden_dim: usize,
    /// Number of hidden ReLU layers in the deformation MLP.
    pub n_hidden_layers: usize,
    /// Canonical time `t_c` at which the deformation is anchored to zero.
    pub canonical_time: f32,
    /// Global multiplier on the predicted offsets (keeps deformations modest).
    pub deform_scale: f32,
}

impl Default for DeformationConfig {
    fn default() -> Self {
        Self {
            pos_freq: 4,
            time_freq: 4,
            hidden_dim: 32,
            n_hidden_layers: 2,
            canonical_time: 0.0,
            deform_scale: 0.1,
        }
    }
}

// ─── Per-Gaussian deformation delta ──────────────────────────────────────────

/// Anchored deformation applied to one canonical Gaussian at a given time.
#[derive(Debug, Clone, Copy, Default)]
pub struct GaussianDelta {
    /// Position offset `δμ`.
    pub d_position: [f32; 3],
    /// Scale offset `δs`.
    pub d_scale: [f32; 3],
    /// Rotation (quaternion) offset `δr`.
    pub d_quaternion: [f32; 4],
}

// ─── DeformationField (MLP) ──────────────────────────────────────────────────

/// The deformation MLP `Φ`: `[γ(x) ⊕ γ(t)] → hidden ReLU layers → 10 outputs`.
#[derive(Debug, Clone)]
pub struct DeformationField {
    pos_cfg: PosEncConfig,
    time_cfg: PosEncConfig,
    /// Hidden `(weight, bias)` layers, each followed by ReLU.
    layers: Vec<(Vec<f32>, Vec<f32>)>,
    /// Output projection `hidden → 10` (linear).
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    in_dim: usize,
    hidden: usize,
}

impl DeformationField {
    /// Build a randomly-initialised deformation field.
    ///
    /// # Errors
    ///
    /// Returns [`NerfError::InvalidFreqLevels`] for zero frequency levels or
    /// [`NerfError::InvalidFeatureDim`] for a zero hidden width.
    pub fn new(cfg: &DeformationConfig, rng: &mut LcgRng) -> NerfResult<Self> {
        if cfg.pos_freq == 0 || cfg.time_freq == 0 {
            return Err(NerfError::InvalidFreqLevels {
                levels: cfg.pos_freq.min(cfg.time_freq),
            });
        }
        if cfg.hidden_dim == 0 {
            return Err(NerfError::InvalidFeatureDim { dim: 0 });
        }
        let pos_cfg = PosEncConfig {
            n_freq: cfg.pos_freq,
            include_input: true,
            input_dim: 3,
        };
        let time_cfg = PosEncConfig {
            n_freq: cfg.time_freq,
            include_input: true,
            input_dim: 1,
        };
        let in_dim = pos_cfg.output_dim() + time_cfg.output_dim();
        let hidden = cfg.hidden_dim;

        let mut layers = Vec::with_capacity(cfg.n_hidden_layers.max(1));
        let mut prev = in_dim;
        for _ in 0..cfg.n_hidden_layers.max(1) {
            layers.push(make_layer(prev, hidden, rng));
            prev = hidden;
        }
        let (out_w, out_b) = make_layer(prev, DEFORM_OUTPUTS, rng);

        Ok(Self {
            pos_cfg,
            time_cfg,
            layers,
            out_w,
            out_b,
            in_dim,
            hidden,
        })
    }

    /// Fourier embedding `γ(t)` of a scalar time (reuses [`positional_encode`]).
    ///
    /// # Errors
    ///
    /// Propagates [`positional_encode`] failures.
    pub fn time_embedding(&self, t: f32) -> NerfResult<Vec<f32>> {
        positional_encode(&[t], &self.time_cfg)
    }

    /// Raw (un-anchored) MLP prediction `Φ(x, t)`.
    ///
    /// # Errors
    ///
    /// Propagates positional-encoding failures, or returns
    /// [`NerfError::Internal`] if the output projection is malformed.
    pub fn forward(&self, position: [f32; 3], t: f32) -> NerfResult<[f32; DEFORM_OUTPUTS]> {
        let pe_pos = positional_encode(&position, &self.pos_cfg)?;
        let pe_t = positional_encode(&[t], &self.time_cfg)?;
        let mut input = Vec::with_capacity(self.in_dim);
        input.extend_from_slice(&pe_pos);
        input.extend_from_slice(&pe_t);

        let mut act = input;
        for (w, b) in &self.layers {
            act = fc_relu(&act, w, b, self.hidden);
        }
        let out = fc_linear(&act, &self.out_w, &self.out_b, DEFORM_OUTPUTS);
        out.try_into().map_err(|_| NerfError::Internal {
            msg: "deformation output projection produced wrong arity".into(),
        })
    }
}

// ─── DeformableGaussians ─────────────────────────────────────────────────────

/// Canonical Gaussians plus a deformation field, renderable at any time.
#[derive(Debug, Clone)]
pub struct DeformableGaussians {
    canonical: Vec<Gaussian3d>,
    field: DeformationField,
    canonical_time: f32,
    deform_scale: f32,
}

impl DeformableGaussians {
    /// Build a deformable Gaussian model over a canonical template.
    ///
    /// # Errors
    ///
    /// Returns [`NerfError::EmptyInput`] for an empty template, or propagates
    /// [`DeformationField::new`] errors.
    pub fn new(
        canonical: Vec<Gaussian3d>,
        cfg: &DeformationConfig,
        rng: &mut LcgRng,
    ) -> NerfResult<Self> {
        if canonical.is_empty() {
            return Err(NerfError::EmptyInput);
        }
        let field = DeformationField::new(cfg, rng)?;
        Ok(Self {
            canonical,
            field,
            canonical_time: cfg.canonical_time,
            deform_scale: cfg.deform_scale,
        })
    }

    /// Number of canonical Gaussians.
    #[must_use]
    pub fn len(&self) -> usize {
        self.canonical.len()
    }

    /// Whether the template is empty (always `false` after construction).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.canonical.is_empty()
    }

    /// Borrow the canonical (undeformed) Gaussians.
    #[must_use]
    pub fn canonical(&self) -> &[Gaussian3d] {
        &self.canonical
    }

    /// Borrow the deformation field.
    #[must_use]
    pub fn field(&self) -> &DeformationField {
        &self.field
    }

    /// Anchored per-Gaussian deformation at time `t`.
    ///
    /// Returns `scale · (Φ(x, t) − Φ(x, t_c))` for every canonical Gaussian,
    /// guaranteeing an exactly-zero delta at the canonical time.
    ///
    /// # Errors
    ///
    /// Propagates deformation-field evaluation failures.
    pub fn deltas(&self, t: f32) -> NerfResult<Vec<GaussianDelta>> {
        let mut out = Vec::with_capacity(self.canonical.len());
        for g in &self.canonical {
            let phi_t = self.field.forward(g.position, t)?;
            let phi_c = self.field.forward(g.position, self.canonical_time)?;
            let mut delta = [0.0_f32; DEFORM_OUTPUTS];
            for (d, (&a, &b)) in delta.iter_mut().zip(phi_t.iter().zip(phi_c.iter())) {
                *d = self.deform_scale * (a - b);
            }
            out.push(GaussianDelta {
                d_position: [delta[0], delta[1], delta[2]],
                d_scale: [delta[3], delta[4], delta[5]],
                d_quaternion: [delta[6], delta[7], delta[8], delta[9]],
            });
        }
        Ok(out)
    }

    /// Deform the canonical Gaussians to time `t`.
    ///
    /// # Errors
    ///
    /// Propagates deformation evaluation, or [`Gaussian3d::new`] validation.
    pub fn deform(&self, t: f32) -> NerfResult<Vec<Gaussian3d>> {
        let deltas = self.deltas(t)?;
        let mut out = Vec::with_capacity(self.canonical.len());
        for (g, d) in self.canonical.iter().zip(deltas.iter()) {
            let position = [
                g.position[0] + d.d_position[0],
                g.position[1] + d.d_position[1],
                g.position[2] + d.d_position[2],
            ];
            let scale = [
                (g.scale[0] + d.d_scale[0]).max(MIN_SCALE),
                (g.scale[1] + d.d_scale[1]).max(MIN_SCALE),
                (g.scale[2] + d.d_scale[2]).max(MIN_SCALE),
            ];
            let quaternion = normalize_quat([
                g.quaternion[0] + d.d_quaternion[0],
                g.quaternion[1] + d.d_quaternion[1],
                g.quaternion[2] + d.d_quaternion[2],
                g.quaternion[3] + d.d_quaternion[3],
            ]);
            out.push(Gaussian3d::new(
                position, scale, quaternion, g.opacity, g.color,
            )?);
        }
        Ok(out)
    }

    /// Deform to time `t` and rasterise with the static 3DGS rasteriser.
    ///
    /// # Errors
    ///
    /// Propagates [`DeformableGaussians::deform`] and [`rasterize`] errors.
    pub fn render(&self, t: f32, cam: &SplatCamera) -> NerfResult<SplatImage> {
        let deformed = self.deform(t)?;
        rasterize(&deformed, cam)
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Normalise a quaternion; a degenerate (near-zero) quaternion maps to identity.
fn normalize_quat(q: [f32; 4]) -> [f32; 4] {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if norm > 1.0e-12 {
        [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    }
}

/// Xavier/He-style normal initialisation scaled by `sqrt(2/fan_in)`.
fn xavier_fill(buf: &mut [f32], fan_in: usize, rng: &mut LcgRng) {
    let scale = (2.0_f32 / fan_in.max(1) as f32).sqrt();
    let mut i = 0;
    while i + 1 < buf.len() {
        let (a, b) = rng.next_normal_pair();
        buf[i] = a * scale;
        buf[i + 1] = b * scale;
        i += 2;
    }
    if i < buf.len() {
        let (a, _) = rng.next_normal_pair();
        buf[i] = a * scale;
    }
}

fn make_layer(in_dim: usize, out_dim: usize, rng: &mut LcgRng) -> (Vec<f32>, Vec<f32>) {
    let mut w = vec![0.0_f32; out_dim * in_dim];
    xavier_fill(&mut w, in_dim, rng);
    (w, vec![0.0_f32; out_dim])
}

/// Fully-connected layer `w·x + b` followed by ReLU.
fn fc_relu(x: &[f32], w: &[f32], b: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    let mut out = vec![0.0_f32; out_dim];
    for (o, (row, &bias)) in out.iter_mut().zip(w.chunks(in_dim).zip(b.iter())) {
        *o = (row
            .iter()
            .zip(x.iter())
            .map(|(&wi, &xi)| wi * xi)
            .sum::<f32>()
            + bias)
            .max(0.0);
    }
    out
}

/// Fully-connected layer `w·x + b`, no activation.
fn fc_linear(x: &[f32], w: &[f32], b: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    let mut out = vec![0.0_f32; out_dim];
    for (o, (row, &bias)) in out.iter_mut().zip(w.chunks(in_dim).zip(b.iter())) {
        *o = row
            .iter()
            .zip(x.iter())
            .map(|(&wi, &xi)| wi * xi)
            .sum::<f32>()
            + bias;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rendering::ray::PinholeCamera;

    fn unit_quat() -> [f32; 4] {
        [1.0, 0.0, 0.0, 0.0]
    }

    fn make_canonical() -> Vec<Gaussian3d> {
        vec![
            Gaussian3d::new(
                [0.0, 0.0, 3.0],
                [0.2, 0.2, 0.2],
                unit_quat(),
                0.8,
                [0.9, 0.1, 0.1],
            )
            .expect("value should be present"),
            Gaussian3d::new(
                [0.3, -0.2, 3.5],
                [0.15, 0.25, 0.2],
                unit_quat(),
                0.7,
                [0.1, 0.8, 0.2],
            )
            .expect("value should be present"),
            Gaussian3d::new(
                [-0.25, 0.1, 4.0],
                [0.2, 0.18, 0.22],
                unit_quat(),
                0.6,
                [0.2, 0.3, 0.9],
            )
            .expect("value should be present"),
        ]
    }

    fn make_model(seed: u64) -> DeformableGaussians {
        let cfg = DeformationConfig {
            pos_freq: 4,
            time_freq: 4,
            hidden_dim: 24,
            n_hidden_layers: 2,
            canonical_time: 0.0,
            deform_scale: 0.1,
        };
        let mut rng = LcgRng::new(seed);
        DeformableGaussians::new(make_canonical(), &cfg, &mut rng).expect("value should be present")
    }

    fn max_pos_offset(a: &[Gaussian3d], b: &[Gaussian3d]) -> f32 {
        a.iter()
            .zip(b.iter())
            .flat_map(|(g, h)| (0..3).map(move |k| (g.position[k] - h.position[k]).abs()))
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn canonical_time_deformation_is_zero() {
        let model = make_model(1);
        let deltas = model.deltas(0.0).expect("deltas should succeed");
        for d in &deltas {
            for v in d.d_position {
                assert!(v.abs() < 1e-6, "δμ must vanish at canonical time, got {v}");
            }
            for v in d.d_scale {
                assert!(v.abs() < 1e-6, "δs must vanish at canonical time, got {v}");
            }
            for v in d.d_quaternion {
                assert!(v.abs() < 1e-6, "δr must vanish at canonical time, got {v}");
            }
        }
        // Deformed == canonical at t_c (unit quats, scale > MIN).
        let deformed = model.deform(0.0).expect("deform should succeed");
        let off = max_pos_offset(&deformed, model.canonical());
        assert!(
            off < 1e-6,
            "deform(t_c) must equal canonical, max offset {off}"
        );
    }

    #[test]
    fn different_time_changes_positions() {
        let model = make_model(2);
        let at0 = model.deform(0.0).expect("deform should succeed");
        let at1 = model.deform(0.7).expect("deform should succeed");
        let off = max_pos_offset(&at0, &at1);
        assert!(
            off > 1e-4,
            "distinct times must deform positions, max offset {off}"
        );
    }

    #[test]
    fn deformation_is_smooth_in_time() {
        // Nearby times must produce nearby offsets relative to a far time.
        let model = make_model(3);
        let base = model.deform(0.30).expect("deform should succeed");
        let near = model.deform(0.31).expect("deform should succeed"); // Δt = 0.01
        let far = model.deform(0.55).expect("deform should succeed"); // Δt = 0.25
        let near_change = max_pos_offset(&base, &near);
        let far_change = max_pos_offset(&base, &far);
        assert!(
            near_change < far_change,
            "nearby time change {near_change} should be smaller than far {far_change}"
        );
        // Continuity: a 10× smaller step yields a much smaller change.
        let tiny = model.deform(0.301).expect("deform should succeed"); // Δt = 0.001
        let tiny_change = max_pos_offset(&base, &tiny);
        assert!(
            tiny_change < near_change,
            "Δt=0.001 change {tiny_change} should be below Δt=0.01 change {near_change}"
        );
    }

    #[test]
    fn time_positional_encoding_works() {
        let model = make_model(4);
        let cfg_dim = PosEncConfig {
            n_freq: 4,
            include_input: true,
            input_dim: 1,
        }
        .output_dim();
        let e0 = model
            .field()
            .time_embedding(0.0)
            .expect("time_embedding should succeed");
        let e1 = model
            .field()
            .time_embedding(0.5)
            .expect("time_embedding should succeed");
        assert_eq!(e0.len(), cfg_dim, "time embedding has 2·L+1 entries");
        assert_eq!(e1.len(), cfg_dim);
        // Distinct times → distinct embeddings.
        let diff: f32 = e0.iter().zip(e1.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-3,
            "embeddings of distinct times must differ, Σ|Δ|={diff}"
        );
    }

    #[test]
    fn deform_preserves_count_and_appearance() {
        let model = make_model(5);
        let deformed = model.deform(0.4).expect("deform should succeed");
        assert_eq!(deformed.len(), model.len());
        assert_eq!(model.len(), 3);
        assert!(!model.is_empty());
        // Opacity and colour are not deformed.
        for (g, c) in deformed.iter().zip(model.canonical().iter()) {
            assert_eq!(g.opacity, c.opacity);
            assert_eq!(g.color, c.color);
        }
    }

    #[test]
    fn deformed_fields_are_finite_and_valid() {
        let model = make_model(6);
        for &t in &[0.0_f32, 0.2, 0.5, 0.9, 1.0] {
            let deformed = model.deform(t).expect("deform should succeed");
            for g in &deformed {
                assert!(g.position.iter().all(|v| v.is_finite()));
                assert!(g.scale.iter().all(|v| v.is_finite() && *v > 0.0));
                assert!(g.quaternion.iter().all(|v| v.is_finite()));
                // Quaternion is unit-norm.
                let n: f32 = g.quaternion.iter().map(|v| v * v).sum();
                assert!(
                    (n - 1.0).abs() < 1e-4,
                    "quaternion must stay unit, |q|²={n}"
                );
            }
        }
    }

    #[test]
    fn deformation_is_deterministic() {
        let model = make_model(7);
        let a = model.deform(0.33).expect("deform should succeed");
        let b = model.deform(0.33).expect("deform should succeed");
        for (g, h) in a.iter().zip(b.iter()) {
            assert_eq!(g.position, h.position);
            assert_eq!(g.scale, h.scale);
            assert_eq!(g.quaternion, h.quaternion);
        }
    }

    #[test]
    fn render_deformed_scene_is_finite() {
        let model = make_model(8);
        let intr = PinholeCamera::new(16.0, 16.0, 8.0, 8.0, 16, 16).expect("new should succeed");
        let cam = SplatCamera::identity(intr, 0.01).expect("identity should succeed");
        let img0 = model.render(0.0, &cam).expect("render should succeed");
        let img1 = model.render(0.6, &cam).expect("render should succeed");
        assert_eq!(img0.rgb.len(), 16 * 16 * 3);
        assert!(img0.rgb.iter().all(|v| v.is_finite()));
        assert!(img1.rgb.iter().all(|v| v.is_finite()));
        // Dynamic scene: deformation generally changes the rendered image.
        let diff: f32 = img0
            .rgb
            .iter()
            .zip(img1.rgb.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff.is_finite());
    }
}
