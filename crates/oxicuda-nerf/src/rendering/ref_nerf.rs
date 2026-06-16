//! Ref-NeRF — structured view-dependent appearance via reflection reparameterisation.
//!
//! Verbin, Hedman, Mildenhall, Srinivasan, Zhang & Barron (2022),
//! "Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance Fields",
//! CVPR (oral).
//!
//! A standard NeRF parameterises view-dependent colour directly by the *viewing*
//! direction `ω_o`, which forces the MLP to memorise the appearance of every
//! specular highlight from scratch for every viewpoint. Ref-NeRF observes that
//! reflected radiance is most naturally indexed by the **reflection direction**
//!
//! ```text
//! ω_r = 2 (ω_o · n) n − ω_o
//! ```
//!
//! (the mirror reflection of the view direction `ω_o` about the surface normal
//! `n`). Indexing the directional appearance by `ω_r` lets a *single* learned
//! function explain a highlight that sweeps across many viewpoints, so the
//! interpolation is dramatically better.
//!
//! Two further ingredients are implemented here as faithful CPU cores:
//!
//! 1. **Integrated Directional Encoding (IDE).** The reflection direction is
//!    encoded with a spherical-harmonic basis whose per-degree amplitude is
//!    attenuated by a learned **roughness** `ρ`. Concretely each degree-`l` band
//!    is multiplied by
//!
//!    ```text
//!    A_l(ρ) = exp(−l (l + 1) ρ / 2),
//!    ```
//!
//!    the closed-form expectation of `Y_l^m` under a von-Mises–Fisher lobe of
//!    concentration `κ = 1/ρ`. A mirror-smooth surface (`ρ → 0`) keeps every
//!    harmonic; a rough surface (`ρ` large) keeps only the low-order bands, i.e.
//!    the directional encoding is *blurred*. This makes the encoding scale with
//!    material roughness, which a fixed SH encoding cannot do.
//! 2. **Diffuse / specular split.** The spatial MLP emits a *view-independent*
//!    diffuse colour `c_d`, a specular *tint* `s`, the roughness `ρ`, a surface
//!    normal `n`, and a bottleneck feature `b`. A directional MLP consumes
//!    `[IDE(ω_r, ρ) ⊕ (n · ω_o) ⊕ b]` and produces the specular colour `c_s`.
//!    The shaded colour is `clamp(c_d + s ⊙ c_s)`.
//!
//! The spherical-harmonic basis is reused verbatim from
//! [`crate::encoding::spherical_harmonics`]; the Fourier position encoding,
//! [`Ray`] and the volume-rendering integral are likewise reused.

use crate::encoding::positional::{PosEncConfig, positional_encode};
use crate::encoding::spherical_harmonics::evaluate_sh;
use crate::error::{NerfError, NerfResult};
use crate::handle::LcgRng;
use crate::rendering::ray::Ray;
use crate::rendering::volume_render::{RenderResult, volume_render};

// ─── Free geometric / encoding functions ─────────────────────────────────────

/// Mirror-reflect the view direction `omega_o` about the surface normal
/// `normal`: `ω_r = 2 (ω_o · n) n − ω_o`.
///
/// This is a Householder reflection across the plane perpendicular to `n`: it is
/// an involution (`reflect(reflect(ω_o)) = ω_o`) and preserves the normal
/// component (`ω_r · n = ω_o · n`).
#[must_use]
pub fn reflect(omega_o: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
    let two_dot = 2.0 * dot3(omega_o, normal);
    [
        two_dot * normal[0] - omega_o[0],
        two_dot * normal[1] - omega_o[1],
        two_dot * normal[2] - omega_o[2],
    ]
}

/// Per-degree IDE attenuation amplitudes `A_l(ρ) = exp(−l (l + 1) ρ / 2)` for
/// degrees `l = 0 … degree`.
///
/// `A_0 = 1` always; the sequence is strictly decreasing in `l` for any
/// `roughness > 0`, and decreasing in `roughness` for every `l ≥ 1`.
#[must_use]
pub fn attenuation_per_degree(degree: usize, roughness: f32) -> Vec<f32> {
    let rho = roughness.max(0.0);
    (0..=degree)
        .map(|l| {
            let band = (l * (l + 1)) as f32;
            (-0.5 * band * rho).exp()
        })
        .collect()
}

/// Integrated Directional Encoding of a (reflection) direction: the real
/// spherical-harmonic basis up to `degree`, with every degree-`l` band scaled by
/// the roughness attenuation `A_l(ρ)`.
///
/// The output length is `(degree + 1)²`. As `roughness` grows, the high-degree
/// entries shrink toward zero while the degree-0 entry is unchanged — a blurred,
/// roughness-aware directional encoding.
///
/// # Errors
///
/// Returns [`NerfError::ZeroRayDirection`] if the direction is degenerate and
/// propagates [`evaluate_sh`] errors (e.g. `degree > 4`).
pub fn ide_encode(direction: [f32; 3], roughness: f32, degree: usize) -> NerfResult<Vec<f32>> {
    let unit = normalize3_checked(direction)?;
    let basis = evaluate_sh(unit[0], unit[1], unit[2], degree)?;
    let amp = attenuation_per_degree(degree, roughness);

    let mut out = Vec::with_capacity(basis.len());
    let mut idx = 0;
    for (l, &a) in amp.iter().enumerate() {
        let count = 2 * l + 1;
        for _ in 0..count {
            out.push(basis[idx] * a);
            idx += 1;
        }
    }
    Ok(out)
}

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for a [`RefNerf`] model.
#[derive(Debug, Clone)]
pub struct RefNerfConfig {
    /// Fourier frequency levels for the spatial position encoding.
    pub pos_freq: usize,
    /// Maximum spherical-harmonic degree of the IDE (`0 ≤ degree ≤ 4`).
    pub sh_degree: usize,
    /// Hidden width of both the spatial and directional MLPs.
    pub hidden_dim: usize,
    /// Number of ReLU hidden layers in the spatial backbone.
    pub n_spatial_layers: usize,
    /// Number of ReLU hidden layers in the directional MLP.
    pub n_dir_layers: usize,
    /// Width of the spatial bottleneck feature fed to the directional MLP.
    pub bottleneck_dim: usize,
}

impl Default for RefNerfConfig {
    fn default() -> Self {
        Self {
            pos_freq: 4,
            sh_degree: 4,
            hidden_dim: 64,
            n_spatial_layers: 3,
            n_dir_layers: 2,
            bottleneck_dim: 16,
        }
    }
}

// ─── Spatial outputs ─────────────────────────────────────────────────────────

/// View-independent quantities emitted by the spatial MLP at a 3-D point.
#[derive(Debug, Clone)]
pub struct SpatialOutputs {
    /// Volume density `σ ≥ 0`.
    pub density: f32,
    /// Diffuse (view-independent) RGB colour in `[0, 1]³`.
    pub diffuse: [f32; 3],
    /// Specular tint `s` in `[0, 1]³` (modulates the directional colour).
    pub tint: [f32; 3],
    /// Surface roughness `ρ ≥ 0` controlling the IDE attenuation.
    pub roughness: f32,
    /// Unit surface normal `n`.
    pub normal: [f32; 3],
    /// Bottleneck feature passed to the directional MLP.
    pub bottleneck: Vec<f32>,
}

// ─── Spatial MLP (multi-head) ────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct SpatialMlp {
    pe_cfg: PosEncConfig,
    backbone: Vec<(Vec<f32>, Vec<f32>)>,
    density: (Vec<f32>, Vec<f32>),
    diffuse: (Vec<f32>, Vec<f32>),
    tint: (Vec<f32>, Vec<f32>),
    roughness: (Vec<f32>, Vec<f32>),
    normal: (Vec<f32>, Vec<f32>),
    bottleneck: (Vec<f32>, Vec<f32>),
    hidden: usize,
    bottleneck_dim: usize,
}

impl SpatialMlp {
    fn new(cfg: &RefNerfConfig, rng: &mut LcgRng) -> Self {
        let pe_cfg = PosEncConfig {
            n_freq: cfg.pos_freq,
            include_input: true,
            input_dim: 3,
        };
        let in_dim = pe_cfg.output_dim();
        let h = cfg.hidden_dim;

        let mut backbone = Vec::with_capacity(cfg.n_spatial_layers.max(1));
        let mut prev = in_dim;
        for _ in 0..cfg.n_spatial_layers.max(1) {
            backbone.push(make_layer(prev, h, rng));
            prev = h;
        }

        Self {
            pe_cfg,
            backbone,
            density: make_layer(h, 1, rng),
            diffuse: make_layer(h, 3, rng),
            tint: make_layer(h, 3, rng),
            roughness: make_layer(h, 1, rng),
            normal: make_layer(h, 3, rng),
            bottleneck: make_layer(h, cfg.bottleneck_dim, rng),
            hidden: h,
            bottleneck_dim: cfg.bottleneck_dim,
        }
    }

    fn forward(&self, position: [f32; 3]) -> NerfResult<SpatialOutputs> {
        let pe = positional_encode(&position, &self.pe_cfg)?;
        let mut act = pe;
        for (w, b) in &self.backbone {
            act = fc_relu(&act, w, b, self.hidden);
        }

        let density = softplus(fc_linear(&act, &self.density.0, &self.density.1, 1)[0]);
        let diffuse = sigmoid3(fc_linear(&act, &self.diffuse.0, &self.diffuse.1, 3));
        let tint = sigmoid3(fc_linear(&act, &self.tint.0, &self.tint.1, 3));
        let roughness = softplus(fc_linear(&act, &self.roughness.0, &self.roughness.1, 1)[0]);
        let normal_raw = fc_linear(&act, &self.normal.0, &self.normal.1, 3);
        let normal = normalize3([normal_raw[0], normal_raw[1], normal_raw[2]]);
        let bottleneck = fc_linear(
            &act,
            &self.bottleneck.0,
            &self.bottleneck.1,
            self.bottleneck_dim,
        );

        Ok(SpatialOutputs {
            density,
            diffuse,
            tint,
            roughness,
            normal,
            bottleneck,
        })
    }
}

// ─── Directional MLP ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DirectionalMlp {
    layers: Vec<(Vec<f32>, Vec<f32>)>,
    out: (Vec<f32>, Vec<f32>),
    hidden: usize,
    in_dim: usize,
}

impl DirectionalMlp {
    fn new(in_dim: usize, cfg: &RefNerfConfig, rng: &mut LcgRng) -> Self {
        let h = cfg.hidden_dim;
        let mut layers = Vec::with_capacity(cfg.n_dir_layers.max(1));
        let mut prev = in_dim;
        for _ in 0..cfg.n_dir_layers.max(1) {
            layers.push(make_layer(prev, h, rng));
            prev = h;
        }
        Self {
            layers,
            out: make_layer(prev, 3, rng),
            hidden: h,
            in_dim,
        }
    }

    fn forward(&self, input: &[f32]) -> NerfResult<[f32; 3]> {
        if input.len() != self.in_dim {
            return Err(NerfError::DimensionMismatch {
                expected: self.in_dim,
                got: input.len(),
            });
        }
        let mut act = input.to_vec();
        for (w, b) in &self.layers {
            act = fc_relu(&act, w, b, self.hidden);
        }
        Ok(sigmoid3(fc_linear(&act, &self.out.0, &self.out.1, 3)))
    }
}

// ─── RefNerf ─────────────────────────────────────────────────────────────────

/// Ref-NeRF model: a multi-head spatial MLP feeding a reflection-direction
/// directional MLP, with roughness-aware IDE.
#[derive(Debug, Clone)]
pub struct RefNerf {
    spatial: SpatialMlp,
    directional: DirectionalMlp,
    sh_degree: usize,
}

impl RefNerf {
    /// Build a randomly-initialised Ref-NeRF model.
    ///
    /// # Errors
    ///
    /// Returns [`NerfError::InvalidFreqLevels`] if `pos_freq == 0`,
    /// [`NerfError::InvalidFeatureDim`] if `sh_degree > 4` or any width is zero.
    pub fn new(cfg: RefNerfConfig, rng: &mut LcgRng) -> NerfResult<Self> {
        if cfg.pos_freq == 0 {
            return Err(NerfError::InvalidFreqLevels { levels: 0 });
        }
        if cfg.sh_degree > 4 {
            return Err(NerfError::InvalidFeatureDim { dim: cfg.sh_degree });
        }
        if cfg.hidden_dim == 0 || cfg.bottleneck_dim == 0 {
            return Err(NerfError::InvalidFeatureDim { dim: 0 });
        }

        let spatial = SpatialMlp::new(&cfg, rng);
        let ide_dim = (cfg.sh_degree + 1) * (cfg.sh_degree + 1);
        // directional input = IDE ⊕ (n · ω_o) ⊕ bottleneck
        let dir_in = ide_dim + 1 + cfg.bottleneck_dim;
        let directional = DirectionalMlp::new(dir_in, &cfg, rng);

        Ok(Self {
            spatial,
            directional,
            sh_degree: cfg.sh_degree,
        })
    }

    /// IDE spherical-harmonic degree of this model.
    #[must_use]
    pub fn sh_degree(&self) -> usize {
        self.sh_degree
    }

    /// Evaluate the view-independent spatial quantities at `position`.
    ///
    /// # Errors
    ///
    /// Propagates positional-encoding failures.
    pub fn query_spatial(&self, position: [f32; 3]) -> NerfResult<SpatialOutputs> {
        self.spatial.forward(position)
    }

    /// View-independent diffuse colour at `position`.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::query_spatial`].
    pub fn diffuse_color(&self, position: [f32; 3]) -> NerfResult<[f32; 3]> {
        Ok(self.query_spatial(position)?.diffuse)
    }

    /// Unit surface normal at `position`.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::query_spatial`].
    pub fn surface_normal(&self, position: [f32; 3]) -> NerfResult<[f32; 3]> {
        Ok(self.query_spatial(position)?.normal)
    }

    /// Reflection direction at `position` for a viewing direction `omega_o`
    /// (pointing from the surface toward the camera).
    ///
    /// # Errors
    ///
    /// Propagates [`Self::query_spatial`].
    pub fn reflection_dir(&self, position: [f32; 3], omega_o: [f32; 3]) -> NerfResult<[f32; 3]> {
        let normal = self.query_spatial(position)?.normal;
        Ok(reflect(normalize3(omega_o), normal))
    }

    /// View-dependent specular contribution `s ⊙ c_s` at `position` for viewing
    /// direction `omega_o`.
    ///
    /// Varies with `omega_o` through the reflection direction and the `n · ω_o`
    /// term; the diffuse colour (see [`Self::diffuse_color`]) does not.
    ///
    /// # Errors
    ///
    /// Propagates spatial / directional evaluation failures.
    pub fn specular_color(&self, position: [f32; 3], omega_o: [f32; 3]) -> NerfResult<[f32; 3]> {
        let spatial = self.query_spatial(position)?;
        self.specular_from_spatial(&spatial, omega_o)
    }

    fn specular_from_spatial(
        &self,
        spatial: &SpatialOutputs,
        omega_o: [f32; 3],
    ) -> NerfResult<[f32; 3]> {
        let omega_o = normalize3(omega_o);
        let omega_r = reflect(omega_o, spatial.normal);
        let ide = ide_encode(omega_r, spatial.roughness, self.sh_degree)?;

        let mut input = Vec::with_capacity(ide.len() + 1 + spatial.bottleneck.len());
        input.extend_from_slice(&ide);
        input.push(dot3(spatial.normal, omega_o));
        input.extend_from_slice(&spatial.bottleneck);

        let spec = self.directional.forward(&input)?;
        Ok([
            spatial.tint[0] * spec[0],
            spatial.tint[1] * spec[1],
            spatial.tint[2] * spec[2],
        ])
    }

    /// Full shaded colour and density at `position` viewed from direction
    /// `omega_o`: `(σ, clamp(c_d + s ⊙ c_s))`.
    ///
    /// # Errors
    ///
    /// Propagates spatial / directional evaluation failures.
    pub fn shade(&self, position: [f32; 3], omega_o: [f32; 3]) -> NerfResult<(f32, [f32; 3])> {
        let spatial = self.query_spatial(position)?;
        let spec = self.specular_from_spatial(&spatial, omega_o)?;
        let rgb = [
            (spatial.diffuse[0] + spec[0]).clamp(0.0, 1.0),
            (spatial.diffuse[1] + spec[1]).clamp(0.0, 1.0),
            (spatial.diffuse[2] + spec[2]).clamp(0.0, 1.0),
        ];
        Ok((spatial.density, rgb))
    }

    /// Volume-render a single ray, shading every sample with the Ref-NeRF model.
    ///
    /// The viewing direction is `ω_o = −ray.dir` (surface → camera). Reuses the
    /// shared [`volume_render`] integral.
    ///
    /// # Errors
    ///
    /// Returns [`NerfError::InvalidSampleCount`] for fewer than one sample and
    /// propagates shading / volume-render failures.
    pub fn render(&self, ray: &Ray, t_vals: &[f32]) -> NerfResult<RenderResult> {
        if t_vals.is_empty() {
            return Err(NerfError::InvalidSampleCount { n: 0 });
        }
        let omega_o = normalize3([-ray.dir[0], -ray.dir[1], -ray.dir[2]]);
        let mut sigma = Vec::with_capacity(t_vals.len());
        let mut color = Vec::with_capacity(t_vals.len() * 3);
        for &t in t_vals {
            let (s, rgb) = self.shade(ray.at(t), omega_o)?;
            sigma.push(s);
            color.extend_from_slice(&rgb);
        }
        volume_render(&sigma, &color, t_vals)
    }
}

// ─── Numeric helpers ─────────────────────────────────────────────────────────

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq > 1e-20 {
        let inv = 1.0 / len_sq.sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        [0.0, 0.0, 1.0]
    }
}

#[inline]
fn normalize3_checked(v: [f32; 3]) -> NerfResult<[f32; 3]> {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq < 1e-20 {
        return Err(NerfError::ZeroRayDirection);
    }
    let inv = 1.0 / len_sq.sqrt();
    Ok([v[0] * inv, v[1] * inv, v[2] * inv])
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn sigmoid3(v: Vec<f32>) -> [f32; 3] {
    [sigmoid(v[0]), sigmoid(v[1]), sigmoid(v[2])]
}

/// Numerically stable softplus `log(1 + exp(x))` (keeps density / roughness ≥ 0).
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

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

fn fc_relu(x: &[f32], w: &[f32], b: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    let mut out = vec![0.0_f32; out_dim];
    for (o, (wo, &bi)) in out.iter_mut().zip(w.chunks(in_dim).zip(b.iter())) {
        *o = (wo
            .iter()
            .zip(x.iter())
            .map(|(&wi, &xi)| wi * xi)
            .sum::<f32>()
            + bi)
            .max(0.0);
    }
    out
}

fn fc_linear(x: &[f32], w: &[f32], b: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    let mut out = vec![0.0_f32; out_dim];
    for (o, (wo, &bi)) in out.iter_mut().zip(w.chunks(in_dim).zip(b.iter())) {
        *o = wo
            .iter()
            .zip(x.iter())
            .map(|(&wi, &xi)| wi * xi)
            .sum::<f32>()
            + bi;
    }
    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> RefNerf {
        let cfg = RefNerfConfig {
            pos_freq: 4,
            sh_degree: 4,
            hidden_dim: 16,
            n_spatial_layers: 2,
            n_dir_layers: 2,
            bottleneck_dim: 8,
        };
        let mut rng = LcgRng::new(7);
        RefNerf::new(cfg, &mut rng).expect("new should succeed")
    }

    fn norm(v: [f32; 3]) -> f32 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    #[test]
    fn reflect_mirror_properties() {
        let omega_o = normalize3([0.3, -0.6, 0.7]);
        let n = normalize3([0.1, 0.9, 0.2]);
        let omega_r = reflect(omega_o, n);
        // Normal component preserved: ω_r · n = ω_o · n.
        assert!(
            (dot3(omega_r, n) - dot3(omega_o, n)).abs() < 1e-5,
            "reflection must preserve the normal component"
        );
        // Reflecting twice is the identity (involution).
        let back = reflect(omega_r, n);
        for k in 0..3 {
            assert!(
                (back[k] - omega_o[k]).abs() < 1e-5,
                "double reflection should return ω_o: {back:?} vs {omega_o:?}"
            );
        }
    }

    #[test]
    fn reflect_preserves_length() {
        let omega_o = normalize3([1.0, 2.0, -3.0]);
        let n = normalize3([-0.4, 0.5, 0.8]);
        let omega_r = reflect(omega_o, n);
        assert!(
            (norm(omega_r) - norm(omega_o)).abs() < 1e-5,
            "a reflection is an isometry"
        );
    }

    #[test]
    fn ide_attenuates_high_frequency_with_roughness() {
        let low = attenuation_per_degree(4, 0.05);
        let high = attenuation_per_degree(4, 1.0);
        // DC term is unattenuated for any roughness.
        assert!((low[0] - 1.0).abs() < 1e-6);
        assert!((high[0] - 1.0).abs() < 1e-6);
        // Monotonically non-increasing in degree.
        for w in high.windows(2) {
            assert!(w[1] <= w[0] + 1e-6, "A_l must decrease with degree");
        }
        // Higher roughness attenuates the high bands more strongly.
        assert!(high[3] < low[3], "rougher → smaller high-degree amplitude");
        assert!(high[4] < low[4]);
        // The effective order collapses: at high roughness band 4 is negligible.
        assert!(
            high[4] < 1e-3,
            "rough surface keeps only low-order harmonics"
        );
    }

    #[test]
    fn ide_encode_blurs_high_frequency() {
        let dir = normalize3([0.2, 0.3, 0.9]);
        let degree = 4;
        let sharp = ide_encode(dir, 0.02, degree).expect("ide_encode should succeed");
        let rough = ide_encode(dir, 1.5, degree).expect("ide_encode should succeed");
        assert_eq!(sharp.len(), (degree + 1) * (degree + 1));

        // Energy in bands l ≥ 2 relative to the (constant) DC term must shrink.
        let dc = sharp[0].abs().max(1e-6);
        let high_energy =
            |enc: &[f32]| -> f32 { enc[4..].iter().map(|v| v * v).sum::<f32>().sqrt() };
        let ratio_sharp = high_energy(&sharp) / dc;
        let ratio_rough = high_energy(&rough) / dc;
        assert!(
            ratio_rough < ratio_sharp,
            "roughness must blur the directional encoding: {ratio_rough} !< {ratio_sharp}"
        );
    }

    #[test]
    fn specular_varies_diffuse_constant() {
        let model = tiny();
        let p = [0.21, -0.13, 0.42];
        let view_a = normalize3([0.0, 0.0, 1.0]);
        let view_b = normalize3([0.8, 0.1, 0.3]);

        let diff_a = model
            .diffuse_color(p)
            .expect("diffuse_color should succeed");
        let diff_b = model
            .diffuse_color(p)
            .expect("diffuse_color should succeed");
        assert_eq!(diff_a, diff_b, "diffuse colour is view-independent");

        let spec_a = model
            .specular_color(p, view_a)
            .expect("specular_color should succeed");
        let spec_b = model
            .specular_color(p, view_b)
            .expect("specular_color should succeed");
        let changed = (0..3).any(|k| (spec_a[k] - spec_b[k]).abs() > 1e-6);
        assert!(changed, "specular colour must vary with view direction");
    }

    #[test]
    fn normals_are_normalized() {
        let model = tiny();
        for p in [[0.0, 0.0, 0.0], [0.5, -0.5, 0.5], [-0.9, 0.3, 0.7]] {
            let n = model
                .surface_normal(p)
                .expect("surface_normal should succeed");
            assert!((norm(n) - 1.0).abs() < 1e-5, "normal must be unit: {n:?}");
        }
    }

    #[test]
    fn shade_finite_and_in_range() {
        let model = tiny();
        let view = normalize3([0.2, 0.3, 0.9]);
        let (sigma, rgb) = model
            .shade([0.1, 0.2, -0.3], view)
            .expect("shade should succeed");
        assert!(
            sigma.is_finite() && sigma >= 0.0,
            "σ must be finite and ≥ 0"
        );
        for c in rgb {
            assert!(
                c.is_finite() && (0.0..=1.0).contains(&c),
                "rgb out of range: {c}"
            );
        }
    }

    #[test]
    fn render_produces_valid_result() {
        let model = tiny();
        let ray =
            Ray::normalized([0.0, 0.0, -2.0], [0.0, 0.0, 1.0]).expect("normalized should succeed");
        let t_vals: Vec<f32> = (0..16).map(|i| 1.0 + i as f32 * 0.1).collect();
        let res = model.render(&ray, &t_vals).expect("render should succeed");
        assert!(
            (0.0..=1.000_01).contains(&res.opacity),
            "opacity {}",
            res.opacity
        );
        assert!(res.depth.is_finite());
        for c in res.rgb {
            assert!(c.is_finite() && (0.0..=1.0).contains(&c));
        }
    }

    #[test]
    fn shading_is_deterministic() {
        let a = tiny();
        let b = tiny();
        let view = normalize3([0.3, 0.4, 0.5]);
        let (sa, ca) = a
            .shade([0.2, 0.1, 0.05], view)
            .expect("shade should succeed");
        let (sb, cb) = b
            .shade([0.2, 0.1, 0.05], view)
            .expect("shade should succeed");
        assert_eq!(sa, sb);
        assert_eq!(ca, cb);
    }

    #[test]
    fn rejects_bad_config() {
        let mut rng = LcgRng::new(1);
        let bad_freq = RefNerfConfig {
            pos_freq: 0,
            ..RefNerfConfig::default()
        };
        assert!(RefNerf::new(bad_freq, &mut rng).is_err());
        let bad_degree = RefNerfConfig {
            sh_degree: 5,
            ..RefNerfConfig::default()
        };
        assert!(RefNerf::new(bad_degree, &mut rng).is_err());
    }
}
