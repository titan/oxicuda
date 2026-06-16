//! pi-GAN — Periodic Implicit Generative Adversarial Networks.
//!
//! Chan, Monteiro, Kellnhofer, Wu & Wetzstein (2021), "pi-GAN: Periodic Implicit
//! Generative Adversarial Networks for 3D-Aware Image Synthesis", CVPR.
//!
//! pi-GAN is a *generative* neural radiance field: a single network produces a
//! whole *family* of 3-D scenes, indexed by a latent code `z`. Its generator is
//! a **FiLM-conditioned SIREN**:
//!
//! ```text
//! h_0      = coordinate (x, y, z)
//! h_{l+1}  = sin( γ_l ⊙ (W_l h_l) + β_l )          (FiLM-SIREN layer)
//! ```
//!
//! The per-layer frequencies `γ_l` and phase shifts `β_l` are *not* free
//! parameters: a **mapping network** predicts them from the latent code `z`,
//!
//! ```text
//! (γ_0, β_0, …, γ_{L-1}, β_{L-1}) = M(z),
//! ```
//!
//! so the latent code reshapes the periodic activations of every layer (FiLM =
//! Feature-wise Linear Modulation). The sinusoidal activations are bounded in
//! `[-1, 1]`, and — following SIREN — `γ_l` is centred on a base frequency
//! `ω₀`, which controls how high-frequency the implicit field can be.
//!
//! On top of the synthesis backbone two heads form the generative radiance field:
//! a softplus **density** head (a function of position only) and a sigmoid
//! **colour** head that additionally consumes the viewing direction. Sampling a
//! new `z` yields a new density/colour field — the source of generative
//! diversity — while a *fixed* `z` is fully deterministic.
//!
//! This is a faithful, compact CPU core of the generator (the GAN discriminator
//! and adversarial training loop are out of scope). The volume-rendering
//! integral and [`Ray`] are reused from the crate.

use crate::error::{NerfError, NerfResult};
use crate::handle::LcgRng;
use crate::rendering::ray::Ray;
use crate::rendering::volume_render::{RenderResult, volume_render};

/// Spatial input dimensionality of the generator (a 3-D coordinate).
const COORD_DIM: usize = 3;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for a [`PiGan`] generator.
#[derive(Debug, Clone)]
pub struct PiGanConfig {
    /// Dimensionality of the latent code `z`.
    pub latent_dim: usize,
    /// Width of every FiLM-SIREN synthesis layer.
    pub hidden_dim: usize,
    /// Number of FiLM-SIREN synthesis layers `L`.
    pub n_synthesis_layers: usize,
    /// Hidden width of the latent mapping network.
    pub mapping_hidden: usize,
    /// Number of hidden ReLU layers in the mapping network.
    pub n_mapping_layers: usize,
    /// SIREN base frequency `ω₀` (the centre of the predicted `γ`).
    pub omega_0: f32,
}

impl Default for PiGanConfig {
    fn default() -> Self {
        Self {
            latent_dim: 16,
            hidden_dim: 32,
            n_synthesis_layers: 3,
            mapping_hidden: 32,
            n_mapping_layers: 2,
            omega_0: 30.0,
        }
    }
}

// ─── FiLM parameters ─────────────────────────────────────────────────────────

/// Per-layer FiLM modulation: a frequency vector `γ` and a phase vector `β`,
/// each of length `hidden_dim`.
#[derive(Debug, Clone)]
pub struct FilmParams {
    /// Per-feature frequency scale `γ_l` (centred on `ω₀`).
    pub gamma: Vec<f32>,
    /// Per-feature phase shift `β_l`.
    pub beta: Vec<f32>,
}

// ─── Mapping network ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct MappingNetwork {
    layers: Vec<(Vec<f32>, Vec<f32>)>,
    out: (Vec<f32>, Vec<f32>),
    hidden: usize,
    latent_dim: usize,
    n_layers: usize,
    width: usize,
    omega_0: f32,
}

impl MappingNetwork {
    fn new(cfg: &PiGanConfig, rng: &mut LcgRng) -> Self {
        let mh = cfg.mapping_hidden;
        let mut layers = Vec::with_capacity(cfg.n_mapping_layers.max(1));
        let mut prev = cfg.latent_dim;
        for _ in 0..cfg.n_mapping_layers.max(1) {
            layers.push(make_layer(prev, mh, rng));
            prev = mh;
        }
        // Two outputs (γ, β) per feature, per synthesis layer.
        let out_dim = 2 * cfg.n_synthesis_layers * cfg.hidden_dim;
        Self {
            layers,
            out: make_layer(prev, out_dim, rng),
            hidden: mh,
            latent_dim: cfg.latent_dim,
            n_layers: cfg.n_synthesis_layers,
            width: cfg.hidden_dim,
            omega_0: cfg.omega_0,
        }
    }

    fn forward(&self, z: &[f32]) -> NerfResult<Vec<FilmParams>> {
        if z.len() != self.latent_dim {
            return Err(NerfError::DimensionMismatch {
                expected: self.latent_dim,
                got: z.len(),
            });
        }
        let mut act = z.to_vec();
        for (w, b) in &self.layers {
            act = fc_relu(&act, w, b, self.hidden);
        }
        let raw = fc_linear(
            &act,
            &self.out.0,
            &self.out.1,
            2 * self.n_layers * self.width,
        );

        // Split into per-layer (γ, β). γ is centred on ω₀ so the synthesis layers
        // start out at the SIREN base frequency; β is an unconstrained phase.
        let mut params = Vec::with_capacity(self.n_layers);
        for l in 0..self.n_layers {
            let base = l * 2 * self.width;
            let gamma: Vec<f32> = (0..self.width)
                .map(|i| self.omega_0 * (1.0 + raw[base + i].tanh()))
                .collect();
            let beta: Vec<f32> = (0..self.width)
                .map(|i| raw[base + self.width + i])
                .collect();
            params.push(FilmParams { gamma, beta });
        }
        Ok(params)
    }
}

// ─── Synthesis (FiLM-SIREN) ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Synthesis {
    /// Per-layer linear weights (no bias — `β` is the phase / bias).
    weights: Vec<Vec<f32>>,
    n_layers: usize,
    width: usize,
}

impl Synthesis {
    fn new(cfg: &PiGanConfig, rng: &mut LcgRng) -> Self {
        let w = cfg.hidden_dim;
        let mut weights = Vec::with_capacity(cfg.n_synthesis_layers);
        // First layer: COORD_DIM → width with the SIREN first-layer init U(-1/n, 1/n).
        weights.push(siren_first(w, COORD_DIM, rng));
        // Hidden layers: width → width with the SIREN hidden init.
        for _ in 1..cfg.n_synthesis_layers {
            weights.push(siren_hidden(w, w, cfg.omega_0, rng));
        }
        Self {
            weights,
            n_layers: cfg.n_synthesis_layers,
            width: w,
        }
    }

    /// Run the FiLM-SIREN stack, returning the final feature and every hidden
    /// activation (each in `[-1, 1]` because of the `sin` non-linearity).
    fn forward(
        &self,
        coord: [f32; 3],
        film: &[FilmParams],
    ) -> NerfResult<(Vec<f32>, Vec<Vec<f32>>)> {
        if film.len() != self.n_layers {
            return Err(NerfError::DimensionMismatch {
                expected: self.n_layers,
                got: film.len(),
            });
        }
        let mut activations = Vec::with_capacity(self.n_layers);
        let mut h: Vec<f32> = coord.to_vec();

        for (l, w) in self.weights.iter().enumerate() {
            let in_dim = h.len();
            let params = &film[l];
            if params.gamma.len() != self.width || params.beta.len() != self.width {
                return Err(NerfError::DimensionMismatch {
                    expected: self.width,
                    got: params.gamma.len(),
                });
            }
            let mut next = vec![0.0_f32; self.width];
            for (j, (wo, slot)) in w.chunks(in_dim).zip(next.iter_mut()).enumerate() {
                let lin = wo
                    .iter()
                    .zip(h.iter())
                    .map(|(&wi, &xi)| wi * xi)
                    .sum::<f32>();
                // h_{l+1} = sin( γ ⊙ (W h) + β )
                *slot = (params.gamma[j] * lin + params.beta[j]).sin();
            }
            activations.push(next.clone());
            h = next;
        }
        Ok((h, activations))
    }
}

// ─── PiGan ───────────────────────────────────────────────────────────────────

/// pi-GAN generator: a latent mapping network feeding a FiLM-SIREN synthesis
/// backbone with density and colour heads.
#[derive(Debug, Clone)]
pub struct PiGan {
    mapping: MappingNetwork,
    synthesis: Synthesis,
    density: (Vec<f32>, Vec<f32>),
    color: (Vec<f32>, Vec<f32>),
    latent_dim: usize,
    hidden_dim: usize,
    n_layers: usize,
}

impl PiGan {
    /// Build a randomly-initialised pi-GAN generator.
    ///
    /// # Errors
    ///
    /// Returns [`NerfError::InvalidFeatureDim`] if any width / layer count is
    /// zero, or [`NerfError::Internal`] for a non-finite `ω₀`.
    pub fn new(cfg: PiGanConfig, rng: &mut LcgRng) -> NerfResult<Self> {
        if cfg.latent_dim == 0
            || cfg.hidden_dim == 0
            || cfg.n_synthesis_layers == 0
            || cfg.mapping_hidden == 0
        {
            return Err(NerfError::InvalidFeatureDim { dim: 0 });
        }
        if !cfg.omega_0.is_finite() || cfg.omega_0 <= 0.0 {
            return Err(NerfError::Internal {
                msg: "omega_0 must be finite and positive".into(),
            });
        }

        let mapping = MappingNetwork::new(&cfg, rng);
        let synthesis = Synthesis::new(&cfg, rng);
        let density = make_layer(cfg.hidden_dim, 1, rng);
        // Colour head consumes [feature ⊕ view direction].
        let color = make_layer(cfg.hidden_dim + 3, 3, rng);

        Ok(Self {
            mapping,
            synthesis,
            density,
            color,
            latent_dim: cfg.latent_dim,
            hidden_dim: cfg.hidden_dim,
            n_layers: cfg.n_synthesis_layers,
        })
    }

    /// Latent dimensionality `dim(z)`.
    #[must_use]
    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    /// Width of every synthesis layer.
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Number of FiLM-SIREN synthesis layers.
    #[must_use]
    pub fn n_synthesis_layers(&self) -> usize {
        self.n_layers
    }

    /// Sample a latent code from the standard normal `N(0, I)`.
    #[must_use]
    pub fn sample_latent(&self, rng: &mut LcgRng) -> Vec<f32> {
        let mut z = vec![0.0_f32; self.latent_dim];
        rng.fill_normal(&mut z);
        z
    }

    /// Predict the per-layer FiLM parameters `{(γ_l, β_l)}` from latent `z`.
    ///
    /// # Errors
    ///
    /// Returns [`NerfError::DimensionMismatch`] if `z.len() != latent_dim`.
    pub fn mapping_forward(&self, z: &[f32]) -> NerfResult<Vec<FilmParams>> {
        self.mapping.forward(z)
    }

    /// Run the FiLM-SIREN synthesis backbone under explicit FiLM parameters,
    /// returning only the final feature.
    ///
    /// # Errors
    ///
    /// Propagates synthesis shape errors.
    pub fn synthesize(&self, coord: [f32; 3], film: &[FilmParams]) -> NerfResult<Vec<f32>> {
        Ok(self.synthesis.forward(coord, film)?.0)
    }

    /// Run the synthesis backbone returning the final feature and all hidden
    /// activations (each bounded in `[-1, 1]`).
    ///
    /// # Errors
    ///
    /// Propagates synthesis shape errors.
    pub fn synthesize_with_activations(
        &self,
        coord: [f32; 3],
        film: &[FilmParams],
    ) -> NerfResult<(Vec<f32>, Vec<Vec<f32>>)> {
        self.synthesis.forward(coord, film)
    }

    /// Evaluate the generative radiance field at `coord` for viewing direction
    /// `view_dir`, conditioned on latent `z`: returns `(σ, rgb)`.
    ///
    /// # Errors
    ///
    /// Propagates mapping / synthesis failures.
    pub fn field(
        &self,
        coord: [f32; 3],
        view_dir: [f32; 3],
        z: &[f32],
    ) -> NerfResult<(f32, [f32; 3])> {
        let film = self.mapping.forward(z)?;
        let feature = self.synthesize(coord, &film)?;
        Ok(self.decode(&feature, view_dir))
    }

    fn decode(&self, feature: &[f32], view_dir: [f32; 3]) -> (f32, [f32; 3]) {
        let sigma = softplus(fc_linear(feature, &self.density.0, &self.density.1, 1)[0]);
        let mut color_in = Vec::with_capacity(feature.len() + 3);
        color_in.extend_from_slice(feature);
        color_in.extend_from_slice(&view_dir);
        let rgb_raw = fc_linear(&color_in, &self.color.0, &self.color.1, 3);
        let rgb = [
            sigmoid(rgb_raw[0]),
            sigmoid(rgb_raw[1]),
            sigmoid(rgb_raw[2]),
        ];
        (sigma, rgb)
    }

    /// Volume-render a ray through the generative field for latent `z`.
    ///
    /// The viewing direction is `ray.dir`. Reuses the shared [`volume_render`]
    /// integral.
    ///
    /// # Errors
    ///
    /// Returns [`NerfError::InvalidSampleCount`] for no samples and propagates
    /// field / volume-render failures.
    pub fn render(&self, ray: &Ray, t_vals: &[f32], z: &[f32]) -> NerfResult<RenderResult> {
        if t_vals.is_empty() {
            return Err(NerfError::InvalidSampleCount { n: 0 });
        }
        let film = self.mapping.forward(z)?;
        let mut sigma = Vec::with_capacity(t_vals.len());
        let mut color = Vec::with_capacity(t_vals.len() * 3);
        for &t in t_vals {
            let feature = self.synthesize(ray.at(t), &film)?;
            let (s, rgb) = self.decode(&feature, ray.dir);
            sigma.push(s);
            color.extend_from_slice(&rgb);
        }
        volume_render(&sigma, &color, t_vals)
    }
}

// ─── Numeric helpers ─────────────────────────────────────────────────────────

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

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

/// SIREN first-layer init: `W ~ U(-1/fan_in, 1/fan_in)`.
fn siren_first(out_dim: usize, in_dim: usize, rng: &mut LcgRng) -> Vec<f32> {
    let bound = 1.0 / in_dim.max(1) as f32;
    let mut w = vec![0.0_f32; out_dim * in_dim];
    for v in &mut w {
        *v = rng.next_f32_range(-bound, bound);
    }
    w
}

/// SIREN hidden-layer init: `W ~ U(-√(6/fan_in)/ω₀, √(6/fan_in)/ω₀)` so that
/// `γ ≈ ω₀` scaling restores unit-variance pre-activations.
fn siren_hidden(out_dim: usize, in_dim: usize, omega_0: f32, rng: &mut LcgRng) -> Vec<f32> {
    let bound = (6.0_f32 / in_dim.max(1) as f32).sqrt() / omega_0;
    let mut w = vec![0.0_f32; out_dim * in_dim];
    for v in &mut w {
        *v = rng.next_f32_range(-bound, bound);
    }
    w
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

    fn tiny() -> PiGan {
        let cfg = PiGanConfig {
            latent_dim: 8,
            hidden_dim: 16,
            n_synthesis_layers: 3,
            mapping_hidden: 16,
            n_mapping_layers: 2,
            omega_0: 30.0,
        };
        let mut rng = LcgRng::new(11);
        PiGan::new(cfg, &mut rng).expect("new should succeed")
    }

    #[test]
    fn sinusoidal_activations_bounded() {
        let model = tiny();
        let mut rng = LcgRng::new(5);
        let z = model.sample_latent(&mut rng);
        let film = model
            .mapping_forward(&z)
            .expect("mapping_forward should succeed");
        let (_, acts) = model
            .synthesize_with_activations([0.3, -0.2, 0.5], &film)
            .expect("value should be present");
        assert_eq!(acts.len(), model.n_synthesis_layers());
        for layer in &acts {
            for &a in layer {
                assert!(
                    (-1.0..=1.0).contains(&a),
                    "sin activation must be in [-1, 1], got {a}"
                );
            }
        }
    }

    #[test]
    fn mapping_produces_correct_shapes() {
        let model = tiny();
        let mut rng = LcgRng::new(6);
        let z = model.sample_latent(&mut rng);
        let film = model
            .mapping_forward(&z)
            .expect("mapping_forward should succeed");
        assert_eq!(film.len(), model.n_synthesis_layers());
        for p in &film {
            assert_eq!(p.gamma.len(), model.hidden_dim());
            assert_eq!(p.beta.len(), model.hidden_dim());
            // γ is centred on ω₀ (= 30) and stays positive (frequency).
            assert!(p.gamma.iter().all(|&g| g > 0.0 && g.is_finite()));
            assert!(p.beta.iter().all(|&b| b.is_finite()));
        }
    }

    #[test]
    fn film_modulation_changes_output() {
        let model = tiny();
        let mut rng = LcgRng::new(9);
        let z = model.sample_latent(&mut rng);
        let film = model
            .mapping_forward(&z)
            .expect("mapping_forward should succeed");

        // Perturb the FiLM frequency / phase of the first layer.
        let mut film2 = film.clone();
        for g in &mut film2[0].gamma {
            *g += 2.5;
        }
        for b in &mut film2[0].beta {
            *b += 1.0;
        }

        let coord = [0.21, -0.34, 0.12];
        let out1 = model
            .synthesize(coord, &film)
            .expect("synthesize should succeed");
        let out2 = model
            .synthesize(coord, &film2)
            .expect("synthesize should succeed");
        let changed = out1.iter().zip(&out2).any(|(a, b)| (a - b).abs() > 1e-5);
        assert!(
            changed,
            "different FiLM (γ, β) must change the synthesis output"
        );
    }

    #[test]
    fn latent_changes_generated_field() {
        let model = tiny();
        let mut rng = LcgRng::new(13);
        let z1 = model.sample_latent(&mut rng);
        let z2 = model.sample_latent(&mut rng);
        let coord = [0.15, 0.25, -0.35];
        let dir = [0.0, 0.0, 1.0];
        let (s1, c1) = model.field(coord, dir, &z1).expect("field should succeed");
        let (s2, c2) = model.field(coord, dir, &z2).expect("field should succeed");
        let changed = (s1 - s2).abs() > 1e-6 || (0..3).any(|k| (c1[k] - c2[k]).abs() > 1e-6);
        assert!(changed, "different latent z must change the rendered field");
    }

    #[test]
    fn field_is_deterministic_under_fixed_z() {
        let model = tiny();
        let mut rng = LcgRng::new(21);
        let z = model.sample_latent(&mut rng);
        let coord = [0.4, -0.1, 0.2];
        let dir = [0.1, 0.2, 0.9];
        let a = model.field(coord, dir, &z).expect("field should succeed");
        let b = model.field(coord, dir, &z).expect("field should succeed");
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
    }

    #[test]
    fn field_is_finite_and_in_range() {
        let model = tiny();
        let mut rng = LcgRng::new(33);
        let z = model.sample_latent(&mut rng);
        let (sigma, rgb) = model
            .field([0.5, 0.5, 0.5], [0.0, 0.0, 1.0], &z)
            .expect("field should succeed");
        assert!(sigma.is_finite() && sigma >= 0.0);
        for c in rgb {
            assert!(c.is_finite() && (0.0..=1.0).contains(&c));
        }
    }

    #[test]
    fn render_produces_valid_result() {
        let model = tiny();
        let mut rng = LcgRng::new(44);
        let z = model.sample_latent(&mut rng);
        let ray =
            Ray::normalized([0.0, 0.0, -2.0], [0.0, 0.0, 1.0]).expect("normalized should succeed");
        let t_vals: Vec<f32> = (0..12).map(|i| 1.0 + i as f32 * 0.1).collect();
        let res = model
            .render(&ray, &t_vals, &z)
            .expect("render should succeed");
        assert!((0.0..=1.000_01).contains(&res.opacity));
        assert!(res.depth.is_finite());
        for c in res.rgb {
            assert!(c.is_finite() && (0.0..=1.0).contains(&c));
        }
    }

    #[test]
    fn rejects_bad_config_and_latent() {
        let mut rng = LcgRng::new(1);
        let bad = PiGanConfig {
            hidden_dim: 0,
            ..PiGanConfig::default()
        };
        assert!(PiGan::new(bad, &mut rng).is_err());

        let model = tiny();
        // Wrong latent length is rejected.
        assert!(model.mapping_forward(&[0.0_f32; 3]).is_err());
    }
}
