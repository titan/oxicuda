//! EmerNeRF — emergent static / dynamic scene decomposition.
//!
//! Yang, Chen, Wang, Liu, Karaman, Pavone & Wang (2023), "EmerNeRF: Emergent
//! Spatial-Temporal Scene Decomposition via Self-Supervision", ICLR 2024.
//!
//! EmerNeRF reconstructs a dynamic scene by splitting it, *without any
//! supervision*, into two coupled fields:
//!
//! * a **static field** `(σ_s, c_s)` that is a function of position only and is
//!   therefore time-invariant (roads, buildings, parked cars …), and
//! * a **dynamic field** `(σ_d, c_d, v, f)` that is conditioned on a time
//!   embedding and additionally predicts a 3-D **scene-flow** vector `v` and a
//!   self-supervised **feature** `f` (moving cars, pedestrians …).
//!
//! The two fields are composited by summing their densities and blending colour
//! by density weight,
//!
//! ```text
//! σ      = σ_s + σ_d
//! c      = (σ_s c_s + σ_d c_d) / (σ_s + σ_d),
//! ```
//!
//! so the rendered radiance is exactly the static plus dynamic contribution —
//! the decomposition "adds up". The scene flow `v(x, t)` warps dynamic features
//! to neighbouring timesteps, `x_{t±Δ} = x ± Δ · v`, which is the basis of
//! EmerNeRF's temporal-consistency self-supervision: the dynamic feature at `t`
//! should be predictable from `t ± 1` by following the flow. The warp is an
//! invertible point correspondence, so warping a point forward by a flow vector
//! and then back recovers the original point — and hence the original feature.
//!
//! The static field reuses the crate's [`NerfMlp`]; the Fourier position / time
//! encodings, [`Ray`] and the volume-rendering integral are reused as well.

use crate::encoding::positional::{PosEncConfig, positional_encode};
use crate::error::{NerfError, NerfResult};
use crate::handle::LcgRng;
use crate::network::nerf_mlp::{NerfMlp, NerfMlpConfig};
use crate::rendering::ray::Ray;
use crate::rendering::volume_render::{RenderResult, volume_render};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for an [`EmerNerf`] model.
#[derive(Debug, Clone)]
pub struct EmerNerfConfig {
    /// Fourier frequency levels for the 3-D position encoding.
    pub pos_freq: usize,
    /// Fourier frequency levels for the scalar time encoding.
    pub time_freq: usize,
    /// Hidden width of both the static and dynamic MLPs.
    pub hidden_dim: usize,
    /// Number of ReLU hidden layers in the dynamic MLP backbone.
    pub n_layers: usize,
    /// Width of the self-supervised dynamic feature vector.
    pub feature_dim: usize,
}

impl Default for EmerNerfConfig {
    fn default() -> Self {
        Self {
            pos_freq: 4,
            time_freq: 4,
            hidden_dim: 32,
            n_layers: 2,
            feature_dim: 8,
        }
    }
}

// ─── Dynamic output ──────────────────────────────────────────────────────────

/// All quantities emitted by the dynamic field at `(x, t)`.
#[derive(Debug, Clone)]
pub struct DynamicOutput {
    /// Dynamic volume density `σ_d ≥ 0`.
    pub density: f32,
    /// Dynamic RGB colour in `[0, 1]³`.
    pub color: [f32; 3],
    /// 3-D scene-flow velocity `v` (displacement per unit time).
    pub flow: [f32; 3],
    /// Self-supervised feature vector (bounded by `tanh`).
    pub feature: Vec<f32>,
}

/// Composited static + dynamic quantities at `(x, t)`.
#[derive(Debug, Clone)]
pub struct Composite {
    /// Static density `σ_s`.
    pub static_density: f32,
    /// Static colour `c_s`.
    pub static_color: [f32; 3],
    /// Dynamic density `σ_d`.
    pub dynamic_density: f32,
    /// Dynamic colour `c_d`.
    pub dynamic_color: [f32; 3],
    /// Combined density `σ_s + σ_d`.
    pub total_density: f32,
    /// Density-weighted blended colour.
    pub total_color: [f32; 3],
}

// ─── Dynamic field (multi-head) ──────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DynamicField {
    pos_cfg: PosEncConfig,
    time_cfg: PosEncConfig,
    backbone: Vec<(Vec<f32>, Vec<f32>)>,
    density: (Vec<f32>, Vec<f32>),
    color: (Vec<f32>, Vec<f32>),
    flow: (Vec<f32>, Vec<f32>),
    feature: (Vec<f32>, Vec<f32>),
    hidden: usize,
    feature_dim: usize,
}

impl DynamicField {
    fn new(cfg: &EmerNerfConfig, rng: &mut LcgRng) -> Self {
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
        let h = cfg.hidden_dim;

        let mut backbone = Vec::with_capacity(cfg.n_layers.max(1));
        let mut prev = in_dim;
        for _ in 0..cfg.n_layers.max(1) {
            backbone.push(make_layer(prev, h, rng));
            prev = h;
        }

        Self {
            pos_cfg,
            time_cfg,
            backbone,
            density: make_layer(h, 1, rng),
            color: make_layer(h, 3, rng),
            flow: make_layer(h, 3, rng),
            feature: make_layer(h, cfg.feature_dim, rng),
            hidden: h,
            feature_dim: cfg.feature_dim,
        }
    }

    fn forward(&self, position: [f32; 3], t: f32) -> NerfResult<DynamicOutput> {
        let pe_pos = positional_encode(&position, &self.pos_cfg)?;
        let pe_t = positional_encode(&[t], &self.time_cfg)?;
        let mut act = Vec::with_capacity(pe_pos.len() + pe_t.len());
        act.extend_from_slice(&pe_pos);
        act.extend_from_slice(&pe_t);
        for (w, b) in &self.backbone {
            act = fc_relu(&act, w, b, self.hidden);
        }

        let density = softplus(fc_linear(&act, &self.density.0, &self.density.1, 1)[0]);
        let color = sigmoid3(fc_linear(&act, &self.color.0, &self.color.1, 3));
        let flow_raw = fc_linear(&act, &self.flow.0, &self.flow.1, 3);
        let flow = [flow_raw[0], flow_raw[1], flow_raw[2]];
        let feature = fc_linear(&act, &self.feature.0, &self.feature.1, self.feature_dim)
            .into_iter()
            .map(f32::tanh)
            .collect();

        Ok(DynamicOutput {
            density,
            color,
            flow,
            feature,
        })
    }
}

// ─── Scene-flow warp ─────────────────────────────────────────────────────────

/// Advect a point along a scene-flow vector by a (signed) time delta:
/// `x' = x + dt · flow`.
///
/// The map is exactly invertible for a fixed flow vector: warping forward by
/// `dt` then backward by `dt` returns the original point.
#[must_use]
pub fn warp(point: [f32; 3], flow: [f32; 3], dt: f32) -> [f32; 3] {
    [
        point[0] + dt * flow[0],
        point[1] + dt * flow[1],
        point[2] + dt * flow[2],
    ]
}

// ─── EmerNerf ────────────────────────────────────────────────────────────────

/// EmerNeRF model: a time-invariant static field and a time-conditioned dynamic
/// field (density + colour + scene-flow + feature), composited together.
#[derive(Debug, Clone)]
pub struct EmerNerf {
    static_field: NerfMlp,
    static_pe: PosEncConfig,
    dynamic: DynamicField,
    feature_dim: usize,
}

impl EmerNerf {
    /// Build a randomly-initialised EmerNeRF model.
    ///
    /// # Errors
    ///
    /// Returns [`NerfError::InvalidFreqLevels`] if either frequency level is
    /// zero, [`NerfError::InvalidFeatureDim`] for a zero width / feature dim,
    /// and propagates [`NerfMlp::new`] errors.
    pub fn new(cfg: EmerNerfConfig, rng: &mut LcgRng) -> NerfResult<Self> {
        if cfg.pos_freq == 0 || cfg.time_freq == 0 {
            return Err(NerfError::InvalidFreqLevels {
                levels: cfg.pos_freq.min(cfg.time_freq),
            });
        }
        if cfg.hidden_dim == 0 || cfg.feature_dim == 0 {
            return Err(NerfError::InvalidFeatureDim { dim: 0 });
        }

        // Static field: NerfMlp over PE(position); the 3-D position is fed as the
        // "direction" features so the field stays a pure function of position
        // (hence time-invariant).
        let static_pe = PosEncConfig {
            n_freq: cfg.pos_freq,
            include_input: true,
            input_dim: 3,
        };
        let static_cfg = NerfMlpConfig {
            xyz_enc_dim: static_pe.output_dim(),
            dir_enc_dim: 3,
            hidden_dim: cfg.hidden_dim,
        };
        let static_field = NerfMlp::new(static_cfg, rng)?;
        let dynamic = DynamicField::new(&cfg, rng);

        Ok(Self {
            static_field,
            static_pe,
            dynamic,
            feature_dim: cfg.feature_dim,
        })
    }

    /// Dynamic feature dimensionality.
    #[must_use]
    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    /// Query the time-invariant static field at `position`: `(σ_s, c_s)`.
    ///
    /// # Errors
    ///
    /// Propagates positional-encoding / [`NerfMlp`] failures.
    pub fn query_static(&self, position: [f32; 3]) -> NerfResult<(f32, [f32; 3])> {
        let pe = positional_encode(&position, &self.static_pe)?;
        self.static_field.forward(&pe, &position)
    }

    /// Query the dynamic field at `(position, t)`.
    ///
    /// # Errors
    ///
    /// Propagates positional-encoding failures.
    pub fn query_dynamic(&self, position: [f32; 3], t: f32) -> NerfResult<DynamicOutput> {
        self.dynamic.forward(position, t)
    }

    /// Scene-flow velocity `v(x, t)` predicted by the dynamic field.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::query_dynamic`].
    pub fn flow(&self, position: [f32; 3], t: f32) -> NerfResult<[f32; 3]> {
        Ok(self.query_dynamic(position, t)?.flow)
    }

    /// Self-supervised dynamic feature `f(x, t)`.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::query_dynamic`].
    pub fn dynamic_feature(&self, position: [f32; 3], t: f32) -> NerfResult<Vec<f32>> {
        Ok(self.query_dynamic(position, t)?.feature)
    }

    /// Composite the static and dynamic fields at `(position, t)`.
    ///
    /// `total_density = σ_s + σ_d` and `total_color` is the density-weighted
    /// blend of the two colours.
    ///
    /// # Errors
    ///
    /// Propagates static / dynamic evaluation failures.
    pub fn composite(&self, position: [f32; 3], t: f32) -> NerfResult<Composite> {
        let (sigma_s, color_s) = self.query_static(position)?;
        let dyn_out = self.query_dynamic(position, t)?;
        let sigma_d = dyn_out.density;
        let color_d = dyn_out.color;

        let total = sigma_s + sigma_d;
        let total_color = if total > 1e-8 {
            let inv = 1.0 / total;
            [
                (sigma_s * color_s[0] + sigma_d * color_d[0]) * inv,
                (sigma_s * color_s[1] + sigma_d * color_d[1]) * inv,
                (sigma_s * color_s[2] + sigma_d * color_d[2]) * inv,
            ]
        } else {
            [
                0.5 * (color_s[0] + color_d[0]),
                0.5 * (color_s[1] + color_d[1]),
                0.5 * (color_s[2] + color_d[2]),
            ]
        };

        Ok(Composite {
            static_density: sigma_s,
            static_color: color_s,
            dynamic_density: sigma_d,
            dynamic_color: color_d,
            total_density: total,
            total_color,
        })
    }

    /// Temporal-consistency probe: the dynamic feature at `(x, t)` and the
    /// dynamic feature at the flow-warped point `(x + Δ·v, t + Δ)`.
    ///
    /// EmerNeRF's self-supervision drives these two features together; the
    /// returned pair is the residual's two operands.
    ///
    /// # Errors
    ///
    /// Propagates dynamic-field evaluation failures.
    pub fn temporal_consistency(
        &self,
        position: [f32; 3],
        t: f32,
        dt: f32,
    ) -> NerfResult<(Vec<f32>, Vec<f32>)> {
        let here = self.query_dynamic(position, t)?;
        let warped_pos = warp(position, here.flow, dt);
        let there = self.query_dynamic(warped_pos, t + dt)?;
        Ok((here.feature, there.feature))
    }

    /// Volume-render a ray at time `t` using the composited density / colour.
    ///
    /// Reuses the shared [`volume_render`] integral.
    ///
    /// # Errors
    ///
    /// Returns [`NerfError::InvalidSampleCount`] for no samples and propagates
    /// composite / volume-render failures.
    pub fn render(&self, ray: &Ray, t_vals: &[f32], t: f32) -> NerfResult<RenderResult> {
        if t_vals.is_empty() {
            return Err(NerfError::InvalidSampleCount { n: 0 });
        }
        let mut sigma = Vec::with_capacity(t_vals.len());
        let mut color = Vec::with_capacity(t_vals.len() * 3);
        for &s in t_vals {
            let comp = self.composite(ray.at(s), t)?;
            sigma.push(comp.total_density);
            color.extend_from_slice(&comp.total_color);
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
fn sigmoid3(v: Vec<f32>) -> [f32; 3] {
    [sigmoid(v[0]), sigmoid(v[1]), sigmoid(v[2])]
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

    fn tiny() -> EmerNerf {
        let cfg = EmerNerfConfig {
            pos_freq: 4,
            time_freq: 4,
            hidden_dim: 16,
            n_layers: 2,
            feature_dim: 8,
        };
        let mut rng = LcgRng::new(17);
        EmerNerf::new(cfg, &mut rng).expect("new should succeed")
    }

    #[test]
    fn decomposition_adds_up() {
        let model = tiny();
        let comp = model
            .composite([0.2, -0.3, 0.4], 0.5)
            .expect("composite should succeed");
        // Total density is exactly the sum of static and dynamic densities.
        assert!(
            (comp.total_density - (comp.static_density + comp.dynamic_density)).abs() < 1e-6,
            "σ_total must equal σ_s + σ_d"
        );
        // Total colour is the density-weighted blend.
        let total = comp.static_density + comp.dynamic_density;
        if total > 1e-8 {
            let inv = 1.0 / total;
            for k in 0..3 {
                let expect = (comp.static_density * comp.static_color[k]
                    + comp.dynamic_density * comp.dynamic_color[k])
                    * inv;
                assert!(
                    (comp.total_color[k] - expect).abs() < 1e-5,
                    "colour must be the density-weighted composite"
                );
            }
        }
    }

    #[test]
    fn static_field_is_time_invariant() {
        let model = tiny();
        let p = [0.1, 0.2, 0.3];
        let a = model.composite(p, 0.0).expect("composite should succeed");
        let b = model.composite(p, 0.9).expect("composite should succeed");
        // The static contribution is identical across time …
        assert_eq!(a.static_density, b.static_density);
        assert_eq!(a.static_color, b.static_color);
        // … while the dynamic contribution differs across time.
        let dyn_changed = (a.dynamic_density - b.dynamic_density).abs() > 1e-6
            || (0..3).any(|k| (a.dynamic_color[k] - b.dynamic_color[k]).abs() > 1e-6);
        assert!(dyn_changed, "dynamic field must vary with time");
    }

    #[test]
    fn dynamic_field_varies_with_time() {
        let model = tiny();
        let p = [0.05, -0.15, 0.25];
        let d0 = model
            .query_dynamic(p, 0.1)
            .expect("query_dynamic should succeed");
        let d1 = model
            .query_dynamic(p, 0.8)
            .expect("query_dynamic should succeed");
        let changed = (d0.density - d1.density).abs() > 1e-6
            || (0..3).any(|k| (d0.color[k] - d1.color[k]).abs() > 1e-6)
            || d0
                .feature
                .iter()
                .zip(&d1.feature)
                .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(changed, "dynamic outputs must change with t");
    }

    #[test]
    fn flow_warp_roundtrip_recovers_point_and_feature() {
        let model = tiny();
        let p = [0.12, -0.21, 0.33];
        let t = 0.4;
        let dt = 0.05;
        let v = model.flow(p, t).expect("flow should succeed");
        // Warp forward by the flow, then backward by the same flow vector.
        let forward = warp(p, v, dt);
        let back = warp(forward, v, -dt);
        for k in 0..3 {
            assert!(
                (back[k] - p[k]).abs() < 1e-5,
                "round-trip warp must recover the original point"
            );
        }
        // The predicted feature at the round-tripped point matches the original.
        let f0 = model
            .dynamic_feature(p, t)
            .expect("dynamic_feature should succeed");
        let f1 = model
            .dynamic_feature(back, t)
            .expect("dynamic_feature should succeed");
        for (a, b) in f0.iter().zip(&f1) {
            assert!(
                (a - b).abs() < 1e-4,
                "feature at the round-tripped point must match"
            );
        }
    }

    #[test]
    fn temporal_consistency_shapes_and_finite() {
        let model = tiny();
        let (here, there) = model
            .temporal_consistency([0.2, 0.1, -0.1], 0.3, 0.05)
            .expect("value should be present");
        assert_eq!(here.len(), model.feature_dim());
        assert_eq!(there.len(), model.feature_dim());
        assert!(here.iter().all(|v| v.is_finite()));
        assert!(there.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn composite_is_finite_and_in_range() {
        let model = tiny();
        let comp = model
            .composite([0.4, 0.4, 0.4], 0.6)
            .expect("composite should succeed");
        assert!(comp.total_density.is_finite() && comp.total_density >= 0.0);
        for c in comp.total_color {
            assert!(c.is_finite() && (0.0..=1.0).contains(&c));
        }
    }

    #[test]
    fn composite_is_deterministic() {
        let a = tiny();
        let b = tiny();
        let ca = a
            .composite([0.3, -0.2, 0.1], 0.25)
            .expect("composite should succeed");
        let cb = b
            .composite([0.3, -0.2, 0.1], 0.25)
            .expect("composite should succeed");
        assert_eq!(ca.total_density, cb.total_density);
        assert_eq!(ca.total_color, cb.total_color);
    }

    #[test]
    fn render_produces_valid_result() {
        let model = tiny();
        let ray =
            Ray::normalized([0.0, 0.0, -2.0], [0.0, 0.0, 1.0]).expect("normalized should succeed");
        let t_vals: Vec<f32> = (0..14).map(|i| 1.0 + i as f32 * 0.1).collect();
        let res = model
            .render(&ray, &t_vals, 0.5)
            .expect("render should succeed");
        assert!((0.0..=1.000_01).contains(&res.opacity));
        assert!(res.depth.is_finite());
        for c in res.rgb {
            assert!(c.is_finite() && (0.0..=1.0).contains(&c));
        }
    }

    #[test]
    fn rejects_bad_config() {
        let mut rng = LcgRng::new(1);
        let bad_freq = EmerNerfConfig {
            time_freq: 0,
            ..EmerNerfConfig::default()
        };
        assert!(EmerNerf::new(bad_freq, &mut rng).is_err());
        let bad_dim = EmerNerfConfig {
            feature_dim: 0,
            ..EmerNerfConfig::default()
        };
        assert!(EmerNerf::new(bad_dim, &mut rng).is_err());
    }
}
