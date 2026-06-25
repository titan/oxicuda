//! E(n)-Equivariant Graph Neural Network (EGNN).
//!
//! Reference: Satorras, Hoogeboom, Welling, *"E(n) Equivariant Graph Neural
//! Networks"*, ICML 2021.
//!
//! Each node `i` carries an invariant feature vector `h_i ∈ R^d` and an
//! equivariant coordinate `x_i ∈ R^3`. Over the directed edge set
//! `(i, j)` the layer computes
//!
//! ```text
//! m_ij = phi_e( h_i, h_j, ||x_i - x_j||^2 )                       (edge message)
//! x_i' = x_i + (1 / N_i) * Σ_j (x_i - x_j) * phi_x(m_ij)          (coord update)
//! m_i  = Σ_j m_ij                                                 (aggregation)
//! h_i' = h_i + phi_h( h_i, m_i )                                  (node update)
//! ```
//!
//! where `N_i` is the in-degree of node `i` (the number of edges `(i, ·)`),
//! and `phi_e`, `phi_x`, `phi_h` are small two-layer MLPs with SiLU
//! activations.
//!
//! # E(3) equivariance / invariance
//!
//! Because the edge message depends on the coordinates only through the
//! *invariant* squared distance `||x_i - x_j||^2`, and the coordinate update
//! only ever scales the *relative* vector `x_i - x_j` by an invariant scalar,
//! the layer is exactly equivariant to the Euclidean group: for any rotation
//! `R` and translation `t`,
//!
//! ```text
//! forward(h, R·x + t)  ==  ( h',  R·x' + t )
//! ```
//!
//! i.e. node features are invariant and coordinates transform with the input.

use crate::error::{Geom3dError, Geom3dResult};
use crate::handle::LcgRng;

// ─── Activations & RNG helpers ────────────────────────────────────────────────

/// SiLU / swish activation: `silu(x) = x * sigmoid(x) = x / (1 + e^{-x})`.
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Draw a uniform `f32` in `[-1, 1)`.
///
/// `LcgRng::next_u32()` spans the full 32-bit range `[0, 2^32)`, so dividing by
/// `2^32` yields a true unit uniform in `[0, 1)` before mapping to the
/// symmetric interval.
#[inline]
fn uniform_pm1(rng: &mut LcgRng) -> f32 {
    let unit = rng.next_u32() as f32 / 4_294_967_296.0_f32; // [0, 1)
    unit * 2.0 - 1.0
}

// ─── Two-layer MLP ────────────────────────────────────────────────────────────

/// Two-layer perceptron `in -> hidden -> out` with a SiLU after the hidden
/// layer and an optional SiLU on the output.
struct Mlp {
    w1: Vec<f32>, // [hidden × in]
    b1: Vec<f32>, // [hidden]
    w2: Vec<f32>, // [out × hidden]
    b2: Vec<f32>, // [out]
    in_dim: usize,
    hidden_dim: usize,
    out_dim: usize,
    final_act: bool,
}

impl Mlp {
    fn new(
        in_dim: usize,
        hidden_dim: usize,
        out_dim: usize,
        final_act: bool,
        rng: &mut LcgRng,
    ) -> Self {
        let s1 = (6.0_f32 / (in_dim + hidden_dim).max(1) as f32).sqrt();
        let mut w1 = vec![0.0_f32; hidden_dim * in_dim];
        for v in &mut w1 {
            *v = uniform_pm1(rng) * s1;
        }
        let s2 = (6.0_f32 / (hidden_dim + out_dim).max(1) as f32).sqrt();
        let mut w2 = vec![0.0_f32; out_dim * hidden_dim];
        for v in &mut w2 {
            *v = uniform_pm1(rng) * s2;
        }
        Self {
            w1,
            b1: vec![0.0_f32; hidden_dim],
            w2,
            b2: vec![0.0_f32; out_dim],
            in_dim,
            hidden_dim,
            out_dim,
            final_act,
        }
    }

    /// Forward pass over a single input vector of length `in_dim`.
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut hidden = vec![0.0_f32; self.hidden_dim];
        for (o, h_o) in hidden.iter_mut().enumerate() {
            let row = &self.w1[o * self.in_dim..(o + 1) * self.in_dim];
            let acc = self.b1[o] + row.iter().zip(input).map(|(w, x)| w * x).sum::<f32>();
            *h_o = silu(acc);
        }

        let mut out = vec![0.0_f32; self.out_dim];
        for (o, out_o) in out.iter_mut().enumerate() {
            let row = &self.w2[o * self.hidden_dim..(o + 1) * self.hidden_dim];
            let acc = self.b2[o] + row.iter().zip(&hidden).map(|(w, x)| w * x).sum::<f32>();
            *out_o = if self.final_act { silu(acc) } else { acc };
        }
        out
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Hyper-parameters for an [`EgnnLayer`].
#[derive(Debug, Clone, Copy)]
pub struct EgnnConfig {
    /// Dimensionality `d` of the node feature vectors `h_i`.
    pub feature_dim: usize,
    /// Dimensionality `d_m` of the edge messages `m_ij`.
    pub message_dim: usize,
    /// Hidden width of the internal MLPs `phi_e`, `phi_x`, `phi_h`.
    pub hidden_dim: usize,
}

impl Default for EgnnConfig {
    fn default() -> Self {
        Self {
            feature_dim: 16,
            message_dim: 16,
            hidden_dim: 32,
        }
    }
}

// ─── EGNN layer ───────────────────────────────────────────────────────────────

/// A single E(n)-equivariant message-passing layer.
pub struct EgnnLayer {
    config: EgnnConfig,
    /// Edge MLP `phi_e: R^{2d+1} -> R^{d_m}` (SiLU on output).
    phi_e: Mlp,
    /// Coordinate MLP `phi_x: R^{d_m} -> R` (linear output scalar).
    phi_x: Mlp,
    /// Node MLP `phi_h: R^{d+d_m} -> R^d` (linear output, added as a residual).
    phi_h: Mlp,
}

impl EgnnLayer {
    /// Create a new layer with Xavier-uniform random weights and zero biases.
    #[must_use]
    pub fn new(config: EgnnConfig, rng: &mut LcgRng) -> Self {
        let d = config.feature_dim;
        let dm = config.message_dim;
        let hidden = config.hidden_dim;
        Self {
            config,
            phi_e: Mlp::new(2 * d + 1, hidden, dm, true, rng),
            phi_x: Mlp::new(dm, hidden, 1, false, rng),
            phi_h: Mlp::new(d + dm, hidden, d, false, rng),
        }
    }

    /// Return the layer configuration.
    #[must_use]
    pub fn config(&self) -> EgnnConfig {
        self.config
    }

    /// Forward pass.
    ///
    /// * `h` — node features, flattened `[n_nodes × feature_dim]`.
    /// * `x` — node coordinates, flattened `[n_nodes × 3]`.
    /// * `edges` — directed edges `(i, j)`; edge `(i, j)` sends a message from
    ///   `j` to `i` and contributes to the update of node `i`.
    /// * `n_nodes` — number of nodes `n`.
    ///
    /// Returns the updated features `h'` (`[n × feature_dim]`) and coordinates
    /// `x'` (`[n × 3]`). Nodes with no incoming edge keep their coordinate
    /// unchanged.
    pub fn forward(
        &self,
        h: &[f32],
        x: &[f32],
        edges: &[(usize, usize)],
        n_nodes: usize,
    ) -> Geom3dResult<(Vec<f32>, Vec<f32>)> {
        let d = self.config.feature_dim;
        let dm = self.config.message_dim;

        if h.len() != n_nodes * d {
            return Err(Geom3dError::DimensionMismatch {
                expected: n_nodes * d,
                got: h.len(),
            });
        }
        if x.len() != n_nodes * 3 {
            return Err(Geom3dError::DimensionMismatch {
                expected: n_nodes * 3,
                got: x.len(),
            });
        }
        for &(i, j) in edges {
            if i >= n_nodes || j >= n_nodes {
                return Err(Geom3dError::Internal(format!(
                    "edge ({i}, {j}) out of bounds for {n_nodes} nodes"
                )));
            }
        }

        let mut coord_delta = vec![0.0_f32; n_nodes * 3];
        let mut degree = vec![0.0_f32; n_nodes];
        let mut agg_msg = vec![0.0_f32; n_nodes * dm];
        let mut e_in = vec![0.0_f32; 2 * d + 1];

        for &(i, j) in edges {
            let dx = x[i * 3] - x[j * 3];
            let dy = x[i * 3 + 1] - x[j * 3 + 1];
            let dz = x[i * 3 + 2] - x[j * 3 + 2];
            let dist2 = dx * dx + dy * dy + dz * dz;

            e_in[..d].copy_from_slice(&h[i * d..i * d + d]);
            e_in[d..2 * d].copy_from_slice(&h[j * d..j * d + d]);
            e_in[2 * d] = dist2;

            let msg = self.phi_e.forward(&e_in);
            let scale = self.phi_x.forward(&msg).first().copied().unwrap_or(0.0);

            coord_delta[i * 3] += dx * scale;
            coord_delta[i * 3 + 1] += dy * scale;
            coord_delta[i * 3 + 2] += dz * scale;
            degree[i] += 1.0;

            let base = i * dm;
            for (acc, &m) in agg_msg[base..base + dm].iter_mut().zip(msg.iter()) {
                *acc += m;
            }
        }

        // Equivariant coordinate update (mean over neighbours).
        let mut x_out = x.to_vec();
        for (i, &n_i) in degree.iter().enumerate() {
            if n_i > 0.0 {
                let inv = 1.0 / n_i;
                x_out[i * 3] = x[i * 3] + coord_delta[i * 3] * inv;
                x_out[i * 3 + 1] = x[i * 3 + 1] + coord_delta[i * 3 + 1] * inv;
                x_out[i * 3 + 2] = x[i * 3 + 2] + coord_delta[i * 3 + 2] * inv;
            }
        }

        // Invariant node update (residual).
        let mut h_out = vec![0.0_f32; n_nodes * d];
        let mut h_in = vec![0.0_f32; d + dm];
        for i in 0..n_nodes {
            h_in[..d].copy_from_slice(&h[i * d..i * d + d]);
            h_in[d..d + dm].copy_from_slice(&agg_msg[i * dm..i * dm + dm]);
            let delta = self.phi_h.forward(&h_in);
            let in_row = &h[i * d..i * d + d];
            let out_row = &mut h_out[i * d..i * d + d];
            for ((out_v, &in_v), &dv) in out_row.iter_mut().zip(in_row.iter()).zip(delta.iter()) {
                *out_v = in_v + dv;
            }
        }

        Ok((h_out, x_out))
    }
}

// ─── Stacked EGNN ─────────────────────────────────────────────────────────────

/// A stack of [`EgnnLayer`]s applied sequentially.
///
/// Equivariance composes, so the whole stack is E(3)-equivariant in the
/// coordinates and invariant in the node features.
pub struct Egnn {
    layers: Vec<EgnnLayer>,
}

impl Egnn {
    /// Build a stack of `n_layers` EGNN layers, each independently initialised.
    #[must_use]
    pub fn new(n_layers: usize, config: EgnnConfig, rng: &mut LcgRng) -> Self {
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(EgnnLayer::new(config, rng));
        }
        Self { layers }
    }

    /// Number of stacked layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Run all layers in sequence, threading `(h, x)` through each.
    pub fn forward(
        &self,
        h: &[f32],
        x: &[f32],
        edges: &[(usize, usize)],
        n_nodes: usize,
    ) -> Geom3dResult<(Vec<f32>, Vec<f32>)> {
        let mut cur_h = h.to_vec();
        let mut cur_x = x.to_vec();
        for layer in &self.layers {
            let (next_h, next_x) = layer.forward(&cur_h, &cur_x, edges, n_nodes)?;
            cur_h = next_h;
            cur_x = next_x;
        }
        Ok((cur_h, cur_x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 90° counter-clockwise rotation about the z-axis: `(x, y, z) -> (-y, x, z)`.
    fn rot_z_90(v: [f32; 3]) -> [f32; 3] {
        [-v[1], v[0], v[2]]
    }

    fn make_features(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut buf = vec![0.0_f32; n * d];
        for v in &mut buf {
            *v = rng.next_u32() as f32 / 4_294_967_296.0_f32 - 0.5; // ≈ [-0.5, 0.5)
        }
        buf
    }

    fn make_coords(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut buf = vec![0.0_f32; n * 3];
        for v in &mut buf {
            *v = rng.next_u32() as f32 / 4_294_967_296.0_f32 * 2.0 - 1.0; // ≈ [-1, 1)
        }
        buf
    }

    fn sample_edges() -> Vec<(usize, usize)> {
        vec![(0, 1), (1, 0), (1, 2), (2, 3), (3, 4), (4, 0), (2, 4)]
    }

    #[test]
    fn egnn_forward_shapes_and_finite() {
        let cfg = EgnnConfig {
            feature_dim: 4,
            message_dim: 8,
            hidden_dim: 16,
        };
        let mut rng = LcgRng::new(1);
        let layer = EgnnLayer::new(cfg, &mut rng);
        let n = 5;
        let h = make_features(n, 4, 2);
        let x = make_coords(n, 3);
        let (h_out, x_out) = layer
            .forward(&h, &x, &sample_edges(), n)
            .expect("value should be present");
        assert_eq!(h_out.len(), n * 4);
        assert_eq!(x_out.len(), n * 3);
        assert!(h_out.iter().all(|v| v.is_finite()));
        assert!(x_out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn egnn_zero_edges_keeps_coords() {
        let cfg = EgnnConfig {
            feature_dim: 4,
            message_dim: 6,
            hidden_dim: 12,
        };
        let mut rng = LcgRng::new(11);
        let layer = EgnnLayer::new(cfg, &mut rng);
        let n = 4;
        let h = make_features(n, 4, 5);
        let x = make_coords(n, 6);
        let (_, x_out) = layer
            .forward(&h, &x, &[], n)
            .expect("forward should succeed");
        assert_eq!(x_out, x, "with no edges, coordinates must be unchanged");
    }

    #[test]
    fn egnn_e3_equivariance() {
        let cfg = EgnnConfig {
            feature_dim: 4,
            message_dim: 8,
            hidden_dim: 16,
        };
        let mut rng = LcgRng::new(7);
        let layer = EgnnLayer::new(cfg, &mut rng);
        let n = 5;
        let h = make_features(n, 4, 21);
        let x = make_coords(n, 22);
        let edges = sample_edges();

        let (h_out, x_out) = layer
            .forward(&h, &x, &edges, n)
            .expect("forward should succeed");

        // Apply a fixed rotation (90° about z) and translation to the inputs.
        let t = [0.3_f32, -0.2, 0.1];
        let mut x_t = vec![0.0_f32; n * 3];
        for i in 0..n {
            let r = rot_z_90([x[i * 3], x[i * 3 + 1], x[i * 3 + 2]]);
            x_t[i * 3] = r[0] + t[0];
            x_t[i * 3 + 1] = r[1] + t[1];
            x_t[i * 3 + 2] = r[2] + t[2];
        }
        let (h_out_t, x_out_t) = layer
            .forward(&h, &x_t, &edges, n)
            .expect("forward should succeed");

        // Features are invariant.
        for (a, b) in h_out.iter().zip(h_out_t.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "features must be E(3)-invariant: {a} vs {b}"
            );
        }
        // Coordinates are equivariant: x_out_t == R·x_out + t.
        for i in 0..n {
            let r = rot_z_90([x_out[i * 3], x_out[i * 3 + 1], x_out[i * 3 + 2]]);
            let expected = [r[0] + t[0], r[1] + t[1], r[2] + t[2]];
            let got = [x_out_t[i * 3], x_out_t[i * 3 + 1], x_out_t[i * 3 + 2]];
            for (g, e) in got.iter().zip(expected.iter()) {
                assert!(
                    (g - e).abs() < 1e-4,
                    "coordinate equivariance broken at node {i}: {g} vs {e}"
                );
            }
        }
    }

    #[test]
    fn egnn_stacked_equivariance() {
        let cfg = EgnnConfig {
            feature_dim: 3,
            message_dim: 6,
            hidden_dim: 12,
        };
        let mut rng = LcgRng::new(99);
        let net = Egnn::new(3, cfg, &mut rng);
        assert_eq!(net.num_layers(), 3);
        let n = 6;
        let h = make_features(n, 3, 31);
        let x = make_coords(n, 32);
        let edges = vec![(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3)];

        let (_, x_out) = net
            .forward(&h, &x, &edges, n)
            .expect("forward should succeed");

        let t = [0.25_f32, 0.4, -0.15];
        let mut x_t = vec![0.0_f32; n * 3];
        for i in 0..n {
            let r = rot_z_90([x[i * 3], x[i * 3 + 1], x[i * 3 + 2]]);
            x_t[i * 3] = r[0] + t[0];
            x_t[i * 3 + 1] = r[1] + t[1];
            x_t[i * 3 + 2] = r[2] + t[2];
        }
        let (_, x_out_t) = net
            .forward(&h, &x_t, &edges, n)
            .expect("forward should succeed");

        for i in 0..n {
            let r = rot_z_90([x_out[i * 3], x_out[i * 3 + 1], x_out[i * 3 + 2]]);
            let expected = [r[0] + t[0], r[1] + t[1], r[2] + t[2]];
            let got = [x_out_t[i * 3], x_out_t[i * 3 + 1], x_out_t[i * 3 + 2]];
            for (g, e) in got.iter().zip(expected.iter()) {
                assert!(
                    (g - e).abs() < 2e-4,
                    "stacked equivariance broken at node {i}: {g} vs {e}"
                );
            }
        }
    }

    #[test]
    fn egnn_deterministic_same_seed() {
        let cfg = EgnnConfig {
            feature_dim: 4,
            message_dim: 4,
            hidden_dim: 8,
        };
        let n = 5;
        let h = make_features(n, 4, 3);
        let x = make_coords(n, 4);
        let edges = sample_edges();

        let mut rng1 = LcgRng::new(123);
        let l1 = EgnnLayer::new(cfg, &mut rng1);
        let mut rng2 = LcgRng::new(123);
        let l2 = EgnnLayer::new(cfg, &mut rng2);

        let (h1, x1) = l1
            .forward(&h, &x, &edges, n)
            .expect("forward should succeed");
        let (h2, x2) = l2
            .forward(&h, &x, &edges, n)
            .expect("forward should succeed");
        assert_eq!(h1, h2);
        assert_eq!(x1, x2);
    }

    #[test]
    fn egnn_dim_mismatch_error() {
        let cfg = EgnnConfig {
            feature_dim: 4,
            message_dim: 4,
            hidden_dim: 8,
        };
        let mut rng = LcgRng::new(1);
        let layer = EgnnLayer::new(cfg, &mut rng);
        // h has wrong length for n=3.
        let h = vec![0.0_f32; 4 * 2];
        let x = vec![0.0_f32; 3 * 3];
        assert!(layer.forward(&h, &x, &[], 3).is_err());
    }

    #[test]
    fn egnn_edge_out_of_bounds_error() {
        let cfg = EgnnConfig {
            feature_dim: 2,
            message_dim: 2,
            hidden_dim: 4,
        };
        let mut rng = LcgRng::new(1);
        let layer = EgnnLayer::new(cfg, &mut rng);
        let n = 3;
        let h = make_features(n, 2, 1);
        let x = make_coords(n, 2);
        let edges = vec![(0, 5)]; // node 5 does not exist
        assert!(layer.forward(&h, &x, &edges, n).is_err());
    }
}
