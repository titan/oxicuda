//! GRAND — GRAph Neural Diffusion (Chamberlain et al., ICML 2021).
//!
//! GRAND reinterprets message passing as the explicit discretisation of a
//! diffusion partial differential equation on the graph:
//!
//! ```text
//!   ∂x(t)/∂t = (A_att − I) · x(t)
//! ```
//!
//! where `A_att` is a **row-stochastic**, attention-weighted adjacency operator
//! restricted to the graph structure (edges together with self-loops). The
//! **GRAND-l** (linear) variant freezes `A_att` to a constant operator computed
//! once from the input features and integrates the linear ODE with a few
//! explicit (forward) Euler steps of size `Δ`:
//!
//! ```text
//!   x_{k+1} = x_k + Δ · (A_att − I) · x_k
//!           = (1 − Δ) · x_k + Δ · A_att · x_k
//! ```
//!
//! For `Δ ∈ (0, 1]` every Euler step is a convex combination of the current
//! node states — the iteration matrix `M = (1 − Δ) I + Δ A_att` is itself
//! row-stochastic — so the dynamics are mass-bounded and smoothing, exactly the
//! behaviour of heat diffusion on a graph. Larger `n_steps` integrates the
//! diffusion further in (virtual) time.
//!
//! The attention coefficients are the parameter-free, numerically-stable scaled
//! dot-product attention used by GRAND:
//!
//! ```text
//!   e_ij = (x_i · x_j) / √d ,      a_ij = softmax_{j ∈ N(i) ∪ {i}}(e_ij)
//! ```
//!
//! Because the operator is built from the *initial* encoded features it is held
//! constant across the integration (the linear `GRAND-l` regime); the layer
//! therefore expects features already projected into the diffusion latent space
//! of dimension `hidden_dim`.

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for a [`GrandLayer`].
#[derive(Debug, Clone, Copy)]
pub struct GrandConfig {
    /// Number of explicit (forward) Euler integration steps.
    ///
    /// `0` is valid and returns the input unchanged (zero integration time).
    pub n_steps: usize,
    /// Integration step size `Δ`.
    ///
    /// For `Δ ∈ (0, 1]` each step is a convex combination of node states, so the
    /// diffusion is mass-bounded and smoothing. Must be finite and `> 0`.
    pub step_size: f32,
    /// Latent diffusion dimension.
    ///
    /// The diffusion runs in a hidden space of this size; the features supplied
    /// to [`GrandLayer::forward`] must already have this dimension.
    pub hidden_dim: usize,
}

// ─── Layer ────────────────────────────────────────────────────────────────────

/// GRAND-l linear graph neural diffusion layer.
pub struct GrandLayer {
    config: GrandConfig,
}

impl GrandLayer {
    /// Construct a GRAND layer from configuration.
    ///
    /// # Errors
    ///
    /// Returns [`GnnError::InvalidLayerConfig`] if `hidden_dim == 0` or if
    /// `step_size` is non-finite or `<= 0`.
    pub fn new(config: GrandConfig) -> GnnResult<Self> {
        if config.hidden_dim == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "GRAND: hidden_dim must be > 0".to_string(),
            ));
        }
        if !config.step_size.is_finite() || config.step_size <= 0.0 {
            return Err(GnnError::InvalidLayerConfig(format!(
                "GRAND: step_size must be finite and > 0, got {}",
                config.step_size
            )));
        }
        Ok(Self { config })
    }

    /// Number of Euler steps integrated by this layer.
    #[inline]
    pub fn n_steps(&self) -> usize {
        self.config.n_steps
    }

    /// Latent diffusion dimension expected by [`forward`](Self::forward).
    #[inline]
    pub fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }

    /// Build the constant, row-stochastic attention operator from the initial
    /// features.
    ///
    /// Returns, for every node `i`, the list of attended nodes
    /// `N(i) ∪ {i}` together with their softmax attention weights
    /// (which sum to one).
    fn attention_rows(
        &self,
        x: &[f32],
        n_nodes: usize,
        dim: usize,
        adjacency: &CsrGraph,
    ) -> GnnResult<Vec<(Vec<usize>, Vec<f32>)>> {
        let inv_sqrt_d = 1.0_f32 / (dim as f32).sqrt();
        let mut rows: Vec<(Vec<usize>, Vec<f32>)> = Vec::with_capacity(n_nodes);

        for i in 0..n_nodes {
            // Unique attended set: graph neighbours of `i` plus the self-loop.
            let raw = adjacency.neighbors(i)?;
            let mut nbrs: Vec<usize> = Vec::with_capacity(raw.len() + 1);
            for &j in raw {
                if !nbrs.contains(&j) {
                    nbrs.push(j);
                }
            }
            if !nbrs.contains(&i) {
                nbrs.push(i);
            }

            // Scaled dot-product scores e_ij = (x_i · x_j) / √d.
            let mut scores: Vec<f32> = Vec::with_capacity(nbrs.len());
            for &j in &nbrs {
                let mut dot = 0.0_f32;
                for c in 0..dim {
                    dot += x[i * dim + c] * x[j * dim + c];
                }
                scores.push(dot * inv_sqrt_d);
            }

            // Numerically-stable softmax over the attended set.
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut weights: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum: f32 = weights.iter().sum();
            if sum > 0.0 {
                for w in weights.iter_mut() {
                    *w /= sum;
                }
            } else {
                let uniform = 1.0_f32 / nbrs.len() as f32;
                for w in weights.iter_mut() {
                    *w = uniform;
                }
            }

            rows.push((nbrs, weights));
        }

        Ok(rows)
    }

    /// Integrate the diffusion `x_{k+1} = (1 − Δ) x_k + Δ A_att x_k`.
    ///
    /// # Arguments
    ///
    /// * `x` — `[n_nodes × dim]` latent node features (row-major).
    /// * `n_nodes` — number of nodes.
    /// * `dim` — feature dimension (must equal `hidden_dim`).
    /// * `adjacency` — CSR graph providing the diffusion topology; self-loops are
    ///   added internally so every node attends to itself.
    ///
    /// # Returns
    ///
    /// `[n_nodes × dim]` diffused node features.
    ///
    /// # Errors
    ///
    /// * [`GnnError::DimensionMismatch`] if `dim != hidden_dim` or if
    ///   `adjacency.n_nodes() != n_nodes`.
    /// * [`GnnError::NodeFeatureMismatch`] if `x.len() != n_nodes * dim`.
    /// * [`GnnError::NonFiniteOutput`] if any output value is NaN or infinite.
    pub fn forward(
        &self,
        x: &[f32],
        n_nodes: usize,
        dim: usize,
        adjacency: &CsrGraph,
    ) -> GnnResult<Vec<f32>> {
        if dim != self.config.hidden_dim {
            return Err(GnnError::DimensionMismatch {
                expected: self.config.hidden_dim,
                got: dim,
            });
        }
        if adjacency.n_nodes() != n_nodes {
            return Err(GnnError::DimensionMismatch {
                expected: adjacency.n_nodes(),
                got: n_nodes,
            });
        }
        if x.len() != n_nodes * dim {
            return Err(GnnError::NodeFeatureMismatch(n_nodes, x.len() / dim.max(1)));
        }

        let step = self.config.step_size;
        let one_minus_step = 1.0_f32 - step;

        // Constant GRAND-l operator from the initial features.
        let rows = self.attention_rows(x, n_nodes, dim, adjacency)?;

        let mut x_cur = x.to_vec();
        for _ in 0..self.config.n_steps {
            let mut x_next = vec![0.0_f32; n_nodes * dim];
            for (i, (nbrs, weights)) in rows.iter().enumerate() {
                // Attention aggregate A_att·x for node i.
                for (idx, &j) in nbrs.iter().enumerate() {
                    let a = weights[idx];
                    for c in 0..dim {
                        x_next[i * dim + c] += a * x_cur[j * dim + c];
                    }
                }
                // Convex Euler blend: (1 − Δ) x_i + Δ (A_att x)_i.
                for c in 0..dim {
                    let agg = x_next[i * dim + c];
                    x_next[i * dim + c] = one_minus_step * x_cur[i * dim + c] + step * agg;
                }
            }
            x_cur = x_next;
        }

        if x_cur.iter().any(|v| !v.is_finite()) {
            return Err(GnnError::NonFiniteOutput("GrandLayer::forward"));
        }
        Ok(x_cur)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn complete_graph(n: usize) -> CsrGraph {
        let mut edges = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    edges.push((i, j));
                }
            }
        }
        CsrGraph::from_edges(n, &edges).expect("test invariant: value must be valid")
    }

    fn ring_graph(n: usize) -> CsrGraph {
        let edges: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| [(i, (i + 1) % n), ((i + 1) % n, i)])
            .collect();
        CsrGraph::from_edges(n, &edges).expect("test invariant: value must be valid")
    }

    fn variance(col: &[f32]) -> f32 {
        let n = col.len() as f32;
        let mean = col.iter().sum::<f32>() / n;
        col.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n
    }

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn new_valid() {
        let cfg = GrandConfig {
            n_steps: 4,
            step_size: 0.25,
            hidden_dim: 3,
        };
        let layer = GrandLayer::new(cfg).expect("test invariant: value must be valid");
        assert_eq!(layer.n_steps(), 4);
        assert_eq!(layer.hidden_dim(), 3);
    }

    #[test]
    fn new_invalid_hidden_dim_zero() {
        let cfg = GrandConfig {
            n_steps: 4,
            step_size: 0.25,
            hidden_dim: 0,
        };
        assert!(GrandLayer::new(cfg).is_err());
    }

    #[test]
    fn new_invalid_step_size_zero() {
        let cfg = GrandConfig {
            n_steps: 4,
            step_size: 0.0,
            hidden_dim: 3,
        };
        assert!(GrandLayer::new(cfg).is_err());
    }

    #[test]
    fn new_invalid_step_size_negative() {
        let cfg = GrandConfig {
            n_steps: 4,
            step_size: -0.1,
            hidden_dim: 3,
        };
        assert!(GrandLayer::new(cfg).is_err());
    }

    // ── (a) Shapes correct & finite ──────────────────────────────────────────

    #[test]
    fn forward_shape_and_finite() {
        let g = ring_graph(5);
        let dim = 3;
        let cfg = GrandConfig {
            n_steps: 5,
            step_size: 0.3,
            hidden_dim: dim,
        };
        let layer = GrandLayer::new(cfg).expect("test invariant: value must be valid");
        let x: Vec<f32> = (0..5 * dim).map(|i| (i as f32) * 0.05).collect();
        let out = layer
            .forward(&x, 5, dim, &g)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 5 * dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── (b) Diffusion smooths: variance decreases on a connected graph ───────

    #[test]
    fn forward_diffusion_smooths_variance() {
        // Complete graph (3-regular + self) with small distinct positive
        // features → near-uniform attention → strong smoothing.
        let n = 4;
        let g = complete_graph(n);
        let dim = 1;
        let cfg = GrandConfig {
            n_steps: 10,
            step_size: 0.5,
            hidden_dim: dim,
        };
        let layer = GrandLayer::new(cfg).expect("test invariant: value must be valid");
        let x = vec![0.1_f32, 0.2, 0.3, 0.4];
        let var_before = variance(&x);
        let out = layer
            .forward(&x, n, dim, &g)
            .expect("test invariant: value must be valid");
        let var_after = variance(&out);
        assert!(
            var_after < var_before,
            "variance should decrease: before={var_before} after={var_after}"
        );
    }

    // ── (c) Mass bounded by convexity of the row-stochastic operator ─────────

    #[test]
    fn forward_mass_is_bounded() {
        let n = 4;
        let g = ring_graph(n);
        let dim = 2;
        let cfg = GrandConfig {
            n_steps: 6,
            step_size: 0.4,
            hidden_dim: dim,
        };
        let layer = GrandLayer::new(cfg).expect("test invariant: value must be valid");
        let x = vec![1.0_f32, 5.0, 2.0, 4.0, 3.0, 1.0, 0.5, 2.0];
        let out = layer
            .forward(&x, n, dim, &g)
            .expect("test invariant: value must be valid");
        // For each feature column the post-diffusion mass must stay within
        // [n·min, n·max] of the original column (convex-combination bound).
        for c in 0..dim {
            let col: Vec<f32> = (0..n).map(|i| x[i * dim + c]).collect();
            let min = col.iter().copied().fold(f32::INFINITY, f32::min);
            let max = col.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mass_after: f32 = (0..n).map(|i| out[i * dim + c]).sum();
            assert!(
                mass_after >= n as f32 * min - 1e-4 && mass_after <= n as f32 * max + 1e-4,
                "mass {mass_after} out of bounds [{}, {}]",
                n as f32 * min,
                n as f32 * max
            );
        }
    }

    #[test]
    fn forward_mass_approx_conserved_on_regular_graph() {
        // On a regular graph with near-uniform attention the operator is close
        // to doubly stochastic, so total mass is approximately conserved.
        let n = 4;
        let g = ring_graph(n);
        let dim = 1;
        let cfg = GrandConfig {
            n_steps: 5,
            step_size: 0.5,
            hidden_dim: dim,
        };
        let layer = GrandLayer::new(cfg).expect("test invariant: value must be valid");
        let x = vec![0.05_f32, 0.1, 0.15, 0.2];
        let mass_before: f32 = x.iter().sum();
        let out = layer
            .forward(&x, n, dim, &g)
            .expect("test invariant: value must be valid");
        let mass_after: f32 = out.iter().sum();
        assert!(
            (mass_after - mass_before).abs() < 0.02,
            "mass before={mass_before} after={mass_after}"
        );
    }

    // ── (d) Zero steps → input unchanged ─────────────────────────────────────

    #[test]
    fn forward_zero_steps_identity() {
        let n = 4;
        let g = complete_graph(n);
        let dim = 2;
        let cfg = GrandConfig {
            n_steps: 0,
            step_size: 0.5,
            hidden_dim: dim,
        };
        let layer = GrandLayer::new(cfg).expect("test invariant: value must be valid");
        let x: Vec<f32> = (0..n * dim).map(|i| (i as f32) * 0.3 - 1.0).collect();
        let out = layer
            .forward(&x, n, dim, &g)
            .expect("test invariant: value must be valid");
        for (o, xi) in out.iter().zip(x.iter()) {
            assert!((o - xi).abs() < 1e-7, "o={o} xi={xi}");
        }
    }

    // ── (e) Isolated node keeps its own features ─────────────────────────────

    #[test]
    fn forward_isolated_node_preserved() {
        // Node 0 is isolated (no edges); nodes 1,2 form a connected pair.
        let g = CsrGraph::from_edges(3, &[(1, 2), (2, 1)])
            .expect("test invariant: value must be valid");
        let dim = 2;
        let cfg = GrandConfig {
            n_steps: 8,
            step_size: 0.5,
            hidden_dim: dim,
        };
        let layer = GrandLayer::new(cfg).expect("test invariant: value must be valid");
        let x = vec![7.0_f32, -3.0, 1.0, 1.0, 2.0, 2.0];
        let out = layer
            .forward(&x, 3, dim, &g)
            .expect("test invariant: value must be valid");
        // Isolated node only attends to itself → exact self-loop fixed point.
        assert!((out[0] - 7.0).abs() < 1e-5, "out[0]={}", out[0]);
        assert!((out[1] - (-3.0)).abs() < 1e-5, "out[1]={}", out[1]);
    }

    // ── (f) dim / adjacency mismatch → error ─────────────────────────────────

    #[test]
    fn forward_dim_mismatch_errors() {
        let g = ring_graph(4);
        let cfg = GrandConfig {
            n_steps: 2,
            step_size: 0.3,
            hidden_dim: 3,
        };
        let layer = GrandLayer::new(cfg).expect("test invariant: value must be valid");
        let x = vec![0.1_f32; 4 * 2]; // dim 2 != hidden_dim 3
        let err = layer.forward(&x, 4, 2, &g);
        assert!(matches!(err, Err(GnnError::DimensionMismatch { .. })));
    }

    #[test]
    fn forward_adjacency_mismatch_errors() {
        let g = ring_graph(4); // 4 nodes
        let cfg = GrandConfig {
            n_steps: 2,
            step_size: 0.3,
            hidden_dim: 2,
        };
        let layer = GrandLayer::new(cfg).expect("test invariant: value must be valid");
        let x = vec![0.1_f32; 5 * 2];
        let err = layer.forward(&x, 5, 2, &g); // n_nodes 5 != graph 4
        assert!(matches!(err, Err(GnnError::DimensionMismatch { .. })));
    }

    #[test]
    fn forward_feature_length_mismatch_errors() {
        let g = ring_graph(4);
        let cfg = GrandConfig {
            n_steps: 2,
            step_size: 0.3,
            hidden_dim: 2,
        };
        let layer = GrandLayer::new(cfg).expect("test invariant: value must be valid");
        let x = vec![0.1_f32; 3 * 2]; // only 3 nodes' worth
        let err = layer.forward(&x, 4, 2, &g);
        assert!(matches!(err, Err(GnnError::NodeFeatureMismatch(..))));
    }

    // ── Attention rows are row-stochastic ─────────────────────────────────────

    #[test]
    fn attention_rows_sum_to_one() {
        let g = complete_graph(4);
        let dim = 2;
        let cfg = GrandConfig {
            n_steps: 1,
            step_size: 0.5,
            hidden_dim: dim,
        };
        let layer = GrandLayer::new(cfg).expect("test invariant: value must be valid");
        let x: Vec<f32> = (0..4 * dim).map(|i| (i as f32) * 0.1).collect();
        let rows = layer
            .attention_rows(&x, 4, dim, &g)
            .expect("test invariant: value must be valid");
        for (_, w) in &rows {
            let s: f32 = w.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row sum={s}");
            assert!(w.iter().all(|&a| a >= 0.0));
        }
    }

    // ── More steps push further toward consensus ──────────────────────────────

    #[test]
    fn forward_more_steps_more_smoothing() {
        let n = 4;
        let g = complete_graph(n);
        let dim = 1;
        let make = |steps: usize| {
            let cfg = GrandConfig {
                n_steps: steps,
                step_size: 0.5,
                hidden_dim: dim,
            };
            let layer = GrandLayer::new(cfg).expect("test invariant: value must be valid");
            let x = vec![0.1_f32, 0.2, 0.3, 0.4];
            layer
                .forward(&x, n, dim, &g)
                .expect("test invariant: value must be valid")
        };
        let out2 = make(2);
        let out12 = make(12);
        assert!(
            variance(&out12) < variance(&out2),
            "more steps should reduce variance further"
        );
    }
}
