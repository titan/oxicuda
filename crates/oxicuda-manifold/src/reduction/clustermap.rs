//! ClusterMap — a unified attraction/repulsion neighbour-embedding optimiser.
//!
//! ClusterMap implements the *unifying contrastive view* of neighbour embeddings
//! described by Damrich, Böhm, Hamprecht & Kobak (2022),
//! "From t-SNE to UMAP with contrastive learning" (a.k.a. the "contrastive
//! neighbour embedding" / CNE family). It frames t-SNE, UMAP and ForceAtlas2 as a
//! single **attraction/repulsion force family** acting on a kNN graph, where a
//! small set of exponents and a temperature interpolate between the methods.
//!
//! # The unified force model
//!
//! Two points `y_i`, `y_j` interact through a *similarity kernel* applied to their
//! squared low-dimensional distance `d² = ‖y_i − y_j‖²`. ClusterMap uses the
//! **generalised Cauchy / Student kernel**
//!
//! ```text
//! k(d²) = 1 / (1 + a · (d²)^b)
//! ```
//!
//! with `a > 0` (scale) and `b > 0` (tail exponent). For `a = b = 1` this is the
//! Student-t (Cauchy) kernel that t-SNE uses; larger `b` makes the tails heavier
//! (UMAP fits `a, b` from `min_dist`/`spread`); `b → 0` flattens toward a constant.
//!
//! Each edge `(i, j)` of the symmetrised kNN graph carries an **attractive** force
//! and every point is repelled from a set of **negatively sampled** non-neighbours.
//! The per-pair contrastive objective (a noise-contrastive / InfoNCE estimator) is
//!
//! ```text
//! L = − Σ_(i,j)∈E  w_ij · α · log  k(d²_ij)                (attraction)
//!     − Σ_(i,m)∈N  γ/T  · log (1 − k(d²_im))               (repulsion)
//! ```
//!
//! where `α` is the [`attraction_exp`](ClusterMapConfig::attraction_exp), `γ` the
//! [`repulsion_exp`](ClusterMapConfig::repulsion_exp), `w_ij` the graph edge weight
//! and `T` the **temperature**, annealed from
//! [`temperature_start`](ClusterMapConfig::temperature_start) down to
//! [`temperature_end`](ClusterMapConfig::temperature_end) across the epochs. A low
//! temperature sharpens repulsion (t-SNE-like, strong global spreading); a high
//! temperature softens it (UMAP/ForceAtlas2-like clustering).
//!
//! ## Exact gradients
//!
//! Let `p = k(d²) = 1/(1 + a·(d²)^b)` and note
//! `dk/d(d²) = −a·b·(d²)^{b−1} · p²`. With `d² = ‖y_i − y_j‖²` we have
//! `∂d²/∂y_i = 2(y_i − y_j)`. Therefore:
//!
//! - **Attraction** (`L_a = −w·α·log p`):
//!   ```text
//!   ∂L_a/∂y_i = w·α · (a·b·(d²)^{b−1}·p) · 2(y_i − y_j)
//!   ```
//!   i.e. a strictly attractive (norm-reducing) pull whose magnitude vanishes as
//!   `d² → 0` (for `b ≥ 1`) and as `p → 0`.
//!
//! - **Repulsion** (`L_r = −(γ/T)·log(1 − p)`):
//!   ```text
//!   ∂L_r/∂y_i = −(γ/T) · ( a·b·(d²)^{b−1}·p² / (1 − p) ) · 2(y_i − y_j)
//!   ```
//!   a strictly repulsive (norm-increasing) push that blows up smoothly as the two
//!   points coincide, preventing collapse. A small `ε` guards `1 − p` and `(d²)`.
//!
//! Each gradient is applied antisymmetrically to both endpoints, conserving the
//! embedding's centre of mass.
//!
//! # Recovering known methods (parameter map)
//!
//! | Target          | `a` | `b`  | `attraction_exp` | `repulsion_exp` | `temperature` |
//! |-----------------|-----|------|------------------|-----------------|---------------|
//! | **t-SNE**       | 1   | 1    | 1                | 1               | low (≈0.05)   |
//! | **UMAP**        | ≈1.6| ≈0.9 | 1                | 1               | high (≈1.0)   |
//! | **ForceAtlas2** | 1   | ≈0.5 | 1                | 1               | high (≈1.5)   |
//!
//! - *t-SNE-like*: the Cauchy kernel (`a=b=1`) with a **low** temperature makes the
//!   normalised repulsion dominate, reproducing the strong inter-cluster repulsion
//!   and well-separated maps characteristic of t-SNE.
//! - *UMAP-like*: a fitted heavier-tailed kernel (`b<1`, `a>1` from `min_dist`) with
//!   a **moderate/high** temperature down-weights repulsion, giving the tighter,
//!   more locally faithful clusters of UMAP with negative sampling.
//! - *ForceAtlas2-like*: a flat-tailed kernel (`b≈0.5`, so similarity decays slowly
//!   with distance) gives the long-range "gravity" pull and soft repulsion of the
//!   ForceAtlas2 graph layout; a high temperature keeps clusters compact. (As
//!   `b→0` the kernel becomes constant and *both* forces vanish, so the preset uses
//!   a small but strictly-positive `b`.)
//!
//! These are *limits / approximations* exposed through the same code path — see the
//! [`ClusterMapPreset`] helper constructors.
//!
//! # Optimiser
//!
//! Per epoch ClusterMap performs one stochastic pass over the graph edges
//! (attraction) interleaved with [`n_negative`](ClusterMapConfig::n_negative)
//! negatively sampled repulsions per edge, using a cosine-annealed learning rate
//! and the per-epoch temperature schedule. Negative samples are drawn with the
//! crate [`LcgRng`]. The embedding is re-centred each epoch.

use crate::error::{ManifoldError, ManifoldResult};
use crate::handle::LcgRng;
use crate::neighbor::knn_brute::knn_brute;

// ──────────────────────────────────────────────────────────────────────────────
// Public configuration
// ──────────────────────────────────────────────────────────────────────────────

/// Initialisation strategy for the ClusterMap embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterMapInit {
    /// Small Gaussian noise, `N(0, σ²)` with `σ = 1e-2`.
    Random,
    /// Top-`out_dim` PCA projection (rescaled), falling back to [`ClusterMapInit::Random`]
    /// when `out_dim > n_features`.
    Pca,
}

/// Hyper-parameter bundle for [`ClusterMap`].
///
/// See the [module documentation](crate::reduction::clustermap) for the meaning of
/// the kernel `(a, b)`, the attraction/repulsion exponents and the temperature
/// schedule, and for the parameter settings that recover t-SNE / UMAP / ForceAtlas2.
#[derive(Debug, Clone)]
pub struct ClusterMapConfig {
    /// Number of nearest neighbours per point used to build the graph (default: 15).
    pub n_neighbors: usize,
    /// Output embedding dimensionality (default: 2).
    pub out_dim: usize,
    /// Number of optimisation epochs (default: 500).
    pub n_epochs: usize,
    /// Initial (peak) learning rate; cosine-annealed to zero (default: 1.0).
    pub learning_rate: f64,
    /// Attractive-force exponent `α` (default: 1.0).
    pub attraction_exp: f64,
    /// Repulsive-force exponent `γ` (default: 1.0).
    pub repulsion_exp: f64,
    /// Kernel scale `a` in `1/(1 + a·d^{2b})` (default: 1.0 → Cauchy).
    pub a: f64,
    /// Kernel tail exponent `b` in `1/(1 + a·d^{2b})` (default: 1.0 → Cauchy).
    pub b: f64,
    /// Temperature at epoch 0 (default: 1.0).
    pub temperature_start: f64,
    /// Temperature at the final epoch; linearly annealed (default: 0.1).
    pub temperature_end: f64,
    /// Number of negative (repulsive) samples per edge per epoch (default: 5).
    pub n_negative: usize,
    /// Embedding initialisation (default: [`ClusterMapInit::Pca`]).
    pub init: ClusterMapInit,
    /// RNG seed used when an explicit RNG is not supplied to `fit_transform`.
    pub seed: u64,
}

impl Default for ClusterMapConfig {
    fn default() -> Self {
        Self {
            n_neighbors: 15,
            out_dim: 2,
            n_epochs: 500,
            learning_rate: 1.0,
            attraction_exp: 1.0,
            repulsion_exp: 1.0,
            a: 1.0,
            b: 1.0,
            temperature_start: 1.0,
            temperature_end: 0.1,
            n_negative: 5,
            init: ClusterMapInit::Pca,
            seed: 0,
        }
    }
}

/// Named presets that instantiate the unified model in the regime of a classic
/// neighbour-embedding method (see the module-level parameter table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterMapPreset {
    /// Cauchy kernel with low temperature — strong, well-separated repulsion.
    Tsne,
    /// Heavier-tailed kernel with moderate temperature — tight local clusters.
    Umap,
    /// Flat kernel with high temperature — graph-layout / spring behaviour.
    ForceAtlas2,
}

impl ClusterMapConfig {
    /// Build a configuration in the regime of `preset`, keeping `n_neighbors`,
    /// `out_dim`, `n_epochs` and `seed` from `self` and overriding the kernel /
    /// temperature / exponents.
    #[must_use]
    pub fn with_preset(mut self, preset: ClusterMapPreset) -> Self {
        match preset {
            ClusterMapPreset::Tsne => {
                self.a = 1.0;
                self.b = 1.0;
                self.attraction_exp = 1.0;
                self.repulsion_exp = 1.0;
                self.temperature_start = 0.10;
                self.temperature_end = 0.05;
            }
            ClusterMapPreset::Umap => {
                self.a = 1.577;
                self.b = 0.895;
                self.attraction_exp = 1.0;
                self.repulsion_exp = 1.0;
                self.temperature_start = 1.0;
                self.temperature_end = 0.8;
            }
            ClusterMapPreset::ForceAtlas2 => {
                // ForceAtlas2 is a graph-layout method: roughly linear spring
                // attraction and a soft long-range repulsion. In the contrastive
                // kernel family this corresponds to a *flat-tailed* kernel
                // (`b < 1`, so the similarity decays slowly with distance, giving
                // the long reach of FA2's gravity) with the repulsion down-weighted
                // by a high temperature so clusters stay compact rather than
                // exploding the way a sharp t-SNE kernel would. `b` is kept small
                // but strictly positive so the kernel derivative (hence both
                // forces) does not vanish.
                self.a = 1.0;
                self.b = 0.5;
                self.attraction_exp = 1.0;
                self.repulsion_exp = 1.0;
                self.temperature_start = 1.5;
                self.temperature_end = 0.5;
            }
        }
        self
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Optimiser handle
// ──────────────────────────────────────────────────────────────────────────────

/// A fitted / fittable ClusterMap optimiser.
///
/// Construct with [`ClusterMap::new`], then call [`ClusterMap::fit_transform`].
/// After fitting, [`ClusterMap::temperature_history`] exposes the annealed
/// temperature schedule that was applied.
#[derive(Debug, Clone)]
pub struct ClusterMap {
    config: ClusterMapConfig,
    temperature_history: Vec<f64>,
    loss_history: Vec<f64>,
}

impl ClusterMap {
    /// Create a new optimiser from a validated configuration.
    ///
    /// # Errors
    /// Returns [`ManifoldError::InvalidParameter`] / [`ManifoldError::InvalidConfiguration`]
    /// for non-positive learning rate, non-positive kernel parameters, `out_dim == 0`,
    /// or a non-finite temperature schedule.
    pub fn new(config: ClusterMapConfig) -> ManifoldResult<Self> {
        validate_config(&config)?;
        Ok(Self {
            config,
            temperature_history: Vec::new(),
            loss_history: Vec::new(),
        })
    }

    /// The configuration this optimiser was built with.
    #[must_use]
    pub fn config(&self) -> &ClusterMapConfig {
        &self.config
    }

    /// The per-epoch temperature schedule recorded during the last `fit_transform`.
    #[must_use]
    pub fn temperature_history(&self) -> &[f64] {
        &self.temperature_history
    }

    /// The per-epoch (approximate) contrastive loss recorded during the last
    /// `fit_transform`, sampled at the start and every few epochs.
    #[must_use]
    pub fn loss_history(&self) -> &[f64] {
        &self.loss_history
    }

    /// Fit the embedding and return it row-major as `n_samples × out_dim`.
    ///
    /// # Parameters
    /// - `data`       — row-major input matrix of shape `[n_samples × n_features]`.
    /// - `n_samples`  — number of rows.
    /// - `n_features` — number of input dimensions.
    /// - `rng`        — RNG used for negative sampling and (random) initialisation.
    ///
    /// # Errors
    /// Returns [`ManifoldError`] for shape/length mismatch, empty input, or
    /// `n_neighbors >= n_samples` (the kNN graph cannot be built).
    pub fn fit_transform(
        &mut self,
        data: &[f64],
        n_samples: usize,
        n_features: usize,
        rng: &mut LcgRng,
    ) -> ManifoldResult<Vec<f64>> {
        // ── 1. Validate inputs ───────────────────────────────────────────────
        if n_samples == 0 || n_features == 0 {
            return Err(ManifoldError::EmptyInput);
        }
        if data.len() != n_samples * n_features {
            return Err(ManifoldError::ShapeMismatch {
                expected: vec![n_samples, n_features],
                got: vec![data.len()],
            });
        }
        if n_samples < 3 {
            return Err(ManifoldError::InvalidParameter {
                name: "n_samples".into(),
                reason: format!("need at least 3 samples, got {n_samples}"),
            });
        }
        if self.config.n_neighbors == 0 || self.config.n_neighbors >= n_samples {
            return Err(ManifoldError::KNeighborsTooLarge {
                k: self.config.n_neighbors,
                n: n_samples,
            });
        }

        let n = n_samples;
        let d_in = n_features;
        let d_out = self.config.out_dim;
        let k = self.config.n_neighbors;

        // ── 2. Build symmetrised kNN graph (edge list + weights) ─────────────
        let graph = build_knn_graph(data, n, d_in, k)?;

        // ── 3. Initialise embedding ──────────────────────────────────────────
        let mut y = init_embedding(data, n, d_in, d_out, self.config.init, rng)?;

        // ── 4. Optimise the unified attraction/repulsion model ───────────────
        self.temperature_history = Vec::with_capacity(self.config.n_epochs.max(1));
        self.loss_history = Vec::new();
        self.run_optimisation(&mut y, &graph, n, d_out, rng);

        Ok(y)
    }

    /// Core SGD loop: attraction over edges + negatively sampled repulsion,
    /// with cosine-annealed learning rate and a linearly annealed temperature.
    fn run_optimisation(
        &mut self,
        y: &mut [f64],
        graph: &KnnGraph,
        n: usize,
        d_out: usize,
        rng: &mut LcgRng,
    ) {
        let cfg = self.config.clone();
        let n_epochs = cfg.n_epochs;

        if n_epochs == 0 {
            // Record the starting temperature and a single loss snapshot so the
            // schedule is never empty.
            self.temperature_history.push(cfg.temperature_start);
            self.loss_history
                .push(self.eval_loss(y, graph, n, d_out, cfg.temperature_start, rng));
            return;
        }

        let n_edges = graph.src.len();
        for epoch in 0..n_epochs {
            let frac = if n_epochs > 1 {
                epoch as f64 / (n_epochs - 1) as f64
            } else {
                0.0
            };
            // Linear temperature annealing start → end.
            let temperature =
                cfg.temperature_start + (cfg.temperature_end - cfg.temperature_start) * frac;
            self.temperature_history.push(temperature);

            // Cosine-annealed learning rate (peak → 0).
            let alpha = 0.5 * cfg.learning_rate * (1.0 + (std::f64::consts::PI * frac).cos());
            let alpha = alpha.max(1e-4 * cfg.learning_rate);

            let inv_t = 1.0 / temperature.max(MIN_TEMPERATURE);

            for e in 0..n_edges {
                let i = graph.src[e];
                let j = graph.dst[e];
                let w = graph.weight[e];

                // ── Attraction along the graph edge ─────────────────────────
                apply_attraction(y, i, j, d_out, w, cfg.attraction_exp, cfg.a, cfg.b, alpha);

                // ── Repulsion from negatively sampled non-neighbours ────────
                for _ in 0..cfg.n_negative {
                    let m = sample_negative(rng, n, i, j);
                    apply_repulsion(
                        y,
                        i,
                        m,
                        d_out,
                        cfg.repulsion_exp * inv_t,
                        cfg.a,
                        cfg.b,
                        alpha,
                    );
                }
            }

            // Keep the centre of mass fixed to prevent drift.
            centre_embedding(y, n, d_out);

            // Sample the loss occasionally for diagnostics.
            if epoch == 0 || (epoch + 1) % LOSS_SAMPLE_EVERY == 0 || epoch + 1 == n_epochs {
                let loss = self.eval_loss(y, graph, n, d_out, temperature, rng);
                self.loss_history.push(loss);
            }
        }
    }

    /// Evaluate the (approximate) contrastive loss with a fresh batch of negative
    /// samples. Used only for diagnostics, so it does not perturb the embedding.
    fn eval_loss(
        &self,
        y: &[f64],
        graph: &KnnGraph,
        n: usize,
        d_out: usize,
        temperature: f64,
        rng: &mut LcgRng,
    ) -> f64 {
        let cfg = &self.config;
        let inv_t = 1.0 / temperature.max(MIN_TEMPERATURE);
        let mut loss = 0.0_f64;
        for e in 0..graph.src.len() {
            let i = graph.src[e];
            let j = graph.dst[e];
            let w = graph.weight[e];
            let d2 = sq_dist(y, i, j, d_out);
            let p = kernel(d2, cfg.a, cfg.b);
            loss -= w * cfg.attraction_exp * (p.max(KERNEL_EPS)).ln();
            for _ in 0..cfg.n_negative {
                let m = sample_negative_const(rng, n, i, j);
                let d2m = sq_dist(y, i, m, d_out);
                let pm = kernel(d2m, cfg.a, cfg.b);
                loss -= cfg.repulsion_exp * inv_t * ((1.0 - pm).max(KERNEL_EPS)).ln();
            }
        }
        loss
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────────

/// Floor for the temperature to avoid division blow-ups.
const MIN_TEMPERATURE: f64 = 1e-6;
/// Numerical guard inside logarithms / `1 − p`.
const KERNEL_EPS: f64 = 1e-12;
/// Floor for `d²` inside the `(d²)^{b−1}` factor when `b < 1`.
const DIST_EPS: f64 = 1e-10;
/// Per-coordinate clamp on the SGD step (UMAP-style gradient clipping).
const STEP_CLAMP: f64 = 4.0;
/// Sample the diagnostic loss every this many epochs.
const LOSS_SAMPLE_EVERY: usize = 25;

// ──────────────────────────────────────────────────────────────────────────────
// Validation
// ──────────────────────────────────────────────────────────────────────────────

fn validate_config(cfg: &ClusterMapConfig) -> ManifoldResult<()> {
    if cfg.out_dim == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "out_dim".into(),
            reason: "must be at least 1".into(),
        });
    }
    if cfg.out_dim > 16 {
        return Err(ManifoldError::InvalidParameter {
            name: "out_dim".into(),
            reason: format!("must be in 1..=16, got {}", cfg.out_dim),
        });
    }
    if !(cfg.learning_rate.is_finite()) || cfg.learning_rate <= 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "learning_rate".into(),
            reason: "must be a positive finite number".into(),
        });
    }
    if !cfg.a.is_finite() || cfg.a <= 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "a".into(),
            reason: "kernel scale must be a positive finite number".into(),
        });
    }
    if !cfg.b.is_finite() || cfg.b < 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "b".into(),
            reason: "kernel tail exponent must be a non-negative finite number".into(),
        });
    }
    if !cfg.attraction_exp.is_finite() || cfg.attraction_exp < 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "attraction_exp".into(),
            reason: "must be a non-negative finite number".into(),
        });
    }
    if !cfg.repulsion_exp.is_finite() || cfg.repulsion_exp < 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "repulsion_exp".into(),
            reason: "must be a non-negative finite number".into(),
        });
    }
    if !cfg.temperature_start.is_finite() || cfg.temperature_start <= 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "temperature_start".into(),
            reason: "must be a positive finite number".into(),
        });
    }
    if !cfg.temperature_end.is_finite() || cfg.temperature_end <= 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "temperature_end".into(),
            reason: "must be a positive finite number".into(),
        });
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// kNN graph
// ──────────────────────────────────────────────────────────────────────────────

/// A symmetrised, weighted directed edge list of the kNN graph.
///
/// Each undirected neighbour relation `{i, j}` appears once as `(i → j)`; the
/// gradients are applied antisymmetrically so both endpoints move.
#[derive(Debug, Clone)]
struct KnnGraph {
    src: Vec<usize>,
    dst: Vec<usize>,
    weight: Vec<f64>,
}

/// Build a symmetrised kNN graph from row-major `data`.
///
/// Edge weights are the UMAP-style membership strengths
/// `exp(−(d_ij − ρ_i) / σ_i)` symmetrised by the probabilistic t-conorm
/// `μ_ij = a + b − a·b`, which yields a graph in `[0, 1]` that emphasises strong
/// local neighbours. Reuses [`knn_brute`] for the neighbour search.
fn build_knn_graph(data: &[f64], n: usize, dim: usize, k: usize) -> ManifoldResult<KnnGraph> {
    let (idx, sq) = knn_brute(data, n, dim, k)?;
    // Euclidean (not squared) distances for the smooth-knn fit.
    let dist: Vec<f64> = sq.iter().map(|v| v.sqrt()).collect();

    // Per-row ρ_i (nearest non-zero distance) and σ_i (bandwidth solving
    // Σ_j exp(−(d_ij − ρ_i)/σ_i) = log2 k).
    let (sigmas, rhos) = smooth_knn_bandwidths(&dist, n, k);

    // Directed memberships v_{i→j} = exp(−(d_ij − ρ_i)/σ_i) into a dense map so we
    // can symmetrise with the t-conorm. We store only the kNN entries.
    let target = (k as f64).log2().max(1e-12);
    let _ = target; // bandwidth already fitted; kept for documentation parity.

    // Build a hash-free symmetric accumulation: store directed weights then merge.
    // membership[i][pos] corresponds to neighbour idx[i*k+pos].
    let mut directed = vec![0.0_f64; n * k];
    for i in 0..n {
        let sigma = sigmas[i].max(1e-12);
        let rho = rhos[i];
        for pos in 0..k {
            let d = dist[i * k + pos];
            let arg = (d - rho).max(0.0);
            directed[i * k + pos] = (-arg / sigma).exp();
        }
    }

    // Symmetrise with probabilistic t-conorm μ = a + b − a·b.
    // For each directed edge (i→j) look up the reverse membership (j→i) if present.
    let mut src = Vec::with_capacity(n * k);
    let mut dst = Vec::with_capacity(n * k);
    let mut weight = Vec::with_capacity(n * k);

    for i in 0..n {
        for pos in 0..k {
            let j = idx[i * k + pos];
            // Only emit each undirected edge once (i < j); when j < i the edge is
            // emitted from j's side (or here if j has no reciprocal entry).
            let a_ij = directed[i * k + pos];
            let a_ji = reverse_membership(&idx, &directed, n, k, j, i);
            let mu = a_ij + a_ji - a_ij * a_ji;
            if mu <= 0.0 {
                continue;
            }
            if i < j {
                src.push(i);
                dst.push(j);
                weight.push(mu);
            } else if i > j {
                // Emit only if j does NOT list i among its neighbours (otherwise
                // j's pass already emitted, or will emit, this undirected edge).
                if reverse_membership(&idx, &directed, n, k, j, i) == 0.0 {
                    src.push(j);
                    dst.push(i);
                    weight.push(mu);
                }
            }
            // i == j cannot occur (knn excludes self).
        }
    }

    if src.is_empty() {
        return Err(ManifoldError::DisconnectedGraph(n));
    }

    Ok(KnnGraph { src, dst, weight })
}

/// Look up the directed membership `v_{a→b}` if `b` is among `a`'s neighbours,
/// else `0.0`.
fn reverse_membership(
    idx: &[usize],
    directed: &[f64],
    _n: usize,
    k: usize,
    a: usize,
    b: usize,
) -> f64 {
    for pos in 0..k {
        if idx[a * k + pos] == b {
            return directed[a * k + pos];
        }
    }
    0.0
}

/// Fit per-row `(σ_i, ρ_i)` so that `Σ_j exp(−(d_ij − ρ_i)/σ_i) = log2 k`.
///
/// Bisection on `σ_i`; `ρ_i` is the smallest strictly-positive neighbour distance.
fn smooth_knn_bandwidths(dist: &[f64], n: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    let target = (k as f64).log2().max(1e-12);
    let mut sigmas = vec![1.0_f64; n];
    let mut rhos = vec![0.0_f64; n];
    for i in 0..n {
        let mut rho = 0.0_f64;
        for pos in 0..k {
            let d = dist[i * k + pos];
            if d > 0.0 {
                rho = d;
                break;
            }
        }
        rhos[i] = rho;

        let mut lo = 0.0_f64;
        let mut hi = f64::INFINITY;
        let mut mid = 1.0_f64;
        for _ in 0..64 {
            let mut s = 0.0_f64;
            for pos in 0..k {
                let d = dist[i * k + pos];
                let arg = (d - rho).max(0.0);
                s += (-arg / mid).exp();
            }
            if (s - target).abs() < 1e-5 {
                break;
            }
            if s > target {
                hi = mid;
                mid = 0.5 * (lo + hi);
            } else {
                lo = mid;
                if hi.is_infinite() {
                    mid *= 2.0;
                } else {
                    mid = 0.5 * (lo + hi);
                }
            }
        }
        sigmas[i] = mid.max(1e-12);
    }
    (sigmas, rhos)
}

// ──────────────────────────────────────────────────────────────────────────────
// Kernel + forces
// ──────────────────────────────────────────────────────────────────────────────

/// Generalised Cauchy / Student kernel `p = 1 / (1 + a·(d²)^b)`.
#[inline]
fn kernel(d2: f64, a: f64, b: f64) -> f64 {
    let pow = if b == 1.0 {
        d2
    } else if b == 0.0 {
        1.0
    } else {
        d2.max(0.0).powf(b)
    };
    1.0 / (1.0 + a * pow)
}

/// The shared kernel factor `a·b·(d²)^{b−1}` appearing in both gradients.
///
/// For `b == 0` the kernel is constant so the derivative vanishes (returns 0).
#[inline]
fn kernel_d_factor(d2: f64, a: f64, b: f64) -> f64 {
    if b == 0.0 {
        return 0.0;
    }
    let dd = d2.max(DIST_EPS);
    let pow_bm1 = if b == 1.0 { 1.0 } else { dd.powf(b - 1.0) };
    a * b * pow_bm1
}

/// Squared Euclidean distance between rows `i` and `j` of embedding `y`.
#[inline]
fn sq_dist(y: &[f64], i: usize, j: usize, d_out: usize) -> f64 {
    let mut s = 0.0_f64;
    let bi = i * d_out;
    let bj = j * d_out;
    for c in 0..d_out {
        let diff = y[bi + c] - y[bj + c];
        s += diff * diff;
    }
    s
}

/// Apply one attractive step pulling `i` and `j` together.
///
/// `∂L_a/∂y_i = w·α · (a·b·(d²)^{b−1}·p) · 2(y_i − y_j)` ⇒ descent moves `y_i` by
/// `−η · ∂L_a/∂y_i` (towards `y_j`).
#[inline]
#[allow(clippy::too_many_arguments)]
fn apply_attraction(
    y: &mut [f64],
    i: usize,
    j: usize,
    d_out: usize,
    w: f64,
    attraction_exp: f64,
    a: f64,
    b: f64,
    alpha: f64,
) {
    if i == j {
        return;
    }
    let d2 = sq_dist(y, i, j, d_out);
    let p = kernel(d2, a, b);
    // grad magnitude scalar (multiplies 2(y_i − y_j)); descent uses −η·grad.
    let g = w * attraction_exp * kernel_d_factor(d2, a, b) * p;
    let coeff = -alpha * 2.0 * g; // step coefficient on (y_i − y_j)
    let bi = i * d_out;
    let bj = j * d_out;
    for c in 0..d_out {
        let diff = y[bi + c] - y[bj + c];
        let step = (coeff * diff).clamp(-STEP_CLAMP, STEP_CLAMP);
        y[bi + c] += step;
        y[bj + c] -= step;
    }
}

/// Apply one repulsive step pushing `i` and `m` apart.
///
/// `∂L_r/∂y_i = −(γ/T) · (a·b·(d²)^{b−1}·p²/(1−p)) · 2(y_i − y_m)` ⇒ descent moves
/// `y_i` by `−η · ∂L_r/∂y_i` (away from `y_m`).
#[inline]
#[allow(clippy::too_many_arguments)]
fn apply_repulsion(
    y: &mut [f64],
    i: usize,
    m: usize,
    d_out: usize,
    rep_scaled: f64,
    a: f64,
    b: f64,
    alpha: f64,
) {
    if i == m {
        return;
    }
    let d2 = sq_dist(y, i, m, d_out);
    let p = kernel(d2, a, b);
    let one_minus_p = (1.0 - p).max(KERNEL_EPS);
    // grad magnitude scalar (multiplies 2(y_i − y_m)); negative ⇒ descent pushes apart.
    let g = -rep_scaled * kernel_d_factor(d2, a, b) * p * p / one_minus_p;
    let coeff = -alpha * 2.0 * g;
    let bi = i * d_out;
    let bm = m * d_out;
    for c in 0..d_out {
        let diff = y[bi + c] - y[bm + c];
        let step = (coeff * diff).clamp(-STEP_CLAMP, STEP_CLAMP);
        y[bi + c] += step;
        y[bm + c] -= step;
    }
}

/// Draw a uniform negative sample in `[0, n)` distinct from `i` and `j`
/// (bounded rejection, then falls back to any non-`i` index).
#[inline]
fn sample_negative(rng: &mut LcgRng, n: usize, i: usize, j: usize) -> usize {
    for _ in 0..8 {
        let m = rng.next_usize(n);
        if m != i && m != j {
            return m;
        }
    }
    // Deterministic fallback.
    let m = (i + 1) % n;
    if m != i { m } else { (i + 2) % n }
}

/// Const-context negative sampler used by the (read-only) loss evaluator.
#[inline]
fn sample_negative_const(rng: &mut LcgRng, n: usize, i: usize, j: usize) -> usize {
    sample_negative(rng, n, i, j)
}

/// Re-centre every output dimension of `y` to zero mean.
fn centre_embedding(y: &mut [f64], n: usize, d_out: usize) {
    if n == 0 {
        return;
    }
    let nf = n as f64;
    for c in 0..d_out {
        let mut mean = 0.0_f64;
        for i in 0..n {
            mean += y[i * d_out + c];
        }
        mean /= nf;
        for i in 0..n {
            y[i * d_out + c] -= mean;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Initialisation
// ──────────────────────────────────────────────────────────────────────────────

fn init_embedding(
    data: &[f64],
    n: usize,
    d_in: usize,
    d_out: usize,
    init: ClusterMapInit,
    rng: &mut LcgRng,
) -> ManifoldResult<Vec<f64>> {
    match init {
        ClusterMapInit::Random => Ok(random_embedding(n, d_out, rng)),
        ClusterMapInit::Pca => {
            if d_out > d_in {
                Ok(random_embedding(n, d_out, rng))
            } else {
                pca_embedding(data, n, d_in, d_out, rng)
            }
        }
    }
}

fn random_embedding(n: usize, d_out: usize, rng: &mut LcgRng) -> Vec<f64> {
    let mut y = vec![0.0_f64; n * d_out];
    for v in &mut y {
        *v = rng.next_normal() * 1e-2;
    }
    y
}

/// PCA initialisation via deflated power iteration on the centred data.
fn pca_embedding(
    data: &[f64],
    n: usize,
    d_in: usize,
    d_out: usize,
    rng: &mut LcgRng,
) -> ManifoldResult<Vec<f64>> {
    let nf = n as f64;
    let mut mean = vec![0.0_f64; d_in];
    for i in 0..n {
        for j in 0..d_in {
            mean[j] += data[i * d_in + j];
        }
    }
    for m in &mut mean {
        *m /= nf;
    }
    let mut centered = vec![0.0_f64; n * d_in];
    for i in 0..n {
        for j in 0..d_in {
            centered[i * d_in + j] = data[i * d_in + j] - mean[j];
        }
    }

    let mut components: Vec<Vec<f64>> = Vec::with_capacity(d_out);
    let mut residual = centered.clone();

    for _ in 0..d_out {
        let mut v: Vec<f64> = (0..d_in).map(|_| rng.next_normal()).collect();
        normalise_vec(&mut v);
        for _ in 0..64 {
            let mut xv = vec![0.0_f64; n];
            for i in 0..n {
                let mut acc = 0.0_f64;
                for j in 0..d_in {
                    acc += residual[i * d_in + j] * v[j];
                }
                xv[i] = acc;
            }
            let mut xtxv = vec![0.0_f64; d_in];
            for i in 0..n {
                let xi = xv[i];
                for j in 0..d_in {
                    xtxv[j] += residual[i * d_in + j] * xi;
                }
            }
            let norm = xtxv.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-14 {
                break;
            }
            for (vi, xi) in v.iter_mut().zip(&xtxv) {
                *vi = xi / norm;
            }
        }
        let mut proj = vec![0.0_f64; n];
        for i in 0..n {
            let mut acc = 0.0_f64;
            for j in 0..d_in {
                acc += residual[i * d_in + j] * v[j];
            }
            proj[i] = acc;
        }
        for i in 0..n {
            let pi = proj[i];
            for j in 0..d_in {
                residual[i * d_in + j] -= pi * v[j];
            }
        }
        components.push(v);
    }

    let mut y = vec![0.0_f64; n * d_out];
    for i in 0..n {
        for c in 0..d_out {
            let mut acc = 0.0_f64;
            for j in 0..d_in {
                acc += centered[i * d_in + j] * components[c][j];
            }
            y[i * d_out + c] = acc;
        }
    }
    // Scale each component to std ≈ 1e-2 (matches the random init magnitude).
    for c in 0..d_out {
        let mut var = 0.0_f64;
        for i in 0..n {
            var += y[i * d_out + c].powi(2);
        }
        var /= nf;
        let std_dev = var.sqrt().max(1e-14);
        for i in 0..n {
            y[i * d_out + c] = y[i * d_out + c] / std_dev * 1e-2;
        }
    }
    Ok(y)
}

fn normalise_vec(v: &mut [f64]) {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-14 {
        for vi in v.iter_mut() {
            *vi /= norm;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Two well-separated isotropic Gaussian blobs in `dim` dimensions.
    /// Cluster A is centred at `+offset` on every axis, cluster B at `−offset`.
    fn two_blobs(n_per: usize, dim: usize, offset: f64, seed: u64) -> Vec<f64> {
        let n = n_per * 2;
        let mut data = vec![0.0_f64; n * dim];
        let mut rng = LcgRng::new(seed);
        for i in 0..n {
            let centre = if i < n_per { offset } else { -offset };
            for d in 0..dim {
                data[i * dim + d] = centre + 0.25 * rng.next_normal();
            }
        }
        data
    }

    fn cluster_centre(emb: &[f64], start: usize, end: usize, d_out: usize) -> Vec<f64> {
        let count = (end - start) as f64;
        let mut c = vec![0.0_f64; d_out];
        for i in start..end {
            for k in 0..d_out {
                c[k] += emb[i * d_out + k];
            }
        }
        for x in &mut c {
            *x /= count;
        }
        c
    }

    fn l2(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn mean_intra(emb: &[f64], start: usize, end: usize, d_out: usize) -> f64 {
        let mut sum = 0.0_f64;
        let mut cnt = 0usize;
        for i in start..end {
            for j in (i + 1)..end {
                sum += l2(
                    &emb[i * d_out..(i + 1) * d_out],
                    &emb[j * d_out..(j + 1) * d_out],
                );
                cnt += 1;
            }
        }
        sum / cnt.max(1) as f64
    }

    // ── Test 1: output shape = n_samples × out_dim ───────────────────────────
    #[test]
    fn test_output_shape() {
        let n_per = 10;
        let n = n_per * 2;
        let dim = 5;
        let data = two_blobs(n_per, dim, 6.0, 1);
        let cfg = ClusterMapConfig {
            n_neighbors: 5,
            n_epochs: 20,
            n_negative: 3,
            ..ClusterMapConfig::default()
        };
        let mut cm = ClusterMap::new(cfg).expect("config valid");
        let mut rng = LcgRng::new(1);
        let emb = cm.fit_transform(&data, n, dim, &mut rng).expect("fit ok");
        assert_eq!(emb.len(), n * 2);
    }

    // ── Test 2: all-finite embedding ─────────────────────────────────────────
    #[test]
    fn test_all_finite() {
        let n_per = 12;
        let n = n_per * 2;
        let dim = 6;
        let data = two_blobs(n_per, dim, 5.0, 2);
        let cfg = ClusterMapConfig {
            n_neighbors: 6,
            n_epochs: 80,
            n_negative: 5,
            ..ClusterMapConfig::default()
        };
        let mut cm = ClusterMap::new(cfg).expect("config valid");
        let mut rng = LcgRng::new(2);
        let emb = cm.fit_transform(&data, n, dim, &mut rng).expect("fit ok");
        for (i, v) in emb.iter().enumerate() {
            assert!(v.is_finite(), "emb[{i}] = {v} is not finite");
        }
    }

    // ── Test 3: two blobs stay separated (Cauchy / default) ──────────────────
    #[test]
    fn test_cluster_separation_default() {
        let n_per = 20;
        let n = n_per * 2;
        let dim = 8;
        let data = two_blobs(n_per, dim, 8.0, 3);
        let cfg = ClusterMapConfig {
            n_neighbors: 8,
            n_epochs: 400,
            n_negative: 5,
            learning_rate: 1.0,
            seed: 3,
            ..ClusterMapConfig::default()
        };
        let mut cm = ClusterMap::new(cfg.clone()).expect("config valid");
        let mut rng = LcgRng::new(3);
        let emb = cm.fit_transform(&data, n, dim, &mut rng).expect("fit ok");
        let d_out = cfg.out_dim;

        let ca = cluster_centre(&emb, 0, n_per, d_out);
        let cb = cluster_centre(&emb, n_per, n, d_out);
        let inter = l2(&ca, &cb);
        let intra = 0.5 * (mean_intra(&emb, 0, n_per, d_out) + mean_intra(&emb, n_per, n, d_out));

        assert!(
            inter > intra,
            "inter-cluster {inter:.4} must exceed intra-cluster {intra:.4}"
        );
    }

    // ── Test 4: t-SNE preset separates clusters ──────────────────────────────
    #[test]
    fn test_tsne_preset_separation() {
        let n_per = 18;
        let n = n_per * 2;
        let dim = 7;
        let data = two_blobs(n_per, dim, 7.0, 4);
        let cfg = ClusterMapConfig {
            n_neighbors: 7,
            n_epochs: 400,
            n_negative: 5,
            seed: 4,
            ..ClusterMapConfig::default()
        }
        .with_preset(ClusterMapPreset::Tsne);
        let mut cm = ClusterMap::new(cfg.clone()).expect("config valid");
        let mut rng = LcgRng::new(4);
        let emb = cm.fit_transform(&data, n, dim, &mut rng).expect("fit ok");
        let d_out = cfg.out_dim;
        let ca = cluster_centre(&emb, 0, n_per, d_out);
        let cb = cluster_centre(&emb, n_per, n, d_out);
        let inter = l2(&ca, &cb);
        let intra = 0.5 * (mean_intra(&emb, 0, n_per, d_out) + mean_intra(&emb, n_per, n, d_out));
        assert!(emb.iter().all(|v| v.is_finite()));
        assert!(
            inter > intra,
            "t-SNE preset: inter {inter:.4} must exceed intra {intra:.4}"
        );
    }

    // ── Test 5: UMAP preset separates clusters ───────────────────────────────
    #[test]
    fn test_umap_preset_separation() {
        let n_per = 18;
        let n = n_per * 2;
        let dim = 7;
        let data = two_blobs(n_per, dim, 7.0, 5);
        let cfg = ClusterMapConfig {
            n_neighbors: 7,
            n_epochs: 400,
            n_negative: 5,
            seed: 5,
            ..ClusterMapConfig::default()
        }
        .with_preset(ClusterMapPreset::Umap);
        let mut cm = ClusterMap::new(cfg.clone()).expect("config valid");
        let mut rng = LcgRng::new(5);
        let emb = cm.fit_transform(&data, n, dim, &mut rng).expect("fit ok");
        let d_out = cfg.out_dim;
        let ca = cluster_centre(&emb, 0, n_per, d_out);
        let cb = cluster_centre(&emb, n_per, n, d_out);
        let inter = l2(&ca, &cb);
        let intra = 0.5 * (mean_intra(&emb, 0, n_per, d_out) + mean_intra(&emb, n_per, n, d_out));
        assert!(emb.iter().all(|v| v.is_finite()));
        assert!(
            inter > intra,
            "UMAP preset: inter {inter:.4} must exceed intra {intra:.4}"
        );
    }

    // ── Test 6: ForceAtlas2 preset produces a valid, separated layout ────────
    #[test]
    fn test_forceatlas2_preset() {
        let n_per = 16;
        let n = n_per * 2;
        let dim = 6;
        let data = two_blobs(n_per, dim, 7.0, 6);
        let cfg = ClusterMapConfig {
            n_neighbors: 6,
            n_epochs: 400,
            n_negative: 5,
            seed: 6,
            ..ClusterMapConfig::default()
        }
        .with_preset(ClusterMapPreset::ForceAtlas2);
        let mut cm = ClusterMap::new(cfg.clone()).expect("config valid");
        let mut rng = LcgRng::new(6);
        let emb = cm.fit_transform(&data, n, dim, &mut rng).expect("fit ok");
        let d_out = cfg.out_dim;
        assert!(emb.iter().all(|v| v.is_finite()));
        let ca = cluster_centre(&emb, 0, n_per, d_out);
        let cb = cluster_centre(&emb, n_per, n, d_out);
        let inter = l2(&ca, &cb);
        let intra = 0.5 * (mean_intra(&emb, 0, n_per, d_out) + mean_intra(&emb, n_per, n, d_out));
        assert!(
            inter > intra,
            "ForceAtlas2 preset: inter {inter:.4} must exceed intra {intra:.4}"
        );
    }

    // ── Test 7: determinism with the same seed ───────────────────────────────
    #[test]
    fn test_determinism_same_seed() {
        let n_per = 12;
        let n = n_per * 2;
        let dim = 5;
        let data = two_blobs(n_per, dim, 6.0, 7);
        let cfg = ClusterMapConfig {
            n_neighbors: 5,
            n_epochs: 60,
            n_negative: 4,
            seed: 99,
            ..ClusterMapConfig::default()
        };
        let mut cm1 = ClusterMap::new(cfg.clone()).expect("config valid");
        let mut cm2 = ClusterMap::new(cfg).expect("config valid");
        let mut r1 = LcgRng::new(99);
        let mut r2 = LcgRng::new(99);
        let e1 = cm1.fit_transform(&data, n, dim, &mut r1).expect("fit 1");
        let e2 = cm2.fit_transform(&data, n, dim, &mut r2).expect("fit 2");
        assert_eq!(e1, e2, "same seed must give identical embeddings");
    }

    // ── Test 8: temperature is annealed monotonically (down) ─────────────────
    #[test]
    fn test_temperature_monotone_decrease() {
        let n_per = 10;
        let n = n_per * 2;
        let dim = 4;
        let data = two_blobs(n_per, dim, 6.0, 8);
        let cfg = ClusterMapConfig {
            n_neighbors: 5,
            n_epochs: 50,
            n_negative: 3,
            temperature_start: 1.0,
            temperature_end: 0.1,
            ..ClusterMapConfig::default()
        };
        let mut cm = ClusterMap::new(cfg).expect("config valid");
        let mut rng = LcgRng::new(8);
        let _ = cm.fit_transform(&data, n, dim, &mut rng).expect("fit ok");
        let hist = cm.temperature_history();
        assert!(hist.len() >= 2, "expected a multi-epoch schedule");
        for w in hist.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-12,
                "temperature must be non-increasing: {} then {}",
                w[0],
                w[1]
            );
        }
        // Endpoints match the configured schedule.
        assert!((hist[0] - 1.0).abs() < 1e-9);
        assert!((hist[hist.len() - 1] - 0.1).abs() < 1e-9);
    }

    // ── Test 9: error — n_neighbors >= n_samples ─────────────────────────────
    #[test]
    fn test_error_neighbors_too_large() {
        let n = 8;
        let dim = 3;
        let data = two_blobs(4, dim, 5.0, 9);
        let cfg = ClusterMapConfig {
            n_neighbors: 8, // == n_samples
            n_epochs: 10,
            ..ClusterMapConfig::default()
        };
        let mut cm = ClusterMap::new(cfg).expect("config valid");
        let mut rng = LcgRng::new(9);
        let res = cm.fit_transform(&data, n, dim, &mut rng);
        assert!(matches!(res, Err(ManifoldError::KNeighborsTooLarge { .. })));
    }

    // ── Test 10: error — out_dim == 0 (at construction) ──────────────────────
    #[test]
    fn test_error_out_dim_zero() {
        let cfg = ClusterMapConfig {
            out_dim: 0,
            ..ClusterMapConfig::default()
        };
        let res = ClusterMap::new(cfg);
        assert!(matches!(res, Err(ManifoldError::InvalidParameter { .. })));
    }

    // ── Test 11: error — data length mismatch ────────────────────────────────
    #[test]
    fn test_error_length_mismatch() {
        let n = 10;
        let dim = 4;
        // Provide one element too few.
        let data = vec![0.0_f64; n * dim - 1];
        let cfg = ClusterMapConfig {
            n_neighbors: 4,
            n_epochs: 5,
            ..ClusterMapConfig::default()
        };
        let mut cm = ClusterMap::new(cfg).expect("config valid");
        let mut rng = LcgRng::new(10);
        let res = cm.fit_transform(&data, n, dim, &mut rng);
        assert!(matches!(res, Err(ManifoldError::ShapeMismatch { .. })));
    }

    // ── Test 12: error — empty input (zero features) ─────────────────────────
    #[test]
    fn test_error_empty_features() {
        let cfg = ClusterMapConfig {
            n_neighbors: 2,
            n_epochs: 5,
            ..ClusterMapConfig::default()
        };
        let mut cm = ClusterMap::new(cfg).expect("config valid");
        let mut rng = LcgRng::new(11);
        let res = cm.fit_transform(&[], 5, 0, &mut rng);
        assert!(matches!(res, Err(ManifoldError::EmptyInput)));
    }

    // ── Test 13: out_dim = 3 produces a finite 3-D embedding ─────────────────
    #[test]
    fn test_three_dimensional_output() {
        let n_per = 12;
        let n = n_per * 2;
        let dim = 6;
        let data = two_blobs(n_per, dim, 6.0, 12);
        let cfg = ClusterMapConfig {
            n_neighbors: 6,
            out_dim: 3,
            n_epochs: 60,
            n_negative: 4,
            ..ClusterMapConfig::default()
        };
        let mut cm = ClusterMap::new(cfg).expect("config valid");
        let mut rng = LcgRng::new(12);
        let emb = cm.fit_transform(&data, n, dim, &mut rng).expect("fit ok");
        assert_eq!(emb.len(), n * 3);
        assert!(emb.iter().all(|v| v.is_finite()));
    }

    // ── Test 14: n_epochs = 0 yields a valid (init) embedding + schedule ─────
    #[test]
    fn test_zero_epochs() {
        let n_per = 8;
        let n = n_per * 2;
        let dim = 4;
        let data = two_blobs(n_per, dim, 5.0, 13);
        let cfg = ClusterMapConfig {
            n_neighbors: 4,
            n_epochs: 0,
            n_negative: 3,
            ..ClusterMapConfig::default()
        };
        let mut cm = ClusterMap::new(cfg.clone()).expect("config valid");
        let mut rng = LcgRng::new(13);
        let emb = cm.fit_transform(&data, n, dim, &mut rng).expect("fit ok");
        assert_eq!(emb.len(), n * cfg.out_dim);
        assert!(emb.iter().all(|v| v.is_finite()));
        assert_eq!(cm.temperature_history().len(), 1);
    }

    // ── Test 15: kernel + gradient sanity ────────────────────────────────────
    #[test]
    fn test_kernel_and_gradient_factor() {
        // Cauchy kernel at d²=0 is 1, decreasing in d².
        assert!((kernel(0.0, 1.0, 1.0) - 1.0).abs() < 1e-12);
        assert!(kernel(1.0, 1.0, 1.0) < kernel(0.5, 1.0, 1.0));
        // b == 0 → constant kernel, zero derivative factor.
        assert_eq!(kernel_d_factor(3.0, 1.0, 0.0), 0.0);
        assert!((kernel(3.0, 1.0, 0.0) - 0.5).abs() < 1e-12);
        // Numerical gradient check of the attraction step direction:
        // moving two points one attraction step must not increase their distance.
        let mut y = vec![0.0, 0.0, 1.0, 0.5]; // two 2-D points
        let before = sq_dist(&y, 0, 1, 2);
        apply_attraction(&mut y, 0, 1, 2, 1.0, 1.0, 1.0, 1.0, 0.1);
        let after = sq_dist(&y, 0, 1, 2);
        assert!(after <= before, "attraction must not increase distance");
        // Repulsion must not decrease distance.
        let mut y2 = vec![0.0, 0.0, 1.0, 0.5];
        let b0 = sq_dist(&y2, 0, 1, 2);
        apply_repulsion(&mut y2, 0, 1, 2, 1.0, 1.0, 1.0, 0.1);
        let a0 = sq_dist(&y2, 0, 1, 2);
        assert!(a0 >= b0, "repulsion must not decrease distance");
    }

    // ── Test 16: loss history recorded and finite ────────────────────────────
    #[test]
    fn test_loss_history_finite() {
        let n_per = 12;
        let n = n_per * 2;
        let dim = 5;
        let data = two_blobs(n_per, dim, 6.0, 14);
        let cfg = ClusterMapConfig {
            n_neighbors: 5,
            n_epochs: 60,
            n_negative: 4,
            seed: 14,
            ..ClusterMapConfig::default()
        };
        let mut cm = ClusterMap::new(cfg).expect("config valid");
        let mut rng = LcgRng::new(14);
        let _ = cm.fit_transform(&data, n, dim, &mut rng).expect("fit ok");
        let lh = cm.loss_history();
        assert!(!lh.is_empty(), "loss history must not be empty");
        assert!(lh.iter().all(|v| v.is_finite()), "loss must be finite");
    }
}
