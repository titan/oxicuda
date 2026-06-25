//! Trainable Instant-NGP hash grid: forward cache + analytic backward pass.
//!
//! Müller, Evans, Schied & Keller (2022), "Instant Neural Graphics Primitives
//! with a Multiresolution Hash Encoding" (SIGGRAPH).
//!
//! The plain [`crate::encoding::hash_grid::HashGrid`] is inference-only. This
//! module makes the encoding *trainable* by deriving the gradient of the
//! trilinear-interpolation read with respect to the hash-table entries and
//! accumulating it back into a gradient buffer of the same shape as the table.
//!
//! # Forward
//!
//! For a query point `x ∈ [0,1]³`, level `l`, feature `f`:
//! ```text
//! out[l, f] = Σ_{c ∈ corners(l)} w_c(x) · T[ off_l + bucket_l(c)·F + f ]
//! ```
//! where `w_c` is the trilinear weight of corner `c` (the eight integer cell
//! corners surrounding `x` at level `l`), `bucket_l(c)` is the spatial hash of
//! that corner, `off_l` the level's base offset into the flat table `T`, and
//! `F` the feature count per entry.
//!
//! # Backward
//!
//! Because the read is *linear* in `T`, the gradient of a scalar loss `L` with
//! respect to a table entry is the upstream gradient `dL/dout[l,f]` scattered
//! through the same trilinear weights:
//! ```text
//! dL/dT[ off_l + bucket_l(c)·F + f ] += w_c(x) · dL/dout[l, f].
//! ```
//! Several corners (or several points in a batch) can hash to the same bucket,
//! so contributions are *accumulated* (`+=`). This module provides:
//!
//! * [`TrainableHashGrid::forward`] — value plus a [`GridCache`] of the per-corner
//!   weights and table indices needed for the backward pass;
//! * [`TrainableHashGrid::backward`] — scatters an upstream gradient into the
//!   internal gradient buffer;
//! * [`TrainableHashGrid::step_sgd`] / [`TrainableHashGrid::step_adam`] — apply a
//!   gradient-descent update and zero the buffer.
//!
//! All RNG uses the crate [`LcgRng`]; no external crates.

use crate::encoding::hash_grid::{HashGrid, HashGridConfig};
use crate::error::{NerfError, NerfResult};
use crate::handle::LcgRng;

const PI2: u64 = 2_654_435_761;
const PI3: u64 = 805_459_861;

// ─── GridCache ─────────────────────────────────────────────────────────────────

/// Per-corner trilinear weights and flat table indices captured during a forward
/// pass, sufficient to scatter an upstream gradient on the backward pass.
///
/// For each level there are eight corners; `indices[level*8 + c]` is the flat
/// *base* index `off_l + bucket·F` of corner `c`, and `weights[level*8 + c]` its
/// trilinear weight `w_c`.
#[derive(Debug, Clone)]
pub struct GridCache {
    /// Flat base index (`off_l + bucket·F`) per `(level, corner)`.
    indices: Vec<usize>,
    /// Trilinear weight per `(level, corner)`.
    weights: Vec<f32>,
    /// Number of levels captured.
    n_levels: usize,
    /// Features per level (`F`).
    n_features: usize,
}

impl GridCache {
    /// Number of levels captured.
    #[must_use]
    pub fn n_levels(&self) -> usize {
        self.n_levels
    }

    /// Features per level.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

// ─── Adam moment state ───────────────────────────────────────────────────────

/// First/second moment buffers for an Adam update over the hash table.
#[derive(Debug, Clone)]
struct AdamState {
    m: Vec<f32>,
    v: Vec<f32>,
    t: u64,
}

// ─── TrainableHashGrid ─────────────────────────────────────────────────────────

/// A hash grid whose table entries can be optimised by gradient descent.
///
/// Wraps an inference [`HashGrid`] and owns a gradient buffer `grad` of the same
/// length as the table plus optional Adam moments.
#[derive(Debug, Clone)]
pub struct TrainableHashGrid {
    grid: HashGrid,
    /// Gradient accumulator, same length as `grid.data`.
    grad: Vec<f32>,
    /// Optional Adam moments (lazily allocated on first `step_adam`).
    adam: Option<AdamState>,
}

impl TrainableHashGrid {
    /// Build a trainable hash grid with a zeroed gradient buffer.
    ///
    /// # Errors
    ///
    /// Propagates [`HashGrid::new`] validation errors.
    pub fn new(cfg: HashGridConfig, rng: &mut LcgRng) -> NerfResult<Self> {
        let grid = HashGrid::new(cfg, rng)?;
        let grad = vec![0.0_f32; grid.data.len()];
        Ok(Self {
            grid,
            grad,
            adam: None,
        })
    }

    /// Wrap an existing inference grid (gradient buffer zeroed).
    #[must_use]
    pub fn from_grid(grid: HashGrid) -> Self {
        let grad = vec![0.0_f32; grid.data.len()];
        Self {
            grid,
            grad,
            adam: None,
        }
    }

    /// Borrow the underlying inference grid.
    #[must_use]
    pub fn grid(&self) -> &HashGrid {
        &self.grid
    }

    /// Output dimension `n_levels · F`.
    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.grid.output_dim()
    }

    /// Borrow the current gradient buffer (length = table length).
    #[must_use]
    pub fn grad(&self) -> &[f32] {
        &self.grad
    }

    /// Zero the gradient accumulator.
    pub fn zero_grad(&mut self) {
        for g in self.grad.iter_mut() {
            *g = 0.0;
        }
    }

    /// Forward query of a single point, returning the encoding and a
    /// [`GridCache`] for the backward pass.
    ///
    /// The encoding equals [`HashGrid::query`]; the cache additionally records
    /// the eight per-level corner weights and table indices.
    ///
    /// # Errors
    ///
    /// Currently infallible for valid configs, but returns
    /// [`NerfError::Internal`] if invariants are violated.
    pub fn forward(&self, xyz: [f32; 3]) -> NerfResult<(Vec<f32>, GridCache)> {
        let cfg = &self.grid.config;
        let t = 1_usize << cfg.log2_hashmap_size;
        let f = cfg.n_features_per_level;
        let n_levels = cfg.n_levels;
        let resolutions = self.grid.level_resolutions();
        if resolutions.len() != n_levels {
            return Err(NerfError::Internal {
                msg: "level resolution count mismatch".into(),
            });
        }

        let mut out = vec![0.0_f32; self.output_dim()];
        let mut indices = vec![0_usize; n_levels * 8];
        let mut weights = vec![0.0_f32; n_levels * 8];

        for (level, &n_l) in resolutions.iter().enumerate() {
            let sx = xyz[0].clamp(0.0, 1.0) * (n_l as f32);
            let sy = xyz[1].clamp(0.0, 1.0) * (n_l as f32);
            let sz = xyz[2].clamp(0.0, 1.0) * (n_l as f32);

            let ix = sx.floor() as i64;
            let iy = sy.floor() as i64;
            let iz = sz.floor() as i64;
            let fx = sx - ix as f32;
            let fy = sy - iy as f32;
            let fz = sz - iz as f32;

            let level_offset = level * t * f;
            let out_base = level * f;

            let mut corner = 0usize;
            for cx in 0_u8..=1 {
                for cy in 0_u8..=1 {
                    for cz in 0_u8..=1 {
                        let xi = ix + i64::from(cx);
                        let yi = iy + i64::from(cy);
                        let zi = iz + i64::from(cz);

                        let bucket = hash_coord(xi, yi, zi, t);
                        let w = trilinear_weight(fx, fy, fz, cx, cy, cz);
                        let base = level_offset + bucket * f;

                        for feat in 0..f {
                            out[out_base + feat] += w * self.grid.data[base + feat];
                        }

                        let slot = level * 8 + corner;
                        indices[slot] = base;
                        weights[slot] = w;
                        corner += 1;
                    }
                }
            }
        }

        let cache = GridCache {
            indices,
            weights,
            n_levels,
            n_features: f,
        };
        Ok((out, cache))
    }

    /// Scatter an upstream gradient `d_out` (length `output_dim`) into the
    /// gradient buffer using the cached corner weights.
    ///
    /// Accumulates (`+=`): call [`Self::zero_grad`] (or a `step_*`) between
    /// optimisation iterations. Multiple corners hashing to the same bucket
    /// correctly sum their contributions.
    ///
    /// # Errors
    ///
    /// [`NerfError::DimensionMismatch`] if `d_out.len() != output_dim`.
    pub fn backward(&mut self, cache: &GridCache, d_out: &[f32]) -> NerfResult<()> {
        if d_out.len() != self.output_dim() {
            return Err(NerfError::DimensionMismatch {
                expected: self.output_dim(),
                got: d_out.len(),
            });
        }
        let f = cache.n_features;
        for level in 0..cache.n_levels {
            let out_base = level * f;
            for corner in 0..8 {
                let slot = level * 8 + corner;
                let base = cache.indices[slot];
                let w = cache.weights[slot];
                if w == 0.0 {
                    continue;
                }
                for feat in 0..f {
                    self.grad[base + feat] += w * d_out[out_base + feat];
                }
            }
        }
        Ok(())
    }

    /// Forward a batch and accumulate gradients for all points in one call.
    ///
    /// `xyz_batch` is flat `[n·3]`; `d_out_batch` is flat `[n·output_dim]`.
    /// Returns the stacked encodings `[n·output_dim]`. Gradients are accumulated
    /// into the internal buffer (not zeroed first).
    ///
    /// # Errors
    ///
    /// [`NerfError::DimensionMismatch`] on shape mismatch.
    pub fn forward_backward_batch(
        &mut self,
        xyz_batch: &[f32],
        d_out_batch: &[f32],
        n: usize,
    ) -> NerfResult<Vec<f32>> {
        let od = self.output_dim();
        if xyz_batch.len() != n * 3 {
            return Err(NerfError::DimensionMismatch {
                expected: n * 3,
                got: xyz_batch.len(),
            });
        }
        if d_out_batch.len() != n * od {
            return Err(NerfError::DimensionMismatch {
                expected: n * od,
                got: d_out_batch.len(),
            });
        }
        let mut out = vec![0.0_f32; n * od];
        for i in 0..n {
            let xyz = [xyz_batch[i * 3], xyz_batch[i * 3 + 1], xyz_batch[i * 3 + 2]];
            let (enc, cache) = self.forward(xyz)?;
            out[i * od..(i + 1) * od].copy_from_slice(&enc);
            self.backward(&cache, &d_out_batch[i * od..(i + 1) * od])?;
        }
        Ok(out)
    }

    /// Plain SGD update `T ← T − lr · grad`, then zero the gradient buffer.
    ///
    /// # Errors
    ///
    /// [`NerfError::InvalidEmbeddingConfig`] if `lr` is non-finite or negative.
    pub fn step_sgd(&mut self, lr: f32) -> NerfResult<()> {
        if !lr.is_finite() || lr < 0.0 {
            return Err(NerfError::InvalidEmbeddingConfig {
                msg: format!("invalid learning rate {lr}"),
            });
        }
        for (w, g) in self.grid.data.iter_mut().zip(self.grad.iter()) {
            *w -= lr * *g;
        }
        self.zero_grad();
        Ok(())
    }

    /// Adam update with bias correction, then zero the gradient buffer.
    ///
    /// Uses the standard moment recursions with `eps` for numerical stability.
    ///
    /// # Errors
    ///
    /// [`NerfError::InvalidEmbeddingConfig`] for invalid hyper-parameters.
    pub fn step_adam(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32) -> NerfResult<()> {
        if !lr.is_finite() || lr < 0.0 {
            return Err(NerfError::InvalidEmbeddingConfig {
                msg: format!("invalid learning rate {lr}"),
            });
        }
        if !(0.0..1.0).contains(&beta1) || !(0.0..1.0).contains(&beta2) {
            return Err(NerfError::InvalidEmbeddingConfig {
                msg: "adam betas must be in [0, 1)".into(),
            });
        }
        if !eps.is_finite() || eps <= 0.0 {
            return Err(NerfError::InvalidEmbeddingConfig {
                msg: "adam eps must be positive".into(),
            });
        }

        let len = self.grid.data.len();
        let state = self.adam.get_or_insert_with(|| AdamState {
            m: vec![0.0_f32; len],
            v: vec![0.0_f32; len],
            t: 0,
        });
        state.t += 1;
        let bc1 = 1.0 - beta1.powi(state.t.min(i32::MAX as u64) as i32);
        let bc2 = 1.0 - beta2.powi(state.t.min(i32::MAX as u64) as i32);

        for i in 0..len {
            let g = self.grad[i];
            state.m[i] = beta1 * state.m[i] + (1.0 - beta1) * g;
            state.v[i] = beta2 * state.v[i] + (1.0 - beta2) * g * g;
            let m_hat = state.m[i] / bc1;
            let v_hat = state.v[i] / bc2;
            self.grid.data[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
        self.zero_grad();
        Ok(())
    }
}

// ─── Internal helpers (mirror encoding::hash_grid) ─────────────────────────────

/// Hash a grid cell coordinate to a bucket index in `[0, t)`.
#[inline]
fn hash_coord(xi: i64, yi: i64, zi: i64, t: usize) -> usize {
    let hx = xi as u64;
    let hy = (yi as u64).wrapping_mul(PI2);
    let hz = (zi as u64).wrapping_mul(PI3);
    (hx ^ hy ^ hz) as usize % t
}

/// Trilinear interpolation weight for corner `(cx, cy, cz)`.
#[inline]
fn trilinear_weight(fx: f32, fy: f32, fz: f32, cx: u8, cy: u8, cz: u8) -> f32 {
    let wx = if cx == 1 { fx } else { 1.0 - fx };
    let wy = if cy == 1 { fy } else { 1.0 - fy };
    let wz = if cz == 1 { fz } else { 1.0 - fz };
    wx * wy * wz
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> HashGridConfig {
        HashGridConfig {
            n_levels: 4,
            n_features_per_level: 2,
            log2_hashmap_size: 10,
            base_resolution: 4,
            max_resolution: 64,
        }
    }

    #[test]
    fn forward_matches_inference_query() {
        let mut rng = LcgRng::new(1);
        let g = TrainableHashGrid::new(cfg(), &mut rng).expect("new");
        let pt = [0.3, 0.55, 0.81];
        let (enc, _cache) = g.forward(pt).expect("forward");
        let reference = g.grid().query(pt).expect("query");
        assert_eq!(enc.len(), reference.len());
        for (a, b) in enc.iter().zip(reference.iter()) {
            assert!((a - b).abs() < 1e-6, "forward != inference: {a} vs {b}");
        }
    }

    #[test]
    fn cached_weights_sum_to_one_per_level() {
        let mut rng = LcgRng::new(2);
        let g = TrainableHashGrid::new(cfg(), &mut rng).expect("new");
        let (_enc, cache) = g.forward([0.27, 0.63, 0.11]).expect("forward");
        for level in 0..cache.n_levels {
            let mut s = 0.0_f32;
            for corner in 0..8 {
                s += cache.weights[level * 8 + corner];
            }
            assert!((s - 1.0).abs() < 1e-5, "level {level} weights sum={s}");
        }
    }

    #[test]
    fn backward_shape_check() {
        let mut rng = LcgRng::new(3);
        let mut g = TrainableHashGrid::new(cfg(), &mut rng).expect("new");
        let (_enc, cache) = g.forward([0.5, 0.5, 0.5]).expect("forward");
        let bad = vec![1.0_f32; g.output_dim() + 1];
        assert!(g.backward(&cache, &bad).is_err());
    }

    /// The analytic gradient of a single output component equals, by finite
    /// differences, exactly the cached trilinear weight of the corner storing
    /// that feature — because the read is linear in the table.
    #[test]
    fn analytic_backward_matches_finite_difference() {
        let mut rng = LcgRng::new(7);
        let mut g = TrainableHashGrid::new(cfg(), &mut rng).expect("new");
        let pt = [0.41, 0.18, 0.73];

        // Loss L = out[target]; dL/dout is a one-hot vector.
        let target = 3usize; // some output index < output_dim
        let od = g.output_dim();
        let mut d_out = vec![0.0_f32; od];
        d_out[target] = 1.0;

        let (_enc, cache) = g.forward(pt).expect("forward");
        g.backward(&cache, &d_out).expect("backward");
        let analytic = g.grad().to_vec();

        // Finite-difference dL/dT[j] for a handful of touched entries.
        let eps = 1e-3_f32;
        let touched: Vec<usize> = {
            let f = cache.n_features;
            let level = target / f;
            let feat = target % f;
            (0..8)
                .map(|c| cache.indices[level * 8 + c] + feat)
                .collect()
        };
        for &j in &touched {
            let base = g.grid.data[j];
            g.grid.data[j] = base + eps;
            let (plus, _) = g.forward(pt).expect("forward+");
            g.grid.data[j] = base - eps;
            let (minus, _) = g.forward(pt).expect("forward-");
            g.grid.data[j] = base;
            let fd = (plus[target] - minus[target]) / (2.0 * eps);
            assert!(
                (analytic[j] - fd).abs() < 1e-3,
                "grad[{j}] analytic={} fd={}",
                analytic[j],
                fd
            );
        }
    }

    /// One SGD step on a single-point regression must strictly reduce the
    /// squared error toward a target encoding.
    #[test]
    fn sgd_reduces_regression_loss() {
        let mut rng = LcgRng::new(11);
        let mut g = TrainableHashGrid::new(cfg(), &mut rng).expect("new");
        let pt = [0.62, 0.34, 0.49];
        let od = g.output_dim();
        let target = vec![0.5_f32; od];

        let loss = |grid: &TrainableHashGrid| -> f32 {
            let (enc, _) = grid.forward(pt).expect("forward");
            enc.iter()
                .zip(target.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
        };

        let before = loss(&g);
        for _ in 0..50 {
            let (enc, cache) = g.forward(pt).expect("forward");
            // dL/dout = 2(out - target)
            let d_out: Vec<f32> = enc
                .iter()
                .zip(target.iter())
                .map(|(a, b)| 2.0 * (a - b))
                .collect();
            g.backward(&cache, &d_out).expect("backward");
            g.step_sgd(0.5).expect("sgd");
        }
        let after = loss(&g);
        assert!(
            after < before,
            "SGD did not reduce loss: before={before}, after={after}"
        );
        assert!(after < 1e-3, "loss not driven low enough: {after}");
    }

    /// Adam should also reduce the same loss and zero the gradient buffer each step.
    #[test]
    fn adam_reduces_loss_and_zeros_grad() {
        let mut rng = LcgRng::new(13);
        let mut g = TrainableHashGrid::new(cfg(), &mut rng).expect("new");
        let pt = [0.2, 0.8, 0.4];
        let od = g.output_dim();
        let target = vec![-0.25_f32; od];

        let mut last = f32::INFINITY;
        for _ in 0..80 {
            let (enc, cache) = g.forward(pt).expect("forward");
            let d_out: Vec<f32> = enc
                .iter()
                .zip(target.iter())
                .map(|(a, b)| 2.0 * (a - b))
                .collect();
            g.backward(&cache, &d_out).expect("backward");
            g.step_adam(0.05, 0.9, 0.999, 1e-8).expect("adam");
            // grad zeroed after step
            assert!(g.grad().iter().all(|&v| v == 0.0));
            let (enc2, _) = g.forward(pt).expect("forward2");
            last = enc2
                .iter()
                .zip(target.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>();
        }
        assert!(last < 1e-2, "Adam loss not low enough: {last}");
    }

    #[test]
    fn batch_forward_backward_accumulates() {
        let mut rng = LcgRng::new(17);
        let mut g = TrainableHashGrid::new(cfg(), &mut rng).expect("new");
        let n = 3;
        let xyz = vec![
            0.1, 0.2, 0.3, //
            0.4, 0.5, 0.6, //
            0.7, 0.8, 0.9, //
        ];
        let od = g.output_dim();
        let d_out = vec![1.0_f32; n * od];
        let out = g.forward_backward_batch(&xyz, &d_out, n).expect("batch");
        assert_eq!(out.len(), n * od);
        // Some gradient entries must be non-zero after accumulation.
        assert!(g.grad().iter().any(|&v| v != 0.0));
    }

    #[test]
    fn invalid_lr_rejected() {
        let mut rng = LcgRng::new(19);
        let mut g = TrainableHashGrid::new(cfg(), &mut rng).expect("new");
        assert!(g.step_sgd(-1.0).is_err());
        assert!(g.step_sgd(f32::NAN).is_err());
        assert!(g.step_adam(0.1, 1.5, 0.999, 1e-8).is_err());
        assert!(g.step_adam(0.1, 0.9, 0.999, 0.0).is_err());
    }
}
