//! Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling.
//!
//! Reference: "Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time
//! Series Modeling", Liu et al., ICLR 2022.
//!
//! Pyraformer builds a multi-scale pyramidal graph over the input sequence, where
//! coarser scales are obtained by averaging consecutive tokens at the previous
//! scale (plus a learnable linear projection). Each node attends only to a small
//! neighbourhood within its own scale together with its parent at the next coarser
//! scale and its children at the next finer scale (the Pyramidal Attention Module,
//! PAM). This yields O(L) overall complexity in the number of attention edges
//! while preserving the ability to capture long-range dependencies through the
//! coarse-scale tokens.
//!
//! This pure-Rust CPU reference implements:
//!
//! 1. **Multi-scale coarsening** — linear pooling that averages groups of
//!    `coarsen_factor` consecutive tokens at the previous scale, then applies a
//!    learnable per-scale linear map `d_model → d_model`.
//! 2. **Pyramidal Attention Module (PAM)** — for each node at scale `s` we form
//!    a sparse attention set that contains its `window_size` intra-scale
//!    neighbours, its single parent at scale `s + 1` (when one exists), and its
//!    `coarsen_factor` children at scale `s − 1` (when they exist). Multi-head
//!    scaled dot-product attention is computed over this set.
//! 3. **Forward** — coarsen the input to all scales, run PAM, and return the
//!    finest-scale `[seq_len, d_model]` representation.
//!
//! All tensors are row-major with `d_model` as the innermost axis.

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for a Pyraformer encoder.
#[derive(Debug, Clone)]
pub struct PyraformerConfig {
    /// Finest-scale sequence length (number of nodes at scale 0).
    pub seq_len: usize,
    /// Token embedding dimension.
    pub d_model: usize,
    /// Number of attention heads (must divide `d_model`).
    pub n_heads: usize,
    /// Pyramid depth: total number of scales (>= 1; scale 0 is finest).
    pub n_scales: usize,
    /// Intra-scale neighbour window radius (in tokens, >= 1).
    pub window_size: usize,
    /// Downsample factor between scales (>= 2, typically 2).
    pub coarsen_factor: usize,
}

impl PyraformerConfig {
    /// Small configuration: `d_model = 16`, `n_heads = 2`, `n_scales = 3`,
    /// `window_size = 2`, `coarsen_factor = 2`.
    #[must_use]
    pub fn tiny(seq_len: usize) -> Self {
        Self {
            seq_len,
            d_model: 16,
            n_heads: 2,
            n_scales: 3,
            window_size: 2,
            coarsen_factor: 2,
        }
    }
}

// ─── Linear weight helper ────────────────────────────────────────────────────

/// Learnable affine map `y = x · W^T + b` with `W` row-major `[out_dim, in_dim]`.
#[derive(Debug, Clone)]
struct Linear {
    weight: Vec<f32>,
    bias: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl Linear {
    fn new(in_dim: usize, out_dim: usize, rng: &mut LcgRng) -> Self {
        let scale = (6.0_f32 / (in_dim + out_dim) as f32).sqrt();
        let mut weight = vec![0.0_f32; out_dim * in_dim];
        rng.fill_normal(&mut weight);
        for w in &mut weight {
            *w *= scale;
        }
        Self {
            weight,
            bias: vec![0.0_f32; out_dim],
            in_dim,
            out_dim,
        }
    }

    fn apply(&self, x: &[f32], out: &mut [f32]) {
        for (oi, slot) in out.iter_mut().enumerate().take(self.out_dim) {
            let row = &self.weight[oi * self.in_dim..(oi + 1) * self.in_dim];
            let mut acc = self.bias[oi];
            for (xv, wv) in x.iter().zip(row.iter()) {
                acc += *xv * *wv;
            }
            *slot = acc;
        }
    }
}

// ─── Multi-head attention weights ────────────────────────────────────────────

/// Q/K/V/O projection weights for a multi-head attention block.
#[derive(Debug, Clone)]
struct AttnWeights {
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_o: Vec<f32>,
}

impl AttnWeights {
    fn new(d_model: usize, rng: &mut LcgRng) -> Self {
        let scale = (2.0_f32 / d_model as f32).sqrt();
        let mut init_mat = || -> Vec<f32> {
            let mut v = vec![0.0_f32; d_model * d_model];
            rng.fill_normal(&mut v);
            for x in &mut v {
                *x *= scale;
            }
            v
        };
        Self {
            w_q: init_mat(),
            w_k: init_mat(),
            w_v: init_mat(),
            w_o: init_mat(),
        }
    }
}

// ─── Pyraformer model ────────────────────────────────────────────────────────

/// Pyraformer encoder.
///
/// Builds a multi-scale pyramidal token graph over the input sequence and
/// applies the Pyramidal Attention Module (PAM) to mix information along the
/// intra-scale neighbours and parent/child cross-scale edges. The finest scale
/// (scale 0) is returned as the output representation.
#[derive(Debug, Clone)]
pub struct Pyraformer {
    /// Per-scale coarsening linear maps (length `n_scales - 1`).
    ///
    /// Index `s` maps an averaged `d_model` vector at scale `s` to the
    /// coarser-scale `d_model` token at scale `s + 1`.
    coarsen: Vec<Linear>,
    /// Per-scale attention weights for the PAM (length `n_scales`).
    attn: Vec<AttnWeights>,
    /// Final output projection `d_model → d_model` applied per token.
    out_proj: Linear,
    /// Cached per-scale node counts (length `n_scales`).
    scale_sizes: Vec<usize>,
    /// Model configuration.
    cfg: PyraformerConfig,
}

impl Pyraformer {
    /// Build a Pyraformer encoder, initialising all weights.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidSequenceLength`] when `seq_len == 0`, or when the
    ///   pyramid would have an empty scale (`seq_len / coarsen_factor^(n_scales-1) < 1`).
    /// - [`TsError::InvalidEmbedDim`] when `d_model == 0`.
    /// - [`TsError::InvalidNumHeads`] when `n_heads == 0`.
    /// - [`TsError::HeadDimMismatch`] when `d_model % n_heads != 0`.
    /// - [`TsError::InvalidPoolSize`] when `n_scales == 0`.
    /// - [`TsError::InvalidKernelSize`] when `window_size == 0`.
    /// - [`TsError::InvalidStride`] when `coarsen_factor < 2`.
    pub fn new(cfg: PyraformerConfig, rng: &mut LcgRng) -> TsResult<Self> {
        if cfg.seq_len == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }
        if cfg.d_model == 0 {
            return Err(TsError::InvalidEmbedDim(0));
        }
        if cfg.n_heads == 0 {
            return Err(TsError::InvalidNumHeads(0));
        }
        if cfg.d_model % cfg.n_heads != 0 {
            return Err(TsError::HeadDimMismatch {
                embed_dim: cfg.d_model,
                n_heads: cfg.n_heads,
            });
        }
        if cfg.n_scales == 0 {
            return Err(TsError::InvalidPoolSize(0));
        }
        if cfg.window_size == 0 {
            return Err(TsError::InvalidKernelSize(0));
        }
        if cfg.coarsen_factor < 2 {
            return Err(TsError::InvalidStride(cfg.coarsen_factor));
        }

        // Compute per-scale node counts using integer division: scale s has
        // `seq_len / coarsen_factor^s` nodes. We require every scale to retain
        // at least one node; the coarsest-scale check enforces this.
        let mut scale_sizes = Vec::with_capacity(cfg.n_scales);
        scale_sizes.push(cfg.seq_len);
        for s in 1..cfg.n_scales {
            let prev = scale_sizes[s - 1];
            let next = prev / cfg.coarsen_factor;
            if next == 0 {
                return Err(TsError::InvalidSequenceLength(cfg.seq_len));
            }
            scale_sizes.push(next);
        }

        // Per-scale coarsening linear maps (`n_scales - 1` of them).
        let coarsen: Vec<Linear> = (0..cfg.n_scales.saturating_sub(1))
            .map(|_| Linear::new(cfg.d_model, cfg.d_model, rng))
            .collect();

        // One attention block per scale.
        let attn: Vec<AttnWeights> = (0..cfg.n_scales)
            .map(|_| AttnWeights::new(cfg.d_model, rng))
            .collect();

        let out_proj = Linear::new(cfg.d_model, cfg.d_model, rng);

        Ok(Self {
            coarsen,
            attn,
            out_proj,
            scale_sizes,
            cfg,
        })
    }

    /// Access the model configuration.
    #[must_use]
    #[inline]
    pub fn config(&self) -> &PyraformerConfig {
        &self.cfg
    }

    /// Per-scale node counts (length `n_scales`); scale 0 first.
    #[must_use]
    #[inline]
    pub fn scale_sizes(&self) -> &[usize] {
        &self.scale_sizes
    }

    /// Total number of nodes across all scales: `Σ_s seq_len / c^s`.
    #[must_use]
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.scale_sizes.iter().sum()
    }

    /// Coarsen the finest-scale input to all scales.
    ///
    /// * Input  `x` — `[seq_len, d_model]` row-major (scale 0).
    /// * Output    — `n_scales` per-scale `[n_s, d_model]` row-major tensors.
    ///
    /// Scale 0 is the input itself. Scale `s + 1` is obtained by averaging
    /// groups of `coarsen_factor` consecutive scale-`s` tokens and applying
    /// the per-scale learnable linear projection `d_model → d_model`.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `x.len() != seq_len * d_model`.
    pub fn coarsen_to_scales(&self, x: &[f32]) -> TsResult<Vec<Vec<f32>>> {
        let d = self.cfg.d_model;
        let expected = self.cfg.seq_len * d;
        if x.len() != expected {
            return Err(TsError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        let mut scales: Vec<Vec<f32>> = Vec::with_capacity(self.cfg.n_scales);
        scales.push(x.to_vec());

        let c = self.cfg.coarsen_factor;
        let inv_c = 1.0_f32 / c as f32;

        for s in 1..self.cfg.n_scales {
            let prev_size = self.scale_sizes[s - 1];
            let next_size = self.scale_sizes[s];
            let prev = &scales[s - 1];
            let mut next = vec![0.0_f32; next_size * d];

            // Use the linear map at index s-1 to project the averaged window.
            let lin_idx = s - 1;
            // Each coarsened token is the linear map of the average of the
            // corresponding `c` tokens at the previous scale.
            let mut avg = vec![0.0_f32; d];
            let mut tok = vec![0.0_f32; d];
            for ni in 0..next_size {
                for v in avg.iter_mut() {
                    *v = 0.0;
                }
                let base = ni * c;
                for ki in 0..c {
                    let src = (base + ki).min(prev_size - 1);
                    for di in 0..d {
                        avg[di] += prev[src * d + di];
                    }
                }
                for v in avg.iter_mut() {
                    *v *= inv_c;
                }
                self.coarsen[lin_idx].apply(&avg, &mut tok);
                next[ni * d..(ni + 1) * d].copy_from_slice(&tok);
            }
            scales.push(next);
        }

        Ok(scales)
    }

    /// Pyramidal Attention Module (PAM).
    ///
    /// For each node at scale `s`, its attention key/value set contains:
    ///
    /// * `window_size` intra-scale neighbours on each side (clamped to scale
    ///   boundary, so the set has at most `2 * window_size + 1` neighbours
    ///   including the node itself),
    /// * its parent at scale `s + 1` (when `s + 1 < n_scales`),
    /// * its `coarsen_factor` children at scale `s − 1` (when `s > 0`).
    ///
    /// Multi-head scaled dot-product attention is applied over this sparse
    /// set, and the per-scale tokens are updated as residual `token + attn`.
    /// Shapes are preserved: each input `[n_s, d_model]` becomes the same.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when any scale's tensor has the wrong length.
    pub fn pam_forward(&self, scales: &[Vec<f32>]) -> TsResult<Vec<Vec<f32>>> {
        if scales.len() != self.cfg.n_scales {
            return Err(TsError::DimensionMismatch {
                expected: self.cfg.n_scales,
                got: scales.len(),
            });
        }
        let d = self.cfg.d_model;
        for (s, sc) in scales.iter().enumerate() {
            let expected = self.scale_sizes[s] * d;
            if sc.len() != expected {
                return Err(TsError::DimensionMismatch {
                    expected,
                    got: sc.len(),
                });
            }
        }

        let n_heads = self.cfg.n_heads;
        let head_dim = d / n_heads;
        let scale_dot = (head_dim as f32).sqrt().recip();
        let window = self.cfg.window_size;
        let c = self.cfg.coarsen_factor;

        let mut output: Vec<Vec<f32>> = scales.iter().map(|sc| sc.to_vec()).collect();

        for s in 0..self.cfg.n_scales {
            let n_s = self.scale_sizes[s];
            let attn = &self.attn[s];

            // Pre-project Q/K/V for the current scale (single-shot per scale).
            let q_self = project_all(&scales[s], n_s, &attn.w_q, d);
            let k_self = project_all(&scales[s], n_s, &attn.w_k, d);
            let v_self = project_all(&scales[s], n_s, &attn.w_v, d);

            // Pre-project K/V for the parent scale (if any) and child scale (if any).
            let (k_parent, v_parent) = if s + 1 < self.cfg.n_scales {
                let n_p = self.scale_sizes[s + 1];
                (
                    Some(project_all(&scales[s + 1], n_p, &attn.w_k, d)),
                    Some(project_all(&scales[s + 1], n_p, &attn.w_v, d)),
                )
            } else {
                (None, None)
            };
            let (k_child, v_child) = if s > 0 {
                let n_c = self.scale_sizes[s - 1];
                (
                    Some(project_all(&scales[s - 1], n_c, &attn.w_k, d)),
                    Some(project_all(&scales[s - 1], n_c, &attn.w_v, d)),
                )
            } else {
                (None, None)
            };

            // Output buffer holds the attention delta (added to the input).
            let mut concat = vec![0.0_f32; n_s * d];

            for ni in 0..n_s {
                // Collect (k_idx_set) used as references into the projected matrices.
                // We avoid building an explicit Vec<(matrix, idx)> by handling each
                // class in turn and concatenating contributions head-wise.

                // Build the index lists for this node.
                let intra_lo = ni.saturating_sub(window);
                let intra_hi = (ni + window + 1).min(n_s);
                let intra: Vec<usize> = (intra_lo..intra_hi).collect();

                let parent_idx: Option<usize> = if s + 1 < self.cfg.n_scales {
                    Some(ni / c)
                } else {
                    None
                };

                let child_indices: Vec<usize> = if s > 0 {
                    let prev_size = self.scale_sizes[s - 1];
                    let base = ni * c;
                    (0..c).map(|k| (base + k).min(prev_size - 1)).collect()
                } else {
                    Vec::new()
                };

                // Compute attention per head, summing values from the three
                // index classes. We scale dot products by 1/sqrt(head_dim),
                // softmax over the union of classes, then weight the values.
                for h in 0..n_heads {
                    let h_off = h * head_dim;
                    // Collect all (kind, idx) — we will compute the dot scores
                    // first, build a softmax over them, then weight the values.

                    // total = intra + (parent?1:0) + child
                    let n_intra = intra.len();
                    let n_parent: usize = if parent_idx.is_some() { 1 } else { 0 };
                    let n_child = child_indices.len();
                    let total = n_intra + n_parent + n_child;
                    let mut scores = vec![0.0_f32; total];

                    // Intra-scale scores.
                    for (slot, &ki) in intra.iter().enumerate() {
                        let mut dot = 0.0_f32;
                        for hd in 0..head_dim {
                            dot += q_self[ni * d + h_off + hd] * k_self[ki * d + h_off + hd];
                        }
                        scores[slot] = dot * scale_dot;
                    }
                    let mut cursor = n_intra;
                    if let (Some(pi), Some(kp)) = (parent_idx, k_parent.as_ref()) {
                        let mut dot = 0.0_f32;
                        for hd in 0..head_dim {
                            dot += q_self[ni * d + h_off + hd] * kp[pi * d + h_off + hd];
                        }
                        scores[cursor] = dot * scale_dot;
                        cursor += 1;
                    }
                    if let Some(kc) = k_child.as_ref() {
                        for &ci in &child_indices {
                            let mut dot = 0.0_f32;
                            for hd in 0..head_dim {
                                dot += q_self[ni * d + h_off + hd] * kc[ci * d + h_off + hd];
                            }
                            scores[cursor] = dot * scale_dot;
                            cursor += 1;
                        }
                    }

                    softmax_row(&mut scores);

                    // Weighted sum of values across the three classes.
                    for hd in 0..head_dim {
                        let mut acc = 0.0_f32;
                        let mut slot = 0;
                        for &ki in &intra {
                            acc += scores[slot] * v_self[ki * d + h_off + hd];
                            slot += 1;
                        }
                        if let (Some(pi), Some(vp)) = (parent_idx, v_parent.as_ref()) {
                            acc += scores[slot] * vp[pi * d + h_off + hd];
                            slot += 1;
                        }
                        if let Some(vc) = v_child.as_ref() {
                            for &ci in &child_indices {
                                acc += scores[slot] * vc[ci * d + h_off + hd];
                                slot += 1;
                            }
                        }
                        concat[ni * d + h_off + hd] = acc;
                    }
                }
            }

            // Apply output projection W_o and residual into `output[s]`.
            let projected = project_all(&concat, n_s, &attn.w_o, d);
            let scale_out = &mut output[s];
            for i in 0..n_s * d {
                scale_out[i] = scales[s][i] + projected[i];
            }
        }

        Ok(output)
    }

    /// Full forward pass: `[seq_len, d_model]` → `[seq_len, d_model]`.
    ///
    /// Coarsens the input to all scales, runs the PAM, and applies a final
    /// per-token output projection to the finest scale.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `x.len() != seq_len * d_model`.
    pub fn forward(&self, x: &[f32]) -> TsResult<Vec<f32>> {
        let d = self.cfg.d_model;
        let scales = self.coarsen_to_scales(x)?;
        let attended = self.pam_forward(&scales)?;

        let finest = attended.first().ok_or_else(|| {
            TsError::Internal("pyraformer: pam_forward returned empty scales".to_string())
        })?;
        let n = self.cfg.seq_len;

        let mut out = vec![0.0_f32; n * d];
        let mut tmp = vec![0.0_f32; d];
        for ti in 0..n {
            self.out_proj.apply(&finest[ti * d..(ti + 1) * d], &mut tmp);
            out[ti * d..(ti + 1) * d].copy_from_slice(&tmp);
        }
        Ok(out)
    }
}

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Apply a single `d × d` linear map (row-major) to a `[n, d]` batch.
fn project_all(x: &[f32], n: usize, w: &[f32], d: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n * d];
    for ti in 0..n {
        for oi in 0..d {
            let row = oi * d;
            let mut acc = 0.0_f32;
            for ki in 0..d {
                acc += x[ti * d + ki] * w[row + ki];
            }
            out[ti * d + oi] = acc;
        }
    }
    out
}

/// Numerically stable in-place softmax over a row.
fn softmax_row(row: &mut [f32]) {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv_sum = if sum == 0.0 { 1.0 } else { sum.recip() };
    for v in row.iter_mut() {
        *v *= inv_sum;
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(2024)
    }

    fn tiny(seq_len: usize) -> PyraformerConfig {
        PyraformerConfig {
            seq_len,
            d_model: 8,
            n_heads: 2,
            n_scales: 3,
            window_size: 2,
            coarsen_factor: 2,
        }
    }

    // 1. coarsen_to_scales returns n_scales tensors with the expected per-scale lengths.
    #[test]
    fn coarsen_scale_lengths() {
        let mut rng = make_rng();
        let cfg = tiny(16);
        let model = Pyraformer::new(cfg.clone(), &mut rng).expect("build");
        let x = vec![0.3_f32; cfg.seq_len * cfg.d_model];
        let scales = model.coarsen_to_scales(&x).expect("coarsen");
        assert_eq!(scales.len(), cfg.n_scales);
        let expected_sizes: [usize; 3] = [16, 8, 4];
        for (s, sc) in scales.iter().enumerate() {
            assert_eq!(sc.len(), expected_sizes[s] * cfg.d_model);
        }
    }

    // 2. Finest-scale tensor length equals seq_len * d_model.
    #[test]
    fn coarsen_finest_length() {
        let mut rng = make_rng();
        let cfg = tiny(16);
        let model = Pyraformer::new(cfg.clone(), &mut rng).expect("build");
        let x = vec![0.5_f32; cfg.seq_len * cfg.d_model];
        let scales = model.coarsen_to_scales(&x).expect("coarsen");
        assert_eq!(scales[0].len(), cfg.seq_len * cfg.d_model);
    }

    // 3. Coarsest-scale tensor has at least d_model entries.
    #[test]
    fn coarsen_coarsest_nonempty() {
        let mut rng = make_rng();
        let cfg = tiny(16);
        let model = Pyraformer::new(cfg.clone(), &mut rng).expect("build");
        let x = vec![0.5_f32; cfg.seq_len * cfg.d_model];
        let scales = model.coarsen_to_scales(&x).expect("coarsen");
        let coarsest = scales.last().expect("scales");
        assert!(coarsest.len() >= cfg.d_model);
    }

    // 4. pam_forward preserves per-scale shapes.
    #[test]
    fn pam_preserves_shapes() {
        let mut rng = make_rng();
        let cfg = tiny(16);
        let model = Pyraformer::new(cfg.clone(), &mut rng).expect("build");
        let x = vec![0.2_f32; cfg.seq_len * cfg.d_model];
        let scales = model.coarsen_to_scales(&x).expect("coarsen");
        let attended = model.pam_forward(&scales).expect("pam");
        assert_eq!(attended.len(), scales.len());
        for (a, b) in attended.iter().zip(scales.iter()) {
            assert_eq!(a.len(), b.len());
        }
    }

    // 5. forward output length == seq_len * d_model.
    #[test]
    fn forward_output_length() {
        let mut rng = make_rng();
        let cfg = tiny(16);
        let model = Pyraformer::new(cfg.clone(), &mut rng).expect("build");
        let x = vec![0.25_f32; cfg.seq_len * cfg.d_model];
        let out = model.forward(&x).expect("forward");
        assert_eq!(out.len(), cfg.seq_len * cfg.d_model);
    }

    // 6. n_nodes formula matches Σ_s seq_len / c^s.
    #[test]
    fn n_nodes_formula() {
        let mut rng = make_rng();
        let cfg = PyraformerConfig {
            seq_len: 32,
            d_model: 8,
            n_heads: 2,
            n_scales: 4,
            window_size: 2,
            coarsen_factor: 2,
        };
        let model = Pyraformer::new(cfg, &mut rng).expect("build");
        // sizes: 32, 16, 8, 4 → sum 60
        assert_eq!(model.scale_sizes(), &[32, 16, 8, 4]);
        assert_eq!(model.n_nodes(), 60);
    }

    // 7. Deterministic given the same seed.
    #[test]
    fn deterministic_given_seed() {
        let cfg = tiny(16);
        let mut rng_a = LcgRng::new(91);
        let mut rng_b = LcgRng::new(91);
        let model_a = Pyraformer::new(cfg.clone(), &mut rng_a).expect("build");
        let model_b = Pyraformer::new(cfg.clone(), &mut rng_b).expect("build");
        let x: Vec<f32> = (0..cfg.seq_len * cfg.d_model)
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let out_a = model_a.forward(&x).expect("forward");
        let out_b = model_b.forward(&x).expect("forward");
        for (a, b) in out_a.iter().zip(out_b.iter()) {
            assert!((a - b).abs() < 1e-6, "non-deterministic: {a} vs {b}");
        }
    }

    // 8. err: seq_len == 0.
    #[test]
    fn err_seq_len_zero() {
        let mut rng = make_rng();
        let cfg = PyraformerConfig {
            seq_len: 0,
            ..tiny(16)
        };
        assert!(matches!(
            Pyraformer::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidSequenceLength(0)
        ));
    }

    // 9. err: d_model == 0.
    #[test]
    fn err_d_model_zero() {
        let mut rng = make_rng();
        let cfg = PyraformerConfig {
            d_model: 0,
            ..tiny(16)
        };
        assert!(matches!(
            Pyraformer::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidEmbedDim(0)
        ));
    }

    // 10. err: n_scales == 0.
    #[test]
    fn err_n_scales_zero() {
        let mut rng = make_rng();
        let cfg = PyraformerConfig {
            n_scales: 0,
            ..tiny(16)
        };
        assert!(matches!(
            Pyraformer::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidPoolSize(0)
        ));
    }

    // 11. err: n_heads == 0.
    #[test]
    fn err_n_heads_zero() {
        let mut rng = make_rng();
        let cfg = PyraformerConfig {
            n_heads: 0,
            ..tiny(16)
        };
        assert!(matches!(
            Pyraformer::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidNumHeads(0)
        ));
    }

    // 12. err: window_size == 0.
    #[test]
    fn err_window_size_zero() {
        let mut rng = make_rng();
        let cfg = PyraformerConfig {
            window_size: 0,
            ..tiny(16)
        };
        assert!(matches!(
            Pyraformer::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidKernelSize(0)
        ));
    }

    // 13. err: coarsen_factor < 2.
    #[test]
    fn err_coarsen_factor_too_small() {
        let mut rng = make_rng();
        let cfg = PyraformerConfig {
            coarsen_factor: 1,
            ..tiny(16)
        };
        assert!(matches!(
            Pyraformer::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidStride(1)
        ));
    }

    // 14. err: d_model % n_heads != 0.
    #[test]
    fn err_head_dim_mismatch() {
        let mut rng = make_rng();
        let cfg = PyraformerConfig {
            d_model: 9,
            n_heads: 2,
            ..tiny(16)
        };
        assert!(matches!(
            Pyraformer::new(cfg, &mut rng).unwrap_err(),
            TsError::HeadDimMismatch { .. }
        ));
    }

    // 15. err: x has the wrong length for coarsen_to_scales/forward.
    #[test]
    fn err_wrong_input_length() {
        let mut rng = make_rng();
        let cfg = tiny(16);
        let model = Pyraformer::new(cfg, &mut rng).expect("build");
        let bad = vec![0.0_f32; 17];
        assert!(matches!(
            model.coarsen_to_scales(&bad).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
        assert!(matches!(
            model.forward(&bad).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    // 16. err: seq_len not divisible by coarsen_factor^(n_scales-1) → coarsest scale 0.
    #[test]
    fn err_seq_len_too_small_for_depth() {
        let mut rng = make_rng();
        // n_scales=4 → need seq_len >= 2^3 = 8 (with coarsen_factor=2). 4 fails.
        let cfg = PyraformerConfig {
            seq_len: 4,
            n_scales: 4,
            d_model: 4,
            n_heads: 1,
            window_size: 1,
            coarsen_factor: 2,
        };
        assert!(matches!(
            Pyraformer::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidSequenceLength(4)
        ));
    }

    // 17. Single scale (n_scales = 1) → forward behaves like windowed attention.
    #[test]
    fn single_scale_forward() {
        let mut rng = make_rng();
        let cfg = PyraformerConfig {
            seq_len: 12,
            d_model: 8,
            n_heads: 2,
            n_scales: 1,
            window_size: 2,
            coarsen_factor: 2,
        };
        let model = Pyraformer::new(cfg.clone(), &mut rng).expect("build");
        assert_eq!(model.scale_sizes(), &[12]);
        let x = vec![0.3_f32; cfg.seq_len * cfg.d_model];
        let scales = model.coarsen_to_scales(&x).expect("coarsen");
        assert_eq!(scales.len(), 1);
        assert_eq!(scales[0].len(), cfg.seq_len * cfg.d_model);
        let out = model.forward(&x).expect("forward");
        assert_eq!(out.len(), cfg.seq_len * cfg.d_model);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // 18. Finest-scale output is finite for random input.
    #[test]
    fn forward_finite_random_input() {
        let mut rng = make_rng();
        let cfg = tiny(16);
        let model = Pyraformer::new(cfg.clone(), &mut rng).expect("build");
        let mut x = vec![0.0_f32; cfg.seq_len * cfg.d_model];
        rng.fill_normal(&mut x);
        let out = model.forward(&x).expect("forward");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite forward output"
        );
    }

    // 19. Changing one input position should change the finest-scale output.
    #[test]
    fn changing_input_changes_output() {
        let mut rng = make_rng();
        let cfg = tiny(16);
        let model = Pyraformer::new(cfg.clone(), &mut rng).expect("build");
        let mut x = vec![0.0_f32; cfg.seq_len * cfg.d_model];
        for (i, xv) in x.iter_mut().enumerate() {
            *xv = (i as f32 * 0.05).cos();
        }
        let out_a = model.forward(&x).expect("forward");
        let mut x2 = x.clone();
        // Perturb the first variate row.
        for xv in x2.iter_mut().take(cfg.d_model) {
            *xv += 0.5;
        }
        let out_b = model.forward(&x2).expect("forward");
        let mut max_diff = 0.0_f32;
        for (a, b) in out_a.iter().zip(out_b.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(max_diff > 1e-4, "perturbed input did not change output");
    }

    // 20. pam_forward errors on a wrong number of scales.
    #[test]
    fn err_pam_wrong_scale_count() {
        let mut rng = make_rng();
        let cfg = tiny(16);
        let model = Pyraformer::new(cfg, &mut rng).expect("build");
        let scales: Vec<Vec<f32>> = Vec::new();
        assert!(matches!(
            model.pam_forward(&scales).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    // 21. pam_forward errors on a wrong per-scale tensor length.
    #[test]
    fn err_pam_wrong_scale_length() {
        let mut rng = make_rng();
        let cfg = tiny(16);
        let model = Pyraformer::new(cfg, &mut rng).expect("build");
        // Build the right number of scales but with the wrong tensor lengths.
        let scales: Vec<Vec<f32>> = vec![vec![0.0_f32; 1]; 3];
        assert!(matches!(
            model.pam_forward(&scales).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }
}
