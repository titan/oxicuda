//! Analytic backward pass (explicit gradients) for the NODE soft oblivious
//! decision tree / ensemble (Popov et al. 2019).
//!
//! A soft oblivious tree routes a sample through `depth` sigmoid-smoothed splits
//! that share, at each level, a single (feature-selection, threshold) pair.  The
//! per-level feature selection uses **entmax-1.5**, so the backward must include
//! the entmax Jacobian.
//!
//! Forward (per tree, per level `ℓ`):
//! ```text
//! p^ℓ = entmax15(feature_logits^ℓ)                 (sparse feature weights)
//! x̃^ℓ = Σ_j p^ℓ_j x_j                              (soft selected feature)
//! b^ℓ = σ(β (x̃^ℓ − threshold^ℓ))                   (soft split decision)
//! leaf_prob[L] = Π_ℓ ( bit_ℓ(L)==1 ? b^ℓ : 1−b^ℓ )
//! out = Σ_L leaf_prob[L] · leaf_value[L]
//! ```
//! The ensemble averages the per-tree outputs.
//!
//! All gradients are checked against central finite differences in the tests.

use super::node::{NodeEnsemble, NodeTree};
use crate::error::TabularResult;

// ─── Gradient container (single tree) ──────────────────────────────────────────

/// Accumulated gradients for one [`NodeTree`].
#[derive(Debug, Clone)]
pub struct NodeTreeGradients {
    /// Gradient w.r.t. per-level feature-selection logits, `[depth * input_dim]`.
    pub feature_logits: Vec<f32>,
    /// Gradient w.r.t. per-level thresholds, `[depth]`.
    pub thresholds: Vec<f32>,
    /// Gradient w.r.t. leaf values, `[2^depth * output_dim]`.
    pub leaf_values: Vec<f32>,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// entmax-1.5 with the active-support weights `s_i = z_i − τ` (so `p_i = s_i²`).
///
/// Returns `(p, s)` where `s_i = 0` off support.  Mirrors the bisection used in
/// [`crate::attention::sparsemax::entmax15`].
fn entmax15_support(z: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let z_max = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let z_min = z.iter().cloned().fold(f32::INFINITY, f32::min);
    let mut lo = z_min - 2.0;
    let mut hi = z_max;
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        let sum: f32 = z.iter().map(|&zi| (zi - mid).max(0.0).powi(2)).sum();
        if sum > 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let tau = 0.5 * (lo + hi);
    let s: Vec<f32> = z.iter().map(|&zi| (zi - tau).max(0.0)).collect();
    let p: Vec<f32> = s.iter().map(|&si| si * si).collect();
    (p, s)
}

/// Backprop through the entmax-1.5 transform defined by `p_i = (z_i − τ)²`.
///
/// Jacobian on the support `S`: `∂p_i/∂z_j = 2 s_i (δ_ij − s_j / Σ_{k∈S} s_k)`,
/// hence `d_z_j = 2 s_j (d_p_j − (Σ_{k∈S} s_k d_p_k) / Σ_{k∈S} s_k)` for `j∈S`,
/// and `0` off support.
fn entmax15_backward(d_p: &[f32], s: &[f32]) -> Vec<f32> {
    let t: f32 = s.iter().sum();
    if t <= 0.0 {
        return vec![0.0_f32; d_p.len()];
    }
    let weighted: f32 = s.iter().zip(d_p.iter()).map(|(&si, &dp)| si * dp).sum();
    let corr = weighted / t;
    s.iter()
        .zip(d_p.iter())
        .map(|(&si, &dp)| {
            if si > 0.0 {
                2.0 * si * (dp - corr)
            } else {
                0.0
            }
        })
        .collect()
}

// ─── Per-level forward cache ───────────────────────────────────────────────────

struct LevelCache {
    feat_probs: Vec<f32>, // p^ℓ                 [input_dim]
    feat_s: Vec<f32>,     // support weights s^ℓ [input_dim]
    b: f32,               // split decision      scalar
}

impl NodeTree {
    /// Forward caching everything needed for backprop.
    fn forward_cached(&self, x: &[f32]) -> TabularResult<(Vec<f32>, Vec<LevelCache>, Vec<f32>)> {
        let depth = self.depth_ref();
        let input_dim = self.input_dim_ref();
        let output_dim = self.output_dim_ref();
        let beta = self.beta_ref();
        let n_leaves = 1usize << depth;

        let mut leaf_probs = vec![1.0_f32; n_leaves];
        let mut levels = Vec::with_capacity(depth);
        let logits = self.feature_logits_ref();
        let thresholds = self.thresholds_ref();

        for level in 0..depth {
            let lvl = &logits[level * input_dim..(level + 1) * input_dim];
            let (feat_probs, feat_s) = entmax15_support(lvl);
            let selected_x: f32 = feat_probs
                .iter()
                .zip(x.iter())
                .map(|(&p, &xi)| p * xi)
                .sum();
            let b = sigmoid(beta * (selected_x - thresholds[level]));
            for (leaf, lp) in leaf_probs.iter_mut().enumerate() {
                let bit = (leaf >> (depth - 1 - level)) & 1;
                *lp *= if bit == 1 { b } else { 1.0 - b };
            }
            levels.push(LevelCache {
                feat_probs,
                feat_s,
                b,
            });
        }

        let leaf_values = self.leaf_values_ref();
        let mut out = vec![0.0_f32; output_dim];
        for (leaf, &lp) in leaf_probs.iter().enumerate() {
            let base = leaf * output_dim;
            for (d, ov) in out.iter_mut().enumerate() {
                *ov += lp * leaf_values[base + d];
            }
        }
        Ok((out, levels, leaf_probs))
    }

    /// Analytic backward pass for a single tree.
    ///
    /// `grad_out` is `dL/d out` (`[output_dim]`).  Returns the parameter
    /// gradients and the gradient w.r.t. the input (`[input_dim]`).
    pub fn backward(
        &self,
        x: &[f32],
        grad_out: &[f32],
    ) -> TabularResult<(NodeTreeGradients, Vec<f32>)> {
        let depth = self.depth_ref();
        let input_dim = self.input_dim_ref();
        let output_dim = self.output_dim_ref();
        let beta = self.beta_ref();
        let n_leaves = 1usize << depth;

        let (_out, levels, leaf_probs) = self.forward_cached(x)?;
        let mut g = NodeTreeGradients {
            feature_logits: vec![0.0_f32; depth * input_dim],
            thresholds: vec![0.0_f32; depth],
            leaf_values: vec![0.0_f32; n_leaves * output_dim],
        };
        let mut d_x = vec![0.0_f32; input_dim];

        // ── Output: out = Σ_L leaf_prob[L] · leaf_value[L] ───────────────────
        let leaf_values = self.leaf_values_ref();
        let mut d_leaf_prob = vec![0.0_f32; n_leaves];
        for leaf in 0..n_leaves {
            let base = leaf * output_dim;
            for d in 0..output_dim {
                let go = grad_out[d];
                g.leaf_values[base + d] += go * leaf_probs[leaf];
                d_leaf_prob[leaf] += go * leaf_values[base + d];
            }
        }

        // ── Each level's b: leaf_prob = Π_ℓ factor_ℓ. ───────────────────────
        // ∂leaf_prob/∂factor_ℓ = leaf_prob / factor_ℓ (use the cached product;
        // since some factors can be ~0, divide via the cached b carefully by
        // reconstructing the product-excluding-level).  depth is small so the
        // O(depth) reconstruction per leaf is cheap and numerically safe.
        for level in 0..depth {
            let b = levels[level].b;
            // d_b accumulated over all leaves.
            let mut d_b = 0.0_f32;
            for (leaf, &dlp) in d_leaf_prob.iter().enumerate() {
                let bit = (leaf >> (depth - 1 - level)) & 1;
                // product of factors excluding this level
                let mut prod_excl = 1.0_f32;
                for (l2, lc) in levels.iter().enumerate() {
                    if l2 == level {
                        continue;
                    }
                    let bit2 = (leaf >> (depth - 1 - l2)) & 1;
                    prod_excl *= if bit2 == 1 { lc.b } else { 1.0 - lc.b };
                }
                // factor = bit==1 ? b : 1-b ; dfactor/db = bit==1 ? 1 : -1
                let sign = if bit == 1 { 1.0 } else { -1.0 };
                d_b += dlp * prod_excl * sign;
            }

            // b = σ(β (x̃ − θ)) ;  db/d(arg) = b(1-b)·β
            let feat_probs = &levels[level].feat_probs;
            let d_arg = d_b * b * (1.0 - b) * beta;

            // arg = x̃ − θ ; ∂/∂θ = −1
            g.thresholds[level] += -d_arg;

            // x̃ = Σ_j p_j x_j ; ∂x̃/∂p_j = x_j ; ∂x̃/∂x_j = p_j
            let mut d_feat_probs = vec![0.0_f32; input_dim];
            for j in 0..input_dim {
                d_feat_probs[j] = d_arg * x[j];
                d_x[j] += d_arg * feat_probs[j];
            }

            // through entmax-1.5 → d_logits for this level
            let d_logits = entmax15_backward(&d_feat_probs, &levels[level].feat_s);
            for (j, &dl) in d_logits.iter().enumerate() {
                g.feature_logits[level * input_dim + j] += dl;
            }
        }

        Ok((g, d_x))
    }
}

impl NodeEnsemble {
    /// Analytic backward for the ensemble (mean over trees).
    ///
    /// `grad_out` is `dL/d out` (`[output_dim]`).  Each tree receives
    /// `grad_out / n_trees`.  Returns per-tree gradients and the summed input
    /// gradient.
    pub fn backward(
        &self,
        x: &[f32],
        grad_out: &[f32],
    ) -> TabularResult<(Vec<NodeTreeGradients>, Vec<f32>)> {
        let n = self.trees_ref().len();
        let scale = 1.0 / n as f32;
        let scaled: Vec<f32> = grad_out.iter().map(|&g| g * scale).collect();
        let mut grads = Vec::with_capacity(n);
        let mut d_x = vec![0.0_f32; x.len()];
        for tree in self.trees_ref() {
            let (g, dx) = tree.backward(x, &scaled)?;
            for (a, &v) in d_x.iter_mut().zip(dx.iter()) {
                *a += v;
            }
            grads.push(g);
        }
        Ok((grads, d_x))
    }
}

// ─── Parameter handle for finite-difference tests ──────────────────────────────

/// Addresses one scalar NODE-tree parameter (test-only).
#[cfg(test)]
pub(crate) enum NodeParam {
    FeatLogit(usize),
    Threshold(usize),
    Leaf(usize),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn tiny() -> (NodeTree, Vec<f32>) {
        let mut rng = LcgRng::new(321);
        // Wider random init so entmax has a clear, stable support away from kinks.
        let mut tree = NodeTree::new(3, 5, 2, &mut rng).expect("new");
        // Bump feature logits so entmax support is well-defined (not all-tied).
        for (i, v) in tree.feature_logits_mut_for_test().iter_mut().enumerate() {
            *v = 0.3 * (i as f32 % 5.0) - 0.5;
        }
        let x = vec![0.7_f32, -0.3, 0.5, 0.1, -0.8];
        (tree, x)
    }

    fn loss(out: &[f32], dir: &[f32]) -> f32 {
        out.iter().zip(dir.iter()).map(|(&a, &b)| a * b).sum()
    }

    #[test]
    fn forward_cached_matches_forward() {
        let (tree, x) = tiny();
        let f1 = tree.forward(&x).expect("forward");
        let (f2, _l, _lp) = tree.forward_cached(&x).expect("cached");
        for (a, b) in f1.iter().zip(f2.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn grad_check_input() {
        let (tree, x) = tiny();
        let dir = vec![0.6_f32, -0.4];
        let (_g, dx) = tree.backward(&x, &dir).expect("bwd");
        let eps = 2e-3_f32;
        for j in 0..x.len() {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[j] += eps;
            xm[j] -= eps;
            let lp = loss(&tree.forward(&xp).expect("f+"), &dir);
            let lm = loss(&tree.forward(&xm).expect("f-"), &dir);
            let fd = (lp - lm) / (2.0 * eps);
            let rel = (fd - dx[j]).abs() / fd.abs().max(dx[j].abs()).max(1e-3);
            assert!(rel < 5e-2, "dx[{j}] analytic={} fd={fd} rel={rel}", dx[j]);
        }
    }

    #[test]
    fn grad_check_parameters() {
        let (mut tree, x) = tiny();
        let dir = vec![0.5_f32, 0.7];
        let (g, _dx) = tree.backward(&x, &dir).expect("bwd");
        let checks: Vec<(&str, NodeParam, f32)> = vec![
            (
                "feat_logit[2]",
                NodeParam::FeatLogit(2),
                g.feature_logits[2],
            ),
            (
                "feat_logit[7]",
                NodeParam::FeatLogit(7),
                g.feature_logits[7],
            ),
            (
                "feat_logit[12]",
                NodeParam::FeatLogit(12),
                g.feature_logits[12],
            ),
            ("threshold[0]", NodeParam::Threshold(0), g.thresholds[0]),
            ("threshold[2]", NodeParam::Threshold(2), g.thresholds[2]),
            ("leaf[0]", NodeParam::Leaf(0), g.leaf_values[0]),
            ("leaf[9]", NodeParam::Leaf(9), g.leaf_values[9]),
            ("leaf[15]", NodeParam::Leaf(15), g.leaf_values[15]),
        ];
        let eps = 2e-3_f32;
        for (label, p, analytic) in checks {
            let orig = tree.param_get(&p);
            tree.param_set(&p, orig + eps);
            let lp = loss(&tree.forward(&x).expect("f+"), &dir);
            tree.param_set(&p, orig - eps);
            let lm = loss(&tree.forward(&x).expect("f-"), &dir);
            tree.param_set(&p, orig);
            let fd = (lp - lm) / (2.0 * eps);
            let rel = (fd - analytic).abs() / fd.abs().max(analytic.abs()).max(1e-3);
            assert!(
                rel < 5e-2,
                "param {label}: analytic={analytic} fd={fd} rel={rel}"
            );
        }
    }

    #[test]
    fn ensemble_backward_shape() {
        let mut rng = LcgRng::new(99);
        let cfg = crate::tree::node::NodeConfig {
            n_trees: 4,
            depth: 3,
            input_dim: 5,
            output_dim: 2,
        };
        let ens = NodeEnsemble::new(cfg, &mut rng).expect("new");
        let x = vec![0.2_f32, 0.4, -0.1, 0.3, 0.0];
        let dir = vec![0.5_f32, -0.5];
        let (grads, dx) = ens.backward(&x, &dir).expect("bwd");
        assert_eq!(grads.len(), 4);
        assert_eq!(dx.len(), 5);
        assert!(dx.iter().all(|v| v.is_finite()));
    }
}
