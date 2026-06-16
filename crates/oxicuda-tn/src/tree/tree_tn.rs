//! Balanced binary Tree Tensor Network (TTN) for hierarchical 1D quantum states.
//!
//! A TTN organises `n = 2^L` physical sites of local dimension `d` as the leaves
//! of a balanced binary tree. Every internal node is a rank-3 isometry that fuses
//! the two child bonds into one parent bond; the root caps the tree with a single
//! top tensor carrying the overall weight. This hierarchical layout captures
//! logarithmic-depth entanglement structure far more compactly than a linear MPS
//! for tree-like correlations (Shi-Duan-Vidal 2006).
//!
//! The decomposition is built top-down by repeated singular-value splittings: the
//! full state tensor is reshaped and SVD'd to separate its left and right halves,
//! the truncated right factor becomes that node's isometry, and the left factor
//! (the remaining "environment") is recursed into. Reconstruction contracts the
//! tree bottom-up.
//!
//! Storage is `f64`, row-major, reusing the crate's [`crate::svd::svd_jacobi`].
//!
//! # References
//! - Shi, Y.-Y., Duan, L.-M. & Vidal, G. (2006). "Classical simulation of quantum
//!   many-body systems with a tree tensor network". *Phys. Rev. A* 74, 022320.
//! - Tagliacozzo, L., Evenbly, G. & Vidal, G. (2009). "Simulation of
//!   two-dimensional quantum systems using a tree tensor network". *PRB* 80, 235127.

use crate::svd::{SvdResult, svd_jacobi};
use crate::{TnError, TnResult};

/// A single isometry node of a tree tensor network.
///
/// Shape is `(d_left, d_right, d_parent)` in row-major order: it fuses the left
/// and right child bonds (`d_left`, `d_right`) into the parent bond `d_parent`.
/// As an isometry, reshaping to `(d_left·d_right, d_parent)` gives a column-
/// orthonormal matrix (`W^† W = I`).
#[derive(Debug, Clone)]
pub struct TtnNode {
    /// Left child bond dimension.
    pub d_left: usize,
    /// Right child bond dimension.
    pub d_right: usize,
    /// Parent bond dimension.
    pub d_parent: usize,
    /// Row-major data of length `d_left * d_right * d_parent`.
    pub data: Vec<f64>,
}

impl TtnNode {
    /// Element `[l, r, p]`.
    #[inline]
    fn get(&self, l: usize, r: usize, p: usize) -> f64 {
        self.data[(l * self.d_right + r) * self.d_parent + p]
    }
}

/// A balanced binary tree tensor network over `2^num_layers` physical sites.
///
/// `nodes[layer]` holds the isometries at tree depth `layer`, ordered left to
/// right. Layer `0` is the row of leaves acting on physical legs; the deepest
/// stored layer feeds the single-column root weight `top`.
#[derive(Debug, Clone)]
pub struct TreeTensorNetwork {
    /// Physical (local Hilbert-space) dimension `d`.
    pub d: usize,
    /// Number of binary-tree layers `L`; there are `2^L` physical sites.
    pub num_layers: usize,
    /// Isometry layers, `nodes[0]` = leaves … `nodes[L-1]` = below the root.
    pub nodes: Vec<Vec<TtnNode>>,
    /// Root weight vector of length equal to the top bond dimension.
    pub top: Vec<f64>,
}

/// Reshape an SVD's `U` (m×k) and fold `S` into `Vt` (k×n), returning the
/// truncated isometry `U[:, :r]` and the remaining matrix `(diag(s)·Vt)[:r, :]`.
fn split_factors(svd: &SvdResult, r: usize) -> (Vec<f64>, Vec<f64>) {
    let m = svd.m;
    let n = svd.n;
    let k = svd.k;
    // Isometry: first r columns of U (m × r).
    let mut iso = vec![0.0_f64; m * r];
    for i in 0..m {
        for j in 0..r {
            iso[i * r + j] = svd.u[i * k + j];
        }
    }
    // Remainder: (r × n) with row j scaled by s[j].
    let mut rem = vec![0.0_f64; r * n];
    for j in 0..r {
        let sv = svd.s[j];
        for col in 0..n {
            rem[j * n + col] = sv * svd.vt[j * n + col];
        }
    }
    (iso, rem)
}

/// Number of singular values to retain: those exceeding a relative tolerance of
/// the largest, capped at `chi_max` and floored at 1.
fn effective_rank(s: &[f64], chi_max: usize) -> usize {
    if s.is_empty() {
        return 0;
    }
    let s_max = s[0].abs().max(1e-300);
    let tol = 1e-12 * s_max;
    let kept = s.iter().take_while(|&&sv| sv.abs() > tol).count();
    kept.clamp(1, chi_max)
}

impl TreeTensorNetwork {
    /// Number of physical sites `2^num_layers`.
    #[inline]
    pub fn n_sites(&self) -> usize {
        1usize << self.num_layers
    }

    /// Decompose a full state vector of length `d^(2^L)` into a TTN with bond
    /// dimension capped at `chi_max`.
    ///
    /// The construction proceeds layer by layer from the leaves up. At each layer
    /// the current set of "blocks" (each a matrix whose columns index the already-
    /// formed bond and whose rows index the rest of the system) is split: adjacent
    /// pairs of physical/bond legs are fused, SVD-truncated to `≤ chi_max`, and the
    /// orthonormal factor stored as that node's isometry while the weighted factor
    /// is carried upward.
    ///
    /// # Errors
    /// - [`TnError::InvalidConfiguration`] if `d < 2`, `num_layers == 0`, or
    ///   `chi_max == 0`.
    /// - [`TnError::ShapeMismatch`] if `state.len() != d^(2^L)`.
    /// - propagates SVD failures.
    pub fn from_state_vector(
        state: &[f64],
        d: usize,
        num_layers: usize,
        chi_max: usize,
    ) -> TnResult<Self> {
        if d < 2 || num_layers == 0 || chi_max == 0 {
            return Err(TnError::InvalidConfiguration(format!(
                "TTN requires d≥2, num_layers≥1, chi_max≥1 (got d={d}, L={num_layers}, χ={chi_max})"
            )));
        }
        let n_sites = 1usize << num_layers;
        let total: usize = d
            .checked_pow(n_sites as u32)
            .ok_or_else(|| TnError::InvalidConfiguration("state size overflow".to_string()))?;
        if state.len() != total {
            return Err(TnError::ShapeMismatch {
                expected: vec![total],
                got: vec![state.len()],
            });
        }

        // `blocks` holds, for the current layer, one matrix per node-to-form. We
        // process the tree breadth-first from the leaves. Initially the entire
        // state is a single column vector with `n_sites` open physical legs.
        //
        // Each layer halves the number of legs by fusing neighbours. We track the
        // per-leg bond dimension `leg_dims` and a single working tensor `work`
        // stored as a flat array whose index decomposes as
        // (leg_0, leg_1, …, leg_{m-1}) row-major.
        let mut leg_dims = vec![d; n_sites];
        let mut work = state.to_vec();

        let mut nodes: Vec<Vec<TtnNode>> = Vec::with_capacity(num_layers);

        for _layer in 0..num_layers {
            let n_pairs = leg_dims.len() / 2;
            let mut layer_nodes = Vec::with_capacity(n_pairs);

            // Fuse the original adjacent pairs left to right. After fusing the pair
            // at the moving position `pos`, the resulting parent leg occupies `pos`
            // and the next original pair begins at `pos + 1`. `work` and `leg_dims`
            // shrink in lock-step and stay contiguous throughout.
            for pos in 0..n_pairs {
                let dl = leg_dims[pos];
                let dr = leg_dims[pos + 1];
                let left_size: usize = leg_dims[..pos].iter().product();
                let right_size: usize = leg_dims[pos + 2..].iter().product();

                // Reshape `work` as (left_size, dl·dr, right_size); SVD with the
                // fused block as the row index, the rest as columns:
                // matrix shape (dl·dr) × (left_size·right_size).
                let fused = dl * dr;
                let rest = left_size * right_size;
                let mut mat = vec![0.0_f64; fused * rest];
                for ls in 0..left_size {
                    for f in 0..fused {
                        for rs in 0..right_size {
                            let w = work[(ls * fused + f) * right_size + rs];
                            mat[f * rest + (ls * right_size + rs)] = w;
                        }
                    }
                }
                let svd = svd_jacobi(&mat, fused, rest)?;
                // Keep only singular values above a relative tolerance, then cap at
                // χ_max. This drops the exact zeros of low-rank (e.g. product)
                // states so their bonds collapse to the true Schmidt rank.
                let r = effective_rank(&svd.s, chi_max).min(fused);
                let (iso, rem) = split_factors(&svd, r);
                layer_nodes.push(TtnNode {
                    d_left: dl,
                    d_right: dr,
                    d_parent: r,
                    data: iso,
                });
                // Re-fold the weighted remainder (r × rest) into a working tensor
                // with leg order (left…, parent, right…).
                let mut new_work = vec![0.0_f64; left_size * r * right_size];
                for p in 0..r {
                    for ls in 0..left_size {
                        for rs in 0..right_size {
                            new_work[(ls * r + p) * right_size + rs] =
                                rem[p * rest + (ls * right_size + rs)];
                        }
                    }
                }
                work = new_work;
                // Replace the fused pair (positions pos, pos+1) by one leg `r`.
                // After this the new parent leg sits at `pos`, so the next
                // original pair begins at `pos + 1` — exactly the next loop index.
                let mut updated = leg_dims[..pos].to_vec();
                updated.push(r);
                updated.extend_from_slice(&leg_dims[pos + 2..]);
                leg_dims = updated;
            }

            nodes.push(layer_nodes);
        }

        // After the last layer exactly one leg remains: that is the root weight.
        let top = work;

        Ok(Self {
            d,
            num_layers,
            nodes,
            top,
        })
    }

    /// Contract the entire tree back into a full state vector of length
    /// `d^(2^L)`.
    ///
    /// # Errors
    /// - propagates index errors via shape consistency (none expected for a TTN
    ///   built by [`Self::from_state_vector`]).
    pub fn to_state_vector(&self) -> TnResult<Vec<f64>> {
        // Start from the root weight as a single-leg tensor, then expand each layer
        // top-down: replace every parent leg by its node's two child legs.
        let mut leg_dims = vec![self.top.len()];
        let mut work = self.top.clone();

        for layer in (0..self.num_layers).rev() {
            let layer_nodes = &self.nodes[layer];

            // Expand parent legs left to right. After expanding a node's parent leg
            // (at the moving position `pos`) into its two child legs, the next
            // node's parent leg sits at `pos + 2`.
            let mut pos = 0usize;
            for node in layer_nodes {
                let left_size: usize = leg_dims[..pos].iter().product();
                let dp = leg_dims[pos];
                let right_size: usize = leg_dims[pos + 1..].iter().product();
                debug_assert_eq!(dp, node.d_parent);

                // work is (left_size, dp, right_size). Expand dp → (dl, dr) via the
                // isometry: new[ls, l, r, rs] = Σ_p work[ls, p, rs]·W[l, r, p].
                let dl = node.d_left;
                let dr = node.d_right;
                let mut expanded = vec![0.0_f64; left_size * dl * dr * right_size];
                for ls in 0..left_size {
                    for rs in 0..right_size {
                        for l in 0..dl {
                            for r_idx in 0..dr {
                                let mut acc = 0.0_f64;
                                for p in 0..dp {
                                    let w = work[(ls * dp + p) * right_size + rs];
                                    acc += w * node.get(l, r_idx, p);
                                }
                                let lr = l * dr + r_idx;
                                expanded[(ls * (dl * dr) + lr) * right_size + rs] = acc;
                            }
                        }
                    }
                }
                work = expanded;
                // Replace the parent leg by (dl, dr).
                let mut updated = leg_dims[..pos].to_vec();
                updated.push(dl);
                updated.push(dr);
                updated.extend_from_slice(&leg_dims[pos + 1..]);
                leg_dims = updated;
                pos += 2;
            }
        }

        Ok(work)
    }

    /// Squared Frobenius norm `⟨ψ|ψ⟩` of the represented state.
    ///
    /// Because every node is an isometry, the norm is carried entirely by the root
    /// weight: `‖ψ‖² = Σ top_p²`.
    pub fn norm_squared(&self) -> f64 {
        self.top.iter().map(|&v| v * v).sum()
    }

    /// The bond dimensions formed at each layer (parent bonds), leaves first.
    pub fn bond_dimensions(&self) -> Vec<Vec<usize>> {
        self.nodes
            .iter()
            .map(|layer| layer.iter().map(|node| node.d_parent).collect())
            .collect()
    }

    /// Verify that every node is a proper isometry: reshaping `(d_left·d_right,
    /// d_parent)` gives `W^† W = I` to within `tol`.
    pub fn check_isometries(&self, tol: f64) -> bool {
        for layer in &self.nodes {
            for node in layer {
                let rows = node.d_left * node.d_right;
                let cols = node.d_parent;
                for p in 0..cols {
                    for q in 0..cols {
                        let mut acc = 0.0_f64;
                        for i in 0..rows {
                            acc += node.data[i * cols + p] * node.data[i * cols + q];
                        }
                        let expected = if p == q { 1.0 } else { 0.0 };
                        if (acc - expected).abs() > tol {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// L2-normalise a vector.
    fn normalize(v: &mut [f64]) {
        let n: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n > 0.0 {
            for x in v.iter_mut() {
                *x /= n;
            }
        }
    }

    /// Deterministic pseudo-random state vector of length `len`.
    fn random_state(len: usize, seed: u64) -> Vec<f64> {
        let mut s = seed | 1;
        let mut v = vec![0.0_f64; len];
        for x in v.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *x = ((s >> 33) as f64) / (1u64 << 31) as f64 - 0.5;
        }
        normalize(&mut v);
        v
    }

    fn rel_err(a: &[f64], b: &[f64]) -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for (x, y) in a.iter().zip(b) {
            num += (x - y) * (x - y);
            den += y * y;
        }
        (num / den.max(1e-300)).sqrt()
    }

    #[test]
    fn ttn_rejects_bad_config() {
        assert!(matches!(
            TreeTensorNetwork::from_state_vector(&[1.0], 1, 1, 4),
            Err(TnError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            TreeTensorNetwork::from_state_vector(&[1.0], 2, 0, 4),
            Err(TnError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            TreeTensorNetwork::from_state_vector(&[1.0], 2, 1, 0),
            Err(TnError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn ttn_rejects_wrong_state_length() {
        // L=2 ⇒ 4 sites ⇒ d^4 = 16 for d=2.
        let bad = vec![0.0; 8];
        assert!(matches!(
            TreeTensorNetwork::from_state_vector(&bad, 2, 2, 4),
            Err(TnError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn ttn_n_sites_is_power_of_two() {
        let psi = random_state(16, 1);
        let ttn = TreeTensorNetwork::from_state_vector(&psi, 2, 2, 4).expect("build");
        assert_eq!(ttn.n_sites(), 4);
        assert_eq!(ttn.nodes.len(), 2);
    }

    #[test]
    fn ttn_reconstructs_two_layer_state() {
        // 4 qubits, full bond dimension ⇒ exact reconstruction.
        let psi = random_state(16, 7);
        let ttn = TreeTensorNetwork::from_state_vector(&psi, 2, 2, 16).expect("build");
        let recon = ttn.to_state_vector().expect("reconstruct");
        assert_eq!(recon.len(), psi.len());
        assert!(
            rel_err(&recon, &psi) < 1e-9,
            "rel err {}",
            rel_err(&recon, &psi)
        );
    }

    #[test]
    fn ttn_reconstructs_three_layer_state() {
        // 8 qubits (d^8 = 256), full bond dimension ⇒ exact.
        let psi = random_state(256, 11);
        let ttn = TreeTensorNetwork::from_state_vector(&psi, 2, 3, 256).expect("build");
        let recon = ttn.to_state_vector().expect("reconstruct");
        assert!(
            rel_err(&recon, &psi) < 1e-8,
            "rel err {}",
            rel_err(&recon, &psi)
        );
    }

    #[test]
    fn ttn_isometries_are_orthonormal() {
        let psi = random_state(16, 13);
        let ttn = TreeTensorNetwork::from_state_vector(&psi, 2, 2, 16).expect("build");
        assert!(ttn.check_isometries(1e-9), "all nodes must be isometries");
    }

    #[test]
    fn ttn_norm_preserved() {
        // The state is normalised, so ‖ψ‖² (carried by the root) must be ≈ 1.
        let psi = random_state(16, 17);
        let ttn = TreeTensorNetwork::from_state_vector(&psi, 2, 2, 16).expect("build");
        assert!(
            (ttn.norm_squared() - 1.0).abs() < 1e-9,
            "norm² = {}",
            ttn.norm_squared()
        );
    }

    #[test]
    fn ttn_product_state_has_trivial_bonds() {
        // A product state |0000⟩ should compress to bond dimension 1 everywhere.
        let mut psi = vec![0.0_f64; 16];
        psi[0] = 1.0;
        let ttn = TreeTensorNetwork::from_state_vector(&psi, 2, 2, 4).expect("build");
        for layer in ttn.bond_dimensions() {
            for chi in layer {
                assert_eq!(chi, 1, "product state bonds should all be 1");
            }
        }
        let recon = ttn.to_state_vector().expect("reconstruct");
        assert!(rel_err(&recon, &psi) < 1e-12);
    }

    #[test]
    fn ttn_truncation_is_lossy_but_bounded() {
        // Truncating an entangled state to χ=1 should still return a unit-norm-ish
        // approximation whose error is finite (not NaN/Inf).
        let psi = random_state(16, 19);
        let ttn = TreeTensorNetwork::from_state_vector(&psi, 2, 2, 1).expect("build");
        let recon = ttn.to_state_vector().expect("reconstruct");
        let e = rel_err(&recon, &psi);
        assert!(e.is_finite());
        assert!(
            e > 0.0,
            "χ=1 truncation of an entangled state should lose info"
        );
        // Bond dims should all be 1.
        for layer in ttn.bond_dimensions() {
            for chi in layer {
                assert_eq!(chi, 1);
            }
        }
    }

    #[test]
    fn ttn_bond_dims_capped_by_chi_max() {
        let psi = random_state(256, 23);
        let chi_max = 3;
        let ttn = TreeTensorNetwork::from_state_vector(&psi, 2, 3, chi_max).expect("build");
        for layer in ttn.bond_dimensions() {
            for chi in layer {
                assert!(chi <= chi_max, "bond {chi} exceeds χ_max {chi_max}");
            }
        }
    }

    #[test]
    fn ttn_d3_two_sites_single_layer() {
        // d=3 qutrits, L=1 ⇒ 2 sites, state length 9. Exact reconstruction.
        let psi = random_state(9, 29);
        let ttn = TreeTensorNetwork::from_state_vector(&psi, 3, 1, 9).expect("build");
        assert_eq!(ttn.n_sites(), 2);
        let recon = ttn.to_state_vector().expect("reconstruct");
        assert!(rel_err(&recon, &psi) < 1e-9);
    }
}
