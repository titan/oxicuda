//! Loopy belief propagation (sum-product) on a 4-connected 2-D grid CRF.
//!
//! Each grid node `i` carries a unary **log-potential** vector
//! `unary[i*K + l]` and every edge shares a pairwise log-potential matrix
//! `pairwise[a*K + b]`. The joint distribution is
//!
//! ```text
//! P(x) ∝ exp( Σ_i unary[i, x_i] + Σ_{(i,j)∈E} pairwise[x_i, x_j] )
//! ```
//!
//! so a *larger* unary entry favours that label and a diagonally dominant
//! pairwise matrix (`pairwise[a,a] > pairwise[a,b]`) is *attractive* (encourages
//! neighbouring nodes to share a label).
//!
//! Messages are passed in log-space (log-sum-exp) with damping until the
//! largest message change drops below `tol` or `max_iter` sweeps elapse. On a
//! loop-free grid (e.g. a `1×N` chain) the fixed point is **exact**; on a grid
//! with cycles it is the standard loopy-BP approximation.

use crate::error::{SeqError, SeqResult};
use crate::hmm::forward_backward::logsumexp;

/// Configuration for loopy belief propagation.
#[derive(Debug, Clone, Copy)]
pub struct LoopyBpConfig {
    /// Maximum number of synchronous message-update sweeps.
    pub max_iter: usize,
    /// Convergence threshold on the largest absolute log-message change.
    pub tol: f64,
    /// Update step in `(0, 1]`: `new = (1−damp)·old + damp·update`. A value of
    /// `1.0` is undamped (standard) BP; smaller values damp the updates.
    pub damping: f64,
}

impl Default for LoopyBpConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-9,
            damping: 0.5,
        }
    }
}

/// Inference output: per-node marginals plus convergence diagnostics.
#[derive(Debug, Clone)]
pub struct LoopyBpResult {
    /// Per-node marginals `[H*W*n_states]`, normalised per node.
    pub marginals: Vec<f64>,
    /// Number of sweeps actually performed.
    pub iterations: usize,
    /// Whether the message change fell below `tol` before `max_iter`.
    pub converged: bool,
}

/// Loopy belief-propagation engine for a fixed grid topology.
///
/// The grid geometry (edge list and per-node incident-message bookkeeping) is
/// built once in [`LoopyBp::new`] and reused across [`LoopyBp::infer`] calls.
#[derive(Debug, Clone)]
pub struct LoopyBp {
    height: usize,
    width: usize,
    n_states: usize,
    config: LoopyBpConfig,
    /// Undirected edges `(u, v)` with `u < v`.
    edges: Vec<(usize, usize)>,
    /// For each node, the `(incoming_slot, edge_index)` of every incident edge.
    incident: Vec<Vec<(usize, usize)>>,
}

impl LoopyBp {
    /// Build an inference engine for a `height × width` grid with `n_states`
    /// labels per node.
    ///
    /// # Errors
    /// * [`SeqError::InvalidConfiguration`] if any dimension is `0` or
    ///   `config.max_iter == 0`.
    /// * [`SeqError::InvalidParameter`] if `config.damping` is outside `(0, 1]`.
    pub fn new(
        height: usize,
        width: usize,
        n_states: usize,
        config: LoopyBpConfig,
    ) -> SeqResult<Self> {
        if height == 0 || width == 0 || n_states == 0 {
            return Err(SeqError::InvalidConfiguration(
                "height, width and n_states must all be > 0".to_string(),
            ));
        }
        if config.max_iter == 0 {
            return Err(SeqError::InvalidConfiguration(
                "max_iter must be > 0".to_string(),
            ));
        }
        if config.damping <= 0.0 || config.damping > 1.0 {
            return Err(SeqError::InvalidParameter {
                name: "damping".to_string(),
                value: config.damping,
            });
        }

        // 4-connected edge list, each stored with the smaller node first.
        let mut edges = Vec::new();
        for r in 0..height {
            for c in 0..width {
                let node = r * width + c;
                if c + 1 < width {
                    edges.push((node, node + 1)); // horizontal
                }
                if r + 1 < height {
                    edges.push((node, node + width)); // vertical
                }
            }
        }

        // Directed message slots: u→v at 2e, v→u at 2e+1. The message arriving
        // *at* a node from edge e is the one pointing toward it.
        let n_nodes = height * width;
        let mut incident: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_nodes];
        for (e, &(u, v)) in edges.iter().enumerate() {
            incident[u].push((2 * e + 1, e)); // v→u arrives at u
            incident[v].push((2 * e, e)); // u→v arrives at v
        }

        Ok(Self {
            height,
            width,
            n_states,
            config,
            edges,
            incident,
        })
    }

    /// Grid height.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Grid width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Number of labels per node.
    pub fn n_states(&self) -> usize {
        self.n_states
    }

    /// Run sum-product BP and return the per-node marginals `[H*W*n_states]`,
    /// normalised so each node's distribution sums to 1.
    ///
    /// # Errors
    /// [`SeqError::ShapeMismatch`] if `unary` or `pairwise` has the wrong length.
    pub fn infer(&self, unary: &[f64], pairwise: &[f64]) -> SeqResult<Vec<f64>> {
        Ok(self.infer_detailed(unary, pairwise)?.marginals)
    }

    /// Run sum-product BP, returning marginals together with the sweep count and
    /// whether convergence was reached. See [`LoopyBp::infer`] for the error
    /// conditions.
    pub fn infer_detailed(&self, unary: &[f64], pairwise: &[f64]) -> SeqResult<LoopyBpResult> {
        let k = self.n_states;
        let n_nodes = self.height * self.width;
        if unary.len() != n_nodes * k {
            return Err(SeqError::ShapeMismatch {
                expected: n_nodes * k,
                got: unary.len(),
            });
        }
        if pairwise.len() != k * k {
            return Err(SeqError::ShapeMismatch {
                expected: k * k,
                got: pairwise.len(),
            });
        }

        let damp = self.config.damping;
        let n_slots = self.edges.len() * 2;
        let mut log_msg = vec![0.0f64; n_slots * k];
        let mut new_log_msg = log_msg.clone();
        let mut terms = vec![0.0f64; k];
        let mut out = vec![0.0f64; k];
        let mut converged = false;
        let mut iterations = 0usize;

        for it in 0..self.config.max_iter {
            iterations = it + 1;
            for (e, &(u, v)) in self.edges.iter().enumerate() {
                // Both directed messages along this edge.
                for &(src, dst, out_slot) in &[(u, v, 2 * e), (v, u, 2 * e + 1)] {
                    let _ = dst;
                    for l_dst in 0..k {
                        for l_src in 0..k {
                            // Oriented pairwise: edge is stored as (u, v).
                            let psi = if src == u {
                                pairwise[l_src * k + l_dst]
                            } else {
                                pairwise[l_dst * k + l_src]
                            };
                            let mut acc = unary[src * k + l_src] + psi;
                            // Product of incoming messages from every neighbour
                            // of `src` except the one on this edge.
                            for &(in_slot, in_edge) in &self.incident[src] {
                                if in_edge == e {
                                    continue;
                                }
                                acc += log_msg[in_slot * k + l_src];
                            }
                            terms[l_src] = acc;
                        }
                        out[l_dst] = logsumexp(&terms);
                    }
                    // Normalise in the log domain for numerical stability.
                    let m = out.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    if m > f64::NEG_INFINITY {
                        for val in out.iter_mut() {
                            *val -= m;
                        }
                    }
                    for l in 0..k {
                        let base = out_slot * k + l;
                        new_log_msg[base] = (1.0 - damp) * log_msg[base] + damp * out[l];
                    }
                }
            }

            let mut max_diff = 0.0f64;
            for idx in 0..log_msg.len() {
                let d = (new_log_msg[idx] - log_msg[idx]).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
            log_msg.copy_from_slice(&new_log_msg);
            if max_diff < self.config.tol {
                converged = true;
                break;
            }
        }

        let marginals = self.node_marginals(unary, &log_msg);
        Ok(LoopyBpResult {
            marginals,
            iterations,
            converged,
        })
    }

    /// Combine the unary potential of each node with its incoming messages and
    /// normalise to a proper distribution.
    fn node_marginals(&self, unary: &[f64], log_msg: &[f64]) -> Vec<f64> {
        let k = self.n_states;
        let n_nodes = self.height * self.width;
        let mut marginals = vec![0.0f64; n_nodes * k];
        let mut log_b = vec![0.0f64; k];
        for i in 0..n_nodes {
            for l in 0..k {
                log_b[l] = unary[i * k + l];
            }
            for &(in_slot, _e) in &self.incident[i] {
                for l in 0..k {
                    log_b[l] += log_msg[in_slot * k + l];
                }
            }
            let m = log_b.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut s = 0.0;
            for l in 0..k {
                let val = (log_b[l] - m).exp();
                marginals[i * k + l] = val;
                s += val;
            }
            for l in 0..k {
                marginals[i * k + l] = if s > 0.0 {
                    marginals[i * k + l] / s
                } else {
                    1.0 / k as f64
                };
            }
        }
        marginals
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact chain marginals by brute-force enumeration of all `k^n` labellings.
    fn brute_force_chain_marginals(
        unary: &[f64],
        pairwise: &[f64],
        n: usize,
        k: usize,
    ) -> Vec<f64> {
        let mut marg = vec![0.0f64; n * k];
        let mut z = 0.0f64;
        let total = k.pow(n as u32);
        let mut labels = vec![0usize; n];
        for code in 0..total {
            let mut x = code;
            for t in 0..n {
                labels[t] = x % k;
                x /= k;
            }
            let mut logp = 0.0f64;
            for t in 0..n {
                logp += unary[t * k + labels[t]];
            }
            for t in 0..n - 1 {
                logp += pairwise[labels[t] * k + labels[t + 1]];
            }
            let p = logp.exp();
            z += p;
            for t in 0..n {
                marg[t * k + labels[t]] += p;
            }
        }
        for v in marg.iter_mut() {
            *v /= z;
        }
        marg
    }

    #[test]
    fn chain_matches_exact_marginals() {
        let n = 4;
        let k = 2;
        let unary = vec![
            0.3, -0.1, // node 0
            -0.4, 0.2, // node 1
            0.5, 0.0, // node 2
            -0.2, 0.6, // node 3
        ];
        // Asymmetric on purpose to exercise oriented pairwise handling.
        let pairwise = vec![0.7, -0.2, -0.3, 0.5];
        let bp = LoopyBp::new(
            1,
            n,
            k,
            LoopyBpConfig {
                max_iter: 500,
                tol: 1e-12,
                damping: 1.0,
            },
        )
        .expect("new");
        let got = bp.infer(&unary, &pairwise).expect("infer");
        let exact = brute_force_chain_marginals(&unary, &pairwise, n, k);
        for idx in 0..n * k {
            assert!(
                (got[idx] - exact[idx]).abs() < 1e-6,
                "idx {idx}: bp {} vs exact {}",
                got[idx],
                exact[idx]
            );
        }
    }

    #[test]
    fn uniform_potentials_give_uniform_marginals() {
        let (h, w, k) = (2, 3, 3);
        let bp = LoopyBp::new(h, w, k, LoopyBpConfig::default()).expect("new");
        let unary = vec![0.0f64; h * w * k];
        let pairwise = vec![0.0f64; k * k];
        let marg = bp.infer(&unary, &pairwise).expect("infer");
        for &m in &marg {
            assert!((m - 1.0 / k as f64).abs() < 1e-9, "got {m}");
        }
    }

    #[test]
    fn strong_unary_propagates_through_attractive_pairwise() {
        let (h, w, k) = (3, 3, 2);
        let bp = LoopyBp::new(h, w, k, LoopyBpConfig::default()).expect("new");
        let mut unary = vec![0.0f64; h * w * k];
        let center = w + 1; // node (1,1)
        unary[center * k] = 4.0; // strongly favours label 0
        // Attractive (Potts) pairwise: same label rewarded.
        let beta = 0.8;
        let pairwise = vec![beta, 0.0, 0.0, beta];
        let marg = bp.infer(&unary, &pairwise).expect("infer");
        // Centre is pinned to label 0.
        assert!(marg[center * k] > 0.9, "centre p0 = {}", marg[center * k]);
        // A direct neighbour is pulled toward label 0 (above the 0.5 prior).
        let nbr = w + 1; // (0,1), directly above the centre
        assert!(marg[nbr * k] > 0.5, "neighbour p0 = {}", marg[nbr * k]);
        // The neighbour is more affected than a far corner.
        let corner = 2 * w; // (2,0)
        assert!(
            marg[nbr * k] >= marg[corner * k] - 1e-9,
            "neighbour {} vs corner {}",
            marg[nbr * k],
            marg[corner * k]
        );
    }

    #[test]
    fn marginals_normalised_and_bounded() {
        let (h, w, k) = (2, 2, 3);
        let bp = LoopyBp::new(h, w, k, LoopyBpConfig::default()).expect("new");
        let unary = vec![
            0.2, -0.3, 0.1, //
            -0.5, 0.4, 0.0, //
            0.3, 0.3, -0.2, //
            0.0, -0.1, 0.5, //
        ];
        let pairwise = vec![0.5, 0.1, 0.0, 0.1, 0.5, 0.1, 0.0, 0.1, 0.5];
        let marg = bp.infer(&unary, &pairwise).expect("infer");
        for i in 0..h * w {
            let mut s = 0.0;
            for l in 0..k {
                let v = marg[i * k + l];
                assert!((0.0..=1.0).contains(&v), "marginal out of range: {v}");
                s += v;
            }
            assert!((s - 1.0).abs() < 1e-9, "node {i} sum {s}");
        }
    }

    #[test]
    fn converges_on_small_grid() {
        let (h, w, k) = (3, 3, 2);
        let bp = LoopyBp::new(h, w, k, LoopyBpConfig::default()).expect("new");
        let mut unary = vec![0.0f64; h * w * k];
        for i in 0..h * w {
            unary[i * k] = 0.1 * (i as f64).cos();
            unary[i * k + 1] = -0.1 * (i as f64).sin();
        }
        let pairwise = vec![0.3, 0.0, 0.0, 0.3]; // weak attractive
        let res = bp.infer_detailed(&unary, &pairwise).expect("infer");
        assert!(
            res.converged,
            "did not converge in {} sweeps",
            res.iterations
        );
        for i in 0..h * w {
            let s: f64 = res.marginals[i * k..(i + 1) * k].iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "node {i} sum {s}");
        }
    }

    #[test]
    fn invalid_dims_and_params_error() {
        assert!(LoopyBp::new(0, 3, 2, LoopyBpConfig::default()).is_err());
        assert!(LoopyBp::new(3, 3, 0, LoopyBpConfig::default()).is_err());
        assert!(
            LoopyBp::new(
                2,
                2,
                2,
                LoopyBpConfig {
                    damping: 1.5,
                    ..LoopyBpConfig::default()
                }
            )
            .is_err()
        );
        let bp = LoopyBp::new(2, 2, 2, LoopyBpConfig::default()).expect("new");
        // Wrong unary length (needs 2*2*2 = 8).
        assert!(bp.infer(&[0.0; 3], &[0.0; 4]).is_err());
        // Wrong pairwise length (needs 2*2 = 4).
        assert!(bp.infer(&[0.0; 8], &[0.0; 3]).is_err());
    }
}
