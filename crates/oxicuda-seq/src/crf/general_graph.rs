//! General-graph CRF training via loopy belief propagation.
//!
//! Supports arbitrary pairwise factor graphs (not just linear chains).
//! Inference uses sum-product message passing; MAP uses max-product.
//!
//! Reference: Koller & Friedman 2009, "Probabilistic Graphical Models",
//! Chapter 11 (Belief Propagation).
//!
//! Graph structure:
//!   - `n_nodes` variable nodes, each with `n_labels` possible states.
//!   - `edges: Vec<(usize, usize)>` — undirected pairwise edges.
//!   - Node potentials φ_i(y_i): `[n_nodes × n_labels]`.
//!   - Edge potentials ψ_{ij}(y_i, y_j): `[n_edges × n_labels × n_labels]`.
//!
//! Loopy BP is approximate for graphs with cycles.
//! Guaranteed exact for trees and linear chains.

use crate::error::{SeqError, SeqResult};

// ─── logsumexp helper ────────────────────────────────────────────────────────

#[inline]
fn logsumexp(xs: &[f64]) -> f64 {
    let m = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let s: f64 = xs.iter().map(|&x| (x - m).exp()).sum();
    m + s.ln()
}

// ─── Graph CRF configuration ─────────────────────────────────────────────────

/// Configuration for the general-graph CRF.
#[derive(Debug, Clone)]
pub struct GraphCrfConfig {
    /// Number of variable nodes in the graph.
    pub n_nodes: usize,
    /// Number of labels per node.
    pub n_labels: usize,
    /// Maximum BP iterations.
    pub max_iter: usize,
    /// Convergence tolerance (max absolute message change).
    pub tol: f64,
    /// Damping factor ∈ [0,1): new_msg = (1-damp)*new + damp*old.
    pub damping: f64,
}

/// One undirected pairwise edge in the factor graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edge {
    pub i: usize,
    pub j: usize,
}

// ─── GeneralGraphCrf ──────────────────────────────────────────────────────────

/// General-graph pairwise CRF with loopy belief propagation inference.
#[derive(Debug, Clone)]
pub struct GeneralGraphCrf {
    config: GraphCrfConfig,
    /// Node unary log-potentials `[n_nodes × n_labels]`.
    pub node_potentials: Vec<f64>,
    /// Edge pairwise log-potentials `[n_edges × n_labels × n_labels]`.
    pub edge_potentials: Vec<f64>,
    /// Edge list.
    pub edges: Vec<Edge>,
}

impl GeneralGraphCrf {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Create a new CRF with zero potentials and the given edges.
    ///
    /// Validates that node indices in `edges` are within `[0, n_nodes)`.
    pub fn new(config: GraphCrfConfig, edges: Vec<Edge>) -> SeqResult<Self> {
        if config.n_nodes == 0 {
            return Err(SeqError::InvalidConfiguration("n_nodes must be > 0".into()));
        }
        if config.n_labels == 0 {
            return Err(SeqError::InvalidConfiguration(
                "n_labels must be > 0".into(),
            ));
        }
        for &Edge { i, j } in &edges {
            if i >= config.n_nodes {
                return Err(SeqError::IndexOutOfBounds {
                    index: i,
                    len: config.n_nodes,
                });
            }
            if j >= config.n_nodes {
                return Err(SeqError::IndexOutOfBounds {
                    index: j,
                    len: config.n_nodes,
                });
            }
        }
        let n_nodes = config.n_nodes;
        let n_labels = config.n_labels;
        let n_edges = edges.len();
        Ok(Self {
            node_potentials: vec![0.0f64; n_nodes * n_labels],
            edge_potentials: vec![0.0f64; n_edges * n_labels * n_labels],
            edges,
            config,
        })
    }

    // ── Potential accessors ───────────────────────────────────────────────────

    /// Set the unary log-potential for node `node`, label `lbl`.
    pub fn set_node_potential(&mut self, node: usize, lbl: usize, val: f64) -> SeqResult<()> {
        let n = self.config.n_labels;
        if node >= self.config.n_nodes {
            return Err(SeqError::IndexOutOfBounds {
                index: node,
                len: self.config.n_nodes,
            });
        }
        if lbl >= n {
            return Err(SeqError::IndexOutOfBounds { index: lbl, len: n });
        }
        self.node_potentials[node * n + lbl] = val;
        Ok(())
    }

    /// Set the pairwise log-potential for edge `e_idx`, labels `(li, lj)`.
    pub fn set_edge_potential(
        &mut self,
        e_idx: usize,
        li: usize,
        lj: usize,
        val: f64,
    ) -> SeqResult<()> {
        let n = self.config.n_labels;
        if e_idx >= self.edges.len() {
            return Err(SeqError::IndexOutOfBounds {
                index: e_idx,
                len: self.edges.len(),
            });
        }
        if li >= n || lj >= n {
            return Err(SeqError::IndexOutOfBounds {
                index: li.max(lj),
                len: n,
            });
        }
        self.edge_potentials[e_idx * n * n + li * n + lj] = val;
        Ok(())
    }

    // ── Sum-product BP ────────────────────────────────────────────────────────

    /// Run sum-product loopy belief propagation.
    ///
    /// Returns marginal log-beliefs `[n_nodes × n_labels]` (normalised so that
    /// `Σ_label exp(beliefs[node * n_labels + label]) = 1`).
    pub fn sum_product_marginals(&self) -> SeqResult<Vec<f64>> {
        let n = self.config.n_labels;
        let n_nodes = self.config.n_nodes;
        let n_edges = self.edges.len();

        // Messages: for each directed edge (i→j) and (j→i), a log-message of
        // length n_labels.  Index: msg[e * 2 + dir][label] where dir=0 → i→j, dir=1 → j→i.
        let mut msgs = vec![vec![0.0f64; n]; n_edges * 2];

        let mut tmp = vec![0.0f64; n];

        for _iter in 0..self.config.max_iter {
            let mut max_delta = 0.0f64;

            for e_idx in 0..n_edges {
                let Edge { i, j } = self.edges[e_idx];
                let ep_base = e_idx * n * n;

                // ── Update msg i→j ────────────────────────────────────────────
                // m_{i→j}(y_j) = log Σ_{y_i} [φ_i(y_i) * ψ_{ij}(y_i, y_j) *
                //                Π_{k≠j} m_{k→i}(y_i)]
                let new_i2j: Vec<f64> = (0..n)
                    .map(|yj| {
                        for yi in 0..n {
                            // Aggregate incoming messages to i (exclude j→i = msgs[e*2+1])
                            let mut incoming_i = self.node_potentials[i * n + yi];
                            for (e2, &Edge { i: ei2, j: ej2 }) in self.edges.iter().enumerate() {
                                if e2 == e_idx {
                                    continue;
                                }
                                if ei2 == i {
                                    // msg j2→i is in e2*2+1
                                    incoming_i += msgs[e2 * 2 + 1][yi];
                                } else if ej2 == i {
                                    // msg i2→j2 direction, msg from e2's i side to i
                                    incoming_i += msgs[e2 * 2][yi];
                                }
                            }
                            tmp[yi] = incoming_i + self.edge_potentials[ep_base + yi * n + yj];
                        }
                        logsumexp(&tmp)
                    })
                    .collect();

                // Normalise
                let lse = logsumexp(&new_i2j);
                let new_i2j: Vec<f64> = new_i2j.iter().map(|&v| v - lse).collect();

                // ── Update msg j→i ────────────────────────────────────────────
                let new_j2i: Vec<f64> = (0..n)
                    .map(|yi| {
                        for yj in 0..n {
                            let mut incoming_j = self.node_potentials[j * n + yj];
                            for (e2, &Edge { i: ei2, j: ej2 }) in self.edges.iter().enumerate() {
                                if e2 == e_idx {
                                    continue;
                                }
                                if ei2 == j {
                                    incoming_j += msgs[e2 * 2 + 1][yj];
                                } else if ej2 == j {
                                    incoming_j += msgs[e2 * 2][yj];
                                }
                            }
                            // ψ_{ij}(yi, yj) is symmetric in undirected graph;
                            // use ψ(yj, yi) for j→i direction
                            tmp[yj] = incoming_j + self.edge_potentials[ep_base + yj * n + yi];
                        }
                        logsumexp(&tmp)
                    })
                    .collect();

                let lse2 = logsumexp(&new_j2i);
                let new_j2i: Vec<f64> = new_j2i.iter().map(|&v| v - lse2).collect();

                // ── Damping + convergence check ────────────────────────────────
                let damp = self.config.damping;
                for l in 0..n {
                    let old_i2j = msgs[e_idx * 2][l];
                    let old_j2i = msgs[e_idx * 2 + 1][l];
                    let updated_i2j = (1.0 - damp) * new_i2j[l] + damp * old_i2j;
                    let updated_j2i = (1.0 - damp) * new_j2i[l] + damp * old_j2i;
                    max_delta = max_delta
                        .max((updated_i2j - old_i2j).abs())
                        .max((updated_j2i - old_j2i).abs());
                    msgs[e_idx * 2][l] = updated_i2j;
                    msgs[e_idx * 2 + 1][l] = updated_j2i;
                }
            }

            if max_delta < self.config.tol {
                break;
            }
        }

        // ── Compute beliefs ───────────────────────────────────────────────────
        let mut beliefs = vec![0.0f64; n_nodes * n];
        for node in 0..n_nodes {
            for l in 0..n {
                let mut b = self.node_potentials[node * n + l];
                for (e_idx, &Edge { i, j }) in self.edges.iter().enumerate() {
                    if i == node {
                        // Incoming message from j (stored as j→i = e_idx*2+1)
                        b += msgs[e_idx * 2 + 1][l];
                    } else if j == node {
                        // Incoming message from i (stored as i→j = e_idx*2)
                        b += msgs[e_idx * 2][l];
                    }
                }
                beliefs[node * n + l] = b;
            }
            // Normalise
            let lse = logsumexp(&beliefs[node * n..(node + 1) * n]);
            for l in 0..n {
                beliefs[node * n + l] -= lse;
            }
        }

        Ok(beliefs)
    }

    // ── Max-product MAP ───────────────────────────────────────────────────────

    /// Run max-product loopy BP to compute approximate MAP assignments.
    ///
    /// Returns `[n_nodes]` with the argmax label for each node.
    pub fn map_decode(&self) -> SeqResult<Vec<usize>> {
        let n = self.config.n_labels;
        let n_nodes = self.config.n_nodes;
        let n_edges = self.edges.len();

        let mut msgs = vec![vec![0.0f64; n]; n_edges * 2];
        let mut tmp = vec![0.0f64; n];

        for _iter in 0..self.config.max_iter {
            let mut max_delta = 0.0f64;

            for e_idx in 0..n_edges {
                let Edge { i, j } = self.edges[e_idx];
                let ep_base = e_idx * n * n;

                let new_i2j: Vec<f64> = (0..n)
                    .map(|yj| {
                        for yi in 0..n {
                            let mut incoming_i = self.node_potentials[i * n + yi];
                            for (e2, &Edge { i: ei2, j: ej2 }) in self.edges.iter().enumerate() {
                                if e2 == e_idx {
                                    continue;
                                }
                                if ei2 == i {
                                    incoming_i += msgs[e2 * 2 + 1][yi];
                                } else if ej2 == i {
                                    incoming_i += msgs[e2 * 2][yi];
                                }
                            }
                            tmp[yi] = incoming_i + self.edge_potentials[ep_base + yi * n + yj];
                        }
                        tmp.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                    })
                    .collect();

                let new_j2i: Vec<f64> = (0..n)
                    .map(|yi| {
                        for yj in 0..n {
                            let mut incoming_j = self.node_potentials[j * n + yj];
                            for (e2, &Edge { i: ei2, j: ej2 }) in self.edges.iter().enumerate() {
                                if e2 == e_idx {
                                    continue;
                                }
                                if ei2 == j {
                                    incoming_j += msgs[e2 * 2 + 1][yj];
                                } else if ej2 == j {
                                    incoming_j += msgs[e2 * 2][yj];
                                }
                            }
                            tmp[yj] = incoming_j + self.edge_potentials[ep_base + yj * n + yi];
                        }
                        tmp.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                    })
                    .collect();

                let damp = self.config.damping;
                for l in 0..n {
                    let old_i2j = msgs[e_idx * 2][l];
                    let old_j2i = msgs[e_idx * 2 + 1][l];
                    let updated_i2j = (1.0 - damp) * new_i2j[l] + damp * old_i2j;
                    let updated_j2i = (1.0 - damp) * new_j2i[l] + damp * old_j2i;
                    max_delta = max_delta
                        .max((updated_i2j - old_i2j).abs())
                        .max((updated_j2i - old_j2i).abs());
                    msgs[e_idx * 2][l] = updated_i2j;
                    msgs[e_idx * 2 + 1][l] = updated_j2i;
                }
            }

            if max_delta < self.config.tol {
                break;
            }
        }

        // MAP: argmax over beliefs
        let mut assignments = vec![0usize; n_nodes];
        for node in 0..n_nodes {
            let mut best_label = 0;
            let mut best_b = f64::NEG_INFINITY;
            let mut b_acc = self.node_potentials[node * n..node * n + n].to_vec();
            for (e_idx, &Edge { i, j }) in self.edges.iter().enumerate() {
                for l in 0..n {
                    if i == node {
                        b_acc[l] += msgs[e_idx * 2 + 1][l];
                    } else if j == node {
                        b_acc[l] += msgs[e_idx * 2][l];
                    }
                }
            }
            for l in 0..n {
                if b_acc[l] > best_b {
                    best_b = b_acc[l];
                    best_label = l;
                }
            }
            assignments[node] = best_label;
        }
        Ok(assignments)
    }

    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.config.n_nodes
    }

    /// Number of edges.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Number of labels.
    pub fn n_labels(&self) -> usize {
        self.config.n_labels
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config(n_nodes: usize, n_labels: usize) -> GraphCrfConfig {
        GraphCrfConfig {
            n_nodes,
            n_labels,
            max_iter: 50,
            tol: 1e-8,
            damping: 0.5,
        }
    }

    fn chain_edges(n: usize) -> Vec<Edge> {
        (0..n - 1).map(|i| Edge { i, j: i + 1 }).collect()
    }

    #[test]
    fn construction_succeeds() {
        let edges = chain_edges(4);
        let crf = GeneralGraphCrf::new(default_config(4, 3), edges);
        assert!(crf.is_ok());
    }

    #[test]
    fn n_nodes_zero_error() {
        let result = GeneralGraphCrf::new(default_config(0, 3), vec![]);
        assert!(result.is_err(), "n_nodes=0 should return Err");
    }

    #[test]
    fn n_labels_zero_error() {
        let result = GeneralGraphCrf::new(default_config(3, 0), vec![]);
        assert!(result.is_err(), "n_labels=0 should return Err");
    }

    #[test]
    fn invalid_edge_node_index_error() {
        let edges = vec![Edge { i: 0, j: 10 }]; // node 10 out of range for 3-node graph
        let result = GeneralGraphCrf::new(default_config(3, 2), edges);
        assert!(
            result.is_err(),
            "edge with out-of-range node should return Err"
        );
    }

    #[test]
    fn marginals_shape() {
        let edges = chain_edges(4);
        let crf = GeneralGraphCrf::new(default_config(4, 3), edges).expect("new");
        let beliefs = crf.sum_product_marginals().expect("marginals");
        assert_eq!(beliefs.len(), 4 * 3);
    }

    #[test]
    fn marginals_normalised() {
        let edges = chain_edges(3);
        let crf = GeneralGraphCrf::new(default_config(3, 2), edges).expect("new");
        let beliefs = crf.sum_product_marginals().expect("marginals");
        for node in 0..3 {
            let sum: f64 = beliefs[node * 2..(node + 1) * 2]
                .iter()
                .map(|&b| b.exp())
                .sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "node {node} marginals sum={sum} should be 1.0"
            );
        }
    }

    #[test]
    fn map_decode_shape() {
        let edges = chain_edges(5);
        let crf = GeneralGraphCrf::new(default_config(5, 4), edges).expect("new");
        let map = crf.map_decode().expect("map_decode");
        assert_eq!(map.len(), 5);
    }

    #[test]
    fn map_decode_valid_labels() {
        let edges = chain_edges(4);
        let crf = GeneralGraphCrf::new(default_config(4, 3), edges).expect("new");
        let map = crf.map_decode().expect("map_decode");
        for &l in &map {
            assert!(l < 3, "map label {l} >= n_labels=3");
        }
    }

    #[test]
    fn strong_node_potential_drives_assignment() {
        // If node 0 has a very strong preference for label 1, MAP should pick 1.
        let mut crf =
            GeneralGraphCrf::new(default_config(2, 2), vec![Edge { i: 0, j: 1 }]).expect("new");
        crf.set_node_potential(0, 0, -10.0).expect("set");
        crf.set_node_potential(0, 1, 10.0).expect("set");
        let map = crf.map_decode().expect("map_decode");
        assert_eq!(map[0], 1, "node 0 should be assigned label 1");
    }

    #[test]
    fn set_potential_out_of_range_error() {
        let mut crf = GeneralGraphCrf::new(default_config(3, 2), vec![]).expect("new");
        let result = crf.set_node_potential(5, 0, 1.0); // node 5 out of range
        assert!(result.is_err());
    }

    #[test]
    fn single_node_marginals() {
        // No edges: marginals from node_potentials only.
        let mut crf = GeneralGraphCrf::new(default_config(1, 3), vec![]).expect("new");
        crf.set_node_potential(0, 0, 0.0).expect("set");
        crf.set_node_potential(0, 1, 1.0).expect("set");
        crf.set_node_potential(0, 2, 2.0).expect("set");
        let beliefs = crf.sum_product_marginals().expect("marginals");
        // Label 2 should have highest belief
        assert!(
            beliefs[2] > beliefs[1],
            "label 2 should have highest marginal"
        );
        assert!(
            beliefs[1] > beliefs[0],
            "label 1 should have higher marginal than 0"
        );
    }

    #[test]
    fn cycle_graph_no_panic() {
        // Triangle: 3 nodes, 3 edges (cycle) — loopy BP, just check it runs.
        let edges = vec![
            Edge { i: 0, j: 1 },
            Edge { i: 1, j: 2 },
            Edge { i: 2, j: 0 },
        ];
        let crf = GeneralGraphCrf::new(default_config(3, 2), edges).expect("new");
        let beliefs = crf.sum_product_marginals().expect("cycle marginals");
        assert_eq!(beliefs.len(), 3 * 2);
        for &b in &beliefs {
            assert!(b.is_finite(), "cycle belief should be finite, got {b}");
        }
    }
}
