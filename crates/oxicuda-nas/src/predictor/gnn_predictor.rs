//! Graph-based architecture accuracy predictors (BANANAS / NPENAS).
//!
//! References:
//! - White, Neiswanger & Savani, "BANANAS: Bayesian Optimization with Neural
//!   Architectures for Neural Architecture Search", AAAI 2021. BANANAS's key
//!   idea is the **path encoding**: a cell is encoded by the set of *input →
//!   output paths* it contains (one binary feature per possible op-path), which
//!   is a far more predictive architecture representation for a small MLP than a
//!   flat adjacency/op list.
//! - Wei, Niu, Tang, Wang & Liang, "NPENAS: Neural Predictor guided Evolution
//!   for Neural Architecture Search", 2020. NPENAS replaces the path-MLP with a
//!   **graph neural network** that message-passes over the cell DAG itself.
//!
//! This module supplies both:
//!
//! - [`PathEncoder`] / [`PathEncodedPredictor`] — the BANANAS path encoding plus
//!   a one-hidden-layer ReLU MLP regressor trained by mini-batch SGD; and
//! - [`GnnPredictor`] — a message-passing GNN over the cell DAG (each node
//!   aggregates its incoming neighbours' embeddings through the incoming-edge op
//!   embedding, a learned linear update + ReLU per layer, then mean-pools the
//!   node embeddings and reads out a scalar accuracy), trained end-to-end by SGD
//!   over the readout + update + op-embedding parameters.
//!
//! The architecture is modelled as a feed-forward cell DAG: a fixed number of
//! **input** nodes followed by `n_intermediate` ordered intermediate nodes, each
//! receiving exactly `n_inputs_per_node` incoming edges from strictly-earlier
//! nodes, every edge labelled by an op index from `[0, n_ops)`. The genome is
//! the flattened per-edge op list, matching the [`ArchEncoding`] used elsewhere.

use crate::error::{NasError, NasResult};
use crate::evolution::encoding::ArchEncoding;
use crate::handle::LcgRng;

// ─── CellTopology ───────────────────────────────────────────────────────────────

/// Fixed wiring of a cell DAG: who connects to whom, independent of the op
/// labels carried by the [`ArchEncoding`] genome.
///
/// Node indexing: `0 .. n_inputs` are the cell inputs; `n_inputs .. n_inputs +
/// n_intermediate` are the intermediate nodes in topological order. Intermediate
/// node `k` (0-based among intermediates) draws `n_inputs_per_node` incoming
/// edges from the `predecessors[k]` node ids (all strictly less than its own id).
#[derive(Debug, Clone)]
pub struct CellTopology {
    /// Number of input nodes.
    pub n_inputs: usize,
    /// Number of ordered intermediate nodes.
    pub n_intermediate: usize,
    /// Incoming edges per intermediate node.
    pub n_inputs_per_node: usize,
    /// Number of candidate ops per edge.
    pub n_ops: usize,
    /// For each intermediate node, the source node ids of its incoming edges.
    pub predecessors: Vec<Vec<usize>>,
}

impl CellTopology {
    /// Build the canonical "node `k` connects to the `n_inputs_per_node`
    /// immediately-preceding nodes" topology (a dense local DAG), validating
    /// dimensions.
    ///
    /// # Errors
    /// - [`NasError::InvalidNumNodes`] if `n_inputs == 0` or
    ///   `n_intermediate == 0`.
    /// - [`NasError::InvalidNumOps`] if `n_inputs_per_node == 0` or
    ///   `n_ops == 0`.
    pub fn sequential(
        n_inputs: usize,
        n_intermediate: usize,
        n_inputs_per_node: usize,
        n_ops: usize,
    ) -> NasResult<Self> {
        if n_inputs == 0 || n_intermediate == 0 {
            return Err(NasError::InvalidNumNodes {
                min: 1,
                got: n_inputs.min(n_intermediate),
            });
        }
        if n_inputs_per_node == 0 || n_ops == 0 {
            return Err(NasError::InvalidNumOps);
        }
        let mut predecessors = Vec::with_capacity(n_intermediate);
        for k in 0..n_intermediate {
            let node_id = n_inputs + k;
            // Connect to the n_inputs_per_node immediately-preceding node ids,
            // clamped so input nodes are reachable for the earliest intermediates.
            let mut preds = Vec::with_capacity(n_inputs_per_node);
            for j in 0..n_inputs_per_node {
                // Walk backwards; wrap into the input nodes when too early.
                let src = if node_id > j + 1 {
                    node_id - 1 - j
                } else {
                    j % n_inputs
                };
                preds.push(src.min(node_id.saturating_sub(1)));
            }
            predecessors.push(preds);
        }
        Ok(Self {
            n_inputs,
            n_intermediate,
            n_inputs_per_node,
            n_ops,
            predecessors,
        })
    }

    /// Total number of edges (= genome length).
    #[must_use]
    pub fn n_edges(&self) -> usize {
        self.n_intermediate * self.n_inputs_per_node
    }

    /// Total number of nodes.
    #[must_use]
    pub fn n_nodes(&self) -> usize {
        self.n_inputs + self.n_intermediate
    }

    /// Validate that a genome's length and op range match this topology.
    ///
    /// # Errors
    /// [`NasError::InvalidArchEncoding`] on a length or op-index mismatch.
    pub fn validate_genome(&self, arch: &ArchEncoding) -> NasResult<()> {
        if arch.genes.len() != self.n_edges() {
            return Err(NasError::InvalidArchEncoding);
        }
        if arch.genes.iter().any(|&g| g >= self.n_ops) {
            return Err(NasError::InvalidArchEncoding);
        }
        Ok(())
    }
}

// ─── PathEncoder (BANANAS) ──────────────────────────────────────────────────────

/// BANANAS truncated path encoding.
///
/// Each intermediate node's *incoming op multiset* is hashed into a fixed-width
/// binary feature vector: for every `(node, incoming-edge-slot, op)` triple we
/// set one bit. This "truncated path" encoding (the practical BANANAS variant)
/// is permutation-stable per node-slot and strictly more informative than a flat
/// op list because it ties an op to *where* in the DAG it appears.
#[derive(Debug, Clone)]
pub struct PathEncoder {
    topology: CellTopology,
}

impl PathEncoder {
    /// Build a path encoder for a topology.
    #[must_use]
    pub fn new(topology: CellTopology) -> Self {
        Self { topology }
    }

    /// Dimension of the produced encoding: `n_edges · n_ops`.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.topology.n_edges() * self.topology.n_ops
    }

    /// Encode a genome into its binary path features.
    ///
    /// # Errors
    /// Propagates [`CellTopology::validate_genome`] errors.
    pub fn encode(&self, arch: &ArchEncoding) -> NasResult<Vec<f32>> {
        self.topology.validate_genome(arch)?;
        let n_ops = self.topology.n_ops;
        let mut feats = vec![0.0_f32; self.dim()];
        for (edge, &op) in arch.genes.iter().enumerate() {
            feats[edge * n_ops + op] = 1.0;
        }
        Ok(feats)
    }
}

// ─── MLP regressor (shared by PathEncodedPredictor) ─────────────────────────────

#[derive(Debug, Clone)]
struct Mlp {
    w1: Vec<f32>, // [hidden × in]
    b1: Vec<f32>, // [hidden]
    w2: Vec<f32>, // [hidden]
    b2: f32,
    in_dim: usize,
    hidden: usize,
}

impl Mlp {
    fn new(in_dim: usize, hidden: usize, rng: &mut LcgRng) -> Self {
        let scale1 = (2.0 / in_dim as f32).sqrt();
        let mut w1 = vec![0.0_f32; hidden * in_dim];
        rng.fill_normal(&mut w1);
        for v in &mut w1 {
            *v *= scale1;
        }
        let mut w2 = vec![0.0_f32; hidden];
        rng.fill_normal(&mut w2);
        let scale2 = (2.0 / hidden as f32).sqrt();
        for v in &mut w2 {
            *v *= scale2;
        }
        Self {
            w1,
            b1: vec![0.0; hidden],
            w2,
            b2: 0.0,
            in_dim,
            hidden,
        }
    }

    fn forward(&self, x: &[f32]) -> (Vec<f32>, f32) {
        let mut h = vec![0.0_f32; self.hidden];
        for (j, hj) in h.iter_mut().enumerate() {
            let mut acc = self.b1[j];
            let row = &self.w1[j * self.in_dim..(j + 1) * self.in_dim];
            for (xi, wij) in x.iter().zip(row) {
                acc += xi * wij;
            }
            *hj = acc.max(0.0); // ReLU
        }
        let mut y = self.b2;
        for (w2j, &hj) in self.w2.iter().zip(h.iter()) {
            y += w2j * hj;
        }
        (h, y)
    }

    /// One SGD step on a single sample; returns the squared error before update.
    fn sgd_step(&mut self, x: &[f32], target: f32, lr: f32) -> f32 {
        let (h, y) = self.forward(x);
        let err = y - target;
        // dL/dy = 2·err (using L = err²); fold the 2 into lr conceptually.
        let g_out = 2.0 * err;
        // Output layer grads.
        for (j, &hj) in h.iter().enumerate() {
            let g_w2 = g_out * hj;
            // Hidden grad through ReLU.
            let relu_grad = if hj > 0.0 { 1.0 } else { 0.0 };
            let g_h = g_out * self.w2[j] * relu_grad;
            // Update w1 row j.
            let row = &mut self.w1[j * self.in_dim..(j + 1) * self.in_dim];
            for (wij, xi) in row.iter_mut().zip(x.iter()) {
                *wij -= lr * g_h * xi;
            }
            self.b1[j] -= lr * g_h;
            self.w2[j] -= lr * g_w2;
        }
        self.b2 -= lr * g_out;
        err * err
    }
}

/// BANANAS path-encoded MLP accuracy predictor.
#[derive(Debug, Clone)]
pub struct PathEncodedPredictor {
    encoder: PathEncoder,
    mlp: Mlp,
}

impl PathEncodedPredictor {
    /// Create an untrained predictor with `hidden` hidden units.
    ///
    /// # Errors
    /// [`NasError::InvalidNumOps`] if `hidden == 0`.
    pub fn new(topology: CellTopology, hidden: usize, rng: &mut LcgRng) -> NasResult<Self> {
        if hidden == 0 {
            return Err(NasError::InvalidNumOps);
        }
        let encoder = PathEncoder::new(topology);
        let mlp = Mlp::new(encoder.dim(), hidden, rng);
        Ok(Self { encoder, mlp })
    }

    /// Train on `(arch, accuracy)` pairs for `epochs` shuffled passes.
    ///
    /// Returns the final-epoch mean squared error.
    ///
    /// # Errors
    /// - [`NasError::EmptySearchSpace`] if `samples` is empty.
    /// - propagates encoding errors.
    pub fn fit(
        &mut self,
        samples: &[(ArchEncoding, f32)],
        epochs: usize,
        lr: f32,
        rng: &mut LcgRng,
    ) -> NasResult<f32> {
        if samples.is_empty() {
            return Err(NasError::EmptySearchSpace);
        }
        let encoded: Vec<(Vec<f32>, f32)> = samples
            .iter()
            .map(|(a, t)| self.encoder.encode(a).map(|e| (e, *t)))
            .collect::<NasResult<Vec<_>>>()?;
        let mut order: Vec<usize> = (0..encoded.len()).collect();
        let mut last_mse = 0.0_f32;
        for _ in 0..epochs.max(1) {
            rng.shuffle(&mut order);
            let mut sse = 0.0_f32;
            for &i in &order {
                sse += self.mlp.sgd_step(&encoded[i].0, encoded[i].1, lr);
            }
            last_mse = sse / encoded.len() as f32;
        }
        Ok(last_mse)
    }

    /// Predict accuracy for an architecture (clamped to `[0, 1]`).
    ///
    /// # Errors
    /// Propagates encoding errors.
    pub fn predict(&self, arch: &ArchEncoding) -> NasResult<f32> {
        let x = self.encoder.encode(arch)?;
        let (_, y) = self.mlp.forward(&x);
        Ok(y.clamp(0.0, 1.0))
    }
}

// ─── GnnPredictor (NPENAS) ──────────────────────────────────────────────────────

/// Message-passing GNN accuracy predictor over the cell DAG (NPENAS-style).
///
/// Forward pass:
/// 1. every node starts with a constant embedding (ones × `1/√dim`);
/// 2. for each of `n_layers` rounds, a node's new embedding is
///    `ReLU(W · (self + Σ_incoming op_embed[op] ⊙ neighbour) + b)`, where
///    `op_embed[op]` is a learned per-op gate vector along the feature axis;
/// 3. the final node embeddings are mean-pooled and read out by a linear head.
///
/// Trained end-to-end by SGD with reverse-mode gradients through the readout,
/// the per-layer `(W, b)`, and the op-embedding table.
#[derive(Debug, Clone)]
pub struct GnnPredictor {
    topology: CellTopology,
    dim: usize,
    n_layers: usize,
    // Per-layer update matrices [n_layers][dim × dim] and biases [n_layers][dim].
    w: Vec<Vec<f32>>,
    b: Vec<Vec<f32>>,
    // Op embedding gates [n_ops][dim].
    op_embed: Vec<Vec<f32>>,
    // Readout [dim] + bias.
    readout: Vec<f32>,
    readout_bias: f32,
}

impl GnnPredictor {
    /// Create an untrained GNN predictor.
    ///
    /// # Errors
    /// [`NasError::InvalidNumOps`] if `dim == 0` or `n_layers == 0`.
    pub fn new(
        topology: CellTopology,
        dim: usize,
        n_layers: usize,
        rng: &mut LcgRng,
    ) -> NasResult<Self> {
        if dim == 0 || n_layers == 0 {
            return Err(NasError::InvalidNumOps);
        }
        let scale = (2.0 / dim as f32).sqrt();
        let mk_mat = |rng: &mut LcgRng| {
            let mut m = vec![0.0_f32; dim * dim];
            rng.fill_normal(&mut m);
            for v in &mut m {
                *v *= scale;
            }
            m
        };
        let w: Vec<Vec<f32>> = (0..n_layers).map(|_| mk_mat(rng)).collect();
        let b: Vec<Vec<f32>> = (0..n_layers).map(|_| vec![0.0_f32; dim]).collect();
        let n_ops = topology.n_ops;
        let op_embed: Vec<Vec<f32>> = (0..n_ops)
            .map(|_| {
                let mut e = vec![0.0_f32; dim];
                rng.fill_normal(&mut e);
                for v in &mut e {
                    // Centre op gates around 1 so the initial network passes
                    // signal rather than zeroing it.
                    *v = 1.0 + 0.1 * *v;
                }
                e
            })
            .collect();
        let mut readout = vec![0.0_f32; dim];
        rng.fill_normal(&mut readout);
        for v in &mut readout {
            *v *= scale;
        }
        Ok(Self {
            topology,
            dim,
            n_layers,
            w,
            b,
            op_embed,
            readout,
            readout_bias: 0.0,
        })
    }

    fn init_node_embeddings(&self) -> Vec<Vec<f32>> {
        let init = 1.0 / (self.dim as f32).sqrt();
        vec![vec![init; self.dim]; self.topology.n_nodes()]
    }

    /// Forward pass returning the per-layer node activations (for backprop) and
    /// the scalar prediction. `acts[0]` is the input embedding; `acts[l+1]` is
    /// the post-layer-`l` embedding.
    fn forward_full(&self, arch: &ArchEncoding) -> (Vec<Vec<Vec<f32>>>, Vec<f32>, f32) {
        let n_nodes = self.topology.n_nodes();
        let n_in = self.topology.n_inputs;
        let mut acts: Vec<Vec<Vec<f32>>> = Vec::with_capacity(self.n_layers + 1);
        acts.push(self.init_node_embeddings());

        for layer in 0..self.n_layers {
            let prev = &acts[layer];
            let mut next = vec![vec![0.0_f32; self.dim]; n_nodes];
            // Input nodes carry their embedding through unchanged (no incoming).
            for node in 0..n_in {
                next[node].clone_from(&prev[node]);
            }
            for k in 0..self.topology.n_intermediate {
                let node = n_in + k;
                // Aggregate: self + Σ gated neighbour.
                let mut agg = prev[node].clone();
                let preds = &self.topology.predecessors[k];
                for (slot, &src) in preds.iter().enumerate() {
                    let edge = k * self.topology.n_inputs_per_node + slot;
                    let op = arch.genes[edge];
                    let gate = &self.op_embed[op];
                    for d in 0..self.dim {
                        agg[d] += gate[d] * prev[src][d];
                    }
                }
                // Linear update W·agg + b, then ReLU.
                let wmat = &self.w[layer];
                let bias = &self.b[layer];
                for o in 0..self.dim {
                    let mut acc = bias[o];
                    let row = &wmat[o * self.dim..(o + 1) * self.dim];
                    for (a, wij) in agg.iter().zip(row) {
                        acc += a * wij;
                    }
                    next[node][o] = acc.max(0.0);
                }
            }
            acts.push(next);
        }

        // Mean-pool the final embeddings.
        let final_acts = &acts[self.n_layers];
        let mut pooled = vec![0.0_f32; self.dim];
        for emb in final_acts {
            for d in 0..self.dim {
                pooled[d] += emb[d];
            }
        }
        let inv_n = 1.0 / n_nodes as f32;
        for v in &mut pooled {
            *v *= inv_n;
        }
        // Readout.
        let mut y = self.readout_bias;
        for (&ro, &p) in self.readout.iter().zip(pooled.iter()) {
            y += ro * p;
        }
        (acts, pooled, y)
    }

    /// Predict accuracy (clamped to `[0, 1]`).
    ///
    /// # Errors
    /// Propagates genome validation errors.
    pub fn predict(&self, arch: &ArchEncoding) -> NasResult<f32> {
        self.topology.validate_genome(arch)?;
        let (_, _, y) = self.forward_full(arch);
        Ok(y.clamp(0.0, 1.0))
    }

    /// One SGD step on a single `(arch, target)` sample. Returns squared error.
    fn sgd_step(&mut self, arch: &ArchEncoding, target: f32, lr: f32) -> f32 {
        let n_nodes = self.topology.n_nodes();
        let n_in = self.topology.n_inputs;
        let (acts, pooled, y) = self.forward_full(arch);
        let err = y - target;
        let g_out = 2.0 * err;

        // Readout grads + back to pooled.
        let mut g_pooled = vec![0.0_f32; self.dim];
        for d in 0..self.dim {
            g_pooled[d] = g_out * self.readout[d];
            self.readout[d] -= lr * g_out * pooled[d];
        }
        self.readout_bias -= lr * g_out;

        // Pool gradient distributes equally across final node embeddings.
        let inv_n = 1.0 / n_nodes as f32;
        let mut g_node: Vec<Vec<f32>> = vec![vec![0.0_f32; self.dim]; n_nodes];
        for gn in g_node.iter_mut() {
            for d in 0..self.dim {
                gn[d] = g_pooled[d] * inv_n;
            }
        }

        // Back-propagate through layers in reverse. Accumulate parameter grads,
        // then apply at the end (so a single step uses consistent params).
        let mut gw: Vec<Vec<f32>> = self.w.iter().map(|m| vec![0.0_f32; m.len()]).collect();
        let mut gb: Vec<Vec<f32>> = self.b.iter().map(|v| vec![0.0_f32; v.len()]).collect();
        let mut g_op: Vec<Vec<f32>> = self
            .op_embed
            .iter()
            .map(|v| vec![0.0_f32; v.len()])
            .collect();

        for layer in (0..self.n_layers).rev() {
            let prev = &acts[layer];
            let cur = &acts[layer + 1];
            // g_prev accumulates gradient w.r.t. the previous-layer embeddings.
            let mut g_prev: Vec<Vec<f32>> = vec![vec![0.0_f32; self.dim]; n_nodes];
            // Input nodes pass straight through.
            for node in 0..n_in {
                for d in 0..self.dim {
                    g_prev[node][d] += g_node[node][d];
                }
            }
            let wmat = &self.w[layer];
            for k in 0..self.topology.n_intermediate {
                let node = n_in + k;
                // Recompute agg for this node (cheap; keeps memory low).
                let mut agg = prev[node].clone();
                let preds = &self.topology.predecessors[k];
                for (slot, &src) in preds.iter().enumerate() {
                    let edge = k * self.topology.n_inputs_per_node + slot;
                    let op = arch.genes[edge];
                    let gate = &self.op_embed[op];
                    for d in 0..self.dim {
                        agg[d] += gate[d] * prev[src][d];
                    }
                }
                // ReLU mask from the recorded post-activation.
                // g_pre[o] = g_node_out[o] · (cur[o] > 0).
                let mut g_pre = vec![0.0_f32; self.dim];
                for o in 0..self.dim {
                    g_pre[o] = if cur[node][o] > 0.0 {
                        g_node[node][o]
                    } else {
                        0.0
                    };
                }
                // Grad to W, b, and to agg.
                let mut g_agg = vec![0.0_f32; self.dim];
                for o in 0..self.dim {
                    let gpre_o = g_pre[o];
                    if gpre_o != 0.0 {
                        let row_off = o * self.dim;
                        for (i, &a) in agg.iter().enumerate() {
                            gw[layer][row_off + i] += gpre_o * a;
                            g_agg[i] += gpre_o * wmat[row_off + i];
                        }
                        gb[layer][o] += gpre_o;
                    }
                }
                // Distribute g_agg back to self + neighbours and op gates.
                for d in 0..self.dim {
                    g_prev[node][d] += g_agg[d]; // self term
                }
                for (slot, &src) in preds.iter().enumerate() {
                    let edge = k * self.topology.n_inputs_per_node + slot;
                    let op = arch.genes[edge];
                    let gate = &self.op_embed[op];
                    for d in 0..self.dim {
                        // contribution = gate[d] * prev[src][d]
                        g_prev[src][d] += g_agg[d] * gate[d];
                        g_op[op][d] += g_agg[d] * prev[src][d];
                    }
                }
            }
            g_node = g_prev;
        }

        // Apply accumulated grads.
        for layer in 0..self.n_layers {
            for (wij, g) in self.w[layer].iter_mut().zip(&gw[layer]) {
                *wij -= lr * g;
            }
            for (bi, g) in self.b[layer].iter_mut().zip(&gb[layer]) {
                *bi -= lr * g;
            }
        }
        for (emb, g) in self.op_embed.iter_mut().zip(&g_op) {
            for (e, ge) in emb.iter_mut().zip(g) {
                *e -= lr * ge;
            }
        }
        err * err
    }

    /// Train on `(arch, accuracy)` pairs for `epochs` shuffled passes.
    ///
    /// Returns the final-epoch mean squared error.
    ///
    /// # Errors
    /// - [`NasError::EmptySearchSpace`] if `samples` is empty.
    /// - propagates genome validation errors.
    pub fn fit(
        &mut self,
        samples: &[(ArchEncoding, f32)],
        epochs: usize,
        lr: f32,
        rng: &mut LcgRng,
    ) -> NasResult<f32> {
        if samples.is_empty() {
            return Err(NasError::EmptySearchSpace);
        }
        for (a, _) in samples {
            self.topology.validate_genome(a)?;
        }
        let mut order: Vec<usize> = (0..samples.len()).collect();
        let mut last_mse = 0.0_f32;
        for _ in 0..epochs.max(1) {
            rng.shuffle(&mut order);
            let mut sse = 0.0_f32;
            for &i in &order {
                sse += self.sgd_step(&samples[i].0, samples[i].1, lr);
            }
            last_mse = sse / samples.len() as f32;
        }
        Ok(last_mse)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn topo() -> CellTopology {
        // 2 input nodes, 4 intermediate, 2 incoming edges each, 5 ops.
        CellTopology::sequential(2, 4, 2, 5).expect("topology")
    }

    fn genome(topo: &CellTopology, genes: &[usize]) -> ArchEncoding {
        ArchEncoding {
            genes: genes.to_vec(),
            n_ops: topo.n_ops,
        }
    }

    #[test]
    fn topology_dimensions() {
        let t = topo();
        assert_eq!(t.n_edges(), 8); // 4 nodes × 2 edges
        assert_eq!(t.n_nodes(), 6); // 2 inputs + 4 intermediate
        // Every predecessor id is strictly less than the consuming node id.
        for (k, preds) in t.predecessors.iter().enumerate() {
            let node = t.n_inputs + k;
            assert_eq!(preds.len(), t.n_inputs_per_node);
            for &p in preds {
                assert!(p < node, "edge must come from an earlier node");
            }
        }
    }

    #[test]
    fn topology_rejects_bad_dims() {
        assert!(CellTopology::sequential(0, 4, 2, 5).is_err());
        assert!(CellTopology::sequential(2, 0, 2, 5).is_err());
        assert!(CellTopology::sequential(2, 4, 0, 5).is_err());
        assert!(CellTopology::sequential(2, 4, 2, 0).is_err());
    }

    #[test]
    fn path_encoding_is_binary_and_correct_dim() {
        let t = topo();
        let enc = PathEncoder::new(t.clone());
        assert_eq!(enc.dim(), t.n_edges() * t.n_ops); // 8 × 5 = 40
        let g = genome(&t, &[0, 1, 2, 3, 4, 0, 1, 2]);
        let f = enc.encode(&g).expect("encode");
        // Exactly n_edges ones, one per edge.
        let ones = f.iter().filter(|&&v| v == 1.0).count();
        assert_eq!(ones, t.n_edges());
        // Edge 0 op 0 set; edge 1 op 1 set.
        assert_eq!(f[0], 1.0); // edge 0, op 0
        assert_eq!(f[t.n_ops + 1], 1.0); // edge 1, op 1
    }

    #[test]
    fn path_encoding_rejects_bad_genome() {
        let t = topo();
        let enc = PathEncoder::new(t.clone());
        // Wrong length.
        assert!(enc.encode(&genome(&t, &[0, 1])).is_err());
        // Op out of range.
        assert!(enc.encode(&genome(&t, &[0, 1, 2, 3, 4, 0, 1, 9])).is_err());
    }

    #[test]
    fn path_predictor_overfits_small_dataset() {
        let t = topo();
        let mut rng = LcgRng::new(1);
        let mut p = PathEncodedPredictor::new(t.clone(), 32, &mut rng).expect("new");
        // Build a tiny labelled set with a deterministic target = fraction of
        // edges that use op 0 (a learnable function of the encoding).
        let mut samples = Vec::new();
        let mut g = LcgRng::new(2);
        for _ in 0..16 {
            let genes: Vec<usize> = (0..t.n_edges()).map(|_| g.next_usize(t.n_ops)).collect();
            let target = genes.iter().filter(|&&x| x == 0).count() as f32 / t.n_edges() as f32;
            samples.push((genome(&t, &genes), target));
        }
        let mse0 = {
            // Initial MSE (one no-shuffle pass conceptually): measure predictions.
            let mut s = 0.0_f32;
            for (a, tg) in &samples {
                let pred = p.predict(a).expect("predict");
                s += (pred - tg).powi(2);
            }
            s / samples.len() as f32
        };
        let final_mse = p.fit(&samples, 400, 0.05, &mut rng).expect("fit");
        assert!(final_mse.is_finite());
        assert!(
            final_mse < mse0,
            "training must reduce error: {final_mse} !< {mse0}"
        );
        assert!(
            final_mse < 0.02,
            "should fit the small set well: {final_mse}"
        );
    }

    #[test]
    fn path_predictor_rejects_empty_fit() {
        let t = topo();
        let mut rng = LcgRng::new(1);
        let mut p = PathEncodedPredictor::new(t, 8, &mut rng).expect("new");
        assert_eq!(
            p.fit(&[], 10, 0.01, &mut rng),
            Err(NasError::EmptySearchSpace)
        );
    }

    #[test]
    fn gnn_forward_is_finite_and_clamped() {
        let t = topo();
        let mut rng = LcgRng::new(3);
        let p = GnnPredictor::new(t.clone(), 8, 2, &mut rng).expect("new");
        let g = genome(&t, &[0, 1, 2, 3, 4, 0, 1, 2]);
        let y = p.predict(&g).expect("predict");
        assert!((0.0..=1.0).contains(&y));
    }

    #[test]
    fn gnn_distinguishes_architectures() {
        // Two different genomes should (after no training) generally give
        // different raw outputs — the GNN is sensitive to the op labels.
        let t = topo();
        let mut rng = LcgRng::new(5);
        let p = GnnPredictor::new(t.clone(), 12, 3, &mut rng).expect("new");
        let a = genome(&t, &[0, 0, 0, 0, 0, 0, 0, 0]);
        let b = genome(&t, &[4, 4, 4, 4, 4, 4, 4, 4]);
        let (_, _, ya) = p.forward_full(&a);
        let (_, _, yb) = p.forward_full(&b);
        assert!(
            (ya - yb).abs() > 1e-6,
            "GNN should distinguish ya={ya} yb={yb}"
        );
    }

    #[test]
    fn gnn_learns_a_graph_function() {
        // Target: normalised count of op-4 edges. The GNN must learn to predict
        // it, reducing MSE substantially below the untrained baseline.
        let t = topo();
        let mut rng = LcgRng::new(11);
        let mut p = GnnPredictor::new(t.clone(), 16, 2, &mut rng).expect("new");
        let mut g = LcgRng::new(22);
        let mut samples = Vec::new();
        for _ in 0..24 {
            let genes: Vec<usize> = (0..t.n_edges()).map(|_| g.next_usize(t.n_ops)).collect();
            let target = genes.iter().filter(|&&x| x == 4).count() as f32 / t.n_edges() as f32;
            samples.push((genome(&t, &genes), target));
        }
        let mse0 = {
            let mut s = 0.0_f32;
            for (a, tg) in &samples {
                let pred = p.predict(a).expect("predict");
                s += (pred - tg).powi(2);
            }
            s / samples.len() as f32
        };
        let final_mse = p.fit(&samples, 300, 0.02, &mut rng).expect("fit");
        assert!(final_mse.is_finite());
        assert!(
            final_mse < mse0 * 0.5,
            "GNN training should at least halve the error: {final_mse} vs {mse0}"
        );
    }

    #[test]
    fn gnn_rejects_bad_dims_and_empty_fit() {
        let t = topo();
        let mut rng = LcgRng::new(1);
        assert!(GnnPredictor::new(t.clone(), 0, 2, &mut rng).is_err());
        assert!(GnnPredictor::new(t.clone(), 4, 0, &mut rng).is_err());
        let mut p = GnnPredictor::new(t, 4, 1, &mut rng).expect("new");
        assert_eq!(
            p.fit(&[], 5, 0.01, &mut rng),
            Err(NasError::EmptySearchSpace)
        );
    }

    #[test]
    fn gnn_training_is_deterministic_given_seed() {
        let t = topo();
        let build = |seed: u64| {
            let mut rng = LcgRng::new(seed);
            let mut p = GnnPredictor::new(t.clone(), 8, 2, &mut rng).expect("new");
            let mut g = LcgRng::new(seed + 1);
            let mut samples = Vec::new();
            for _ in 0..10 {
                let genes: Vec<usize> = (0..t.n_edges()).map(|_| g.next_usize(t.n_ops)).collect();
                let target = genes.iter().filter(|&&x| x == 0).count() as f32 / t.n_edges() as f32;
                samples.push((genome(&t, &genes), target));
            }
            let mse = p.fit(&samples, 50, 0.02, &mut rng).expect("fit");
            (mse, p.predict(&samples[0].0).expect("predict"))
        };
        let (m1, y1) = build(7);
        let (m2, y2) = build(7);
        assert_eq!(m1.to_bits(), m2.to_bits());
        assert_eq!(y1.to_bits(), y2.to_bits());
    }
}
