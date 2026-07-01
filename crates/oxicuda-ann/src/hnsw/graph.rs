/// HNSW multi-layer proximity graph.
///
/// Vectors are stored flat in `vectors` (row-major, `[n_nodes, dim]`).
/// `layers[node_id][layer]` stores neighbor IDs at that layer.
pub struct HnswGraph {
    /// Neighbor lists: `layers[node_id][layer] = Vec<neighbor_id>`.
    pub layers: Vec<Vec<Vec<u32>>>,
    pub dim: usize,
    /// Flat vector storage `[n, dim]`.
    pub vectors: Vec<f32>,
    pub m: usize,
    pub m_max0: usize,
    pub ef_construction: usize,
    pub ef: usize,
    /// Current entry-point node ID (the one at the top layer).
    pub entry_point: Option<u32>,
    /// Max layer index currently in use.
    pub max_layer: usize,
}

impl HnswGraph {
    #[must_use]
    pub fn new(dim: usize, m: usize, ef_construction: usize, ef: usize) -> Self {
        Self {
            layers: Vec::new(),
            dim,
            vectors: Vec::new(),
            m,
            m_max0: 2 * m,
            ef_construction,
            ef,
            entry_point: None,
            max_layer: 0,
        }
    }

    #[must_use]
    pub fn n_nodes(&self) -> usize {
        self.layers.len()
    }

    #[must_use]
    pub fn get_vector(&self, id: u32) -> &[f32] {
        let s = id as usize * self.dim;
        &self.vectors[s..s + self.dim]
    }

    #[must_use]
    pub fn get_neighbors(&self, id: u32, layer: usize) -> &[u32] {
        let id = id as usize;
        if id >= self.layers.len() || layer >= self.layers[id].len() {
            return &[];
        }
        &self.layers[id][layer]
    }

    pub fn set_neighbors(&mut self, id: u32, layer: usize, nbrs: Vec<u32>) {
        let id = id as usize;
        while self.layers[id].len() <= layer {
            self.layers[id].push(Vec::new());
        }
        self.layers[id][layer] = nbrs;
    }

    /// Allocate a new node with `n_layers` layer slots.
    pub fn add_node(&mut self, v: &[f32], n_layers: usize) -> u32 {
        let id = self.layers.len() as u32;
        self.vectors.extend_from_slice(v);
        self.layers.push(vec![Vec::new(); n_layers]);
        id
    }

    pub fn l2_sq(&self, a: u32, b: u32) -> f32 {
        let va = self.get_vector(a);
        let vb = self.get_vector(b);
        va.iter()
            .zip(vb.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum()
    }

    pub fn l2_sq_query(&self, query: &[f32], node: u32) -> f32 {
        let vn = self.get_vector(node);
        query
            .iter()
            .zip(vn.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_graph_is_empty() {
        let g = HnswGraph::new(4, 8, 200, 10);
        assert_eq!(g.n_nodes(), 0);
        assert_eq!(g.dim, 4);
        assert_eq!(g.m, 8);
        assert_eq!(g.m_max0, 16, "m_max0 must equal 2*m");
        assert_eq!(g.ef_construction, 200);
        assert_eq!(g.ef, 10);
        assert!(g.entry_point.is_none());
        assert_eq!(g.max_layer, 0);
    }

    #[test]
    fn add_node_increments_count_and_returns_sequential_ids() {
        let mut g = HnswGraph::new(2, 8, 200, 10);
        let id0 = g.add_node(&[1.0_f32, 2.0], 1);
        assert_eq!(id0, 0);
        assert_eq!(g.n_nodes(), 1);

        let id1 = g.add_node(&[3.0_f32, 4.0], 2);
        assert_eq!(id1, 1);
        assert_eq!(g.n_nodes(), 2);
    }

    #[test]
    fn add_node_layer_slot_count_matches_n_layers() {
        let mut g = HnswGraph::new(2, 8, 200, 10);
        g.add_node(&[0.0_f32, 0.0], 3); // 3 layer slots: 0, 1, 2
        g.add_node(&[1.0_f32, 0.0], 1); // 1 layer slot: 0 only

        assert_eq!(g.layers[0].len(), 3, "node 0 should have 3 layer slots");
        assert_eq!(g.layers[1].len(), 1, "node 1 should have 1 layer slot");
    }

    #[test]
    fn get_vector_round_trips() {
        let mut g = HnswGraph::new(3, 8, 200, 10);
        let v = vec![1.5_f32, -2.3, 0.7];
        g.add_node(&v, 1);
        assert_eq!(
            g.get_vector(0),
            v.as_slice(),
            "stored vector must round-trip"
        );
    }

    #[test]
    fn get_neighbors_returns_empty_for_out_of_range_layer() {
        let mut g = HnswGraph::new(2, 8, 200, 10);
        g.add_node(&[0.0_f32, 0.0], 1); // only layer 0 allocated
        assert!(
            g.get_neighbors(0, 5).is_empty(),
            "out-of-range layer must return empty slice"
        );
    }

    #[test]
    fn get_neighbors_returns_empty_for_nonexistent_node() {
        let g = HnswGraph::new(2, 8, 200, 10);
        assert!(
            g.get_neighbors(99, 0).is_empty(),
            "nonexistent node must return empty slice"
        );
    }

    #[test]
    fn set_and_get_neighbors_round_trips() {
        let mut g = HnswGraph::new(2, 8, 200, 10);
        g.add_node(&[0.0_f32, 0.0], 2);
        g.add_node(&[1.0_f32, 0.0], 2);
        g.add_node(&[0.0_f32, 1.0], 2);

        g.set_neighbors(0, 0, vec![1, 2]);
        g.set_neighbors(0, 1, vec![1]);

        assert_eq!(
            g.get_neighbors(0, 0),
            &[1u32, 2],
            "layer-0 neighbors mismatch"
        );
        assert_eq!(g.get_neighbors(0, 1), &[1u32], "layer-1 neighbors mismatch");
        assert!(
            g.get_neighbors(0, 2).is_empty(),
            "unset layer should return empty"
        );
    }

    #[test]
    fn l2_sq_known_value() {
        let mut g = HnswGraph::new(2, 8, 200, 10);
        g.add_node(&[0.0_f32, 0.0], 1);
        g.add_node(&[3.0_f32, 4.0], 1);
        // l2_sq([0,0],[3,4]) = 9 + 16 = 25
        let d = g.l2_sq(0, 1);
        assert!((d - 25.0_f32).abs() < 1e-5, "l2_sq expected 25 got {d}");
    }

    #[test]
    fn l2_sq_query_matches_l2_sq() {
        let mut g = HnswGraph::new(2, 8, 200, 10);
        g.add_node(&[1.0_f32, 2.0], 1);
        g.add_node(&[4.0_f32, 6.0], 1);
        let query: Vec<f32> = g.get_vector(0).to_vec();
        let from_stored = g.l2_sq(0, 1);
        let from_query = g.l2_sq_query(&query, 1);
        assert!(
            (from_stored - from_query).abs() < 1e-6,
            "l2_sq and l2_sq_query must agree: {from_stored} vs {from_query}"
        );
    }
}
