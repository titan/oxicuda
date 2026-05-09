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
