/// Product Quantization codebook: `m` subspaces, `ksub` codewords each of size `dsub`.
pub struct PqCodebook {
    /// Flat storage `[m, ksub, dsub]`.
    centroids: Vec<f32>,
    pub m: usize,
    pub ksub: usize,
    pub dsub: usize,
}

impl PqCodebook {
    /// Create an uninitialised codebook of the given shape.
    #[must_use]
    pub fn new(m: usize, ksub: usize, dsub: usize) -> Self {
        Self {
            centroids: vec![0.0_f32; m * ksub * dsub],
            m,
            ksub,
            dsub,
        }
    }

    /// Mutable access to centroid slice for subspace `s`, code `c`.
    pub fn centroid_mut(&mut self, s: usize, c: usize) -> &mut [f32] {
        let off = (s * self.ksub + c) * self.dsub;
        &mut self.centroids[off..off + self.dsub]
    }

    /// Immutable slice for subspace `s`, code `c`.
    #[must_use]
    pub fn centroid(&self, s: usize, c: usize) -> &[f32] {
        let off = (s * self.ksub + c) * self.dsub;
        &self.centroids[off..off + self.dsub]
    }

    /// Raw centroid storage for external use (e.g. PTX kernel input).
    #[must_use]
    pub fn centroids_raw(&self) -> &[f32] {
        &self.centroids
    }
}
