//! Online hyperdimensional clustering (streaming centroids).
//!
//! An online, k-means-like clustering scheme over binary ±1 hypervectors. Each
//! cluster keeps an `i32` accumulator and a thresholded binary centroid. An
//! incoming HV is assigned to the nearest current centroid by cosine similarity,
//! added into that cluster's accumulator, and the affected centroid is
//! re-thresholded. Processing is sequential / single-pass and deterministic for
//! a fixed seed (random centroid initialisation and threshold tie-breaks both
//! draw from the supplied [`crate::handle::LcgRng`]).

use crate::distance::cosine::cosine_binary;
use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;
use crate::vector::binary::{random_binary, threshold_binary};

/// Online HD clustering state: per-cluster accumulators, centroids, and counts.
pub struct HdCluster {
    /// Dimension of hypervectors.
    dim: usize,
    /// Number of clusters.
    n_clusters: usize,
    /// Per-cluster `i32` accumulator (running sum of assigned HVs).
    accumulators: Vec<Vec<i32>>,
    /// Per-cluster thresholded binary centroid.
    centroids: Vec<Vec<i8>>,
    /// Per-cluster count of assigned points.
    counts: Vec<usize>,
}

impl HdCluster {
    /// Create a new clustering with `n_clusters` randomly-initialised centroids
    /// over `dim`-dimensional HVs.
    ///
    /// # Errors
    /// Returns [`HdcError::EmptyInput`] if `n_clusters == 0`, and
    /// [`HdcError::ZeroDimension`] if `dim == 0`.
    pub fn new(n_clusters: usize, dim: usize, rng: &mut LcgRng) -> HdcResult<Self> {
        if n_clusters == 0 {
            return Err(HdcError::EmptyInput);
        }
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        let mut centroids = Vec::with_capacity(n_clusters);
        for _ in 0..n_clusters {
            centroids.push(random_binary(dim, rng)?);
        }
        Ok(Self {
            dim,
            n_clusters,
            accumulators: vec![vec![0i32; dim]; n_clusters],
            centroids,
            counts: vec![0usize; n_clusters],
        })
    }

    /// Index of the centroid most similar (cosine) to `hv`. Ties resolve to the
    /// smallest cluster index.
    ///
    /// # Errors
    /// Returns [`HdcError::DimensionMismatch`] if `hv.len() != dim`.
    pub fn assign(&self, hv: &[i8]) -> HdcResult<usize> {
        if hv.len() != self.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.dim,
                got: hv.len(),
            });
        }
        let mut best_idx = 0usize;
        let mut best_sim = f32::NEG_INFINITY;
        for (idx, centroid) in self.centroids.iter().enumerate() {
            let sim = cosine_binary(hv, centroid)?;
            if sim > best_sim {
                best_sim = sim;
                best_idx = idx;
            }
        }
        Ok(best_idx)
    }

    /// Assign `hv` to its nearest centroid, fold it into that cluster's
    /// accumulator, re-threshold the centroid, bump the count, and return the
    /// chosen cluster index.
    ///
    /// # Errors
    /// Returns [`HdcError::DimensionMismatch`] if `hv.len() != dim`.
    pub fn update(&mut self, hv: &[i8], rng: &mut LcgRng) -> HdcResult<usize> {
        let c = self.assign(hv)?;
        for (a, &v) in self.accumulators[c].iter_mut().zip(hv.iter()) {
            *a += v as i32;
        }
        self.centroids[c] = threshold_binary(&self.accumulators[c], rng)?;
        self.counts[c] += 1;
        Ok(c)
    }

    /// Run [`Self::update`] over every HV in `data` (single pass).
    ///
    /// # Errors
    /// Returns [`HdcError::DimensionMismatch`] if any item's length differs from
    /// `dim`.
    pub fn fit(&mut self, data: &[Vec<i8>], rng: &mut LcgRng) -> HdcResult<()> {
        for hv in data {
            self.update(hv, rng)?;
        }
        Ok(())
    }

    /// Binary centroid of cluster `c`.
    ///
    /// # Errors
    /// Returns [`HdcError::ClassNotFound`] if `c >= n_clusters`.
    pub fn centroid(&self, c: usize) -> HdcResult<&[i8]> {
        if c >= self.n_clusters {
            return Err(HdcError::ClassNotFound(c));
        }
        Ok(&self.centroids[c])
    }

    /// Number of points assigned to cluster `c`.
    ///
    /// # Errors
    /// Returns [`HdcError::ClassNotFound`] if `c >= n_clusters`.
    pub fn cluster_size(&self, c: usize) -> HdcResult<usize> {
        if c >= self.n_clusters {
            return Err(HdcError::ClassNotFound(c));
        }
        Ok(self.counts[c])
    }

    /// Number of clusters.
    #[must_use]
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    /// Hypervector dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Per-cluster assignment counts.
    #[must_use]
    pub fn counts(&self) -> &[usize] {
        &self.counts
    }

    /// Clustering inertia: the sum over points of `1 - cosine(point, assigned
    /// centroid)`. Lower values indicate tighter clusters.
    ///
    /// # Errors
    /// Returns [`HdcError::DimensionMismatch`] if any item's length differs from
    /// `dim`.
    pub fn inertia(&self, data: &[Vec<i8>]) -> HdcResult<f32> {
        let mut total = 0f32;
        for hv in data {
            let c = self.assign(hv)?;
            let sim = cosine_binary(hv, &self.centroids[c])?;
            total += 1.0 - sim;
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::vector::binary::random_binary;

    #[test]
    fn new_rejects_zero_clusters() {
        let mut rng = LcgRng::new(1);
        let err = HdCluster::new(0, 64, &mut rng);
        assert!(matches!(err, Err(HdcError::EmptyInput)));
    }

    #[test]
    fn new_rejects_zero_dim() {
        let mut rng = LcgRng::new(1);
        let err = HdCluster::new(3, 0, &mut rng);
        assert!(matches!(err, Err(HdcError::ZeroDimension)));
    }

    #[test]
    fn assign_returns_index_in_range() {
        let mut rng = LcgRng::new(5);
        let hc = HdCluster::new(4, 64, &mut rng).expect("new");
        let q = random_binary(64, &mut rng).expect("rand");
        let c = hc.assign(&q).expect("assign");
        assert!(c < 4);
    }

    #[test]
    fn update_returns_cluster_and_bumps_count() {
        let mut rng = LcgRng::new(6);
        let mut hc = HdCluster::new(3, 64, &mut rng).expect("new");
        let q = random_binary(64, &mut rng).expect("rand");
        let c = hc.update(&q, &mut rng).expect("update");
        assert!(c < 3);
        assert_eq!(hc.cluster_size(c).expect("size"), 1);
    }

    #[test]
    fn counts_sum_to_num_points_after_fit() {
        let mut rng = LcgRng::new(7);
        let mut hc = HdCluster::new(3, 128, &mut rng).expect("new");
        let data: Vec<Vec<i8>> = (0..20)
            .map(|_| random_binary(128, &mut rng).expect("rand"))
            .collect();
        hc.fit(&data, &mut rng).expect("fit");
        let total: usize = hc.counts().iter().sum();
        assert_eq!(total, 20);
    }

    #[test]
    fn point_identical_to_centroid_assigns_to_it() {
        let mut rng = LcgRng::new(8);
        let hc = HdCluster::new(3, 64, &mut rng).expect("new");
        // Use cluster 1's own centroid as the query — sim 1.0 there.
        let centroid1 = hc.centroid(1).expect("centroid").to_vec();
        let assigned = hc.assign(&centroid1).expect("assign");
        let sim = cosine_binary(&centroid1, hc.centroid(assigned).expect("c")).expect("cos");
        assert!((sim - 1.0).abs() < 1e-6, "sim={sim}");
    }

    #[test]
    fn two_separated_groups_mostly_co_assign() {
        let mut rng = LcgRng::new(2025);
        let dim = 512;
        let mut hc = HdCluster::new(2, dim, &mut rng).expect("new");
        let base_a = random_binary(dim, &mut rng).expect("a");
        let base_b = random_binary(dim, &mut rng).expect("b");
        // Build two groups of slightly-noisy variants.
        let mut group_a = Vec::new();
        let mut group_b = Vec::new();
        for s in 0..10 {
            let mut a = base_a.clone();
            let mut b = base_b.clone();
            for f in 0..5 {
                let idx = (s * 5 + f) % dim;
                a[idx] = -a[idx];
                b[idx] = -b[idx];
            }
            group_a.push(a);
            group_b.push(b);
        }
        // Interleave for a fair stream.
        let mut data = Vec::new();
        for i in 0..10 {
            data.push(group_a[i].clone());
            data.push(group_b[i].clone());
        }
        hc.fit(&data, &mut rng).expect("fit");
        // Each group's members should land predominantly in one cluster.
        let a_assignments: Vec<usize> = group_a.iter().map(|h| hc.assign(h).expect("a")).collect();
        let b_assignments: Vec<usize> = group_b.iter().map(|h| hc.assign(h).expect("b")).collect();
        let a_in0 = a_assignments.iter().filter(|&&c| c == 0).count();
        let b_in0 = b_assignments.iter().filter(|&&c| c == 0).count();
        // Majority of group A in one cluster, majority of group B in the other.
        let a_major = a_in0.max(group_a.len() - a_in0);
        let b_major = b_in0.max(group_b.len() - b_in0);
        assert!(
            a_major >= 7,
            "group A not cohesive: {a_in0}/10 in cluster 0"
        );
        assert!(
            b_major >= 7,
            "group B not cohesive: {b_in0}/10 in cluster 0"
        );
    }

    #[test]
    fn centroid_out_of_range_class_not_found() {
        let mut rng = LcgRng::new(9);
        let hc = HdCluster::new(2, 64, &mut rng).expect("new");
        let err = hc.centroid(5);
        assert!(matches!(err, Err(HdcError::ClassNotFound(5))));
    }

    #[test]
    fn update_dimension_mismatch_rejected() {
        let mut rng = LcgRng::new(10);
        let mut hc = HdCluster::new(2, 64, &mut rng).expect("new");
        let q: Vec<i8> = vec![1i8; 32];
        let err = hc.update(&q, &mut rng);
        assert!(matches!(
            err,
            Err(HdcError::DimensionMismatch {
                expected: 64,
                got: 32
            })
        ));
    }

    #[test]
    fn inertia_low_for_well_separated_data() {
        let mut rng = LcgRng::new(2026);
        let dim = 512;
        let mut hc = HdCluster::new(2, dim, &mut rng).expect("new");
        let base_a = random_binary(dim, &mut rng).expect("a");
        let base_b = random_binary(dim, &mut rng).expect("b");
        let mut data = Vec::new();
        for s in 0..10 {
            let mut a = base_a.clone();
            let mut b = base_b.clone();
            for f in 0..3 {
                let idx = (s * 3 + f) % dim;
                a[idx] = -a[idx];
                b[idx] = -b[idx];
            }
            data.push(a);
            data.push(b);
        }
        hc.fit(&data, &mut rng).expect("fit");
        let two_cluster = hc.inertia(&data).expect("inertia2");

        // Single-cluster baseline: all points share one centroid.
        let mut hc1 = HdCluster::new(1, dim, &mut rng).expect("new1");
        hc1.fit(&data, &mut rng).expect("fit1");
        let one_cluster = hc1.inertia(&data).expect("inertia1");

        assert!(
            two_cluster < one_cluster,
            "2-cluster inertia {two_cluster} not below 1-cluster {one_cluster}"
        );
    }

    #[test]
    fn cluster_size_matches_manual_count() {
        let mut rng = LcgRng::new(12);
        let dim = 128;
        let mut hc = HdCluster::new(3, dim, &mut rng).expect("new");
        let data: Vec<Vec<i8>> = (0..15)
            .map(|_| random_binary(dim, &mut rng).expect("rand"))
            .collect();
        // Record assignment at update time.
        let mut manual = [0usize; 3];
        for hv in &data {
            let c = hc.update(hv, &mut rng).expect("update");
            manual[c] += 1;
        }
        for (c, &expected) in manual.iter().enumerate() {
            assert_eq!(hc.cluster_size(c).expect("size"), expected);
        }
    }

    #[test]
    fn fit_over_empty_data_is_noop_ok() {
        let mut rng = LcgRng::new(13);
        let mut hc = HdCluster::new(2, 64, &mut rng).expect("new");
        let empty: Vec<Vec<i8>> = Vec::new();
        hc.fit(&empty, &mut rng).expect("fit");
        assert_eq!(hc.counts().iter().sum::<usize>(), 0);
    }

    #[test]
    fn rethresholding_moves_centroid_toward_added_point() {
        let mut rng = LcgRng::new(14);
        let dim = 256;
        let mut hc = HdCluster::new(1, dim, &mut rng).expect("new");
        let target = random_binary(dim, &mut rng).expect("target");
        let before = cosine_binary(&target, hc.centroid(0).expect("c0")).expect("cos0");
        // Repeatedly add the same target point.
        for _ in 0..20 {
            hc.update(&target, &mut rng).expect("update");
        }
        let after = cosine_binary(&target, hc.centroid(0).expect("c0b")).expect("cos0b");
        assert!(
            after > before,
            "centroid did not move toward point: before={before} after={after}"
        );
        // After enough additions of one identical point, the centroid equals it.
        assert!((after - 1.0).abs() < 1e-6, "after={after}");
    }
}
