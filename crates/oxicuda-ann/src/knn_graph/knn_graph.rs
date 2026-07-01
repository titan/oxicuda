use crate::handle::LcgRng;

/// k-NN graph: each node stores its k nearest neighbors as `(neighbor_id, dist_sq)`.
pub struct KnnGraph {
    adj: Vec<Vec<(u32, f32)>>,
    pub k: usize,
}

impl KnnGraph {
    /// Brute-force O(n²) construction.
    pub fn build_brute(data: &[f32], n: usize, dim: usize, k: usize) -> KnnGraph {
        let actual_k = k.min(n.saturating_sub(1));
        let mut adj = Vec::with_capacity(n);

        for i in 0..n {
            let vi = &data[i * dim..(i + 1) * dim];
            let mut dists: Vec<(u32, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let vj = &data[j * dim..(j + 1) * dim];
                    let d: f32 = vi
                        .iter()
                        .zip(vj.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    (j as u32, d)
                })
                .collect();
            dists.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            dists.truncate(actual_k);
            adj.push(dists);
        }

        KnnGraph { adj, k: actual_k }
    }

    /// NN-Descent approximate k-NN graph construction.
    ///
    /// Iterates: for each node, examines neighbors-of-neighbors as new candidates.
    /// Converges when the fraction of improved edges falls below `delta`.
    pub fn build_nn_descent(
        data: &[f32],
        n: usize,
        dim: usize,
        k: usize,
        max_iter: usize,
        delta: f32,
        rng: &mut LcgRng,
    ) -> KnnGraph {
        let actual_k = k.min(n.saturating_sub(1));

        // Initialize with random neighbors
        let mut adj: Vec<Vec<(u32, f32)>> = (0..n)
            .map(|i| {
                let vi = &data[i * dim..(i + 1) * dim];
                let mut nbrs = Vec::with_capacity(actual_k);
                let mut used = vec![false; n];
                used[i] = true;
                while nbrs.len() < actual_k {
                    let j = rng.next_u32() as usize % n;
                    if !used[j] {
                        used[j] = true;
                        let vj = &data[j * dim..(j + 1) * dim];
                        let d: f32 = vi
                            .iter()
                            .zip(vj.iter())
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum();
                        nbrs.push((j as u32, d));
                    }
                }
                nbrs.sort_unstable_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                nbrs
            })
            .collect();

        let total_edges = (n * actual_k) as f32;

        for _ in 0..max_iter {
            let mut improvements = 0usize;

            // Collect candidates from neighbors-of-neighbors
            let old_adj = adj.clone();

            for i in 0..n {
                let vi = &data[i * dim..(i + 1) * dim];

                // Gather candidates: neighbors of neighbors of i
                let mut candidates: Vec<u32> = Vec::new();
                for &(nbr, _) in &old_adj[i] {
                    for &(nn, _) in &old_adj[nbr as usize] {
                        if nn != i as u32 {
                            candidates.push(nn);
                        }
                    }
                }

                // Deduplicate
                candidates.sort_unstable();
                candidates.dedup();

                let worst_dist = adj[i].last().map_or(f32::INFINITY, |(_, d)| *d);

                for cand in candidates {
                    let cand_u = cand as usize;
                    if cand_u == i {
                        continue;
                    }
                    let vc = &data[cand_u * dim..(cand_u + 1) * dim];
                    let d: f32 = vi
                        .iter()
                        .zip(vc.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();

                    if d < worst_dist || adj[i].len() < actual_k {
                        // Check if already in adj[i]
                        let already = adj[i].iter().any(|(id, _)| *id == cand);
                        if !already {
                            adj[i].push((cand, d));
                            adj[i].sort_unstable_by(|a, b| {
                                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            adj[i].truncate(actual_k);
                            improvements += 1;
                        }
                    }
                }
            }

            if improvements as f32 / total_edges < delta {
                break;
            }
        }

        KnnGraph { adj, k: actual_k }
    }

    #[must_use]
    pub fn neighbors(&self, id: usize) -> &[(u32, f32)] {
        if id < self.adj.len() {
            &self.adj[id]
        } else {
            &[]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Brute-force k nearest neighbors of point `i` in `data` (self excluded), sorted by dist_sq.
    fn brute_knn_ids(data: &[f32], n: usize, dim: usize, i: usize, k: usize) -> Vec<u32> {
        let vi = &data[i * dim..(i + 1) * dim];
        let mut dists: Vec<(f32, u32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let vj = &data[j * dim..(j + 1) * dim];
                let d: f32 = vi
                    .iter()
                    .zip(vj.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                (d, j as u32)
            })
            .collect();
        dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k);
        dists.into_iter().map(|(_, id)| id).collect()
    }

    #[test]
    fn brute_k_neighbor_count_exact() {
        // n=5, k=2 — each node should have exactly 2 neighbors (n-1 >= k).
        let data: Vec<f32> = (0..5).flat_map(|i| [i as f32, 0.0_f32]).collect();
        let g = KnnGraph::build_brute(&data, 5, 2, 2);
        assert_eq!(g.k, 2);
        for i in 0..5 {
            assert_eq!(
                g.neighbors(i).len(),
                2,
                "node {i} should have exactly 2 neighbors"
            );
        }
    }

    #[test]
    fn brute_no_self_neighbor() {
        let data: Vec<f32> = (0..8).flat_map(|i| [i as f32, (8 - i) as f32]).collect();
        let g = KnnGraph::build_brute(&data, 8, 2, 3);
        for i in 0..8 {
            assert!(
                g.neighbors(i).iter().all(|&(id, _)| id != i as u32),
                "node {i} must not appear as its own neighbor"
            );
        }
    }

    #[test]
    fn brute_matches_ground_truth_line() {
        // 6 collinear points; nearest neighbors are analytically obvious.
        // data:  0:[0,0]  1:[1,0]  2:[2,0]  3:[10,0]  4:[11,0]  5:[12,0]
        let data: Vec<f32> = vec![
            0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 10.0, 0.0, 11.0, 0.0, 12.0, 0.0,
        ];
        let g = KnnGraph::build_brute(&data, 6, 2, 2);

        // Node 0: nearest are 1 (dist_sq=1) then 2 (dist_sq=4)
        let nb0: Vec<u32> = g.neighbors(0).iter().map(|&(id, _)| id).collect();
        assert_eq!(nb0, vec![1, 2], "node 0 neighbors mismatch: {nb0:?}");

        // Node 3: nearest are 4 (dist_sq=1) then 5 (dist_sq=4)
        let nb3: Vec<u32> = g.neighbors(3).iter().map(|&(id, _)| id).collect();
        assert_eq!(nb3, vec![4, 5], "node 3 neighbors mismatch: {nb3:?}");
    }

    #[test]
    fn brute_k_saturates_at_n_minus_one() {
        // n=3, k=5 => actual_k = min(5, 2) = 2; g.k should be 2.
        let data: Vec<f32> = vec![0.0_f32, 0.0, 1.0, 0.0, 2.0, 0.0];
        let g = KnnGraph::build_brute(&data, 3, 2, 5);
        assert_eq!(g.k, 2, "k must saturate at n-1=2");
        for i in 0..3 {
            assert_eq!(
                g.neighbors(i).len(),
                2,
                "node {i} should have 2 neighbors (saturated)"
            );
        }
    }

    #[test]
    fn neighbors_out_of_range_returns_empty() {
        let data: Vec<f32> = vec![0.0_f32, 0.0, 1.0, 0.0];
        let g = KnnGraph::build_brute(&data, 2, 2, 1);
        assert!(
            g.neighbors(99).is_empty(),
            "out-of-range id must return empty slice"
        );
    }

    #[test]
    fn brute_distances_finite_and_nonneg() {
        let data: Vec<f32> = (0..10)
            .flat_map(|i| [i as f32 * 1.7, i as f32 * -0.9])
            .collect();
        let g = KnnGraph::build_brute(&data, 10, 2, 3);
        for i in 0..10 {
            for &(_, d) in g.neighbors(i) {
                assert!(d.is_finite(), "node {i}: distance is not finite");
                assert!(d >= 0.0, "node {i}: distance is negative ({d})");
            }
        }
    }

    #[test]
    fn brute_matches_per_node_ground_truth() {
        // 8 points on a 1D line; verify every node's neighbors match brute_knn_ids.
        let n = 8_usize;
        let dim = 1_usize;
        let k = 3_usize;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let g = KnnGraph::build_brute(&data, n, dim, k);
        for i in 0..n {
            let mut expected = brute_knn_ids(&data, n, dim, i, k);
            let mut got: Vec<u32> = g.neighbors(i).iter().map(|&(id, _)| id).collect();
            expected.sort_unstable();
            got.sort_unstable();
            assert_eq!(got, expected, "node {i}: got={got:?} expected={expected:?}");
        }
    }

    #[test]
    fn nn_descent_no_self_neighbor() {
        let data: Vec<f32> = (0..10).flat_map(|i| [i as f32, 0.0_f32]).collect();
        let mut rng = LcgRng::new(101);
        let g = KnnGraph::build_nn_descent(&data, 10, 2, 3, 10, 0.01, &mut rng);
        for i in 0..10 {
            assert!(
                g.neighbors(i).iter().all(|&(id, _)| id != i as u32),
                "node {i} must not appear as its own neighbor in NN-Descent"
            );
        }
    }

    #[test]
    fn nn_descent_k_neighbor_count_correct() {
        // Structural invariant: every node has exactly actual_k out-neighbors after build.
        let n = 10_usize;
        let dim = 2_usize;
        let k = 2_usize;
        let data: Vec<f32> = (0..n).flat_map(|i| [i as f32, 0.0_f32]).collect();
        let mut rng = LcgRng::new(77);
        let g = KnnGraph::build_nn_descent(&data, n, dim, k, 30, 0.001, &mut rng);
        // actual_k = min(k, n-1) = 2 for n=10
        assert_eq!(g.k, 2);
        for i in 0..n {
            assert_eq!(
                g.neighbors(i).len(),
                2,
                "node {i} should have exactly 2 neighbors after NN-Descent"
            );
        }
    }

    #[test]
    fn nn_descent_achieves_meaningful_recall() {
        // NN-Descent is approximate; we assert recall >= 50% (typically much higher).
        // With n=10 collinear points, k=2 and 30 iterations, the algorithm achieves
        // high recall on most nodes even with an adversarial random seed.
        let n = 10_usize;
        let dim = 2_usize;
        let k = 2_usize;
        let data: Vec<f32> = (0..n).flat_map(|i| [i as f32, 0.0_f32]).collect();
        let mut rng = LcgRng::new(77);
        let approx = KnnGraph::build_nn_descent(&data, n, dim, k, 30, 0.001, &mut rng);
        let brute = KnnGraph::build_brute(&data, n, dim, k);

        let mut hits = 0_usize;
        let mut total = 0_usize;
        for i in 0..n {
            let brute_set: std::collections::HashSet<u32> =
                brute.neighbors(i).iter().map(|&(id, _)| id).collect();
            for &(id, _) in approx.neighbors(i) {
                total += 1;
                if brute_set.contains(&id) {
                    hits += 1;
                }
            }
        }
        let recall = hits as f64 / total.max(1) as f64;
        assert!(
            recall >= 0.5,
            "NN-Descent recall {recall:.2} is below 0.5 (hits={hits}/{total})"
        );
    }
}
