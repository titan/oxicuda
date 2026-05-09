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
