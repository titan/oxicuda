//! `oxicuda-ann` — Approximate Nearest Neighbor & vector-search primitives for OxiCUDA.
//!
//! Pure-Rust implementation of HNSW, IVF, Product Quantization, IVFPQ, LSH, k-NN graph,
//! and top-K selection, suitable for CPU simulation and PTX kernel generation for GPU execution.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-ann
//! ├── distance/    — L2, inner product distance metrics
//! ├── flat/        — Brute-force flat index (exact search baseline)
//! ├── hnsw/        — Hierarchical Navigable Small World graph (insert + search)
//! ├── hnsw_pq      — Compressed HNSW with per-node PQ codes + ADC scoring
//! ├── ivf/         — Inverted File Index (coarse quantizer + probing)
//! ├── ivfpq/       — IVF with Product Quantization re-ranking
//! ├── kmeans/      — k-Means clustering (used for IVF/PQ training)
//! ├── knn_graph/   — Brute-force and NN-Descent k-NN graph construction
//! ├── lsh/         — Random Projection LSH and MinHash
//! ├── ngt/         — NGT/ANNG approximate neighborhood graph (incremental build + ε-greedy search)
//! ├── pq/          — Product Quantization (train, encode, ADC)
//! ├── quantize/    — Scalar quantization utilities
//! ├── topk/        — Parallel top-K heap selection
//! ├── vamana       — DiskANN in-memory Vamana graph (α-pruned RobustPrune + greedy search)
//! ├── handle       — LcgRng (deterministic PRNG)
//! ├── error        — AnnError / AnnResult
//! └── ptx_kernels  — GPU PTX kernel strings (7 kernels × 6 SM versions)
//! ```

pub mod distance;
pub mod error;
pub mod flat;
pub mod handle;
pub mod hnsw;
pub mod hnsw_pq;
pub mod ivf;
pub mod ivfpq;
pub mod kmeans;
pub mod knn_graph;
pub mod lsh;
pub mod ngt;
pub mod pq;
pub mod ptx_kernels;
pub mod quantize;
pub mod topk;
pub mod vamana;

#[cfg(test)]
mod e2e_tests {
    use crate::distance::l2::l2_sq;
    use crate::flat::flat::FlatIndex;
    use crate::handle::LcgRng;
    use crate::hnsw::graph::HnswGraph;
    use crate::hnsw::insert::hnsw_insert;
    use crate::hnsw::search::hnsw_search;
    use crate::ivf::ivf::IvfIndex;
    use crate::kmeans::kmeans::KMeans;
    use crate::knn_graph::knn_graph::KnnGraph;
    use crate::lsh::minhash::MinHash;
    use crate::lsh::random_proj::RandomProjLsh;
    use crate::pq::adc::{adc_distance, build_adc_table};
    use crate::pq::encode::encode_vector;
    use crate::pq::train::train_pq;
    use crate::ptx_kernels::{
        hnsw_neighbor_eval_ptx, ip_distance_batch_ptx, ivf_assign_ptx, l2_distance_batch_ptx,
        lsh_random_proj_ptx, pq_adc_table_ptx, topk_select_ptx,
    };

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    fn rand_vecs(n: usize, dim: usize, rng: &mut LcgRng) -> Vec<f32> {
        (0..n * dim).map(|_| rng.next_f32()).collect()
    }

    // Test 1: FlatIndex inserts 10 vectors and finds nearest correctly (dist=0 for exact match)
    #[test]
    fn t01_flat_exact_match() {
        let mut rng = make_rng();
        let dim = 8;
        let mut idx = FlatIndex::new(dim);
        let vecs = rand_vecs(10, dim, &mut rng);
        for i in 0..10 {
            idx.add(&vecs[i * dim..(i + 1) * dim]);
        }
        // Query with the 3rd vector exactly
        let res = idx.search_l2(&vecs[3 * dim..4 * dim], 1).unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, 3, "should find id=3");
        assert!(res[0].1.abs() < 1e-6, "dist should be ~0");
    }

    // Test 2: FlatIndex top-k returns k results
    #[test]
    fn t02_flat_topk_count() {
        let mut rng = make_rng();
        let dim = 4;
        let mut idx = FlatIndex::new(dim);
        let vecs = rand_vecs(20, dim, &mut rng);
        for i in 0..20 {
            idx.add(&vecs[i * dim..(i + 1) * dim]);
        }
        let res = idx.search_l2(&vecs[0..dim], 5).unwrap();
        assert_eq!(res.len(), 5);
        // sorted ascending
        for w in res.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }

    // Test 3: k-means converges to 2 clear clusters in 1D
    #[test]
    fn t03_kmeans_two_clusters() {
        let mut rng = make_rng();
        // 50 points near 0, 50 points near 10
        let mut data = Vec::with_capacity(100);
        for _ in 0..50 {
            data.push(rng.next_f32() * 0.5);
        }
        for _ in 0..50 {
            data.push(10.0 + rng.next_f32() * 0.5);
        }
        let km = KMeans::fit(&data, 100, 1, 2, 100, &mut rng).unwrap();
        let c0 = km.centroids()[0];
        let c1 = km.centroids()[1];
        let lo = c0.min(c1);
        let hi = c0.max(c1);
        assert!(lo < 1.5, "low centroid={lo}");
        assert!(hi > 9.0, "high centroid={hi}");
    }

    // Test 4: PQ codebook trains without error
    #[test]
    fn t04_pq_train_no_error() {
        let mut rng = make_rng();
        let n = 200;
        let dim = 8;
        let m = 2;
        let ksub = 16;
        let data = rand_vecs(n, dim, &mut rng);
        let cb = train_pq(&data, n, dim, m, ksub, 20, &mut rng);
        assert!(cb.is_ok(), "PQ training failed: {:?}", cb.err());
        let cb = cb.unwrap();
        assert_eq!(cb.m, m);
        assert_eq!(cb.ksub, ksub);
    }

    // Test 5: PQ encode + ADC: distance estimate is finite and positive for non-identical vectors
    #[test]
    fn t05_pq_encode_adc_finite() {
        let mut rng = make_rng();
        let n = 200;
        let dim = 8;
        let m = 2;
        let ksub = 16;
        let data = rand_vecs(n, dim, &mut rng);
        let cb = train_pq(&data, n, dim, m, ksub, 20, &mut rng).unwrap();

        let query = rand_vecs(1, dim, &mut rng);
        let candidate = rand_vecs(1, dim, &mut rng);
        let code = encode_vector(&candidate, &cb);
        let table = build_adc_table(&query, &cb);
        let dist = adc_distance(&code, &table, m, ksub);
        assert!(dist.is_finite(), "dist={dist}");
        assert!(dist >= 0.0, "dist={dist}");
    }

    // Test 6: IVF train + add + search returns k results
    #[test]
    fn t06_ivf_search_returns_k() {
        let mut rng = make_rng();
        let n = 100;
        let dim = 8;
        let n_lists = 4;
        let data = rand_vecs(n, dim, &mut rng);

        let mut idx = IvfIndex::new(n_lists, dim);
        idx.train(&data, n, &mut rng).unwrap();
        for i in 0..n {
            idx.add(&data[i * dim..(i + 1) * dim], i);
        }

        let query = rand_vecs(1, dim, &mut rng);
        let res = idx.search(&query, 5, 2).unwrap();
        assert_eq!(res.len(), 5);
    }

    // Test 7: HNSW insert 100 vectors + search finds exact self (dist ≈ 0)
    #[test]
    fn t07_hnsw_exact_self() {
        let mut rng = make_rng();
        let dim = 8;
        let mut graph = HnswGraph::new(dim, 16, 200, 50);
        let data = rand_vecs(100, dim, &mut rng);
        for i in 0..100 {
            hnsw_insert(&mut graph, &data[i * dim..(i + 1) * dim], &mut rng);
        }

        // Search for the 50th vector
        let query = &data[50 * dim..51 * dim];
        let res = hnsw_search(&graph, query, 1).unwrap();
        assert!(!res.is_empty());
        // The found vector should have ~0 distance from query
        let found_id = res[0].0 as usize;
        let found_vec = &data[found_id * dim..(found_id + 1) * dim];
        let d = l2_sq(query, found_vec).unwrap();
        assert!(d < 1e-5, "dist={d} for found_id={found_id}");
    }

    // Test 8: HNSW search top-5 recall ≥ 80% vs brute-force ground truth
    #[test]
    fn t08_hnsw_recall() {
        let mut rng = make_rng();
        let dim = 8;
        let n = 50;
        let mut graph = HnswGraph::new(dim, 16, 200, 50);
        let data = rand_vecs(n, dim, &mut rng);
        for i in 0..n {
            hnsw_insert(&mut graph, &data[i * dim..(i + 1) * dim], &mut rng);
        }

        let k = 5;
        let mut total_recall = 0usize;
        let n_queries = 10;

        let queries = rand_vecs(n_queries, dim, &mut rng);
        for qi in 0..n_queries {
            let q = &queries[qi * dim..(qi + 1) * dim];

            // Brute-force top-k
            let mut bf: Vec<(usize, f32)> = (0..n)
                .map(|i| {
                    let v = &data[i * dim..(i + 1) * dim];
                    let d: f32 = q.iter().zip(v.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                    (i, d)
                })
                .collect();
            bf.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let ground_truth: std::collections::HashSet<usize> =
                bf.iter().take(k).map(|(i, _)| *i).collect();

            // HNSW search
            let hnsw_res = hnsw_search(&graph, q, k).unwrap();
            let hits = hnsw_res
                .iter()
                .filter(|(id, _)| ground_truth.contains(&(*id as usize)))
                .count();
            total_recall += hits;
        }

        let recall_at_k = total_recall as f32 / (n_queries * k) as f32;
        assert!(recall_at_k >= 0.8, "recall={recall_at_k:.2} < 0.8");
    }

    // Test 9: LSH: same vector hashes identically
    #[test]
    fn t09_lsh_deterministic() {
        let mut rng = make_rng();
        let lsh = RandomProjLsh::new(64, 8, &mut rng);
        let mut rng2 = make_rng();
        let v = rand_vecs(1, 8, &mut rng2);
        assert_eq!(lsh.hash(&v), lsh.hash(&v));
    }

    // Test 10: MinHash: Jaccard({1,2,3}, {1,2,3}) = 1.0 ± 0.05
    #[test]
    fn t10_minhash_identical_jaccard() {
        let mut rng = make_rng();
        let mh = MinHash::new(256, &mut rng);
        let s = vec![1u32, 2, 3];
        let sig1 = mh.hash(&s);
        let sig2 = mh.hash(&s);
        let j = MinHash::jaccard_estimate(&sig1, &sig2);
        assert!((j - 1.0).abs() <= 0.05, "j={j}");
    }

    // Test 11: NN-Descent graph: each node's nearest neighbor distance ≤ brute-force 5th NN
    #[test]
    fn t11_nndescent_quality() {
        let mut rng = make_rng();
        let n = 30;
        let dim = 4;
        let k = 3;
        let data = rand_vecs(n, dim, &mut rng);

        let bf = KnnGraph::build_brute(&data, n, dim, 5);
        let nnd = KnnGraph::build_nn_descent(&data, n, dim, k, 10, 0.001, &mut rng);

        for i in 0..n {
            let bf_5th = bf.neighbors(i).last().map_or(f32::INFINITY, |(_, d)| *d);
            let nnd_1st = nnd.neighbors(i).first().map_or(f32::INFINITY, |(_, d)| *d);
            assert!(
                nnd_1st <= bf_5th + 1e-5,
                "node={i} nnd_1st={nnd_1st} bf_5th={bf_5th}"
            );
        }
    }

    // Test 12: PTX kernels generate non-empty strings for all 6 SM versions
    #[test]
    fn t12_ptx_nonempty_all_sm() {
        for sm in [75u32, 80, 86, 89, 90, 100] {
            assert!(!l2_distance_batch_ptx(sm).is_empty(), "sm={sm}");
            assert!(!ip_distance_batch_ptx(sm).is_empty(), "sm={sm}");
            assert!(!pq_adc_table_ptx(sm).is_empty(), "sm={sm}");
            assert!(!hnsw_neighbor_eval_ptx(sm).is_empty(), "sm={sm}");
            assert!(!ivf_assign_ptx(sm).is_empty(), "sm={sm}");
            assert!(!lsh_random_proj_ptx(sm).is_empty(), "sm={sm}");
            assert!(!topk_select_ptx(sm).is_empty(), "sm={sm}");
        }
    }
}
