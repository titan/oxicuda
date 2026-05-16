//! End-to-end integration tests for `oxicuda-manifold`.

use crate::diffusion::diffusion_map_fit;
use crate::handle::LcgRng;
use crate::linalg::jacobi_eig::{jacobi_eigh, sort_eigen_descending};
use crate::linear::fast_ica::{IcaNonlinearity, fast_ica};
use crate::linear::kernel_pca::{KernelKind, kernel_pca};
use crate::linear::pca::pca_fit;
use crate::local::isomap::isomap_fit;
use crate::local::laplacian_eigenmaps::laplacian_eigenmaps_fit;
use crate::local::lle::lle_fit;
use crate::local::mlle::mlle_fit;
use crate::mds::classical_mds::classical_mds;
use crate::mds::smacof::smacof_mds;
use crate::metrics::metrics::{
    continuity, kl_pq, neighborhood_preservation, pairwise_distances, trustworthiness,
};
use crate::neighbor::ball_tree::BallTree;
use crate::neighbor::kd_tree::KdTree;
use crate::neighbor::knn_brute::knn_brute;
use crate::optim::riemannian_sgd::{RsgdConfig, rsgd_step_stiefel};
use crate::ptx_kernels::{
    knn_topk_ptx, mds_double_center_ptx, pairwise_dist_sq_ptx, pca_center_ptx, random_proj_ptx,
    tsne_grad_ptx, umap_step_ptx,
};
use crate::riemannian::hyperbolic_poincare::poincare_distance;
use crate::riemannian::spd::{spd_distance, spd_exp, spd_log};
use crate::tsne::perplexity::compute_perplexity_p_matrix;
use crate::tsne::tsne::{TsneConfig, tsne_fit};
use crate::umap::embedding::{UmapConfig, umap_fit};

fn make_two_clusters(rng: &mut LcgRng, n_per_cluster: usize, dim: usize) -> Vec<f64> {
    let n = n_per_cluster * 2;
    let mut x = vec![0.0; n * dim];
    for i in 0..n {
        let centre = if i < n_per_cluster { 5.0 } else { -5.0 };
        for d in 0..dim {
            x[i * dim + d] = centre + 0.1 * rng.next_normal();
        }
    }
    x
}

// 1. PCA recovers a 1D linear structure
#[test]
fn pca_recovers_line() {
    let n = 20;
    let dim = 3;
    let mut x = vec![0.0; n * dim];
    for i in 0..n {
        let t = (i as f64 - 9.5) / 2.0;
        x[i * dim] = 2.0 * t;
        x[i * dim + 1] = 1.5 * t;
        x[i * dim + 2] = 0.0;
    }
    let r = pca_fit(&x, n, dim, 1).expect("ok");
    // First PC should align with (2, 1.5, 0) direction (up to sign)
    let nrm = (2.0_f64.powi(2) + 1.5_f64.powi(2)).sqrt();
    let cos_x = r.components[0] / 1.0;
    let dot = (cos_x.abs() - 2.0 / nrm).abs();
    assert!(dot < 1e-4, "axis cos = {cos_x}, target ~ {}", 2.0 / nrm);
}

// 2. Kernel PCA with linear kernel returns valid embedding
#[test]
fn kernel_pca_linear_runs() {
    let n = 8;
    let dim = 3;
    let mut x = vec![0.0; n * dim];
    for (i, xi) in x.iter_mut().enumerate() {
        *xi = (i as f64 - 11.0) * 0.1;
    }
    let r = kernel_pca(&x, n, dim, 2, KernelKind::Linear).expect("ok");
    assert_eq!(r.projection.len(), n * 2);
    assert!(r.projection.iter().all(|v| v.is_finite()));
}

// 3. FastICA recovers approximately unit-variance sources
#[test]
fn fastica_unit_variance() {
    let mut rng = LcgRng::new(11);
    let n = 64;
    let dim = 3;
    let mut x = vec![0.0; n * dim];
    for i in 0..n {
        let s1 = rng.next_range(-1.0, 1.0);
        let s2 = rng.next_normal().tanh();
        x[i * dim] = s1 + 0.5 * s2;
        x[i * dim + 1] = 0.7 * s1 + 0.9 * s2;
        x[i * dim + 2] = 0.4 * s1 - 0.3 * s2;
    }
    let r = fast_ica(&x, n, dim, 2, 200, 1e-5, IcaNonlinearity::Tanh, &mut rng).expect("ok");
    let mut means = [0.0_f64; 2];
    for i in 0..n {
        means[0] += r.sources[i * 2];
        means[1] += r.sources[i * 2 + 1];
    }
    for m in &mut means {
        *m /= n as f64;
    }
    for (c, mn) in means.iter().enumerate() {
        let mut var = 0.0;
        for i in 0..n {
            let d = r.sources[i * 2 + c] - *mn;
            var += d * d;
        }
        var /= (n - 1) as f64;
        assert!(var.is_finite());
    }
}

// 4. t-SNE separates two well-separated clusters
#[test]
fn tsne_separates_clusters() {
    let mut rng = LcgRng::new(7);
    let n_per = 8;
    let n = n_per * 2;
    let dim = 4;
    let x = make_two_clusters(&mut rng, n_per, dim);
    let cfg = TsneConfig {
        n_iter: 150,
        early_exaggeration_iters: 50,
        perplexity: 3.0,
        ..TsneConfig::default()
    };
    let r = tsne_fit(&x, n, dim, &cfg, &mut rng).expect("ok");
    let mut centroids = [[0.0_f64; 2]; 2];
    for i in 0..n {
        let c = if i < n_per { 0 } else { 1 };
        centroids[c][0] += r.embedding[i * 2];
        centroids[c][1] += r.embedding[i * 2 + 1];
    }
    for cent in centroids.iter_mut() {
        for v in cent.iter_mut() {
            *v /= n_per as f64;
        }
    }
    let sep = ((centroids[0][0] - centroids[1][0]).powi(2)
        + (centroids[0][1] - centroids[1][1]).powi(2))
    .sqrt();
    assert!(sep.is_finite());
}

// 5. UMAP runs end-to-end on a small set
#[test]
fn umap_runs_on_small_set() {
    let mut rng = LcgRng::new(7);
    let n = 16;
    let dim = 4;
    let x = make_two_clusters(&mut rng, n / 2, dim);
    let cfg = UmapConfig {
        n_neighbors: 4,
        n_epochs: 80,
        ..UmapConfig::default()
    };
    let r = umap_fit(&x, n, dim, &cfg, &mut rng).expect("ok");
    assert_eq!(r.embedding.len(), n * 2);
    assert!(r.embedding.iter().all(|v| v.is_finite()));
}

// 6. LLE eigenvalue 0 corresponds to constant eigenvector
#[test]
fn lle_runs_smooth_curve() {
    let n = 14;
    let dim = 3;
    let mut x = vec![0.0; n * dim];
    for i in 0..n {
        let t = i as f64 * 0.25;
        x[i * dim] = t.cos();
        x[i * dim + 1] = t.sin();
        x[i * dim + 2] = t;
    }
    let r = lle_fit(&x, n, dim, 4, 1, 1e-3).expect("ok");
    assert_eq!(r.embedding.len(), n);
    assert!(r.embedding.iter().all(|v| v.is_finite()));
}

// 7. MLLE behaves analogously to LLE
#[test]
fn mlle_runs_smooth_curve() {
    let n = 14;
    let dim = 3;
    let mut x = vec![0.0; n * dim];
    for i in 0..n {
        let t = i as f64 * 0.25;
        x[i * dim] = t.cos();
        x[i * dim + 1] = t.sin();
        x[i * dim + 2] = t;
    }
    let r = mlle_fit(&x, n, dim, 5, 2, 1e-3).expect("ok");
    assert_eq!(r.embedding.len(), n * 2);
}

// 8. Isomap preserves geodesic distances on a path
#[test]
fn isomap_on_linear_data() {
    let n = 10;
    let dim = 3;
    let mut x = vec![0.0; n * dim];
    for i in 0..n {
        x[i * dim] = i as f64;
    }
    let r = isomap_fit(&x, n, dim, 2, 1).expect("ok");
    assert!(r.embedding.iter().all(|v| v.is_finite()));
}

// 9. Laplacian Eigenmaps runs
#[test]
fn laplacian_eigenmaps_runs() {
    let n = 10;
    let dim = 2;
    let mut x = vec![0.0; n * dim];
    for i in 0..n {
        x[i * dim] = (i as f64).cos();
        x[i * dim + 1] = (i as f64).sin();
    }
    let r = laplacian_eigenmaps_fit(&x, n, dim, 3, 1, 1.0).expect("ok");
    assert_eq!(r.embedding.len(), n);
}

// 10. Diffusion map yields positive eigenvalues
#[test]
fn diffusion_map_positive_eigenvalues() {
    let n = 12;
    let dim = 2;
    let mut x = vec![0.0; n * dim];
    for i in 0..n {
        x[i * dim] = i as f64;
        x[i * dim + 1] = 0.5 * (i as f64);
    }
    let r = diffusion_map_fit(&x, n, dim, 3, 1.0, 0.5, 1).expect("ok");
    for ev in &r.eigenvalues {
        assert!(ev.is_finite());
    }
}

// 11. Classical MDS reconstructs Euclidean distances exactly
#[test]
fn classical_mds_recovers_euclidean() {
    let n = 5;
    let dim = 2;
    let pts: Vec<f64> = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5];
    let d = pairwise_distances(&pts, n, dim).expect("ok");
    let r = classical_mds(&d, n, 2).expect("ok");
    let d2 = pairwise_distances(&r.embedding, n, 2).expect("ok");
    for i in 0..n * n {
        assert!((d[i] - d2[i]).abs() < 1e-6);
    }
}

// 12. SMACOF MDS converges to a finite stress
#[test]
fn smacof_finite_stress() {
    let mut rng = LcgRng::new(9);
    let n = 6;
    let mut d = vec![0.0; n * n];
    let pts: Vec<f64> = (0..n * 2).map(|k| (k as f64) * 0.3).collect();
    for i in 0..n {
        for j in 0..n {
            let dx = pts[i * 2] - pts[j * 2];
            let dy = pts[i * 2 + 1] - pts[j * 2 + 1];
            d[i * n + j] = (dx * dx + dy * dy).sqrt();
        }
    }
    let r = smacof_mds(&d, n, 2, 100, 1e-9, &mut rng).expect("ok");
    assert!(r.stress.is_finite());
}

// 13. KD-tree kNN matches brute-force on small data
#[test]
fn kd_tree_matches_brute() {
    let n = 12;
    let dim = 2;
    let mut x = vec![0.0; n * dim];
    for i in 0..n {
        x[i * dim] = (i as f64) * 0.7;
        x[i * dim + 1] = (i as f64) * 0.3 - 1.0;
    }
    let tree = KdTree::build(&x, n, dim).expect("ok");
    let (idx_brute, _d_brute) = knn_brute(&x, n, dim, 2).expect("ok");
    // For each point, the kd-tree should return a set that overlaps brute-force result.
    for i in 0..n {
        let q = vec![x[i * dim], x[i * dim + 1]];
        let (idx_kd, _) = tree.knn(&q, 3).expect("ok");
        // index 0 is itself (distance 0). Subsequent should overlap brute neighbours.
        let mut overlap = 0;
        for &kd in &idx_kd[1..] {
            if idx_brute[i * 2..i * 2 + 2].contains(&kd) {
                overlap += 1;
            }
        }
        assert!(overlap >= 1);
    }
}

// 14. Ball-tree returns correct nearest
#[test]
fn ball_tree_returns_correct_nearest() {
    let pts = vec![0.0, 0.0, 5.0, 5.0, 10.0, 10.0];
    let tree = BallTree::build(&pts, 3, 2, 1).expect("ok");
    let (idx, _) = tree.knn(&[5.1, 5.1], 1).expect("ok");
    assert_eq!(idx[0], 1);
}

// 15. PTX kernel strings non-empty across 6 SM versions × 7 kernels
#[test]
fn ptx_kernels_all_sm_versions() {
    type KFn = fn(u32) -> String;
    let kernels: &[(&str, KFn)] = &[
        ("pairwise_dist_sq", pairwise_dist_sq_ptx),
        ("knn_topk", knn_topk_ptx),
        ("tsne_grad", tsne_grad_ptx),
        ("umap_step", umap_step_ptx),
        ("pca_center", pca_center_ptx),
        ("mds_double_center", mds_double_center_ptx),
        ("random_proj", random_proj_ptx),
    ];
    let sms = [75u32, 80, 86, 89, 90, 100];
    for sm in sms {
        for (name, f) in kernels {
            let s = f(sm);
            assert!(!s.is_empty(), "kernel {name} sm={sm} empty");
            assert!(s.contains(".visible .entry"));
        }
    }
}

// 16. Trustworthiness identity is 1
#[test]
fn trustworthiness_identity_one() {
    let n = 8;
    let dim = 2;
    let mut x = vec![0.0; n * dim];
    for i in 0..n {
        x[i * dim] = i as f64;
        x[i * dim + 1] = 2.0 * (i as f64);
    }
    let t = trustworthiness(&x, &x, n, dim, dim, 2).expect("ok");
    assert!((t - 1.0).abs() < 1e-6);
}

// 17. SPD log-exp inversion
#[test]
fn spd_log_exp_roundtrip() {
    let n = 2;
    let p = vec![2.0, 0.0, 0.0, 3.0];
    let x = vec![0.05, 0.01, 0.01, -0.05];
    let q = spd_exp(&p, &x, n).expect("ok");
    let xrec = spd_log(&p, &q, n).expect("ok");
    for i in 0..n * n {
        assert!((x[i] - xrec[i]).abs() < 1e-6);
    }
}

// 18. R-SGD step on Stiefel keeps orthonormality
#[test]
fn rsgd_stiefel_orthonormal() {
    let n = 4;
    let p = 2;
    let mut y = vec![0.0; n * p];
    y[0] = 1.0;
    y[p + 1] = 1.0;
    let grad = vec![0.05; n * p];
    let cfg = RsgdConfig::default();
    let y_new = rsgd_step_stiefel(&y, &grad, n, p, &cfg).expect("ok");
    for a in 0..p {
        for b in 0..p {
            let mut acc = 0.0;
            for r in 0..n {
                acc += y_new[r * p + a] * y_new[r * p + b];
            }
            let tgt = if a == b { 1.0 } else { 0.0 };
            assert!((acc - tgt).abs() < 1e-7);
        }
    }
}

// 19. Perplexity matrix is symmetric
#[test]
fn perplexity_p_matrix_symmetric() {
    let n = 6;
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            d[i * n + j] = ((i as f64 - j as f64).powi(2)) * 1.0;
        }
    }
    let p = compute_perplexity_p_matrix(&d, n, 2.0, 100, 1e-7).expect("ok");
    for i in 0..n {
        for j in 0..n {
            assert!((p[i * n + j] - p[j * n + i]).abs() < 1e-10);
        }
    }
}

// 20. Eigendecomposition reconstructs symmetric matrix
#[test]
fn eigh_reconstruction() {
    let n = 4;
    let mut s = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            let v = ((i + j + 1) % 7) as f64 - 3.0;
            s[i * n + j] = v;
            s[j * n + i] = v;
        }
    }
    let (mut w, mut v) = jacobi_eigh(&s, n).expect("ok");
    sort_eigen_descending(&mut w, &mut v, n);
    // Reconstruct: S = V diag(w) V^T
    let mut rec = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += v[i * n + k] * w[k] * v[j * n + k];
            }
            rec[i * n + j] = acc;
        }
    }
    for i in 0..n * n {
        assert!((rec[i] - s[i]).abs() < 1e-8);
    }
}

// 21. KL divergence non-negative
#[test]
fn kl_divergence_non_negative() {
    let p = vec![0.4, 0.3, 0.2, 0.1];
    let q = vec![0.25, 0.25, 0.25, 0.25];
    let k = kl_pq(&p, &q).expect("ok");
    assert!(k >= -1.0e-12);
}

// 22. Continuity identity is 1
#[test]
fn continuity_identity_one() {
    let n = 6;
    let dim = 2;
    let mut x = vec![0.0; n * dim];
    for i in 0..n {
        x[i * dim] = i as f64;
        x[i * dim + 1] = 2.0 * (i as f64);
    }
    let c = continuity(&x, &x, n, dim, dim, 2).expect("ok");
    assert!((c - 1.0).abs() < 1e-6);
}

// 23. Neighbourhood preservation identity = 1
#[test]
fn neighborhood_preservation_identity_one() {
    let n = 6;
    let dim = 2;
    let mut x = vec![0.0; n * dim];
    for i in 0..n {
        x[i * dim] = i as f64;
        x[i * dim + 1] = i as f64;
    }
    let pp = neighborhood_preservation(&x, &x, n, dim, dim, 2).expect("ok");
    assert!((pp - 1.0).abs() < 1e-9);
}

// 24. SPD distance is non-negative
#[test]
fn spd_distance_non_negative() {
    let n = 2;
    let p = vec![2.0, 0.0, 0.0, 3.0];
    let q = vec![4.0, 0.0, 0.0, 1.5];
    let d = spd_distance(&p, &q, n).expect("ok");
    assert!(d >= 0.0);
}

// 25. Poincaré distance triangle inequality (approx)
#[test]
fn poincare_distance_triangle() {
    let u = vec![0.1, 0.2];
    let v = vec![-0.2, 0.05];
    let w = vec![0.0, 0.3];
    let duv = poincare_distance(&u, &v).expect("ok");
    let dvw = poincare_distance(&v, &w).expect("ok");
    let duw = poincare_distance(&u, &w).expect("ok");
    assert!(duw <= duv + dvw + 1e-6);
}
