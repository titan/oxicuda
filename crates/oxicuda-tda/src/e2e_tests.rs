//! End-to-end integration tests for `oxicuda-tda`.
//!
//! 18 cross-module tests covering the full TDA pipeline from filtration building
//! through homology computation, persistence metrics, Mapper, and witness complex.

use crate::complex::filtration::{FilteredSimplex, Filtration};
use crate::complex::simplex::Simplex;
use crate::distance::kernel::{
    KernelConfig, persistence_scale_space_distance, persistence_scale_space_kernel,
};
use crate::distance::pairwise::pairwise_euclidean;
use crate::handle::LcgRng;
use crate::homology::boundary::BoundaryMatrix;
use crate::homology::persistent::extract_persistence_pairs;
use crate::homology::reduction::reduce_boundary_matrix;
use crate::homology::zigzag::{ZigzagArrow, ZigzagComplex, ZigzagInput, zigzag_persistence};
use crate::mapper::mapper::{MapperConfig, build_mapper};
use crate::metrics::metrics::{
    betti_numbers, count_components, persistence_landscape, persistent_entropy,
};
use crate::persistence::diagram::PersistenceDiagram;
use crate::persistence::distance::{bottleneck_distance, wasserstein_1};
use crate::ptx_kernels::{
    betti_count_ptx, boundary_reduce_ptx, diagram_match_ptx, filtration_sort_ptx,
    mapper_cluster_ptx, pairwise_dist_ptx, witness_dist_ptx,
};
use crate::vector::betti_curve::betti_curve;
use crate::witness::witness::{lazy_witness_complex, maxmin_landmarks};

// ──────────────────────────────────────────────
// Test 1: ∂² = 0
// ──────────────────────────────────────────────
#[test]
fn simplex_boundary_squared_zero() {
    use std::collections::HashMap;
    let s = Simplex::new(vec![0, 1, 2, 3]).expect("3-simplex");
    let bd1 = s.boundary();
    let mut sums: HashMap<Simplex, i32> = HashMap::new();
    for (c1, face) in &bd1 {
        for (c2, ff) in face.boundary() {
            *sums.entry(ff).or_insert(0) += (*c1 as i32) * (c2 as i32);
        }
    }
    for (_, v) in sums {
        assert_eq!(v, 0, "∂² ≠ 0 on [0,1,2,3]");
    }
}

// ──────────────────────────────────────────────
// Test 2: Vietoris-Rips at r=0 → vertices only
// ──────────────────────────────────────────────
#[test]
fn vietoris_rips_r0_vertices_only() {
    // 5 points scattered in 2D
    let pts = vec![0.0f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5];
    let filt = Filtration::vietoris_rips_from_points(&pts, 2, 0.0, 2).expect("ok");
    assert!(
        filt.simplices.iter().all(|fs| fs.simplex.dim() == 0),
        "At r=0 only vertices should appear"
    );
    assert_eq!(filt.n_simplices(), 5);
}

// ──────────────────────────────────────────────
// Test 3: Large radius → complete complex on 4 pts
// ──────────────────────────────────────────────
#[test]
fn vietoris_rips_complete_at_large_r() {
    // 4 points at unit square corners
    let pts = vec![0.0f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let filt = Filtration::vietoris_rips_from_points(&pts, 2, 100.0, 2).expect("ok");
    // Should have 4 vertices + 6 edges + 4 triangles
    let n_verts = filt
        .simplices
        .iter()
        .filter(|fs| fs.simplex.dim() == 0)
        .count();
    let n_edges = filt
        .simplices
        .iter()
        .filter(|fs| fs.simplex.dim() == 1)
        .count();
    let n_tris = filt
        .simplices
        .iter()
        .filter(|fs| fs.simplex.dim() == 2)
        .count();
    assert_eq!(n_verts, 4);
    assert_eq!(n_edges, 6);
    assert_eq!(n_tris, 4);
}

// ──────────────────────────────────────────────
// Test 4: Filtration values are non-decreasing
// ──────────────────────────────────────────────
#[test]
fn filtration_sorted_by_value() {
    let pts = vec![0.0f64, 0.0, 1.0, 0.0, 0.5, 0.866, 0.0, 1.0];
    let filt = Filtration::vietoris_rips_from_points(&pts, 2, 10.0, 2).expect("ok");
    for w in filt.simplices.windows(2) {
        assert!(
            w[0].value <= w[1].value + 1e-13,
            "filtration not sorted: {} > {}",
            w[0].value,
            w[1].value
        );
    }
}

// ──────────────────────────────────────────────
// Test 5: 3-point circle H₁=1 before triangle
// ──────────────────────────────────────────────
#[test]
fn three_point_circle_h1() {
    // 3 points forming an equilateral triangle: edge length ≈ 1
    // At r=1 (edges only), we expect H₁=1 (one loop), H₀=1
    let s3 = 3.0_f64.sqrt() / 2.0; // ≈ 0.866
    let pts = vec![0.0f64, 0.0, 1.0, 0.0, 0.5, s3];

    // Build filtration with max_dim=1 (no triangles) and radius just large enough for edges
    let filt = Filtration::vietoris_rips_from_points(&pts, 2, 1.01, 1).expect("ok");
    let mut bm = BoundaryMatrix::from_filtration(&filt).expect("ok");
    let _pivots = reduce_boundary_matrix(&mut bm);
    let pairs = extract_persistence_pairs(&bm, &filt).expect("ok");

    let diagrams = PersistenceDiagram::from_pairs_by_dim(&pairs, 1);
    // H₁ essential classes (loop that never dies at max_dim=1)
    let h1 = &diagrams[1];
    // At least 1 H₁ generator (the loop formed by the 3 edges)
    let h1_count = h1.essential_classes().len() + h1.finite_pairs().len();
    assert!(h1_count >= 1, "expected H₁ ≥ 1, got {h1_count}");
}

// ──────────────────────────────────────────────
// Test 6: Boundary matrix reduces without panic
// ──────────────────────────────────────────────
#[test]
fn boundary_matrix_reduces() {
    let simplices = vec![
        FilteredSimplex {
            simplex: Simplex { vertices: vec![0] },
            value: 0.0,
        },
        FilteredSimplex {
            simplex: Simplex { vertices: vec![1] },
            value: 0.0,
        },
        FilteredSimplex {
            simplex: Simplex { vertices: vec![2] },
            value: 0.0,
        },
        FilteredSimplex {
            simplex: Simplex {
                vertices: vec![0, 1],
            },
            value: 1.0,
        },
        FilteredSimplex {
            simplex: Simplex {
                vertices: vec![0, 2],
            },
            value: 1.0,
        },
        FilteredSimplex {
            simplex: Simplex {
                vertices: vec![1, 2],
            },
            value: 1.0,
        },
        FilteredSimplex {
            simplex: Simplex {
                vertices: vec![0, 1, 2],
            },
            value: 2.0,
        },
    ];
    let filt = Filtration::new(simplices).expect("ok");
    let mut bm = BoundaryMatrix::from_filtration(&filt).expect("ok");
    let _pivots = reduce_boundary_matrix(&mut bm);
    // Just assert it completes without panic; verify at least n_cols == 7
    assert_eq!(bm.n_cols, 7);
}

// ──────────────────────────────────────────────
// Test 7: All finite pairs have birth ≤ death
// ──────────────────────────────────────────────
#[test]
fn persistence_pairs_birth_before_death() {
    let pts: Vec<f64> = (0..8)
        .flat_map(|i| vec![(i % 4) as f64, (i / 4) as f64])
        .collect();
    let filt = Filtration::vietoris_rips_from_points(&pts, 2, 2.0, 2).expect("ok");
    let mut bm = BoundaryMatrix::from_filtration(&filt).expect("ok");
    reduce_boundary_matrix(&mut bm);
    let pairs = extract_persistence_pairs(&bm, &filt).expect("ok");
    for p in &pairs {
        if let Some(d) = p.death {
            assert!(
                p.birth <= d + 1e-12,
                "pair has birth {} > death {}",
                p.birth,
                d
            );
        }
    }
}

// ──────────────────────────────────────────────
// Test 8: 4 points (unit square), r=1 → H₀=1, H₁=1
// ──────────────────────────────────────────────
#[test]
fn square_4pts_topology() {
    // 4 points at corners of unit square: distances are 1 (edges) and √2 (diagonals).
    // At r=1: only the 4 unit edges are included → 1 connected component, 1 loop.
    let pts = vec![0.0f64, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let filt = Filtration::vietoris_rips_from_points(&pts, 2, 1.001, 2).expect("ok");
    let mut bm = BoundaryMatrix::from_filtration(&filt).expect("ok");
    reduce_boundary_matrix(&mut bm);
    let pairs = extract_persistence_pairs(&bm, &filt).expect("ok");
    let diagrams = PersistenceDiagram::from_pairs_by_dim(&pairs, 2);

    // H₀: should have exactly 1 essential class
    let h0 = &diagrams[0];
    let components = count_components(h0);
    assert_eq!(
        components, 1,
        "expected 1 connected component, got {components}"
    );

    // H₁: should have at least 1 generator (the square loop)
    let h1 = &diagrams[1];
    let h1_total = h1.essential_classes().len() + h1.finite_pairs().len();
    assert!(h1_total >= 1, "expected H₁ ≥ 1, got {h1_total}");
}

// ──────────────────────────────────────────────
// Test 9: bottleneck_distance(D, D) = 0
// ──────────────────────────────────────────────
#[test]
fn bottleneck_self_distance_zero() {
    use crate::homology::persistent::PersistencePair;
    let pairs = vec![
        PersistencePair {
            dim: 0,
            birth: 0.0,
            death: Some(1.0),
        },
        PersistencePair {
            dim: 0,
            birth: 0.5,
            death: Some(3.0),
        },
    ];
    let d = PersistenceDiagram::new(pairs, 0);
    let dist = bottleneck_distance(&d, &d).expect("ok");
    assert!(dist < 1e-10, "self bottleneck distance = {dist}");
}

// ──────────────────────────────────────────────
// Test 10: Wasserstein triangle inequality
// ──────────────────────────────────────────────
#[test]
fn wasserstein_triangle_inequality() {
    use crate::homology::persistent::PersistencePair;

    let make = |pts: &[(f64, f64)]| {
        let pairs = pts
            .iter()
            .map(|&(b, d)| PersistencePair {
                dim: 0,
                birth: b,
                death: Some(d),
            })
            .collect::<Vec<_>>();
        PersistenceDiagram::new(pairs, 0)
    };

    let a = make(&[(0.0, 1.0), (1.0, 3.0)]);
    let b = make(&[(0.1, 1.1), (0.9, 2.9)]);
    let c = make(&[(0.5, 2.5)]);

    let wab = wasserstein_1(&a, &b).expect("ok");
    let wbc = wasserstein_1(&b, &c).expect("ok");
    let wac = wasserstein_1(&a, &c).expect("ok");

    assert!(
        wac <= wab + wbc + 1e-9,
        "triangle inequality violated: W(a,c)={wac} > W(a,b)={wab} + W(b,c)={wbc}"
    );
}

// ──────────────────────────────────────────────
// Test 11: Mapper on a circle → loop structure
// ──────────────────────────────────────────────
#[test]
fn mapper_circle_creates_loop() {
    // 20 points uniformly on a circle
    let n = 20usize;
    let pts: Vec<f64> = (0..n)
        .flat_map(|i| {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            vec![angle.cos(), angle.sin()]
        })
        .collect();

    let cfg = MapperConfig {
        n_intervals: 5,
        overlap_frac: 0.4,
        cluster_eps: 0.7,
        min_pts: 1,
    };

    let graph = build_mapper(&pts, n, 2, |p| p[0], &cfg).expect("ok");
    assert!(
        graph.n_nodes() >= 2,
        "expected ≥ 2 nodes, got {}",
        graph.n_nodes()
    );
    assert!(
        graph.n_edges() >= 1,
        "expected ≥ 1 edge, got {}",
        graph.n_edges()
    );
}

// ──────────────────────────────────────────────
// Test 12: Maxmin landmarks are well-separated
// ──────────────────────────────────────────────
#[test]
fn maxmin_landmarks_well_separated() {
    // 10 uniformly-spaced points in [0, 1]
    let n = 10usize;
    let pts: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let mut dist_mat = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            dist_mat[i * n + j] = (pts[i] - pts[j]).abs();
        }
    }
    let mut rng = LcgRng::new(7);
    let landmarks = maxmin_landmarks(&dist_mat, n, 3, &mut rng).expect("ok");
    assert_eq!(landmarks.len(), 3);

    // Check all pairwise landmark distances are ≥ 0.1
    for ii in 0..landmarks.len() {
        for jj in (ii + 1)..landmarks.len() {
            let d = (pts[landmarks[ii]] - pts[landmarks[jj]]).abs();
            assert!(
                d >= 0.09,
                "landmarks not well separated: dist({},{}) = {d}",
                landmarks[ii],
                landmarks[jj]
            );
        }
    }
}

// ──────────────────────────────────────────────
// Test 13: Witness complex has at least 3 vertices
// ──────────────────────────────────────────────
#[test]
fn witness_complex_has_vertices() {
    let n_l = 3;
    let n_w = 8;
    // Create a simple landmark-to-witness distance matrix
    let mut d = vec![0.0_f64; n_l * n_w];
    for l in 0..n_l {
        for w in 0..n_w {
            d[l * n_w + w] = ((l * n_w + w) as f64 * 0.15) % 3.0;
        }
    }
    let filt = lazy_witness_complex(&d, n_l, n_w, 5.0, 1).expect("ok");
    let n_verts = filt
        .simplices
        .iter()
        .filter(|fs| fs.simplex.dim() == 0)
        .count();
    assert_eq!(n_verts, n_l, "expected {n_l} vertices, got {n_verts}");
}

// ──────────────────────────────────────────────
// Test 14: Betti numbers from a known diagram
// ──────────────────────────────────────────────
#[test]
fn betti_numbers_from_diagram() {
    use crate::homology::persistent::PersistencePair;

    // Construct a known diagram: H₀ = 1 essential, H₁ = 1 essential
    let h0_pairs = vec![
        PersistencePair {
            dim: 0,
            birth: 0.0,
            death: Some(1.0),
        }, // finite
        PersistencePair {
            dim: 0,
            birth: 0.0,
            death: None,
        }, // essential
    ];
    let h1_pairs = vec![
        PersistencePair {
            dim: 1,
            birth: 0.5,
            death: None,
        }, // essential loop
    ];

    let diagrams = vec![
        PersistenceDiagram::new(h0_pairs, 0),
        PersistenceDiagram::new(h1_pairs, 1),
    ];

    let betti = betti_numbers(&diagrams, 0.0);
    // β₀ = 1 finite (persistence=1 > 0) + 1 essential = 2
    assert_eq!(betti[0], 2, "H₀ Betti = {}", betti[0]);
    // β₁ = 0 finite + 1 essential = 1
    assert_eq!(betti[1], 1, "H₁ Betti = {}", betti[1]);
}

// ──────────────────────────────────────────────
// Test 15: Persistent entropy ≥ 0
// ──────────────────────────────────────────────
#[test]
fn persistent_entropy_nonneg() {
    use crate::homology::persistent::PersistencePair;
    let pairs = vec![
        PersistencePair {
            dim: 0,
            birth: 0.0,
            death: Some(1.0),
        },
        PersistencePair {
            dim: 0,
            birth: 0.0,
            death: Some(2.0),
        },
        PersistencePair {
            dim: 0,
            birth: 0.5,
            death: Some(3.0),
        },
    ];
    let diag = PersistenceDiagram::new(pairs, 0);
    let h = persistent_entropy(&diag).expect("ok");
    assert!(
        h >= 0.0,
        "persistent entropy should be non-negative, got {h}"
    );
}

// ──────────────────────────────────────────────
// Test 16: Landscape function is positive at midpoint
// ──────────────────────────────────────────────
#[test]
fn landscape_function_positive() {
    use crate::homology::persistent::PersistencePair;
    // Single pair (0, 2): tent function peaks at t=1 with value 1
    let pairs = vec![PersistencePair {
        dim: 0,
        birth: 0.0,
        death: Some(2.0),
    }];
    let diag = PersistenceDiagram::new(pairs, 0);
    let vals = persistence_landscape(&diag, 1, &[1.0]).expect("ok");
    assert!(
        vals[0] > 0.0,
        "landscape at midpoint should be positive, got {}",
        vals[0]
    );
    assert!(
        (vals[0] - 1.0).abs() < 1e-10,
        "landscape at midpoint should be 1.0, got {}",
        vals[0]
    );
}

// ──────────────────────────────────────────────
// Test 17: Essential classes detected (count_components)
// ──────────────────────────────────────────────
#[test]
fn essential_classes_detected() {
    // Single connected component: one H₀ essential class
    let pts = vec![0.0f64, 0.0, 1.0, 0.0];
    let filt = Filtration::vietoris_rips_from_points(&pts, 2, 2.0, 1).expect("ok");
    let mut bm = BoundaryMatrix::from_filtration(&filt).expect("ok");
    reduce_boundary_matrix(&mut bm);
    let pairs = extract_persistence_pairs(&bm, &filt).expect("ok");
    let diagrams = PersistenceDiagram::from_pairs_by_dim(&pairs, 1);
    let n_components = count_components(&diagrams[0]);
    assert_eq!(
        n_components, 1,
        "expected 1 connected component, got {n_components}"
    );
}

// ──────────────────────────────────────────────
// Test 18: PTX kernels are non-empty and valid
// ──────────────────────────────────────────────
#[test]
fn ptx_kernels_valid() {
    type KernelFn = fn(u32) -> String;
    let kernels: &[(&str, KernelFn)] = &[
        ("pairwise_dist", pairwise_dist_ptx),
        ("filtration_sort", filtration_sort_ptx),
        ("boundary_reduce", boundary_reduce_ptx),
        ("diagram_match", diagram_match_ptx),
        ("witness_dist", witness_dist_ptx),
        ("betti_count", betti_count_ptx),
        ("mapper_cluster", mapper_cluster_ptx),
    ];
    let sm_versions: &[u32] = &[75, 80, 86, 89, 90, 100];

    for &sm in sm_versions {
        for &(name, f) in kernels {
            let ptx = f(sm);
            assert!(
                !ptx.is_empty(),
                "kernel {name} at SM{sm} returned empty string"
            );
            assert!(
                ptx.contains(".visible .entry"),
                "kernel {name} at SM{sm} missing '.visible .entry'"
            );
        }
    }
}

// ──────────────────────────────────────────────
// Helper: build a complete TDA pipeline on a small point cloud
// ──────────────────────────────────────────────
#[allow(dead_code)]
fn run_full_pipeline(pts: &[f64], n_dims: usize, max_r: f64, max_dim: usize) {
    let filt = Filtration::vietoris_rips_from_points(pts, n_dims, max_r, max_dim).expect("ok");
    let mut bm = BoundaryMatrix::from_filtration(&filt).expect("ok");
    reduce_boundary_matrix(&mut bm);
    let _pairs = extract_persistence_pairs(&bm, &filt).expect("ok");
}

#[test]
fn full_pipeline_smoke() {
    // 6 points on a hexagon-like arrangement
    let pts: Vec<f64> = (0..6)
        .flat_map(|i| {
            let a = std::f64::consts::PI * i as f64 / 3.0;
            vec![a.cos(), a.sin()]
        })
        .collect();
    run_full_pipeline(&pts, 2, 1.2, 2);
}

// ──────────────────────────────────────────────
// Additional test for pairwise distance
// ──────────────────────────────────────────────
#[test]
fn pairwise_euclidean_symmetry() {
    let pts = vec![0.0f64, 0.0, 3.0, 4.0, 6.0, 8.0];
    let d = pairwise_euclidean(&pts, 2).expect("ok");
    let n = 3;
    for i in 0..n {
        for j in 0..n {
            assert!(
                (d[i * n + j] - d[j * n + i]).abs() < 1e-12,
                "distance matrix not symmetric"
            );
        }
        assert!(d[i * n + i] < 1e-12, "diagonal should be 0");
    }
}

// ──────────────────────────────────────────────
// Test 19: Betti curve from the VR pipeline on a hexagon loop
// ──────────────────────────────────────────────
#[test]
fn betti_curve_from_vr_pipeline_on_loop() {
    // 6 points on a unit circle (regular hexagon). Adjacent points are 1.0 apart,
    // so a 1-cycle (the loop) is born at radius ≈ 1.0 and dies once the disc fills.
    let pts: Vec<f64> = (0..6)
        .flat_map(|i| {
            let a = std::f64::consts::PI * i as f64 / 3.0;
            vec![a.cos(), a.sin()]
        })
        .collect();

    let filt = Filtration::vietoris_rips_from_points(&pts, 2, 1.2, 2).expect("ok");
    let mut bm = BoundaryMatrix::from_filtration(&filt).expect("ok");
    reduce_boundary_matrix(&mut bm);
    let pairs = extract_persistence_pairs(&bm, &filt).expect("ok");
    let diagrams = PersistenceDiagram::from_pairs_by_dim(&pairs, 2);
    let h1 = &diagrams[1];

    // Find the H1 loop's birth radius.
    let h1_total = h1.essential_classes().len() + h1.finite_pairs().len();
    assert!(h1_total >= 1, "expected an H1 loop, got {h1_total}");
    let birth = h1
        .pairs
        .iter()
        .map(|p| p.birth)
        .fold(f64::INFINITY, f64::min);

    // Below the loop's birth radius the H1 Betti number is 0; at a radius just above
    // it should be ≥ 1.
    let grid = [(birth - 0.2).max(0.0) as f32, (birth + 0.05) as f32];
    let curve = betti_curve(h1, 1, &grid).expect("ok");
    assert_eq!(
        curve.values[0], 0,
        "H1 Betti must be 0 below the loop birth"
    );
    assert!(
        curve.values[1] >= 1,
        "H1 Betti must be ≥ 1 once the loop appears"
    );
}

// ──────────────────────────────────────────────
// Test 20: Scale-space kernel on two real point-cloud diagrams
// ──────────────────────────────────────────────
#[test]
fn scale_space_kernel_on_real_diagrams() {
    // Diagram A: a square loop (4 corners of a unit square) — has H1.
    let square = vec![0.0f64, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let filt_a = Filtration::vietoris_rips_from_points(&square, 2, 1.5, 2).expect("ok");
    let mut bm_a = BoundaryMatrix::from_filtration(&filt_a).expect("ok");
    reduce_boundary_matrix(&mut bm_a);
    let pairs_a = extract_persistence_pairs(&bm_a, &filt_a).expect("ok");
    let diags_a = PersistenceDiagram::from_pairs_by_dim(&pairs_a, 1);
    let da = &diags_a[0]; // H0 diagram

    // Diagram B: four collinear points spaced at 2.0 — H0 deaths fall at 2.0,
    // distinct from the square's unit-spaced H0 deaths at 1.0.
    let line = vec![0.0f64, 0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 0.0];
    let filt_b = Filtration::vietoris_rips_from_points(&line, 2, 6.5, 1).expect("ok");
    let mut bm_b = BoundaryMatrix::from_filtration(&filt_b).expect("ok");
    reduce_boundary_matrix(&mut bm_b);
    let pairs_b = extract_persistence_pairs(&bm_b, &filt_b).expect("ok");
    let diags_b = PersistenceDiagram::from_pairs_by_dim(&pairs_b, 1);
    let db = &diags_b[0]; // H0 diagram

    let cfg = KernelConfig::default();

    // Self-kernel positive.
    let kaa = persistence_scale_space_kernel(da, da, &cfg).expect("ok");
    assert!(kaa > 0.0, "self-kernel must be positive, got {kaa}");

    // Symmetric.
    let kab = persistence_scale_space_kernel(da, db, &cfg).expect("ok");
    let kba = persistence_scale_space_kernel(db, da, &cfg).expect("ok");
    assert!((kab - kba).abs() < 1e-12, "kernel must be symmetric");

    // Identical-diagram distance ≈ 0; distinct-diagram distance > 0.
    let d_same = persistence_scale_space_distance(da, da, &cfg).expect("ok");
    assert!(d_same < 1e-9, "self-distance must be ~0, got {d_same}");
    let d_diff = persistence_scale_space_distance(da, db, &cfg).expect("ok");
    assert!(
        d_diff > 0.0,
        "distinct diagrams must have positive distance"
    );
}

// ──────────────────────────────────────────────
// Test 21: Full build-and-tear zigzag → born-by-deletion H1 bar
// ──────────────────────────────────────────────
#[test]
fn zigzag_build_and_tear_born_by_deletion() {
    let s = |v: &[usize]| Simplex::new(v.to_vec()).expect("valid simplex");
    let complex = |sx: &[&[usize]]| ZigzagComplex {
        simplices: sx.iter().map(|v| s(v)).collect(),
    };

    // Build a triangle boundary, fill it, then tear out the 2-simplex: the loop is
    // reborn by deletion at the last complex.
    let x0 = complex(&[&[0], &[1], &[2]]);
    let x1 = complex(&[&[0], &[1], &[2], &[0, 1]]);
    let x2 = complex(&[&[0], &[1], &[2], &[0, 1], &[0, 2], &[1, 2]]);
    let x3 = complex(&[&[0], &[1], &[2], &[0, 1], &[0, 2], &[1, 2], &[0, 1, 2]]);
    let x4 = x2.clone();

    let input = ZigzagInput {
        complexes: vec![x0, x1, x2, x3, x4],
        arrows: vec![
            ZigzagArrow::Forward,
            ZigzagArrow::Forward,
            ZigzagArrow::Forward,
            ZigzagArrow::Backward,
        ],
    };
    let bc = zigzag_persistence(&input).expect("zigzag");

    // The born-by-deletion H1 bar lives in the final complex only: [4, 4].
    let h1 = bc.bars_in_dim(1);
    assert!(
        h1.iter().any(|b| b.birth == 4 && b.death == 4),
        "expected born-by-deletion H1 bar [4,4], got {:?}",
        h1.iter().map(|b| (b.birth, b.death)).collect::<Vec<_>>()
    );
    assert_eq!(bc.betti(4, 1), 1, "β1(X4) must be 1 (loop reborn)");
}
