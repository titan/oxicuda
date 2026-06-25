//! Worked examples reproducing textbook TDA results on the CPU.
//!
//! The crate ships no `examples/` directory, so these textbook reproductions are delivered
//! as runnable, asserted `#[test]` functions.  Each one computes the *real* topology
//! end-to-end (Vietoris-Rips → boundary reduction → persistence, or filter → cover →
//! clustering → nerve) — nothing here is a hand-written diagram:
//!
//! 1. [`two_circles_persistence_recovers_two_loops`] — sample two disjoint noisy circles,
//!    run Vietoris-Rips persistent homology, and verify the diagram recovers the textbook
//!    "two circles" topology: two connected components (H0) and exactly two long-lived
//!    H1 loops (one per circle) that stand clearly above the short-lived sampling noise.
//! 2. [`mapper_recovers_circle_topology`] — sample one noisy circle, run the Mapper
//!    algorithm with the x-coordinate filter, and verify the resulting nerve is a single
//!    cycle (one connected component, first Betti number 1, every node of degree two).

use crate::complex::filtration::Filtration;
use crate::handle::LcgRng;
use crate::homology::boundary::BoundaryMatrix;
use crate::homology::persistent::extract_persistence_pairs;
use crate::homology::reduction::reduce_boundary_matrix;
use crate::mapper::mapper::{MapperConfig, build_mapper};
use crate::metrics::metrics::count_components;
use crate::persistence::diagram::PersistenceDiagram;

/// Sample `n_per` points from each of two unit circles centred at `(0,0)` and
/// `(centre_gap, 0)`, with deterministic radial and angular noise drawn from the crate's
/// full-range LCG.  Returns `(points_row_major_2d, n_pts)`.
fn sample_two_circles(
    seed: u64,
    n_per: usize,
    centre_gap: f64,
    rad_noise: f64,
    ang_noise: f64,
) -> (Vec<f64>, usize) {
    let mut rng = LcgRng::new(seed);
    let radius = 1.0_f64;
    let centres = [(0.0_f64, 0.0_f64), (centre_gap, 0.0_f64)];
    let mut pts = Vec::with_capacity(2 * n_per * 2);
    for &(cx, cy) in &centres {
        for i in 0..n_per {
            let base = 2.0 * std::f64::consts::PI * i as f64 / n_per as f64;
            // Symmetric noise in [-rad_noise, rad_noise] / [-ang_noise, ang_noise].
            let dr = (rng.next_f64() - 0.5) * 2.0 * rad_noise;
            let da = (rng.next_f64() - 0.5) * 2.0 * ang_noise;
            let angle = base + da;
            let rr = radius + dr;
            pts.push(cx + rr * angle.cos());
            pts.push(cy + rr * angle.sin());
        }
    }
    (pts, 2 * n_per)
}

/// Textbook "two circles" persistence: two disjoint noisy circles must produce a
/// persistence diagram with two persistent connected components (H0) and exactly two
/// long-lived loops (H1), one per circle, clearly separated from the sampling noise.
#[test]
fn two_circles_persistence_recovers_two_loops() {
    // 16 points per circle, unit radius, centres 6.0 apart (gap between the circles is
    // 6 - 1 - 1 = 4 >> max_radius, so the two circles never merge).  Radial noise ±0.16,
    // angular noise ±0.12 rad — enough to be non-trivial, small enough to keep the loops.
    let n_per = 16;
    let max_radius = 2.5; // > √3·R (loop fill scale ≈ 1.73) and < inter-circle gap (4.0)
    let (pts, _n_pts) = sample_two_circles(20_260_621, n_per, 6.0, 0.16, 0.12);

    // Full host-side persistence pipeline.
    let filt = Filtration::vietoris_rips_from_points(&pts, 2, max_radius, 2).expect("VR ok");
    let mut bm = BoundaryMatrix::from_filtration(&filt).expect("boundary ok");
    reduce_boundary_matrix(&mut bm);
    let pairs = extract_persistence_pairs(&bm, &filt).expect("pairs ok");
    let diagrams = PersistenceDiagram::from_pairs_by_dim(&pairs, 2);
    let h0 = &diagrams[0];
    let h1 = &diagrams[1];

    // ── H0: two circles → exactly two persistent connected components. ──
    let components = count_components(h0);
    assert_eq!(
        components, 2,
        "two disjoint circles must give 2 persistent H0 components, got {components}"
    );

    // The H0 *noise* is the short-lived merging of nearby samples into their own circle;
    // every finite H0 bar must be short (well below the unit radius).
    let h0_noise_floor = h0
        .finite_pairs()
        .iter()
        .map(|p| p.persistence(0.0)) // finite ⇒ death − birth, no unwrap needed
        .fold(0.0_f64, f64::max);
    assert!(
        h0_noise_floor < 0.7,
        "H0 sampling-noise bars must be short-lived, got max {h0_noise_floor}"
    );

    // ── H1: exactly two long-lived loops, one per circle. ──
    let mut h1_pers: Vec<f64> = h1.pairs.iter().map(|p| p.persistence(max_radius)).collect();
    h1_pers.sort_by(|a, b| b.total_cmp(a));

    // No spurious noise loops at all: the only H1 features are the two real loops.
    assert_eq!(
        h1.pairs.len(),
        2,
        "expected exactly two H1 loops (one per circle), got {} with persistences {h1_pers:?}",
        h1.pairs.len()
    );

    // Both loops are long-lived (persistence ≈ death − birth ≈ 1.0).
    let significant = h1_pers.iter().filter(|&&p| p > 0.7).count();
    assert_eq!(
        significant, 2,
        "expected two significant (long-lived) H1 loops, got {significant} from {h1_pers:?}"
    );

    // Both loops stand clearly above the sampling-noise scale: each H1 loop must outlive
    // the worst H0 noise bar by a wide margin — this is the signal/noise separation.
    for &loop_pers in &h1_pers {
        assert!(
            loop_pers > 1.5 * h0_noise_floor,
            "H1 loop persistence {loop_pers} not clearly above noise floor {h0_noise_floor}"
        );
        // No H1 bar may sit in the "noise band"; both are genuine loops.
        assert!(
            loop_pers > 0.7,
            "H1 bar {loop_pers} fell into the noise band — not a clean loop"
        );
    }
}

/// Textbook "Mapper recovers a circle": the Mapper nerve of a noisy circle, built with the
/// x-coordinate filter, must be a single cycle (one component, β₁ = 1, every node of
/// degree two).
#[test]
fn mapper_recovers_circle_topology() {
    // 40 points on a unit circle with deterministic radial noise ±0.06.
    let mut rng = LcgRng::new(424_242);
    let n = 40usize;
    let radius = 1.0_f64;
    let noise = 0.06_f64;
    let mut pts = Vec::with_capacity(n * 2);
    for i in 0..n {
        let base = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
        let dr = (rng.next_f64() - 0.5) * 2.0 * noise;
        let rr = radius + dr;
        pts.push(rr * base.cos());
        pts.push(rr * base.sin());
    }

    // x-coordinate filter: each interior vertical strip meets the circle in a top arc and
    // a bottom arc (two clusters), while the leftmost/rightmost strips meet it in a single
    // arc.  Overlapping intervals chain matching arcs, closing the two arcs into one cycle.
    // cluster_eps = 0.5 keeps each arc connected (adjacent samples ≈ 0.157 apart) while
    // keeping the well-separated top/bottom arcs in distinct clusters.
    let cfg = MapperConfig {
        n_intervals: 6,
        overlap_frac: 0.4,
        cluster_eps: 0.5,
        min_pts: 1,
    };
    let graph = build_mapper(&pts, n, 2, |p| p[0], &cfg).expect("mapper ok");

    // A non-degenerate graph.
    assert!(
        graph.n_nodes() >= 4,
        "expected a non-trivial nerve, got {} nodes",
        graph.n_nodes()
    );

    // Single connected component (the circle is connected).
    let n_comp = graph.connected_components().len();
    assert_eq!(
        n_comp, 1,
        "circle nerve must be connected, got {n_comp} components"
    );

    // First Betti number 1: the nerve has exactly one independent loop.
    let b1 = graph.betti_1();
    assert_eq!(
        b1, 1,
        "Mapper must recover a single loop (β₁ = 1), got β₁ = {b1}"
    );

    // A pure cycle: |edges| == |nodes| and every node has degree exactly 2 (closed chain).
    assert_eq!(
        graph.n_edges(),
        graph.n_nodes(),
        "a single cycle has |E| == |V|, got {} edges, {} nodes",
        graph.n_edges(),
        graph.n_nodes()
    );
    for node in 0..graph.n_nodes() {
        let deg = graph.degree(node);
        assert_eq!(
            deg, 2,
            "every node of a closed-chain cycle must have degree 2, node {node} has degree {deg}"
        );
    }
}
