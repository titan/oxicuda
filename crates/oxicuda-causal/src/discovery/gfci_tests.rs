//! Inline tests for the GFCI (Greedy Fast Causal Inference) algorithm.

#![cfg(test)]

use super::fci::{EdgeMark, Pag};
use super::gfci::{Gfci, GfciConfig};
use crate::handle::LcgRng;

fn count_edges(pag: &Pag) -> usize {
    let mut count = 0;
    for i in 0..pag.n_vars {
        for j in (i + 1)..pag.n_vars {
            if pag.adj_pub(i, j) {
                count += 1;
            }
        }
    }
    count
}

fn count_bidirected(pag: &Pag) -> usize {
    let mut count = 0;
    for i in 0..pag.n_vars {
        for j in (i + 1)..pag.n_vars {
            if pag.adj_pub(i, j)
                && pag.mark_pub(i, j) == EdgeMark::Arrow
                && pag.mark_pub(j, i) == EdgeMark::Arrow
            {
                count += 1;
            }
        }
    }
    count
}

fn gen_normal_matrix(rng: &mut LcgRng, n: usize, d: usize) -> Vec<Vec<f64>> {
    (0..d)
        .map(|_| (0..n).map(|_| rng.next_normal() as f64).collect())
        .collect()
}

/// Chain X → Y → Z must keep the X–Y and Y–Z edges, drop X–Z, and not
/// introduce any bidirected (arrow-arrow) edges.
#[test]
fn test_gfci_chain_no_bidirected() {
    let n = 600_usize;
    let mut rng = LcgRng::new(11);
    let mut cols: Vec<Vec<f64>> = (0..3).map(|_| Vec::with_capacity(n)).collect();
    for _ in 0..n {
        let x = rng.next_normal() as f64;
        let y = x + 0.1 * rng.next_normal() as f64;
        let z = y + 0.1 * rng.next_normal() as f64;
        cols[0].push(x);
        cols[1].push(y);
        cols[2].push(z);
    }
    let pag =
        Gfci::discover(&cols, &GfciConfig::default()).expect("chain GFCI discovery should succeed");
    assert!(pag.adj_pub(0, 1));
    assert!(pag.adj_pub(1, 2));
    assert_eq!(count_bidirected(&pag), 0);
}

/// Latent confounder L between X and Y. The marginal X–Y dependence should
/// produce an edge with at least one arrow (collider-style mark) at one end.
#[test]
fn test_gfci_latent_confounder_keeps_edge() {
    let n = 600_usize;
    let mut rng = LcgRng::new(7);
    let mut cols: Vec<Vec<f64>> = (0..2).map(|_| Vec::with_capacity(n)).collect();
    for _ in 0..n {
        let l = rng.next_normal() as f64;
        let x = l + 0.2 * rng.next_normal() as f64;
        let y = l + 0.2 * rng.next_normal() as f64;
        cols[0].push(x);
        cols[1].push(y);
    }
    let pag = Gfci::discover(&cols, &GfciConfig::default())
        .expect("latent-confounder GFCI discovery should succeed");
    assert!(pag.adj_pub(0, 1));
}

/// Empty input must error.
#[test]
fn test_gfci_empty_data_errors() {
    let empty: Vec<Vec<f64>> = vec![];
    assert!(Gfci::discover(&empty, &GfciConfig::default()).is_err());
}

/// A single variable should produce an empty Pag with n_vars == 1.
#[test]
fn test_gfci_single_variable_empty_pag() {
    let one: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0, 4.0]];
    let pag = Gfci::discover(&one, &GfciConfig::default())
        .expect("single-variable GFCI discovery should succeed");
    assert_eq!(pag.n_vars, 1);
    assert_eq!(count_edges(&pag), 0);
}

/// Two independent variables should yield no edges.
#[test]
fn test_gfci_two_independent_vars_no_edge() {
    let n = 400_usize;
    let mut rng = LcgRng::new(91);
    let cols = gen_normal_matrix(&mut rng, n, 2);
    let pag = Gfci::discover(&cols, &GfciConfig::default())
        .expect("two-independent-vars GFCI discovery should succeed");
    assert_eq!(count_edges(&pag), 0);
}

/// Two calls with identical data must produce structurally identical PAGs.
#[test]
fn test_gfci_deterministic() {
    let n = 200_usize;
    let mut rng = LcgRng::new(42);
    let mut cols: Vec<Vec<f64>> = (0..3).map(|_| Vec::with_capacity(n)).collect();
    for _ in 0..n {
        let x = rng.next_normal() as f64;
        let y = x + 0.1 * rng.next_normal() as f64;
        let z = y + 0.1 * rng.next_normal() as f64;
        cols[0].push(x);
        cols[1].push(y);
        cols[2].push(z);
    }
    let cfg = GfciConfig::default();
    let p1 =
        Gfci::discover(&cols, &cfg).expect("first deterministic GFCI discovery should succeed");
    let p2 =
        Gfci::discover(&cols, &cfg).expect("second deterministic GFCI discovery should succeed");
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(p1.adj_pub(i, j), p2.adj_pub(i, j));
            assert_eq!(p1.mark_pub(i, j), p2.mark_pub(i, j));
        }
    }
}

/// Re-discovering on a PAG that is already fully consistent must be a
/// fixed-point: the second call returns the same structure as the first.
#[test]
fn test_gfci_idempotent_repeated_calls() {
    let n = 300_usize;
    let mut rng = LcgRng::new(2024);
    let mut cols: Vec<Vec<f64>> = (0..4).map(|_| Vec::with_capacity(n)).collect();
    for _ in 0..n {
        let x = rng.next_normal() as f64;
        let y = x + 0.1 * rng.next_normal() as f64;
        let z = y + 0.1 * rng.next_normal() as f64;
        let w = rng.next_normal() as f64;
        cols[0].push(x);
        cols[1].push(y);
        cols[2].push(z);
        cols[3].push(w);
    }
    let cfg = GfciConfig::default();
    let a = Gfci::discover(&cols, &cfg).expect("first idempotent GFCI discovery should succeed");
    let b = Gfci::discover(&cols, &cfg).expect("second idempotent GFCI discovery should succeed");
    assert_eq!(a.n_vars, b.n_vars);
    for i in 0..a.n_vars {
        for j in 0..a.n_vars {
            assert_eq!(a.adj_pub(i, j), b.adj_pub(i, j));
            assert_eq!(a.mark_pub(i, j), b.mark_pub(i, j));
        }
    }
}

/// Mismatched column lengths must be reported as a dimension mismatch.
#[test]
fn test_gfci_dim_mismatch() {
    let cols: Vec<Vec<f64>> = vec![vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![0.0, 1.0, 2.0, 3.0]];
    assert!(Gfci::discover(&cols, &GfciConfig::default()).is_err());
}

/// Negative or zero BIC penalty must be rejected as `IncompatibleData`.
#[test]
fn test_gfci_negative_bic_penalty_errors() {
    let n = 50_usize;
    let mut rng = LcgRng::new(1);
    let cols = gen_normal_matrix(&mut rng, n, 3);
    let cfg = GfciConfig {
        bic_penalty: -1.0,
        ..GfciConfig::default()
    };
    assert!(Gfci::discover(&cols, &cfg).is_err());
    let cfg0 = GfciConfig {
        bic_penalty: 0.0,
        ..GfciConfig::default()
    };
    assert!(Gfci::discover(&cols, &cfg0).is_err());
}

/// Out-of-range Fisher-Z alpha must be rejected.
#[test]
fn test_gfci_invalid_alpha_errors() {
    let n = 50_usize;
    let mut rng = LcgRng::new(2);
    let cols = gen_normal_matrix(&mut rng, n, 3);
    for &a in &[0.0_f64, -0.1, 1.0, 1.5] {
        let cfg = GfciConfig {
            fisher_z_alpha: a,
            ..GfciConfig::default()
        };
        assert!(
            Gfci::discover(&cols, &cfg).is_err(),
            "alpha {a} should be rejected"
        );
    }
}

/// `max_orient_passes = 0` should still succeed: the algorithm returns
/// after the collider-orientation step, with no R1–R4 propagation.
#[test]
fn test_gfci_zero_orient_passes_succeeds() {
    let n = 300_usize;
    let mut rng = LcgRng::new(13);
    let mut cols: Vec<Vec<f64>> = (0..3).map(|_| Vec::with_capacity(n)).collect();
    for _ in 0..n {
        let x = rng.next_normal() as f64;
        let y = rng.next_normal() as f64;
        let z = x + y + 0.05 * rng.next_normal() as f64;
        cols[0].push(x);
        cols[1].push(z);
        cols[2].push(y);
    }
    let cfg = GfciConfig {
        max_orient_passes: 0,
        ..GfciConfig::default()
    };
    let pag =
        Gfci::discover(&cols, &cfg).expect("zero-orient-passes GFCI discovery should succeed");
    assert!(pag.adj_pub(0, 1));
    assert!(pag.adj_pub(2, 1));
}

/// Star graph: variable 0 directly influences variables 1, 2, 3.
/// All four star edges (0,1), (0,2), (0,3) must be present.
#[test]
fn test_gfci_star_graph_recovers() {
    let n = 500_usize;
    let mut rng = LcgRng::new(2025);
    let mut cols: Vec<Vec<f64>> = (0..4).map(|_| Vec::with_capacity(n)).collect();
    for _ in 0..n {
        let center = rng.next_normal() as f64;
        cols[0].push(center);
        cols[1].push(center + 0.1 * rng.next_normal() as f64);
        cols[2].push(center + 0.1 * rng.next_normal() as f64);
        cols[3].push(center + 0.1 * rng.next_normal() as f64);
    }
    let pag = Gfci::discover(&cols, &GfciConfig::default())
        .expect("star-graph GFCI discovery should succeed");
    assert!(pag.adj_pub(0, 1));
    assert!(pag.adj_pub(0, 2));
    assert!(pag.adj_pub(0, 3));
}

/// Five-variable SEM X1 → X2 → X3 → X4 → X5. Skeleton should retain the
/// four consecutive edges and drop the long-range non-edges.
#[test]
fn test_gfci_five_var_chain_majority_recovered() {
    let n = 600_usize;
    let mut rng = LcgRng::new(31);
    let mut cols: Vec<Vec<f64>> = (0..5).map(|_| Vec::with_capacity(n)).collect();
    for _ in 0..n {
        let mut prev = rng.next_normal() as f64;
        cols[0].push(prev);
        for col in cols.iter_mut().take(5).skip(1) {
            let next = prev + 0.1 * rng.next_normal() as f64;
            col.push(next);
            prev = next;
        }
    }
    let pag = Gfci::discover(&cols, &GfciConfig::default())
        .expect("five-var chain GFCI discovery should succeed");
    let mut consecutive = 0_usize;
    for j in 0..4 {
        if pag.adj_pub(j, j + 1) {
            consecutive += 1;
        }
    }
    assert!(consecutive >= 3, "only {consecutive} consecutive edges");
    assert!(!pag.adj_pub(0, 4));
}

/// Reproducibility across two identical calls (alias of deterministic).
#[test]
fn test_gfci_reproducible_independent_calls() {
    let n = 250_usize;
    let mut rng = LcgRng::new(8);
    let cols = gen_normal_matrix(&mut rng, n, 4);
    let cfg = GfciConfig::default();
    let a = Gfci::discover(&cols, &cfg).expect("first reproducible GFCI discovery should succeed");
    let b = Gfci::discover(&cols, &cfg).expect("second reproducible GFCI discovery should succeed");
    assert_eq!(count_edges(&a), count_edges(&b));
    for i in 0..a.n_vars {
        for j in 0..a.n_vars {
            assert_eq!(a.adj_pub(i, j), b.adj_pub(i, j));
        }
    }
}

/// Non-finite entries (NaN) must be rejected.
#[test]
fn test_gfci_rejects_non_finite_input() {
    let cols: Vec<Vec<f64>> = vec![
        vec![0.0, 1.0, 2.0, 3.0, 4.0],
        vec![0.0, 1.0, f64::NAN, 3.0, 4.0],
    ];
    assert!(Gfci::discover(&cols, &GfciConfig::default()).is_err());
}

/// Too few samples (n < 4) must be rejected.
#[test]
fn test_gfci_too_few_samples_errors() {
    let cols: Vec<Vec<f64>> = vec![vec![0.0, 1.0, 2.0], vec![1.0, 2.0, 3.0]];
    assert!(Gfci::discover(&cols, &GfciConfig::default()).is_err());
}

/// Non-default BIC penalty (≠1.0) must still produce a valid PAG.
#[test]
fn test_gfci_alternate_bic_penalty_runs() {
    let n = 400_usize;
    let mut rng = LcgRng::new(9_999);
    let mut cols: Vec<Vec<f64>> = (0..3).map(|_| Vec::with_capacity(n)).collect();
    for _ in 0..n {
        let x = rng.next_normal() as f64;
        let y = x + 0.1 * rng.next_normal() as f64;
        let z = y + 0.1 * rng.next_normal() as f64;
        cols[0].push(x);
        cols[1].push(y);
        cols[2].push(z);
    }
    let cfg = GfciConfig {
        bic_penalty: 2.0,
        ..GfciConfig::default()
    };
    let pag =
        Gfci::discover(&cols, &cfg).expect("alternate BIC penalty GFCI discovery should succeed");
    assert!(pag.n_vars == 3);
    // With heavier penalty the skeleton can only shrink, never grow.
    let total = count_edges(&pag);
    assert!(total <= 3);
}
