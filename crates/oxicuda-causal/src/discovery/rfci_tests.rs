//! Inline tests for the Really Fast Causal Inference (RFCI) algorithm.

#![cfg(test)]

use super::fci::{EdgeMark, Fci, FciConfig, Pag};
use super::rfci::{Rfci, RfciConfig};
use crate::handle::LcgRng;

fn gen_normal_matrix(rng: &mut LcgRng, n: usize, d: usize) -> Vec<Vec<f64>> {
    (0..d)
        .map(|_| (0..n).map(|_| rng.next_normal() as f64).collect())
        .collect()
}

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

/// X → Y → Z. The skeleton should drop the X–Z edge (X ⫫ Z | Y) and keep
/// X–Y and Y–Z.
#[test]
fn test_rfci_chain_xyz_skeleton() {
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
    let rfci = Rfci::new(RfciConfig::default()).unwrap();
    let pag = rfci.fit(&cols).unwrap();
    assert!(pag.adj_pub(0, 1));
    assert!(pag.adj_pub(1, 2));
    assert!(!pag.adj_pub(0, 2));
}

/// X ← L → Y with L unobserved (latent confounder). RFCI should output
/// `X *-* Y` (no separation possible) — exact bidirected marks depend on
/// the orientation rules; we test that the edge is preserved.
#[test]
fn test_rfci_latent_confounder_keeps_edge() {
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
    let rfci = Rfci::new(RfciConfig::default()).unwrap();
    let pag = rfci.fit(&cols).unwrap();
    assert!(pag.adj_pub(0, 1));
}

/// X → Z ← Y with X ⫫ Y. The skeleton retains X–Z and Y–Z, and the
/// unshielded triple X — Z — Y must be oriented as a collider with arrows
/// pointing into Z (Z is the middle node, index 1 here).
#[test]
fn test_rfci_collider_orientation_three_vars() {
    let n = 600_usize;
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
    let rfci = Rfci::new(RfciConfig::default()).unwrap();
    let pag = rfci.fit(&cols).unwrap();
    assert!(pag.adj_pub(0, 1));
    assert!(pag.adj_pub(2, 1));
    assert!(!pag.adj_pub(0, 2));
    assert_eq!(pag.mark_pub(0, 1), EdgeMark::Arrow);
    assert_eq!(pag.mark_pub(2, 1), EdgeMark::Arrow);
}

/// Three completely independent normals should produce no edges.
#[test]
fn test_rfci_three_independent_vars() {
    let n = 400_usize;
    let mut rng = LcgRng::new(101);
    let cols = gen_normal_matrix(&mut rng, n, 3);
    let rfci = Rfci::new(RfciConfig::default()).unwrap();
    let pag = rfci.fit(&cols).unwrap();
    assert_eq!(count_edges(&pag), 0);
}

/// Five-variable Markov chain X1 → X2 → X3 → X4 → X5. Only the four
/// consecutive edges should remain.
#[test]
fn test_rfci_five_var_chain() {
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
    let rfci = Rfci::new(RfciConfig::default()).unwrap();
    let pag = rfci.fit(&cols).unwrap();
    for j in 0..4 {
        assert!(pag.adj_pub(j, j + 1), "missing edge {j}-{}", j + 1);
    }
    assert!(!pag.adj_pub(0, 4));
    assert!(!pag.adj_pub(0, 3));
}

/// A high alpha (almost always reject independence) should keep many
/// edges; a low alpha should drop most of them in pure-noise data.
#[test]
fn test_rfci_alpha_sensitivity() {
    let n = 50_usize;
    let mut rng = LcgRng::new(2025);
    let cols = gen_normal_matrix(&mut rng, n, 4);
    let dense = Rfci::new(RfciConfig {
        alpha: 0.5,
        max_cond_set_size: 2,
    })
    .unwrap();
    let sparse = Rfci::new(RfciConfig {
        alpha: 1e-6,
        max_cond_set_size: 2,
    })
    .unwrap();
    let dense_pag = dense.fit(&cols).unwrap();
    let sparse_pag = sparse.fit(&cols).unwrap();
    assert!(count_edges(&dense_pag) >= count_edges(&sparse_pag));
}

/// `max_cond_set_size = 0` should only run order-0 (marginal) tests.
#[test]
fn test_rfci_max_cond_set_size_respected() {
    let n = 200_usize;
    let mut rng = LcgRng::new(55);
    let mut cols: Vec<Vec<f64>> = (0..3).map(|_| Vec::with_capacity(n)).collect();
    for _ in 0..n {
        let x = rng.next_normal() as f64;
        let y = x + 0.05 * rng.next_normal() as f64;
        let z = y + 0.05 * rng.next_normal() as f64;
        cols[0].push(x);
        cols[1].push(y);
        cols[2].push(z);
    }
    // With max_cond_set_size = 0, X ⫫ Z given {Y} test is never run; X–Z
    // remains because their marginal correlation is non-zero in this DGP.
    let rfci0 = Rfci::new(RfciConfig {
        alpha: 0.05,
        max_cond_set_size: 0,
    })
    .unwrap();
    let pag0 = rfci0.fit(&cols).unwrap();
    assert!(pag0.adj_pub(0, 2));
    // With max_cond_set_size = 1, the conditioning test fires and the
    // X–Z edge disappears.
    let rfci1 = Rfci::new(RfciConfig {
        alpha: 0.05,
        max_cond_set_size: 1,
    })
    .unwrap();
    let pag1 = rfci1.fit(&cols).unwrap();
    assert!(!pag1.adj_pub(0, 2));
}

/// Empty data (zero variables) must error.
#[test]
fn test_rfci_empty_data_errors() {
    let rfci = Rfci::new(RfciConfig::default()).unwrap();
    let empty: Vec<Vec<f64>> = vec![];
    assert!(rfci.fit(&empty).is_err());
}

/// A single variable cannot form any edge; we treat it as an invalid
/// graph size for this discovery routine.
#[test]
fn test_rfci_single_variable_errors() {
    let rfci = Rfci::new(RfciConfig::default()).unwrap();
    let one: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0, 4.0]];
    assert!(rfci.fit(&one).is_err());
}

/// Fit twice — same input, same output, structurally.
#[test]
fn test_rfci_idempotent_fit() {
    let n = 200_usize;
    let mut rng = LcgRng::new(99);
    let cols = gen_normal_matrix(&mut rng, n, 4);
    let rfci = Rfci::new(RfciConfig::default()).unwrap();
    let a = rfci.fit(&cols).unwrap();
    let b = rfci.fit(&cols).unwrap();
    assert_eq!(a.n_vars, b.n_vars);
    for i in 0..a.n_vars {
        for j in 0..a.n_vars {
            assert_eq!(a.adj_pub(i, j), b.adj_pub(i, j));
            assert_eq!(a.mark_pub(i, j), b.mark_pub(i, j));
        }
    }
}

/// Output must be deterministic for fixed inputs.
#[test]
fn test_rfci_deterministic_output() {
    let cols: Vec<Vec<f64>> = vec![
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![0.1, 0.9, 2.1, 2.9, 4.1, 4.9, 6.1, 6.9, 8.1, 8.9],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    ];
    let rfci = Rfci::new(RfciConfig::default()).unwrap();
    let p1 = rfci.fit(&cols).unwrap();
    let p2 = rfci.fit(&cols).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(p1.mark_pub(i, j), p2.mark_pub(i, j));
            assert_eq!(p1.adj_pub(i, j), p2.adj_pub(i, j));
        }
    }
}

/// RFCI and FCI must agree on the skeleton of a simple three-variable
/// chain, since RFCI only differs from FCI in the *refinement* step.
#[test]
fn test_rfci_skeleton_agrees_with_fci_on_chain() {
    let n = 600_usize;
    let mut rng = LcgRng::new(2024);
    let mut data_rm = vec![0.0_f64; n * 3];
    let mut cols: Vec<Vec<f64>> = (0..3).map(|_| Vec::with_capacity(n)).collect();
    for i in 0..n {
        let x = rng.next_normal() as f64;
        let y = x + 0.1 * rng.next_normal() as f64;
        let z = y + 0.1 * rng.next_normal() as f64;
        data_rm[i * 3] = x;
        data_rm[i * 3 + 1] = y;
        data_rm[i * 3 + 2] = z;
        cols[0].push(x);
        cols[1].push(y);
        cols[2].push(z);
    }
    let fci = Fci::new(FciConfig::default()).unwrap();
    let fci_pag = fci.fit(&data_rm, n, 3).unwrap();
    let rfci = Rfci::new(RfciConfig::default()).unwrap();
    let rfci_pag = rfci.fit(&cols).unwrap();
    for i in 0..3 {
        for j in (i + 1)..3 {
            assert_eq!(
                fci_pag.adj_pub(i, j),
                rfci_pag.adj_pub(i, j),
                "skeleton mismatch at ({i},{j})"
            );
        }
    }
}

/// Non-finite inputs (NaN) must be rejected as invalid config rather than
/// silently propagated.
#[test]
fn test_rfci_rejects_non_finite_input() {
    let rfci = Rfci::new(RfciConfig::default()).unwrap();
    let cols: Vec<Vec<f64>> = vec![
        vec![0.0, 1.0, 2.0, 3.0, 4.0],
        vec![0.0, 1.0, f64::NAN, 3.0, 4.0],
    ];
    assert!(rfci.fit(&cols).is_err());
}

/// Mismatched column lengths must be reported as a dimension mismatch.
#[test]
fn test_rfci_dim_mismatch() {
    let rfci = Rfci::new(RfciConfig::default()).unwrap();
    let cols: Vec<Vec<f64>> = vec![vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![0.0, 1.0, 2.0, 3.0]];
    assert!(rfci.fit(&cols).is_err());
}

/// Construction with an out-of-range alpha must fail.
#[test]
fn test_rfci_invalid_alpha() {
    assert!(
        Rfci::new(RfciConfig {
            alpha: 0.0,
            max_cond_set_size: 2,
        })
        .is_err()
    );
    assert!(
        Rfci::new(RfciConfig {
            alpha: 1.5,
            max_cond_set_size: 2,
        })
        .is_err()
    );
}

/// Too few samples must be rejected (Fisher-Z needs n > cond_set_size + 3).
#[test]
fn test_rfci_too_few_samples() {
    let rfci = Rfci::new(RfciConfig::default()).unwrap();
    let cols: Vec<Vec<f64>> = vec![vec![0.0, 1.0], vec![1.0, 2.0]];
    assert!(rfci.fit(&cols).is_err());
}
