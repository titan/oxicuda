//! Inline tests for the FCI algorithm and PAG data structure.

#![cfg(test)]

use super::fci::{EdgeMark, Fci, FciConfig, Pag};
use super::fci_numeric::normal_quantile_two_sided;
use crate::handle::LcgRng;

fn make_data(rng: &mut LcgRng, n: usize, d: usize) -> Vec<f64> {
    let mut data = vec![0.0_f64; n * d];
    for v in data.iter_mut() {
        *v = rng.next_normal() as f64;
    }
    data
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

#[test]
fn test_collider_orientation_three_vars() {
    let n = 400_usize;
    let d = 3_usize;
    let mut rng = LcgRng::new(11);
    let mut data = vec![0.0_f64; n * d];
    for i in 0..n {
        let x = rng.next_normal() as f64;
        let z = rng.next_normal() as f64;
        let y = x + z + 0.05 * rng.next_normal() as f64;
        data[i * d] = x;
        data[i * d + 1] = y;
        data[i * d + 2] = z;
    }
    let fci = Fci::new(FciConfig::default()).unwrap();
    let pag = fci.fit(&data, n, d).unwrap();
    assert!(pag.adj_pub(0, 1));
    assert!(pag.adj_pub(2, 1));
    assert_eq!(pag.mark_pub(0, 1), EdgeMark::Arrow);
    assert_eq!(pag.mark_pub(2, 1), EdgeMark::Arrow);
}

#[test]
fn test_chain_no_collider() {
    let n = 400_usize;
    let d = 3_usize;
    let mut rng = LcgRng::new(42);
    let mut data = vec![0.0_f64; n * d];
    for i in 0..n {
        let x = rng.next_normal() as f64;
        let y = x + 0.1 * rng.next_normal() as f64;
        let z = y + 0.1 * rng.next_normal() as f64;
        data[i * d] = x;
        data[i * d + 1] = y;
        data[i * d + 2] = z;
    }
    let fci = Fci::new(FciConfig::default()).unwrap();
    let pag = fci.fit(&data, n, d).unwrap();
    assert!(!pag.adj_pub(0, 2));
    assert!(pag.adj_pub(0, 1));
    assert!(pag.adj_pub(1, 2));
}

#[test]
fn test_latent_confounder_marginalised() {
    let n = 600_usize;
    let mut rng = LcgRng::new(7);
    let d = 3_usize;
    let mut data = vec![0.0_f64; n * d];
    for i in 0..n {
        let l = rng.next_normal() as f64;
        let a = l + 0.2 * rng.next_normal() as f64;
        let b = l + 0.2 * rng.next_normal() as f64;
        let c = a + 0.2 * rng.next_normal() as f64;
        data[i * d] = a;
        data[i * d + 1] = b;
        data[i * d + 2] = c;
    }
    let fci = Fci::new(FciConfig::default()).unwrap();
    let pag = fci.fit(&data, n, d).unwrap();
    assert!(count_edges(&pag) >= 1);
}

#[test]
fn test_large_alpha_keeps_many_edges() {
    let n = 50_usize;
    let d = 4_usize;
    let mut rng = LcgRng::new(123);
    let data = make_data(&mut rng, n, d);
    let fci = Fci::new(FciConfig {
        alpha: 0.5,
        max_cond_set_size: 2,
    })
    .unwrap();
    let pag = fci.fit(&data, n, d).unwrap();
    assert!(count_edges(&pag) >= 1);
}

#[test]
fn test_small_alpha_on_noise_yields_near_empty() {
    let n = 100_usize;
    let d = 4_usize;
    let mut rng = LcgRng::new(9_999);
    let data = make_data(&mut rng, n, d);
    let fci = Fci::new(FciConfig {
        alpha: 0.001,
        max_cond_set_size: 2,
    })
    .unwrap();
    let pag = fci.fit(&data, n, d).unwrap();
    let max_edges = d * (d - 1) / 2;
    assert!(count_edges(&pag) <= max_edges / 2 + 1);
}

#[test]
fn test_adjacency_symmetric() {
    let n = 80_usize;
    let d = 4_usize;
    let mut rng = LcgRng::new(2);
    let data = make_data(&mut rng, n, d);
    let fci = Fci::new(FciConfig::default()).unwrap();
    let pag = fci.fit(&data, n, d).unwrap();
    for i in 0..d {
        for j in 0..d {
            assert_eq!(pag.adjacency[i * d + j], pag.adjacency[j * d + i]);
        }
    }
}

#[test]
fn test_sep_set_determines_orientation_explicit() {
    let n = 500_usize;
    let d = 4_usize;
    let mut rng = LcgRng::new(31);
    let mut data = vec![0.0_f64; n * d];
    for i in 0..n {
        let x = rng.next_normal() as f64;
        let z = rng.next_normal() as f64;
        let w = rng.next_normal() as f64;
        let y = x + z + 0.05 * rng.next_normal() as f64;
        data[i * d] = x;
        data[i * d + 1] = y;
        data[i * d + 2] = z;
        data[i * d + 3] = w;
    }
    let fci = Fci::new(FciConfig::default()).unwrap();
    let pag = fci.fit(&data, n, d).unwrap();
    assert_eq!(pag.mark_pub(0, 1), EdgeMark::Arrow);
    assert_eq!(pag.mark_pub(2, 1), EdgeMark::Arrow);
    assert!(!pag.adj_pub(3, 0) || count_edges(&pag) <= 4);
}

#[test]
fn test_five_node_sem_skeleton() {
    let n = 600_usize;
    let d = 5_usize;
    let mut rng = LcgRng::new(101);
    let mut data = vec![0.0_f64; n * d];
    for i in 0..n {
        let x0 = rng.next_normal() as f64;
        let x1 = 0.8 * x0 + 0.3 * rng.next_normal() as f64;
        let x2 = 0.8 * x0 + 0.3 * rng.next_normal() as f64;
        let x3 = 0.7 * x1 + 0.7 * x2 + 0.2 * rng.next_normal() as f64;
        let x4 = rng.next_normal() as f64;
        data[i * d] = x0;
        data[i * d + 1] = x1;
        data[i * d + 2] = x2;
        data[i * d + 3] = x3;
        data[i * d + 4] = x4;
    }
    let fci = Fci::new(FciConfig::default()).unwrap();
    let pag = fci.fit(&data, n, d).unwrap();
    assert!(pag.adj_pub(0, 1));
    assert!(pag.adj_pub(0, 2));
    assert!(pag.adj_pub(1, 3));
    assert!(pag.adj_pub(2, 3));
    let neighbors_4: Vec<usize> = (0..d).filter(|&j| j != 4 && pag.adj_pub(4, j)).collect();
    assert!(neighbors_4.len() <= 1);
}

#[test]
fn test_rule_r1_applies() {
    let mut pag = Pag::empty_pub(3);
    pag.set_adj_pub(0, 1, true);
    pag.set_adj_pub(1, 2, true);
    pag.set_mark_pub(0, 1, EdgeMark::Arrow);
    pag.set_mark_pub(1, 0, EdgeMark::Tail);
    pag.set_mark_pub(1, 2, EdgeMark::Circle);
    pag.set_mark_pub(2, 1, EdgeMark::Circle);
    let _ = Fci::rule_r1_pub(&mut pag);
    assert_eq!(pag.mark_pub(2, 1), EdgeMark::Tail);
    assert_eq!(pag.mark_pub(1, 2), EdgeMark::Arrow);
}

#[test]
fn test_rule_r2_applies() {
    let mut pag = Pag::empty_pub(3);
    pag.set_adj_pub(0, 1, true);
    pag.set_adj_pub(1, 2, true);
    pag.set_adj_pub(0, 2, true);
    pag.set_mark_pub(1, 0, EdgeMark::Tail);
    pag.set_mark_pub(0, 1, EdgeMark::Arrow);
    pag.set_mark_pub(2, 1, EdgeMark::Tail);
    pag.set_mark_pub(1, 2, EdgeMark::Arrow);
    pag.set_mark_pub(0, 2, EdgeMark::Circle);
    pag.set_mark_pub(2, 0, EdgeMark::Circle);
    let _ = Fci::rule_r2_pub(&mut pag);
    assert_eq!(pag.mark_pub(0, 2), EdgeMark::Arrow);
}

#[test]
fn test_rule_r3_applies() {
    let mut pag = Pag::empty_pub(4);
    pag.set_adj_pub(0, 1, true);
    pag.set_adj_pub(2, 1, true);
    pag.set_adj_pub(0, 3, true);
    pag.set_adj_pub(2, 3, true);
    pag.set_adj_pub(3, 1, true);
    pag.set_mark_pub(0, 1, EdgeMark::Arrow);
    pag.set_mark_pub(2, 1, EdgeMark::Arrow);
    pag.set_mark_pub(1, 0, EdgeMark::Circle);
    pag.set_mark_pub(1, 2, EdgeMark::Circle);
    pag.set_mark_pub(0, 3, EdgeMark::Circle);
    pag.set_mark_pub(3, 0, EdgeMark::Circle);
    pag.set_mark_pub(2, 3, EdgeMark::Circle);
    pag.set_mark_pub(3, 2, EdgeMark::Circle);
    pag.set_mark_pub(3, 1, EdgeMark::Circle);
    pag.set_mark_pub(1, 3, EdgeMark::Circle);
    let _ = Fci::rule_r3_pub(&mut pag);
    assert_eq!(pag.mark_pub(3, 1), EdgeMark::Arrow);
}

#[test]
fn test_rule_r4_applies() {
    let mut pag = Pag::empty_pub(4);
    pag.set_adj_pub(0, 1, true);
    pag.set_adj_pub(1, 2, true);
    pag.set_adj_pub(1, 3, true);
    pag.set_adj_pub(2, 3, true);
    pag.set_mark_pub(0, 1, EdgeMark::Arrow);
    pag.set_mark_pub(1, 0, EdgeMark::Arrow);
    pag.set_mark_pub(1, 2, EdgeMark::Arrow);
    pag.set_mark_pub(2, 1, EdgeMark::Arrow);
    pag.set_mark_pub(1, 3, EdgeMark::Arrow);
    pag.set_mark_pub(3, 1, EdgeMark::Tail);
    pag.set_mark_pub(2, 3, EdgeMark::Circle);
    pag.set_mark_pub(3, 2, EdgeMark::Circle);
    let changed = Fci::rule_r4_pub(&mut pag);
    assert!(changed);
    assert_eq!(pag.mark_pub(2, 3), EdgeMark::Arrow);
}

#[test]
fn test_invalid_config_alpha_zero_returns_err() {
    let res = Fci::new(FciConfig {
        alpha: 0.0,
        max_cond_set_size: 3,
    });
    assert!(res.is_err());
}

#[test]
fn test_invalid_config_alpha_one_returns_err() {
    let res = Fci::new(FciConfig {
        alpha: 1.0,
        max_cond_set_size: 3,
    });
    assert!(res.is_err());
    let res2 = Fci::new(FciConfig {
        alpha: -0.1,
        max_cond_set_size: 3,
    });
    assert!(res2.is_err());
}

#[test]
fn test_wrong_data_length_returns_err() {
    let fci = Fci::new(FciConfig::default()).unwrap();
    let data = vec![0.0_f64; 10];
    let res = fci.fit(&data, 10, 3);
    assert!(res.is_err());
}

#[test]
fn test_single_variable_handled() {
    let fci = Fci::new(FciConfig::default()).unwrap();
    let pag = fci.fit(&[], 0, 1).unwrap();
    assert_eq!(pag.n_vars, 1);
    assert_eq!(count_edges(&pag), 0);
}

#[test]
fn test_zero_variable_returns_err() {
    let fci = Fci::new(FciConfig::default()).unwrap();
    let res = fci.fit(&[], 0, 0);
    assert!(res.is_err());
}

#[test]
fn test_normal_quantile_basic() {
    let z = normal_quantile_two_sided(0.05);
    assert!((z - 1.96).abs() < 1e-2);
    let z2 = normal_quantile_two_sided(0.01);
    assert!((z2 - 2.576).abs() < 1e-2);
}
