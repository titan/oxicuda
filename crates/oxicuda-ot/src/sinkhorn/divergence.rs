//! Debiased Sinkhorn divergence.
//!
//! ```text
//! S_ε(a, b) = OT_ε(a, b) − ½ (OT_ε(a, a) + OT_ε(b, b))
//! ```
//!
//! This subtracts the entropic self-cost so that `S_ε(a, a) = 0`, removing
//! the bias of vanilla entropic OT and recovering a true divergence between
//! distributions.

use crate::error::OtResult;
use crate::sinkhorn::sinkhorn::{SinkhornConfig, sinkhorn};

/// Compute the debiased Sinkhorn divergence between histograms `a` and `b`.
///
/// `c_ab` is the cost matrix between source `a` and target `b` (shape `m × n`).
/// `c_aa` is the self-cost of `a` (shape `m × m`); `c_bb` is the self-cost of
/// `b` (shape `n × n`). The function runs Sinkhorn three times and returns
/// `OT_ε(a,b) − ½(OT_ε(a,a) + OT_ε(b,b))`.
pub fn sinkhorn_divergence(
    c_ab: &[f32],
    c_aa: &[f32],
    c_bb: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    cfg: &SinkhornConfig,
) -> OtResult<f32> {
    let r_ab = sinkhorn(c_ab, a, b, m, n, cfg)?;
    let r_aa = sinkhorn(c_aa, a, a, m, m, cfg)?;
    let r_bb = sinkhorn(c_bb, b, b, n, n, cfg)?;
    Ok(r_ab.cost - 0.5 * (r_aa.cost + r_bb.cost))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn squared_distance_matrix(x: &[f32], y: &[f32], dim: usize) -> Vec<f32> {
        let nx = x.len() / dim;
        let ny = y.len() / dim;
        let mut c = vec![0.0_f32; nx * ny];
        for i in 0..nx {
            for j in 0..ny {
                let mut s = 0.0_f32;
                for d in 0..dim {
                    let diff = x[i * dim + d] - y[j * dim + d];
                    s += diff * diff;
                }
                c[i * ny + j] = s;
            }
        }
        c
    }

    #[test]
    fn divergence_self_is_near_zero() {
        let dim = 1;
        let x = vec![0.0_f32, 1.0, 2.0];
        let a = vec![1.0_f32 / 3.0; 3];
        let c_aa = squared_distance_matrix(&x, &x, dim);
        let cfg = SinkhornConfig {
            eps: 0.5,
            max_iter: 2000,
            tol: 1e-4,
        };
        let s = sinkhorn_divergence(&c_aa, &c_aa, &c_aa, &a, &a, 3, 3, &cfg).expect("ok");
        assert!(s.abs() < 1e-3, "S(a,a)={s} should be ≈ 0");
    }

    #[test]
    fn divergence_symmetric() {
        let dim = 1;
        let x = vec![0.0_f32, 1.0];
        let y = vec![1.0_f32, 2.0];
        let a = vec![0.5_f32, 0.5];
        let b = vec![0.5_f32, 0.5];
        let c_ab = squared_distance_matrix(&x, &y, dim);
        let c_ba = squared_distance_matrix(&y, &x, dim);
        let c_aa = squared_distance_matrix(&x, &x, dim);
        let c_bb = squared_distance_matrix(&y, &y, dim);
        let cfg = SinkhornConfig {
            eps: 0.5,
            max_iter: 2000,
            tol: 1e-4,
        };
        let s_ab = sinkhorn_divergence(&c_ab, &c_aa, &c_bb, &a, &b, 2, 2, &cfg).expect("ok");
        let s_ba = sinkhorn_divergence(&c_ba, &c_bb, &c_aa, &b, &a, 2, 2, &cfg).expect("ok");
        assert!((s_ab - s_ba).abs() < 1e-3, "S(a,b)={s_ab} ≠ S(b,a)={s_ba}");
    }

    #[test]
    fn divergence_non_negative_for_disjoint() {
        let dim = 1;
        let x = vec![0.0_f32];
        let y = vec![5.0_f32];
        let a = vec![1.0_f32];
        let b = vec![1.0_f32];
        let c_ab = squared_distance_matrix(&x, &y, dim);
        let c_aa = vec![0.0_f32];
        let c_bb = vec![0.0_f32];
        let cfg = SinkhornConfig {
            eps: 0.5,
            max_iter: 200,
            tol: 1e-5,
        };
        let s = sinkhorn_divergence(&c_ab, &c_aa, &c_bb, &a, &b, 1, 1, &cfg).expect("ok");
        assert!(s >= -1e-3, "expected non-negative divergence, got {s}");
    }
}
