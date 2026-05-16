//! Higher-Order Orthogonal Iteration (HOOI) — iterative refinement of HOSVD.
//!
//! Alternating: with `U_a, U_b` fixed, the optimal `U_c` is the leading-`r_c` left
//! singular vectors of the unfolding of `T ×_a U_a^T ×_b U_b^T`.

use crate::TnResult;
use crate::tucker::hosvd::{
    TuckerResult, hosvd, mode_apply_left_transpose, mode_unfold_then_svd_left,
};

/// Refine an HOSVD via HOOI. `max_iter` iterations are performed; convergence is
/// measured by the change in the Frobenius norm of the core tensor.
pub fn hooi(
    t: &[f64],
    d0: usize,
    d1: usize,
    d2: usize,
    r0: usize,
    r1: usize,
    r2: usize,
    max_iter: usize,
    tol: f64,
) -> TnResult<TuckerResult> {
    let mut res = hosvd(t, d0, d1, d2, r0, r1, r2)?;
    let mut prev_norm = fro(&res.core);
    for _ in 0..max_iter {
        // Update U0 from T ×_1 U1^T ×_2 U2^T
        let t1 = apply_two_modes(t, &res.u1, &res.u2, d0, d1, d2, r1, r2);
        let u0 = mode_unfold_then_svd_left(&t1, d0, r1, r2, 0, r0)?;
        res.u0 = u0;
        // Update U1
        let t2 = apply_two_modes_alt(t, &res.u0, &res.u2, d0, d1, d2, r0, r2, "01_2");
        let u1 = mode_unfold_then_svd_left(&t2, r0, d1, r2, 1, r1)?;
        res.u1 = u1;
        // Update U2
        let t3 = apply_two_modes_alt(t, &res.u0, &res.u1, d0, d1, d2, r0, r1, "01_1");
        let u2 = mode_unfold_then_svd_left(&t3, r0, r1, d2, 2, r2)?;
        res.u2 = u2;
        // Update core
        res.core = mode_apply_left_transpose(t, &res.u0, &res.u1, &res.u2, d0, d1, d2, r0, r1, r2);
        let cur_norm = fro(&res.core);
        if (cur_norm - prev_norm).abs() < tol {
            break;
        }
        prev_norm = cur_norm;
    }
    Ok(res)
}

fn fro(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Compute `T ×_1 U1^T ×_2 U2^T` → shape `(d0, r1, r2)`.
fn apply_two_modes(
    t: &[f64],
    u1: &[f64],
    u2: &[f64],
    d0: usize,
    d1: usize,
    d2: usize,
    r1: usize,
    r2: usize,
) -> Vec<f64> {
    let mut t1 = vec![0.0; d0 * r1 * d2];
    for i in 0..d0 {
        for b in 0..r1 {
            for k in 0..d2 {
                let mut acc = 0.0;
                for j in 0..d1 {
                    acc += u1[j * r1 + b] * t[(i * d1 + j) * d2 + k];
                }
                t1[(i * r1 + b) * d2 + k] = acc;
            }
        }
    }
    let mut out = vec![0.0; d0 * r1 * r2];
    for i in 0..d0 {
        for b in 0..r1 {
            for c in 0..r2 {
                let mut acc = 0.0;
                for k in 0..d2 {
                    acc += u2[k * r2 + c] * t1[(i * r1 + b) * d2 + k];
                }
                out[(i * r1 + b) * r2 + c] = acc;
            }
        }
    }
    out
}

/// Compute `T ×_0 U0^T ×_2 U2^T` (mode == "01_2") or `T ×_0 U0^T ×_1 U1^T` (mode == "01_1").
#[allow(clippy::too_many_arguments)]
fn apply_two_modes_alt(
    t: &[f64],
    u0: &[f64],
    other: &[f64],
    d0: usize,
    d1: usize,
    d2: usize,
    r0: usize,
    r_other: usize,
    which: &str,
) -> Vec<f64> {
    // Apply U0 first
    let mut t1 = vec![0.0; r0 * d1 * d2];
    for a in 0..r0 {
        for j in 0..d1 {
            for k in 0..d2 {
                let mut acc = 0.0;
                for i in 0..d0 {
                    acc += u0[i * r0 + a] * t[(i * d1 + j) * d2 + k];
                }
                t1[(a * d1 + j) * d2 + k] = acc;
            }
        }
    }
    match which {
        "01_2" => {
            // Apply U2 on last mode
            let mut out = vec![0.0; r0 * d1 * r_other];
            for a in 0..r0 {
                for j in 0..d1 {
                    for c in 0..r_other {
                        let mut acc = 0.0;
                        for k in 0..d2 {
                            acc += other[k * r_other + c] * t1[(a * d1 + j) * d2 + k];
                        }
                        out[(a * d1 + j) * r_other + c] = acc;
                    }
                }
            }
            out
        }
        _ => {
            // "01_1": apply U1 on middle mode
            let mut out = vec![0.0; r0 * r_other * d2];
            for a in 0..r0 {
                for b in 0..r_other {
                    for k in 0..d2 {
                        let mut acc = 0.0;
                        for j in 0..d1 {
                            acc += other[j * r_other + b] * t1[(a * d1 + j) * d2 + k];
                        }
                        out[(a * r_other + b) * d2 + k] = acc;
                    }
                }
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::tucker::hosvd::tucker_reconstruct;

    #[test]
    fn hooi_improves_full_rank() {
        let mut rng = LcgRng::new(11);
        let d0 = 3;
        let d1 = 3;
        let d2 = 3;
        let data: Vec<f64> = (0..d0 * d1 * d2).map(|_| rng.next_normal()).collect();
        let res = hooi(&data, d0, d1, d2, d0, d1, d2, 4, 1e-10).expect("ok");
        let rec = tucker_reconstruct(&res);
        let diff: f64 = data
            .iter()
            .zip(&rec)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        assert!(diff < 1e-8);
    }
}
