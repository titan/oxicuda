//! Loopy belief propagation on a pairwise MRF (sum-product + max-product).

use super::mrf::Mrf;
use crate::error::{SeqError, SeqResult};

/// BP configuration.
#[derive(Debug, Clone, Copy)]
pub struct BpConfig {
    pub max_iter: usize,
    pub tol: f64,
    pub damping: f64,
}

impl Default for BpConfig {
    fn default() -> Self {
        Self {
            max_iter: 50,
            tol: 1e-5,
            damping: 0.5,
        }
    }
}

/// Result of BP marginal inference.
#[derive(Debug, Clone)]
pub struct BpResult {
    pub marginals: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

/// Loopy sum-product BP on a pairwise MRF.  Computes approximate node marginals.
///
/// Internally uses *log-space* messages so values do not under/overflow on
/// high-energy graphs.
pub fn loopy_bp_marginals(mrf: &Mrf, cfg: &BpConfig) -> SeqResult<BpResult> {
    if cfg.max_iter == 0 {
        return Err(SeqError::InvalidConfiguration(
            "max_iter must be > 0".to_string(),
        ));
    }
    let nl = mrf.n_labels;
    let l2 = nl * nl;
    // Directed messages per (edge_idx, direction): u→v at idx*2, v→u at idx*2+1.
    let n_messages = mrf.edges.len() * 2;
    let mut log_msg = vec![0.0; n_messages * nl];
    let mut new_log_msg = log_msg.clone();
    let mut converged = false;
    let mut iters = 0;

    for it in 0..cfg.max_iter {
        iters = it + 1;
        for (e_idx, &(u, v)) in mrf.edges.iter().enumerate() {
            for &(src, _dst, msg_idx, _opp_idx) in &[
                (u, v, e_idx * 2, e_idx * 2 + 1),
                (v, u, e_idx * 2 + 1, e_idx * 2),
            ] {
                // Build message[l_dst] = logsumexp_{l_src}(unary[src][l_src] +
                //                       pairwise[edge][l_u, l_v] (oriented) +
                //                       Σ_k≠e log_msg[k→src][l_src]).
                let mut out = vec![f64::NEG_INFINITY; nl];
                for l_dst in 0..nl {
                    let mut terms = vec![0.0; nl];
                    for l_src in 0..nl {
                        let mut acc = -mrf.unary[src * nl + l_src];
                        // pairwise oriented: edge is stored as (u, v); apply correct order.
                        let psi = if src == u {
                            mrf.pairwise[e_idx * l2 + l_src * nl + l_dst]
                        } else {
                            mrf.pairwise[e_idx * l2 + l_dst * nl + l_src]
                        };
                        acc -= psi;
                        // Incoming messages from all neighbours of `src` except this edge.
                        for (k_idx, &(uu, vv)) in mrf.edges.iter().enumerate() {
                            if k_idx == e_idx {
                                continue;
                            }
                            let in_msg = if uu == src {
                                &log_msg[(k_idx * 2 + 1) * nl..]
                            } else if vv == src {
                                &log_msg[(k_idx * 2) * nl..]
                            } else {
                                continue;
                            };
                            acc += in_msg[l_src];
                        }
                        terms[l_src] = acc;
                    }
                    out[l_dst] = logsumexp_in(&terms);
                }
                // Normalise (log-domain)
                let m = out.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                for v in out.iter_mut() {
                    *v -= m;
                }
                // Damping + write
                for l in 0..nl {
                    new_log_msg[msg_idx * nl + l] =
                        (1.0 - cfg.damping) * log_msg[msg_idx * nl + l] + cfg.damping * out[l];
                }
            }
        }
        let mut max_diff = 0.0_f64;
        for k in 0..log_msg.len() {
            let d = (new_log_msg[k] - log_msg[k]).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        log_msg.copy_from_slice(&new_log_msg);
        if max_diff < cfg.tol {
            converged = true;
            break;
        }
    }

    // Compute marginals
    let mut marginals = vec![0.0; mrf.n_nodes * nl];
    for i in 0..mrf.n_nodes {
        let mut log_b = vec![0.0; nl];
        for l in 0..nl {
            log_b[l] = -mrf.unary[i * nl + l];
        }
        for (e_idx, &(u, v)) in mrf.edges.iter().enumerate() {
            if u == i {
                for l in 0..nl {
                    log_b[l] += log_msg[(e_idx * 2 + 1) * nl + l];
                }
            }
            if v == i {
                for l in 0..nl {
                    log_b[l] += log_msg[(e_idx * 2) * nl + l];
                }
            }
        }
        let m = log_b.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut s = 0.0;
        let mut exps = vec![0.0; nl];
        for l in 0..nl {
            exps[l] = (log_b[l] - m).exp();
            s += exps[l];
        }
        for l in 0..nl {
            marginals[i * nl + l] = if s > 0.0 {
                exps[l] / s
            } else {
                1.0 / nl as f64
            };
        }
    }
    Ok(BpResult {
        marginals,
        iterations: iters,
        converged,
    })
}

/// Loopy max-product BP for MAP inference.  Returns the MAP labelling.
pub fn loopy_bp_map(mrf: &Mrf, cfg: &BpConfig) -> SeqResult<Vec<usize>> {
    if cfg.max_iter == 0 {
        return Err(SeqError::InvalidConfiguration(
            "max_iter must be > 0".to_string(),
        ));
    }
    let nl = mrf.n_labels;
    let l2 = nl * nl;
    let n_messages = mrf.edges.len() * 2;
    let mut log_msg = vec![0.0; n_messages * nl];
    let mut new_log_msg = log_msg.clone();

    for _ in 0..cfg.max_iter {
        for (e_idx, &(u, v)) in mrf.edges.iter().enumerate() {
            for &(src, dst, msg_idx) in &[(u, v, e_idx * 2), (v, u, e_idx * 2 + 1)] {
                let _ = dst;
                let mut out = vec![f64::NEG_INFINITY; nl];
                for l_dst in 0..nl {
                    let mut best = f64::NEG_INFINITY;
                    for l_src in 0..nl {
                        let mut acc = -mrf.unary[src * nl + l_src];
                        let psi = if src == u {
                            mrf.pairwise[e_idx * l2 + l_src * nl + l_dst]
                        } else {
                            mrf.pairwise[e_idx * l2 + l_dst * nl + l_src]
                        };
                        acc -= psi;
                        for (k_idx, &(uu, vv)) in mrf.edges.iter().enumerate() {
                            if k_idx == e_idx {
                                continue;
                            }
                            let in_msg = if uu == src {
                                &log_msg[(k_idx * 2 + 1) * nl..]
                            } else if vv == src {
                                &log_msg[(k_idx * 2) * nl..]
                            } else {
                                continue;
                            };
                            acc += in_msg[l_src];
                        }
                        if acc > best {
                            best = acc;
                        }
                    }
                    out[l_dst] = best;
                }
                let m = out.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                for v in out.iter_mut() {
                    *v -= m;
                }
                for l in 0..nl {
                    new_log_msg[msg_idx * nl + l] =
                        (1.0 - cfg.damping) * log_msg[msg_idx * nl + l] + cfg.damping * out[l];
                }
            }
        }
        log_msg.copy_from_slice(&new_log_msg);
    }

    // Decode
    let mut labels = vec![0usize; mrf.n_nodes];
    for i in 0..mrf.n_nodes {
        let mut best_l = 0usize;
        let mut best_v = f64::NEG_INFINITY;
        for l in 0..nl {
            let mut acc = -mrf.unary[i * nl + l];
            for (e_idx, &(u, v)) in mrf.edges.iter().enumerate() {
                if u == i {
                    acc += log_msg[(e_idx * 2 + 1) * nl + l];
                }
                if v == i {
                    acc += log_msg[(e_idx * 2) * nl + l];
                }
            }
            if acc > best_v {
                best_v = acc;
                best_l = l;
            }
        }
        labels[i] = best_l;
    }
    Ok(labels)
}

fn logsumexp_in(xs: &[f64]) -> f64 {
    let m = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let s: f64 = xs.iter().map(|x| (x - m).exp()).sum();
    m + s.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bp_marginals_normalise() {
        let m = Mrf::new(
            3,
            2,
            vec![(0, 1), (1, 2)],
            vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        )
        .expect("ok");
        let res = loopy_bp_marginals(&m, &BpConfig::default()).expect("ok");
        for i in 0..m.n_nodes {
            let s: f64 = res.marginals[i * m.n_labels..(i + 1) * m.n_labels]
                .iter()
                .sum();
            assert!((s - 1.0).abs() < 1e-6, "row sum {s}");
        }
    }

    #[test]
    fn bp_map_runs() {
        let m = Mrf::new(
            3,
            2,
            vec![(0, 1), (1, 2)],
            vec![0.0, 5.0, 5.0, 0.0, 0.0, 5.0],
            vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        )
        .expect("ok");
        let labels = loopy_bp_map(&m, &BpConfig::default()).expect("ok");
        assert_eq!(labels.len(), 3);
    }
}
