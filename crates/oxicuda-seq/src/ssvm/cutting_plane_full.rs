//! Full n-slack cutting-plane structural SVM optimiser
//! (Joachims, Finley, Yu 2009, "Cutting-plane training of structural SVMs").
//!
//! The training objective is the standard n-slack QP
//!
//! ```text
//! min_{w, ξ ≥ 0}  ½ ‖w‖² + C · Σ_i ξ_i
//! s.t.            ∀ i, ∀ (Δψ_{ij}, ℓ_{ij}) ∈ working_set_i :
//!                 ⟨w, Δψ_{ij}⟩ ≥ ℓ_{ij} − ξ_i
//! ```
//!
//! where `Δψ = ψ(x_i, y_i) − ψ(x_i, ŷ)` and `ℓ` is the structured loss.  At
//! each outer iteration, the separation oracle for example `i` returns a
//! candidate `(Δψ, ℓ)`; the constraint is added if it is violated by more
//! than `ε`.  The QP master problem over the accumulated working set is
//! solved in its **dual form** using projected gradient ascent on
//! `α_{ij} ≥ 0`, `Σ_j α_{ij} ≤ C`, with the primal recovered via
//! `w = Σ_{ij} α_{ij} Δψ_{ij}`.
//!
//! The "Full" prefix distinguishes this from the simpler sub-gradient
//! `CuttingPlaneConfig` already in [`super::cutting_plane`].

use crate::error::{SeqError, SeqResult};

/// Configuration for [`FullCuttingPlaneSvm`].
#[derive(Debug, Clone, Copy)]
pub struct FullCuttingPlaneConfig {
    /// Regularisation trade-off `C > 0`.  Larger ⇒ less regularisation.
    pub c_reg: f32,
    /// Constraint-violation tolerance.  A new constraint is added only if it
    /// is violated by more than `epsilon`.
    pub epsilon: f32,
    /// Maximum outer (separation) iterations.
    pub max_iter: usize,
}

impl Default for FullCuttingPlaneConfig {
    fn default() -> Self {
        Self {
            c_reg: 1.0,
            epsilon: 1e-3,
            max_iter: 100,
        }
    }
}

/// Output of [`FullCuttingPlaneSvm::train`].
#[derive(Debug, Clone)]
pub struct FullCuttingPlaneResult {
    /// Final weight vector.
    pub w: Vec<f32>,
    /// Per-example slack variables `ξ_i ≥ 0`.
    pub xi: Vec<f32>,
    /// Outer iterations actually performed.
    pub iterations: usize,
    /// Total number of constraints accumulated across all examples.
    pub n_constraints: usize,
    /// `true` if the last outer iteration added zero constraints.
    pub converged: bool,
}

/// Stateless namespace for the full cutting-plane SVM routines.
pub struct FullCuttingPlaneSvm;

impl FullCuttingPlaneSvm {
    /// Solve the QP master
    ///
    /// ```text
    ///   min_{w, ξ ≥ 0} ½ ‖w‖² + C Σ_i ξ_i
    ///   s.t.           ∀ i, ∀ j : ⟨w, Δψ_{ij}⟩ + ξ_i ≥ ℓ_{ij}
    /// ```
    ///
    /// over the supplied per-example working sets.  `constraints[i]` is the
    /// working set for example `i`; each entry is `(Δψ, ℓ)`.  Returns
    /// `(w, xi)` of lengths `n_features` and `n_examples` respectively.
    ///
    /// The dual is
    ///
    /// ```text
    ///   max_{α ≥ 0, Σ_j α_{ij} ≤ C}
    ///       Σ_{ij} α_{ij} ℓ_{ij}  −  ½ ‖Σ_{ij} α_{ij} Δψ_{ij}‖²
    /// ```
    ///
    /// and is solved by projected gradient ascent on `α`.  The primal `w` is
    /// recovered as `w = Σ_{ij} α_{ij} Δψ_{ij}` and `ξ_i` from the active
    /// constraints.
    pub fn qp_master(
        constraints: &[Vec<(Vec<f32>, f32)>],
        n_features: usize,
        c_reg: f32,
    ) -> SeqResult<(Vec<f32>, Vec<f32>)> {
        if c_reg <= 0.0 || c_reg.is_nan() {
            return Err(SeqError::InvalidParameter {
                name: "c_reg".to_string(),
                value: c_reg as f64,
            });
        }
        let n_examples = constraints.len();
        // Validate constraint shape.
        for (i, ws) in constraints.iter().enumerate() {
            for (j, (dpsi, _)) in ws.iter().enumerate() {
                if dpsi.len() != n_features {
                    return Err(SeqError::ShapeMismatch {
                        expected: n_features,
                        got: dpsi.len(),
                    });
                }
                let _ = (i, j);
            }
        }
        if n_examples == 0 {
            // Trivial case: no examples ⇒ no constraints ⇒ w = 0, xi = [].
            return Ok((vec![0.0; n_features], Vec::new()));
        }
        let total_c: usize = constraints.iter().map(|ws| ws.len()).sum();
        if total_c == 0 {
            return Ok((vec![0.0; n_features], vec![0.0; n_examples]));
        }

        // Flatten (i, j) indexing.
        // alpha[k] ↔ constraint k = (i_k, j_k).
        let mut alpha: Vec<f32> = vec![0.0; total_c];
        let mut owner: Vec<usize> = Vec::with_capacity(total_c);
        let mut loss: Vec<f32> = Vec::with_capacity(total_c);
        let mut dpsi: Vec<&Vec<f32>> = Vec::with_capacity(total_c);
        for (i, ws) in constraints.iter().enumerate() {
            for (psi, ell) in ws.iter() {
                owner.push(i);
                loss.push(*ell);
                dpsi.push(psi);
            }
        }

        // Projected gradient ascent on the dual.
        // Gradient of the dual w.r.t. α_k:
        //   g_k = ℓ_k − ⟨w, Δψ_k⟩,    where w = Σ_k α_k Δψ_k.
        // After each step project onto the simplex-like set
        //   α ≥ 0, Σ_{k: owner_k = i} α_k ≤ C
        // by clamping non-negatively then renormalising per-owner if its sum
        // exceeds C.
        //
        // Step size: a conservative bound on the dual Hessian diagonal is the
        // self-inner-product `⟨Δψ_k, Δψ_k⟩`.  Use the inverse of the largest
        // such norm-squared as the global step.
        let mut max_norm_sq: f32 = 0.0;
        for d in &dpsi {
            let mut n2 = 0.0_f32;
            for v in d.iter() {
                n2 += *v * *v;
            }
            if n2 > max_norm_sq {
                max_norm_sq = n2;
            }
        }
        let step: f32 = if max_norm_sq > 0.0 {
            1.0 / max_norm_sq
        } else {
            1.0
        };

        // Inner solve: iterate to convergence on α.
        let inner_iters = 500;
        let inner_tol: f32 = 1e-6;
        let mut prev_obj = f32::NEG_INFINITY;
        for _it in 0..inner_iters {
            // Compute w = Σ_k α_k Δψ_k.
            let mut w = vec![0.0_f32; n_features];
            for k in 0..total_c {
                let a = alpha[k];
                if a == 0.0 {
                    continue;
                }
                let pk = dpsi[k];
                for f in 0..n_features {
                    w[f] += a * pk[f];
                }
            }
            // Gradient.
            let mut grad = vec![0.0_f32; total_c];
            for k in 0..total_c {
                let mut dot = 0.0_f32;
                let pk = dpsi[k];
                for f in 0..n_features {
                    dot += w[f] * pk[f];
                }
                grad[k] = loss[k] - dot;
            }
            // Ascent step.
            for k in 0..total_c {
                alpha[k] += step * grad[k];
                if alpha[k] < 0.0 {
                    alpha[k] = 0.0;
                }
            }
            // Per-owner cap: renormalise if the per-example sum exceeds C.
            // Track sums per example.
            let mut sum_i = vec![0.0_f32; n_examples];
            for k in 0..total_c {
                sum_i[owner[k]] += alpha[k];
            }
            for k in 0..total_c {
                let i = owner[k];
                let s = sum_i[i];
                if s > c_reg && s > 0.0 {
                    let scale = c_reg / s;
                    alpha[k] *= scale;
                }
            }

            // Check dual-objective change for early stopping.
            let mut obj = 0.0_f32;
            for k in 0..total_c {
                obj += alpha[k] * loss[k];
            }
            // Recompute w · w with the (possibly rescaled) alphas.
            let mut w = vec![0.0_f32; n_features];
            for k in 0..total_c {
                let a = alpha[k];
                if a == 0.0 {
                    continue;
                }
                let pk = dpsi[k];
                for f in 0..n_features {
                    w[f] += a * pk[f];
                }
            }
            let mut ww = 0.0_f32;
            for f in 0..n_features {
                ww += w[f] * w[f];
            }
            obj -= 0.5 * ww;
            if (obj - prev_obj).abs() < inner_tol {
                break;
            }
            prev_obj = obj;
        }

        // Recover (w, xi).
        let mut w = vec![0.0_f32; n_features];
        for k in 0..total_c {
            let a = alpha[k];
            if a == 0.0 {
                continue;
            }
            let pk = dpsi[k];
            for f in 0..n_features {
                w[f] += a * pk[f];
            }
        }
        // ξ_i = max(0, max_j (ℓ_{ij} − ⟨w, Δψ_{ij}⟩)).
        let mut xi = vec![0.0_f32; n_examples];
        for (i, ws) in constraints.iter().enumerate() {
            let mut best = 0.0_f32;
            for (psi, ell) in ws.iter() {
                let mut dot = 0.0_f32;
                for f in 0..n_features {
                    dot += w[f] * psi[f];
                }
                let v = *ell - dot;
                if v > best {
                    best = v;
                }
            }
            xi[i] = best;
        }

        Ok((w, xi))
    }

    /// Train an n-slack cutting-plane structural SVM.
    ///
    /// `separation_oracle(w, i)` returns the most-violating constraint
    /// `(Δψ, ℓ)` for example `i` at the current `w`.  Implementations should
    /// solve `ŷ = argmax_y [ℓ(y_i, y) + ⟨w, ψ(x_i, y) − ψ(x_i, y_i)⟩]` and
    /// return `(ψ(x_i, y_i) − ψ(x_i, ŷ), ℓ(y_i, ŷ))`.
    pub fn train<O>(
        n_examples: usize,
        n_features: usize,
        separation_oracle: O,
        cfg: &FullCuttingPlaneConfig,
    ) -> SeqResult<FullCuttingPlaneResult>
    where
        O: Fn(&[f32], usize) -> SeqResult<(Vec<f32>, f32)>,
    {
        if n_examples == 0 {
            return Err(SeqError::InvalidParameter {
                name: "n_examples".to_string(),
                value: 0.0,
            });
        }
        if n_features == 0 {
            return Err(SeqError::InvalidParameter {
                name: "n_features".to_string(),
                value: 0.0,
            });
        }
        if cfg.c_reg <= 0.0 || cfg.c_reg.is_nan() {
            return Err(SeqError::InvalidParameter {
                name: "c_reg".to_string(),
                value: cfg.c_reg as f64,
            });
        }
        if cfg.epsilon <= 0.0 || cfg.epsilon.is_nan() {
            return Err(SeqError::InvalidParameter {
                name: "epsilon".to_string(),
                value: cfg.epsilon as f64,
            });
        }
        if cfg.max_iter == 0 {
            return Err(SeqError::InvalidParameter {
                name: "max_iter".to_string(),
                value: 0.0,
            });
        }

        let mut working_set: Vec<Vec<(Vec<f32>, f32)>> = vec![Vec::new(); n_examples];
        let mut w: Vec<f32> = vec![0.0; n_features];
        let mut xi: Vec<f32> = vec![0.0; n_examples];
        let mut iterations = 0_usize;
        let mut converged = false;

        for it in 0..cfg.max_iter {
            iterations = it + 1;
            let mut added = false;
            for i in 0..n_examples {
                let (dpsi, ell) = separation_oracle(&w, i)?;
                if dpsi.len() != n_features {
                    return Err(SeqError::ShapeMismatch {
                        expected: n_features,
                        got: dpsi.len(),
                    });
                }
                // Current margin gap on this candidate.
                let mut dot = 0.0_f32;
                for f in 0..n_features {
                    dot += w[f] * dpsi[f];
                }
                let viol = ell - dot;
                if viol > xi[i] + cfg.epsilon {
                    working_set[i].push((dpsi, ell));
                    added = true;
                }
            }
            if !added {
                converged = true;
                break;
            }
            let (w_new, xi_new) = Self::qp_master(&working_set, n_features, cfg.c_reg)?;
            w = w_new;
            xi = xi_new;
        }

        let n_constraints: usize = working_set.iter().map(|ws| ws.len()).sum();
        Ok(FullCuttingPlaneResult {
            w,
            xi,
            iterations,
            n_constraints,
            converged,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn default_cfg() -> FullCuttingPlaneConfig {
        FullCuttingPlaneConfig {
            c_reg: 1.0,
            epsilon: 1e-3,
            max_iter: 50,
        }
    }

    #[test]
    fn train_zero_oracle_returns_zero_weights() {
        let oracle =
            |_w: &[f32], _i: usize| -> SeqResult<(Vec<f32>, f32)> { Ok((vec![0.0; 3], 0.0)) };
        let res = FullCuttingPlaneSvm::train(2, 3, oracle, &default_cfg()).expect("ok");
        for v in &res.w {
            assert!(v.abs() < 1e-7);
        }
        for v in &res.xi {
            assert!(v.abs() < 1e-7);
        }
        assert!(res.converged);
        assert_eq!(res.n_constraints, 0);
        assert_eq!(res.iterations, 1);
    }

    #[test]
    fn qp_master_single_constraint_closed_form() {
        // Single example with a single constraint (Δψ, ℓ).  The dual is
        // max_{0 ≤ α ≤ C} α ℓ − ½ α² ‖Δψ‖²
        // The unconstrained optimum is α* = ℓ / ‖Δψ‖².  If α* > C we clamp to
        // C.  Here ℓ = 1, Δψ = (1, 0) ⇒ ‖Δψ‖² = 1, C = 0.5 ⇒ α = 0.5 ⇒
        // w = α · Δψ = (0.5, 0).
        let constraints: Vec<Vec<(Vec<f32>, f32)>> = vec![vec![(vec![1.0, 0.0], 1.0)]];
        let (w, xi) = FullCuttingPlaneSvm::qp_master(&constraints, 2, 0.5).expect("ok");
        assert!((w[0] - 0.5).abs() < 1e-3);
        assert!(w[1].abs() < 1e-3);
        // ξ = max(0, ℓ − w·Δψ) = 1 − 0.5 = 0.5.
        assert!((xi[0] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn qp_master_single_constraint_within_cap() {
        // Same example but C = 10 so α reaches its unconstrained optimum α* = 1.
        // w = (1, 0); ξ = max(0, 1 − 1) = 0.
        let constraints: Vec<Vec<(Vec<f32>, f32)>> = vec![vec![(vec![1.0, 0.0], 1.0)]];
        let (w, xi) = FullCuttingPlaneSvm::qp_master(&constraints, 2, 10.0).expect("ok");
        assert!((w[0] - 1.0).abs() < 1e-3);
        assert!(w[1].abs() < 1e-3);
        assert!(xi[0].abs() < 1e-3);
    }

    #[test]
    fn train_synthetic_separable_problem_converges() {
        // Two examples; the oracle returns one constant constraint per
        // example, then nothing more once the margin is satisfied.
        // Example 0: Δψ = (1, 0), ℓ = 1.
        // Example 1: Δψ = (0, 1), ℓ = 1.
        // After one master solve with large C the weight is w ≈ (1, 1) and
        // both margins are satisfied.
        let calls: Cell<usize> = Cell::new(0);
        let oracle = |w: &[f32], i: usize| -> SeqResult<(Vec<f32>, f32)> {
            calls.set(calls.get() + 1);
            let dpsi = if i == 0 {
                vec![1.0_f32, 0.0]
            } else {
                vec![0.0_f32, 1.0]
            };
            let ell = 1.0_f32;
            let mut dot = 0.0_f32;
            for f in 0..2 {
                dot += w[f] * dpsi[f];
            }
            // Return *something*; the caller decides if it is violating.
            let _ = ell - dot;
            Ok((dpsi, ell))
        };
        let cfg = FullCuttingPlaneConfig {
            c_reg: 10.0,
            epsilon: 1e-3,
            max_iter: 10,
        };
        let res = FullCuttingPlaneSvm::train(2, 2, oracle, &cfg).expect("ok");
        assert!(res.iterations >= 1);
        assert!(res.converged);
        // The QP master with two orthogonal constraints and large C gives
        // w = (1, 1).
        assert!((res.w[0] - 1.0).abs() < 0.1);
        assert!((res.w[1] - 1.0).abs() < 0.1);
        assert!(calls.get() >= 2);
    }

    #[test]
    fn train_constraints_grow_with_iterations() {
        // An oracle that always returns a violating constraint at the same
        // Δψ but a different ℓ shifted by the iteration count would force
        // monotone growth.  We use a fresh constraint each call to drive the
        // working set up.
        let counter: Cell<usize> = Cell::new(0);
        let oracle = |_w: &[f32], _i: usize| -> SeqResult<(Vec<f32>, f32)> {
            let c = counter.get();
            counter.set(c + 1);
            // Δψ_k = e_0 always, ℓ grows so the constraint stays violating.
            Ok((vec![1.0_f32, 0.0], 10.0 + c as f32))
        };
        let cfg = FullCuttingPlaneConfig {
            c_reg: 1.0,
            epsilon: 1e-3,
            max_iter: 3,
        };
        let res = FullCuttingPlaneSvm::train(2, 2, oracle, &cfg).expect("ok");
        assert!(res.n_constraints >= 2);
    }

    #[test]
    fn train_converged_flag_true_when_no_new_violation() {
        // First call returns a strong violation; subsequent calls return the
        // same Δψ so the constraint is already satisfied → no new
        // constraint added → converged = true.
        let calls: Cell<usize> = Cell::new(0);
        let oracle = |_w: &[f32], _i: usize| -> SeqResult<(Vec<f32>, f32)> {
            let c = calls.get();
            calls.set(c + 1);
            Ok((vec![1.0_f32, 0.0], 1.0))
        };
        let cfg = FullCuttingPlaneConfig {
            c_reg: 10.0,
            epsilon: 1e-3,
            max_iter: 10,
        };
        let res = FullCuttingPlaneSvm::train(1, 2, oracle, &cfg).expect("ok");
        assert!(res.converged);
    }

    #[test]
    fn train_is_deterministic() {
        let oracle =
            |_w: &[f32], _i: usize| -> SeqResult<(Vec<f32>, f32)> { Ok((vec![0.5_f32, 0.5], 0.5)) };
        let cfg = default_cfg();
        let r1 = FullCuttingPlaneSvm::train(2, 2, oracle, &cfg).expect("ok");
        let r2 = FullCuttingPlaneSvm::train(2, 2, oracle, &cfg).expect("ok");
        assert_eq!(r1.iterations, r2.iterations);
        assert_eq!(r1.n_constraints, r2.n_constraints);
        for i in 0..2 {
            assert!((r1.w[i] - r2.w[i]).abs() < 1e-7);
        }
    }

    #[test]
    fn err_n_examples_zero() {
        let oracle = |_w: &[f32], _i: usize| -> SeqResult<(Vec<f32>, f32)> { Ok((vec![0.0], 0.0)) };
        let cfg = default_cfg();
        let r = FullCuttingPlaneSvm::train(0, 1, oracle, &cfg);
        assert!(matches!(r, Err(SeqError::InvalidParameter { .. })));
    }

    #[test]
    fn err_n_features_zero() {
        let oracle = |_w: &[f32], _i: usize| -> SeqResult<(Vec<f32>, f32)> { Ok((vec![], 0.0)) };
        let cfg = default_cfg();
        let r = FullCuttingPlaneSvm::train(1, 0, oracle, &cfg);
        assert!(matches!(r, Err(SeqError::InvalidParameter { .. })));
    }

    #[test]
    fn err_c_reg_non_positive() {
        let oracle = |_w: &[f32], _i: usize| -> SeqResult<(Vec<f32>, f32)> { Ok((vec![0.0], 0.0)) };
        let mut cfg = default_cfg();
        cfg.c_reg = 0.0;
        let r = FullCuttingPlaneSvm::train(1, 1, oracle, &cfg);
        assert!(matches!(r, Err(SeqError::InvalidParameter { .. })));
    }

    #[test]
    fn err_epsilon_non_positive() {
        let oracle = |_w: &[f32], _i: usize| -> SeqResult<(Vec<f32>, f32)> { Ok((vec![0.0], 0.0)) };
        let mut cfg = default_cfg();
        cfg.epsilon = 0.0;
        let r = FullCuttingPlaneSvm::train(1, 1, oracle, &cfg);
        assert!(matches!(r, Err(SeqError::InvalidParameter { .. })));
    }

    #[test]
    fn err_max_iter_zero() {
        let oracle = |_w: &[f32], _i: usize| -> SeqResult<(Vec<f32>, f32)> { Ok((vec![0.0], 0.0)) };
        let mut cfg = default_cfg();
        cfg.max_iter = 0;
        let r = FullCuttingPlaneSvm::train(1, 1, oracle, &cfg);
        assert!(matches!(r, Err(SeqError::InvalidParameter { .. })));
    }

    #[test]
    fn single_example_single_constraint() {
        let oracle =
            |_w: &[f32], _i: usize| -> SeqResult<(Vec<f32>, f32)> { Ok((vec![1.0_f32], 1.0)) };
        let cfg = FullCuttingPlaneConfig {
            c_reg: 10.0,
            epsilon: 1e-3,
            max_iter: 10,
        };
        let res = FullCuttingPlaneSvm::train(1, 1, oracle, &cfg).expect("ok");
        assert!((res.w[0] - 1.0).abs() < 0.05);
    }

    #[test]
    fn multiple_examples_accumulate_per_example_constraints() {
        // Each example yields a distinct violating Δψ.  After one iteration
        // each per-example working set has 1 constraint, total = 3.
        let oracle = |_w: &[f32], i: usize| -> SeqResult<(Vec<f32>, f32)> {
            let mut dpsi = vec![0.0_f32; 3];
            if i < 3 {
                dpsi[i] = 1.0;
            }
            Ok((dpsi, 1.0))
        };
        let cfg = FullCuttingPlaneConfig {
            c_reg: 10.0,
            epsilon: 1e-3,
            max_iter: 5,
        };
        let res = FullCuttingPlaneSvm::train(3, 3, oracle, &cfg).expect("ok");
        assert!(res.n_constraints >= 3);
    }

    #[test]
    fn xi_is_non_negative() {
        let oracle =
            |_w: &[f32], _i: usize| -> SeqResult<(Vec<f32>, f32)> { Ok((vec![1.0_f32, 0.0], 0.5)) };
        let cfg = default_cfg();
        let res = FullCuttingPlaneSvm::train(2, 2, oracle, &cfg).expect("ok");
        for v in &res.xi {
            assert!(*v >= 0.0);
        }
    }

    #[test]
    fn xi_length_equals_n_examples() {
        let oracle =
            |_w: &[f32], _i: usize| -> SeqResult<(Vec<f32>, f32)> { Ok((vec![0.0_f32, 0.0], 0.0)) };
        let cfg = default_cfg();
        let res = FullCuttingPlaneSvm::train(5, 2, oracle, &cfg).expect("ok");
        assert_eq!(res.xi.len(), 5);
        assert_eq!(res.w.len(), 2);
    }

    #[test]
    fn convergence_within_max_iter() {
        let oracle =
            |_w: &[f32], _i: usize| -> SeqResult<(Vec<f32>, f32)> { Ok((vec![1.0_f32, 0.0], 1.0)) };
        let cfg = FullCuttingPlaneConfig {
            c_reg: 10.0,
            epsilon: 1e-3,
            max_iter: 100,
        };
        let res = FullCuttingPlaneSvm::train(2, 2, oracle, &cfg).expect("ok");
        assert!(res.iterations <= cfg.max_iter);
    }

    #[test]
    fn qp_master_empty_constraints() {
        let constraints: Vec<Vec<(Vec<f32>, f32)>> = vec![Vec::new(); 3];
        let (w, xi) = FullCuttingPlaneSvm::qp_master(&constraints, 4, 1.0).expect("ok");
        assert_eq!(w.len(), 4);
        assert_eq!(xi.len(), 3);
        for v in &w {
            assert!(v.abs() < 1e-9);
        }
        for v in &xi {
            assert!(v.abs() < 1e-9);
        }
    }

    #[test]
    fn qp_master_shape_mismatch_errors() {
        // Δψ has 2 entries but n_features = 3.
        let constraints: Vec<Vec<(Vec<f32>, f32)>> = vec![vec![(vec![1.0, 0.0], 1.0)]];
        let r = FullCuttingPlaneSvm::qp_master(&constraints, 3, 1.0);
        assert!(matches!(r, Err(SeqError::ShapeMismatch { .. })));
    }

    #[test]
    fn qp_master_c_reg_non_positive_errors() {
        let constraints: Vec<Vec<(Vec<f32>, f32)>> = vec![Vec::new()];
        let r = FullCuttingPlaneSvm::qp_master(&constraints, 2, -1.0);
        assert!(matches!(r, Err(SeqError::InvalidParameter { .. })));
    }

    #[test]
    fn oracle_shape_mismatch_propagates() {
        let oracle = |_w: &[f32], _i: usize| -> SeqResult<(Vec<f32>, f32)> {
            Ok((vec![1.0_f32], 1.0)) // wrong length: 1 vs n_features = 2
        };
        let cfg = default_cfg();
        let r = FullCuttingPlaneSvm::train(1, 2, oracle, &cfg);
        assert!(matches!(r, Err(SeqError::ShapeMismatch { .. })));
    }

    #[test]
    fn oracle_error_propagates() {
        let oracle =
            |_w: &[f32], _i: usize| -> SeqResult<(Vec<f32>, f32)> { Err(SeqError::EmptyInput) };
        let cfg = default_cfg();
        let r = FullCuttingPlaneSvm::train(1, 2, oracle, &cfg);
        assert!(matches!(r, Err(SeqError::EmptyInput)));
    }
}
