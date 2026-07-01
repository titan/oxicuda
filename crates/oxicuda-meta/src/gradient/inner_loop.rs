use crate::error::{MetaError, MetaResult};
use crate::gradient::finite_diff::fd_gradient;

pub fn inner_sgd_step(params: &[f32], grads: &[f32], lr: f32) -> MetaResult<Vec<f32>> {
    if params.len() != grads.len() {
        return Err(MetaError::DimensionMismatch {
            expected: params.len(),
            got: grads.len(),
        });
    }
    if lr <= 0.0 || !lr.is_finite() {
        return Err(MetaError::InvalidLr { lr });
    }
    Ok(params
        .iter()
        .zip(grads.iter())
        .map(|(&p, &g)| p - lr * g)
        .collect())
}

pub fn multi_step_inner<F>(
    params: Vec<f32>,
    f: &F,
    lr: f32,
    n_steps: usize,
    eps: f32,
) -> MetaResult<Vec<f32>>
where
    F: Fn(&[f32]) -> f32,
{
    if lr <= 0.0 || !lr.is_finite() {
        return Err(MetaError::InvalidLr { lr });
    }
    let mut current = params;
    for _ in 0..n_steps {
        let grad = fd_gradient(&current, f, eps);
        current = inner_sgd_step(&current, &grad, lr)?;
    }
    Ok(current)
}

pub fn cross_entropy_loss(logits: &[f32], labels: &[u32], n_classes: usize) -> MetaResult<f32> {
    if n_classes == 0 {
        return Err(MetaError::InvalidEpisodeConfig {
            msg: "n_classes must be > 0".into(),
        });
    }
    if logits.len() != labels.len() * n_classes {
        return Err(MetaError::DimensionMismatch {
            expected: labels.len() * n_classes,
            got: logits.len(),
        });
    }

    let n = labels.len();
    if n == 0 {
        return Err(MetaError::EmptySupport);
    }

    let mut total_loss = 0.0_f32;

    for (i, &lbl) in labels.iter().enumerate() {
        let row = &logits[i * n_classes..(i + 1) * n_classes];
        let max_logit = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&z| (z - max_logit).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        if sum_exp == 0.0 {
            return Err(MetaError::NanEncountered {
                context: "cross_entropy sum_exp is zero".into(),
            });
        }
        let log_prob = (exps[lbl as usize] / sum_exp).ln();
        if !log_prob.is_finite() {
            return Err(MetaError::NanEncountered {
                context: "cross_entropy log_prob is non-finite".into(),
            });
        }
        total_loss -= log_prob;
    }

    Ok(total_loss / n as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // Quadratic toy: L(θ) = ½‖θ − t‖², ∇L(θ) = (θ − t).
    fn quadratic_loss(target: &[f32]) -> impl Fn(&[f32]) -> f32 + '_ {
        move |p: &[f32]| {
            0.5 * p
                .iter()
                .zip(target.iter())
                .map(|(&pi, &ti)| (pi - ti) * (pi - ti))
                .sum::<f32>()
        }
    }

    // Pins θ₁ = θ₀ − α·g EXACTLY. With g = (θ₀ − t) this is one SGD step on
    // the quadratic L = ½‖θ − t‖² whose gradient is (θ − t).
    #[test]
    fn inner_loop_single_step_exact_identity() {
        let theta0 = vec![3.0_f32, -1.0, 2.0];
        let target = [1.0_f32, 0.0, 5.0];
        let grad: Vec<f32> = theta0
            .iter()
            .zip(target.iter())
            .map(|(&p, &ti)| p - ti)
            .collect();
        let lr = 0.1_f32;
        let theta1 = inner_sgd_step(&theta0, &grad, lr).expect("sgd step");
        for i in 0..theta0.len() {
            let expected = theta0[i] - lr * grad[i];
            assert_eq!(theta1[i], expected, "θ₁ must equal θ₀ − α·g exactly at {i}");
        }
    }

    // Pins the "no movement" identity θ₁ = θ₀ − α·0 = θ₀. The impl rejects
    // lr = 0 (see inner_loop_rejects_nonpositive_lr), so the genuine
    // "unchanged" property is the zero-gradient case, valid for any α.
    #[test]
    fn inner_loop_zero_grad_leaves_params_unchanged() {
        let theta0 = vec![1.5_f32, -2.0, 0.0, 7.25];
        let zero = vec![0.0_f32; theta0.len()];
        let updated = inner_sgd_step(&theta0, &zero, 0.5).expect("sgd step");
        assert_eq!(updated, theta0, "zero gradient must leave params unchanged");
    }

    // The impl pins lr ∈ (0, ∞): lr ≤ 0 or non-finite ⇒ InvalidLr.
    #[test]
    fn inner_loop_rejects_nonpositive_lr() {
        let theta0 = vec![1.0_f32, 2.0];
        let grad = vec![0.5_f32, 0.5];
        assert!(matches!(
            inner_sgd_step(&theta0, &grad, 0.0),
            Err(MetaError::InvalidLr { .. })
        ));
        assert!(matches!(
            inner_sgd_step(&theta0, &grad, -0.1),
            Err(MetaError::InvalidLr { .. })
        ));
        assert!(matches!(
            inner_sgd_step(&theta0, &grad, f32::NAN),
            Err(MetaError::InvalidLr { .. })
        ));
    }

    #[test]
    fn inner_loop_rejects_dim_mismatch() {
        let r = inner_sgd_step(&[1.0, 2.0], &[1.0], 0.1);
        assert!(matches!(
            r,
            Err(MetaError::DimensionMismatch {
                expected: 2,
                got: 1
            })
        ));
    }

    // K steps of GD on the convex quadratic strictly DECREASE the loss for a
    // small lr (monotone). Each k restarts from θ₀ and does k steps, so the
    // sequence of losses is exactly the per-step trajectory.
    #[test]
    fn inner_loop_multi_step_monotone_decrease() {
        let target = vec![2.0_f32, -1.0, 0.5];
        let f = quadratic_loss(&target);
        let theta0 = vec![5.0_f32, 5.0, 5.0];
        let lr = 0.05_f32;
        let mut prev = f(&theta0);
        for k in 1..=6 {
            let adapted = multi_step_inner(theta0.clone(), &f, lr, k, 1e-4).expect("multi step");
            let loss = f(&adapted);
            assert!(
                loss < prev,
                "loss must strictly decrease at k={k}: {loss} >= {prev}"
            );
            prev = loss;
        }
    }

    // Central-difference gradient of a quadratic is analytically exact, so one
    // multi_step pass reproduces θ₁ = θ₀ − α(θ₀ − t). Use a small, well-
    // conditioned problem so the library's f32 fd (eps = 1e-4) is accurate.
    #[test]
    fn inner_loop_multi_step_one_step_matches_analytic() {
        let target = vec![0.1_f32, 0.2, -0.1];
        let f = quadratic_loss(&target);
        let theta0 = vec![0.3_f32, -0.5, 0.2];
        let lr = 0.05_f32;
        let adapted = multi_step_inner(theta0.clone(), &f, lr, 1, 1e-4).expect("multi step");
        let true_grad: Vec<f32> = theta0
            .iter()
            .zip(target.iter())
            .map(|(&p, &ti)| p - ti)
            .collect();
        let expected = inner_sgd_step(&theta0, &true_grad, lr).expect("sgd step");
        for i in 0..theta0.len() {
            assert!(
                (adapted[i] - expected[i]).abs() < 1e-5,
                "fd single step {} != analytic {} at {i}",
                adapted[i],
                expected[i]
            );
        }
    }

    // Many steps converge toward the target t (the unique minimizer).
    #[test]
    fn inner_loop_converges_to_minimizer() {
        let target = vec![2.0_f32, -1.0, 0.5];
        let f = quadratic_loss(&target);
        let theta0 = vec![5.0_f32, 5.0, 5.0];
        let adapted = multi_step_inner(theta0, &f, 0.1, 400, 1e-4).expect("multi step");
        for i in 0..target.len() {
            assert!(
                (adapted[i] - target[i]).abs() < 1e-3,
                "did not converge at {i}: {} vs {}",
                adapted[i],
                target[i]
            );
        }
    }

    // Uniform logits ⇒ uniform softmax ⇒ CE loss = ln(n_classes); a confident
    // correct prediction ⇒ CE loss ≈ 0.
    #[test]
    fn cross_entropy_known_values() {
        let n_classes = 4;
        let labels = vec![0_u32, 3];
        let uniform = vec![0.0_f32; labels.len() * n_classes];
        let loss = cross_entropy_loss(&uniform, &labels, n_classes).expect("ce");
        assert!(
            (loss - (n_classes as f32).ln()).abs() < 1e-5,
            "uniform CE must be ln(C): got {loss}"
        );

        let confident = vec![
            100.0_f32, 0.0, 0.0, 0.0, // label 0
            0.0, 0.0, 0.0, 100.0, // label 3
        ];
        let loss2 = cross_entropy_loss(&confident, &labels, n_classes).expect("ce");
        assert!(loss2 < 1e-3, "confident correct CE must be ≈0: got {loss2}");
    }

    #[test]
    fn cross_entropy_rejects_bad_args() {
        assert!(matches!(
            cross_entropy_loss(&[0.0, 0.0], &[0], 0),
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
        assert!(matches!(
            cross_entropy_loss(&[0.0, 0.0, 0.0], &[0], 2),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    // Determinism with a fixed seed and finiteness of outputs.
    #[test]
    fn inner_loop_deterministic_and_finite() {
        let mut rng = LcgRng::new(123);
        let theta0: Vec<f32> = (0..5).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let target: Vec<f32> = (0..5).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let f = quadratic_loss(&target);
        let a = multi_step_inner(theta0.clone(), &f, 0.05, 10, 1e-4).expect("multi step");
        let b = multi_step_inner(theta0, &f, 0.05, 10, 1e-4).expect("multi step");
        assert_eq!(a, b, "multi_step_inner must be deterministic");
        assert!(a.iter().all(|v| v.is_finite()), "outputs must be finite");
    }
}
