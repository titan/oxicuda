use crate::error::{MetaError, MetaResult};
use crate::gradient::finite_diff::fd_gradient;
use crate::gradient::inner_loop::inner_sgd_step;

pub struct AnilConfig {
    pub inner_lr: f32,
    pub n_inner_steps: usize,
    pub feat_dim: usize,
    pub n_classes: usize,
}

fn head_loss(
    head_params: &[f32],
    support_feats: &[f32],
    support_y: &[u32],
    feat_dim: usize,
    n_classes: usize,
) -> f32 {
    let n_support = support_y.len();
    if n_support == 0 {
        return 0.0;
    }
    let w = &head_params[..n_classes * feat_dim];
    let b = &head_params[n_classes * feat_dim..];

    let mut loss = 0.0_f32;
    for (s, feat) in support_feats.chunks(feat_dim).enumerate() {
        let mut logits = vec![0.0_f32; n_classes];
        for c in 0..n_classes {
            let row = &w[c * feat_dim..(c + 1) * feat_dim];
            logits[c] = row
                .iter()
                .zip(feat.iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum::<f32>()
                + b[c];
        }
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&z| (z - max_l).exp()).collect();
        let sum_e: f32 = exps.iter().sum();
        if sum_e > 0.0 {
            let lbl = support_y[s] as usize;
            let lp = (exps[lbl] / sum_e).ln();
            if lp.is_finite() {
                loss -= lp;
            }
        }
    }
    loss / n_support as f32
}

pub fn anil_adapt_head(
    head_params: &[f32],
    support_feats: &[f32],
    support_y: &[u32],
    cfg: &AnilConfig,
) -> MetaResult<Vec<f32>> {
    if cfg.inner_lr <= 0.0 {
        return Err(MetaError::InvalidLr { lr: cfg.inner_lr });
    }
    let expected = cfg.n_classes * cfg.feat_dim + cfg.n_classes;
    if head_params.len() != expected {
        return Err(MetaError::DimensionMismatch {
            expected,
            got: head_params.len(),
        });
    }

    let mut adapted = head_params.to_vec();
    for _ in 0..cfg.n_inner_steps {
        let f = |p: &[f32]| head_loss(p, support_feats, support_y, cfg.feat_dim, cfg.n_classes);
        let grad = fd_gradient(&adapted, &f, 1e-4);
        adapted = inner_sgd_step(&adapted, &grad, cfg.inner_lr)?;
    }
    Ok(adapted)
}

pub fn anil_meta_update(
    head_params: &[f32],
    task_feats: &[(Vec<f32>, Vec<u32>)],
    outer_lr: f32,
    cfg: &AnilConfig,
) -> MetaResult<Vec<f32>> {
    if task_feats.is_empty() {
        return Err(MetaError::EmptySupport);
    }
    if outer_lr <= 0.0 || !outer_lr.is_finite() {
        return Err(MetaError::InvalidLr { lr: outer_lr });
    }

    let n_params = head_params.len();
    let n_tasks = task_feats.len() as f32;
    let mut meta_grad = vec![0.0_f32; n_params];

    for (support_feats, support_y) in task_feats {
        let adapted = anil_adapt_head(head_params, support_feats, support_y, cfg)?;
        let f = |p: &[f32]| head_loss(p, support_feats, support_y, cfg.feat_dim, cfg.n_classes);
        let task_grad = fd_gradient(&adapted, &f, 1e-4);
        for (mg, &tg) in meta_grad.iter_mut().zip(task_grad.iter()) {
            *mg += tg / n_tasks;
        }
    }

    Ok(head_params
        .iter()
        .zip(meta_grad.iter())
        .map(|(&p, &g)| p - outer_lr * g)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    const N_CLASSES: usize = 2;
    const FEAT_DIM: usize = 3;
    const HEAD_LEN: usize = N_CLASSES * FEAT_DIM + N_CLASSES;

    fn base_head() -> Vec<f32> {
        vec![0.2, -0.1, 0.05, 0.1, -0.2, 0.15, 0.0, 0.0]
    }

    // Two samples, FEAT_DIM features each: these are the FROZEN backbone outputs.
    fn support() -> (Vec<f32>, Vec<u32>) {
        (vec![1.0, 0.5, -0.5, -0.3, 0.8, 0.2], vec![0_u32, 1])
    }

    fn cfg(n_inner_steps: usize, inner_lr: f32) -> AnilConfig {
        AnilConfig {
            inner_lr,
            n_inner_steps,
            feat_dim: FEAT_DIM,
            n_classes: N_CLASSES,
        }
    }

    // A linear backbone: raw inputs → features. Used only to exhibit a concrete
    // frozen body in the ANIL frozen-backbone test.
    fn forward_body(raw: &[f32], body: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let mut feats = Vec::new();
        for sample in raw.chunks(in_dim) {
            for o in 0..out_dim {
                let row = &body[o * in_dim..(o + 1) * in_dim];
                feats.push(
                    row.iter()
                        .zip(sample.iter())
                        .map(|(&w, &x)| w * x)
                        .sum::<f32>(),
                );
            }
        }
        feats
    }

    // ANIL inner loop is one head-only SGD step on the (frozen) features:
    // adapted = head − α·∇_head L(head). Pinned EXACTLY for n_inner_steps = 1.
    #[test]
    fn anil_one_step_head_adaptation_exact() {
        let head = base_head();
        let (feats, sy) = support();
        let c = cfg(1, 0.05);
        let f = |p: &[f32]| head_loss(p, &feats, &sy, FEAT_DIM, N_CLASSES);
        let grad = fd_gradient(&head, &f, 1e-4);
        let expected = inner_sgd_step(&head, &grad, c.inner_lr).expect("sgd step");
        let adapted = anil_adapt_head(&head, &feats, &sy, &c).expect("adapt");
        for i in 0..HEAD_LEN {
            assert_eq!(
                adapted[i], expected[i],
                "head must adapt by one SGD step at {i}"
            );
        }
    }

    // DEFINING ANIL PROPERTY ("Almost No Inner Loop"): only the HEAD adapts; the
    // backbone/body is byte-identical before and after, and adaptation has
    // head-only dimensionality.
    #[test]
    fn anil_freezes_body_adapts_head() {
        // Concrete frozen backbone (3x2 linear map) producing the features.
        let body = vec![0.3_f32, -0.2, 0.1, 0.4, -0.5, 0.6];
        let raw = vec![1.0_f32, -1.0, 0.5, 0.25]; // 2 samples x in_dim=2
        let feats = forward_body(&raw, &body, 2, FEAT_DIM);
        let (_, sy) = support();
        let head = base_head();

        let body_before = body.clone();
        let full_before: Vec<f32> = body.iter().chain(head.iter()).copied().collect();

        let adapted = anil_adapt_head(&head, &feats, &sy, &cfg(3, 0.1)).expect("adapt");

        // adaptation is head-sized only — it never expands to include the body.
        assert_eq!(
            adapted.len(),
            HEAD_LEN,
            "ANIL adapts head-only dimensionality"
        );
        let full_after: Vec<f32> = body.iter().chain(adapted.iter()).copied().collect();

        let bl = body.len();
        assert_eq!(body, body_before, "backbone params must be frozen");
        assert_eq!(
            &full_before[..bl],
            &full_after[..bl],
            "body slice of the full model must be byte-identical"
        );
        assert_ne!(
            &full_before[bl..],
            &full_after[bl..],
            "head slice of the full model must change"
        );
    }

    // Head-only inner loop strictly decreases the head loss (monotone) for a
    // small lr.
    #[test]
    fn anil_inner_loop_monotone_decrease() {
        let head = base_head();
        let (feats, sy) = support();
        let mut prev = head_loss(&head, &feats, &sy, FEAT_DIM, N_CLASSES);
        for k in 1..=5 {
            let adapted = anil_adapt_head(&head, &feats, &sy, &cfg(k, 0.1)).expect("adapt");
            let loss = head_loss(&adapted, &feats, &sy, FEAT_DIM, N_CLASSES);
            assert!(
                loss < prev,
                "head loss must strictly decrease at k={k}: {loss} >= {prev}"
            );
            prev = loss;
        }
    }

    // ANIL meta-gradient (head-only, first-order) = ∇_head L at the ADAPTED head.
    #[test]
    fn anil_meta_grad_is_gradient_at_adapted_head() {
        let head = base_head();
        let (feats, sy) = support();
        let c = cfg(3, 0.1);
        let outer_lr = 0.1_f32;
        let adapted = anil_adapt_head(&head, &feats, &sy, &c).expect("adapt");
        let f = |p: &[f32]| head_loss(p, &feats, &sy, FEAT_DIM, N_CLASSES);
        let grad_at_adapted = fd_gradient(&adapted, &f, 1e-4);
        let updated = anil_meta_update(&head, &[(feats, sy)], outer_lr, &c).expect("meta");
        for i in 0..HEAD_LEN {
            let meta_grad = (head[i] - updated[i]) / outer_lr;
            assert!(
                (meta_grad - grad_at_adapted[i]).abs() < 1e-4,
                "ANIL meta-grad must be ∇_head L at adapted head at {i}: {} vs {}",
                meta_grad,
                grad_at_adapted[i]
            );
        }
    }

    #[test]
    fn anil_deterministic_and_finite() {
        let mut rng = LcgRng::new(55);
        let head: Vec<f32> = (0..HEAD_LEN).map(|_| rng.next_f32() - 0.5).collect();
        let (feats, sy) = support();
        let a = anil_adapt_head(&head, &feats, &sy, &cfg(3, 0.05)).expect("adapt");
        let b = anil_adapt_head(&head, &feats, &sy, &cfg(3, 0.05)).expect("adapt");
        assert_eq!(a, b, "anil_adapt_head must be deterministic");
        assert!(a.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn anil_rejects_bad_args() {
        let head = base_head();
        let (feats, sy) = support();
        // Wrong head length ⇒ DimensionMismatch.
        let short = vec![0.0_f32; HEAD_LEN - 1];
        assert!(matches!(
            anil_adapt_head(&short, &feats, &sy, &cfg(1, 0.05)),
            Err(MetaError::DimensionMismatch { .. })
        ));
        // Non-positive inner_lr ⇒ InvalidLr.
        assert!(matches!(
            anil_adapt_head(&head, &feats, &sy, &cfg(1, 0.0)),
            Err(MetaError::InvalidLr { .. })
        ));
        // Empty task list ⇒ EmptySupport.
        assert!(matches!(
            anil_meta_update(&head, &[], 0.1, &cfg(1, 0.05)),
            Err(MetaError::EmptySupport)
        ));
    }
}
