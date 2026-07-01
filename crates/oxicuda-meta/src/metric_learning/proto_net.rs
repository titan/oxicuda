use crate::error::{MetaError, MetaResult};

pub fn compute_prototypes(
    support_feats: &[f32],
    support_y: &[u32],
    n_way: usize,
    k_shot: usize,
    feat_dim: usize,
) -> MetaResult<Vec<f32>> {
    if support_feats.is_empty() {
        return Err(MetaError::EmptySupport);
    }
    if support_feats.len() != n_way * k_shot * feat_dim {
        return Err(MetaError::DimensionMismatch {
            expected: n_way * k_shot * feat_dim,
            got: support_feats.len(),
        });
    }
    if support_y.len() != n_way * k_shot {
        return Err(MetaError::DimensionMismatch {
            expected: n_way * k_shot,
            got: support_y.len(),
        });
    }

    let mut prototypes = vec![0.0_f32; n_way * feat_dim];
    let mut counts = vec![0_usize; n_way];

    for (s, feat) in support_feats.chunks(feat_dim).enumerate() {
        let cls = support_y[s] as usize;
        if cls >= n_way {
            return Err(MetaError::Internal {
                msg: format!("support label {cls} >= n_way {n_way}"),
            });
        }
        let proto_row = &mut prototypes[cls * feat_dim..(cls + 1) * feat_dim];
        for (p, &f) in proto_row.iter_mut().zip(feat.iter()) {
            *p += f;
        }
        counts[cls] += 1;
    }

    for (c, &cnt) in counts.iter().enumerate() {
        if cnt > 0 {
            let proto_row = &mut prototypes[c * feat_dim..(c + 1) * feat_dim];
            for p in proto_row.iter_mut() {
                *p /= cnt as f32;
            }
        }
    }

    Ok(prototypes)
}

pub fn proto_predict(
    query_feats: &[f32],
    prototypes: &[f32],
    n_way: usize,
    feat_dim: usize,
) -> MetaResult<Vec<u32>> {
    if !query_feats.len().is_multiple_of(feat_dim) {
        return Err(MetaError::DimensionMismatch {
            expected: query_feats.len() / feat_dim * feat_dim,
            got: query_feats.len(),
        });
    }
    if prototypes.len() != n_way * feat_dim {
        return Err(MetaError::DimensionMismatch {
            expected: n_way * feat_dim,
            got: prototypes.len(),
        });
    }

    let n_query = query_feats.len() / feat_dim;
    let mut preds = Vec::with_capacity(n_query);

    for q_feat in query_feats.chunks(feat_dim) {
        let mut best_cls = 0;
        let mut best_dist = f32::INFINITY;
        for cls in 0..n_way {
            let proto = &prototypes[cls * feat_dim..(cls + 1) * feat_dim];
            let dist: f32 = q_feat
                .iter()
                .zip(proto.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();
            if dist < best_dist {
                best_dist = dist;
                best_cls = cls;
            }
        }
        preds.push(best_cls as u32);
    }

    Ok(preds)
}

pub fn proto_loss(
    query_feats: &[f32],
    query_y: &[u32],
    prototypes: &[f32],
    n_way: usize,
    feat_dim: usize,
) -> MetaResult<f32> {
    if query_feats.is_empty() {
        return Err(MetaError::EmptySupport);
    }
    let n_query = query_y.len();
    if query_feats.len() != n_query * feat_dim {
        return Err(MetaError::DimensionMismatch {
            expected: n_query * feat_dim,
            got: query_feats.len(),
        });
    }
    if prototypes.len() != n_way * feat_dim {
        return Err(MetaError::DimensionMismatch {
            expected: n_way * feat_dim,
            got: prototypes.len(),
        });
    }

    let mut total_loss = 0.0_f32;

    for (q, q_feat) in query_feats.chunks(feat_dim).enumerate() {
        // logits[k] = -d²(q, c_k)
        let neg_dists: Vec<f32> = (0..n_way)
            .map(|cls| {
                let proto = &prototypes[cls * feat_dim..(cls + 1) * feat_dim];
                let d2: f32 = q_feat
                    .iter()
                    .zip(proto.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                -d2
            })
            .collect();

        let max_logit = neg_dists.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = neg_dists.iter().map(|&z| (z - max_logit).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        if sum_exp == 0.0 {
            return Err(MetaError::NanEncountered {
                context: "proto_loss sum_exp is zero".into(),
            });
        }
        let lbl = query_y[q] as usize;
        let log_prob = (exps[lbl] / sum_exp).ln();
        if !log_prob.is_finite() {
            return Err(MetaError::NanEncountered {
                context: "proto_loss log_prob is non-finite".into(),
            });
        }
        total_loss -= log_prob;
    }

    Ok(total_loss / n_query as f32)
}

#[cfg(test)]
mod tests {
    use super::{compute_prototypes, proto_loss, proto_predict};
    use crate::error::MetaError;

    // ── compute_prototypes: exact-mean property ──────────────────────────────

    #[test]
    fn prototypes_exact_mean_single_class() {
        // Two shots for class 0: [0,0] and [2,2]; mean must be exactly [1,1].
        let support_feats = vec![0.0_f32, 0.0, 2.0, 2.0];
        let support_y = vec![0_u32, 0];
        let protos = compute_prototypes(&support_feats, &support_y, 1, 2, 2)
            .expect("compute_prototypes must succeed for valid inputs");
        assert_eq!(
            protos.len(),
            2,
            "prototype vector must have feat_dim elements"
        );
        assert_eq!(
            protos[0], 1.0_f32,
            "prototype[0] must be exact arithmetic mean 1.0"
        );
        assert_eq!(
            protos[1], 1.0_f32,
            "prototype[1] must be exact arithmetic mean 1.0"
        );
    }

    #[test]
    fn prototypes_two_classes_correct_means() {
        // Class 0 shots: [0,0],[2,2] → mean [1,1]; class 1 shots: [8,8],[10,10] → mean [9,9].
        let support_feats = vec![0.0_f32, 0.0, 2.0, 2.0, 8.0, 8.0, 10.0, 10.0];
        let support_y = vec![0_u32, 0, 1, 1];
        let protos = compute_prototypes(&support_feats, &support_y, 2, 2, 2)
            .expect("compute_prototypes must succeed");
        assert_eq!(protos.len(), 4);
        assert_eq!(protos[0], 1.0_f32, "class-0 proto dim-0");
        assert_eq!(protos[1], 1.0_f32, "class-0 proto dim-1");
        assert_eq!(protos[2], 9.0_f32, "class-1 proto dim-0");
        assert_eq!(protos[3], 9.0_f32, "class-1 proto dim-1");
    }

    // ── proto_predict: nearest-prototype assignment ──────────────────────────

    #[test]
    fn proto_predict_assigns_nearest_class() {
        // proto_0=[1,1], proto_1=[9,9]; query at [1.5,1.5] is much closer to class 0.
        let protos = vec![1.0_f32, 1.0, 9.0, 9.0];
        // dist(query, proto_0) = 0.5² + 0.5² = 0.5
        // dist(query, proto_1) = 7.5² + 7.5² = 112.5
        let query = vec![1.5_f32, 1.5];
        let preds = proto_predict(&query, &protos, 2, 2).expect("proto_predict must succeed");
        assert_eq!(
            preds,
            vec![0_u32],
            "query near class 0 must predict class 0"
        );
    }

    #[test]
    fn proto_predict_deterministic() {
        // Same inputs must always yield the same predictions.
        let protos = vec![0.0_f32, 0.0, 5.0, 5.0];
        let query = vec![0.1_f32, 0.1, 4.9, 4.9];
        let p1 = proto_predict(&query, &protos, 2, 2).expect("first predict must succeed");
        let p2 = proto_predict(&query, &protos, 2, 2).expect("second predict must succeed");
        assert_eq!(p1, p2, "predictions must be deterministic");
    }

    // ── proto_loss: analytic properties ─────────────────────────────────────

    #[test]
    fn proto_loss_nonneg_and_finite() {
        // Negative log-softmax is always ≥ 0 and finite.
        let protos = vec![1.0_f32, 1.0, 9.0, 9.0];
        let query = vec![1.5_f32, 1.5];
        let labels = vec![0_u32];
        let loss = proto_loss(&query, &labels, &protos, 2, 2).expect("proto_loss must succeed");
        assert!(loss >= 0.0, "loss must be non-negative, got {loss}");
        assert!(loss.is_finite(), "loss must be finite, got {loss}");
    }

    #[test]
    fn proto_loss_smaller_when_query_near_true_prototype() {
        // proto_0=[0,0], proto_1=[4,4]; true label=0.
        // Near query [0.5,0.5]: d²(q,p0)=0.5,  d²(q,p1)=24.5  → exp gap exp(-24) ≈ 3.8e-11
        // Far  query [3.5,3.5]: d²(q,p0)=24.5, d²(q,p1)=0.5   → exp gap reversed
        // Distances kept well below the f32 exp underflow threshold (~87.3 for normals).
        let protos = vec![0.0_f32, 0.0, 4.0, 4.0];
        let labels = vec![0_u32];
        let loss_near = proto_loss(&[0.5_f32, 0.5], &labels, &protos, 2, 2)
            .expect("proto_loss near must succeed");
        let loss_far = proto_loss(&[3.5_f32, 3.5], &labels, &protos, 2, 2)
            .expect("proto_loss far must succeed");
        assert!(
            loss_near < loss_far,
            "query near true prototype (loss={loss_near}) must give lower loss than far (loss={loss_far})"
        );
    }

    // ── error variants ───────────────────────────────────────────────────────

    #[test]
    fn compute_prototypes_empty_returns_error() {
        let result = compute_prototypes(&[], &[], 2, 1, 2);
        assert!(
            matches!(result, Err(MetaError::EmptySupport)),
            "empty support must return EmptySupport, got {result:?}"
        );
    }

    #[test]
    fn compute_prototypes_dim_mismatch_returns_error() {
        // Supply 3 floats but n_way*k_shot*feat_dim = 2*2*2 = 8.
        let result = compute_prototypes(&[0.0_f32; 3], &[0_u32; 4], 2, 2, 2);
        assert!(
            matches!(result, Err(MetaError::DimensionMismatch { .. })),
            "size mismatch must return DimensionMismatch, got {result:?}"
        );
    }
}
