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
