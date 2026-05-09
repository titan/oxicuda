use crate::error::{MetaError, MetaResult};

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> MetaResult<f32> {
    if a.len() != b.len() {
        return Err(MetaError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum();
    let norm_a: f32 = a.iter().map(|&ai| ai * ai).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&bi| bi * bi).sum::<f32>().sqrt();
    Ok(dot / (norm_a * norm_b + 1e-8))
}

pub fn matching_net_attention(
    query_feat: &[f32],
    support_feats: &[f32],
    support_y: &[u32],
    n_way: usize,
    temp: f32,
) -> MetaResult<Vec<f32>> {
    let feat_dim = query_feat.len();
    if feat_dim == 0 {
        return Err(MetaError::InvalidFeatDim { dim: 0 });
    }
    let n_support = support_y.len();
    if n_support == 0 {
        return Err(MetaError::EmptySupport);
    }
    if support_feats.len() != n_support * feat_dim {
        return Err(MetaError::DimensionMismatch {
            expected: n_support * feat_dim,
            got: support_feats.len(),
        });
    }

    // Compute cosine similarities and apply temperature
    let mut sims = Vec::with_capacity(n_support);
    for s_feat in support_feats.chunks(feat_dim) {
        let sim = cosine_similarity(query_feat, s_feat)?;
        sims.push(sim * temp);
    }

    // Softmax over support examples
    let max_sim = sims.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = sims.iter().map(|&s| (s - max_sim).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    let attn: Vec<f32> = exps.iter().map(|&e| e / sum_exp).collect();

    // Blend into class distribution
    let mut class_probs = vec![0.0_f32; n_way];
    for (s, (&a, &lbl)) in attn.iter().zip(support_y.iter()).enumerate() {
        let _ = s;
        let cls = lbl as usize;
        if cls < n_way {
            class_probs[cls] += a;
        }
    }

    Ok(class_probs)
}

pub fn matching_net_predict(
    query_feats: &[f32],
    support_feats: &[f32],
    support_y: &[u32],
    n_way: usize,
    feat_dim: usize,
    temp: f32,
) -> MetaResult<Vec<u32>> {
    if !query_feats.len().is_multiple_of(feat_dim) {
        return Err(MetaError::DimensionMismatch {
            expected: query_feats.len() / feat_dim * feat_dim,
            got: query_feats.len(),
        });
    }
    let n_query = query_feats.len() / feat_dim;
    let mut preds = Vec::with_capacity(n_query);

    for q_feat in query_feats.chunks(feat_dim) {
        let class_probs = matching_net_attention(q_feat, support_feats, support_y, n_way, temp)?;
        let best_cls = class_probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        preds.push(best_cls as u32);
    }

    Ok(preds)
}
