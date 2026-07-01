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

#[cfg(test)]
mod tests {
    use super::{cosine_similarity, matching_net_attention, matching_net_predict};
    use crate::error::MetaError;

    // ── cosine_similarity: analytic identities ───────────────────────────────

    #[test]
    fn cosine_similarity_identical_vectors_is_one() {
        // cos(v, v) = ||v||² / (||v||² + ε). With unit vector, denominator rounds
        // to 1.0 in f32 (ε=1e-8 < f32 machine epsilon ≈ 1.19e-7), so result is 1.0.
        let v = vec![1.0_f32, 0.0, 0.0];
        let sim = cosine_similarity(&v, &v).expect("cosine_similarity must succeed");
        assert!(
            (sim - 1.0_f32).abs() < 1e-5,
            "cos(v,v) must be ~1.0, got {sim}"
        );
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors_is_zero() {
        // cos([1,0], [0,1]) = 0 / (1·1 + ε) = 0.
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let sim = cosine_similarity(&a, &b).expect("cosine_similarity must succeed");
        assert!(
            sim.abs() < 1e-6,
            "cos of orthogonal vectors must be 0, got {sim}"
        );
    }

    #[test]
    fn cosine_similarity_dim_mismatch_returns_error() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        let result = cosine_similarity(&a, &b);
        assert!(
            matches!(result, Err(MetaError::DimensionMismatch { .. })),
            "dimension mismatch must return DimensionMismatch, got {result:?}"
        );
    }

    // ── matching_net_attention: distribution properties ──────────────────────

    #[test]
    fn attention_class_probs_sum_to_one() {
        // With all support labels valid (< n_way), the blended class probs must sum to 1.
        // Support: class 0 → [1,0], class 1 → [0,1].
        let support_feats = vec![1.0_f32, 0.0, 0.0, 1.0];
        let support_y = vec![0_u32, 1];
        let query = vec![0.5_f32, 0.5];
        let class_probs = matching_net_attention(&query, &support_feats, &support_y, 2, 1.0)
            .expect("matching_net_attention must succeed");
        let total: f32 = class_probs.iter().sum();
        assert!(
            (total - 1.0_f32).abs() < 1e-5,
            "class probs must sum to 1.0, got {total}"
        );
    }

    #[test]
    fn attention_class_probs_nonneg() {
        // Softmax outputs are always non-negative.
        let support_feats = vec![1.0_f32, 0.0, 0.0, 1.0];
        let support_y = vec![0_u32, 1];
        let query = vec![0.3_f32, 0.7];
        let class_probs = matching_net_attention(&query, &support_feats, &support_y, 2, 1.0)
            .expect("matching_net_attention must succeed");
        for (i, &p) in class_probs.iter().enumerate() {
            assert!(p >= 0.0, "class_probs[{i}] must be non-negative, got {p}");
        }
    }

    #[test]
    fn identical_query_yields_highest_attention_on_its_class() {
        // Support: class 0 → [1,0], class 1 → [0,1].
        // Query = [1,0] is identical to the class-0 support example.
        // cos(query, s0) = 1.0, cos(query, s1) = 0.0; after softmax class_probs[0] > class_probs[1].
        let support_feats = vec![1.0_f32, 0.0, 0.0, 1.0];
        let support_y = vec![0_u32, 1];
        let query = vec![1.0_f32, 0.0];
        let class_probs = matching_net_attention(&query, &support_feats, &support_y, 2, 1.0)
            .expect("matching_net_attention must succeed");
        assert!(
            class_probs[0] > class_probs[1],
            "class 0 must receive higher attention than class 1 when query matches class-0 support \
             (probs: {:?})",
            class_probs
        );
    }

    // ── matching_net_predict: end-to-end classification ──────────────────────

    #[test]
    fn predict_assigns_query_to_nearest_class() {
        // Support: class 0 → [1,0], class 1 → [0,1].
        // Query [1,0] is identical to class-0 support → must predict class 0.
        let support_feats = vec![1.0_f32, 0.0, 0.0, 1.0];
        let support_y = vec![0_u32, 1];
        let query_feats = vec![1.0_f32, 0.0];
        let preds = matching_net_predict(&query_feats, &support_feats, &support_y, 2, 2, 1.0)
            .expect("matching_net_predict must succeed");
        assert_eq!(
            preds,
            vec![0_u32],
            "query matching class-0 support must predict class 0"
        );
    }

    #[test]
    fn predict_deterministic_for_same_input() {
        // Same support and query must yield the same predictions on every call.
        let support_feats = vec![1.0_f32, 0.0, 0.0, 1.0];
        let support_y = vec![0_u32, 1];
        let query_feats = vec![0.9_f32, 0.1, 0.1, 0.9];
        let p1 = matching_net_predict(&query_feats, &support_feats, &support_y, 2, 2, 1.0)
            .expect("first predict must succeed");
        let p2 = matching_net_predict(&query_feats, &support_feats, &support_y, 2, 2, 1.0)
            .expect("second predict must succeed");
        assert_eq!(p1, p2, "predictions must be deterministic");
    }
}
