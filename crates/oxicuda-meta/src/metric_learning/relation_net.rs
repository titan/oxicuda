use crate::episode::types::FewShotEpisode;
use crate::error::{MetaError, MetaResult};
use crate::handle::LcgRng;

pub struct RelationNet {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    feat_dim: usize,
    hidden_dim: usize,
}

impl RelationNet {
    pub fn new(feat_dim: usize, hidden_dim: usize, rng: &mut LcgRng) -> Self {
        let in_dim = 2 * feat_dim;
        let limit1 = (6.0_f32 / (in_dim + hidden_dim) as f32).sqrt();
        let mut w1 = vec![0.0_f32; hidden_dim * in_dim];
        for v in w1.iter_mut() {
            *v = (rng.next_f32() * 2.0 - 1.0) * limit1;
        }
        let b1 = vec![0.0_f32; hidden_dim];

        let limit2 = (6.0_f32 / (hidden_dim + 1) as f32).sqrt();
        let mut w2 = vec![0.0_f32; hidden_dim];
        for v in w2.iter_mut() {
            *v = (rng.next_f32() * 2.0 - 1.0) * limit2;
        }
        let b2 = vec![0.0_f32; 1];

        Self {
            w1,
            b1,
            w2,
            b2,
            feat_dim,
            hidden_dim,
        }
    }

    pub fn relation_score(&self, query_feat: &[f32], support_feat: &[f32]) -> MetaResult<f32> {
        if query_feat.len() != self.feat_dim {
            return Err(MetaError::DimensionMismatch {
                expected: self.feat_dim,
                got: query_feat.len(),
            });
        }
        if support_feat.len() != self.feat_dim {
            return Err(MetaError::DimensionMismatch {
                expected: self.feat_dim,
                got: support_feat.len(),
            });
        }

        let in_dim = 2 * self.feat_dim;

        // h = ReLU(W1 * concat(q, s) + b1)
        let mut h = vec![0.0_f32; self.hidden_dim];
        for (j, (hj, bj)) in h.iter_mut().zip(self.b1.iter()).enumerate() {
            let row = &self.w1[j * in_dim..(j + 1) * in_dim];
            let val: f32 = row[..self.feat_dim]
                .iter()
                .zip(query_feat.iter())
                .map(|(&w, &x)| w * x)
                .sum::<f32>()
                + row[self.feat_dim..]
                    .iter()
                    .zip(support_feat.iter())
                    .map(|(&w, &x)| w * x)
                    .sum::<f32>()
                + bj;
            *hj = val.max(0.0);
        }

        // score = sigmoid(W2 * h + b2)
        let pre_sig: f32 = h
            .iter()
            .zip(self.w2.iter())
            .map(|(&hi, &wi)| hi * wi)
            .sum::<f32>()
            + self.b2[0];
        let score = 1.0 / (1.0 + (-pre_sig).exp());
        Ok(score)
    }

    pub fn predict_episode(&self, episode: &FewShotEpisode) -> MetaResult<Vec<u32>> {
        let cfg = &episode.config;
        let fd = cfg.feat_dim;
        let n_way = cfg.n_way;
        let k_shot = cfg.k_shot;
        let n_query = cfg.n_query;

        // Compute class prototypes from support
        let mut prototypes = vec![0.0_f32; n_way * fd];
        for (s, feat) in episode.support_x.chunks(fd).enumerate() {
            let cls = episode.support_y[s] as usize;
            let proto_row = &mut prototypes[cls * fd..(cls + 1) * fd];
            for (p, &f) in proto_row.iter_mut().zip(feat.iter()) {
                *p += f;
            }
        }
        for c in 0..n_way {
            let proto_row = &mut prototypes[c * fd..(c + 1) * fd];
            for p in proto_row.iter_mut() {
                *p /= k_shot as f32;
            }
        }

        let mut preds = Vec::with_capacity(n_way * n_query);
        for q_feat in episode.query_x.chunks(fd) {
            let mut best_cls = 0;
            let mut best_score = f32::NEG_INFINITY;
            for cls in 0..n_way {
                let proto = &prototypes[cls * fd..(cls + 1) * fd];
                let score = self.relation_score(q_feat, proto)?;
                if score > best_score {
                    best_score = score;
                    best_cls = cls;
                }
            }
            preds.push(best_cls as u32);
        }

        Ok(preds)
    }

    pub fn relation_loss(&self, episode: &FewShotEpisode) -> MetaResult<f32> {
        let cfg = &episode.config;
        let fd = cfg.feat_dim;
        let n_way = cfg.n_way;
        let k_shot = cfg.k_shot;

        // Compute class prototypes
        let mut prototypes = vec![0.0_f32; n_way * fd];
        for (s, feat) in episode.support_x.chunks(fd).enumerate() {
            let cls = episode.support_y[s] as usize;
            let proto_row = &mut prototypes[cls * fd..(cls + 1) * fd];
            for (p, &f) in proto_row.iter_mut().zip(feat.iter()) {
                *p += f;
            }
        }
        for c in 0..n_way {
            let proto_row = &mut prototypes[c * fd..(c + 1) * fd];
            for p in proto_row.iter_mut() {
                *p /= k_shot as f32;
            }
        }

        let mut total_loss = 0.0_f32;
        let mut n_pairs = 0_usize;

        for (q, q_feat) in episode.query_x.chunks(fd).enumerate() {
            let true_cls = episode.query_y[q] as usize;
            for cls in 0..n_way {
                let proto = &prototypes[cls * fd..(cls + 1) * fd];
                let score = self.relation_score(q_feat, proto)?;
                let target = if cls == true_cls { 1.0_f32 } else { 0.0_f32 };
                let diff = score - target;
                total_loss += diff * diff;
                n_pairs += 1;
            }
        }

        if n_pairs == 0 {
            return Err(MetaError::EmptySupport);
        }

        Ok(total_loss / n_pairs as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::RelationNet;
    use crate::episode::types::{EpisodeConfig, FewShotEpisode};
    use crate::error::MetaError;
    use crate::handle::LcgRng;

    fn make_net(feat_dim: usize, hidden_dim: usize, seed: u64) -> RelationNet {
        let mut rng = LcgRng::new(seed);
        RelationNet::new(feat_dim, hidden_dim, &mut rng)
    }

    fn make_episode(
        n_way: usize,
        k_shot: usize,
        n_query: usize,
        feat_dim: usize,
        seed: u64,
    ) -> FewShotEpisode {
        let mut rng = LcgRng::new(seed);
        let n_support = n_way * k_shot;
        let n_q_total = n_way * n_query;
        let support_x: Vec<f32> = (0..n_support * feat_dim).map(|_| rng.next_f32()).collect();
        // Labels assigned round-robin per class (k_shot consecutive shots per class).
        let support_y: Vec<u32> = (0..n_support).map(|i| (i / k_shot) as u32).collect();
        let query_x: Vec<f32> = (0..n_q_total * feat_dim).map(|_| rng.next_f32()).collect();
        let query_y: Vec<u32> = (0..n_q_total).map(|i| (i / n_query) as u32).collect();
        FewShotEpisode {
            config: EpisodeConfig {
                n_way,
                k_shot,
                n_query,
                feat_dim,
            },
            support_x,
            support_y,
            query_x,
            query_y,
        }
    }

    // ── relation_score: sigmoid range guarantee ──────────────────────────────

    #[test]
    fn score_always_in_unit_interval() {
        // sigmoid(x) ∈ (0,1) for all finite x; we assert [0,1] to handle f32 extremes.
        let net = make_net(4, 8, 42);
        let mut rng = LcgRng::new(7);
        for _ in 0..50 {
            let q: Vec<f32> = (0..4).map(|_| rng.next_f32() * 4.0 - 2.0).collect();
            let s: Vec<f32> = (0..4).map(|_| rng.next_f32() * 4.0 - 2.0).collect();
            let score = net
                .relation_score(&q, &s)
                .expect("relation_score must succeed");
            assert!(
                (0.0..=1.0).contains(&score),
                "sigmoid output must be in [0,1], got {score}"
            );
        }
    }

    #[test]
    fn zero_input_gives_exactly_half_score() {
        // Analytic proof: b1=zeros, b2=[0], so for zero inputs every pre-ReLU activation
        // is 0, h=zeros, pre_sigmoid=0, sigmoid(0)=0.5 regardless of W1,W2.
        let net = make_net(6, 16, 99);
        let zeros_q = vec![0.0_f32; 6];
        let zeros_s = vec![0.0_f32; 6];
        let score = net
            .relation_score(&zeros_q, &zeros_s)
            .expect("relation_score on zeros must succeed");
        assert_eq!(
            score, 0.5_f32,
            "zero-input score must be exactly 0.5 (b1=b2=0 ⇒ sigmoid(0))"
        );
    }

    #[test]
    fn score_is_finite_for_random_inputs() {
        // No NaN or infinity must arise from valid sized inputs.
        let net = make_net(8, 12, 31);
        let mut rng = LcgRng::new(13);
        for _ in 0..100 {
            let q: Vec<f32> = (0..8).map(|_| rng.next_f32()).collect();
            let s: Vec<f32> = (0..8).map(|_| rng.next_f32()).collect();
            let score = net
                .relation_score(&q, &s)
                .expect("relation_score must succeed");
            assert!(score.is_finite(), "score must be finite, got {score}");
        }
    }

    #[test]
    fn score_deterministic_with_fixed_seed() {
        // Two identically seeded networks must produce the same score for the same inputs.
        let net_a = make_net(4, 8, 55);
        let net_b = make_net(4, 8, 55);
        let q = vec![0.1_f32, 0.2, 0.3, 0.4];
        let s = vec![0.4_f32, 0.3, 0.2, 0.1];
        let sa = net_a
            .relation_score(&q, &s)
            .expect("net_a score must succeed");
        let sb = net_b
            .relation_score(&q, &s)
            .expect("net_b score must succeed");
        assert_eq!(
            sa, sb,
            "identically seeded nets must yield identical scores"
        );
    }

    // ── relation_loss: MSE bounds ────────────────────────────────────────────

    #[test]
    fn relation_loss_nonneg_and_bounded_by_one() {
        // score ∈ [0,1], target ∈ {0,1}: (score-target)² ∈ [0,1]; mean ∈ [0,1].
        let net = make_net(4, 8, 77);
        let episode = make_episode(3, 2, 2, 4, 19);
        let loss = net
            .relation_loss(&episode)
            .expect("relation_loss must succeed");
        assert!(loss >= 0.0, "MSE loss must be non-negative, got {loss}");
        assert!(
            loss <= 1.0,
            "MSE loss of sigmoid vs {{0,1}} targets must be ≤ 1, got {loss}"
        );
        assert!(loss.is_finite(), "loss must be finite, got {loss}");
    }

    // ── predict_episode: output shape and class validity ────────────────────

    #[test]
    fn predict_episode_correct_size_and_valid_class_indices() {
        // Predictions must have length n_way*n_query and all indices < n_way.
        let n_way = 3;
        let n_query = 2;
        let net = make_net(4, 8, 11);
        let episode = make_episode(n_way, 2, n_query, 4, 23);
        let preds = net
            .predict_episode(&episode)
            .expect("predict_episode must succeed");
        assert_eq!(
            preds.len(),
            n_way * n_query,
            "prediction count must equal n_way*n_query={}, got {}",
            n_way * n_query,
            preds.len()
        );
        for (i, &p) in preds.iter().enumerate() {
            assert!(
                (p as usize) < n_way,
                "prediction[{i}]={p} must be a valid class index < n_way={n_way}"
            );
        }
    }

    // ── error variants ───────────────────────────────────────────────────────

    #[test]
    fn relation_score_dim_mismatch_returns_error() {
        // Net expects feat_dim=4; passing size-3 query must return DimensionMismatch.
        let net = make_net(4, 8, 1);
        let q_wrong = vec![0.0_f32; 3];
        let s_ok = vec![0.0_f32; 4];
        let result = net.relation_score(&q_wrong, &s_ok);
        assert!(
            matches!(result, Err(MetaError::DimensionMismatch { .. })),
            "wrong query size must return DimensionMismatch, got {result:?}"
        );
    }
}
