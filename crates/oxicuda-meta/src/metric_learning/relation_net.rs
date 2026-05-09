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
