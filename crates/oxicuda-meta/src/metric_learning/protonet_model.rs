//! Struct-based Prototypical Network with MLP embedding backbone.
//!
//! Unlike the functional `proto_net` API, this module provides a `ProtoNet` struct
//! with learned embedding weights, Kaiming-initialized MLP layers, L2-normalized
//! embeddings, and episodic cross-entropy training loss.

use crate::error::{MetaError, MetaResult};
use crate::handle::LcgRng;

/// Configuration for a `ProtoNet` model.
#[derive(Debug, Clone)]
pub struct ProtoNetConfig {
    /// Dimensionality of the raw input features.
    pub d_input: usize,
    /// Dimensionality of the embedding space.
    pub d_embed: usize,
    /// Number of MLP layers (>= 1).
    pub n_layers: usize,
    /// Number of classes in each episode.
    pub n_way: usize,
    /// Number of support examples per class.
    pub n_shot: usize,
}

/// Prototypical Network with a fully-connected MLP embedding backbone.
///
/// The network maps inputs to an L2-normalized embedding space and classifies
/// query examples by nearest-prototype (Euclidean distance) in that space.
#[derive(Debug, Clone)]
pub struct ProtoNet {
    /// Weight matrices for each MLP layer, stored row-major.
    /// `embed_w[l]` has shape `[d_out × d_in]`.
    pub embed_w: Vec<Vec<f32>>,
    /// Bias vectors for each MLP layer.
    /// `embed_b[l]` has length `d_out`.
    pub embed_b: Vec<Vec<f32>>,
    /// Model configuration.
    pub config: ProtoNetConfig,
}

impl ProtoNet {
    /// Construct a new `ProtoNet` with Kaiming-initialized weights.
    ///
    /// # Errors
    ///
    /// Returns [`MetaError::InvalidFeatDim`] if `d_input == 0` or `d_embed == 0`.
    /// Returns [`MetaError::InvalidKShot`] if `n_shot < 1`.
    /// Returns [`MetaError::InvalidNWay`] if `n_way < 2`.
    /// Returns [`MetaError::InvalidEpisodeConfig`] if `n_layers < 1`.
    pub fn new(config: ProtoNetConfig, rng: &mut LcgRng) -> MetaResult<Self> {
        if config.d_input == 0 {
            return Err(MetaError::InvalidFeatDim { dim: 0 });
        }
        if config.d_embed == 0 {
            return Err(MetaError::InvalidFeatDim { dim: 0 });
        }
        if config.n_layers < 1 {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "n_layers must be >= 1".into(),
            });
        }
        if config.n_way < 2 {
            return Err(MetaError::InvalidNWay {
                n_way: config.n_way,
            });
        }
        if config.n_shot < 1 {
            return Err(MetaError::InvalidKShot {
                k_shot: config.n_shot,
            });
        }

        let mut embed_w = Vec::with_capacity(config.n_layers);
        let mut embed_b = Vec::with_capacity(config.n_layers);

        for layer in 0..config.n_layers {
            let fan_in = if layer == 0 {
                config.d_input
            } else {
                config.d_embed
            };
            let fan_out = config.d_embed;
            // Kaiming He initialization: scale = sqrt(2 / fan_in)
            let scale = (2.0_f32 / fan_in as f32).sqrt();

            let w_size = fan_out * fan_in;
            let w: Vec<f32> = (0..w_size)
                .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
                .collect();
            let b = vec![0.0_f32; fan_out];

            embed_w.push(w);
            embed_b.push(b);
        }

        Ok(Self {
            embed_w,
            embed_b,
            config,
        })
    }

    /// Forward pass through the MLP, with ReLU on all layers except the last,
    /// followed by L2 normalization.
    ///
    /// # Errors
    ///
    /// Returns [`MetaError::DimensionMismatch`] if `x.len() != d_input`.
    /// Returns [`MetaError::NanEncountered`] if the embedding norm is zero.
    pub fn embed(&self, x: &[f32]) -> MetaResult<Vec<f32>> {
        if x.len() != self.config.d_input {
            return Err(MetaError::DimensionMismatch {
                expected: self.config.d_input,
                got: x.len(),
            });
        }

        let mut activation: Vec<f32> = x.to_vec();

        for (layer_idx, (w, b)) in self.embed_w.iter().zip(self.embed_b.iter()).enumerate() {
            let fan_in = activation.len();
            let fan_out = self.config.d_embed;
            let mut next = vec![0.0_f32; fan_out];
            for o in 0..fan_out {
                let mut sum = b[o];
                for i in 0..fan_in {
                    sum += w[o * fan_in + i] * activation[i];
                }
                next[o] = sum;
            }
            // ReLU on all layers except the final one
            if layer_idx < self.config.n_layers - 1 {
                for v in next.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
            activation = next;
        }

        // L2 normalize
        let norm_sq: f32 = activation.iter().map(|&v| v * v).sum();
        let norm = norm_sq.sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return Err(MetaError::NanEncountered {
                context: "embed: L2 norm is zero or non-finite".into(),
            });
        }
        for v in activation.iter_mut() {
            *v /= norm;
        }

        Ok(activation)
    }

    /// Compute class prototypes as the mean of embedded support examples per class.
    ///
    /// `support` has shape `[n_way * n_shot * d_input]` with examples ordered
    /// class-major: first `n_shot` examples belong to class 0, next to class 1, etc.
    ///
    /// Returns a flat vector of shape `[n_way * d_embed]`.
    ///
    /// # Errors
    ///
    /// Returns [`MetaError::DimensionMismatch`] if `support.len() != n_way * n_shot * d_input`.
    /// Returns [`MetaError::InvalidNWay`] if `n_way < 2`.
    /// Returns [`MetaError::InvalidKShot`] if `n_shot < 1`.
    pub fn prototypes(&self, support: &[f32], n_way: usize, n_shot: usize) -> MetaResult<Vec<f32>> {
        if n_way < 2 {
            return Err(MetaError::InvalidNWay { n_way });
        }
        if n_shot < 1 {
            return Err(MetaError::InvalidKShot { k_shot: n_shot });
        }
        let expected = n_way * n_shot * self.config.d_input;
        if support.len() != expected {
            return Err(MetaError::DimensionMismatch {
                expected,
                got: support.len(),
            });
        }

        let d = self.config.d_embed;
        let mut protos = vec![0.0_f32; n_way * d];

        for cls in 0..n_way {
            for shot in 0..n_shot {
                let start = (cls * n_shot + shot) * self.config.d_input;
                let x = &support[start..start + self.config.d_input];
                let emb = self.embed(x)?;
                for (j, &v) in emb.iter().enumerate() {
                    protos[cls * d + j] += v;
                }
            }
            let scale = 1.0 / n_shot as f32;
            for j in 0..d {
                protos[cls * d + j] *= scale;
            }
        }

        Ok(protos)
    }

    /// Predict the class of a query example using nearest-prototype (L2) distance.
    ///
    /// `prototypes` is a flat vector of shape `[n_way * d_embed]`.
    /// Returns the predicted class index in `[0, n_way)`.
    ///
    /// # Errors
    ///
    /// Returns [`MetaError::DimensionMismatch`] if sizes are inconsistent.
    /// Returns [`MetaError::InvalidNWay`] if `n_way < 2`.
    pub fn predict(&self, query: &[f32], prototypes: &[f32], n_way: usize) -> MetaResult<usize> {
        if n_way < 2 {
            return Err(MetaError::InvalidNWay { n_way });
        }
        let d = self.config.d_embed;
        if prototypes.len() != n_way * d {
            return Err(MetaError::DimensionMismatch {
                expected: n_way * d,
                got: prototypes.len(),
            });
        }

        let q_emb = self.embed(query)?;

        let mut best_cls = 0usize;
        let mut best_dist = f32::INFINITY;

        for cls in 0..n_way {
            let proto = &prototypes[cls * d..(cls + 1) * d];
            let dist: f32 = q_emb
                .iter()
                .zip(proto.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();
            if dist < best_dist {
                best_dist = dist;
                best_cls = cls;
            }
        }

        Ok(best_cls)
    }

    /// Compute the episodic negative log-softmax loss using squared L2 distances
    /// as negative logits (Snell et al. 2017, Eq. 2).
    ///
    /// `support` shape: `[n_way * n_shot * d_input]`
    /// `queries` shape: `[n_query * d_input]`
    /// `query_labels` shape: `[n_query]`, values in `[0, n_way)`
    ///
    /// # Errors
    ///
    /// Returns [`MetaError::DimensionMismatch`] on shape mismatches.
    /// Returns [`MetaError::NanEncountered`] if loss is non-finite.
    pub fn episode_loss(
        &self,
        support: &[f32],
        queries: &[f32],
        query_labels: &[usize],
        n_way: usize,
        n_shot: usize,
        n_query: usize,
    ) -> MetaResult<f32> {
        if n_way < 2 {
            return Err(MetaError::InvalidNWay { n_way });
        }
        if n_shot < 1 {
            return Err(MetaError::InvalidKShot { k_shot: n_shot });
        }
        if n_query == 0 {
            return Err(MetaError::InvalidQuerySize { size: 0 });
        }
        let expected_q = n_query * self.config.d_input;
        if queries.len() != expected_q {
            return Err(MetaError::DimensionMismatch {
                expected: expected_q,
                got: queries.len(),
            });
        }
        if query_labels.len() != n_query {
            return Err(MetaError::DimensionMismatch {
                expected: n_query,
                got: query_labels.len(),
            });
        }

        let protos = self.prototypes(support, n_way, n_shot)?;
        let d = self.config.d_embed;

        let mut total_loss = 0.0_f32;

        for (q_idx, &true_cls) in query_labels.iter().enumerate() {
            let q_start = q_idx * self.config.d_input;
            let q_x = &queries[q_start..q_start + self.config.d_input];
            let q_emb = self.embed(q_x)?;

            // Negative squared L2 distances as logits
            let neg_dists: Vec<f32> = (0..n_way)
                .map(|cls| {
                    let proto = &protos[cls * d..(cls + 1) * d];
                    let sq_dist: f32 = q_emb
                        .iter()
                        .zip(proto.iter())
                        .map(|(&a, &b)| (a - b) * (a - b))
                        .sum();
                    -sq_dist
                })
                .collect();

            // Numerically-stable softmax and log-softmax
            let max_nd = neg_dists.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = neg_dists.iter().map(|&nd| (nd - max_nd).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            if !sum_exp.is_finite() || sum_exp == 0.0 {
                return Err(MetaError::NanEncountered {
                    context: "episode_loss: sum_exp is zero or non-finite".into(),
                });
            }

            if true_cls >= n_way {
                return Err(MetaError::InvalidEpisodeConfig {
                    msg: format!("query label {true_cls} >= n_way {n_way}"),
                });
            }
            let log_prob = (exps[true_cls] / sum_exp).ln();
            if !log_prob.is_finite() {
                return Err(MetaError::NanEncountered {
                    context: "episode_loss: log_prob is non-finite".into(),
                });
            }
            total_loss -= log_prob;
        }

        let loss = total_loss / n_query as f32;
        if !loss.is_finite() {
            return Err(MetaError::NanEncountered {
                context: "episode_loss: final loss is non-finite".into(),
            });
        }
        Ok(loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_net(d_input: usize, d_embed: usize, n_layers: usize) -> ProtoNet {
        let cfg = ProtoNetConfig {
            d_input,
            d_embed,
            n_layers,
            n_way: 3,
            n_shot: 2,
        };
        ProtoNet::new(cfg, &mut LcgRng::new(42)).expect("value should be present")
    }

    #[test]
    fn embed_shape() {
        let net = make_net(8, 4, 2);
        let x = vec![0.5_f32; 8];
        let emb = net.embed(&x).expect("embed should succeed");
        assert_eq!(emb.len(), 4, "embed output length must equal d_embed");
    }

    #[test]
    fn prototypes_shape() {
        let net = make_net(8, 4, 2);
        let support = vec![0.1_f32; 3 * 2 * 8]; // n_way=3, n_shot=2
        let protos = net
            .prototypes(&support, 3, 2)
            .expect("prototypes should succeed");
        assert_eq!(
            protos.len(),
            3 * 4,
            "prototypes length must be n_way * d_embed"
        );
    }

    #[test]
    fn predict_in_range() {
        let net = make_net(6, 4, 2);
        let support = vec![0.2_f32; 3 * 2 * 6];
        let protos = net
            .prototypes(&support, 3, 2)
            .expect("prototypes should succeed");
        let query = vec![0.3_f32; 6];
        let pred = net
            .predict(&query, &protos, 3)
            .expect("predict should succeed");
        assert!(pred < 3, "predicted class must be in [0, n_way)");
    }

    #[test]
    fn episode_loss_finite() {
        let net = make_net(6, 4, 2);
        let support = vec![0.1_f32; 3 * 2 * 6];
        let queries = vec![0.2_f32; 4 * 6];
        let labels = vec![0usize, 1, 2, 0];
        let loss = net
            .episode_loss(&support, &queries, &labels, 3, 2, 4)
            .expect("value should be present");
        assert!(loss.is_finite(), "episode loss must be finite, got {loss}");
    }

    #[test]
    fn episode_loss_nonneg() {
        let net = make_net(6, 4, 2);
        let support = vec![0.1_f32; 3 * 2 * 6];
        let queries = vec![0.2_f32; 4 * 6];
        let labels = vec![0usize, 1, 2, 0];
        let loss = net
            .episode_loss(&support, &queries, &labels, 3, 2, 4)
            .expect("value should be present");
        assert!(loss >= 0.0, "episode loss must be non-negative, got {loss}");
    }

    #[test]
    fn single_shot_works() {
        let cfg = ProtoNetConfig {
            d_input: 4,
            d_embed: 4,
            n_layers: 1,
            n_way: 2,
            n_shot: 1,
        };
        let net = ProtoNet::new(cfg, &mut LcgRng::new(7)).expect("value should be present");
        let support = vec![0.5_f32; 2 * 4];
        let protos = net
            .prototypes(&support, 2, 1)
            .expect("prototypes should succeed");
        assert_eq!(protos.len(), 2 * 4);
    }

    #[test]
    fn same_class_predicts_correctly() {
        // Build a network, embed a support example, use it as the query too.
        // The query == the only support example for its class, so it should predict that class.
        let cfg = ProtoNetConfig {
            d_input: 4,
            d_embed: 4,
            n_layers: 1,
            n_way: 2,
            n_shot: 1,
        };
        let net = ProtoNet::new(cfg, &mut LcgRng::new(17)).expect("value should be present");
        // Two clearly-separated support classes
        let mut support = vec![0.0_f32; 2 * 4];
        // class 0: [1,0,0,0], class 1: [0,1,0,0]
        support[0] = 1.0;
        support[5] = 1.0;
        let protos = net
            .prototypes(&support, 2, 1)
            .expect("prototypes should succeed");
        // Query identical to class-0 support
        let query = vec![1.0_f32, 0.0, 0.0, 0.0];
        let pred = net
            .predict(&query, &protos, 2)
            .expect("predict should succeed");
        assert_eq!(
            pred, 0,
            "query identical to class-0 support should predict class 0"
        );
    }

    #[test]
    fn n_way_1_trivial() {
        let cfg = ProtoNetConfig {
            d_input: 4,
            d_embed: 4,
            n_layers: 1,
            n_way: 1,
            n_shot: 1,
        };
        let result = ProtoNet::new(cfg, &mut LcgRng::new(1));
        assert!(
            matches!(result, Err(MetaError::InvalidNWay { .. })),
            "n_way=1 must return InvalidNWay"
        );
    }

    #[test]
    fn d_embed_0_error() {
        let cfg = ProtoNetConfig {
            d_input: 4,
            d_embed: 0,
            n_layers: 1,
            n_way: 2,
            n_shot: 1,
        };
        let result = ProtoNet::new(cfg, &mut LcgRng::new(1));
        assert!(result.is_err(), "d_embed=0 must return Err");
    }

    #[test]
    fn prototypes_finite() {
        let net = make_net(8, 4, 2);
        let support = vec![0.3_f32; 3 * 2 * 8];
        let protos = net
            .prototypes(&support, 3, 2)
            .expect("prototypes should succeed");
        for &v in &protos {
            assert!(v.is_finite(), "all prototype values must be finite");
        }
    }
}
