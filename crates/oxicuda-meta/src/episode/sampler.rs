use crate::episode::types::{EpisodeConfig, FewShotEpisode};
use crate::error::{MetaError, MetaResult};
use crate::handle::LcgRng;

pub struct EpisodeSampler {
    pub config: EpisodeConfig,
}

impl EpisodeSampler {
    pub fn new(config: EpisodeConfig) -> MetaResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn sample(
        &self,
        data: &[f32],
        labels: &[u32],
        n_classes: usize,
        rng: &mut LcgRng,
    ) -> MetaResult<FewShotEpisode> {
        let cfg = &self.config;
        let n_total = labels.len();
        if n_total == 0 {
            return Err(MetaError::EmptySupport);
        }
        if data.len() != n_total * cfg.feat_dim {
            return Err(MetaError::DimensionMismatch {
                expected: n_total * cfg.feat_dim,
                got: data.len(),
            });
        }
        if n_classes < cfg.n_way {
            return Err(MetaError::InsufficientClasses {
                need: cfg.n_way,
                got: n_classes,
            });
        }

        let examples_per_class = cfg.k_shot + cfg.n_query;

        // Gather indices per class
        let mut class_indices: Vec<Vec<usize>> = (0..n_classes).map(|_| Vec::new()).collect();
        for (idx, &lbl) in labels.iter().enumerate() {
            if (lbl as usize) < n_classes {
                class_indices[lbl as usize].push(idx);
            }
        }

        for (cls, indices) in class_indices.iter().enumerate() {
            if indices.len() < examples_per_class {
                return Err(MetaError::InsufficientExamples {
                    cls,
                    need: examples_per_class,
                    got: indices.len(),
                });
            }
        }

        // Fisher-Yates to select n_way classes
        let mut class_order: Vec<usize> = (0..n_classes).collect();
        for i in (1..n_classes).rev() {
            let j = rng.next_usize(i + 1);
            class_order.swap(i, j);
        }
        let chosen_classes = &class_order[..cfg.n_way];

        let fd = cfg.feat_dim;
        let mut support_x = vec![0.0_f32; cfg.n_way * cfg.k_shot * fd];
        let mut support_y = vec![0_u32; cfg.n_way * cfg.k_shot];
        let mut query_x = vec![0.0_f32; cfg.n_way * cfg.n_query * fd];
        let mut query_y = vec![0_u32; cfg.n_way * cfg.n_query];

        for (way_idx, &cls) in chosen_classes.iter().enumerate() {
            let indices = &class_indices[cls];
            let mut perm: Vec<usize> = (0..indices.len()).collect();
            for i in (1..perm.len()).rev() {
                let j = rng.next_usize(i + 1);
                perm.swap(i, j);
            }

            for k in 0..cfg.k_shot {
                let src_idx = indices[perm[k]];
                let dst_start = (way_idx * cfg.k_shot + k) * fd;
                let src_start = src_idx * fd;
                support_x[dst_start..dst_start + fd]
                    .copy_from_slice(&data[src_start..src_start + fd]);
                support_y[way_idx * cfg.k_shot + k] = way_idx as u32;
            }

            for q in 0..cfg.n_query {
                let src_idx = indices[perm[cfg.k_shot + q]];
                let dst_start = (way_idx * cfg.n_query + q) * fd;
                let src_start = src_idx * fd;
                query_x[dst_start..dst_start + fd]
                    .copy_from_slice(&data[src_start..src_start + fd]);
                query_y[way_idx * cfg.n_query + q] = way_idx as u32;
            }
        }

        Ok(FewShotEpisode {
            config: cfg.clone(),
            support_x,
            support_y,
            query_x,
            query_y,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episode::types::EpisodeConfig;
    use crate::error::MetaError;
    use crate::handle::LcgRng;
    use std::collections::HashSet;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build a flat data/label pool where every example has a unique fingerprint:
    /// example at global index `idx` has feature `f` set to `(idx * feat_dim + f) as f32`.
    /// This makes support/query disjointness verifiable by simple value comparison.
    fn make_pool(n_classes: usize, per_class: usize, feat_dim: usize) -> (Vec<f32>, Vec<u32>) {
        let n_total = n_classes * per_class;
        let mut data = vec![0.0_f32; n_total * feat_dim];
        let mut labels = vec![0_u32; n_total];
        for cls in 0..n_classes {
            for ex in 0..per_class {
                let idx = cls * per_class + ex;
                labels[idx] = cls as u32;
                for f in 0..feat_dim {
                    data[idx * feat_dim + f] = (idx * feat_dim + f) as f32;
                }
            }
        }
        (data, labels)
    }

    fn default_cfg() -> EpisodeConfig {
        EpisodeConfig {
            n_way: 3,
            k_shot: 2,
            n_query: 4,
            feat_dim: 5,
        }
    }

    // -----------------------------------------------------------------------
    // N-way / K-shot count tests — the highest-value correctness checks
    // -----------------------------------------------------------------------

    #[test]
    fn episode_support_has_exactly_n_way_distinct_classes() {
        let cfg = default_cfg();
        let (data, labels) = make_pool(5, 10, cfg.feat_dim);
        let sampler = EpisodeSampler::new(cfg.clone()).expect("new ok");
        let mut rng = LcgRng::new(1);
        let ep = sampler
            .sample(&data, &labels, 5, &mut rng)
            .expect("sample ok");
        let unique: HashSet<u32> = ep.support_y.iter().copied().collect();
        assert_eq!(
            unique.len(),
            cfg.n_way,
            "support must span exactly n_way={} distinct classes, got {}",
            cfg.n_way,
            unique.len()
        );
        for &y in &ep.support_y {
            assert!(
                (y as usize) < cfg.n_way,
                "support label {y} out of remapped range [0, {})",
                cfg.n_way
            );
        }
    }

    #[test]
    fn episode_support_has_exactly_k_shot_examples_per_class() {
        let cfg = default_cfg();
        let (data, labels) = make_pool(5, 10, cfg.feat_dim);
        let sampler = EpisodeSampler::new(cfg.clone()).expect("new ok");
        let mut rng = LcgRng::new(2);
        let ep = sampler
            .sample(&data, &labels, 5, &mut rng)
            .expect("sample ok");
        assert_eq!(
            ep.support_y.len(),
            cfg.n_way * cfg.k_shot,
            "total support-set size must be n_way * k_shot"
        );
        let mut counts = vec![0_usize; cfg.n_way];
        for &y in &ep.support_y {
            counts[y as usize] += 1;
        }
        for (cls, &cnt) in counts.iter().enumerate() {
            assert_eq!(
                cnt, cfg.k_shot,
                "class {cls}: expected k_shot={} support examples, got {cnt}",
                cfg.k_shot
            );
        }
    }

    #[test]
    fn episode_query_has_exactly_n_query_examples_per_class() {
        let cfg = default_cfg();
        let (data, labels) = make_pool(5, 10, cfg.feat_dim);
        let sampler = EpisodeSampler::new(cfg.clone()).expect("new ok");
        let mut rng = LcgRng::new(3);
        let ep = sampler
            .sample(&data, &labels, 5, &mut rng)
            .expect("sample ok");
        assert_eq!(
            ep.query_y.len(),
            cfg.n_way * cfg.n_query,
            "total query-set size must be n_way * n_query"
        );
        let mut counts = vec![0_usize; cfg.n_way];
        for &y in &ep.query_y {
            counts[y as usize] += 1;
        }
        for (cls, &cnt) in counts.iter().enumerate() {
            assert_eq!(
                cnt, cfg.n_query,
                "class {cls}: expected n_query={} query examples, got {cnt}",
                cfg.n_query
            );
        }
    }

    // -----------------------------------------------------------------------
    // Disjointness test — no example may appear in both support and query
    // -----------------------------------------------------------------------

    #[test]
    fn support_and_query_feature_vectors_are_disjoint() {
        // make_pool assigns unique feature fingerprints, so identical slices
        // imply the same original example was used twice.
        let cfg = default_cfg();
        let (data, labels) = make_pool(5, 10, cfg.feat_dim);
        let sampler = EpisodeSampler::new(cfg.clone()).expect("new ok");
        let mut rng = LcgRng::new(4);
        let ep = sampler
            .sample(&data, &labels, 5, &mut rng)
            .expect("sample ok");
        let fd = cfg.feat_dim;
        let n_s = cfg.n_way * cfg.k_shot;
        let n_q = cfg.n_way * cfg.n_query;
        for si in 0..n_s {
            let s_slice = &ep.support_x[si * fd..(si + 1) * fd];
            for qi in 0..n_q {
                let q_slice = &ep.query_x[qi * fd..(qi + 1) * fd];
                assert_ne!(
                    s_slice, q_slice,
                    "support[{si}] and query[{qi}] must not be the same example (disjointness violated)"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Determinism test
    // -----------------------------------------------------------------------

    #[test]
    fn sample_is_deterministic_with_fixed_seed() {
        let cfg = default_cfg();
        let (data, labels) = make_pool(5, 10, cfg.feat_dim);
        let sampler = EpisodeSampler::new(cfg).expect("new ok");
        let mut rng_a = LcgRng::new(99);
        let mut rng_b = LcgRng::new(99);
        let ep_a = sampler
            .sample(&data, &labels, 5, &mut rng_a)
            .expect("sample a ok");
        let ep_b = sampler
            .sample(&data, &labels, 5, &mut rng_b)
            .expect("sample b ok");
        assert_eq!(
            ep_a.support_x, ep_b.support_x,
            "support_x must be deterministic"
        );
        assert_eq!(
            ep_a.support_y, ep_b.support_y,
            "support_y must be deterministic"
        );
        assert_eq!(ep_a.query_x, ep_b.query_x, "query_x must be deterministic");
        assert_eq!(ep_a.query_y, ep_b.query_y, "query_y must be deterministic");
    }

    // -----------------------------------------------------------------------
    // Error-handling tests
    // -----------------------------------------------------------------------

    #[test]
    fn insufficient_classes_returns_error() {
        // n_way=5 but only 3 classes available → InsufficientClasses
        let cfg = EpisodeConfig {
            n_way: 5,
            k_shot: 2,
            n_query: 2,
            feat_dim: 4,
        };
        let (data, labels) = make_pool(3, 10, cfg.feat_dim);
        let sampler = EpisodeSampler::new(cfg).expect("new ok");
        let mut rng = LcgRng::new(0);
        assert!(
            matches!(
                sampler.sample(&data, &labels, 3, &mut rng),
                Err(MetaError::InsufficientClasses { .. })
            ),
            "fewer classes than n_way must return InsufficientClasses"
        );
    }

    #[test]
    fn insufficient_examples_per_class_returns_error() {
        // k_shot + n_query = 10 but only 3 examples per class → InsufficientExamples
        let cfg = EpisodeConfig {
            n_way: 2,
            k_shot: 5,
            n_query: 5,
            feat_dim: 4,
        };
        let (data, labels) = make_pool(2, 3, cfg.feat_dim);
        let sampler = EpisodeSampler::new(cfg).expect("new ok");
        let mut rng = LcgRng::new(0);
        assert!(
            matches!(
                sampler.sample(&data, &labels, 2, &mut rng),
                Err(MetaError::InsufficientExamples { .. })
            ),
            "fewer examples than k_shot+n_query must return InsufficientExamples"
        );
    }

    #[test]
    fn invalid_n_way_less_than_two_returns_error_on_construction() {
        let cfg = EpisodeConfig {
            n_way: 1,
            k_shot: 1,
            n_query: 1,
            feat_dim: 4,
        };
        assert!(
            matches!(EpisodeSampler::new(cfg), Err(MetaError::InvalidNWay { .. })),
            "n_way < 2 must be rejected at construction time"
        );
    }
}
