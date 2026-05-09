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
