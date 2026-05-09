use crate::error::MetaError;

#[derive(Debug, Clone)]
pub struct EpisodeConfig {
    pub n_way: usize,
    pub k_shot: usize,
    pub n_query: usize,
    pub feat_dim: usize,
}

impl EpisodeConfig {
    pub fn validate(&self) -> Result<(), MetaError> {
        if self.n_way < 2 {
            return Err(MetaError::InvalidNWay { n_way: self.n_way });
        }
        if self.k_shot < 1 {
            return Err(MetaError::InvalidKShot {
                k_shot: self.k_shot,
            });
        }
        if self.feat_dim < 1 {
            return Err(MetaError::InvalidFeatDim { dim: self.feat_dim });
        }
        if self.n_query < 1 {
            return Err(MetaError::InvalidQuerySize { size: self.n_query });
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FewShotEpisode {
    pub config: EpisodeConfig,
    pub support_x: Vec<f32>,
    pub support_y: Vec<u32>,
    pub query_x: Vec<f32>,
    pub query_y: Vec<u32>,
}

impl FewShotEpisode {
    pub fn n_support(&self) -> usize {
        self.config.n_way * self.config.k_shot
    }

    pub fn n_query_total(&self) -> usize {
        self.config.n_way * self.config.n_query
    }

    pub fn support_for_class(&self, cls: usize) -> &[f32] {
        let fd = self.config.feat_dim;
        let k = self.config.k_shot;
        let start = cls * k * fd;
        let end = start + k * fd;
        &self.support_x[start..end]
    }
}
