use crate::error::{RlhfError, RlhfResult};

pub struct RewardNormalizer {
    mean: f64,
    m2: f64,
    count: u64,
}

impl Default for RewardNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl RewardNormalizer {
    pub fn new() -> Self {
        Self {
            mean: 0.0,
            m2: 0.0,
            count: 0,
        }
    }

    pub fn update(&mut self, r: f32) {
        self.count += 1;
        let delta = f64::from(r) - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = f64::from(r) - self.mean;
        self.m2 += delta * delta2;
    }

    pub fn normalize(&self, r: f32) -> RlhfResult<f32> {
        if self.count < 2 {
            return Err(RlhfError::RewardNormFailed {
                msg: "need at least 2 samples to normalize".into(),
            });
        }
        let variance = self.m2 / (self.count - 1) as f64;
        let std_dev = variance.sqrt();
        let eps = 1e-8_f64;
        let normalized = (f64::from(r) - self.mean) / (std_dev + eps);
        let out = normalized as f32;
        if out.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        Ok(out)
    }

    pub fn normalize_batch(&self, rs: &[f32]) -> RlhfResult<Vec<f32>> {
        rs.iter().map(|&r| self.normalize(r)).collect()
    }
}
