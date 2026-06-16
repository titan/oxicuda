use crate::error::{RlhfError, RlhfResult};

pub struct RewardNormConfig {
    pub eps: f32,
}

pub struct RunningRewardStats {
    mean: f32,
    var: f32,
    m2: f32,
    count: usize,
    config: RewardNormConfig,
}

impl RunningRewardStats {
    pub fn new(config: RewardNormConfig) -> RlhfResult<Self> {
        if !config.eps.is_finite() || config.eps <= 0.0 {
            return Err(RlhfError::RewardNormFailed {
                msg: format!("eps must be > 0 and finite, got {}", config.eps),
            });
        }
        Ok(Self {
            mean: 0.0,
            var: 0.0,
            m2: 0.0,
            count: 0,
            config,
        })
    }

    pub fn update(&mut self, rewards: &[f32]) {
        for &x in rewards {
            self.count += 1;
            let delta = x - self.mean;
            self.mean += delta / self.count as f32;
            let delta2 = x - self.mean;
            self.m2 += delta * delta2;
            // Update variance: unbiased when count > 1, biased otherwise
            self.var = if self.count > 1 {
                self.m2 / (self.count - 1) as f32
            } else {
                0.0
            };
        }
    }

    pub fn normalize(&self, rewards: &[f32]) -> RlhfResult<Vec<f32>> {
        if self.count == 0 {
            return Err(RlhfError::RewardNormFailed {
                msg: "no data seen yet; call update() before normalize()".to_string(),
            });
        }
        let std = (self.var + self.config.eps).sqrt();
        let normalized: Vec<f32> = rewards.iter().map(|&r| (r - self.mean) / std).collect();
        for &v in &normalized {
            if !v.is_finite() {
                return Err(RlhfError::NanEncountered);
            }
        }
        Ok(normalized)
    }

    pub fn mean(&self) -> f32 {
        self.mean
    }

    pub fn std(&self) -> f32 {
        (self.var + self.config.eps).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stats(eps: f32) -> RunningRewardStats {
        RunningRewardStats::new(RewardNormConfig { eps }).expect("valid config")
    }

    #[test]
    fn normalize_output_finite() {
        let mut stats = make_stats(1e-8);
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        stats.update(&data);
        let normed = stats.normalize(&data).expect("ok");
        for v in &normed {
            assert!(v.is_finite(), "output must be finite, got {v}");
        }
    }

    #[test]
    fn zero_mean_after_normalize() {
        let mut stats = make_stats(1e-8);
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        stats.update(&data);
        let normed = stats.normalize(&data).expect("ok");
        let mean_normed = normed.iter().sum::<f32>() / normed.len() as f32;
        assert!(
            mean_normed.abs() < 1e-3,
            "mean after normalization should be ~0, got {mean_normed}"
        );
    }

    #[test]
    fn unit_std_after_normalize() {
        let mut stats = make_stats(1e-8);
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        stats.update(&data);
        let normed = stats.normalize(&data).expect("ok");
        let mean_n = normed.iter().sum::<f32>() / normed.len() as f32;
        let var_n = normed.iter().map(|&x| (x - mean_n).powi(2)).sum::<f32>() / normed.len() as f32;
        let std_n = var_n.sqrt();
        assert!(
            (std_n - 1.0).abs() < 0.05,
            "std after normalization should be ~1, got {std_n}"
        );
    }

    #[test]
    fn running_updates_converge() {
        // Known distribution: values 0..N, mean = (N-1)/2
        let mut stats = make_stats(1e-8);
        let n = 1000usize;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        stats.update(&data);
        let expected_mean = (n - 1) as f32 / 2.0;
        assert!(
            (stats.mean() - expected_mean).abs() < 0.1,
            "mean should converge to {expected_mean}, got {}",
            stats.mean()
        );
    }

    #[test]
    fn empty_rewards_ok() {
        let mut stats = make_stats(1e-8);
        // Update with data first so count > 0
        stats.update(&[1.0, 2.0, 3.0]);
        let mean_before = stats.mean();
        let count_before = stats.count;
        // Update with empty slice — should be a no-op
        stats.update(&[]);
        assert_eq!(stats.count, count_before, "count should not change");
        assert!(
            (stats.mean() - mean_before).abs() < 1e-6,
            "mean should not change after empty update"
        );
    }

    #[test]
    fn std_correct() {
        // data = [1, 3, 5]: mean=3, var=(4+0+4)/2=4, std=2, std_eps=sqrt(4+1e-8)≈2
        let mut stats = make_stats(1e-8);
        stats.update(&[1.0, 3.0, 5.0]);
        assert!(
            (stats.mean() - 3.0).abs() < 1e-5,
            "mean should be 3, got {}",
            stats.mean()
        );
        let expected_std = (4.0f32 + 1e-8f32).sqrt();
        assert!(
            (stats.std() - expected_std).abs() < 1e-4,
            "std should be {expected_std}, got {}",
            stats.std()
        );
    }

    #[test]
    fn eps_prevents_div_by_zero() {
        // Constant rewards → var=0, but eps prevents div by zero
        let mut stats = make_stats(1e-4);
        stats.update(&[5.0, 5.0, 5.0, 5.0]);
        let normed = stats.normalize(&[5.0, 5.0]).expect("ok");
        for v in &normed {
            assert!(v.is_finite(), "should not divide by zero, got {v}");
        }
    }

    #[test]
    fn multiple_updates_stable() {
        // Multiple small batch updates should give same result as one large update
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();

        let mut stats_single = make_stats(1e-8);
        stats_single.update(&data);

        let mut stats_batched = make_stats(1e-8);
        for chunk in data.chunks(10) {
            stats_batched.update(chunk);
        }

        assert!(
            (stats_single.mean() - stats_batched.mean()).abs() < 1e-3,
            "means should match: {} vs {}",
            stats_single.mean(),
            stats_batched.mean()
        );
        assert!(
            (stats_single.std() - stats_batched.std()).abs() < 0.1,
            "stds should match: {} vs {}",
            stats_single.std(),
            stats_batched.std()
        );
    }
}
