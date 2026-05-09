use std::collections::BTreeSet;

use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

pub struct PopularityNegSampler {
    pub cdf: Vec<f32>,
    pub n_items: usize,
}

impl PopularityNegSampler {
    pub fn new(item_counts: &[usize]) -> RecsysResult<Self> {
        if item_counts.is_empty() {
            return Err(RecsysError::EmptyInput);
        }
        let total: usize = item_counts.iter().sum();
        if total == 0 {
            return Err(RecsysError::EmptyInput);
        }
        let total_f = total as f32;
        let mut cdf = Vec::with_capacity(item_counts.len());
        let mut running = 0.0_f32;
        for &c in item_counts {
            running += c as f32 / total_f;
            cdf.push(running);
        }
        // Ensure last entry is exactly 1.0
        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }
        let n_items = item_counts.len();
        Ok(Self { cdf, n_items })
    }

    pub fn sample(
        &self,
        user: usize,
        user_positives: &BTreeSet<usize>,
        rng: &mut LcgRng,
    ) -> RecsysResult<usize> {
        for _ in 0..100 {
            let u01 = rng.next_f32();
            let idx = self.cdf.partition_point(|&c| c < u01);
            let candidate = idx.min(self.n_items - 1);
            if !user_positives.contains(&candidate) {
                return Ok(candidate);
            }
        }
        Err(RecsysError::NoNegativeAvailable { user })
    }
}
