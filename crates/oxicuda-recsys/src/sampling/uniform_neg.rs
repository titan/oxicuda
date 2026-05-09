use std::collections::BTreeSet;

use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

pub struct UniformNegSampler {
    pub n_items: usize,
}

impl UniformNegSampler {
    pub fn new(n_items: usize) -> RecsysResult<Self> {
        if n_items == 0 {
            return Err(RecsysError::InvalidNumItems { n: n_items });
        }
        Ok(Self { n_items })
    }

    pub fn sample(
        &self,
        user: usize,
        user_positives: &BTreeSet<usize>,
        rng: &mut LcgRng,
    ) -> RecsysResult<usize> {
        for _ in 0..100 {
            let candidate = (rng.next_u32() as usize) % self.n_items;
            if !user_positives.contains(&candidate) {
                return Ok(candidate);
            }
        }
        Err(RecsysError::NoNegativeAvailable { user })
    }
}
