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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use std::collections::BTreeSet;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    /// empty slice must yield EmptyInput
    #[test]
    fn new_empty_input_returns_error() {
        assert!(matches!(
            PopularityNegSampler::new(&[]),
            Err(RecsysError::EmptyInput)
        ));
    }

    /// all-zero counts: total==0 must also yield EmptyInput
    #[test]
    fn new_all_zero_counts_returns_error() {
        assert!(matches!(
            PopularityNegSampler::new(&[0_usize, 0, 0]),
            Err(RecsysError::EmptyInput)
        ));
    }

    /// CDF must be non-decreasing and its last entry must be exactly 1.0
    #[test]
    fn cdf_is_monotonic_and_normalized() {
        let counts = [1_usize, 2, 3, 4];
        let sampler = PopularityNegSampler::new(&counts).expect("valid counts should succeed");
        assert_eq!(sampler.cdf.len(), 4, "CDF length must equal item count");
        for i in 1..sampler.cdf.len() {
            assert!(
                sampler.cdf[i] >= sampler.cdf[i - 1],
                "CDF non-decreasing violated at index {i}: {} < {}",
                sampler.cdf[i],
                sampler.cdf[i - 1]
            );
        }
        let last = *sampler.cdf.last().expect("non-empty CDF");
        assert_eq!(last, 1.0_f32, "last CDF entry must be exactly 1.0");
    }

    /// An item with higher count must contribute a larger CDF step (more sampling mass).
    /// counts=[1,3]: item0 carries 1/4=0.25 mass, item1 carries 3/4=0.75 mass.
    #[test]
    fn higher_count_produces_larger_cdf_step() {
        let counts = [1_usize, 3];
        let sampler = PopularityNegSampler::new(&counts).expect("valid counts should succeed");
        let step0 = sampler.cdf[0];
        let step1 = sampler.cdf[1] - sampler.cdf[0];
        assert!(
            step1 > step0,
            "item1 (count=3) must have larger CDF step than item0 (count=1): step0={step0}, step1={step1}"
        );
        let eps = 1e-5_f32;
        assert!(
            (step0 - 0.25_f32).abs() < eps,
            "step0={step0} expected 0.25"
        );
        assert!(
            (step1 - 0.75_f32).abs() < eps,
            "step1={step1} expected 0.75"
        );
    }

    /// Degenerate catalog with a single item: partition_point on [1.0] always returns 0
    /// for any u01 in [0,1), so the sole item must always be sampled.
    #[test]
    fn degenerate_single_item_always_sampled() {
        let counts = [7_usize];
        let sampler = PopularityNegSampler::new(&counts).expect("valid counts should succeed");
        let positives: BTreeSet<usize> = BTreeSet::new();
        let mut rng = make_rng();
        for _ in 0..50 {
            let neg = sampler
                .sample(0, &positives, &mut rng)
                .expect("sample must succeed with no positives");
            assert_eq!(neg, 0, "only item in the catalog must always be returned");
        }
    }

    /// sample must NEVER return an item whose index is in user_positives.
    /// Stress test: 200 draws with the two highest-mass items marked positive.
    #[test]
    fn sample_never_returns_positive_item() {
        // counts: item0=10, item1=1, item2=5, item3=2, item4=8, item5=3
        // Items 0 and 4 have the most mass but are marked positive.
        let counts = [10_usize, 1, 5, 2, 8, 3];
        let sampler = PopularityNegSampler::new(&counts).expect("valid counts should succeed");
        let positives: BTreeSet<usize> = [0_usize, 4].iter().copied().collect();
        let mut rng = LcgRng::new(12345);
        for i in 0..200 {
            let neg = sampler
                .sample(0, &positives, &mut rng)
                .expect("sample must succeed");
            assert!(
                !positives.contains(&neg),
                "iteration {i}: sampled positive item {neg}"
            );
        }
    }

    /// Two RNGs seeded identically must produce an identical sample sequence.
    #[test]
    fn sample_deterministic_with_fixed_seed() {
        // pi-like counts to avoid any symmetric bias.
        let counts = [3_usize, 1, 4, 1, 5, 9, 2, 6];
        let sampler = PopularityNegSampler::new(&counts).expect("valid counts should succeed");
        let positives: BTreeSet<usize> = std::iter::once(7_usize).collect();
        let mut rng_a = LcgRng::new(99);
        let mut rng_b = LcgRng::new(99);
        for _ in 0..30 {
            let a = sampler
                .sample(0, &positives, &mut rng_a)
                .expect("sample A must succeed");
            let b = sampler
                .sample(0, &positives, &mut rng_b)
                .expect("sample B must succeed");
            assert_eq!(a, b, "same seed must produce identical samples");
        }
    }

    /// When every item is in the user's positive set, sample must exhaust its retry
    /// budget (100 attempts) and return NoNegativeAvailable for the given user id.
    #[test]
    fn no_negative_available_when_all_items_positive() {
        let counts = [1_usize, 2, 3];
        let sampler = PopularityNegSampler::new(&counts).expect("valid counts should succeed");
        let positives: BTreeSet<usize> = [0_usize, 1, 2].iter().copied().collect();
        let mut rng = make_rng();
        assert!(
            matches!(
                sampler.sample(0, &positives, &mut rng),
                Err(RecsysError::NoNegativeAvailable { user: 0 })
            ),
            "must return NoNegativeAvailable when no negatives exist"
        );
    }
}
