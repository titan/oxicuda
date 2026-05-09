use crate::error::{RlhfError, RlhfResult};

pub struct PreferencePair {
    pub chosen_logp: f32,
    pub rejected_logp: f32,
    pub ref_chosen_logp: f32,
    pub ref_rejected_logp: f32,
}

pub struct PairBatch {
    pub chosen_logps: Vec<f32>,
    pub rejected_logps: Vec<f32>,
    pub ref_chosen_logps: Vec<f32>,
    pub ref_rejected_logps: Vec<f32>,
}

impl PairBatch {
    pub fn new(
        chosen_logps: Vec<f32>,
        rejected_logps: Vec<f32>,
        ref_chosen_logps: Vec<f32>,
        ref_rejected_logps: Vec<f32>,
    ) -> RlhfResult<Self> {
        let n = chosen_logps.len();
        if rejected_logps.len() != n {
            return Err(RlhfError::MismatchedPairLength {
                chosen: n,
                rejected: rejected_logps.len(),
            });
        }
        if ref_chosen_logps.len() != n {
            return Err(RlhfError::DimensionMismatch {
                expected: n,
                got: ref_chosen_logps.len(),
            });
        }
        if ref_rejected_logps.len() != n {
            return Err(RlhfError::DimensionMismatch {
                expected: n,
                got: ref_rejected_logps.len(),
            });
        }
        Ok(Self {
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        })
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.chosen_logps.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chosen_logps.is_empty()
    }
}
