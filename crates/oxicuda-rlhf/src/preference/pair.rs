use crate::error::{RlhfError, RlhfResult};

pub struct PreferencePair {
    pub chosen_logp: f32,
    pub rejected_logp: f32,
    pub ref_chosen_logp: f32,
    pub ref_rejected_logp: f32,
}

#[derive(Debug)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::RlhfError;

    // ── PairBatch::new with matching lengths → Ok ─────────────────────────────

    #[test]
    fn pair_batch_new_matching_lengths() {
        let batch = PairBatch::new(
            vec![-0.5_f32, -1.0],
            vec![-1.5_f32, -2.0],
            vec![-0.3_f32, -0.6],
            vec![-1.2_f32, -1.8],
        )
        .expect("matching lengths must succeed");
        assert_eq!(batch.len(), 2, "batch length must be 2");
        assert!(
            !batch.is_empty(),
            "non-empty batch must not report is_empty"
        );
    }

    // ── Empty vectors → len=0, is_empty=true ─────────────────────────────────

    #[test]
    fn pair_batch_empty_is_empty() {
        let batch =
            PairBatch::new(vec![], vec![], vec![], vec![]).expect("empty vectors must succeed");
        assert_eq!(batch.len(), 0, "empty batch len must be 0");
        assert!(batch.is_empty(), "empty batch must report is_empty=true");
    }

    // ── Data is preserved exactly in the batch ────────────────────────────────

    #[test]
    fn pair_batch_data_preserved() {
        let chosen = vec![-0.1_f32, -0.2, -0.3];
        let rejected = vec![-1.1_f32, -1.2, -1.3];
        let ref_chosen = vec![-0.5_f32, -0.6, -0.7];
        let ref_rejected = vec![-1.5_f32, -1.6, -1.7];
        let batch = PairBatch::new(
            chosen.clone(),
            rejected.clone(),
            ref_chosen.clone(),
            ref_rejected.clone(),
        )
        .expect("valid batch");
        assert_eq!(batch.chosen_logps, chosen);
        assert_eq!(batch.rejected_logps, rejected);
        assert_eq!(batch.ref_chosen_logps, ref_chosen);
        assert_eq!(batch.ref_rejected_logps, ref_rejected);
    }

    // ── Mismatched rejected length → MismatchedPairLength ────────────────────

    #[test]
    fn mismatched_rejected_length_errors() {
        let err = PairBatch::new(
            vec![-0.5_f32, -1.0],
            vec![-1.5_f32],
            vec![-0.3_f32, -0.6],
            vec![-1.2_f32, -1.8],
        )
        .expect_err("rejected length mismatch must error");
        assert!(
            matches!(
                err,
                RlhfError::MismatchedPairLength {
                    chosen: 2,
                    rejected: 1
                }
            ),
            "expected MismatchedPairLength(chosen=2,rejected=1), got {err:?}"
        );
    }

    // ── Mismatched ref_chosen length → DimensionMismatch ─────────────────────

    #[test]
    fn mismatched_ref_chosen_length_errors() {
        let err = PairBatch::new(
            vec![-0.5_f32, -1.0],
            vec![-1.5_f32, -2.0],
            vec![-0.3_f32],
            vec![-1.2_f32, -1.8],
        )
        .expect_err("ref_chosen length mismatch must error");
        assert!(
            matches!(
                err,
                RlhfError::DimensionMismatch {
                    expected: 2,
                    got: 1
                }
            ),
            "expected DimensionMismatch(expected=2,got=1) for ref_chosen, got {err:?}"
        );
    }

    // ── Mismatched ref_rejected length → DimensionMismatch ───────────────────

    #[test]
    fn mismatched_ref_rejected_length_errors() {
        let err = PairBatch::new(
            vec![-0.5_f32, -1.0],
            vec![-1.5_f32, -2.0],
            vec![-0.3_f32, -0.6],
            vec![-1.2_f32],
        )
        .expect_err("ref_rejected length mismatch must error");
        assert!(
            matches!(
                err,
                RlhfError::DimensionMismatch {
                    expected: 2,
                    got: 1
                }
            ),
            "expected DimensionMismatch(expected=2,got=1) for ref_rejected, got {err:?}"
        );
    }

    // ── PreferencePair direct construction and field access ───────────────────

    #[test]
    fn preference_pair_fields() {
        let p = PreferencePair {
            chosen_logp: -0.5_f32,
            rejected_logp: -2.0_f32,
            ref_chosen_logp: -0.3_f32,
            ref_rejected_logp: -1.5_f32,
        };
        assert_eq!(p.chosen_logp, -0.5_f32);
        assert_eq!(p.rejected_logp, -2.0_f32);
        assert_eq!(p.ref_chosen_logp, -0.3_f32);
        assert_eq!(p.ref_rejected_logp, -1.5_f32);
    }
}
