use crate::error::{RlhfError, RlhfResult};

pub fn masked_token_ce(logits: &[f32], label: u32, n_vocab: usize) -> RlhfResult<f32> {
    if logits.len() != n_vocab {
        return Err(RlhfError::DimensionMismatch {
            expected: n_vocab,
            got: logits.len(),
        });
    }
    if label as usize >= n_vocab {
        return Err(RlhfError::Internal {
            msg: format!("label {label} >= n_vocab {n_vocab}"),
        });
    }
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
    let log_sum_exp = sum_exp.ln() + max_logit;
    let ce = log_sum_exp - logits[label as usize];
    if ce.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(ce)
}

pub fn sft_loss(logits: &[f32], labels: &[u32], mask: &[u8], n_vocab: usize) -> RlhfResult<f32> {
    if labels.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    let t = labels.len();
    if logits.len() != t * n_vocab {
        return Err(RlhfError::DimensionMismatch {
            expected: t * n_vocab,
            got: logits.len(),
        });
    }
    if mask.len() != t {
        return Err(RlhfError::DimensionMismatch {
            expected: t,
            got: mask.len(),
        });
    }
    for &m in mask {
        if m > 1 {
            return Err(RlhfError::InvalidMaskValue);
        }
    }
    let mut loss_sum = 0.0_f32;
    let mut mask_sum = 0u32;
    for (pos, ((&label, &m), token_logits)) in labels
        .iter()
        .zip(mask.iter())
        .zip(logits.chunks_exact(n_vocab))
        .enumerate()
    {
        let _ = pos;
        if m == 0 {
            continue;
        }
        loss_sum += masked_token_ce(token_logits, label, n_vocab)?;
        mask_sum += 1;
    }
    if mask_sum == 0 {
        return Ok(0.0);
    }
    let loss = loss_sum / mask_sum as f32;
    if loss.is_nan() {
        return Err(RlhfError::NanEncountered);
    }
    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::RlhfError;

    // ── masked_token_ce: uniform logits → CE = ln(V) exactly ─────────────────

    #[test]
    fn uniform_logits_ce_equals_ln_vocab() {
        // For V classes all with logit 0:
        //   sum_exp = V, log_sum_exp = ln(V), CE = ln(V) − 0 = ln(V).
        let v = 4_usize;
        let logits = vec![0.0_f32; v];
        let ce = masked_token_ce(&logits, 0, v).expect("valid uniform inputs");
        let expected = (v as f32).ln();
        assert!(
            (ce - expected).abs() < 1e-5,
            "uniform logits V={v}: CE expected ln({v})={expected:.6}, got {ce:.6}"
        );
    }

    // ── masked_token_ce: confident-correct logit → CE ≈ 0 ───────────────────

    #[test]
    fn confident_correct_ce_near_zero() {
        // logits=[50, 0, 0], label=0: model is very confident in class 0.
        // CE = log_sum_exp − logits[0] ≈ 50 − 50 = 0 (exp(−50) terms vanish).
        let logits = vec![50.0_f32, 0.0, 0.0];
        let ce = masked_token_ce(&logits, 0, 3).expect("confident-correct inputs");
        assert!(ce < 1e-4, "confident-correct CE expected ≈0, got {ce}");
    }

    // ── masked_token_ce: hand-computed analytic value ────────────────────────

    #[test]
    fn two_class_ce_analytic() {
        // logits=[1.0, 2.0], label=1 (correct class), n_vocab=2.
        // max=2, sum_exp = exp(−1) + 1 = 1.36787944, log_sum_exp = ln(1.36787944)+2
        // CE = log_sum_exp − logits[1] = ln(1 + exp(−1)) ≈ 0.31326169
        let logits = vec![1.0_f32, 2.0];
        let ce = masked_token_ce(&logits, 1, 2).expect("analytic 2-class inputs");
        let expected = (1.0_f32 + (-1.0_f32).exp()).ln();
        assert!(
            (ce - expected).abs() < 1e-5,
            "2-class analytic CE: expected {expected:.6}, got {ce:.6}"
        );
    }

    // ── sft_loss: mask=0 excludes positions from the average ─────────────────

    #[test]
    fn mask_excludes_position_from_average() {
        // Token 0: uniform V=2 logits → CE = ln(2) ≈ 0.693147
        // Token 1: very confident correct (label=0) → CE ≈ 0
        // mask=[1,0]: only token 0 contributes → loss = ln(2)
        // mask=[1,1]: both tokens → loss ≈ ln(2)/2
        // Confirms that the denominator (mask_sum) changes, not just the numerator.
        let logits = vec![
            0.0_f32, 0.0, // token 0: uniform over 2 classes
            100.0, -100.0, // token 1: confident class 0
        ];
        let labels = vec![0_u32, 0];
        let loss_10 = sft_loss(&logits, &labels, &[1_u8, 0], 2).expect("mask=[1,0]");
        let loss_11 = sft_loss(&logits, &labels, &[1_u8, 1], 2).expect("mask=[1,1]");
        let ln2 = 2.0_f32.ln();
        assert!(
            (loss_10 - ln2).abs() < 1e-4,
            "mask=[1,0]: loss expected ln(2)={ln2:.6}, got {loss_10:.6}"
        );
        assert!(
            (loss_11 - ln2 / 2.0).abs() < 1e-3,
            "mask=[1,1]: loss expected ≈ln(2)/2={:.6}, got {loss_11:.6}",
            ln2 / 2.0
        );
        assert!(
            loss_10 > loss_11,
            "mask=[1,0] loss ({loss_10}) must exceed mask=[1,1] loss ({loss_11})"
        );
    }

    // ── sft_loss: all-zero mask → exactly 0.0 ────────────────────────────────

    #[test]
    fn all_zero_mask_returns_zero() {
        let logits = vec![1.0_f32, 0.0, 0.5, 0.5];
        let labels = vec![0_u32, 1];
        let loss = sft_loss(&logits, &labels, &[0_u8, 0], 2).expect("all-masked input");
        assert_eq!(loss, 0.0_f32, "all-zero mask must return exactly 0.0");
    }

    // ── sft_loss: dimension mismatches → DimensionMismatch ───────────────────

    #[test]
    fn dimension_mismatch_errors() {
        // logits length ≠ t * n_vocab (t=1, n_vocab=3 → expect 3, got 2)
        let err = sft_loss(&[0.0_f32, 0.0], &[0_u32], &[1_u8], 3)
            .expect_err("logits length mismatch must error");
        assert!(
            matches!(err, RlhfError::DimensionMismatch { .. }),
            "expected DimensionMismatch for logits, got {err:?}"
        );
        // mask length ≠ t (t=1, mask.len()=2)
        let err2 = sft_loss(&[0.0_f32, 0.0], &[0_u32], &[1_u8, 1], 2)
            .expect_err("mask length mismatch must error");
        assert!(
            matches!(err2, RlhfError::DimensionMismatch { .. }),
            "expected DimensionMismatch for mask, got {err2:?}"
        );
    }

    // ── masked_token_ce: label ≥ n_vocab → Internal error ───────────────────

    #[test]
    fn label_out_of_bounds_errors() {
        let err = masked_token_ce(&[0.0_f32, 0.0, 0.0], 5, 3)
            .expect_err("label=5 >= n_vocab=3 must error");
        assert!(
            matches!(err, RlhfError::Internal { .. }),
            "expected Internal error for out-of-bounds label, got {err:?}"
        );
    }
}
