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
