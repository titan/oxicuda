//! Spiking spatial pooling — `Max` (any spike fires output) or `Avg` (spike count fraction).

use crate::error::{SnnError, SnnResult};

/// Pooling reduction kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolKind {
    /// Output = max over the window (any spike => 1).
    Max,
    /// Output = mean over the window (fraction of spikes).
    Avg,
}

/// Pool a 2-D spike map. Single channel; for multi-channel call once per channel.
///
/// `spikes` length must equal `in_h * in_w`. Output length must equal `(in_h/kh) * (in_w/kw)`.
pub fn spike_pool(
    spikes: &[f32],
    in_h: usize,
    in_w: usize,
    kh: usize,
    kw: usize,
    kind: PoolKind,
    out: &mut [f32],
) -> SnnResult<()> {
    if kh == 0 || kw == 0 {
        return Err(SnnError::BadDim { got: 0 });
    }
    if !in_h.is_multiple_of(kh) || !in_w.is_multiple_of(kw) {
        return Err(SnnError::OutOfRange {
            name: "input shape divisible by kernel".to_string(),
            val: in_h as f32,
        });
    }
    if spikes.len() != in_h * in_w {
        return Err(SnnError::BadShape {
            expected: in_h * in_w,
            got: spikes.len(),
        });
    }
    let oh = in_h / kh;
    let ow = in_w / kw;
    if out.len() != oh * ow {
        return Err(SnnError::BadShape {
            expected: oh * ow,
            got: out.len(),
        });
    }
    let win_area = (kh * kw) as f32;
    for r in 0..oh {
        for c in 0..ow {
            let mut acc = match kind {
                PoolKind::Max => f32::MIN,
                PoolKind::Avg => 0.0_f32,
            };
            for u in 0..kh {
                for v in 0..kw {
                    let val = spikes[(r * kh + u) * in_w + (c * kw + v)];
                    match kind {
                        PoolKind::Max => {
                            if val > acc {
                                acc = val;
                            }
                        }
                        PoolKind::Avg => acc += val,
                    }
                }
            }
            let out_val = match kind {
                PoolKind::Max => acc,
                PoolKind::Avg => acc / win_area,
            };
            out[r * ow + c] = out_val;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_zeros_returns_zeros() {
        let s = vec![0.0_f32; 16];
        let mut out = vec![0.0_f32; 4];
        spike_pool(&s, 4, 4, 2, 2, PoolKind::Max, &mut out).expect("ok");
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn max_ones_returns_ones() {
        let s = vec![1.0_f32; 16];
        let mut out = vec![0.0_f32; 4];
        spike_pool(&s, 4, 4, 2, 2, PoolKind::Max, &mut out).expect("ok");
        for &v in &out {
            assert_eq!(v, 1.0);
        }
    }

    #[test]
    fn avg_half_pattern() {
        // alternating row pattern: 1 0 1 0 / 1 0 1 0 / ... → 2x2 avg = 0.5
        let s: Vec<f32> = (0..16)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let mut out = vec![0.0_f32; 4];
        spike_pool(&s, 4, 4, 2, 2, PoolKind::Avg, &mut out).expect("ok");
        for &v in &out {
            assert!((v - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn unaligned_kernel_errors() {
        let s = vec![0.0_f32; 9];
        let mut out = vec![0.0_f32; 1];
        assert!(spike_pool(&s, 3, 3, 2, 2, PoolKind::Max, &mut out).is_err());
    }
}
