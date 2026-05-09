//! Analytic FLOP / parameter accountant for [`OpKind`] primitives.
//!
//! Counts multiply-add (MAC) operations as 2 FLOPs each, matching the standard
//! "MAdds × 2" convention used by FBNet and many hardware-aware NAS papers.
//!
//! All formulas assume `H × W` input spatial resolution and stride 1
//! (NAS reduction cells are handled by the caller via the `LayerSpec.h/w`
//! values reflecting the post-stride spatial size).

use crate::error::{NasError, NasResult};
use crate::ops::OpKind;
use crate::predictor::predictor_io::LayerSpec;

/// Cost summary for a single layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpCost {
    /// Number of trainable parameters.
    pub params: u64,
    /// Number of FLOPs (= 2 × MACs by convention).
    pub flops: u64,
}

impl OpCost {
    /// Combine two costs additively.
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        Self {
            params: self.params.saturating_add(other.params),
            flops: self.flops.saturating_add(other.flops),
        }
    }

    /// Zero cost (unit element under [`Self::merge`]).
    #[must_use]
    pub fn zero() -> Self {
        Self {
            params: 0,
            flops: 0,
        }
    }
}

/// Compute the parameter and FLOP counts for a single layer.
///
/// Formulas:
/// - `Zero` / `Identity` / pooling — 0 params, FLOPs include element-wise compares.
/// - `SepConv KxK` — depthwise: `K² · in_ch + 2 · K² · in_ch · H · W` MACs;
///   pointwise: `in_ch · out_ch + 2 · in_ch · out_ch · H · W` MACs.
/// - `DilConv KxK` — same as `SepConv KxK` (dilation does not change FLOPs).
///
/// # Errors
/// - [`NasError::DimensionMismatch`] if `Identity` is requested with `in_ch != out_ch`.
pub fn op_cost(layer: &LayerSpec) -> NasResult<OpCost> {
    let LayerSpec {
        op,
        in_channels: cin,
        out_channels: cout,
        h,
        w,
    } = *layer;
    let hw = (h as u64) * (w as u64);
    match op {
        OpKind::Zero => Ok(OpCost {
            params: 0,
            flops: 0,
        }),
        OpKind::Identity => {
            if cin != cout {
                return Err(NasError::DimensionMismatch {
                    expected: cin,
                    got: cout,
                });
            }
            Ok(OpCost {
                params: 0,
                flops: 0,
            })
        }
        OpKind::MaxPool3x3 | OpKind::AvgPool3x3 => Ok(OpCost {
            params: 0,
            // 3×3 window per output element; treat each compare/add as 1 FLOP.
            flops: 9 * (cout as u64) * hw,
        }),
        OpKind::SepConv3x3 | OpKind::DilConv3x3 => Ok(sep_conv_cost(cin, cout, hw, 3)),
        OpKind::SepConv5x5 | OpKind::DilConv5x5 => Ok(sep_conv_cost(cin, cout, hw, 5)),
    }
}

fn sep_conv_cost(cin: usize, cout: usize, hw: u64, k: u64) -> OpCost {
    let cin64 = cin as u64;
    let cout64 = cout as u64;
    let k2 = k * k;
    // Depthwise: per input channel, K² weights; FLOPs = 2 · K² · cin · HW.
    let dw_params = k2 * cin64;
    let dw_flops = 2 * k2 * cin64 * hw;
    // Pointwise: cin × cout 1×1 conv; FLOPs = 2 · cin · cout · HW.
    let pw_params = cin64 * cout64;
    let pw_flops = 2 * cin64 * cout64 * hw;
    OpCost {
        params: dw_params.saturating_add(pw_params),
        flops: dw_flops.saturating_add(pw_flops),
    }
}

/// Compute total params + FLOPs across a stack of layers.
///
/// # Errors
/// Propagates errors from [`op_cost`]; returns [`NasError::EmptySearchSpace`]
/// for empty input.
pub fn total_cost(layers: &[LayerSpec]) -> NasResult<OpCost> {
    if layers.is_empty() {
        return Err(NasError::EmptySearchSpace);
    }
    let mut total = OpCost::zero();
    for layer in layers {
        total = total.merge(op_cost(layer)?);
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_cost_zero_is_free() {
        let l = LayerSpec::new(OpKind::Zero, 4, 4, 8, 8);
        let c = op_cost(&l).unwrap();
        assert_eq!(c.params, 0);
        assert_eq!(c.flops, 0);
    }

    #[test]
    fn op_cost_identity_requires_matching_channels() {
        let ok = LayerSpec::new(OpKind::Identity, 4, 4, 8, 8);
        let bad = LayerSpec::new(OpKind::Identity, 4, 8, 8, 8);
        assert!(op_cost(&ok).is_ok());
        assert!(op_cost(&bad).is_err());
    }

    #[test]
    fn op_cost_avg_pool_proportional_to_spatial() {
        let l1 = LayerSpec::new(OpKind::AvgPool3x3, 4, 4, 8, 8);
        let l2 = LayerSpec::new(OpKind::AvgPool3x3, 4, 4, 16, 16);
        let c1 = op_cost(&l1).unwrap();
        let c2 = op_cost(&l2).unwrap();
        assert_eq!(c2.flops, 4 * c1.flops);
    }

    #[test]
    fn op_cost_sep_conv_3x3_formula() {
        let cin = 16;
        let cout = 32;
        let h = 8;
        let w = 8;
        let l = LayerSpec::new(OpKind::SepConv3x3, cin, cout, h, w);
        let c = op_cost(&l).unwrap();
        let hw = (h * w) as u64;
        let expected_dw_params = 9 * cin as u64;
        let expected_pw_params = (cin * cout) as u64;
        assert_eq!(c.params, expected_dw_params + expected_pw_params);
        let expected_dw_flops = 2 * 9 * cin as u64 * hw;
        let expected_pw_flops = 2 * (cin * cout) as u64 * hw;
        assert_eq!(c.flops, expected_dw_flops + expected_pw_flops);
    }

    #[test]
    fn op_cost_sep_conv_5x5_more_expensive_than_3x3() {
        let l3 = LayerSpec::new(OpKind::SepConv3x3, 8, 8, 8, 8);
        let l5 = LayerSpec::new(OpKind::SepConv5x5, 8, 8, 8, 8);
        let c3 = op_cost(&l3).unwrap();
        let c5 = op_cost(&l5).unwrap();
        assert!(c5.flops > c3.flops);
        assert!(c5.params > c3.params);
    }

    #[test]
    fn op_cost_dil_conv_same_as_sep_conv() {
        let l_sep = LayerSpec::new(OpKind::SepConv3x3, 8, 16, 8, 8);
        let l_dil = LayerSpec::new(OpKind::DilConv3x3, 8, 16, 8, 8);
        assert_eq!(op_cost(&l_sep).unwrap(), op_cost(&l_dil).unwrap());
    }

    #[test]
    fn total_cost_sums_layers() {
        let layers = vec![
            LayerSpec::new(OpKind::SepConv3x3, 8, 16, 16, 16),
            LayerSpec::new(OpKind::SepConv3x3, 16, 16, 16, 16),
            LayerSpec::new(OpKind::AvgPool3x3, 16, 16, 16, 16),
        ];
        let total = total_cost(&layers).unwrap();
        let manual = op_cost(&layers[0])
            .unwrap()
            .merge(op_cost(&layers[1]).unwrap())
            .merge(op_cost(&layers[2]).unwrap());
        assert_eq!(total, manual);
    }

    #[test]
    fn total_cost_rejects_empty() {
        assert!(total_cost(&[]).is_err());
    }

    #[test]
    fn op_cost_merge_associative() {
        let a = OpCost {
            params: 10,
            flops: 200,
        };
        let b = OpCost {
            params: 5,
            flops: 80,
        };
        let c = OpCost {
            params: 1,
            flops: 4,
        };
        let abc = a.merge(b).merge(c);
        let bca = b.merge(c).merge(a);
        assert_eq!(abc, bca);
    }
}
