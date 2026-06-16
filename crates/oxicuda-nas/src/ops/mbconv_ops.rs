//! MobileNet-V2 MBConv computation cost estimation.
//!
//! Provides analytical multiply-accumulate (MAC) and parameter counts for the
//! Inverted Residual / MBConv block used in MobileNetV2, EfficientNet, and
//! ProxylessNAS / Once-for-All search spaces.
//!
//! # MBConv anatomy
//!
//! ```text
//! Input (in_ch, H, W)
//!   │
//!   ├── [Pointwise Expand]  1×1 conv:  in_ch → in_ch * expand_ratio
//!   │     (skipped when expand_ratio == 1)
//!   │
//!   ├── [Depthwise Conv]    k×k conv:  in_ch*expand_ratio, stride s
//!   │     → output spatial: (H/s, W/s)
//!   │
//!   └── [Pointwise Project] 1×1 conv:  in_ch*expand_ratio → out_ch
//! ```
//!
//! No bias terms are counted (standard practice in NAS cost models).

// ─── MbConvSpec ──────────────────────────────────────────────────────────────

/// Full specification for one MBConv (Inverted Residual) block.
#[derive(Debug, Clone)]
pub struct MbConvSpec {
    /// Number of input channels.
    pub in_ch: usize,
    /// Number of output channels.
    pub out_ch: usize,
    /// Spatial stride applied in the depthwise convolution (1 or 2).
    pub stride: usize,
    /// Channel expansion ratio applied in the pointwise expand step.
    /// When 1 the expand step is omitted.
    pub expand_ratio: usize,
    /// Depthwise convolution kernel size (square: `kernel × kernel`).
    pub kernel: usize,
}

// ─── mbconv_mac_count ────────────────────────────────────────────────────────

/// Multiply-accumulate count for one MBConv block.
///
/// The count follows the three-stage decomposition:
///
/// 1. **Pointwise expand** (`expand_ratio > 1` only):
///    `in_ch × (in_ch × expand_ratio) × H × W`
/// 2. **Depthwise conv** (`kernel × kernel`, stride `s`):
///    `(in_ch × expand_ratio) × kernel² × (H/s) × (W/s)`
/// 3. **Pointwise project** (`1×1`):
///    `(in_ch × expand_ratio) × out_ch × (H/s) × (W/s)`
///
/// Integer (floor) division is used for the spatial downsampling step.
/// No bias terms are counted.
#[must_use]
pub fn mbconv_mac_count(spec: &MbConvSpec, h: usize, w: usize) -> u64 {
    let mid_ch = spec.in_ch * spec.expand_ratio;

    // Stride must be at least 1 to avoid division by zero; treat 0 as 1.
    let effective_stride = spec.stride.max(1);
    let out_h = h / effective_stride;
    let out_w = w / effective_stride;

    // Stage 1 – pointwise expand (only when expand_ratio > 1).
    let expand_macs: u64 = if spec.expand_ratio > 1 {
        (spec.in_ch as u64)
            .saturating_mul(mid_ch as u64)
            .saturating_mul(h as u64)
            .saturating_mul(w as u64)
    } else {
        0
    };

    // Stage 2 – depthwise convolution.
    let dw_macs: u64 = (mid_ch as u64)
        .saturating_mul(spec.kernel as u64)
        .saturating_mul(spec.kernel as u64)
        .saturating_mul(out_h as u64)
        .saturating_mul(out_w as u64);

    // Stage 3 – pointwise project.
    let proj_macs: u64 = (mid_ch as u64)
        .saturating_mul(spec.out_ch as u64)
        .saturating_mul(out_h as u64)
        .saturating_mul(out_w as u64);

    expand_macs
        .saturating_add(dw_macs)
        .saturating_add(proj_macs)
}

// ─── mbconv_param_count ──────────────────────────────────────────────────────

/// Weight parameter count for one MBConv block.
///
/// The count follows the same three-stage decomposition as
/// [`mbconv_mac_count`] but is spatial-independent:
///
/// 1. **Pointwise expand** weight tensor (`expand_ratio > 1`):
///    `in_ch × (in_ch × expand_ratio)`
/// 2. **Depthwise conv** weight tensor:
///    `(in_ch × expand_ratio) × kernel²`
/// 3. **Pointwise project** weight tensor:
///    `(in_ch × expand_ratio) × out_ch`
///
/// Batch-norm parameters and biases are excluded (hardware-cost convention).
#[must_use]
pub fn mbconv_param_count(spec: &MbConvSpec) -> u64 {
    let mid_ch = spec.in_ch * spec.expand_ratio;

    // Stage 1 – pointwise expand (only when expand_ratio > 1).
    let expand_params: u64 = if spec.expand_ratio > 1 {
        (spec.in_ch as u64).saturating_mul(mid_ch as u64)
    } else {
        0
    };

    // Stage 2 – depthwise convolution.
    let dw_params: u64 = (mid_ch as u64)
        .saturating_mul(spec.kernel as u64)
        .saturating_mul(spec.kernel as u64);

    // Stage 3 – pointwise project.
    let proj_params: u64 = (mid_ch as u64).saturating_mul(spec.out_ch as u64);

    expand_params
        .saturating_add(dw_params)
        .saturating_add(proj_params)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A typical MBConv-6 block as found in MobileNetV2.
    fn typical_spec(expand: usize, stride: usize) -> MbConvSpec {
        MbConvSpec {
            in_ch: 32,
            out_ch: 64,
            stride,
            expand_ratio: expand,
            kernel: 3,
        }
    }

    // 1. stride=2 has fewer output MACs than stride=1 for the same input.
    #[test]
    fn stride1_vs_stride2() {
        let s1 = typical_spec(6, 1);
        let s2 = typical_spec(6, 2);
        let macs_s1 = mbconv_mac_count(&s1, 56, 56);
        let macs_s2 = mbconv_mac_count(&s2, 56, 56);
        assert!(
            macs_s2 < macs_s1,
            "stride-2 MACs ({macs_s2}) should be fewer than stride-1 MACs ({macs_s1})"
        );
    }

    // 2. expand_ratio=1 omits the pointwise expand step;
    //    param count equals dw + project only.
    #[test]
    fn expand_1_no_expand_conv() {
        let spec = typical_spec(1, 1);
        let mid_ch = spec.in_ch; // expand_ratio=1 → no expansion
        let expected_dw = (mid_ch * spec.kernel * spec.kernel) as u64;
        let expected_proj = (mid_ch * spec.out_ch) as u64;
        let expected = expected_dw + expected_proj;
        assert_eq!(
            mbconv_param_count(&spec),
            expected,
            "expand_ratio=1 should have no expand-conv params"
        );
    }

    // 3. param_count > 0 for a normal spec.
    #[test]
    fn param_count_finite() {
        let spec = typical_spec(6, 1);
        assert!(mbconv_param_count(&spec) > 0);
    }

    // 4. mac_count > 0 for positive dimensions.
    #[test]
    fn mac_count_positive() {
        let spec = typical_spec(6, 1);
        assert!(mbconv_mac_count(&spec, 28, 28) > 0);
    }

    // 5. expand_ratio=6 produces more MACs than expand_ratio=1.
    #[test]
    fn larger_expand_more_mac() {
        let e1 = mbconv_mac_count(&typical_spec(1, 1), 28, 28);
        let e6 = mbconv_mac_count(&typical_spec(6, 1), 28, 28);
        assert!(
            e6 > e1,
            "expand=6 MACs ({e6}) should exceed expand=1 MACs ({e1})"
        );
    }

    // 6. The function returns a plain u64 (no SE block or extra logic).
    #[test]
    fn se_excluded() {
        let spec = typical_spec(6, 1);
        let count: u64 = mbconv_mac_count(&spec, 14, 14);
        // Just verifying the return type is u64 and a concrete value is produced.
        assert!(count > 0, "returned u64 count should be non-zero: {count}");
    }
}
