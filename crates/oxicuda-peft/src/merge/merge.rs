use crate::lora::lora::LoraLinear;

/// Compute the weighted sum of LoRA delta matrices across multiple adapters.
///
/// Each adapter's `lora_delta()` is weighted by the corresponding entry in `weights`.
/// Returns the merged delta as a flat `[out_features × in_features]` matrix.
/// All adapters must have identical `in_features` and `out_features`.
#[must_use]
pub fn merge_loras(loras: &[&LoraLinear], weights: &[f32]) -> Vec<f32> {
    if loras.is_empty() {
        return Vec::new();
    }
    let out_features = loras[0].out_features;
    let in_features = loras[0].in_features;
    let n = out_features * in_features;
    let mut result = vec![0.0_f32; n];
    for (&w, lora) in weights.iter().zip(loras.iter()) {
        let delta = lora.lora_delta();
        for (r, d) in result.iter_mut().zip(delta.iter()) {
            *r += w * d;
        }
    }
    result
}

/// Add a weighted combination of deltas to a base weight matrix.
///
/// Returns `base_w + Σ_i weight_i * delta_i`. All vectors must have the same length.
#[must_use]
pub fn linear_merge(base_w: &[f32], deltas: &[(Vec<f32>, f32)]) -> Vec<f32> {
    let mut result = base_w.to_vec();
    for (delta, weight) in deltas {
        for (r, d) in result.iter_mut().zip(delta.iter()) {
            *r += weight * d;
        }
    }
    result
}

/// TIES (Trim, Elect Sign, Disjoint Merge) model merging.
///
/// 1. **Trim**: keep only the top `density` fraction of elements by absolute magnitude in each delta.
/// 2. **Elect**: resolve sign conflicts by majority vote (sum of trimmed deltas → take sign of sum).
/// 3. **Merge**: average only the deltas that agree with the elected sign.
///
/// Returns the merged weight vector `base_w + merged_delta`.
#[must_use]
pub fn ties_merge(base_w: &[f32], deltas: &[(Vec<f32>, f32)], density: f32) -> Vec<f32> {
    if deltas.is_empty() || base_w.is_empty() {
        return base_w.to_vec();
    }
    let n = base_w.len();
    let density = density.clamp(0.0, 1.0);

    // Step 1: Trim each delta to keep only top-density elements by magnitude.
    let trimmed: Vec<Vec<f32>> = deltas
        .iter()
        .map(|(delta, _w)| trim_by_density(delta, density))
        .collect();

    // Step 2: Elect sign per element — sign of the sum of all trimmed deltas.
    let mut sign_sum = vec![0.0_f32; n];
    for trimmed_delta in &trimmed {
        for (s, &v) in sign_sum.iter_mut().zip(trimmed_delta.iter()) {
            *s += v;
        }
    }
    // sign_sum[i] > 0 → elected sign +1; < 0 → -1; == 0 → 0

    // Step 3: For each adapter, keep only elements that agree with the elected sign.
    // Then average only contributors.
    let mut merged_delta = vec![0.0_f32; n];
    let mut counts = vec![0u32; n];
    for (trimmed_delta, (_, w)) in trimmed.iter().zip(deltas.iter()) {
        for i in 0..n {
            let v = trimmed_delta[i];
            let elected = sign_sum[i];
            // Only include if same sign or elected sign is zero
            let agrees =
                (elected == 0.0) || (elected > 0.0 && v > 0.0) || (elected < 0.0 && v < 0.0);
            if agrees && v != 0.0 {
                merged_delta[i] += w * v;
                counts[i] += 1;
            }
        }
    }
    // Normalise by count of contributors
    for i in 0..n {
        if counts[i] > 1 {
            merged_delta[i] /= counts[i] as f32;
        }
    }

    // Add to base
    base_w
        .iter()
        .zip(merged_delta.iter())
        .map(|(b, d)| b + d)
        .collect()
}

/// Keep only the top `density` fraction of elements by absolute value; zero out the rest.
fn trim_by_density(delta: &[f32], density: f32) -> Vec<f32> {
    let n = delta.len();
    if n == 0 {
        return Vec::new();
    }
    let keep = ((n as f32 * density).ceil() as usize).min(n);
    // Collect (|v|, idx) to find threshold
    let mut magnitudes: Vec<(f32, usize)> = delta
        .iter()
        .enumerate()
        .map(|(i, &v)| (v.abs(), i))
        .collect();
    magnitudes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut mask = vec![false; n];
    for &(_, idx) in magnitudes.iter().take(keep) {
        mask[idx] = true;
    }
    delta
        .iter()
        .enumerate()
        .map(|(i, &v)| if mask[i] { v } else { 0.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::lora::lora::{LoraConfig, LoraLinear};

    /// Build a `LoraLinear` with caller-supplied A and B matrices, overwriting the
    /// randomly-initialised ones.  `scale` is the effective LoRA scale (α / r).
    fn make_lora_with_ab(
        in_features: usize,
        out_features: usize,
        rank: usize,
        scale: f32,
        a_vals: Vec<f32>,
        b_vals: Vec<f32>,
    ) -> LoraLinear {
        let mut rng = LcgRng::new(0);
        let cfg = LoraConfig {
            r: rank,
            alpha: scale * rank as f32,
            init_scale: 0.01,
        };
        let mut lora = LoraLinear::new(in_features, out_features, &cfg, &mut rng);
        lora.a = a_vals;
        lora.b = b_vals;
        lora
    }

    // ──── linear_merge ─────────────────────────────────────────────────────

    #[test]
    fn linear_merge_no_deltas_returns_base() {
        let base = [1.0_f32, 2.0, 3.0];
        let result = linear_merge(&base, &[]);
        assert_eq!(
            result.as_slice(),
            &base,
            "no deltas: result must equal base"
        );
    }

    #[test]
    fn linear_merge_single_delta_analytic() {
        // base=[1,2], delta=([1,-1], w=2.0) → [1+2·1, 2+2·(-1)] = [3, 0]
        let base = [1.0_f32, 2.0];
        let result = linear_merge(&base, &[(vec![1.0_f32, -1.0], 2.0)]);
        assert!(
            (result[0] - 3.0).abs() < 1e-6,
            "expected 3.0, got {}",
            result[0]
        );
        assert!((result[1]).abs() < 1e-6, "expected 0.0, got {}", result[1]);
    }

    #[test]
    fn linear_merge_two_deltas_analytic() {
        // base=[0,0], d1=([1,2], 1.0), d2=([-1,3], 2.0)
        // result = [0 + 1·1 + 2·(-1), 0 + 1·2 + 2·3] = [-1, 8]
        let base = [0.0_f32, 0.0];
        let result = linear_merge(
            &base,
            &[(vec![1.0_f32, 2.0], 1.0), (vec![-1.0_f32, 3.0], 2.0)],
        );
        assert!(
            (result[0] - (-1.0)).abs() < 1e-6,
            "expected -1.0, got {}",
            result[0]
        );
        assert!(
            (result[1] - 8.0).abs() < 1e-6,
            "expected 8.0, got {}",
            result[1]
        );
    }

    // ──── merge_loras ──────────────────────────────────────────────────────

    #[test]
    fn merge_loras_empty_returns_empty() {
        let result = merge_loras(&[], &[]);
        assert!(result.is_empty(), "empty adapters must return empty result");
    }

    #[test]
    fn merge_loras_single_adapter_analytic() {
        // rank=1, in=2, out=2, scale=1.0
        // A=[1.0, 0.0] (shape [1×2]), B=[2.0, 0.0] (shape [2×1])
        // lora_delta[i,j] = scale · B[i,0] · A[0,j]:
        //   [0,0] = 1.0·2.0·1.0 = 2.0
        //   [0,1] = 1.0·2.0·0.0 = 0.0
        //   [1,0] = 1.0·0.0·1.0 = 0.0
        //   [1,1] = 1.0·0.0·0.0 = 0.0
        // merge_loras([lora],[1.0]) = 1.0·[2,0,0,0] = [2,0,0,0]
        let lora = make_lora_with_ab(2, 2, 1, 1.0, vec![1.0_f32, 0.0], vec![2.0_f32, 0.0]);
        let result = merge_loras(&[&lora], &[1.0]);
        assert_eq!(result.len(), 4, "out×in = 4 elements");
        assert!(
            (result[0] - 2.0).abs() < 1e-6,
            "result[0,0] expected 2.0, got {}",
            result[0]
        );
        assert!(
            result[1].abs() < 1e-6,
            "result[0,1] expected 0.0, got {}",
            result[1]
        );
        assert!(
            result[2].abs() < 1e-6,
            "result[1,0] expected 0.0, got {}",
            result[2]
        );
        assert!(
            result[3].abs() < 1e-6,
            "result[1,1] expected 0.0, got {}",
            result[3]
        );
    }

    #[test]
    fn merge_loras_two_adapters_weighted() {
        // rank=1, in=1, out=1, scale=1.0
        // adapter_a: A=[1.0], B=[1.0] → delta=[1.0]
        // adapter_b: A=[2.0], B=[1.0] → delta=[2.0]
        // merge_loras([a,b], [0.5, 0.25]) = 0.5·1.0 + 0.25·2.0 = 1.0
        let a = make_lora_with_ab(1, 1, 1, 1.0, vec![1.0_f32], vec![1.0_f32]);
        let b = make_lora_with_ab(1, 1, 1, 1.0, vec![2.0_f32], vec![1.0_f32]);
        let result = merge_loras(&[&a, &b], &[0.5, 0.25]);
        assert_eq!(result.len(), 1, "out×in = 1 element");
        assert!(
            (result[0] - 1.0).abs() < 1e-6,
            "expected 1.0, got {}",
            result[0]
        );
    }

    // ──── ties_merge ───────────────────────────────────────────────────────

    #[test]
    fn ties_merge_empty_base_returns_empty() {
        let result = ties_merge(&[], &[], 1.0);
        assert!(result.is_empty(), "empty base must produce empty result");
    }

    #[test]
    fn ties_merge_no_sign_conflict_averages_contributors() {
        // base=[0,0], deltas: ([2,0], 1.0) and ([4,0], 1.0), density=1.0
        // trim: both trimmed as-is (keep all 2 elements)
        // sign_sum = [6, 0] → elected[0]=+1, elected[1]=0
        // Both adapters agree at pos 0 (both positive); pos 1 value=0 → skipped.
        // merged[0] = (1.0·2 + 1.0·4) / 2 = 3.0  (2 contributors → divide by 2)
        // result = [0+3, 0+0] = [3.0, 0.0]
        let base = [0.0_f32, 0.0];
        let result = ties_merge(
            &base,
            &[(vec![2.0_f32, 0.0], 1.0), (vec![4.0_f32, 0.0], 1.0)],
            1.0,
        );
        assert!(
            (result[0] - 3.0).abs() < 1e-6,
            "expected 3.0, got {}",
            result[0]
        );
        assert!(result[1].abs() < 1e-6, "expected 0.0, got {}", result[1]);
    }

    #[test]
    fn ties_merge_sign_conflict_rejects_minority_adapter() {
        // base=[0.0], d_pos=[3.0], d_neg=[-1.0], density=1.0
        // sign_sum = [3+(-1)] = [2] > 0 → elected sign = +1
        // d_pos agrees (+), d_neg disagrees (−) → only d_pos contributes
        // merged[0] = 1.0·3.0 (count=1, no division)
        // result = [0 + 3.0] = [3.0]
        let base = [0.0_f32];
        let result = ties_merge(&base, &[(vec![3.0_f32], 1.0), (vec![-1.0_f32], 1.0)], 1.0);
        assert!(
            (result[0] - 3.0).abs() < 1e-6,
            "minority-sign adapter must be excluded; expected 3.0, got {}",
            result[0]
        );
    }

    #[test]
    fn ties_merge_density_trims_low_magnitude_elements() {
        // delta=[1.0, 5.0, 3.0, 0.5], density=0.5
        // keep = ceil(4·0.5) = 2
        // magnitudes sorted desc: 5.0(idx1), 3.0(idx2), 1.0(idx0), 0.5(idx3)
        // Keep indices 1,2; zero out 0,3 → trimmed=[0.0, 5.0, 3.0, 0.0]
        // sign_sum=[0, 5, 3, 0]: elected[0]=0, elected[1]=+1, elected[2]=+1, elected[3]=0
        // single adapter, w=1.0:
        //   pos0: v=0 → skipped; pos1: v=5>0 agrees → merged[1]=5, cnt[1]=1
        //   pos2: v=3>0 agrees → merged[2]=3, cnt[2]=1; pos3: v=0 → skipped
        // no cnt>1, no division
        // result = [10+0, 10+5, 10+3, 10+0] = [10, 15, 13, 10]
        let base = [10.0_f32, 10.0, 10.0, 10.0];
        let result = ties_merge(&base, &[(vec![1.0_f32, 5.0, 3.0, 0.5], 1.0)], 0.5);
        assert!(
            (result[0] - 10.0).abs() < 1e-6,
            "pos0 trimmed→0, expected 10.0, got {}",
            result[0]
        );
        assert!(
            (result[1] - 15.0).abs() < 1e-6,
            "pos1 kept→15.0, got {}",
            result[1]
        );
        assert!(
            (result[2] - 13.0).abs() < 1e-6,
            "pos2 kept→13.0, got {}",
            result[2]
        );
        assert!(
            (result[3] - 10.0).abs() < 1e-6,
            "pos3 trimmed→0, expected 10.0, got {}",
            result[3]
        );
    }
}
