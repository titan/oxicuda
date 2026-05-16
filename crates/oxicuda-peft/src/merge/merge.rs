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
