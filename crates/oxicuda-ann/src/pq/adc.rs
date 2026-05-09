use super::codebook::PqCodebook;

/// Build the asymmetric distance computation table `[m × ksub]`.
///
/// `table[s * ksub + c]` = L2² from query sub-vector `s` to centroid `(s, c)`.
pub fn build_adc_table(query: &[f32], cb: &PqCodebook) -> Vec<f32> {
    let mut table = vec![0.0_f32; cb.m * cb.ksub];
    for s in 0..cb.m {
        let q_sub = &query[s * cb.dsub..(s + 1) * cb.dsub];
        for c in 0..cb.ksub {
            let centroid = cb.centroid(s, c);
            let d: f32 = q_sub
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            table[s * cb.ksub + c] = d;
        }
    }
    table
}

/// Compute approximate L2² distance from pre-built ADC table and PQ codes.
pub fn adc_distance(codes: &[u8], table: &[f32], m: usize, ksub: usize) -> f32 {
    let mut dist = 0.0_f32;
    for (s, &c) in codes.iter().take(m).enumerate() {
        dist += table[s * ksub + c as usize];
    }
    dist
}
