use super::codebook::PqCodebook;

/// Encode one vector into m bytes (one byte per subspace = nearest centroid index).
pub fn encode_vector(v: &[f32], cb: &PqCodebook) -> Vec<u8> {
    let mut codes = Vec::with_capacity(cb.m);
    for s in 0..cb.m {
        let sub = &v[s * cb.dsub..(s + 1) * cb.dsub];
        let mut best_c = 0u8;
        let mut best_d = f32::INFINITY;
        for c in 0..cb.ksub {
            let centroid = cb.centroid(s, c);
            let d: f32 = sub
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            if d < best_d {
                best_d = d;
                best_c = c as u8;
            }
        }
        codes.push(best_c);
    }
    codes
}

/// Encode `n` vectors (row-major `[n, dim]`) into `[n, m]` byte matrix.
pub fn encode_batch(data: &[f32], n: usize, cb: &PqCodebook) -> Vec<u8> {
    let dim = cb.m * cb.dsub;
    let mut out = Vec::with_capacity(n * cb.m);
    for i in 0..n {
        let v = &data[i * dim..(i + 1) * dim];
        out.extend(encode_vector(v, cb));
    }
    out
}
