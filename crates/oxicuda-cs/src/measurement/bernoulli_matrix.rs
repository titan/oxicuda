//! Bernoulli/Rademacher sensing matrix `Φᵢⱼ ~ ±1/√m`.

use crate::error::{CsError, CsResult};
use crate::handle::LcgRng;

/// Build a random `m × n` Rademacher matrix with entries ±1/√m uniformly.
pub fn bernoulli_matrix(m: usize, n: usize, rng: &mut LcgRng) -> CsResult<Vec<f64>> {
    if m == 0 || n == 0 {
        return Err(CsError::EmptyInput);
    }
    let scale = 1.0 / (m as f64).sqrt();
    let mut a = vec![0.0_f64; m * n];
    for v in a.iter_mut() {
        *v = if rng.next_bool() { scale } else { -scale };
    }
    Ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bernoulli_matrix_entries() {
        let mut rng = LcgRng::new(31);
        let m = 8;
        let a = bernoulli_matrix(m, 4, &mut rng).expect("ok");
        let scale = 1.0 / (m as f64).sqrt();
        for v in &a {
            assert!((v.abs() - scale).abs() < 1.0e-12);
        }
    }
}
