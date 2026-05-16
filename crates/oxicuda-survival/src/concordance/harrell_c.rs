//! Harrell's C-index.

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};

/// Harrell C-index: fraction of comparable pairs (i, j) where higher predicted risk η
/// goes with shorter observed time. Ties in η contribute 0.5.
pub fn harrell_c_index(data: &Dataset, eta: &[f64]) -> SurvivalResult<f64> {
    if data.len() != eta.len() {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![data.len()],
            got: vec![eta.len()],
        });
    }
    let n = data.len();
    let mut concordant = 0.0_f64;
    let mut comparable = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let ti = data.observations[i].time;
            let tj = data.observations[j].time;
            let ei = data.observations[i].event;
            // pair is comparable if observation i is uncensored and ti < tj
            if !ei || ti >= tj {
                continue;
            }
            comparable += 1.0;
            if eta[i] > eta[j] {
                concordant += 1.0;
            } else if (eta[i] - eta[j]).abs() < 1.0e-12 {
                concordant += 0.5;
            }
        }
    }
    if comparable == 0.0 {
        return Err(SurvivalError::NumericalInstability(
            "no comparable pairs".to_string(),
        ));
    }
    Ok(concordant / comparable)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_index_perfect_ranking_one() {
        // η aligned with reverse time: shortest time has highest risk
        let d = Dataset::from_arrays(&[1.0, 2.0, 3.0, 4.0], &[true, true, true, true]).expect("ok");
        let eta = vec![3.0, 2.0, 1.0, 0.0];
        let c = harrell_c_index(&d, &eta).expect("ok");
        assert!((c - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn c_index_reverse_ranking_zero() {
        let d = Dataset::from_arrays(&[1.0, 2.0, 3.0, 4.0], &[true, true, true, true]).expect("ok");
        let eta = vec![0.0, 1.0, 2.0, 3.0];
        let c = harrell_c_index(&d, &eta).expect("ok");
        assert!(c < 0.05);
    }

    #[test]
    fn c_index_all_tied_eta_half() {
        let d = Dataset::from_arrays(&[1.0, 2.0, 3.0, 4.0], &[true, true, true, true]).expect("ok");
        let eta = vec![1.0, 1.0, 1.0, 1.0];
        let c = harrell_c_index(&d, &eta).expect("ok");
        assert!((c - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn c_index_rejects_size_mismatch() {
        let d = Dataset::from_arrays(&[1.0, 2.0], &[true, true]).expect("ok");
        assert!(harrell_c_index(&d, &[1.0]).is_err());
    }
}
