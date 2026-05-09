//! Latin Hypercube Sampling.

use crate::handle::LcgRng;

/// Generate LHS samples.
///
/// Returns a flat `[n × d]` array where each marginal dimension hits exactly one
/// cell of a uniform partition into `n` equal intervals.
///
/// Algorithm (Fisher-Yates shuffle per dimension):
/// 1. For each dim j: create a random permutation of 0..n.
/// 2. Point `i` in dim `j`: `(perm[j][i] + U[0,1)) / n`.
pub fn latin_hypercube_sample(n: usize, d: usize, rng: &mut LcgRng) -> Vec<f32> {
    let mut out = vec![0.0_f32; n * d];
    let n_inv = 1.0 / n as f32;

    for j in 0..d {
        // Create permutation of 0..n
        let mut perm: Vec<usize> = (0..n).collect();
        rng.shuffle(&mut perm);

        for i in 0..n {
            let u = rng.next_f32();
            out[i * d + j] = (perm[i] as f32 + u) * n_inv;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lhs_output_shape() {
        let mut rng = LcgRng::new(1);
        let samples = latin_hypercube_sample(10, 3, &mut rng);
        assert_eq!(samples.len(), 30);
    }

    #[test]
    fn lhs_in_unit_hypercube() {
        let mut rng = LcgRng::new(2);
        let n = 50;
        let d = 4;
        let samples = latin_hypercube_sample(n, d, &mut rng);
        for &v in &samples {
            assert!((0.0..=1.0).contains(&v), "Sample {v} not in [0,1]");
        }
    }

    #[test]
    fn lhs_marginal_coverage_1d() {
        // Each bin [k/n, (k+1)/n) should be hit exactly once
        let mut rng = LcgRng::new(3);
        let n = 10;
        let samples = latin_hypercube_sample(n, 1, &mut rng);
        let mut hits = vec![false; n];
        for &v in &samples {
            let bin = (v * n as f32).floor() as usize;
            let bin = bin.min(n - 1);
            assert!(!hits[bin], "Bin {bin} hit more than once");
            hits[bin] = true;
        }
        assert!(hits.iter().all(|&h| h), "Not all bins covered");
    }

    #[test]
    fn lhs_marginal_coverage_2d() {
        let mut rng = LcgRng::new(4);
        let n = 20;
        let d = 2;
        let samples = latin_hypercube_sample(n, d, &mut rng);
        for j in 0..d {
            let mut hits = vec![false; n];
            for i in 0..n {
                let v = samples[i * d + j];
                let bin = (v * n as f32).floor() as usize;
                let bin = bin.min(n - 1);
                hits[bin] = true;
            }
            assert!(
                hits.iter().all(|&h| h),
                "Marginal dim {j}: not all bins covered"
            );
        }
    }

    #[test]
    fn lhs_deterministic_with_seed() {
        let mut rng1 = LcgRng::new(42);
        let mut rng2 = LcgRng::new(42);
        let s1 = latin_hypercube_sample(5, 2, &mut rng1);
        let s2 = latin_hypercube_sample(5, 2, &mut rng2);
        assert_eq!(s1, s2);
    }

    #[test]
    fn lhs_n1_returns_one_sample() {
        let mut rng = LcgRng::new(5);
        let s = latin_hypercube_sample(1, 3, &mut rng);
        assert_eq!(s.len(), 3);
        for &v in &s {
            assert!((0.0..=1.0).contains(&v));
        }
    }
}
