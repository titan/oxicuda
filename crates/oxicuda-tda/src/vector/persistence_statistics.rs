//! Persistence statistics: a fixed-length scalar-feature summary of a diagram.
//!
//! Many machine-learning pipelines need a *small, fixed-length* feature vector
//! from a persistence diagram rather than a full functional summary (landscape,
//! image, Betti curve). **Persistence statistics** are the standard such
//! representation: a handful of order-statistics and moments computed over the
//! diagram's births, deaths, persistences (`d − b`), and midpoints
//! (`(b + d) / 2`). They are permutation-invariant (each is a symmetric function
//! of the points) and cheap to compute, which makes them a strong baseline
//! feature set for classification and regression on topological signatures.
//!
//! The [`PersistenceStatistics`] vector reports, over the multiset of finite
//! persistence values `{λᵢ = dᵢ − bᵢ}` and midpoints `{mᵢ}`:
//!
//! | field | definition |
//! |-------|------------|
//! | `n_points`            | number of finite pairs |
//! | `total_persistence`   | `Σ λᵢ` |
//! | `max_persistence`     | `max λᵢ` |
//! | `mean_persistence`    | `(1/n) Σ λᵢ` |
//! | `std_persistence`     | population standard deviation of `λ` |
//! | `median_persistence`  | median of `λ` |
//! | `persistent_entropy`  | `−Σ pᵢ ln pᵢ`, `pᵢ = λᵢ / Σλ` |
//! | `mean_midpoint`       | `(1/n) Σ mᵢ` |
//! | `min_birth`           | `min bᵢ` |
//! | `max_death`           | `max dᵢ` |
//!
//! Persistent entropy follows Rucco et al. (2016); the remaining moments are the
//! conventional persistence-statistics features used throughout applied TDA
//! (e.g. Atienza et al. 2020, "On the stability of persistent entropy and new
//! summary functions for topological data analysis").
//!
//! Reference: N. Atienza, R. González-Díaz, M. Soriano-Trigueros, "On the
//! stability of persistent entropy and new summary functions for TDA", Pattern
//! Recognition, 2020.

use crate::error::{TdaError, TdaResult};
use crate::persistence::diagram::PersistenceDiagram;

/// A fixed-length statistical summary of a persistence diagram.
///
/// All fields are `0` for an empty diagram, so the [`PersistenceStatistics::to_vec`]
/// representation always has the same length (10) regardless of diagram size.
#[derive(Debug, Clone, PartialEq)]
pub struct PersistenceStatistics {
    /// Number of finite persistence pairs.
    pub n_points: usize,
    /// Sum of persistences `Σ (dᵢ − bᵢ)`.
    pub total_persistence: f64,
    /// Largest persistence.
    pub max_persistence: f64,
    /// Mean persistence.
    pub mean_persistence: f64,
    /// Population standard deviation of the persistences.
    pub std_persistence: f64,
    /// Median persistence.
    pub median_persistence: f64,
    /// Persistent entropy `−Σ pᵢ ln pᵢ` with `pᵢ = λᵢ / Σλ`.
    pub persistent_entropy: f64,
    /// Mean of the point midpoints `(bᵢ + dᵢ) / 2`.
    pub mean_midpoint: f64,
    /// Smallest birth value.
    pub min_birth: f64,
    /// Largest death value.
    pub max_death: f64,
}

impl PersistenceStatistics {
    /// Flatten the statistics into a fixed-length feature vector (length 10) in
    /// the field order documented on the struct.
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.n_points as f64,
            self.total_persistence,
            self.max_persistence,
            self.mean_persistence,
            self.std_persistence,
            self.median_persistence,
            self.persistent_entropy,
            self.mean_midpoint,
            self.min_birth,
            self.max_death,
        ]
    }

    /// The all-zero statistics used for an empty diagram.
    fn empty() -> Self {
        Self {
            n_points: 0,
            total_persistence: 0.0,
            max_persistence: 0.0,
            mean_persistence: 0.0,
            std_persistence: 0.0,
            median_persistence: 0.0,
            persistent_entropy: 0.0,
            mean_midpoint: 0.0,
            min_birth: 0.0,
            max_death: 0.0,
        }
    }
}

/// Compute the [`PersistenceStatistics`] of a diagram over its **finite** pairs.
///
/// Essential (infinite-persistence) classes are ignored. The returned struct
/// always has the documented fields; an empty diagram yields all zeros.
///
/// # Errors
/// [`TdaError::NanFiltrationValue`] when a finite pair has a NaN birth or death.
pub fn persistence_statistics(diagram: &PersistenceDiagram) -> TdaResult<PersistenceStatistics> {
    // Gather finite (birth, death, persistence, midpoint).
    let mut births = Vec::new();
    let mut deaths = Vec::new();
    let mut pers = Vec::new();
    let mut mids = Vec::new();
    for p in &diagram.pairs {
        if let Some(d) = p.death {
            if p.birth.is_nan() || d.is_nan() {
                return Err(TdaError::NanFiltrationValue);
            }
            let lam = (d - p.birth).max(0.0);
            births.push(p.birth);
            deaths.push(d);
            pers.push(lam);
            mids.push(0.5 * (p.birth + d));
        }
    }

    let n = pers.len();
    if n == 0 {
        return Ok(PersistenceStatistics::empty());
    }
    let n_f = n as f64;

    let total: f64 = pers.iter().sum();
    let max = pers.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = total / n_f;
    let var = pers.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n_f;
    let std = var.sqrt();

    // Median (sorted copy).
    let mut sorted = pers.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    };

    // Persistent entropy (Rucco 2016): −Σ pᵢ ln pᵢ, pᵢ = λᵢ / Σλ.
    let entropy = if total > 0.0 {
        let mut h = 0.0;
        for &lam in &pers {
            if lam > 0.0 {
                let p = lam / total;
                h -= p * p.ln();
            }
        }
        h
    } else {
        0.0
    };

    let mean_midpoint = mids.iter().sum::<f64>() / n_f;
    let min_birth = births.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_death = deaths.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    Ok(PersistenceStatistics {
        n_points: n,
        total_persistence: total,
        max_persistence: max,
        mean_persistence: mean,
        std_persistence: std,
        median_persistence: median,
        persistent_entropy: entropy,
        mean_midpoint,
        min_birth,
        max_death,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::homology::persistent::PersistencePair;

    fn diagram(points: &[(f64, Option<f64>)]) -> PersistenceDiagram {
        let pairs = points
            .iter()
            .map(|&(b, d)| PersistencePair {
                dim: 1,
                birth: b,
                death: d,
            })
            .collect();
        PersistenceDiagram::new(pairs, 1)
    }

    // 1. The flat vector always has length 10.
    #[test]
    fn output_shape() {
        let d = diagram(&[(0.0, Some(3.0)), (1.0, Some(2.0))]);
        let s = persistence_statistics(&d).expect("persistence_statistics should succeed");
        assert_eq!(s.to_vec().len(), 10);
    }

    // 2. An empty diagram yields all-zero statistics.
    #[test]
    fn empty_diagram_zero() {
        let d = diagram(&[]);
        let s = persistence_statistics(&d).expect("persistence_statistics should succeed");
        assert_eq!(s, PersistenceStatistics::empty());
        for &x in &s.to_vec() {
            assert_eq!(x, 0.0);
        }
    }

    // 3. n_points counts only finite pairs.
    #[test]
    fn counts_finite_only() {
        let d = diagram(&[(0.0, Some(3.0)), (0.0, None), (1.0, Some(2.0))]);
        let s = persistence_statistics(&d).expect("persistence_statistics should succeed");
        assert_eq!(s.n_points, 2);
    }

    // 4. Total / max / mean persistence are correct.
    #[test]
    fn total_max_mean() {
        // Persistences: 3, 6, 1 → total 10, max 6, mean 10/3.
        let d = diagram(&[(0.0, Some(3.0)), (2.0, Some(8.0)), (1.0, Some(2.0))]);
        let s = persistence_statistics(&d).expect("persistence_statistics should succeed");
        assert!(
            (s.total_persistence - 10.0).abs() < 1e-12,
            "{}",
            s.total_persistence
        );
        assert!(
            (s.max_persistence - 6.0).abs() < 1e-12,
            "{}",
            s.max_persistence
        );
        assert!(
            (s.mean_persistence - 10.0 / 3.0).abs() < 1e-12,
            "{}",
            s.mean_persistence
        );
    }

    // 5. Standard deviation is zero for identical persistences and positive
    //    otherwise.
    #[test]
    fn std_behaviour() {
        // All three pairs have persistence 1 (identical λ ⇒ zero variance).
        let same = diagram(&[(0.0, Some(1.0)), (2.0, Some(3.0)), (5.0, Some(6.0))]);
        let s = persistence_statistics(&same).expect("persistence_statistics should succeed");
        assert!(s.std_persistence.abs() < 1e-12, "std {}", s.std_persistence);
        let varied = diagram(&[(0.0, Some(1.0)), (0.0, Some(5.0))]);
        let v = persistence_statistics(&varied).expect("persistence_statistics should succeed");
        assert!(v.std_persistence > 0.0, "std should be > 0");
    }

    // 6. Median is the middle order statistic (odd count) and the mean of the two
    //    middle values (even count).
    #[test]
    fn median_correct() {
        let odd = diagram(&[(0.0, Some(1.0)), (0.0, Some(5.0)), (0.0, Some(3.0))]);
        let so = persistence_statistics(&odd).expect("persistence_statistics should succeed");
        assert!(
            (so.median_persistence - 3.0).abs() < 1e-12,
            "{}",
            so.median_persistence
        );
        let even = diagram(&[
            (0.0, Some(1.0)),
            (0.0, Some(3.0)),
            (0.0, Some(5.0)),
            (0.0, Some(9.0)),
        ]);
        let se = persistence_statistics(&even).expect("persistence_statistics should succeed");
        assert!(
            (se.median_persistence - 4.0).abs() < 1e-12,
            "{}",
            se.median_persistence
        );
    }

    // 7. Persistent entropy is non-negative and maximised by equal persistences.
    #[test]
    fn entropy_nonneg_and_max_uniform() {
        let equal = diagram(&[(0.0, Some(2.0)), (0.0, Some(2.0)), (0.0, Some(2.0))]);
        let se = persistence_statistics(&equal).expect("persistence_statistics should succeed");
        // Equal pᵢ = 1/3 → entropy = ln 3.
        assert!(
            (se.persistent_entropy - 3f64.ln()).abs() < 1e-12,
            "{}",
            se.persistent_entropy
        );
        let skewed = diagram(&[(0.0, Some(0.01)), (0.0, Some(10.0)), (0.0, Some(0.01))]);
        let ss = persistence_statistics(&skewed).expect("persistence_statistics should succeed");
        assert!(
            ss.persistent_entropy >= 0.0,
            "entropy {}",
            ss.persistent_entropy
        );
        assert!(
            ss.persistent_entropy < se.persistent_entropy,
            "skewed entropy should be lower"
        );
    }

    // 8. min_birth and max_death track the diagram extremes.
    #[test]
    fn birth_death_extremes() {
        let d = diagram(&[(2.0, Some(5.0)), (-1.0, Some(3.0)), (0.0, Some(9.0))]);
        let s = persistence_statistics(&d).expect("persistence_statistics should succeed");
        assert!((s.min_birth - (-1.0)).abs() < 1e-12, "{}", s.min_birth);
        assert!((s.max_death - 9.0).abs() < 1e-12, "{}", s.max_death);
    }

    // 9. NaN filtration value errors.
    #[test]
    fn nan_value_error() {
        let d = diagram(&[(0.0, Some(f64::NAN))]);
        let err = persistence_statistics(&d);
        assert!(
            matches!(err, Err(TdaError::NanFiltrationValue)),
            "got {err:?}"
        );
    }

    // 10. Permutation invariance of the whole statistics vector.
    #[test]
    fn permutation_invariant() {
        let a = diagram(&[(0.0, Some(3.0)), (1.0, Some(7.0)), (2.0, Some(4.0))]);
        let b = diagram(&[(2.0, Some(4.0)), (0.0, Some(3.0)), (1.0, Some(7.0))]);
        let va = persistence_statistics(&a)
            .expect("persistence_statistics should succeed")
            .to_vec();
        let vb = persistence_statistics(&b)
            .expect("persistence_statistics should succeed")
            .to_vec();
        for (x, y) in va.iter().zip(&vb) {
            assert!((x - y).abs() < 1e-12, "{x} vs {y}");
        }
    }

    // 11. mean_midpoint equals the average of (b+d)/2.
    #[test]
    fn mean_midpoint_correct() {
        // Midpoints: (0+4)/2 = 2, (2+6)/2 = 4 → mean 3.
        let d = diagram(&[(0.0, Some(4.0)), (2.0, Some(6.0))]);
        let s = persistence_statistics(&d).expect("persistence_statistics should succeed");
        assert!((s.mean_midpoint - 3.0).abs() < 1e-12, "{}", s.mean_midpoint);
    }
}
