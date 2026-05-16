//! TDA summary statistics: Betti numbers, persistent entropy, persistence landscape,
//! total persistence, and connected component count.

use crate::error::{TdaError, TdaResult};
use crate::persistence::diagram::PersistenceDiagram;

/// Compute Betti numbers from a slice of persistence diagrams.
///
/// `β_d` = number of pairs in diagram `d` with persistence > `threshold`
/// plus the number of essential (infinite-persistence) classes.
pub fn betti_numbers(diagrams: &[PersistenceDiagram], threshold: f64) -> Vec<usize> {
    diagrams
        .iter()
        .map(|diag| {
            let finite = diag
                .finite_pairs()
                .iter()
                .filter(|p| p.death.unwrap_or(0.0) - p.birth > threshold)
                .count();
            let essential = diag.essential_classes().len();
            finite + essential
        })
        .collect()
}

/// Persistent entropy of a persistence diagram.
///
/// `H = -Σ_i (l_i / L) * log(l_i / L)`
///
/// where `l_i = death_i - birth_i` are the lifetimes of **finite** pairs and
/// `L = Σ l_i`.  Returns `TdaError::EmptyComplex` if there are no finite pairs with
/// positive lifetime.
pub fn persistent_entropy(diagram: &PersistenceDiagram) -> TdaResult<f64> {
    let lifetimes: Vec<f64> = diagram
        .finite_pairs()
        .iter()
        .map(|p| p.death.unwrap_or(0.0) - p.birth)
        .filter(|&l| l > 0.0)
        .collect();

    if lifetimes.is_empty() {
        return Err(TdaError::EmptyComplex);
    }

    let total: f64 = lifetimes.iter().sum();
    if total <= 0.0 {
        return Err(TdaError::EmptyComplex);
    }

    let entropy = lifetimes
        .iter()
        .map(|&l| {
            let p = l / total;
            -p * p.ln()
        })
        .sum::<f64>();

    Ok(entropy)
}

/// Persistence landscape function L_k(t).
///
/// The k-th persistence landscape (1-indexed) at point `t` is the k-th largest value of
/// `tent_k(t) = max(0, min(t - birth, death - t))` over all finite pairs `(birth, death)`.
///
/// Essential classes are treated as half-infinite triangles truncated at the maximum finite
/// death value.
pub fn persistence_landscape(
    diagram: &PersistenceDiagram,
    k: usize,
    t_values: &[f64],
) -> TdaResult<Vec<f64>> {
    if k == 0 {
        return Err(TdaError::ParameterOutOfRange(
            "k must be ≥ 1 (1-indexed)".to_owned(),
        ));
    }

    let max_death = diagram
        .finite_pairs()
        .iter()
        .filter_map(|p| p.death)
        .fold(0.0_f64, f64::max);

    // Collect pairs (birth, death) for tent functions
    let pairs: Vec<(f64, f64)> = diagram
        .pairs
        .iter()
        .filter(|p| p.birth < p.death.unwrap_or(max_death))
        .map(|p| (p.birth, p.death.unwrap_or(max_death)))
        .collect();

    let result = t_values
        .iter()
        .map(|&t| {
            // Compute tent value for each pair
            let mut vals: Vec<f64> = pairs
                .iter()
                .map(|&(b, d)| {
                    let v = (t - b).min(d - t);
                    v.max(0.0)
                })
                .collect();
            // Sort descending, take k-th (0-indexed: k-1)
            vals.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            vals.get(k - 1).copied().unwrap_or(0.0)
        })
        .collect();

    Ok(result)
}

/// L2 distance between two persistence landscapes sampled on a uniform grid.
///
/// `||L1 - L2||_2 = sqrt(dt * Σ (L1[i] - L2[i])^2)`
pub fn landscape_distance(l1: &[f64], l2: &[f64], dt: f64) -> TdaResult<f64> {
    if l1.len() != l2.len() {
        return Err(TdaError::DimensionMismatch {
            expected: l1.len(),
            got: l2.len(),
        });
    }
    if dt <= 0.0 {
        return Err(TdaError::ParameterOutOfRange(
            "dt must be positive".to_owned(),
        ));
    }
    let sq_sum: f64 = l1
        .iter()
        .zip(l2.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum();
    Ok((sq_sum * dt).sqrt())
}

/// Total persistence (sum of p-th powers of lifetimes) for a diagram.
///
/// `T_p = Σ (death_i - birth_i)^p` over all finite pairs.
pub fn total_persistence(diagram: &PersistenceDiagram, p: f64) -> f64 {
    diagram
        .finite_pairs()
        .iter()
        .map(|pair| {
            let lifetime = pair.death.unwrap_or(0.0) - pair.birth;
            lifetime.powf(p)
        })
        .sum()
}

/// Number of connected components (β₀) from the 0-dimensional persistence diagram.
///
/// β₀ = number of essential (infinite-persistence) classes in dimension 0,
/// which corresponds to the number of connected components of the underlying space.
pub fn count_components(dim0_diagram: &PersistenceDiagram) -> usize {
    dim0_diagram.essential_classes().len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::homology::persistent::PersistencePair;
    use crate::persistence::diagram::PersistenceDiagram;

    fn make_diag(pairs: &[(f64, f64)]) -> PersistenceDiagram {
        let ps = pairs
            .iter()
            .map(|&(b, d)| PersistencePair {
                dim: 0,
                birth: b,
                death: Some(d),
            })
            .collect();
        PersistenceDiagram::new(ps, 0)
    }

    #[test]
    fn entropy_positive() {
        let d = make_diag(&[(0.0, 1.0), (0.5, 2.0)]);
        let h = persistent_entropy(&d).expect("ok");
        assert!(h >= 0.0);
    }

    #[test]
    fn landscape_midpoint_positive() {
        // Single pair (0, 2): tent function peaks at t=1 with value 1
        let d = make_diag(&[(0.0, 2.0)]);
        let vals = persistence_landscape(&d, 1, &[1.0]).expect("ok");
        assert!((vals[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn betti_threshold() {
        let d = make_diag(&[(0.0, 0.1), (0.0, 5.0)]);
        let diags = vec![d];
        let betti = betti_numbers(&diags, 1.0);
        assert_eq!(betti[0], 1); // only the 5-unit bar survives threshold=1
    }
}
