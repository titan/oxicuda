//! Persistence diagram: birth-death pairs grouped by homological dimension.

use crate::homology::persistent::PersistencePair;

/// A persistence diagram for a single homological dimension.
///
/// Contains all persistence pairs (finite and essential) for dimension `dimension`.
#[derive(Debug, Clone)]
pub struct PersistenceDiagram {
    pub pairs: Vec<PersistencePair>,
    /// Which H_d this diagram represents.
    pub dimension: usize,
}

impl PersistenceDiagram {
    /// Construct a diagram from a list of persistence pairs and a dimension tag.
    pub fn new(pairs: Vec<PersistencePair>, dimension: usize) -> Self {
        Self { pairs, dimension }
    }

    /// All finite pairs (cycles that are killed within the filtration).
    pub fn finite_pairs(&self) -> Vec<&PersistencePair> {
        self.pairs.iter().filter(|p| p.death.is_some()).collect()
    }

    /// All essential (infinite-persistence) classes.
    pub fn essential_classes(&self) -> Vec<&PersistencePair> {
        self.pairs.iter().filter(|p| p.death.is_none()).collect()
    }

    /// Maximum finite persistence value in this diagram.  Returns 0 if no finite pairs.
    pub fn max_persistence(&self) -> f64 {
        self.pairs
            .iter()
            .filter_map(|p| p.death.map(|d| d - p.birth))
            .fold(0.0_f64, f64::max)
    }

    /// Partition a flat `Vec<PersistencePair>` into one `PersistenceDiagram` per dimension.
    ///
    /// Returns a `Vec` of length `max_dim + 1` where index `d` contains all pairs with
    /// `dim == d`.
    pub fn from_pairs_by_dim(all_pairs: &[PersistencePair], max_dim: usize) -> Vec<Self> {
        let mut diagrams: Vec<Vec<PersistencePair>> = vec![Vec::new(); max_dim + 1];
        for pair in all_pairs {
            if pair.dim <= max_dim {
                diagrams[pair.dim].push(pair.clone());
            }
        }
        diagrams
            .into_iter()
            .enumerate()
            .map(|(d, pairs)| Self::new(pairs, d))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pair(birth: f64, death: Option<f64>) -> PersistencePair {
        PersistencePair {
            dim: 0,
            birth,
            death,
        }
    }

    #[test]
    fn from_pairs_by_dim_splits_correctly() {
        let pairs = vec![
            PersistencePair {
                dim: 0,
                birth: 0.0,
                death: Some(1.0),
            },
            PersistencePair {
                dim: 1,
                birth: 0.5,
                death: Some(2.0),
            },
            PersistencePair {
                dim: 0,
                birth: 0.0,
                death: None,
            },
        ];
        let diagrams = PersistenceDiagram::from_pairs_by_dim(&pairs, 1);
        assert_eq!(diagrams[0].pairs.len(), 2);
        assert_eq!(diagrams[1].pairs.len(), 1);
    }

    #[test]
    fn essential_and_finite_filter() {
        let pairs = vec![make_pair(0.0, Some(1.0)), make_pair(0.0, None)];
        let diag = PersistenceDiagram::new(pairs, 0);
        assert_eq!(diag.finite_pairs().len(), 1);
        assert_eq!(diag.essential_classes().len(), 1);
    }
}
