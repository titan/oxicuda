//! Barcode representation of a persistence diagram.
//!
//! A barcode is a multi-set of intervals [birth, death) (or [birth, ∞) for essential classes),
//! one per homological generator.

use crate::persistence::diagram::PersistenceDiagram;

/// A single bar in a barcode: [birth, death).
///
/// `death = f64::INFINITY` for essential (never-dying) classes.
#[derive(Debug, Clone)]
pub struct Bar {
    pub birth: f64,
    pub death: f64,
    pub dim: usize,
}

impl Bar {
    /// Persistence lifetime of the bar.
    pub fn lifetime(&self) -> f64 {
        self.death - self.birth
    }
}

/// A collection of bars for a single persistence diagram.
#[derive(Debug, Clone)]
pub struct Barcode {
    pub bars: Vec<Bar>,
}

impl Barcode {
    /// Build a barcode from a `PersistenceDiagram`.
    ///
    /// Essential classes use `f64::INFINITY` as the death value.
    /// `max_death` is unused (we represent essential pairs as infinity).
    pub fn from_diagram(diag: &PersistenceDiagram, _max_death: f64) -> Self {
        let bars = diag
            .pairs
            .iter()
            .map(|p| Bar {
                birth: p.birth,
                death: p.death.unwrap_or(f64::INFINITY),
                dim: p.dim,
            })
            .collect();
        Self { bars }
    }

    /// Persistence lifetimes for all bars.
    pub fn lifetimes(&self) -> Vec<f64> {
        self.bars.iter().map(|b| b.lifetime()).collect()
    }

    /// Count bars with finite persistence strictly greater than `threshold`.
    pub fn count_significant(&self, threshold: f64) -> usize {
        self.bars
            .iter()
            .filter(|b| b.death.is_finite() && b.lifetime() > threshold)
            .count()
    }

    /// Number of finite bars (death < ∞).
    pub fn n_finite(&self) -> usize {
        self.bars.iter().filter(|b| b.death.is_finite()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::homology::persistent::PersistencePair;
    use crate::persistence::diagram::PersistenceDiagram;

    #[test]
    fn barcode_essential_is_infinity() {
        let pairs = vec![PersistencePair {
            dim: 0,
            birth: 0.0,
            death: None,
        }];
        let diag = PersistenceDiagram::new(pairs, 0);
        let bc = Barcode::from_diagram(&diag, 10.0);
        assert_eq!(bc.bars[0].death, f64::INFINITY);
    }

    #[test]
    fn count_significant() {
        let pairs = vec![
            PersistencePair {
                dim: 0,
                birth: 0.0,
                death: Some(0.1),
            },
            PersistencePair {
                dim: 0,
                birth: 0.0,
                death: Some(5.0),
            },
        ];
        let diag = PersistenceDiagram::new(pairs, 0);
        let bc = Barcode::from_diagram(&diag, 100.0);
        assert_eq!(bc.count_significant(1.0), 1); // only the 5.0-0.0=5 bar
    }
}
