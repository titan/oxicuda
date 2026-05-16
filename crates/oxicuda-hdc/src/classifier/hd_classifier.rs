//! HD Classifier: one prototype HV per class with error-corrective online update.

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;
use crate::vector::binary::threshold_binary;

/// HD Classifier with one prototype binary HV per class.
///
/// Training: accumulate example HVs per class, then build binary prototypes.
/// Classification: argmax cosine_similarity(query, prototypes).
/// Online update: adjust accumulators on misclassification and rebuild.
pub struct HdClassifier {
    /// Number of classes.
    pub n_classes: usize,
    /// Dimension of hypervectors.
    dim: usize,
    /// Per-class i32 accumulator (sum of training HVs).
    accumulators: Vec<Vec<i32>>,
    /// Thresholded binary prototypes (built from accumulators).
    prototypes: Vec<Vec<i8>>,
    /// Training example counts per class.
    counts: Vec<usize>,
}

impl HdClassifier {
    /// Create a new HD classifier.
    pub fn new(n_classes: usize, dim: usize) -> HdcResult<Self> {
        if n_classes == 0 {
            return Err(HdcError::EmptyInput);
        }
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        Ok(Self {
            n_classes,
            dim,
            accumulators: vec![vec![0i32; dim]; n_classes],
            prototypes: vec![vec![1i8; dim]; n_classes],
            counts: vec![0usize; n_classes],
        })
    }

    /// Add a training example for the given class.
    pub fn add_example(&mut self, class: usize, hv: &[i8]) -> HdcResult<()> {
        if class >= self.n_classes {
            return Err(HdcError::ClassNotFound(class));
        }
        if hv.len() != self.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.dim,
                got: hv.len(),
            });
        }
        for (a, &v) in self.accumulators[class].iter_mut().zip(hv.iter()) {
            *a += v as i32;
        }
        self.counts[class] += 1;
        Ok(())
    }

    /// Build binary prototypes from accumulators (threshold majority vote).
    pub fn build_prototypes(&mut self, rng: &mut LcgRng) -> HdcResult<()> {
        for c in 0..self.n_classes {
            self.prototypes[c] = threshold_binary(&self.accumulators[c], rng)?;
        }
        Ok(())
    }

    /// Classify: return the class with highest cosine(query, prototype).
    pub fn classify(&self, query: &[i8]) -> HdcResult<usize> {
        if query.len() != self.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let mut best_class = 0usize;
        let mut best_sim = f32::NEG_INFINITY;
        for c in 0..self.n_classes {
            let dot: i64 = self.prototypes[c]
                .iter()
                .zip(query.iter())
                .map(|(&p, &q)| (p as i64) * (q as i64))
                .sum();
            let sim = dot as f32 / self.dim as f32;
            if sim > best_sim {
                best_sim = sim;
                best_class = c;
            }
        }
        Ok(best_class)
    }

    /// Online update on misclassification:
    /// - Add query HV to true class accumulator.
    /// - Subtract query HV from predicted class accumulator.
    /// - Rebuild prototypes for both classes.
    pub fn online_update(
        &mut self,
        query: &[i8],
        true_class: usize,
        predicted: usize,
        rng: &mut LcgRng,
    ) -> HdcResult<()> {
        if true_class >= self.n_classes {
            return Err(HdcError::ClassNotFound(true_class));
        }
        if predicted >= self.n_classes {
            return Err(HdcError::ClassNotFound(predicted));
        }
        if query.len() != self.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        // Update true class (add)
        for (a, &v) in self.accumulators[true_class].iter_mut().zip(query.iter()) {
            *a += v as i32;
        }
        self.counts[true_class] += 1;
        // Update predicted class (subtract)
        for (a, &v) in self.accumulators[predicted].iter_mut().zip(query.iter()) {
            *a -= v as i32;
        }
        // Rebuild affected prototypes
        self.prototypes[true_class] = threshold_binary(&self.accumulators[true_class], rng)?;
        self.prototypes[predicted] = threshold_binary(&self.accumulators[predicted], rng)?;
        Ok(())
    }

    /// Return the prototype HV for the given class.
    pub fn prototype(&self, class: usize) -> HdcResult<&[i8]> {
        if class >= self.n_classes {
            return Err(HdcError::ClassNotFound(class));
        }
        Ok(&self.prototypes[class])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::vector::binary::random_binary;

    #[test]
    fn classifier_two_distinct_classes() {
        let mut rng = LcgRng::new(80);
        let dim = 512;
        let mut clf = HdClassifier::new(2, dim).expect("new");

        // Generate class prototype HVs
        let proto0 = random_binary(dim, &mut rng).expect("proto0");
        let proto1 = random_binary(dim, &mut rng).expect("proto1");

        // Train with noisy versions of each prototype
        for _ in 0..5 {
            clf.add_example(0, &proto0).expect("add class0");
            clf.add_example(1, &proto1).expect("add class1");
        }
        clf.build_prototypes(&mut rng).expect("build");

        let pred0 = clf.classify(&proto0).expect("classify 0");
        let pred1 = clf.classify(&proto1).expect("classify 1");
        assert_eq!(pred0, 0);
        assert_eq!(pred1, 1);
    }
}
