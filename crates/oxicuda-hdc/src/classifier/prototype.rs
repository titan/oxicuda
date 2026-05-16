//! Single class prototype with incremental update for HD classification.

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;
use crate::vector::binary::threshold_binary;

/// Single class prototype with incremental accumulator and online update.
pub struct Prototype {
    /// Class label.
    pub class: usize,
    /// Accumulator for training examples (i32 to avoid overflow).
    acc: Vec<i32>,
    /// Number of training examples added.
    pub count: usize,
    /// Thresholded binary HV (None until build() is called).
    hv: Option<Vec<i8>>,
    /// Dimension of the hypervectors.
    dim: usize,
}

impl Prototype {
    /// Create a new prototype for the given class and dimension.
    pub fn new(class: usize, dim: usize) -> HdcResult<Self> {
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        Ok(Self {
            class,
            acc: vec![0i32; dim],
            count: 0,
            hv: None,
            dim,
        })
    }

    /// Add a training example HV to the accumulator.
    pub fn add(&mut self, hv: &[i8]) -> HdcResult<()> {
        if hv.len() != self.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.dim,
                got: hv.len(),
            });
        }
        for (a, &v) in self.acc.iter_mut().zip(hv.iter()) {
            *a += v as i32;
        }
        self.count += 1;
        Ok(())
    }

    /// Threshold the accumulator into a binary prototype HV.
    pub fn build(&mut self, rng: &mut LcgRng) -> HdcResult<()> {
        self.hv = Some(threshold_binary(&self.acc, rng)?);
        Ok(())
    }

    /// Return the prototype HV (error if not yet built).
    pub fn hv(&self) -> HdcResult<&[i8]> {
        self.hv.as_deref().ok_or(HdcError::PrototypeNotBuilt)
    }

    /// Cosine similarity between the prototype and a query HV.
    /// For binary HVs: cosine = dot(proto, query) / dim.
    pub fn cosine(&self, query: &[i8]) -> HdcResult<f32> {
        let proto = self.hv()?;
        if query.len() != self.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let dot: i64 = proto
            .iter()
            .zip(query.iter())
            .map(|(&p, &q)| (p as i64) * (q as i64))
            .sum();
        Ok(dot as f32 / self.dim as f32)
    }

    /// Subtract an HV from the accumulator (for online error-correction retraining).
    pub fn subtract(&mut self, hv: &[i8]) -> HdcResult<()> {
        if hv.len() != self.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.dim,
                got: hv.len(),
            });
        }
        for (a, &v) in self.acc.iter_mut().zip(hv.iter()) {
            *a -= v as i32;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::vector::binary::random_binary;

    #[test]
    fn prototype_build_and_cosine_self() {
        let mut rng = LcgRng::new(70);
        let mut proto = Prototype::new(0, 256).expect("new");
        let hv = random_binary(256, &mut rng).expect("hv");
        proto.add(&hv).expect("add");
        proto.build(&mut rng).expect("build");
        let sim = proto.cosine(&hv).expect("cosine");
        // single example: prototype should be close to the example
        assert!(sim > 0.9, "sim={sim}");
    }
}
