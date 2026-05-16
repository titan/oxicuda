//! Record-based (feature-value) encoding for HDC.
//!
//! Each feature i has a feature position HV f_i.
//! Each value v for feature i has a value HV hv(v).
//! Record = Bundle(bind(f_0, hv(val_0)), bind(f_1, hv(val_1)), ..., bind(f_n, hv(val_n))).

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;
use crate::ops::binding::binary_bind;
use crate::ops::bundling::bundle_binary;
use crate::vector::binary::random_binary;

/// Record-based encoder for fixed-schema feature vectors.
pub struct RecordEncoder {
    /// Hypervector dimension.
    dim: usize,
    /// HV per feature position (n_features entries).
    feature_hvs: Vec<Vec<i8>>,
    /// HV per (feature, value_bin) pair: outer index = feature, inner = value bucket.
    value_hvs: Vec<Vec<Vec<i8>>>,
}

impl RecordEncoder {
    /// Create a new record encoder.
    ///
    /// - `n_features`: number of features in the record schema.
    /// - `n_values_per_feature`: number of discrete value buckets per feature.
    /// - `dim`: hypervector dimension.
    /// - `rng`: random number generator for initialization.
    pub fn new(
        n_features: usize,
        n_values_per_feature: usize,
        dim: usize,
        rng: &mut LcgRng,
    ) -> HdcResult<Self> {
        if n_features == 0 {
            return Err(HdcError::EmptyInput);
        }
        if n_values_per_feature == 0 {
            return Err(HdcError::EmptyInput);
        }
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        let mut feature_hvs = Vec::with_capacity(n_features);
        for _ in 0..n_features {
            feature_hvs.push(random_binary(dim, rng)?);
        }
        let mut value_hvs = Vec::with_capacity(n_features);
        for _ in 0..n_features {
            let mut feat_vals = Vec::with_capacity(n_values_per_feature);
            for _ in 0..n_values_per_feature {
                feat_vals.push(random_binary(dim, rng)?);
            }
            value_hvs.push(feat_vals);
        }
        Ok(Self {
            dim,
            feature_hvs,
            value_hvs,
        })
    }

    /// Encode a record: features as discretized bucket indices.
    ///
    /// `feature_values`: slice of bucket indices, one per feature.
    pub fn encode(&self, feature_values: &[usize], rng: &mut LcgRng) -> HdcResult<Vec<i8>> {
        if feature_values.len() != self.feature_hvs.len() {
            return Err(HdcError::DimensionMismatch {
                expected: self.feature_hvs.len(),
                got: feature_values.len(),
            });
        }
        let n_values = self.value_hvs[0].len();
        let mut bound_hvs: Vec<Vec<i8>> = Vec::with_capacity(feature_values.len());
        for (feat_idx, &val_idx) in feature_values.iter().enumerate() {
            if val_idx >= n_values {
                return Err(HdcError::FeatureIndexOutOfRange {
                    feat: val_idx,
                    max: n_values,
                });
            }
            let feat_hv = &self.feature_hvs[feat_idx];
            let val_hv = &self.value_hvs[feat_idx][val_idx];
            let bound = binary_bind(feat_hv, val_hv)?;
            bound_hvs.push(bound);
        }
        bundle_binary(&bound_hvs, rng)
    }

    /// Return the dimension of the encoder.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn record_encoder_different_values_differ() {
        let mut rng = LcgRng::new(90);
        let enc = RecordEncoder::new(3, 4, 256, &mut rng).expect("new");
        let r1 = enc.encode(&[0, 1, 2], &mut rng).expect("encode r1");
        let r2 = enc.encode(&[1, 2, 3], &mut rng).expect("encode r2");
        // Records should differ (not identical)
        let equal = r1.iter().zip(r2.iter()).all(|(a, b)| a == b);
        assert!(!equal, "distinct records produced identical HVs");
    }
}
