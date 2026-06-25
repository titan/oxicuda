//! Bark-style hierarchical acoustic-token *layout* over an RVQ codec.
//!
//! # Scope (read this)
//!
//! This module provides **only the codec token structure** — the deterministic
//! split of an EnCodec/SoundStream-style RVQ code stack into "coarse" and "fine"
//! acoustic tiers, and the exact round-trip back to a reconstruction.  It is a
//! thin structural wrapper over [`ResidualVectorQuantizer`] stages.
//!
//! The **trained autoregressive transformers** that actually *generate* Bark's
//! semantic / coarse / fine tokens from text — the real Bark models (Suno 2023)
//! — are **NOT implemented here**.  They require training-scale data and are not
//! CPU-unit-verifiable, so they remain an honest `[ ]` in `TODO.md`.  Bark's
//! *semantic* tokens additionally come from a separate self-supervised model
//! (HuBERT-like); the closest CPU primitive in this crate is
//! [`crate::encoder::KMeansQuantizer`] (acoustic-unit discovery), but the
//! trained semantic transformer itself is out of scope.
//!
//! What *is* verifiable, and is tested here: regrouping the flat RVQ codes into
//! coarse/fine tiers is loss-free (round-trip identical to the flat codec), and
//! a coarse-only decode is never more accurate than the full decode (the RVQ
//! nestedness property surfaced through the tier split).

use crate::codec::rvq::ResidualVectorQuantizer;
use crate::error::{AudioError, AudioResult};

// ─── BarkAcousticTokens ──────────────────────────────────────────────────────

/// Acoustic tokens split into a coarse tier (first `n_coarse` RVQ stages) and a
/// fine tier (the remaining stages).
///
/// This mirrors Bark/EnCodec's coarse-vs-fine acoustic codebook hierarchy: the
/// coarse tier carries the dominant structure, the fine tier the residual
/// detail.  The values are exactly the per-stage RVQ code indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BarkAcousticTokens {
    /// Code indices for the coarse tier (RVQ stages `0..n_coarse`).
    pub coarse: Vec<usize>,
    /// Code indices for the fine tier (RVQ stages `n_coarse..n_quantizers`).
    pub fine: Vec<usize>,
}

impl BarkAcousticTokens {
    /// Flatten the two tiers back into the underlying flat RVQ code stack
    /// (coarse stages first, then fine stages).
    #[must_use]
    pub fn flatten(&self) -> Vec<usize> {
        let mut all = Vec::with_capacity(self.coarse.len() + self.fine.len());
        all.extend_from_slice(&self.coarse);
        all.extend_from_slice(&self.fine);
        all
    }

    /// Total number of code indices across both tiers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.coarse.len() + self.fine.len()
    }

    /// Whether both tiers are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.coarse.is_empty() && self.fine.is_empty()
    }
}

// ─── BarkCodec ───────────────────────────────────────────────────────────────

/// Thin coarse/fine acoustic-token wrapper over a [`ResidualVectorQuantizer`].
///
/// The boundary `n_coarse` partitions the RVQ stages into the coarse and fine
/// tiers.  Encoding/decoding simply regroups the flat RVQ codes; the trained
/// token-generation transformers are out of scope (see the module docs).
#[derive(Debug, Clone)]
pub struct BarkCodec {
    /// Underlying residual vector quantizer.
    rvq: ResidualVectorQuantizer,
    /// Number of leading stages assigned to the coarse tier.
    n_coarse: usize,
}

impl BarkCodec {
    /// Wrap an RVQ, assigning its first `n_coarse` stages to the coarse tier.
    ///
    /// # Errors
    ///
    /// [`AudioError::ShapeMismatch`] if `n_coarse > rvq.n_quantizers()`.
    pub fn new(rvq: ResidualVectorQuantizer, n_coarse: usize) -> AudioResult<Self> {
        if n_coarse > rvq.n_quantizers() {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "bark: n_coarse={} > n_quantizers={}",
                    n_coarse,
                    rvq.n_quantizers()
                ),
            });
        }
        Ok(Self { rvq, n_coarse })
    }

    /// Encode `x` into coarse + fine acoustic tiers.
    ///
    /// # Errors
    ///
    /// [`AudioError::ShapeMismatch`] if `x.len() != dim`.
    pub fn encode(&self, x: &[f32]) -> AudioResult<BarkAcousticTokens> {
        let codes = self.rvq.encode(x)?;
        let (coarse, fine) = codes.split_at(self.n_coarse);
        Ok(BarkAcousticTokens {
            coarse: coarse.to_vec(),
            fine: fine.to_vec(),
        })
    }

    /// Decode both tiers back to a reconstruction (identical to decoding the
    /// equivalent flat RVQ code stack).
    ///
    /// # Errors
    ///
    /// Propagates [`ResidualVectorQuantizer::decode`] errors.
    pub fn decode(&self, tokens: &BarkAcousticTokens) -> AudioResult<Vec<f32>> {
        self.rvq.decode(&tokens.flatten())
    }

    /// Decode only the coarse tier — a real lower-fidelity RVQ preview using the
    /// first `n_coarse` stages.
    ///
    /// # Errors
    ///
    /// Propagates [`ResidualVectorQuantizer::decode`] errors.
    pub fn decode_coarse(&self, tokens: &BarkAcousticTokens) -> AudioResult<Vec<f32>> {
        self.rvq.decode(&tokens.coarse)
    }

    /// Number of coarse-tier stages.
    #[must_use]
    pub fn n_coarse(&self) -> usize {
        self.n_coarse
    }

    /// Number of fine-tier stages.
    #[must_use]
    pub fn n_fine(&self) -> usize {
        self.rvq.n_quantizers() - self.n_coarse
    }

    /// Borrow the underlying residual vector quantizer.
    #[must_use]
    pub fn rvq(&self) -> &ResidualVectorQuantizer {
        &self.rvq
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn dist(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum::<f32>()
            .sqrt()
    }

    /// The tier split is loss-free: round-trip equals the flat RVQ round-trip.
    #[test]
    fn tier_split_matches_flat_rvq() {
        let mut rng = LcgRng::new(314);
        let n_q = 6usize;
        let rvq = ResidualVectorQuantizer::new(n_q, 8, 8, &mut rng).expect("new ok");

        let mut x = vec![0.0_f32; 8];
        LcgRng::new(15).fill_normal(&mut x);

        let flat_codes = rvq.encode(&x).expect("encode ok");
        let flat_hat = rvq.decode(&flat_codes).expect("decode ok");

        let bark = BarkCodec::new(rvq, 2).expect("bark ok");
        assert_eq!(bark.n_coarse(), 2);
        assert_eq!(bark.n_fine(), 4);

        let tokens = bark.encode(&x).expect("bark encode ok");
        assert_eq!(tokens.coarse.len(), 2);
        assert_eq!(tokens.fine.len(), 4);
        assert_eq!(tokens.len(), n_q);
        assert!(!tokens.is_empty());
        assert_eq!(
            tokens.flatten(),
            flat_codes,
            "tier regroup must be loss-free"
        );

        let bark_hat = bark.decode(&tokens).expect("bark decode ok");
        assert_eq!(bark_hat, flat_hat, "tier decode != flat decode");
    }

    /// A coarse-only decode is never more accurate than the full decode
    /// (the RVQ nestedness property via the tier split).
    #[test]
    fn coarse_only_not_more_accurate() {
        let mut rng = LcgRng::new(2718);
        let rvq = ResidualVectorQuantizer::new(5, 8, 7, &mut rng).expect("new ok");
        let bark = BarkCodec::new(rvq, 2).expect("bark ok");

        let mut x = vec![0.0_f32; 7];
        LcgRng::new(99).fill_normal(&mut x);
        let tokens = bark.encode(&x).expect("encode ok");

        let full_err = dist(&x, &bark.decode(&tokens).expect("decode ok"));
        let coarse_err = dist(&x, &bark.decode_coarse(&tokens).expect("decode coarse ok"));
        assert!(
            coarse_err >= full_err - 1e-6,
            "coarse {coarse_err} < full {full_err}"
        );
        assert!(full_err.is_finite() && coarse_err.is_finite());
    }

    /// Boundary validation.
    #[test]
    fn invalid_coarse_boundary() {
        let mut rng = LcgRng::new(7);
        let rvq = ResidualVectorQuantizer::new(3, 4, 4, &mut rng).expect("new ok");
        assert!(matches!(
            BarkCodec::new(rvq, 4).unwrap_err(),
            AudioError::ShapeMismatch { .. }
        ));
    }
}
