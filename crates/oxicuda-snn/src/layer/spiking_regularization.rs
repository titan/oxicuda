//! Stochastic regularisers for spiking networks: spike dropout and stochastic
//! depth.
//!
//! Both regularisers behave differently in training and inference, mirroring the
//! conventions of their artificial-network ancestors but adapted to spike
//! trains.
//!
//! * **Spike dropout** (Srivastava et al. 2014, adapted to binary spikes):
//!   during training each spike is independently zeroed with probability `p`,
//!   forcing downstream neurons not to rely on any single afferent. Kept spikes
//!   stay exactly binary. At inference nothing is dropped; instead every output
//!   is scaled by the keep probability `1 − p` so the *expected* drive matches
//!   the training-time average.
//!
//! * **Stochastic depth** (Huang et al. 2016, "Deep Networks with Stochastic
//!   Depth"): a residual block `out = identity + f(x)` is, during training, kept
//!   with survival probability `p` and otherwise skipped entirely
//!   (`out = identity`). The whole branch shares a single Bernoulli draw — it is
//!   the *block* that is dropped, not individual units. At inference the residual
//!   branch is scaled by `p`, giving the expected output `identity + p · f(x)`.
//!   This lets very deep spiking residual stacks train with a shorter expected
//!   depth.
//!
//! Both types draw from the crate [`crate::handle::LcgRng`], whose `next_f32` is uniform on
//! `[0, 1)`, so a Bernoulli event with probability `q` is simply `u < q`.

use crate::error::{SnnError, SnnResult};
use crate::handle::LcgRng;

/// Spike-train dropout with the classic (non-inverted) train/inference split.
#[derive(Debug, Clone, Copy)]
pub struct SpikingDropout {
    /// Drop probability `p ∈ [0, 1)`.
    pub p: f32,
    /// Whether the layer is in training mode (drop) or inference mode (scale).
    pub training: bool,
}

impl SpikingDropout {
    /// Create a dropout layer in training mode.
    ///
    /// Returns [`SnnError::OutOfRange`] unless `p` is finite and in `[0, 1)`.
    pub fn new(p: f32) -> SnnResult<Self> {
        if !p.is_finite() || !(0.0..1.0).contains(&p) {
            return Err(SnnError::OutOfRange {
                name: "p".to_string(),
                val: p,
            });
        }
        Ok(Self { p, training: true })
    }

    /// Keep probability `1 − p`.
    #[must_use]
    pub fn keep_prob(&self) -> f32 {
        1.0 - self.p
    }

    /// Switch to training mode (spikes are randomly dropped).
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Switch to inference mode (no drops; outputs scaled by `1 − p`).
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Explicitly set the training flag.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Apply dropout, writing the result to `out`.
    ///
    /// In training mode each spike survives with probability `1 − p` and is
    /// otherwise set to zero; kept values are passed through unchanged (so a
    /// binary input stays binary). In inference mode every value is scaled by
    /// `1 − p` and the RNG is left untouched.
    ///
    /// Returns [`SnnError::IncompatibleLength`] if `spikes` and `out` differ.
    pub fn forward(&self, spikes: &[f32], rng: &mut LcgRng, out: &mut [f32]) -> SnnResult<()> {
        if spikes.len() != out.len() {
            return Err(SnnError::IncompatibleLength {
                a: spikes.len(),
                b: out.len(),
            });
        }
        if self.training {
            for (s, o) in spikes.iter().zip(out.iter_mut()) {
                let u = rng.next_f32();
                *o = if u < self.p { 0.0 } else { *s };
            }
        } else {
            let keep = self.keep_prob();
            for (s, o) in spikes.iter().zip(out.iter_mut()) {
                *o = *s * keep;
            }
        }
        Ok(())
    }
}

/// Stochastic depth over a spiking residual block `out = identity + residual`.
#[derive(Debug, Clone, Copy)]
pub struct StochasticDepth {
    /// Survival probability `p ∈ [0, 1]` of keeping the residual branch.
    pub survival_prob: f32,
    /// Whether the layer is in training mode (Bernoulli skip) or inference mode
    /// (expected/scaled output).
    pub training: bool,
}

impl StochasticDepth {
    /// Create a stochastic-depth layer in training mode.
    ///
    /// Returns [`SnnError::OutOfRange`] unless `survival_prob` is finite and in
    /// `[0, 1]`.
    pub fn new(survival_prob: f32) -> SnnResult<Self> {
        if !survival_prob.is_finite() || !(0.0..=1.0).contains(&survival_prob) {
            return Err(SnnError::OutOfRange {
                name: "survival_prob".to_string(),
                val: survival_prob,
            });
        }
        Ok(Self {
            survival_prob,
            training: true,
        })
    }

    /// Switch to training mode (Bernoulli block skipping).
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Switch to inference mode (residual scaled by the survival probability).
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Explicitly set the training flag.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Combine a residual block with its identity short-cut, writing to `out`.
    ///
    /// In training mode a single Bernoulli draw decides whether the *whole*
    /// residual branch is kept (`out = identity + residual`) or skipped
    /// (`out = identity`). In inference mode the residual is scaled by the
    /// survival probability, yielding the expected output
    /// `identity + p · residual`; the RNG is left untouched.
    ///
    /// Returns [`SnnError::IncompatibleLength`] if the three slices disagree in
    /// length.
    pub fn forward(
        &self,
        identity: &[f32],
        residual: &[f32],
        rng: &mut LcgRng,
        out: &mut [f32],
    ) -> SnnResult<()> {
        if identity.len() != residual.len() {
            return Err(SnnError::IncompatibleLength {
                a: identity.len(),
                b: residual.len(),
            });
        }
        if identity.len() != out.len() {
            return Err(SnnError::IncompatibleLength {
                a: identity.len(),
                b: out.len(),
            });
        }
        let scale = if self.training {
            if rng.next_f32() < self.survival_prob {
                1.0
            } else {
                0.0
            }
        } else {
            self.survival_prob
        };
        for ((id, r), o) in identity.iter().zip(residual.iter()).zip(out.iter_mut()) {
            *o = *id + scale * *r;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dropout_zeros_configured_fraction() {
        let drop = SpikingDropout::new(0.3).expect("ctor");
        let n = 8000;
        let spikes = vec![1.0_f32; n];
        let mut out = vec![0.0_f32; n];
        let mut rng = LcgRng::new(1);
        drop.forward(&spikes, &mut rng, &mut out).expect("forward");
        let zeros = out.iter().filter(|&&s| s == 0.0).count();
        let frac = zeros as f32 / n as f32;
        assert!(
            (frac - 0.3).abs() < 0.03,
            "dropped fraction {frac} far from p=0.3"
        );
    }

    #[test]
    fn dropout_eval_scales_and_keeps_all() {
        let mut drop = SpikingDropout::new(0.25).expect("ctor");
        drop.eval();
        let spikes = vec![1.0_f32; 100];
        let mut out = vec![0.0_f32; 100];
        let mut rng = LcgRng::new(2);
        drop.forward(&spikes, &mut rng, &mut out).expect("forward");
        for &o in &out {
            assert!((o - 0.75).abs() < 1e-6, "eval should scale by keep=0.75");
        }
    }

    #[test]
    fn dropout_kept_spikes_stay_binary() {
        let drop = SpikingDropout::new(0.5).expect("ctor");
        let spikes: Vec<f32> = (0..200).map(|i| (i % 2) as f32).collect();
        let mut out = vec![0.0_f32; 200];
        let mut rng = LcgRng::new(3);
        drop.forward(&spikes, &mut rng, &mut out).expect("forward");
        for &o in &out {
            assert!(o == 0.0 || o == 1.0, "training output not binary: {o}");
        }
    }

    #[test]
    fn dropout_p_zero_keeps_everything() {
        let drop = SpikingDropout::new(0.0).expect("ctor");
        let spikes = vec![1.0_f32; 64];
        let mut out = vec![0.0_f32; 64];
        let mut rng = LcgRng::new(4);
        drop.forward(&spikes, &mut rng, &mut out).expect("forward");
        assert_eq!(out, spikes);
    }

    #[test]
    fn dropout_determinism_under_fixed_seed() {
        let drop = SpikingDropout::new(0.4).expect("ctor");
        let spikes = vec![1.0_f32; 500];
        let mut oa = vec![0.0_f32; 500];
        let mut ob = vec![0.0_f32; 500];
        let mut ra = LcgRng::new(123);
        let mut rb = LcgRng::new(123);
        drop.forward(&spikes, &mut ra, &mut oa).expect("a");
        drop.forward(&spikes, &mut rb, &mut ob).expect("b");
        assert_eq!(oa, ob);
    }

    #[test]
    fn dropout_rejects_bad_p_and_shape() {
        assert!(matches!(
            SpikingDropout::new(1.0),
            Err(SnnError::OutOfRange { .. })
        ));
        assert!(matches!(
            SpikingDropout::new(-0.1),
            Err(SnnError::OutOfRange { .. })
        ));
        assert!(matches!(
            SpikingDropout::new(f32::NAN),
            Err(SnnError::OutOfRange { .. })
        ));
        let drop = SpikingDropout::new(0.3).expect("ctor");
        let mut rng = LcgRng::new(5);
        let mut out = vec![0.0_f32; 3];
        assert!(matches!(
            drop.forward(&[1.0; 4], &mut rng, &mut out),
            Err(SnnError::IncompatibleLength { .. })
        ));
    }

    #[test]
    fn stochastic_depth_eval_is_identity_scaled_residual() {
        let mut sd = StochasticDepth::new(0.7).expect("ctor");
        sd.eval();
        let identity = vec![0.2_f32, 0.5, 1.0, 0.0];
        let residual = vec![1.0_f32, 1.0, 0.0, 2.0];
        let mut out = vec![0.0_f32; 4];
        let mut rng = LcgRng::new(6);
        sd.forward(&identity, &residual, &mut rng, &mut out)
            .expect("forward");
        for i in 0..4 {
            let expected = identity[i] + 0.7 * residual[i];
            assert!((out[i] - expected).abs() < 1e-6, "i={i}");
        }
    }

    #[test]
    fn stochastic_depth_survival_respected_statistically() {
        let sd = StochasticDepth::new(0.6).expect("ctor");
        let identity = vec![0.0_f32];
        let residual = vec![1.0_f32];
        let mut rng = LcgRng::new(7);
        let trials = 6000;
        let mut survived = 0usize;
        let mut out = vec![0.0_f32; 1];
        for _ in 0..trials {
            sd.forward(&identity, &residual, &mut rng, &mut out)
                .expect("forward");
            // out = 0 + scale*1 = scale ∈ {0,1}.
            if out[0] == 1.0 {
                survived += 1;
            }
        }
        let frac = survived as f32 / trials as f32;
        assert!(
            (frac - 0.6).abs() < 0.03,
            "survival fraction {frac} far from p=0.6"
        );
    }

    #[test]
    fn stochastic_depth_training_is_identity_or_full_residual() {
        let sd = StochasticDepth::new(0.5).expect("ctor");
        let identity = vec![0.3_f32, 0.7, 0.1];
        let residual = vec![1.0_f32, 2.0, 3.0];
        let mut rng = LcgRng::new(8);
        let mut out = vec![0.0_f32; 3];
        for _ in 0..200 {
            sd.forward(&identity, &residual, &mut rng, &mut out)
                .expect("forward");
            let is_identity = out
                .iter()
                .zip(identity.iter())
                .all(|(&o, &id)| (o - id).abs() < 1e-6);
            let is_full = out
                .iter()
                .zip(identity.iter().zip(residual.iter()))
                .all(|(&o, (&id, &r))| (o - (id + r)).abs() < 1e-6);
            assert!(
                is_identity || is_full,
                "training output must be identity or identity+residual: {out:?}"
            );
        }
    }

    #[test]
    fn stochastic_depth_determinism_and_finite() {
        let sd = StochasticDepth::new(0.5).expect("ctor");
        let identity = vec![0.1_f32; 16];
        let residual = vec![0.9_f32; 16];
        let mut ra = LcgRng::new(321);
        let mut rb = LcgRng::new(321);
        let mut oa = vec![0.0_f32; 16];
        let mut ob = vec![0.0_f32; 16];
        for _ in 0..10 {
            sd.forward(&identity, &residual, &mut ra, &mut oa)
                .expect("a");
            sd.forward(&identity, &residual, &mut rb, &mut ob)
                .expect("b");
            assert_eq!(oa, ob);
            for &v in &oa {
                assert!(v.is_finite());
            }
        }
    }

    #[test]
    fn stochastic_depth_rejects_bad_prob_and_shape() {
        assert!(matches!(
            StochasticDepth::new(1.5),
            Err(SnnError::OutOfRange { .. })
        ));
        assert!(matches!(
            StochasticDepth::new(-0.2),
            Err(SnnError::OutOfRange { .. })
        ));
        // p = 1.0 is valid (always survive).
        assert!(StochasticDepth::new(1.0).is_ok());
        let sd = StochasticDepth::new(0.5).expect("ctor");
        let mut rng = LcgRng::new(9);
        let mut out = vec![0.0_f32; 3];
        assert!(matches!(
            sd.forward(&[0.0; 3], &[0.0; 2], &mut rng, &mut out),
            Err(SnnError::IncompatibleLength { .. })
        ));
        assert!(matches!(
            sd.forward(&[0.0; 3], &[0.0; 3], &mut rng, &mut [0.0; 4]),
            Err(SnnError::IncompatibleLength { .. })
        ));
    }
}
