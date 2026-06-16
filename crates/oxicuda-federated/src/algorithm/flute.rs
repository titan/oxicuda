//! FLUTE-style personalized federated learning: shared body + local heads.
//!
//! Dimitriadis et al., "FLUTE: A Scalable, Extensible Framework for
//! High-Performance Federated Learning Simulation" (and the associated
//! personalised-FL recipe), 2022.
//!
//! The model is split into two parts:
//! - a **shared body** (feature extractor) that is federated-averaged across
//!   clients every round, and
//! - a per-client **personalised head** (classifier / regressor) that never
//!   leaves the client.
//!
//! Each client trains body + head locally on its own (heterogeneous) data; the
//! server aggregates **only the bodies** (weighted by sample count), while heads
//! stay local. Fast personalisation of a fresh client is expressed as a
//! **task vector** — the delta of a head relative to a base head — which can be
//! transplanted (and scaled) onto another base without any fine-tuning.
//!
//! Concrete model used here (kept small and analytic for testing):
//! `ŷ = H · (B · x)` — a linear body `B` (`d_feat × d_in`) feeding a linear head
//! `H` (`d_out × d_feat`), trained with full-batch gradient descent on mean
//! squared error.

use crate::error::{FedError, FedResult};

/// Configuration for FLUTE personalised training.
#[derive(Debug, Clone)]
pub struct FluteConfig {
    /// Input feature dimension.
    pub d_in: usize,
    /// Shared-representation (body output) dimension.
    pub d_feat: usize,
    /// Output dimension (head output).
    pub d_out: usize,
    /// Local learning rate `> 0`.
    pub learning_rate: f32,
    /// Number of local full-batch gradient-descent epochs.
    pub local_epochs: usize,
}

impl FluteConfig {
    /// Create a validated FLUTE configuration.
    ///
    /// # Errors
    /// - [`FedError::DimensionMismatch`] if any dimension is zero.
    /// - [`FedError::InvalidWeight`] if `learning_rate` is not positive/finite.
    pub fn new(
        d_in: usize,
        d_feat: usize,
        d_out: usize,
        learning_rate: f32,
        local_epochs: usize,
    ) -> FedResult<Self> {
        if d_in == 0 || d_feat == 0 || d_out == 0 {
            return Err(FedError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if !(learning_rate > 0.0 && learning_rate.is_finite()) {
            return Err(FedError::InvalidWeight {
                weight: learning_rate,
            });
        }
        Ok(Self {
            d_in,
            d_feat,
            d_out,
            learning_rate,
            local_epochs,
        })
    }

    /// Number of flat parameters in the shared body (`d_feat · d_in`).
    #[must_use]
    pub fn body_len(&self) -> usize {
        self.d_feat * self.d_in
    }

    /// Number of flat parameters in a head (`d_out · d_feat`).
    #[must_use]
    pub fn head_len(&self) -> usize {
        self.d_out * self.d_feat
    }
}

/// A FLUTE model: a shared linear body plus a (personalised) linear head.
///
/// `body` is row-major `d_feat × d_in`; `head` is row-major `d_out × d_feat`.
#[derive(Debug, Clone)]
pub struct FluteModel {
    /// Input feature dimension.
    pub d_in: usize,
    /// Shared-representation dimension.
    pub d_feat: usize,
    /// Output dimension.
    pub d_out: usize,
    /// Shared body parameters (`d_feat · d_in`).
    pub body: Vec<f32>,
    /// Personalised head parameters (`d_out · d_feat`).
    pub head: Vec<f32>,
}

impl FluteModel {
    /// Create a zero-initialised model for the given configuration.
    #[must_use]
    pub fn new(cfg: &FluteConfig) -> Self {
        Self {
            d_in: cfg.d_in,
            d_feat: cfg.d_feat,
            d_out: cfg.d_out,
            body: vec![0.0_f32; cfg.body_len()],
            head: vec![0.0_f32; cfg.head_len()],
        }
    }

    /// Build a model from explicit body and head parameters.
    ///
    /// # Errors
    /// [`FedError::DimensionMismatch`] if either vector has the wrong length.
    pub fn from_parts(
        d_in: usize,
        d_feat: usize,
        d_out: usize,
        body: Vec<f32>,
        head: Vec<f32>,
    ) -> FedResult<Self> {
        if body.len() != d_feat * d_in {
            return Err(FedError::DimensionMismatch {
                expected: d_feat * d_in,
                got: body.len(),
            });
        }
        if head.len() != d_out * d_feat {
            return Err(FedError::DimensionMismatch {
                expected: d_out * d_feat,
                got: head.len(),
            });
        }
        Ok(Self {
            d_in,
            d_feat,
            d_out,
            body,
            head,
        })
    }

    /// Compute the shared representation `h = B · x`.
    ///
    /// # Errors
    /// [`FedError::DimensionMismatch`] if `x.len() != d_in`.
    pub fn features(&self, x: &[f32]) -> FedResult<Vec<f32>> {
        if x.len() != self.d_in {
            return Err(FedError::DimensionMismatch {
                expected: self.d_in,
                got: x.len(),
            });
        }
        let mut h = vec![0.0_f32; self.d_feat];
        for (row, hf) in self.body.chunks_exact(self.d_in).zip(h.iter_mut()) {
            let mut acc = 0.0_f64;
            for (&w, &xi) in row.iter().zip(x.iter()) {
                acc += (w as f64) * (xi as f64);
            }
            *hf = acc as f32;
        }
        Ok(h)
    }

    /// Forward pass `ŷ = H · (B · x)`.
    ///
    /// # Errors
    /// [`FedError::DimensionMismatch`] if `x.len() != d_in`.
    pub fn forward(&self, x: &[f32]) -> FedResult<Vec<f32>> {
        let h = self.features(x)?;
        let mut y = vec![0.0_f32; self.d_out];
        for (row, yo) in self.head.chunks_exact(self.d_feat).zip(y.iter_mut()) {
            let mut acc = 0.0_f64;
            for (&w, &hf) in row.iter().zip(h.iter()) {
                acc += (w as f64) * (hf as f64);
            }
            *yo = acc as f32;
        }
        Ok(y)
    }
}

/// A client's locally trained update offered to the server.
///
/// Only [`FluteClientUpdate::body`] participates in server aggregation;
/// [`FluteClientUpdate::head`] is carried for bookkeeping but is **ignored**.
#[derive(Debug, Clone)]
pub struct FluteClientUpdate {
    /// Unique client identifier.
    pub client_id: usize,
    /// Locally trained shared-body parameters.
    pub body: Vec<f32>,
    /// Locally trained personalised head (ignored by aggregation).
    pub head: Vec<f32>,
    /// Local dataset size (aggregation weight); must be `> 0`.
    pub n_samples: usize,
}

/// A labelled sample: input features paired with target outputs.
pub type FluteSample = (Vec<f32>, Vec<f32>);

/// Stateless handle providing the FLUTE primitives.
pub struct Flute;

impl Flute {
    /// Mean squared error of `model` over `data` (`(1/N) Σ ‖ŷ − y‖²`).
    ///
    /// # Errors
    /// - [`FedError::EmptyClientList`] if `data` is empty.
    /// - [`FedError::DimensionMismatch`] on input/target shape mismatch.
    pub fn mse(model: &FluteModel, data: &[FluteSample]) -> FedResult<f32> {
        if data.is_empty() {
            return Err(FedError::EmptyClientList);
        }
        let mut total = 0.0_f64;
        for (x, y) in data {
            if y.len() != model.d_out {
                return Err(FedError::DimensionMismatch {
                    expected: model.d_out,
                    got: y.len(),
                });
            }
            let yhat = model.forward(x)?;
            for (&a, &b) in yhat.iter().zip(y.iter()) {
                let e = (a - b) as f64;
                total += e * e;
            }
        }
        Ok((total / data.len() as f64) as f32)
    }

    /// Train a copy of `model` (body **and** head) on local `data` with
    /// full-batch gradient descent for `cfg.local_epochs` epochs.
    ///
    /// # Errors
    /// - [`FedError::EmptyClientList`] if `data` is empty.
    /// - [`FedError::DimensionMismatch`] on shape mismatch (model vs config or
    ///   sample shapes).
    pub fn local_update(
        model: &FluteModel,
        data: &[FluteSample],
        cfg: &FluteConfig,
    ) -> FedResult<FluteModel> {
        if data.is_empty() {
            return Err(FedError::EmptyClientList);
        }
        if model.d_in != cfg.d_in || model.d_feat != cfg.d_feat || model.d_out != cfg.d_out {
            return Err(FedError::DimensionMismatch {
                expected: cfg.body_len() + cfg.head_len(),
                got: model.body.len() + model.head.len(),
            });
        }
        let d_in = cfg.d_in;
        let d_feat = cfg.d_feat;
        let d_out = cfg.d_out;
        let mut trained = model.clone();

        for _epoch in 0..cfg.local_epochs {
            let mut grad_body = vec![0.0_f64; trained.body.len()];
            let mut grad_head = vec![0.0_f64; trained.head.len()];

            for (x, y) in data {
                if x.len() != d_in {
                    return Err(FedError::DimensionMismatch {
                        expected: d_in,
                        got: x.len(),
                    });
                }
                if y.len() != d_out {
                    return Err(FedError::DimensionMismatch {
                        expected: d_out,
                        got: y.len(),
                    });
                }
                let h = trained.features(x)?;
                let yhat = trained.forward(x)?;

                // Output error e = ŷ − y.
                let err: Vec<f64> = yhat
                    .iter()
                    .zip(y.iter())
                    .map(|(&a, &b)| (a - b) as f64)
                    .collect();

                // Head gradient: ∂L/∂H[o,f] = e_o · h_f ;
                // back-propagated feature error dh_f = Σ_o H[o,f] · e_o.
                let mut dh = vec![0.0_f64; d_feat];
                for ((hrow, ghrow), &e_o) in trained
                    .head
                    .chunks_exact(d_feat)
                    .zip(grad_head.chunks_exact_mut(d_feat))
                    .zip(err.iter())
                {
                    for ((gh, &w), (dh_f, &hf)) in ghrow
                        .iter_mut()
                        .zip(hrow.iter())
                        .zip(dh.iter_mut().zip(h.iter()))
                    {
                        *gh += e_o * (hf as f64);
                        *dh_f += e_o * (w as f64);
                    }
                }

                // Body gradient: ∂L/∂B[f,i] = dh_f · x_i.
                for (gbrow, &dh_f) in grad_body.chunks_exact_mut(d_in).zip(dh.iter()) {
                    for (gb, &xi) in gbrow.iter_mut().zip(x.iter()) {
                        *gb += dh_f * (xi as f64);
                    }
                }
            }

            // Gradient of (1/N)Σ‖ŷ−y‖² scales the per-sample sum by 2/N.
            let scale = 2.0 * cfg.learning_rate as f64 / data.len() as f64;
            for (w, &g) in trained.body.iter_mut().zip(grad_body.iter()) {
                *w = (*w as f64 - scale * g) as f32;
            }
            for (w, &g) in trained.head.iter_mut().zip(grad_head.iter()) {
                *w = (*w as f64 - scale * g) as f32;
            }
        }

        Ok(trained)
    }

    /// Server aggregation: sample-weighted mean of the client **bodies** only.
    ///
    /// Heads are deliberately ignored. Returns the new shared body.
    ///
    /// # Errors
    /// - [`FedError::EmptyClientList`] if `updates` is empty.
    /// - [`FedError::DimensionMismatch`] if body lengths differ.
    /// - [`FedError::InvalidWeight`] if any `n_samples` is zero.
    pub fn aggregate_body(updates: &[FluteClientUpdate]) -> FedResult<Vec<f32>> {
        if updates.is_empty() {
            return Err(FedError::EmptyClientList);
        }
        let body_len = updates[0].body.len();
        let mut weight_sum = 0.0_f64;
        for u in updates {
            if u.body.len() != body_len {
                return Err(FedError::DimensionMismatch {
                    expected: body_len,
                    got: u.body.len(),
                });
            }
            if u.n_samples == 0 {
                return Err(FedError::InvalidWeight { weight: 0.0 });
            }
            weight_sum += u.n_samples as f64;
        }
        let mut acc = vec![0.0_f64; body_len];
        for u in updates {
            let w = u.n_samples as f64;
            for (a, &b) in acc.iter_mut().zip(u.body.iter()) {
                *a += w * (b as f64);
            }
        }
        let inv = 1.0 / weight_sum;
        Ok(acc.iter().map(|&a| (a * inv) as f32).collect())
    }

    /// Compute the **task vector** of a head: `adapted − base`.
    ///
    /// The task vector encodes the personalisation applied on top of a base head
    /// and can be transplanted onto another base via [`Flute::apply_task_vector`].
    ///
    /// # Errors
    /// [`FedError::DimensionMismatch`] if the heads differ in length.
    pub fn task_vector(base_head: &[f32], adapted_head: &[f32]) -> FedResult<Vec<f32>> {
        if base_head.len() != adapted_head.len() {
            return Err(FedError::DimensionMismatch {
                expected: base_head.len(),
                got: adapted_head.len(),
            });
        }
        Ok(adapted_head
            .iter()
            .zip(base_head.iter())
            .map(|(&a, &b)| a - b)
            .collect())
    }

    /// Apply a scaled task vector to a base head: `base + scale · task_vector`.
    ///
    /// With `scale = 1.0` this exactly reconstructs the originally adapted head;
    /// other scales interpolate / extrapolate the personalisation for fast,
    /// fine-tuning-free client adaptation.
    ///
    /// # Errors
    /// [`FedError::DimensionMismatch`] if `base_head` and `task_vector` differ.
    pub fn apply_task_vector(
        base_head: &[f32],
        task_vector: &[f32],
        scale: f32,
    ) -> FedResult<Vec<f32>> {
        if base_head.len() != task_vector.len() {
            return Err(FedError::DimensionMismatch {
                expected: base_head.len(),
                got: task_vector.len(),
            });
        }
        Ok(base_head
            .iter()
            .zip(task_vector.iter())
            .map(|(&b, &t)| b + scale * t)
            .collect())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> FluteConfig {
        FluteConfig::new(2, 2, 1, 0.1, 300).expect("test invariant: valid config")
    }

    // Identity body so that features pass inputs straight through.
    fn identity_body_model(cfg: &FluteConfig) -> FluteModel {
        FluteModel::from_parts(
            cfg.d_in,
            cfg.d_feat,
            cfg.d_out,
            vec![1.0, 0.0, 0.0, 1.0], // 2×2 identity
            vec![0.0, 0.0],           // 1×2 zero head
        )
        .expect("test invariant: identity model")
    }

    fn client0_data() -> Vec<FluteSample> {
        // wants y = x0
        vec![
            (vec![1.0, 0.0], vec![1.0]),
            (vec![0.0, 1.0], vec![0.0]),
            (vec![1.0, 1.0], vec![1.0]),
        ]
    }

    fn client1_data() -> Vec<FluteSample> {
        // wants y = x1 (conflicts with client0)
        vec![
            (vec![1.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 1.0], vec![1.0]),
        ]
    }

    #[test]
    fn config_validation() {
        assert!(FluteConfig::new(0, 2, 1, 0.1, 10).is_err());
        assert!(matches!(
            FluteConfig::new(2, 2, 1, 0.0, 10),
            Err(FedError::InvalidWeight { .. })
        ));
        let c = cfg();
        assert_eq!(c.body_len(), 4);
        assert_eq!(c.head_len(), 2);
    }

    #[test]
    fn from_parts_shape_mismatch() {
        // Requirement (e): shape mismatch → error.
        assert!(matches!(
            FluteModel::from_parts(2, 2, 1, vec![1.0, 2.0], vec![0.0, 0.0]),
            Err(FedError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            FluteModel::from_parts(2, 2, 1, vec![1.0, 2.0, 3.0, 4.0], vec![0.0]),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn forward_two_layer_linear() {
        // body = [[1,2],[3,4]], head = [[1,1]] → ŷ = (1·h0 + 1·h1)
        let m = FluteModel::from_parts(2, 2, 1, vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 1.0])
            .expect("model");
        // x = [1,1] → h = [3, 7] → y = 10
        let y = m.forward(&[1.0, 1.0]).expect("forward");
        assert!((y[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn aggregate_body_weighted_mean_and_heads_distinct() {
        // Requirement (a): body averaged across clients == manual weighted mean,
        // heads remain per-client distinct.
        let updates = vec![
            FluteClientUpdate {
                client_id: 0,
                body: vec![1.0, 2.0, 3.0, 4.0],
                head: vec![1.0, 0.0],
                n_samples: 1,
            },
            FluteClientUpdate {
                client_id: 1,
                body: vec![3.0, 4.0, 5.0, 6.0],
                head: vec![0.0, 1.0],
                n_samples: 3,
            },
        ];
        let agg = Flute::aggregate_body(&updates).expect("aggregate");
        // (1·[1,2,3,4] + 3·[3,4,5,6]) / 4 = [2.5, 3.5, 4.5, 5.5]
        let expected = [2.5_f32, 3.5, 4.5, 5.5];
        for (a, e) in agg.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5, "agg={a}, expected={e}");
        }
        // Heads stay distinct.
        assert_ne!(updates[0].head, updates[1].head);
    }

    #[test]
    fn aggregate_ignores_heads() {
        // Requirement (c): aggregation ignores heads — changing heads cannot
        // change the aggregated body.
        let body_a = vec![0.5_f32, 1.5, 2.5, 3.5];
        let mk = |head: Vec<f32>| FluteClientUpdate {
            client_id: 0,
            body: body_a.clone(),
            head,
            n_samples: 2,
        };
        let updates_x = vec![mk(vec![10.0, -10.0]), mk(vec![0.0, 0.0])];
        let updates_y = vec![mk(vec![-99.0, 42.0]), mk(vec![7.0, 7.0])];
        let agg_x = Flute::aggregate_body(&updates_x).expect("agg x");
        let agg_y = Flute::aggregate_body(&updates_y).expect("agg y");
        assert_eq!(agg_x, agg_y, "aggregated body must not depend on heads");
    }

    #[test]
    fn aggregate_empty_errors() {
        // Requirement (d): client count 0 → error.
        assert!(matches!(
            Flute::aggregate_body(&[]),
            Err(FedError::EmptyClientList)
        ));
    }

    #[test]
    fn aggregate_body_length_mismatch_errors() {
        // Requirement (e): shape mismatch → error.
        let updates = vec![
            FluteClientUpdate {
                client_id: 0,
                body: vec![1.0, 2.0, 3.0, 4.0],
                head: vec![0.0],
                n_samples: 1,
            },
            FluteClientUpdate {
                client_id: 1,
                body: vec![1.0, 2.0],
                head: vec![0.0],
                n_samples: 1,
            },
        ];
        assert!(matches!(
            Flute::aggregate_body(&updates),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn aggregate_zero_samples_errors() {
        let updates = vec![FluteClientUpdate {
            client_id: 0,
            body: vec![1.0, 2.0, 3.0, 4.0],
            head: vec![0.0, 0.0],
            n_samples: 0,
        }];
        assert!(matches!(
            Flute::aggregate_body(&updates),
            Err(FedError::InvalidWeight { .. })
        ));
    }

    #[test]
    fn local_update_reduces_local_loss() {
        let cfg = cfg();
        let base = identity_body_model(&cfg);
        let data = client0_data();
        let before = Flute::mse(&base, &data).expect("mse before");
        let trained = Flute::local_update(&base, &data, &cfg).expect("local update");
        let after = Flute::mse(&trained, &data).expect("mse after");
        assert!(
            after < before,
            "local training should reduce MSE: {before} -> {after}"
        );
    }

    #[test]
    fn personalized_head_beats_shared_head() {
        // Requirement (b): a client's personalised head fits its local data
        // better than the shared/global (averaged) head.
        let cfg = cfg();
        let base = identity_body_model(&cfg);
        let data0 = client0_data();
        let data1 = client1_data();

        let model0 = Flute::local_update(&base, &data0, &cfg).expect("train c0");
        let model1 = Flute::local_update(&base, &data1, &cfg).expect("train c1");

        // Shared/global head = plain average of the two personalised heads.
        let shared_head: Vec<f32> = model0
            .head
            .iter()
            .zip(model1.head.iter())
            .map(|(&a, &b)| 0.5 * (a + b))
            .collect();

        // Same (personalised) body, but swap in the shared head.
        let model0_shared = FluteModel::from_parts(
            model0.d_in,
            model0.d_feat,
            model0.d_out,
            model0.body.clone(),
            shared_head,
        )
        .expect("shared-head model");

        let personalised_mse = Flute::mse(&model0, &data0).expect("personalised mse");
        let shared_mse = Flute::mse(&model0_shared, &data0).expect("shared mse");
        assert!(
            personalised_mse < shared_mse,
            "personalised head should fit local data better: personalised={personalised_mse}, shared={shared_mse}"
        );
    }

    #[test]
    fn task_vector_round_trip() {
        let base = vec![1.0_f32, 2.0, 3.0];
        let adapted = vec![1.5_f32, 2.5, 2.0];
        let tv = Flute::task_vector(&base, &adapted).expect("task vector");
        assert!((tv[0] - 0.5).abs() < 1e-6);
        assert!((tv[1] - 0.5).abs() < 1e-6);
        assert!((tv[2] - (-1.0)).abs() < 1e-6);
        // Applying with scale 1 reconstructs the adapted head.
        let recon = Flute::apply_task_vector(&base, &tv, 1.0).expect("apply");
        for (r, a) in recon.iter().zip(adapted.iter()) {
            assert!((r - a).abs() < 1e-6);
        }
        // Scale 0 returns the base unchanged.
        let zero = Flute::apply_task_vector(&base, &tv, 0.0).expect("apply 0");
        for (z, b) in zero.iter().zip(base.iter()) {
            assert!((z - b).abs() < 1e-6);
        }
    }

    #[test]
    fn task_vector_adapts_toward_personalization() {
        // A scaled task vector moves a fresh client's head toward the source
        // client's personalisation without any local fine-tuning.
        let cfg = cfg();
        let base = identity_body_model(&cfg);
        let data0 = client0_data();
        let source = Flute::local_update(&base, &data0, &cfg).expect("train source");
        let tv = Flute::task_vector(&base.head, &source.head).expect("task vector");
        let adapted_head = Flute::apply_task_vector(&base.head, &tv, 1.0).expect("apply");

        let fresh = FluteModel::from_parts(
            source.d_in,
            source.d_feat,
            source.d_out,
            source.body.clone(),
            adapted_head,
        )
        .expect("fresh model");

        let base_mse = Flute::mse(&base, &data0).expect("base mse");
        let adapted_mse = Flute::mse(&fresh, &data0).expect("adapted mse");
        assert!(
            adapted_mse < base_mse,
            "task-vector adaptation should improve fit: base={base_mse}, adapted={adapted_mse}"
        );
    }

    #[test]
    fn task_vector_dim_mismatch() {
        assert!(matches!(
            Flute::task_vector(&[1.0, 2.0], &[1.0]),
            Err(FedError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            Flute::apply_task_vector(&[1.0, 2.0], &[1.0], 1.0),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn local_update_empty_data_errors() {
        let cfg = cfg();
        let base = identity_body_model(&cfg);
        assert!(matches!(
            Flute::local_update(&base, &[], &cfg),
            Err(FedError::EmptyClientList)
        ));
    }
}
