//! FedDF: Ensemble distillation for robust federated model fusion.
//!
//! Lin et al., "Ensemble Distillation for Robust Model Fusion in Federated
//! Learning", NeurIPS 2020.
//!
//! Instead of (or after) plain parameter averaging, FedDF distills the
//! *ensemble* of client models into the global ("student") model using
//! **unlabeled public data**. The student's tempered-softmax outputs are driven
//! to match the averaged client-ensemble ("teacher") predictions through
//! knowledge distillation (KD) over public batches. This is robust to client
//! architecture heterogeneity in the original paper; here the global model is a
//! compact linear soft-max classifier and clients share that architecture so the
//! ensemble teacher and student logits are directly comparable.
//!
//! Pipeline:
//! 1. **Ensemble teacher** — for each public input, average the per-client
//!    tempered-softmax probability vectors (`ensemble_soft_labels`).
//! 2. **KD step** — minimise the cross-entropy `H(q, p_θ)` between the fixed
//!    teacher distribution `q` and the student distribution `p_θ` over the
//!    public minibatches, where the gradient w.r.t. a student logit is
//!    `(p - q)` (the `1/T` factor of the soft-max derivative is folded into the
//!    learning rate).
//! 3. **Loop** — repeat for `distill_steps` epochs across `public_batches`
//!    minibatches of the public set (`distill`).

use crate::error::{FedError, FedResult};

/// A compact linear soft-max classifier: `logits = W · x + b`.
///
/// Parameters are stored as a single flat `Vec<f32>` laid out as the row-major
/// weight matrix `W` (`n_classes × n_features`) immediately followed by the
/// bias vector `b` (`n_classes`). Both the global student model and every
/// client teacher model use this representation so their logits are directly
/// comparable on shared public inputs.
#[derive(Debug, Clone)]
pub struct LinearModel {
    /// Input feature dimension.
    pub n_features: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Flat parameters: `[W (n_classes·n_features) ; b (n_classes)]`.
    pub params: Vec<f32>,
}

impl LinearModel {
    /// Number of flat parameters for a `(n_features, n_classes)` linear model.
    #[must_use]
    pub fn param_len(n_features: usize, n_classes: usize) -> usize {
        n_classes * n_features + n_classes
    }

    /// Create a zero-initialised model (uniform predictions for any input).
    #[must_use]
    pub fn new(n_features: usize, n_classes: usize) -> Self {
        Self {
            n_features,
            n_classes,
            params: vec![0.0_f32; Self::param_len(n_features, n_classes)],
        }
    }

    /// Build a model from explicit flat parameters.
    ///
    /// # Errors
    /// [`FedError::DimensionMismatch`] if `params.len()` does not match the
    /// `(n_features, n_classes)` layout.
    pub fn from_params(n_features: usize, n_classes: usize, params: Vec<f32>) -> FedResult<Self> {
        let expected = Self::param_len(n_features, n_classes);
        if params.len() != expected {
            return Err(FedError::DimensionMismatch {
                expected,
                got: params.len(),
            });
        }
        Ok(Self {
            n_features,
            n_classes,
            params,
        })
    }

    /// Borrow the weight matrix slice (`n_classes · n_features` entries).
    #[must_use]
    fn weights(&self) -> &[f32] {
        &self.params[..self.n_classes * self.n_features]
    }

    /// Borrow the bias slice (`n_classes` entries).
    #[must_use]
    fn bias(&self) -> &[f32] {
        &self.params[self.n_classes * self.n_features..]
    }

    /// Compute raw logits `W · x + b` for a single input.
    ///
    /// # Errors
    /// [`FedError::DimensionMismatch`] if `x.len() != n_features`.
    pub fn logits(&self, x: &[f32]) -> FedResult<Vec<f32>> {
        if x.len() != self.n_features {
            return Err(FedError::DimensionMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }
        let bias = self.bias();
        let mut out = vec![0.0_f32; self.n_classes];
        for ((row, &b), o) in self
            .weights()
            .chunks_exact(self.n_features)
            .zip(bias.iter())
            .zip(out.iter_mut())
        {
            let mut acc = b as f64;
            for (&w, &xi) in row.iter().zip(x.iter()) {
                acc += (w as f64) * (xi as f64);
            }
            *o = acc as f32;
        }
        Ok(out)
    }

    /// Argmax class index of the logits for a single input.
    ///
    /// # Errors
    /// Propagates [`LinearModel::logits`] errors.
    pub fn predict(&self, x: &[f32]) -> FedResult<usize> {
        let logits = self.logits(x)?;
        Ok(argmax(&logits))
    }
}

/// Index of the maximum element (first on ties); `0` for an empty slice.
#[must_use]
pub fn argmax(values: &[f32]) -> usize {
    let mut best = 0_usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best
}

/// Numerically stable tempered soft-max: `softmax(logits / temperature)`.
///
/// # Errors
/// [`FedError::InvalidWeight`] if `temperature` is not positive and finite.
pub fn softmax_with_temperature(logits: &[f32], temperature: f32) -> FedResult<Vec<f32>> {
    if !(temperature > 0.0 && temperature.is_finite()) {
        return Err(FedError::InvalidWeight {
            weight: temperature,
        });
    }
    if logits.is_empty() {
        return Ok(Vec::new());
    }
    let inv_t = 1.0_f64 / temperature as f64;
    let mut max_scaled = f64::NEG_INFINITY;
    for &z in logits {
        let s = (z as f64) * inv_t;
        if s > max_scaled {
            max_scaled = s;
        }
    }
    let mut exps = vec![0.0_f64; logits.len()];
    let mut sum = 0.0_f64;
    for (e, &z) in exps.iter_mut().zip(logits.iter()) {
        let v = ((z as f64) * inv_t - max_scaled).exp();
        *e = v;
        sum += v;
    }
    let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    Ok(exps.iter().map(|&e| (e * inv_sum) as f32).collect())
}

/// Configuration for FedDF ensemble distillation.
#[derive(Debug, Clone)]
pub struct FedDfConfig {
    /// Number of public minibatches the public set is split into per epoch.
    pub public_batches: usize,
    /// Number of distillation epochs (passes over the public set).
    pub distill_steps: usize,
    /// Soft-max temperature `T > 0` for both teacher and student.
    pub temperature: f32,
    /// Distillation learning rate `> 0`.
    pub lr: f32,
}

impl FedDfConfig {
    /// Create a validated FedDF configuration.
    ///
    /// # Errors
    /// - [`FedError::InvalidWeight`] if `temperature` or `lr` is not positive
    ///   and finite.
    /// - [`FedError::Internal`] if `public_batches` is zero.
    pub fn new(
        public_batches: usize,
        distill_steps: usize,
        temperature: f32,
        lr: f32,
    ) -> FedResult<Self> {
        if !(temperature > 0.0 && temperature.is_finite()) {
            return Err(FedError::InvalidWeight {
                weight: temperature,
            });
        }
        if !(lr > 0.0 && lr.is_finite()) {
            return Err(FedError::InvalidWeight { weight: lr });
        }
        if public_batches == 0 {
            return Err(FedError::Internal(
                "public_batches must be at least 1".to_string(),
            ));
        }
        Ok(Self {
            public_batches,
            distill_steps,
            temperature,
            lr,
        })
    }
}

/// Stateless handle providing the FedDF distillation primitives.
pub struct FedDf;

impl FedDf {
    /// Validate that every client shares the global model's architecture.
    fn check_clients(global: &LinearModel, clients: &[LinearModel]) -> FedResult<()> {
        if clients.is_empty() {
            return Err(FedError::EmptyClientList);
        }
        let expected = global.params.len();
        for c in clients {
            if c.n_features != global.n_features || c.n_classes != global.n_classes {
                return Err(FedError::DimensionMismatch {
                    expected,
                    got: c.params.len(),
                });
            }
        }
        Ok(())
    }

    /// Validate that every public input matches the model feature dimension.
    fn check_inputs(global: &LinearModel, inputs: &[Vec<f32>]) -> FedResult<()> {
        if inputs.is_empty() {
            return Err(FedError::Internal("public dataset is empty".to_string()));
        }
        for x in inputs {
            if x.len() != global.n_features {
                return Err(FedError::DimensionMismatch {
                    expected: global.n_features,
                    got: x.len(),
                });
            }
        }
        Ok(())
    }

    /// Aggregate the client ensemble into per-input soft-label distributions.
    ///
    /// For each public input the tempered soft-max probability vectors of all
    /// clients are averaged, producing the teacher distribution `q` that the
    /// student is distilled toward.
    ///
    /// # Errors
    /// - [`FedError::EmptyClientList`] if `clients` is empty.
    /// - [`FedError::DimensionMismatch`] on architecture / input-shape mismatch.
    /// - [`FedError::InvalidWeight`] if `temperature` is non-positive.
    pub fn ensemble_soft_labels(
        clients: &[LinearModel],
        inputs: &[Vec<f32>],
        temperature: f32,
    ) -> FedResult<Vec<Vec<f32>>> {
        if clients.is_empty() {
            return Err(FedError::EmptyClientList);
        }
        if !(temperature > 0.0 && temperature.is_finite()) {
            return Err(FedError::InvalidWeight {
                weight: temperature,
            });
        }
        let n_classes = clients[0].n_classes;
        let n_features = clients[0].n_features;
        for c in clients.iter().skip(1) {
            if c.n_classes != n_classes || c.n_features != n_features {
                return Err(FedError::DimensionMismatch {
                    expected: clients[0].params.len(),
                    got: c.params.len(),
                });
            }
        }
        let inv_k = 1.0_f64 / clients.len() as f64;
        let mut labels = Vec::with_capacity(inputs.len());
        for x in inputs {
            let mut acc = vec![0.0_f64; n_classes];
            for c in clients {
                let logits = c.logits(x)?;
                let probs = softmax_with_temperature(&logits, temperature)?;
                for (a, &p) in acc.iter_mut().zip(probs.iter()) {
                    *a += p as f64;
                }
            }
            labels.push(acc.iter().map(|&a| (a * inv_k) as f32).collect());
        }
        Ok(labels)
    }

    /// Mean KL divergence `D_KL(q ‖ p)` between teacher `q` and a per-input
    /// student distribution `p`, averaged over the batch.
    ///
    /// `1e-12` clamping keeps the logarithm finite for zero-probability entries.
    #[must_use]
    pub fn mean_kl(teacher: &[Vec<f32>], student: &[Vec<f32>]) -> f32 {
        if teacher.is_empty() {
            return 0.0;
        }
        let mut total = 0.0_f64;
        for (q, p) in teacher.iter().zip(student.iter()) {
            for (&qi, &pi) in q.iter().zip(p.iter()) {
                if qi > 0.0 {
                    let pi_safe = (pi as f64).max(1e-12);
                    total += (qi as f64) * ((qi as f64).max(1e-12).ln() - pi_safe.ln());
                }
            }
        }
        (total / teacher.len() as f64) as f32
    }

    /// Mean KD divergence between the ensemble teacher and the current global
    /// student model over the public inputs.
    ///
    /// # Errors
    /// Propagates validation errors from [`FedDf::ensemble_soft_labels`] and
    /// [`LinearModel::logits`].
    pub fn mean_kd_divergence(
        global: &LinearModel,
        clients: &[LinearModel],
        inputs: &[Vec<f32>],
        temperature: f32,
    ) -> FedResult<f32> {
        Self::check_clients(global, clients)?;
        Self::check_inputs(global, inputs)?;
        let teacher = Self::ensemble_soft_labels(clients, inputs, temperature)?;
        let student = Self::student_distribution(global, inputs, temperature)?;
        Ok(Self::mean_kl(&teacher, &student))
    }

    /// Per-input student tempered-softmax distributions for the global model.
    fn student_distribution(
        global: &LinearModel,
        inputs: &[Vec<f32>],
        temperature: f32,
    ) -> FedResult<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(inputs.len());
        for x in inputs {
            let logits = global.logits(x)?;
            out.push(softmax_with_temperature(&logits, temperature)?);
        }
        Ok(out)
    }

    /// Distil the client ensemble into `global` over the unlabeled public set.
    ///
    /// The ensemble teacher distribution is computed **once** (it does not depend
    /// on the student), then for `cfg.distill_steps` epochs the public inputs are
    /// processed in `cfg.public_batches` minibatches; each minibatch performs one
    /// full gradient step that drives the student distribution toward the
    /// teacher. Returns the final mean KD divergence over the whole public set.
    ///
    /// # Errors
    /// - [`FedError::EmptyClientList`] if `clients` is empty.
    /// - [`FedError::Internal`] if `inputs` is empty.
    /// - [`FedError::DimensionMismatch`] on architecture / input-shape mismatch.
    pub fn distill(
        global: &mut LinearModel,
        clients: &[LinearModel],
        inputs: &[Vec<f32>],
        cfg: &FedDfConfig,
    ) -> FedResult<f32> {
        Self::check_clients(global, clients)?;
        Self::check_inputs(global, inputs)?;

        let teacher = Self::ensemble_soft_labels(clients, inputs, cfg.temperature)?;
        let n_features = global.n_features;
        let n_classes = global.n_classes;
        let weight_len = n_classes * n_features;

        // Minibatch boundaries: ceil(n / public_batches) inputs per batch.
        let n = inputs.len();
        let batch_size = n.div_ceil(cfg.public_batches).max(1);

        for _epoch in 0..cfg.distill_steps {
            let mut start = 0_usize;
            while start < n {
                let end = (start + batch_size).min(n);
                let batch_x = &inputs[start..end];
                let batch_q = &teacher[start..end];
                start = end;

                // Accumulate the average KD gradient over this minibatch.
                let mut grad = vec![0.0_f64; global.params.len()];
                let m = batch_x.len();
                if m == 0 {
                    continue;
                }
                for (x, q) in batch_x.iter().zip(batch_q.iter()) {
                    let logits = global.logits(x)?;
                    let p = softmax_with_temperature(&logits, cfg.temperature)?;
                    // Gradient w.r.t. logit c is (p_c - q_c); the 1/T factor of
                    // the soft-max derivative is folded into the learning rate.
                    let (grad_w, grad_b) = grad.split_at_mut(weight_len);
                    for (c, (&pc, &qc)) in p.iter().zip(q.iter()).enumerate() {
                        let g_c = (pc - qc) as f64;
                        let row = &mut grad_w[c * n_features..(c + 1) * n_features];
                        for (gw, &xi) in row.iter_mut().zip(x.iter()) {
                            *gw += g_c * (xi as f64);
                        }
                        grad_b[c] += g_c;
                    }
                }

                let scale = cfg.lr as f64 / m as f64;
                for (param, &g) in global.params.iter_mut().zip(grad.iter()) {
                    *param = (*param as f64 - scale * g) as f32;
                }
            }
        }

        let student = Self::student_distribution(global, inputs, cfg.temperature)?;
        Ok(Self::mean_kl(&teacher, &student))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn teacher_model() -> LinearModel {
        // 3 features, 2 classes. Class 0 favours feature 0, class 1 favours
        // feature 1; feature 2 is a small shared bias channel.
        // W row-major: class0 = [2,-2,0], class1 = [-2,2,0]; bias = [0,0].
        LinearModel::from_params(3, 2, vec![2.0, -2.0, 0.0, -2.0, 2.0, 0.0, 0.0, 0.0])
            .expect("test invariant: valid teacher model")
    }

    fn public_inputs() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0],
            vec![0.8, 0.2, 1.0],
            vec![0.3, 0.7, 1.0],
        ]
    }

    #[test]
    fn linear_model_param_len() {
        assert_eq!(LinearModel::param_len(3, 2), 8);
        assert_eq!(LinearModel::new(4, 5).params.len(), 25);
    }

    #[test]
    fn linear_model_logits_zero_model_is_uniform() {
        let m = LinearModel::new(3, 2);
        let z = m.logits(&[1.0, 2.0, 3.0]).expect("logits");
        assert!(z.iter().all(|&v| v.abs() < 1e-7));
        let p = softmax_with_temperature(&z, 1.0).expect("softmax");
        assert!((p[0] - 0.5).abs() < 1e-6 && (p[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn linear_model_logits_dim_mismatch() {
        let m = LinearModel::new(3, 2);
        assert!(matches!(
            m.logits(&[1.0, 2.0]),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn from_params_dim_mismatch() {
        assert!(matches!(
            LinearModel::from_params(3, 2, vec![0.0; 7]),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn softmax_sums_to_one_and_temperature_flattens() {
        let logits = vec![2.0_f32, 0.0, -1.0];
        let p1 = softmax_with_temperature(&logits, 1.0).expect("softmax t=1");
        let sum: f32 = p1.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // High temperature flattens toward uniform → max prob shrinks.
        let p_hot = softmax_with_temperature(&logits, 10.0).expect("softmax t=10");
        let max1 = p1.iter().cloned().fold(0.0_f32, f32::max);
        let max_hot = p_hot.iter().cloned().fold(0.0_f32, f32::max);
        assert!(
            max_hot < max1,
            "temperature should flatten the distribution"
        );
    }

    #[test]
    fn config_rejects_nonpositive_temperature() {
        // Requirement (c): temperature > 0 enforced.
        assert!(matches!(
            FedDfConfig::new(1, 10, 0.0, 0.1),
            Err(FedError::InvalidWeight { .. })
        ));
        assert!(matches!(
            FedDfConfig::new(1, 10, -1.0, 0.1),
            Err(FedError::InvalidWeight { .. })
        ));
        assert!(FedDfConfig::new(1, 10, 2.0, 0.1).is_ok());
    }

    #[test]
    fn config_rejects_nonpositive_lr_and_zero_batches() {
        assert!(matches!(
            FedDfConfig::new(1, 10, 1.0, 0.0),
            Err(FedError::InvalidWeight { .. })
        ));
        assert!(matches!(
            FedDfConfig::new(0, 10, 1.0, 0.1),
            Err(FedError::Internal(_))
        ));
    }

    #[test]
    fn ensemble_soft_labels_averages_clients() {
        // Two clients with opposite confident logits → ensemble is uniform.
        let c0 = LinearModel::from_params(1, 2, vec![10.0, -10.0, 0.0, 0.0]).expect("c0");
        let c1 = LinearModel::from_params(1, 2, vec![-10.0, 10.0, 0.0, 0.0]).expect("c1");
        let inputs = vec![vec![1.0_f32]];
        let labels = FedDf::ensemble_soft_labels(&[c0, c1], &inputs, 1.0).expect("labels");
        assert_eq!(labels.len(), 1);
        assert!((labels[0][0] - 0.5).abs() < 1e-4);
        assert!((labels[0][1] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn ensemble_soft_labels_empty_clients_errors() {
        // Requirement (d): empty client list → error.
        let inputs = vec![vec![1.0_f32]];
        assert!(matches!(
            FedDf::ensemble_soft_labels(&[], &inputs, 1.0),
            Err(FedError::EmptyClientList)
        ));
    }

    #[test]
    fn distill_empty_clients_errors() {
        // Requirement (d): empty client list → error in the full loop.
        let mut global = LinearModel::new(3, 2);
        let inputs = public_inputs();
        let cfg = FedDfConfig::new(1, 5, 1.0, 0.1).expect("cfg");
        assert!(matches!(
            FedDf::distill(&mut global, &[], &inputs, &cfg),
            Err(FedError::EmptyClientList)
        ));
    }

    #[test]
    fn distill_shape_mismatch_errors() {
        // Requirement (e): dim/shape mismatch → error.
        let mut global = LinearModel::new(3, 2);
        let client = teacher_model();
        // Public input with wrong feature length.
        let bad_inputs = vec![vec![1.0_f32, 2.0]];
        let cfg = FedDfConfig::new(1, 5, 1.0, 0.1).expect("cfg");
        assert!(matches!(
            FedDf::distill(
                &mut global,
                std::slice::from_ref(&client),
                &bad_inputs,
                &cfg
            ),
            Err(FedError::DimensionMismatch { .. })
        ));
        // Client with mismatched architecture.
        let mut global2 = LinearModel::new(3, 2);
        let wrong_client = LinearModel::new(4, 2);
        assert!(matches!(
            FedDf::distill(&mut global2, &[wrong_client], &public_inputs(), &cfg),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn distill_reduces_kd_divergence() {
        // Requirement (a): after distillation predictions move CLOSER to the
        // ensemble average → mean KD divergence decreases.
        let mut global = LinearModel::new(3, 2);
        // Two genuinely different clients.
        let c0 = teacher_model();
        let c1 = LinearModel::from_params(3, 2, vec![1.0, 0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
            .expect("c1");
        let clients = vec![c0, c1];
        let inputs = public_inputs();

        let initial = FedDf::mean_kd_divergence(&global, &clients, &inputs, 1.0).expect("init kl");
        let cfg = FedDfConfig::new(1, 40, 1.0, 0.1).expect("cfg");
        let final_kl = FedDf::distill(&mut global, &clients, &inputs, &cfg).expect("distill");
        assert!(
            final_kl < initial,
            "KD divergence should decrease: initial={initial}, final={final_kl}"
        );
        assert!(final_kl >= 0.0, "KL must be non-negative");
    }

    #[test]
    fn distill_converges_to_unanimous_ensemble() {
        // Requirement (b): if all clients agree, the global converges to that
        // agreement (predictions match, divergence collapses).
        let teacher = teacher_model();
        let clients = vec![teacher.clone(), teacher.clone(), teacher.clone()];
        let inputs = public_inputs();
        let mut global = LinearModel::new(3, 2);

        let initial = FedDf::mean_kd_divergence(&global, &clients, &inputs, 1.0).expect("init");
        let cfg = FedDfConfig::new(1, 400, 1.0, 0.3).expect("cfg");
        let final_kl = FedDf::distill(&mut global, &clients, &inputs, &cfg).expect("distill");

        assert!(
            final_kl < initial * 0.5,
            "should collapse divergence: initial={initial}, final={final_kl}"
        );
        // Student argmax should match the unanimous teacher argmax everywhere.
        for x in &inputs {
            assert_eq!(
                global.predict(x).expect("student predict"),
                teacher.predict(x).expect("teacher predict"),
                "argmax mismatch for input {x:?}"
            );
        }
    }

    #[test]
    fn distill_minibatches_match_full_batch_direction() {
        // Splitting the public set into batches still reduces divergence.
        let teacher = teacher_model();
        let clients = vec![teacher.clone()];
        let inputs = public_inputs();
        let mut global = LinearModel::new(3, 2);
        let initial = FedDf::mean_kd_divergence(&global, &clients, &inputs, 1.0).expect("init");
        let cfg = FedDfConfig::new(2, 50, 1.0, 0.2).expect("cfg");
        let final_kl = FedDf::distill(&mut global, &clients, &inputs, &cfg).expect("distill");
        assert!(
            final_kl < initial,
            "minibatch distillation should still reduce KL"
        );
    }

    #[test]
    fn argmax_picks_largest() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
        assert_eq!(argmax(&[2.0, 2.0, 1.0]), 0);
        assert_eq!(argmax(&[]), 0);
    }
}
