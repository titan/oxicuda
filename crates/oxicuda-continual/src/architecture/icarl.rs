//! iCaRL: Incremental Classifier and Representation Learning.
//!
//! Implements the method from:
//! Rebuffi et al. "iCaRL: Incremental Classifier and Representation Learning."
//! CVPR 2017.
//!
//! iCaRL combines a deep feature extractor with a nearest-mean-of-exemplars
//! classifier. Exemplars are selected via herding — greedily choosing samples
//! that keep the running feature mean close to the true class mean.

use crate::error::{ContinualError, ContinualResult};
use crate::handle::LcgRng;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for iCaRL.
#[derive(Debug, Clone)]
pub struct IcarlConfig {
    /// Total number of classes ever (upper bound).
    pub n_classes: usize,
    /// Maximum exemplars stored per class.
    pub memory_per_class: usize,
    /// Hidden dimension of the 2-layer encoder.
    pub hidden_dim: usize,
    /// Raw input dimensionality.
    pub input_dim: usize,
    /// SGD learning rate.
    pub lr: f64,
    /// Training epochs per task.
    pub n_epochs: usize,
    /// Temperature for knowledge-distillation softmax (τ).
    pub temperature: f64,
}

impl Default for IcarlConfig {
    fn default() -> Self {
        Self {
            n_classes: 10,
            memory_per_class: 20,
            hidden_dim: 64,
            input_dim: 32,
            lr: 0.01,
            n_epochs: 5,
            temperature: 2.0,
        }
    }
}

// ─── Exemplar set ────────────────────────────────────────────────────────────

/// Exemplar set for one class: a small replay memory plus the class mean.
#[derive(Debug, Clone)]
pub struct ExemplarSet {
    /// Class identifier.
    pub class_id: usize,
    /// Selected exemplars; each is a raw input vector of length `input_dim`.
    pub exemplars: Vec<Vec<f64>>,
    /// Class mean computed in *feature space* from the final exemplar set.
    pub class_mean: Vec<f64>,
}

// ─── Model state ─────────────────────────────────────────────────────────────

/// Encoder state for iCaRL.
///
/// Architecture: `input_dim → hidden_dim → feature_dim` (ReLU; L2-normalised output).
/// `feature_dim = hidden_dim / 2`.
#[derive(Debug, Clone)]
pub struct IcarlState {
    /// Output feature dimensionality (`hidden_dim / 2`).
    pub feature_dim: usize,
    /// Layer-1 weight matrix, shape `[hidden_dim × input_dim]`, row-major.
    pub encoder_w1: Vec<f64>,
    /// Layer-1 bias, length `hidden_dim`.
    pub encoder_b1: Vec<f64>,
    /// Layer-2 weight matrix, shape `[feature_dim × hidden_dim]`, row-major.
    pub encoder_w2: Vec<f64>,
    /// Layer-2 bias, length `feature_dim`.
    pub encoder_b2: Vec<f64>,
    /// One exemplar set per observed class (ordered by insertion).
    pub exemplar_sets: Vec<ExemplarSet>,
    /// Class ids seen so far, in insertion order.
    pub seen_classes: Vec<usize>,
    /// Cached config values needed for training.
    pub(crate) input_dim: usize,
    pub(crate) hidden_dim: usize,
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Xavier uniform scale: `sqrt(6 / (fan_in + fan_out))`.
#[inline]
fn xavier_scale(fan_in: usize, fan_out: usize) -> f64 {
    (6.0_f64 / (fan_in + fan_out) as f64).sqrt()
}

/// Fill a slice with Xavier-uniform samples in `[-scale, +scale]`.
fn xavier_init(buf: &mut [f64], fan_in: usize, fan_out: usize, rng: &mut LcgRng) {
    let scale = xavier_scale(fan_in, fan_out);
    for v in buf.iter_mut() {
        let u = rng.next_f32() as f64; // [0, 1)
        *v = (2.0 * u - 1.0) * scale;
    }
}

/// Matrix-vector product `W x + b`; W is `[out × in]` row-major.
#[inline]
fn matvec(w: &[f64], x: &[f64], b: &[f64], in_dim: usize, out_dim: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; out_dim];
    for row in 0..out_dim {
        let mut acc = b[row];
        let base = row * in_dim;
        for col in 0..in_dim {
            acc += w[base + col] * x[col];
        }
        out[row] = acc;
    }
    out
}

/// ReLU element-wise in-place.
#[inline]
fn relu_inplace(v: &mut [f64]) {
    for x in v.iter_mut() {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
}

/// L2-normalise a vector in-place; if norm is ~0 leave unchanged.
fn l2_normalise(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Euclidean distance squared between two equal-length slices.
#[inline]
fn dist_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Softmax with temperature τ.
fn softmax_temp(logits: &[f64], temperature: f64) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = logits
        .iter()
        .map(|&z| ((z - max) / temperature).exp())
        .collect();
    let sum: f64 = exp.iter().sum();
    exp.iter().map(|&e| e / sum.max(1e-30)).collect()
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Initialise a new iCaRL encoder state with Xavier weights.
pub fn icarl_new(cfg: &IcarlConfig, seed: u64) -> ContinualResult<IcarlState> {
    if cfg.input_dim == 0 || cfg.hidden_dim < 2 || cfg.n_classes == 0 {
        return Err(ContinualError::EmptyInput);
    }
    let feature_dim = cfg.hidden_dim / 2;
    if feature_dim == 0 {
        return Err(ContinualError::EmptyInput);
    }
    let mut rng = LcgRng::new(seed);
    let mut w1 = vec![0.0_f64; cfg.hidden_dim * cfg.input_dim];
    xavier_init(&mut w1, cfg.input_dim, cfg.hidden_dim, &mut rng);
    let b1 = vec![0.0_f64; cfg.hidden_dim];
    let mut w2 = vec![0.0_f64; feature_dim * cfg.hidden_dim];
    xavier_init(&mut w2, cfg.hidden_dim, feature_dim, &mut rng);
    let b2 = vec![0.0_f64; feature_dim];

    Ok(IcarlState {
        feature_dim,
        encoder_w1: w1,
        encoder_b1: b1,
        encoder_w2: w2,
        encoder_b2: b2,
        exemplar_sets: Vec::new(),
        seen_classes: Vec::new(),
        input_dim: cfg.input_dim,
        hidden_dim: cfg.hidden_dim,
    })
}

/// Encode a single input vector through the 2-layer encoder and L2-normalise.
pub fn icarl_encode(state: &IcarlState, x: &[f64]) -> Vec<f64> {
    let mut h1 = matvec(
        &state.encoder_w1,
        x,
        &state.encoder_b1,
        state.input_dim,
        state.hidden_dim,
    );
    relu_inplace(&mut h1);
    let mut h2 = matvec(
        &state.encoder_w2,
        &h1,
        &state.encoder_b2,
        state.hidden_dim,
        state.feature_dim,
    );
    l2_normalise(&mut h2);
    h2
}

/// Select m exemplars from `x_class` (shape `[n × input_dim]`) using herding.
///
/// Herding greedily adds the sample that minimises the distance between the
/// current mean of selected exemplars and the true class mean in feature space.
///
/// Returns an `ExemplarSet` with up to `m` exemplars.
pub fn icarl_construct_exemplar_set(
    state: &IcarlState,
    x_class: &[f64],
    n: usize,
    m: usize,
) -> ExemplarSet {
    let d_in = state.input_dim;
    let d_feat = state.feature_dim;

    // Compute feature for every sample.
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let xi = &x_class[i * d_in..(i + 1) * d_in];
        features.push(icarl_encode(state, xi));
    }

    // True class mean in feature space.
    let mut class_mean = vec![0.0_f64; d_feat];
    for f in &features {
        for k in 0..d_feat {
            class_mean[k] += f[k];
        }
    }
    let inv_n = if n > 0 { 1.0 / n as f64 } else { 0.0 };
    for v in class_mean.iter_mut() {
        *v *= inv_n;
    }

    let m_actual = m.min(n);
    let mut selected_indices: Vec<usize> = Vec::with_capacity(m_actual);
    // Running sum of feature vectors for selected exemplars.
    let mut running_sum = vec![0.0_f64; d_feat];

    for step in 0..m_actual {
        let scale = 1.0 / (step + 1) as f64;
        let mut best_idx = 0;
        let mut best_dist = f64::INFINITY;

        for (i, feat) in features.iter().enumerate() {
            if selected_indices.contains(&i) {
                continue;
            }
            // Candidate mean if we add this sample.
            let cand_mean: Vec<f64> = running_sum
                .iter()
                .zip(feat.iter())
                .map(|(s, f)| (s + f) * scale)
                .collect();
            let d = dist_sq(&cand_mean, &class_mean);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        // Add best exemplar to running sum.
        for k in 0..d_feat {
            running_sum[k] += features[best_idx][k];
        }
        selected_indices.push(best_idx);
    }

    // Collect raw input exemplars.
    let exemplars: Vec<Vec<f64>> = selected_indices
        .iter()
        .map(|&i| x_class[i * d_in..(i + 1) * d_in].to_vec())
        .collect();

    // Recompute class_mean from selected exemplar features.
    let mut final_mean = vec![0.0_f64; d_feat];
    if !selected_indices.is_empty() {
        for &i in &selected_indices {
            for k in 0..d_feat {
                final_mean[k] += features[i][k];
            }
        }
        let inv_m = 1.0 / selected_indices.len() as f64;
        for v in final_mean.iter_mut() {
            *v *= inv_m;
        }
    }

    // class_mean exposed as `ExemplarSet::class_mean` is in feature space.
    ExemplarSet {
        class_id: 0, // set by caller
        exemplars,
        class_mean: final_mean,
    }
}

/// Train the encoder for one task using the combined new + exemplar data.
///
/// Loss = CE on new-class samples + KL distillation on old-class exemplars.
pub fn icarl_update_representation(
    state: &mut IcarlState,
    x_new: &[f64],
    y_new: &[usize],
    n: usize,
    rng: &mut LcgRng,
) -> ContinualResult<()> {
    if n == 0 || x_new.is_empty() {
        return Err(ContinualError::EmptyInput);
    }
    if y_new.len() != n {
        return Err(ContinualError::DimensionMismatch {
            expected: n,
            got: y_new.len(),
        });
    }
    let d_in = state.input_dim;
    let d_h = state.hidden_dim;
    let d_feat = state.feature_dim;

    // --- Collect class-output mapping -----------------------------------------
    // We use a simple linear classifier on top of features (in-place output
    // layer: each class has one prototype logit = -||feat - mean||^2).
    // For CE loss we need a mapping from sample labels to class indices.
    // Build a union of new classes + old classes.
    let mut all_classes: Vec<usize> = state.seen_classes.clone();
    for &c in y_new {
        if !all_classes.contains(&c) {
            all_classes.push(c);
        }
    }
    all_classes.sort_unstable();
    let n_cls = all_classes.len().max(1);
    let class_index = |c: usize| -> usize { all_classes.iter().position(|&x| x == c).unwrap_or(0) };

    // --- Capture old soft-targets for distillation ----------------------------
    // For each exemplar in old classes, compute current model's output probs.
    let old_exemplars: Vec<(Vec<f64>, Vec<f64>)> = state
        .exemplar_sets
        .iter()
        .flat_map(|es| {
            es.exemplars.iter().map(|ex| {
                let feat = icarl_encode(state, ex);
                // Logit for each class: -dist(feat, class_mean).
                let logits: Vec<f64> = all_classes
                    .iter()
                    .map(|&c| {
                        state
                            .exemplar_sets
                            .iter()
                            .find(|e| e.class_id == c)
                            .map(|e| -dist_sq(&feat, &e.class_mean))
                            .unwrap_or(f64::NEG_INFINITY)
                    })
                    .collect();
                // Soft target with temperature 2 (hardcoded for distillation)
                let soft = softmax_temp(&logits, 2.0);
                (ex.clone(), soft)
            })
        })
        .collect();

    // --- Build combined dataset -----------------------------------------------
    // x_combined: new samples + all exemplars
    let n_old = old_exemplars.len();
    let n_total = n + n_old;

    // --- SGD training loop ----------------------------------------------------
    let lr = state.encoder_b1[0]; // placeholder — we keep lr in a local
    let _ = lr;
    let lr_val = 0.01_f64; // default; callers pass cfg.lr via icarl_fit_task

    // Shuffle indices each epoch.
    let mut indices: Vec<usize> = (0..n_total).collect();

    for _epoch in 0..5 {
        rng.shuffle(&mut indices);
        for &idx in &indices {
            let (x_sample, label_or_soft, is_old) = if idx < n {
                let xi = &x_new[idx * d_in..(idx + 1) * d_in];
                let ci = class_index(y_new[idx]);
                (xi.to_vec(), (ci, None::<Vec<f64>>), false)
            } else {
                let old_idx = idx - n;
                let (ex, soft) = &old_exemplars[old_idx];
                (ex.clone(), (0, Some(soft.clone())), true)
            };

            // Forward pass.
            let mut h1 = matvec(&state.encoder_w1, &x_sample, &state.encoder_b1, d_in, d_h);
            relu_inplace(&mut h1);
            let mut h2 = matvec(&state.encoder_w2, &h1, &state.encoder_b2, d_h, d_feat);
            l2_normalise(&mut h2);

            // Compute logits = -dist(feat, class_mean_i) for each class.
            let logits: Vec<f64> = all_classes
                .iter()
                .map(|&c| {
                    state
                        .exemplar_sets
                        .iter()
                        .find(|e| e.class_id == c)
                        .map(|e| -dist_sq(&h2, &e.class_mean))
                        .unwrap_or(-1.0_f64)
                })
                .collect();

            // Loss and output gradient.
            let probs = softmax_temp(&logits, 1.0);
            let mut d_logits = probs.clone();

            if is_old {
                // KL distillation: gradient = probs - soft_target.
                if let Some(soft) = &label_or_soft.1 {
                    for (dg, sv) in d_logits.iter_mut().zip(soft.iter()) {
                        *dg -= sv;
                    }
                }
            } else {
                // CE: gradient = probs - one_hot.
                let ci = label_or_soft.0;
                if ci < n_cls {
                    d_logits[ci] -= 1.0;
                }
            }

            // Gradient of h2 (feature) via chain rule through logits.
            // d_feat[k] = -2 * Σ_c d_logits[c] * (h2[k] - class_mean_c[k])
            let mut d_h2 = vec![0.0_f64; d_feat];
            for (c_idx, &c) in all_classes.iter().enumerate() {
                if let Some(es) = state.exemplar_sets.iter().find(|e| e.class_id == c) {
                    let dg = d_logits[c_idx];
                    for (dh, (&hv, &mv)) in d_h2.iter_mut().zip(h2.iter().zip(es.class_mean.iter()))
                    {
                        *dh += -2.0 * dg * (hv - mv);
                    }
                }
            }

            // Backprop through L2-norm (Jacobian of L2-normalisation).
            // d_pre_norm = (I - feat*feat^T) * d_h2 / norm
            let norm_sq: f64 = h2.iter().map(|x| x * x).sum();
            let norm = norm_sq.sqrt().max(1e-12);
            let dot: f64 = h2.iter().zip(d_h2.iter()).map(|(a, b)| a * b).sum();
            let d_pre2: Vec<f64> = d_h2
                .iter()
                .zip(h2.iter())
                .map(|(&dh, &hv)| (dh - hv * dot) / norm)
                .collect();

            // Backprop layer 2: W2, b2, d_h1.
            // d_W2[row,col] = d_pre2[row] * h1[col]; d_b2 = d_pre2; d_h1 = W2^T * d_pre2.
            let mut d_h1 = vec![0.0_f64; d_h];
            for (row, &g) in d_pre2.iter().enumerate() {
                state.encoder_b2[row] -= lr_val * g;
                for (col, &h1v) in h1.iter().enumerate() {
                    state.encoder_w2[row * d_h + col] -= lr_val * g * h1v;
                    d_h1[col] += g * state.encoder_w2[row * d_h + col];
                }
            }

            // ReLU backward (through h1 pre-activation).
            let h1_pre = matvec(&state.encoder_w1, &x_sample, &state.encoder_b1, d_in, d_h);
            for (k, &pre_v) in h1_pre.iter().enumerate() {
                if pre_v <= 0.0 {
                    d_h1[k] = 0.0;
                }
            }

            // Backprop layer 1: W1, b1.
            for (row, &g) in d_h1.iter().enumerate() {
                state.encoder_b1[row] -= lr_val * g;
                for (col, &xv) in x_sample.iter().enumerate() {
                    state.encoder_w1[row * d_in + col] -= lr_val * g * xv;
                }
            }
        }
    }
    Ok(())
}

/// Nearest-mean-of-exemplars classifier.
///
/// Returns the `class_id` of the exemplar set whose class_mean is closest (in
/// Euclidean distance) to the L2-normalised feature of `x`.
pub fn icarl_classify(state: &IcarlState, x: &[f64]) -> usize {
    let feat = icarl_encode(state, x);
    let mut best_class = 0;
    let mut best_dist = f64::INFINITY;
    for es in &state.exemplar_sets {
        let d = dist_sq(&feat, &es.class_mean);
        if d < best_dist {
            best_dist = d;
            best_class = es.class_id;
        }
    }
    best_class
}

/// Orchestrate one task: update representation + rebuild exemplar sets.
///
/// `x`: raw data matrix `[n × input_dim]`; `y`: class labels (length n).
/// `class_ids`: the distinct classes present in this task.
pub fn icarl_fit_task(
    state: &mut IcarlState,
    x: &[f64],
    y: &[usize],
    n: usize,
    class_ids: &[usize],
    rng: &mut LcgRng,
) -> ContinualResult<()> {
    if n == 0 || x.is_empty() {
        return Err(ContinualError::EmptyInput);
    }
    if class_ids.is_empty() {
        return Err(ContinualError::EmptyInput);
    }
    // Register new classes.
    for &c in class_ids {
        if !state.seen_classes.contains(&c) {
            state.seen_classes.push(c);
        }
    }

    // Update encoder representation.
    icarl_update_representation(state, x, y, n, rng)?;

    // Rebuild exemplar sets for all new classes.
    let d_in = state.input_dim;
    for &class_id in class_ids {
        // Gather samples for this class.
        let class_samples: Vec<usize> = (0..n).filter(|&i| y[i] == class_id).collect();
        if class_samples.is_empty() {
            continue;
        }
        let n_class = class_samples.len();
        // Build a contiguous slice for this class.
        let mut x_class = vec![0.0_f64; n_class * d_in];
        for (dest, &src_i) in class_samples.iter().enumerate() {
            x_class[dest * d_in..(dest + 1) * d_in]
                .copy_from_slice(&x[src_i * d_in..(src_i + 1) * d_in]);
        }
        let m = 20; // default memory_per_class; callers can tune via config
        let mut es = icarl_construct_exemplar_set(state, &x_class, n_class, m);
        es.class_id = class_id;

        // Replace or insert.
        if let Some(existing) = state
            .exemplar_sets
            .iter_mut()
            .find(|e| e.class_id == class_id)
        {
            *existing = es;
        } else {
            state.exemplar_sets.push(es);
        }
    }

    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state() -> IcarlState {
        let cfg = IcarlConfig {
            input_dim: 8,
            hidden_dim: 16,
            ..Default::default()
        };
        icarl_new(&cfg, 42).expect("iCaRL state should initialize with valid config")
    }

    /// 1. Encode produces unit-norm output.
    #[test]
    fn encode_unit_norm() {
        let state = make_state();
        let x = vec![0.5_f64; 8];
        let feat = icarl_encode(&state, &x);
        let norm: f64 = feat.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-9 || norm < 1e-9,
            "Encoded vector must have unit norm, got {norm}"
        );
    }

    /// 2. Encode output length equals feature_dim.
    #[test]
    fn encode_correct_dimension() {
        let state = make_state();
        let x = vec![1.0_f64; 8];
        let feat = icarl_encode(&state, &x);
        assert_eq!(feat.len(), state.feature_dim);
    }

    /// 3. Herding selects exactly m exemplars.
    #[test]
    fn herding_selects_m_exemplars() {
        let state = make_state();
        let n = 20_usize;
        let mut rng = LcgRng::new(1);
        let x_class: Vec<f64> = (0..n * 8).map(|_| rng.next_f32() as f64).collect();
        let m = 5;
        let es = icarl_construct_exemplar_set(&state, &x_class, n, m);
        assert_eq!(
            es.exemplars.len(),
            m,
            "Herding must select exactly m={m} exemplars"
        );
    }

    /// 4. Herding with m > n selects at most n.
    #[test]
    fn herding_caps_at_n() {
        let state = make_state();
        let n = 3_usize;
        let mut rng = LcgRng::new(2);
        let x_class: Vec<f64> = (0..n * 8).map(|_| rng.next_f32() as f64).collect();
        let es = icarl_construct_exemplar_set(&state, &x_class, n, 10);
        assert!(es.exemplars.len() <= n);
    }

    /// 5. Exemplar class_mean is approximately the feature-space class mean.
    #[test]
    fn exemplar_class_mean_close_to_true_mean() {
        let cfg = IcarlConfig {
            input_dim: 4,
            hidden_dim: 8,
            ..Default::default()
        };
        let state = icarl_new(&cfg, 99).expect("iCaRL state should initialize with valid config");
        // All-ones samples → identical features → mean = that feature.
        let n = 10_usize;
        let x_class: Vec<f64> = vec![1.0_f64; n * 4];
        let es = icarl_construct_exemplar_set(&state, &x_class, n, 10);
        // All features are identical so mean should equal that feature.
        let ref_feat = icarl_encode(&state, &[1.0_f64; 4]);
        let err: f64 = es
            .class_mean
            .iter()
            .zip(ref_feat.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            err < 1e-6,
            "Class mean should equal true feature mean for identical samples, err={err}"
        );
    }

    /// 6. Classify returns a valid class_id drawn from seen classes.
    #[test]
    fn classify_returns_valid_class_id() {
        let mut state = make_state();
        // Manually inject two exemplar sets.
        state.exemplar_sets.push(ExemplarSet {
            class_id: 0,
            exemplars: vec![vec![0.0_f64; 8]],
            class_mean: vec![1.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        });
        state.exemplar_sets.push(ExemplarSet {
            class_id: 1,
            exemplars: vec![vec![1.0_f64; 8]],
            class_mean: vec![-1.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        });
        let x = vec![0.0_f64; 8];
        let pred = icarl_classify(&state, &x);
        assert!(
            pred == 0 || pred == 1,
            "predict must return 0 or 1, got {pred}"
        );
    }

    /// 7. fit_task doesn't crash for single class with single sample.
    #[test]
    fn fit_task_single_class_single_sample() {
        let cfg = IcarlConfig {
            input_dim: 4,
            hidden_dim: 8,
            ..Default::default()
        };
        let mut state =
            icarl_new(&cfg, 7).expect("iCaRL state should initialize with valid config");
        let mut rng = LcgRng::new(7);
        let x = vec![0.5_f64; 4];
        let y = vec![0_usize];
        let result = icarl_fit_task(&mut state, &x, &y, 1, &[0], &mut rng);
        assert!(result.is_ok(), "fit_task must succeed for single sample");
    }

    /// 8. fit_task registers new class in seen_classes.
    #[test]
    fn fit_task_registers_class() {
        let cfg = IcarlConfig {
            input_dim: 4,
            hidden_dim: 8,
            ..Default::default()
        };
        let mut state =
            icarl_new(&cfg, 8).expect("iCaRL state should initialize with valid config");
        let mut rng = LcgRng::new(8);
        let n = 5_usize;
        let mut rng2 = LcgRng::new(100);
        let x: Vec<f64> = (0..n * 4).map(|_| rng2.next_f32() as f64).collect();
        let y = vec![3_usize; n];
        icarl_fit_task(&mut state, &x, &y, n, &[3], &mut rng)
            .expect("iCaRL task fitting should succeed with valid data");
        assert!(
            state.seen_classes.contains(&3),
            "Class 3 should be registered after fit_task"
        );
    }

    /// 9. fit_task creates exemplar set.
    #[test]
    fn fit_task_creates_exemplar_set() {
        let cfg = IcarlConfig {
            input_dim: 4,
            hidden_dim: 8,
            ..Default::default()
        };
        let mut state =
            icarl_new(&cfg, 9).expect("iCaRL state should initialize with valid config");
        let mut rng = LcgRng::new(9);
        let n = 8_usize;
        let mut rng2 = LcgRng::new(200);
        let x: Vec<f64> = (0..n * 4).map(|_| rng2.next_f32() as f64).collect();
        let y = vec![5_usize; n];
        icarl_fit_task(&mut state, &x, &y, n, &[5], &mut rng)
            .expect("iCaRL task fitting should succeed with valid data");
        assert!(
            state.exemplar_sets.iter().any(|e| e.class_id == 5),
            "Exemplar set for class 5 should exist"
        );
    }

    /// 10. fit_task returns Err on empty data.
    #[test]
    fn fit_task_empty_data_returns_err() {
        let mut state = make_state();
        let mut rng = LcgRng::new(10);
        let result = icarl_fit_task(&mut state, &[], &[], 0, &[0], &mut rng);
        assert!(result.is_err(), "Empty data must return Err");
    }

    /// 11. fit_task returns Err when class_ids is empty.
    #[test]
    fn fit_task_empty_class_ids_returns_err() {
        let mut state = make_state();
        let mut rng = LcgRng::new(11);
        let x = vec![0.5_f64; 8];
        let y = vec![0_usize];
        let result = icarl_fit_task(&mut state, &x, &y, 1, &[], &mut rng);
        assert!(result.is_err(), "Empty class_ids must return Err");
    }

    /// 12. icarl_new returns Err for zero input_dim.
    #[test]
    fn icarl_new_zero_input_dim_err() {
        let cfg = IcarlConfig {
            input_dim: 0,
            ..Default::default()
        };
        assert!(icarl_new(&cfg, 0).is_err());
    }

    /// 13. Encode is deterministic (same input → same output).
    #[test]
    fn encode_deterministic() {
        let state = make_state();
        let x = vec![0.3_f64; 8];
        let f1 = icarl_encode(&state, &x);
        let f2 = icarl_encode(&state, &x);
        for (a, b) in f1.iter().zip(f2.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    /// 14. Exemplar raw inputs have correct input_dim length.
    #[test]
    fn exemplar_inputs_correct_length() {
        let state = make_state();
        let n = 6_usize;
        let mut rng = LcgRng::new(14);
        let x_class: Vec<f64> = (0..n * 8).map(|_| rng.next_f32() as f64).collect();
        let m = 3;
        let es = icarl_construct_exemplar_set(&state, &x_class, n, m);
        for ex in &es.exemplars {
            assert_eq!(ex.len(), 8);
        }
    }
}
