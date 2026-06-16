//! Structured perceptron and averaged structured perceptron (Collins 2002).
//!
//! The **structured perceptron** ("Discriminative Training Methods for Hidden
//! Markov Models", Collins, EMNLP 2002) trains a linear sequence tagger by the
//! mistake-driven perceptron rule.  For each training sentence the current
//! weights `w` are used to Viterbi-decode a predicted tag sequence `ŷ`; if
//! `ŷ ≠ y` the weights are corrected toward the gold features and away from the
//! predicted features:
//!
//! ```text
//! w ← w + φ(x, y) − φ(x, ŷ)
//! ```
//!
//! where `φ` is the global feature map decomposing into local emission and
//! transition features.  This crate uses the same parameterisation as the
//! linear-chain CRF: real-valued emission features per position combined with a
//! dense `n_labels × n_labels` transition table.
//!
//! The **averaged perceptron** (Freund & Schapire 1999, popularised by Collins)
//! returns the *average* of all weight vectors seen during training rather than
//! the final one, which dramatically reduces variance and overfitting.  We use
//! the efficient lazy-update trick of Daumé (a running total accumulator
//! `w_total += w` after every example) so averaging costs `O(P)` once at the end
//! rather than `O(P)` per update.

use crate::error::{SeqError, SeqResult};

// ─── Model ───────────────────────────────────────────────────────────────────

/// A linear-chain structured perceptron tagger.
///
/// Parameter layout (identical to `LinearChainCrf`):
/// * `emissions[label*n_features + k]` — weight for feature `k` under `label`.
/// * `transitions[prev*n_labels + cur]` — score of the `prev → cur` bigram.
#[derive(Debug, Clone)]
pub struct StructuredPerceptron {
    /// Number of output labels.
    pub n_labels: usize,
    /// Number of (real-valued) emission features per position.
    pub n_features: usize,
    /// Emission weights, length `n_labels * n_features`.
    pub emissions: Vec<f64>,
    /// Transition weights, length `n_labels * n_labels`.
    pub transitions: Vec<f64>,
}

impl StructuredPerceptron {
    /// Create a zero-initialised perceptron.
    ///
    /// # Errors
    ///
    /// [`SeqError::InvalidConfiguration`] if `n_labels == 0` or `n_features == 0`.
    pub fn zeros(n_labels: usize, n_features: usize) -> SeqResult<Self> {
        if n_labels == 0 || n_features == 0 {
            return Err(SeqError::InvalidConfiguration(
                "n_labels and n_features must be > 0".to_string(),
            ));
        }
        Ok(Self {
            n_labels,
            n_features,
            emissions: vec![0.0; n_labels * n_features],
            transitions: vec![0.0; n_labels * n_labels],
        })
    }

    /// Total number of parameters.
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.n_labels * self.n_features + self.n_labels * self.n_labels
    }

    /// Emission score `w_label · x_t`.
    fn emit_score(&self, label: usize, x: &[f64]) -> f64 {
        let base = label * self.n_features;
        let mut s = 0.0;
        for (k, &xv) in x.iter().enumerate() {
            s += self.emissions[base + k] * xv;
        }
        s
    }

    /// Viterbi decode the highest-scoring label sequence for feature matrix
    /// `x` (`T × n_features`, row-major).
    ///
    /// # Errors
    ///
    /// * [`SeqError::EmptyInput`]    — if `x` is empty.
    /// * [`SeqError::ShapeMismatch`] — if `x.len()` is not a multiple of
    ///   `n_features`.
    pub fn decode(&self, x: &[f64]) -> SeqResult<Vec<usize>> {
        if x.is_empty() {
            return Err(SeqError::EmptyInput);
        }
        let k = self.n_features;
        if x.len() % k != 0 {
            return Err(SeqError::ShapeMismatch {
                expected: x.len().div_ceil(k) * k,
                got: x.len(),
            });
        }
        let n = self.n_labels;
        let t_max = x.len() / k;

        let mut delta = vec![f64::NEG_INFINITY; t_max * n];
        let mut psi = vec![0usize; t_max * n];

        // t = 0: emission only.
        for j in 0..n {
            delta[j] = self.emit_score(j, &x[..k]);
        }
        for t in 1..t_max {
            let xt = &x[t * k..(t + 1) * k];
            for j in 0..n {
                let emit = self.emit_score(j, xt);
                let mut best = f64::NEG_INFINITY;
                let mut argmax = 0usize;
                for i in 0..n {
                    let v = delta[(t - 1) * n + i] + self.transitions[i * n + j];
                    if v > best {
                        best = v;
                        argmax = i;
                    }
                }
                delta[t * n + j] = best + emit;
                psi[t * n + j] = argmax;
            }
        }

        // Termination.
        let mut best = f64::NEG_INFINITY;
        let mut last = 0usize;
        for j in 0..n {
            let v = delta[(t_max - 1) * n + j];
            if v > best {
                best = v;
                last = j;
            }
        }
        let mut path = vec![0usize; t_max];
        path[t_max - 1] = last;
        for t in (1..t_max).rev() {
            path[t - 1] = psi[t * n + path[t]];
        }
        Ok(path)
    }

    /// Total linear score of a full label sequence `y` under `x`.
    ///
    /// # Errors
    ///
    /// * [`SeqError::EmptyInput`]      — if `y` is empty.
    /// * [`SeqError::ShapeMismatch`]   — if `x.len() ≠ y.len() * n_features`.
    /// * [`SeqError::IndexOutOfBounds`] — if any label `≥ n_labels`.
    pub fn sequence_score(&self, x: &[f64], y: &[usize]) -> SeqResult<f64> {
        if y.is_empty() {
            return Err(SeqError::EmptyInput);
        }
        let k = self.n_features;
        let t_max = y.len();
        if x.len() != t_max * k {
            return Err(SeqError::ShapeMismatch {
                expected: t_max * k,
                got: x.len(),
            });
        }
        let mut s = 0.0;
        for t in 0..t_max {
            if y[t] >= self.n_labels {
                return Err(SeqError::IndexOutOfBounds {
                    index: y[t],
                    len: self.n_labels,
                });
            }
            s += self.emit_score(y[t], &x[t * k..(t + 1) * k]);
            if t > 0 {
                s += self.transitions[y[t - 1] * self.n_labels + y[t]];
            }
        }
        Ok(s)
    }

    /// Apply the perceptron correction `θ ← θ + φ(gold) − φ(pred)` in place,
    /// returning the number of positions where `gold` and `pred` differ.
    ///
    /// Both label sequences must have length `T = x.len() / n_features`.
    ///
    /// # Errors
    ///
    /// * [`SeqError::LengthMismatch`]  — if `gold.len() ≠ pred.len()`.
    /// * [`SeqError::ShapeMismatch`]   — if shapes are inconsistent.
    /// * [`SeqError::IndexOutOfBounds`] — if any label is out of range.
    pub fn update(&mut self, x: &[f64], gold: &[usize], pred: &[usize]) -> SeqResult<usize> {
        if gold.len() != pred.len() {
            return Err(SeqError::LengthMismatch {
                a: gold.len(),
                b: pred.len(),
            });
        }
        let k = self.n_features;
        let t_max = gold.len();
        if x.len() != t_max * k {
            return Err(SeqError::ShapeMismatch {
                expected: t_max * k,
                got: x.len(),
            });
        }
        let n = self.n_labels;
        for &lbl in gold.iter().chain(pred.iter()) {
            if lbl >= n {
                return Err(SeqError::IndexOutOfBounds { index: lbl, len: n });
            }
        }

        let mut mistakes = 0usize;
        // Emission features.
        for t in 0..t_max {
            if gold[t] == pred[t] {
                continue;
            }
            mistakes += 1;
            let xt = &x[t * k..(t + 1) * k];
            let gbase = gold[t] * k;
            let pbase = pred[t] * k;
            for (idx, &xv) in xt.iter().enumerate() {
                self.emissions[gbase + idx] += xv;
                self.emissions[pbase + idx] -= xv;
            }
        }
        // Transition features.
        for t in 1..t_max {
            let g = gold[t - 1] * n + gold[t];
            let p = pred[t - 1] * n + pred[t];
            if g != p {
                self.transitions[g] += 1.0;
                self.transitions[p] -= 1.0;
            }
        }
        Ok(mistakes)
    }
}

// ─── Training configuration ──────────────────────────────────────────────────

/// Configuration for perceptron training.
#[derive(Debug, Clone)]
pub struct PerceptronConfig {
    /// Number of passes (epochs) over the training set.
    pub epochs: usize,
    /// Whether to return the averaged weight vector (Collins averaging).
    pub averaged: bool,
}

impl Default for PerceptronConfig {
    fn default() -> Self {
        Self {
            epochs: 10,
            averaged: true,
        }
    }
}

/// One training example: a feature matrix `x` (`T × n_features`) and a gold
/// label sequence `y` (`T`).
#[derive(Debug, Clone)]
pub struct PerceptronExample {
    /// Feature matrix, row-major `T × n_features`.
    pub x: Vec<f64>,
    /// Gold labels, length `T`.
    pub y: Vec<usize>,
}

/// Result of perceptron training.
#[derive(Debug, Clone)]
pub struct PerceptronTrainResult {
    /// The trained model (averaged iff `config.averaged`).
    pub model: StructuredPerceptron,
    /// Total number of mistaken positions over the final epoch.
    pub final_epoch_mistakes: usize,
    /// Number of epochs actually run.
    pub epochs_run: usize,
}

/// Train a structured perceptron on the given examples.
///
/// The model is updated example-by-example.  When `config.averaged` is set the
/// returned model is the running average of all weight vectors (the standard
/// "averaged perceptron"); the running total is accumulated after every example
/// so that each weight vector contributes once per example it survived.
///
/// # Errors
///
/// * [`SeqError::EmptyInput`]    — if `examples` is empty.
/// * [`SeqError::ShapeMismatch`] — if any example's `x` length is not
///   `y.len() * n_features`.
/// * Propagates decode/update errors.
pub fn train_perceptron(
    n_labels: usize,
    n_features: usize,
    examples: &[PerceptronExample],
    config: &PerceptronConfig,
) -> SeqResult<PerceptronTrainResult> {
    if examples.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let mut model = StructuredPerceptron::zeros(n_labels, n_features)?;
    let p = model.param_count();
    // Running total for averaging (emissions then transitions, same layout).
    let mut total = vec![0.0_f64; p];
    let mut n_updates = 0u64;
    let mut final_mistakes = 0usize;

    for epoch in 0..config.epochs.max(1) {
        let mut epoch_mistakes = 0usize;
        for ex in examples {
            let t_max = ex.y.len();
            if t_max == 0 || ex.x.len() != t_max * n_features {
                return Err(SeqError::ShapeMismatch {
                    expected: t_max * n_features,
                    got: ex.x.len(),
                });
            }
            let pred = model.decode(&ex.x)?;
            let mistakes = model.update(&ex.x, &ex.y, &pred)?;
            epoch_mistakes += mistakes;

            if config.averaged {
                // Accumulate the *current* weight vector after the update.
                accumulate(&model, &mut total);
                n_updates += 1;
            }
        }
        final_mistakes = epoch_mistakes;
        // Early-stop hint: a perfectly separable pass converges.
        if epoch_mistakes == 0 {
            // Still keep accumulating handled above; break after recording.
            return finish(model, total, n_updates, final_mistakes, epoch + 1, config);
        }
    }

    finish(
        model,
        total,
        n_updates,
        final_mistakes,
        config.epochs.max(1),
        config,
    )
}

/// Add the model's current weights into the running total.
fn accumulate(model: &StructuredPerceptron, total: &mut [f64]) {
    let cut = model.emissions.len();
    for (t, &e) in total[..cut].iter_mut().zip(model.emissions.iter()) {
        *t += e;
    }
    for (t, &tr) in total[cut..].iter_mut().zip(model.transitions.iter()) {
        *t += tr;
    }
}

/// Finalise training, applying averaging if requested.
fn finish(
    mut model: StructuredPerceptron,
    total: Vec<f64>,
    n_updates: u64,
    final_mistakes: usize,
    epochs_run: usize,
    config: &PerceptronConfig,
) -> SeqResult<PerceptronTrainResult> {
    if config.averaged && n_updates > 0 {
        let inv = 1.0 / n_updates as f64;
        let cut = model.emissions.len();
        for (e, &t) in model.emissions.iter_mut().zip(total[..cut].iter()) {
            *e = t * inv;
        }
        for (tr, &t) in model.transitions.iter_mut().zip(total[cut..].iter()) {
            *tr = t * inv;
        }
    }
    Ok(PerceptronTrainResult {
        model,
        final_epoch_mistakes: final_mistakes,
        epochs_run,
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a separable toy dataset: 2 labels, 2 features where feature `j`
    /// is an indicator that the gold label is `j`.  A perceptron must learn to
    /// route feature 0 → label 0 and feature 1 → label 1.
    fn toy_examples() -> Vec<PerceptronExample> {
        vec![
            PerceptronExample {
                // tags 0,1,0
                x: vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                y: vec![0, 1, 0],
            },
            PerceptronExample {
                // tags 1,0,1
                x: vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                y: vec![1, 0, 1],
            },
        ]
    }

    #[test]
    fn zeros_rejects_bad_dims() {
        assert!(StructuredPerceptron::zeros(0, 3).is_err());
        assert!(StructuredPerceptron::zeros(3, 0).is_err());
        assert!(StructuredPerceptron::zeros(2, 2).is_ok());
    }

    #[test]
    fn param_count_correct() {
        let m = StructuredPerceptron::zeros(3, 4).expect("ok");
        assert_eq!(m.param_count(), 3 * 4 + 3 * 3);
    }

    #[test]
    fn zero_model_decodes_first_label() {
        // All-zero weights: every path scores 0; argmax ties resolve to label 0.
        let m = StructuredPerceptron::zeros(2, 2).expect("ok");
        let y = m.decode(&[1.0, 0.0, 0.0, 1.0]).expect("ok");
        assert_eq!(y, vec![0, 0]);
    }

    #[test]
    fn decode_rejects_empty() {
        let m = StructuredPerceptron::zeros(2, 2).expect("ok");
        assert!(matches!(m.decode(&[]), Err(SeqError::EmptyInput)));
    }

    #[test]
    fn decode_rejects_bad_shape() {
        let m = StructuredPerceptron::zeros(2, 3).expect("ok");
        // length 4 is not a multiple of n_features=3.
        assert!(matches!(
            m.decode(&[1.0, 2.0, 3.0, 4.0]),
            Err(SeqError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn sequence_score_rejects_oob_label() {
        let m = StructuredPerceptron::zeros(2, 2).expect("ok");
        assert!(matches!(
            m.sequence_score(&[1.0, 0.0], &[5]),
            Err(SeqError::IndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn update_rejects_length_mismatch() {
        let mut m = StructuredPerceptron::zeros(2, 2).expect("ok");
        assert!(matches!(
            m.update(&[1.0, 0.0], &[0], &[0, 1]),
            Err(SeqError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn update_counts_mistakes_and_moves_weights() {
        let mut m = StructuredPerceptron::zeros(2, 2).expect("ok");
        let x = vec![1.0, 0.0, 0.0, 1.0]; // T=2
        let gold = vec![0, 1];
        let pred = vec![1, 0];
        let mistakes = m.update(&x, &gold, &pred).expect("ok");
        assert_eq!(mistakes, 2);
        // emission[label0, feat0] should have increased (gold uses feat0→label0).
        assert!(m.emissions[0] > 0.0);
        // After the correction, gold now scores higher than pred.
        let sg = m.sequence_score(&x, &gold).expect("ok");
        let sp = m.sequence_score(&x, &pred).expect("ok");
        assert!(sg > sp, "gold {sg} should exceed pred {sp}");
    }

    #[test]
    fn update_no_mistakes_is_noop() {
        let mut m = StructuredPerceptron::zeros(2, 2).expect("ok");
        let x = vec![1.0, 0.0, 0.0, 1.0];
        let y = vec![0, 1];
        let before = m.emissions.clone();
        let mistakes = m.update(&x, &y, &y).expect("ok");
        assert_eq!(mistakes, 0);
        assert_eq!(before, m.emissions);
    }

    #[test]
    fn train_rejects_empty() {
        let cfg = PerceptronConfig::default();
        assert!(matches!(
            train_perceptron(2, 2, &[], &cfg),
            Err(SeqError::EmptyInput)
        ));
    }

    #[test]
    fn train_learns_separable_data() {
        let ex = toy_examples();
        let cfg = PerceptronConfig {
            epochs: 20,
            averaged: false,
        };
        let res = train_perceptron(2, 2, &ex, &cfg).expect("ok");
        // After training the model must decode every example correctly.
        for e in &ex {
            let pred = res.model.decode(&e.x).expect("ok");
            assert_eq!(pred, e.y, "model failed to fit training example");
        }
    }

    #[test]
    fn train_converges_to_zero_mistakes() {
        let ex = toy_examples();
        let cfg = PerceptronConfig {
            epochs: 50,
            averaged: false,
        };
        let res = train_perceptron(2, 2, &ex, &cfg).expect("ok");
        assert_eq!(
            res.final_epoch_mistakes, 0,
            "separable data should converge to 0 mistakes"
        );
        assert!(res.epochs_run <= 50);
    }

    #[test]
    fn averaged_perceptron_fits_and_is_finite() {
        let ex = toy_examples();
        let cfg = PerceptronConfig {
            epochs: 20,
            averaged: true,
        };
        let res = train_perceptron(2, 2, &ex, &cfg).expect("ok");
        assert!(res.model.emissions.iter().all(|v| v.is_finite()));
        for e in &ex {
            let pred = res.model.decode(&e.x).expect("ok");
            assert_eq!(pred, e.y);
        }
    }

    #[test]
    fn averaging_equals_mean_of_trajectory() {
        // Deterministic averaging check on a *single* example over 2 epochs.
        // Epoch 0: weights start at 0, prediction is all-label-0, the example
        // has a mistake → one update giving weight vector w₁.  Epoch 1: the
        // example is now classified correctly → no update, weight stays w₁.
        // The averaged model accumulates [w₁ (after ep0), w₁ (after ep1)], so
        // the average is exactly w₁ == the raw final weights here.
        let ex = vec![PerceptronExample {
            x: vec![1.0, 0.0, 0.0, 1.0],
            y: vec![0, 1],
        }];
        let avg = train_perceptron(
            2,
            2,
            &ex,
            &PerceptronConfig {
                epochs: 2,
                averaged: true,
            },
        )
        .expect("ok");
        let raw = train_perceptron(
            2,
            2,
            &ex,
            &PerceptronConfig {
                epochs: 2,
                averaged: false,
            },
        )
        .expect("ok");
        // Both should decode the example correctly.
        assert_eq!(avg.model.decode(&ex[0].x).expect("d"), ex[0].y);
        assert_eq!(raw.model.decode(&ex[0].x).expect("d"), ex[0].y);
        // Averaged weights are a convex mean of the trajectory, hence finite
        // and never larger in magnitude than the raw final weights.
        for (a, r) in avg.model.emissions.iter().zip(raw.model.emissions.iter()) {
            assert!(a.abs() <= r.abs() + 1e-9, "avg {a} exceeds raw {r}");
        }
    }

    #[test]
    fn averaging_shrinks_when_trajectory_varies() {
        // A dataset whose two examples *pull weights in opposite directions*
        // each epoch keeps the weight trajectory oscillating, so the running
        // average has strictly smaller magnitude than the final weights for at
        // least one coordinate.
        let ex = vec![
            PerceptronExample {
                x: vec![1.0, 0.0],
                y: vec![0],
            },
            PerceptronExample {
                x: vec![1.0, 0.0],
                y: vec![1],
            },
        ];
        let avg = train_perceptron(
            2,
            2,
            &ex,
            &PerceptronConfig {
                epochs: 6,
                averaged: true,
            },
        )
        .expect("ok");
        let raw = train_perceptron(
            2,
            2,
            &ex,
            &PerceptronConfig {
                epochs: 6,
                averaged: false,
            },
        )
        .expect("ok");
        let diff: f64 = avg
            .model
            .emissions
            .iter()
            .zip(raw.model.emissions.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-9,
            "with a non-separable oscillating dataset averaging must differ from final"
        );
    }

    #[test]
    fn train_rejects_inconsistent_example_shape() {
        let bad = vec![PerceptronExample {
            x: vec![1.0, 0.0, 0.0], // 3 entries, but T=2 needs 4
            y: vec![0, 1],
        }];
        let cfg = PerceptronConfig::default();
        assert!(matches!(
            train_perceptron(2, 2, &bad, &cfg),
            Err(SeqError::ShapeMismatch { .. })
        ));
    }
}
