//! RNN-T (Transducer) loss — Graves 2012, §3.1.
//!
//! Computes the log-likelihood of a target label sequence given a `[T, U+1, V]`
//! joint-network log-probability tensor using log-domain forward (α) recursion.
//! The loss is the negative log-likelihood.

use crate::error::{AudioError, AudioResult};

// ─── Helper ──────────────────────────────────────────────────────────────────

/// Numerically-stable log-sum-exp of two log-domain values.
///
/// `log_add(a, b) = max(a,b) + ln(1 + exp(-|a-b|))`
///
/// Returns `NEG_INFINITY` when both inputs are `NEG_INFINITY`.
#[inline]
fn log_add(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY && b == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let larger = a.max(b);
    let smaller = a.min(b);
    larger + (1.0_f32 + (smaller - larger).exp()).ln()
}

// ─── Public types ─────────────────────────────────────────────────────────────

/// Configuration for RNN-T loss computation.
#[derive(Debug, Clone, Copy)]
pub struct RnntConfig {
    /// Index of the blank symbol in the vocabulary.
    ///
    /// Conventionally `vocab_size - 1`, but any valid index is supported.
    pub blank_id: usize,
}

/// Result of a successful RNN-T loss computation.
#[derive(Debug, Clone)]
pub struct RnntResult {
    /// Negative log-likelihood (the loss value; lower = better).
    pub loss: f32,
    /// Number of input frames T.
    pub n_frames: usize,
    /// Number of target label symbols U.
    pub n_labels: usize,
}

// ─── Core algorithm ──────────────────────────────────────────────────────────

/// Compute the RNN-T loss (Graves 2012, §3.1).
///
/// # Arguments
///
/// - `log_probs`: Log-probabilities from the joint network, shape `[T, U+1, V]`
///   flattened row-major.  Element `(t, u, v)` is accessed as
///   `log_probs[(t * (u_labels + 1) + u) * vocab_size + v]`.
/// - `labels`: Target label sequence of length `u_labels`, each in `[0, V-1]`
///   and not equal to `blank_id`.
/// - `t_frames`: `T` — number of encoder frames.
/// - `u_labels`: `U` — number of target symbols (length of `labels`).
/// - `vocab_size`: `V` — vocabulary size including the blank symbol.
/// - `cfg`: Algorithm configuration (blank symbol index).
///
/// # Returns
///
/// `RnntResult` with `loss = -log P(y | x)`.
///
/// # Errors
///
/// - [`AudioError::EmptyInput`] if `T == 0` or `U == 0`.
/// - [`AudioError::DimensionMismatch`] if `log_probs.len() ≠ T * (U+1) * V`.
/// - [`AudioError::BlankOutOfRange`] if `blank_id ≥ vocab_size`.
/// - [`AudioError::NonFinite`] if the terminal log-probability is NaN.
pub fn rnnt_loss(
    log_probs: &[f32],
    labels: &[usize],
    t_frames: usize,
    u_labels: usize,
    vocab_size: usize,
    cfg: &RnntConfig,
) -> AudioResult<RnntResult> {
    // ── Validation ────────────────────────────────────────────────────────────
    if t_frames == 0 {
        return Err(AudioError::EmptyInput {
            msg: "rnnt_loss: t_frames == 0".into(),
        });
    }
    if u_labels == 0 {
        return Err(AudioError::EmptyInput {
            msg: "rnnt_loss: u_labels == 0".into(),
        });
    }
    if cfg.blank_id >= vocab_size {
        return Err(AudioError::BlankOutOfRange {
            blank: cfg.blank_id,
            vocab: vocab_size,
        });
    }
    let expected_len = t_frames * (u_labels + 1) * vocab_size;
    if log_probs.len() != expected_len {
        return Err(AudioError::DimensionMismatch {
            expected: expected_len,
            got: log_probs.len(),
        });
    }

    let blank = cfg.blank_id;
    let u1 = u_labels + 1; // U+1

    // Accessor: lp(t, u, v).
    let lp = |t: usize, u: usize, v: usize| -> f32 { log_probs[(t * u1 + u) * vocab_size + v] };

    // ── Forward (α) lattice ───────────────────────────────────────────────────
    // α[t * u1 + u] = log P(emit labels y[0..u] using frames x[0..t]).
    //
    // Boundary conditions:
    //   α(0, 0) = 0                                          (start)
    //   α(t, 0) = α(t-1, 0) + lp(t-1, 0, blank)            for t >= 1
    //   α(0, u) = α(0, u-1) + lp(0, u-1, labels[u-1])      for u >= 1
    //
    // General recurrence for t >= 1, u >= 1:
    //   α(t, u) = log_add(
    //       α(t-1, u) + lp(t-1, u, blank),      // emit blank at frame t-1
    //       α(t, u-1) + lp(t, u-1, labels[u-1]) // emit label at frame t
    //   )

    let total = t_frames * u1;
    let mut alpha = vec![f32::NEG_INFINITY; total];

    // Boundary: α(0, 0) = 0.
    alpha[0] = 0.0_f32;

    // Boundary: α(t, 0) for t in 1..T.
    for t in 1..t_frames {
        let prev = alpha[(t - 1) * u1]; // α(t-1, 0)
        alpha[t * u1] = prev + lp(t - 1, 0, blank);
    }

    // Boundary: α(0, u) for u in 1..U+1.
    for u in 1..u1 {
        let prev = alpha[u - 1]; // α(0, u-1)
        alpha[u] = prev + lp(0, u - 1, labels[u - 1]);
    }

    // General recurrence.
    for t in 1..t_frames {
        for u in 1..u1 {
            let from_blank = alpha[(t - 1) * u1 + u] + lp(t - 1, u, blank);
            let from_label = alpha[t * u1 + (u - 1)] + lp(t, u - 1, labels[u - 1]);
            alpha[t * u1 + u] = log_add(from_blank, from_label);
        }
    }

    // Terminal: α(T-1, U) + lp(T-1, U, blank).
    let terminal_alpha = alpha[(t_frames - 1) * u1 + u_labels];
    let terminal_lp = lp(t_frames - 1, u_labels, blank);
    let log_likelihood = terminal_alpha + terminal_lp;

    if log_likelihood.is_nan() {
        return Err(AudioError::NonFinite {
            msg: format!(
                "rnnt_loss: log-likelihood is NaN (terminal_alpha={terminal_alpha}, terminal_lp={terminal_lp})"
            ),
        });
    }

    Ok(RnntResult {
        loss: -log_likelihood,
        n_frames: t_frames,
        n_labels: u_labels,
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a log-probability tensor uniformly: log(1/V) everywhere.
    fn uniform_log_probs(t: usize, u1: usize, v: usize) -> Vec<f32> {
        let lp = -(v as f32).ln();
        vec![lp; t * u1 * v]
    }

    /// Build small random log-probs using a deterministic LCG (no external crate).
    fn pseudo_log_probs(t: usize, u1: usize, v: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let mut next = || -> f32 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as f32 / u32::MAX as f32
        };
        let total = t * u1 * v;
        let mut raw = vec![0.0_f32; total];
        // Build row-by-row so each row (t,u) is normalised.
        for row in 0..t * u1 {
            let start = row * v;
            let mut row_sum = 0.0_f32;
            for j in 0..v {
                let val = next() + 0.01_f32; // ensure positive
                raw[start + j] = val;
                row_sum += val;
            }
            // Convert to log-probs.
            for j in 0..v {
                raw[start + j] = (raw[start + j] / row_sum).ln();
            }
        }
        raw
    }

    /// Config with blank at the last position (V-1).
    fn cfg_last(v: usize) -> RnntConfig {
        RnntConfig { blank_id: v - 1 }
    }

    // ── log_add helper tests ──────────────────────────────────────────────────

    #[test]
    fn rnnt_log_add_symmetry() {
        let a = -1.0_f32;
        let b = -2.0_f32;
        let ab = log_add(a, b);
        let ba = log_add(b, a);
        assert!(
            (ab - ba).abs() < 1e-6,
            "log_add not symmetric: {ab} vs {ba}"
        );
    }

    #[test]
    fn rnnt_log_add_neg_inf() {
        let result = log_add(f32::NEG_INFINITY, f32::NEG_INFINITY);
        assert_eq!(result, f32::NEG_INFINITY);
    }

    #[test]
    fn rnnt_log_add_dominates() {
        // log_add(0.0, -100.0) should be very close to 0.0.
        let result = log_add(0.0_f32, -100.0_f32);
        assert!(result.abs() < 1e-3, "expected ≈ 0.0, got {result}");
    }

    // ── Error cases ───────────────────────────────────────────────────────────

    #[test]
    fn rnnt_err_empty_frames() {
        let lp = vec![0.0_f32; 0];
        let labels = [1_usize];
        let err = rnnt_loss(&lp, &labels, 0, 1, 3, &RnntConfig { blank_id: 2 }).unwrap_err();
        assert!(matches!(err, AudioError::EmptyInput { .. }));
    }

    #[test]
    fn rnnt_err_empty_labels() {
        let lp = uniform_log_probs(3, 1, 3); // U=0 → u1=1
        let err = rnnt_loss(&lp, &[], 3, 0, 3, &RnntConfig { blank_id: 2 }).unwrap_err();
        assert!(matches!(err, AudioError::EmptyInput { .. }));
    }

    #[test]
    fn rnnt_err_dim_mismatch() {
        // Correct size would be T*(U+1)*V = 3*3*4 = 36, but we provide 10.
        let lp = vec![0.0_f32; 10];
        let labels = [1_usize, 2];
        let err = rnnt_loss(&lp, &labels, 3, 2, 4, &RnntConfig { blank_id: 3 }).unwrap_err();
        assert!(matches!(err, AudioError::DimensionMismatch { .. }));
    }

    #[test]
    fn rnnt_err_blank_out_of_range() {
        let v = 4_usize;
        let t = 2_usize;
        let u = 1_usize;
        let lp = uniform_log_probs(t, u + 1, v);
        let labels = [1_usize];
        let err = rnnt_loss(&lp, &labels, t, u, v, &RnntConfig { blank_id: 4 }).unwrap_err();
        assert!(matches!(err, AudioError::BlankOutOfRange { .. }));
    }

    // ── Correctness tests ─────────────────────────────────────────────────────

    #[test]
    fn rnnt_loss_basic_finite() {
        let t = 3_usize;
        let u = 2_usize;
        let v = 5_usize;
        let lp = pseudo_log_probs(t, u + 1, v, 0xDEAD_BEEF);
        let labels = [1_usize, 2];
        let result =
            rnnt_loss(&lp, &labels, t, u, v, &cfg_last(v)).expect("value should be present");
        assert!(
            result.loss.is_finite(),
            "loss must be finite: {}",
            result.loss
        );
    }

    #[test]
    fn rnnt_loss_positive() {
        // The negative log-prob of a sub-1 probability should be > 0.
        let t = 4_usize;
        let u = 2_usize;
        let v = 6_usize;
        let lp = uniform_log_probs(t, u + 1, v);
        let labels = [1_usize, 2];
        let result =
            rnnt_loss(&lp, &labels, t, u, v, &cfg_last(v)).expect("value should be present");
        assert!(
            result.loss > 0.0,
            "loss should be positive; got {}",
            result.loss
        );
    }

    #[test]
    fn rnnt_loss_single_frame_single_label() {
        // T=1, U=1, V=3, blank=2.
        // The tensor has shape [1, 2, 3] (T=1, U+1=2, V=3).
        // lp(0, 0, label=0) + lp(0, 1, blank=2) gives the only valid path.
        let t = 1_usize;
        let u = 1_usize;
        let v = 3_usize;
        let blank = 2_usize;
        // Use simple uniform: each row = log(1/3).
        let lp_val = -(v as f32).ln();
        let lp = vec![lp_val; t * (u + 1) * v];
        let labels = [0_usize];
        let result = rnnt_loss(&lp, &labels, t, u, v, &RnntConfig { blank_id: blank })
            .expect("rnnt_loss should succeed");

        // Manual:
        //   α(0,0) = 0
        //   α(0,1) = 0 + lp(0,0,label=0) = lp_val
        // terminal = α(0,1) + lp(0,1,blank=2)
        //          = lp_val + lp_val
        // loss = -(2 * lp_val)
        let expected_loss = -2.0_f32 * lp_val;
        assert!(
            (result.loss - expected_loss).abs() < 1e-5,
            "expected loss={expected_loss}, got {}",
            result.loss
        );
    }

    #[test]
    fn rnnt_loss_blank_only_path() {
        // Blank-dominant: give blank very high log-prob, label low.
        let t = 3_usize;
        let u = 1_usize;
        let v = 4_usize;
        let blank = 3_usize;
        // Build rows where blank has log(0.9) and other symbols share log(0.1/3).
        let lp_blank = 0.9_f32.ln();
        let lp_other = (0.1_f32 / 3.0).ln();
        let row: Vec<f32> = (0..v)
            .map(|vi| if vi == blank { lp_blank } else { lp_other })
            .collect();
        let lp: Vec<f32> = (0..t * (u + 1)).flat_map(|_| row.clone()).collect();
        let labels = [1_usize];
        let result = rnnt_loss(&lp, &labels, t, u, v, &RnntConfig { blank_id: blank })
            .expect("rnnt_loss should succeed");
        assert!(
            result.loss.is_finite(),
            "loss must be finite: {}",
            result.loss
        );
        assert!(result.loss > 0.0, "loss must be positive");
    }

    #[test]
    fn rnnt_loss_uniform_probs() {
        // With uniform log-probs = log(1/V), the forward variable accumulates
        // predictably.  Loss should be finite and positive for any small config.
        let t = 5_usize;
        let u = 3_usize;
        let v = 4_usize;
        let lp = uniform_log_probs(t, u + 1, v);
        let labels = [1_usize, 2, 3];
        let result =
            rnnt_loss(&lp, &labels, t, u, v, &cfg_last(v)).expect("value should be present");
        assert!(result.loss.is_finite());
        assert!(result.loss > 0.0);
    }

    #[test]
    fn rnnt_loss_scale_invariant_shape() {
        let t = 4_usize;
        let u = 3_usize;
        let v = 6_usize;
        let lp = pseudo_log_probs(t, u + 1, v, 12345);
        let labels = [1_usize, 2, 3];
        let result =
            rnnt_loss(&lp, &labels, t, u, v, &cfg_last(v)).expect("value should be present");
        assert!(result.loss.is_finite());
        assert_eq!(result.n_frames, t);
        assert_eq!(result.n_labels, u);
    }

    #[test]
    fn rnnt_loss_two_frames_two_labels() {
        // T=2, U=2, V=4, blank=3.
        // Manual calculation of the α grid to verify implementation.
        let t = 2_usize;
        let u = 2_usize;
        let v = 4_usize;
        let blank = 3_usize;
        // Use distinct values per row so we can trace manually.
        // Row (t=0,u=0): [lp_a, lp_b, lp_c, lp_bk]
        // Normalise by construction: let each row be uniform for simplicity.
        let lp_val = -(v as f32).ln();
        let lp = uniform_log_probs(t, u + 1, v);

        // Manual:
        //   α(0,0)=0
        //   α(0,1)=0+lp(0,0,label[0]=0) = lp_val + lp_val ... wait, label 0 maps to vocab[0]
        // We use labels = [0,1] (non-blank).
        let labels = [0_usize, 1];
        let result = rnnt_loss(&lp, &labels, t, u, v, &RnntConfig { blank_id: blank })
            .expect("rnnt_loss should succeed");

        // All rows are uniform → loss must be finite and positive.
        assert!(result.loss.is_finite(), "loss finite: {}", result.loss);
        assert!(result.loss > 0.0, "loss positive: {}", result.loss);

        // For uniform: each lp = lp_val. The exact value is deterministic.
        // α(0,0)=0, α(0,1)=lp_val, α(0,2)=2*lp_val
        // α(1,0)=lp_val (from blank), ...
        // terminal = α(1,2) + lp(1,2,blank=3)
        // Verify it matches direct forward recursion result.
        let a00 = 0.0_f32;
        let a01 = a00 + lp_val; // lp(0,0,label[0])
        let a02 = a01 + lp_val; // lp(0,1,label[1])
        let a10 = a00 + lp_val; // lp(0,0,blank)
        let a11 = log_add(a10 + lp_val, a01 + lp_val); // blank or label
        let a12 = log_add(a11 + lp_val, a02 + lp_val); // blank or label
        let terminal = a12 + lp_val; // lp(1,2,blank)
        let expected_loss = -terminal;
        assert!(
            (result.loss - expected_loss).abs() < 1e-5,
            "expected={expected_loss}, got={}",
            result.loss
        );
    }

    #[test]
    fn rnnt_blank_id_zero() {
        // blank_id = 0 (first position in vocabulary).
        let t = 3_usize;
        let u = 1_usize;
        let v = 4_usize;
        let lp = pseudo_log_probs(t, u + 1, v, 999);
        let labels = [2_usize]; // non-blank label
        let result = rnnt_loss(&lp, &labels, t, u, v, &RnntConfig { blank_id: 0 })
            .expect("rnnt_loss should succeed");
        assert!(result.loss.is_finite());
        assert!(result.loss > 0.0);
    }

    #[test]
    fn rnnt_loss_large_t() {
        let t = 20_usize;
        let u = 5_usize;
        let v = 8_usize;
        let lp = pseudo_log_probs(t, u + 1, v, 0xABCD);
        let labels = [1_usize, 2, 3, 4, 5];
        let result =
            rnnt_loss(&lp, &labels, t, u, v, &cfg_last(v)).expect("value should be present");
        assert!(result.loss.is_finite(), "loss should be finite for T=20");
    }

    #[test]
    fn rnnt_loss_deterministic() {
        let t = 4_usize;
        let u = 2_usize;
        let v = 5_usize;
        let lp = pseudo_log_probs(t, u + 1, v, 77777);
        let labels = [1_usize, 2];
        let cfg = cfg_last(v);
        let r1 = rnnt_loss(&lp, &labels, t, u, v, &cfg).expect("rnnt_loss should succeed");
        let r2 = rnnt_loss(&lp, &labels, t, u, v, &cfg).expect("rnnt_loss should succeed");
        assert_eq!(r1.loss, r2.loss, "rnnt_loss must be deterministic");
    }

    #[test]
    fn rnnt_loss_boundary_alphas() {
        // Verify α(0,0)=0 and α(1,0)=lp(0,0,blank) from boundary conditions.
        let t = 3_usize;
        let u = 2_usize;
        let v = 5_usize;
        let blank = v - 1;
        // Use uniform log-probs so lp(t,u,v) = lp_val for all t,u,v.
        let lp = uniform_log_probs(t, u + 1, v);
        let labels = [1_usize, 2];
        // We compute the full forward lattice here to check boundaries.
        // For a minimal sanity check: run the full loss and check it's correct.
        let result = rnnt_loss(&lp, &labels, t, u, v, &RnntConfig { blank_id: blank })
            .expect("rnnt_loss should succeed");

        // Manually compute α(1,0) from boundary rule: 0 + lp(0,0,blank) = lp_val.
        // The only term that uses α(1,0) is in the general cell (1,1):
        //   from_blank = α(0,1) + lp(0,1,blank)
        //   from_label = α(1,0) + lp(1,0,label[0])
        // We can verify the loss is finite, which implies boundary was correctly set.
        assert!(result.loss.is_finite());

        // Verify boundary: the loss with a tiny T should still produce a finite result.
        let t_tiny = 1_usize;
        let u_tiny = 1_usize;
        let lp_tiny = uniform_log_probs(t_tiny, u_tiny + 1, v);
        let labels_tiny = [1_usize];
        let r2 = rnnt_loss(
            &lp_tiny,
            &labels_tiny,
            t_tiny,
            u_tiny,
            v,
            &RnntConfig { blank_id: blank },
        )
        .expect("value should be present");
        // α(0,0)=0, α(0,1)=lp_val, terminal=α(0,1)+lp(0,1,blank)=2*lp_val
        // loss = -2*lp_val = 2*ln(V)
        let expected = 2.0_f32 * (v as f32).ln();
        assert!(
            (r2.loss - expected).abs() < 1e-4,
            "boundary: expected={expected}, got={}",
            r2.loss
        );
    }
}
