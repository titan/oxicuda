//! Zero-cost proxies for predictor-free architecture ranking.
//!
//! These proxies score a candidate architecture from signals collected on a
//! *single* untrained (or barely-initialised) network at one minibatch — no
//! training loop, no held-out evaluation. They are forward-only (plus, for some,
//! a single backward pass) and therefore orders of magnitude cheaper than the
//! train-then-evaluate cycle used by classic NAS.
//!
//! All scores follow the convention "higher is better"; [`rank_architectures`]
//! turns a slice of scores into a descending ranking.
//!
//! Implemented proxies:
//! - [`naswot_score`] — NASWOT logdet of the binary-activation kernel
//!   (Mellor et al., "Neural Architecture Search without Training", ICML 2021).
//! - [`snip_score`] — SNIP connection sensitivity `Σ|g·w|`
//!   (Lee et al., "SNIP: Single-shot Network Pruning", ICLR 2019).
//! - [`grasp_score`] — GraSP gradient-signal preservation `Σ −(Hg)·w`
//!   (Wang et al., "Picking Winning Tickets Before Training by Preserving
//!   Gradient Flow", ICLR 2020).
//! - [`synflow_score`] — SynFlow path-norm saliency `Σ|w·∂R/∂w|`
//!   (Tanaka et al., "Pruning neural networks without any data by iteratively
//!   conserving synaptic flow", NeurIPS 2020).
//!
//! The caller is responsible for producing the activation / gradient / weight
//! vectors; this module performs no autodiff and runs no network.

use crate::error::{NasError, NasResult};

/// Ridge added to the diagonal of the NASWOT kernel before the Cholesky
/// factorisation.
///
/// The raw kernel `K` can be singular (e.g. when several samples produce the
/// *same* binary activation code, making rows of `K` linearly dependent), so a
/// strictly-positive ridge `λI` is required to keep `K + λI` positive definite
/// and `log|K + λI|` finite. The value `1e-3` matches the small jitter used by
/// the reference NASWOT implementation; it is large enough to survive `f32`
/// rounding on a rank-deficient kernel yet small enough not to dominate the
/// logdet for well-conditioned kernels.
pub const NASWOT_RIDGE: f32 = 1e-3;

// ─── NASWOT ────────────────────────────────────────────────────────────────────

/// NASWOT score: the log-determinant of the binary-activation kernel.
///
/// `activation_patterns[i]` is the binary ReLU activation code of minibatch
/// sample `i` across `N_units` rectifier units (`true` = unit active,
/// `false` = unit inactive). For a batch of `n` samples the kernel is the
/// `n × n` matrix
///
/// ```text
/// K_ij = N_units − HammingDistance(c_i, c_j)
///      = #{ units where c_i and c_j agree }
/// ```
///
/// which equals the number of units sharing the same activation state between
/// samples `i` and `j` (Mellor et al., 2021, eq. 2, written there in terms of
/// `N_A` = the number of agreeing-active plus agreeing-inactive units).
///
/// The score is `log|K + λI|`, with `λ = `[`NASWOT_RIDGE`]. Architectures that
/// induce *distinct* activation regions for distinct inputs (a near-diagonal,
/// well-conditioned `K`) score higher than ones that collapse many inputs to
/// the same code (a near-rank-1 `K` whose logdet is dominated by the ridge).
///
/// The determinant is computed from a self-contained ridge-stabilised Cholesky
/// factorisation `K + λI = L Lᵀ`, returning `log|K + λI| = 2 Σ_i ln L_ii`.
///
/// # Errors
/// - [`NasError::EmptySearchSpace`] if `activation_patterns` is empty.
/// - [`NasError::DimensionMismatch`] if the inner activation codes have
///   differing lengths.
pub fn naswot_score(activation_patterns: &[Vec<bool>]) -> NasResult<f32> {
    let n = activation_patterns.len();
    if n == 0 {
        return Err(NasError::EmptySearchSpace);
    }
    let n_units = activation_patterns[0].len();
    for code in activation_patterns {
        if code.len() != n_units {
            return Err(NasError::DimensionMismatch {
                expected: n_units,
                got: code.len(),
            });
        }
    }

    // Build the symmetric agreement kernel K (n × n) with the ridge already on
    // the diagonal so the in-place Cholesky factors K + λI directly.
    let mut k_mat = vec![0.0_f32; n * n];
    for i in 0..n {
        let ci = &activation_patterns[i];
        for j in i..n {
            let cj = &activation_patterns[j];
            let mut agree = 0usize;
            for (&a, &b) in ci.iter().zip(cj.iter()) {
                if a == b {
                    agree += 1;
                }
            }
            let mut v = agree as f32;
            if i == j {
                v += NASWOT_RIDGE;
            }
            k_mat[i * n + j] = v;
            k_mat[j * n + i] = v;
        }
    }

    logdet_spd_cholesky(&k_mat, n)
}

/// Log-determinant of a symmetric positive-definite matrix `A` (`n × n`,
/// row-major) via Cholesky: `A = L Lᵀ`, `log|A| = 2 Σ_i ln L_ii`.
///
/// Self-contained (no external linear-algebra dependency). The matrix is assumed
/// to already carry any required ridge on its diagonal.
///
/// # Errors
/// - [`NasError::DimensionMismatch`] if `a.len() != n * n`.
/// - [`NasError::Internal`] if a non-positive pivot is encountered (the matrix
///   is not positive definite even after the caller's regularisation).
fn logdet_spd_cholesky(a: &[f32], n: usize) -> NasResult<f32> {
    if a.len() != n * n {
        return Err(NasError::DimensionMismatch {
            expected: n * n,
            got: a.len(),
        });
    }
    if n == 0 {
        return Ok(0.0);
    }
    // Lower-triangular Cholesky factor, stored dense row-major.
    let mut l = vec![0.0_f32; n * n];
    let mut logdet = 0.0_f64;
    for i in 0..n {
        for j in 0..=i {
            // sum_{k<j} L_ik L_jk
            let mut sum = 0.0_f32;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let diag = a[i * n + i] - sum;
                if diag <= 0.0 {
                    return Err(NasError::Internal(
                        "NASWOT kernel not positive definite (non-positive Cholesky pivot)".into(),
                    ));
                }
                let l_ii = diag.sqrt();
                l[i * n + j] = l_ii;
                logdet += 2.0 * (l_ii as f64).ln();
            } else {
                let l_jj = l[j * n + j];
                // l_jj > 0 guaranteed: column j's diagonal was set in a prior
                // (i == j) iteration before any off-diagonal of column j is used.
                l[i * n + j] = (a[i * n + j] - sum) / l_jj;
            }
        }
    }
    Ok(logdet as f32)
}

// ─── SNIP ──────────────────────────────────────────────────────────────────────

/// SNIP connection-sensitivity score `Σ_i |g_i · w_i|` (Lee et al., 2019).
///
/// `grads[i]` is the loss gradient `∂L/∂w_i` and `weights[i]` the corresponding
/// weight, both evaluated on one minibatch of the untrained network. The
/// saliency `|g·w|` approximates the change in loss when connection `i` is
/// removed; summing it gives a single architecture score (higher = more
/// salient connections = better).
///
/// # Errors
/// - [`NasError::EmptySearchSpace`] if either slice is empty.
/// - [`NasError::DimensionMismatch`] if `grads.len() != weights.len()`.
pub fn snip_score(grads: &[f32], weights: &[f32]) -> NasResult<f32> {
    if grads.is_empty() || weights.is_empty() {
        return Err(NasError::EmptySearchSpace);
    }
    if grads.len() != weights.len() {
        return Err(NasError::DimensionMismatch {
            expected: grads.len(),
            got: weights.len(),
        });
    }
    let score: f32 = grads
        .iter()
        .zip(weights.iter())
        .map(|(&g, &w)| (g * w).abs())
        .sum();
    Ok(score)
}

// ─── GraSP ───────────────────────────────────────────────────────────────────────

/// GraSP gradient-signal-preservation score `Σ_i −(Hg)_i · w_i` (Wang et al., 2020).
///
/// `hessian_grad_product[i]` is the `i`-th component of the Hessian–gradient
/// product `Hg` (the directional curvature of the loss along the gradient), and
/// `weights[i]` the corresponding weight. GraSP measures how removing a
/// connection changes the *gradient flow*; its saliency is `−(Hg)·w`, with the
/// minus sign so that connections whose removal would *increase* gradient flow
/// score positively. The architecture score is the sum of per-connection
/// saliencies (higher = better preserved gradient flow).
///
/// # Errors
/// - [`NasError::EmptySearchSpace`] if either slice is empty.
/// - [`NasError::DimensionMismatch`] if the slice lengths differ.
pub fn grasp_score(hessian_grad_product: &[f32], weights: &[f32]) -> NasResult<f32> {
    if hessian_grad_product.is_empty() || weights.is_empty() {
        return Err(NasError::EmptySearchSpace);
    }
    if hessian_grad_product.len() != weights.len() {
        return Err(NasError::DimensionMismatch {
            expected: hessian_grad_product.len(),
            got: weights.len(),
        });
    }
    let score: f32 = hessian_grad_product
        .iter()
        .zip(weights.iter())
        .map(|(&hg, &w)| -(hg * w))
        .sum();
    Ok(score)
}

// ─── SynFlow ───────────────────────────────────────────────────────────────────

/// SynFlow path-norm saliency `Σ_i |w_i · synflow_grad_i|` (Tanaka et al., 2020).
///
/// SynFlow defines the data-free objective `R = 𝟙ᵀ (∏_l |W_l|) 𝟙` (the sum over
/// all input→output paths of the product of absolute weights). `synflow_grads[i]`
/// is `∂R/∂w_i` evaluated with *positive* weights, and the per-parameter
/// saliency is `w_i · ∂R/∂w_i`. For positive weights and positive grads the
/// product is positive; we take the absolute value for robustness and sum to a
/// single architecture score (higher = better synaptic-flow preservation).
///
/// # Errors
/// - [`NasError::EmptySearchSpace`] if either slice is empty.
/// - [`NasError::DimensionMismatch`] if the slice lengths differ.
pub fn synflow_score(weights: &[f32], synflow_grads: &[f32]) -> NasResult<f32> {
    if weights.is_empty() || synflow_grads.is_empty() {
        return Err(NasError::EmptySearchSpace);
    }
    if weights.len() != synflow_grads.len() {
        return Err(NasError::DimensionMismatch {
            expected: weights.len(),
            got: synflow_grads.len(),
        });
    }
    let score: f32 = weights
        .iter()
        .zip(synflow_grads.iter())
        .map(|(&w, &g)| (w * g).abs())
        .sum();
    Ok(score)
}

// ─── Proxy selector + ranking ────────────────────────────────────────────────────

/// The available zero-cost proxy metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZeroCostProxy {
    /// NASWOT logdet kernel score ([`naswot_score`]).
    Naswot,
    /// SNIP connection sensitivity ([`snip_score`]).
    Snip,
    /// GraSP gradient-signal preservation ([`grasp_score`]).
    Grasp,
    /// SynFlow path-norm saliency ([`synflow_score`]).
    Synflow,
}

impl ZeroCostProxy {
    /// All four proxy variants in canonical order.
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[Self::Naswot, Self::Snip, Self::Grasp, Self::Synflow]
    }

    /// Human-readable name of the proxy.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Naswot => "naswot",
            Self::Snip => "snip",
            Self::Grasp => "grasp",
            Self::Synflow => "synflow",
        }
    }
}

/// Rank architectures by their zero-cost scores, best first.
///
/// Returns the indices `0..scores.len()` sorted so that the architecture with
/// the **highest** score comes first ("higher is better" for every proxy in
/// this module). Ties are broken by **lower original index first**, giving a
/// fully deterministic, stable ordering. Non-finite scores (`NaN`) are treated
/// as the smallest possible value and therefore sink to the end.
#[must_use]
pub fn rank_architectures(scores: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        let sa = scores[a];
        let sb = scores[b];
        // Descending by score; NaN sinks to the bottom.
        match sb.partial_cmp(&sa) {
            Some(ord) if ord != std::cmp::Ordering::Equal => ord,
            Some(_) => a.cmp(&b), // equal scores → lower index first
            None => {
                // At least one is NaN. Order finite-before-NaN, else by index.
                match (sa.is_nan(), sb.is_nan()) {
                    (false, true) => std::cmp::Ordering::Less,
                    (true, false) => std::cmp::Ordering::Greater,
                    _ => a.cmp(&b),
                }
            }
        }
    });
    idx
}

// ─── Tests ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force log-determinant via LU with partial pivoting, for cross-checks.
    fn logdet_reference(a: &[f32], n: usize) -> f64 {
        let mut m: Vec<f64> = a.iter().map(|&v| v as f64).collect();
        let mut logdet = 0.0_f64;
        for i in 0..n {
            // partial pivot
            let mut piv = i;
            let mut best = m[i * n + i].abs();
            for r in (i + 1)..n {
                let v = m[r * n + i].abs();
                if v > best {
                    best = v;
                    piv = r;
                }
            }
            if piv != i {
                for c in 0..n {
                    m.swap(i * n + c, piv * n + c);
                }
            }
            let d = m[i * n + i];
            logdet += d.abs().ln();
            for r in (i + 1)..n {
                let f = m[r * n + i] / d;
                for c in i..n {
                    m[r * n + c] -= f * m[i * n + c];
                }
            }
        }
        logdet
    }

    #[test]
    fn naswot_orthogonal_beats_correlated() {
        // 4 samples, 4 units. "Orthogonal" → distinct one-hot codes.
        let orthogonal = vec![
            vec![true, false, false, false],
            vec![false, true, false, false],
            vec![false, false, true, false],
            vec![false, false, false, true],
        ];
        // Highly correlated → nearly the same code.
        let correlated = vec![
            vec![true, true, true, false],
            vec![true, true, true, false],
            vec![true, true, true, false],
            vec![true, true, false, false],
        ];
        let s_orth = naswot_score(&orthogonal).expect("orthogonal naswot");
        let s_corr = naswot_score(&correlated).expect("correlated naswot");
        assert!(
            s_orth > s_corr,
            "orthogonal {s_orth} should exceed correlated {s_corr}"
        );
    }

    #[test]
    fn naswot_identical_patterns_finite() {
        // All identical → raw K is rank-1 (singular). Ridge must keep logdet finite.
        let identical = vec![vec![true, false, true, true]; 5];
        let s = naswot_score(&identical).expect("identical naswot must be finite");
        assert!(
            s.is_finite(),
            "logdet must be finite for singular kernel: {s}"
        );
    }

    #[test]
    fn naswot_distinct_beats_identical() {
        let distinct = vec![
            vec![true, true, false, false],
            vec![false, false, true, true],
            vec![true, false, true, false],
        ];
        let identical = vec![vec![true, true, false, false]; 3];
        let s_distinct = naswot_score(&distinct).expect("distinct");
        let s_identical = naswot_score(&identical).expect("identical");
        assert!(
            s_distinct > s_identical,
            "distinct {s_distinct} should beat identical {s_identical}"
        );
    }

    #[test]
    fn naswot_logdet_matches_reference() {
        // Recompute K + λI exactly and compare logdet to an independent LU.
        let patterns = vec![
            vec![true, false, true, false, true],
            vec![false, true, true, false, false],
            vec![true, true, false, false, true],
        ];
        let n = patterns.len();
        let n_units = patterns[0].len();
        let mut k = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let agree = (0..n_units)
                    .filter(|&u| patterns[i][u] == patterns[j][u])
                    .count();
                k[i * n + j] = agree as f32 + if i == j { NASWOT_RIDGE } else { 0.0 };
            }
        }
        let reference = logdet_reference(&k, n);
        let got = naswot_score(&patterns).expect("naswot") as f64;
        assert!(
            (got - reference).abs() < 1e-2,
            "naswot {got} vs reference logdet {reference}"
        );
    }

    #[test]
    fn naswot_empty_errors() {
        let empty: Vec<Vec<bool>> = Vec::new();
        assert_eq!(naswot_score(&empty), Err(NasError::EmptySearchSpace));
    }

    #[test]
    fn naswot_ragged_errors() {
        let ragged = vec![vec![true, false], vec![true, false, true]];
        assert!(matches!(
            naswot_score(&ragged),
            Err(NasError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn snip_matches_hand_computed() {
        // |(-2)*3| + |1*(-4)| + |0*5| + |2*0.5| = 6 + 4 + 0 + 1 = 11
        let grads = [-2.0, 1.0, 0.0, 2.0];
        let weights = [3.0, -4.0, 5.0, 0.5];
        let s = snip_score(&grads, &weights).expect("snip");
        assert!((s - 11.0).abs() < 1e-6, "snip = {s}");
    }

    #[test]
    fn snip_zero_weights_is_zero() {
        let grads = [1.0, 2.0, 3.0];
        let weights = [0.0, 0.0, 0.0];
        let s = snip_score(&grads, &weights).expect("snip");
        assert_eq!(s, 0.0);
    }

    #[test]
    fn snip_length_mismatch_errors() {
        let grads = [1.0, 2.0];
        let weights = [1.0, 2.0, 3.0];
        assert!(matches!(
            snip_score(&grads, &weights),
            Err(NasError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn snip_empty_errors() {
        assert_eq!(snip_score(&[], &[]), Err(NasError::EmptySearchSpace));
    }

    #[test]
    fn grasp_sign_convention() {
        // score = Σ -(Hg)*w. With Hg = [1, -2], w = [3, 4]:
        // -(1*3) - (-2*4) = -3 + 8 = 5.
        let hg = [1.0, -2.0];
        let w = [3.0, 4.0];
        let s = grasp_score(&hg, &w).expect("grasp");
        assert!((s - 5.0).abs() < 1e-6, "grasp = {s}");
    }

    #[test]
    fn grasp_positive_when_removal_helps_flow() {
        // All -(Hg)*w positive when Hg and w have opposite signs.
        let hg = [-1.0, -2.0, -0.5];
        let w = [1.0, 2.0, 4.0];
        let s = grasp_score(&hg, &w).expect("grasp");
        assert!(s > 0.0, "grasp = {s}");
    }

    #[test]
    fn grasp_length_mismatch_errors() {
        assert!(matches!(
            grasp_score(&[1.0], &[1.0, 2.0]),
            Err(NasError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn synflow_positive_for_positive_inputs() {
        let weights = [0.5, 1.0, 2.0, 0.25];
        let grads = [1.0, 0.5, 0.1, 4.0];
        let s = synflow_score(&weights, &grads).expect("synflow");
        assert!(s > 0.0, "synflow = {s}");
        // hand: 0.5 + 0.5 + 0.2 + 1.0 = 2.2
        assert!((s - 2.2).abs() < 1e-6, "synflow = {s}");
    }

    #[test]
    fn synflow_length_mismatch_errors() {
        assert!(matches!(
            synflow_score(&[1.0, 2.0], &[1.0]),
            Err(NasError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn synflow_empty_errors() {
        assert_eq!(synflow_score(&[], &[]), Err(NasError::EmptySearchSpace));
    }

    #[test]
    fn rank_descending_with_low_index_tiebreak() {
        // scores: idx0=1.0, idx1=3.0, idx2=3.0, idx3=2.0
        // descending: 3.0 (idx1), 3.0 (idx2), 2.0 (idx3), 1.0 (idx0)
        // tie between idx1 and idx2 broken by lower index → idx1 before idx2.
        let scores = [1.0, 3.0, 3.0, 2.0];
        let order = rank_architectures(&scores);
        assert_eq!(order, vec![1, 2, 3, 0]);
    }

    #[test]
    fn rank_empty_is_empty() {
        let order = rank_architectures(&[]);
        assert!(order.is_empty());
    }

    #[test]
    fn rank_nan_sinks_to_end() {
        let scores = [1.0, f32::NAN, 2.0];
        let order = rank_architectures(&scores);
        // 2.0 (idx2), 1.0 (idx0), NaN (idx1) last.
        assert_eq!(order, vec![2, 0, 1]);
    }

    #[test]
    fn rank_all_equal_preserves_index_order() {
        let scores = [5.0, 5.0, 5.0];
        assert_eq!(rank_architectures(&scores), vec![0, 1, 2]);
    }

    #[test]
    fn proxy_enum_names_and_all() {
        assert_eq!(ZeroCostProxy::all().len(), 4);
        assert_eq!(ZeroCostProxy::Naswot.name(), "naswot");
        assert_eq!(ZeroCostProxy::Snip.name(), "snip");
        assert_eq!(ZeroCostProxy::Grasp.name(), "grasp");
        assert_eq!(ZeroCostProxy::Synflow.name(), "synflow");
    }

    #[test]
    fn naswot_single_sample_finite() {
        let one = vec![vec![true, false, true]];
        let s = naswot_score(&one).expect("single-sample naswot");
        // K = [3 + λ]; logdet = ln(3 + λ).
        assert!((s - (3.0_f32 + NASWOT_RIDGE).ln()).abs() < 1e-4, "s = {s}");
    }
}
