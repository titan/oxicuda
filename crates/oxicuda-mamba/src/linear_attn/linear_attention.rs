//! Causal linear attention (Katharopoulos et al. 2020) and gated linear
//! attention (GLA, Yang et al. 2023).
//!
//! # Background
//!
//! Softmax attention has `O(L²)` cost.  **Linear attention** ("Transformers are
//! RNNs", Katharopoulos, Vyas, Pappas & Fleuret 2020) replaces the softmax
//! similarity `exp(qᵀk)` with a kernel `φ(q)ᵀ φ(k)` for a non-negative feature
//! map `φ`.  The default feature map is `φ(x) = elu(x) + 1`, which keeps the
//! similarities positive.  Causal linear attention then admits a constant-memory
//! recurrence over a running key-value state `S` and a running key-sum `z`:
//!
//! ```text
//! S_t = S_{t-1} + φ(kₜ)ᵀ vₜ            (S_t ∈ ℝ^{d_φ × d_v})
//! z_t = z_{t-1} + φ(kₜ)                (z_t ∈ ℝ^{d_φ})
//! oₜ  = φ(qₜ) S_t / (φ(qₜ) · z_t + ε)
//! ```
//!
//! **Gated linear attention** (GLA) generalises this with a per-step,
//! per-feature **forget gate** `α_t ∈ (0, 1)^{d_φ}` applied to the state,
//! interpolating between full memory (`α = 1`, plain linear attention) and a
//! decaying memory:
//!
//! ```text
//! S_t = diag(α_t) S_{t-1} + φ(kₜ)ᵀ vₜ
//! z_t = α_t ⊙ z_{t-1} + φ(kₜ)
//! ```
//!
//! Both forms are exposed in parallel (quadratic, for verification / training)
//! and recurrent (linear, for inference) variants.
//!
//! # Layout
//!
//! Flat row-major tensors: `q`, `k` are `[L × d_k]`, `v` is `[L × d_v]`, output
//! is `[L × d_v]`.  The feature map preserves dimension, so `d_φ = d_k`.

use crate::error::{MambaError, MambaResult};

/// Small denominator floor preventing division by zero in the normalizer.
const EPS: f32 = 1e-6;

// ─── Feature maps ────────────────────────────────────────────────────────────

/// Non-negative feature map `φ` applied element-wise to queries and keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureMap {
    /// `φ(x) = elu(x) + 1` (Katharopoulos default): `x+1` if `x>0`, else `eˣ`.
    EluPlusOne,
    /// `φ(x) = relu(x)` (non-negative, sparse).
    Relu,
    /// `φ(x) = x` (identity; no positivity guarantee — caller's responsibility).
    Identity,
}

impl FeatureMap {
    /// Apply the feature map to a single scalar.
    #[inline]
    #[must_use]
    pub fn apply_scalar(self, x: f32) -> f32 {
        match self {
            FeatureMap::EluPlusOne => {
                if x > 0.0 {
                    x + 1.0
                } else {
                    x.exp()
                }
            }
            FeatureMap::Relu => x.max(0.0),
            FeatureMap::Identity => x,
        }
    }

    /// Apply the feature map element-wise to a slice, into `out`.
    fn apply_into(self, x: &[f32], out: &mut [f32]) {
        for (o, &xi) in out.iter_mut().zip(x.iter()) {
            *o = self.apply_scalar(xi);
        }
    }
}

// ─── LinearAttentionConfig ───────────────────────────────────────────────────

/// Configuration for a single-head linear-attention computation.
#[derive(Debug, Clone)]
pub struct LinearAttentionConfig {
    /// Sequence length `L`.
    pub seq_len: usize,
    /// Query / key dimension `d_k` (= feature dim `d_φ`).
    pub d_k: usize,
    /// Value dimension `d_v`.
    pub d_v: usize,
    /// Feature map `φ`.
    pub feature_map: FeatureMap,
}

impl LinearAttentionConfig {
    /// Create a new config with the `elu+1` feature map.
    ///
    /// # Errors
    ///
    /// * [`MambaError::InvalidSeqLen`]   — if `seq_len == 0`.
    /// * [`MambaError::InvalidModelDim`] — if `d_k == 0` or `d_v == 0`.
    pub fn new(seq_len: usize, d_k: usize, d_v: usize) -> MambaResult<Self> {
        if seq_len == 0 {
            return Err(MambaError::InvalidSeqLen(0));
        }
        if d_k == 0 || d_v == 0 {
            return Err(MambaError::InvalidModelDim(d_k.min(d_v)));
        }
        Ok(Self {
            seq_len,
            d_k,
            d_v,
            feature_map: FeatureMap::EluPlusOne,
        })
    }

    /// Override the feature map.
    #[must_use]
    pub fn with_feature_map(mut self, fm: FeatureMap) -> Self {
        self.feature_map = fm;
        self
    }
}

#[inline]
fn check_shapes(cfg: &LinearAttentionConfig, q: &[f32], k: &[f32], v: &[f32]) -> MambaResult<()> {
    let lk = cfg.seq_len * cfg.d_k;
    let lv = cfg.seq_len * cfg.d_v;
    if q.len() != lk {
        return Err(MambaError::DimensionMismatch {
            expected: lk,
            got: q.len(),
        });
    }
    if k.len() != lk {
        return Err(MambaError::DimensionMismatch {
            expected: lk,
            got: k.len(),
        });
    }
    if v.len() != lv {
        return Err(MambaError::DimensionMismatch {
            expected: lv,
            got: v.len(),
        });
    }
    Ok(())
}

// ─── Recurrent (linear) causal attention ─────────────────────────────────────

/// Causal linear attention via the constant-memory recurrence (Katharopoulos).
///
/// Returns `[L × d_v]`.
///
/// # Errors
///
/// [`MambaError::DimensionMismatch`] on shape disagreement.
pub fn linear_attention_recurrent(
    cfg: &LinearAttentionConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> MambaResult<Vec<f32>> {
    check_shapes(cfg, q, k, v)?;
    let l = cfg.seq_len;
    let dk = cfg.d_k;
    let dv = cfg.d_v;
    let fm = cfg.feature_map;

    let mut s = vec![0.0_f32; dk * dv]; // running S = Σ φ(k)ᵀ v
    let mut z = vec![0.0_f32; dk]; // running z = Σ φ(k)
    let mut phi_q = vec![0.0_f32; dk];
    let mut phi_k = vec![0.0_f32; dk];
    let mut out = vec![0.0_f32; l * dv];

    for t in 0..l {
        fm.apply_into(&q[t * dk..t * dk + dk], &mut phi_q);
        fm.apply_into(&k[t * dk..t * dk + dk], &mut phi_k);
        let v_t = &v[t * dv..t * dv + dv];

        // S += φ(k)ᵀ v ; z += φ(k)
        for i in 0..dk {
            let ki = phi_k[i];
            z[i] += ki;
            let row = &mut s[i * dv..i * dv + dv];
            for j in 0..dv {
                row[j] += ki * v_t[j];
            }
        }
        // numerator = φ(q) S ; denom = φ(q)·z
        let mut denom = 0.0_f32;
        for i in 0..dk {
            denom += phi_q[i] * z[i];
        }
        denom += EPS;
        let o_t = &mut out[t * dv..t * dv + dv];
        for i in 0..dk {
            let qi = phi_q[i];
            let row = &s[i * dv..i * dv + dv];
            for j in 0..dv {
                o_t[j] += qi * row[j];
            }
        }
        for o in o_t.iter_mut() {
            *o /= denom;
        }
    }
    Ok(out)
}

// ─── Parallel (quadratic) causal attention ───────────────────────────────────

/// Causal linear attention via the explicit `O(L²)` masked form (verification).
///
/// `oₜ = Σ_{s≤t} (φ(qₜ)·φ(kₛ)) vₛ / (Σ_{s≤t} φ(qₜ)·φ(kₛ) + ε)`.
///
/// # Errors
///
/// [`MambaError::DimensionMismatch`] on shape disagreement.
pub fn linear_attention_parallel(
    cfg: &LinearAttentionConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> MambaResult<Vec<f32>> {
    check_shapes(cfg, q, k, v)?;
    let l = cfg.seq_len;
    let dk = cfg.d_k;
    let dv = cfg.d_v;
    let fm = cfg.feature_map;

    // Pre-compute φ(k) once.
    let mut phi_k_all = vec![0.0_f32; l * dk];
    for t in 0..l {
        let dst = &mut phi_k_all[t * dk..t * dk + dk];
        fm.apply_into(&k[t * dk..t * dk + dk], dst);
    }

    let mut phi_q = vec![0.0_f32; dk];
    let mut out = vec![0.0_f32; l * dv];
    for t in 0..l {
        fm.apply_into(&q[t * dk..t * dk + dk], &mut phi_q);
        let o_t = &mut out[t * dv..t * dv + dv];
        let mut denom = 0.0_f32;
        for s in 0..=t {
            let phi_k_s = &phi_k_all[s * dk..s * dk + dk];
            let mut sim = 0.0_f32;
            for i in 0..dk {
                sim += phi_q[i] * phi_k_s[i];
            }
            denom += sim;
            let v_s = &v[s * dv..s * dv + dv];
            for j in 0..dv {
                o_t[j] += sim * v_s[j];
            }
        }
        denom += EPS;
        for o in o_t.iter_mut() {
            *o /= denom;
        }
    }
    Ok(out)
}

// ─── Gated linear attention (GLA) ────────────────────────────────────────────

/// Gated linear attention (recurrent form) with a per-feature forget gate.
///
/// `gates` is `[L × d_k]` with each entry in `(0, 1]`; the state is decayed
/// element-wise per feature row: `S ← diag(α_t) S + φ(kₜ)ᵀ vₜ`.  When all gates
/// equal `1` this reduces exactly to [`linear_attention_recurrent`].
///
/// # Errors
///
/// * [`MambaError::DimensionMismatch`] — on any shape disagreement.
/// * [`MambaError::Internal`]          — if any gate is outside `(0, 1]`.
pub fn gated_linear_attention(
    cfg: &LinearAttentionConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gates: &[f32],
) -> MambaResult<Vec<f32>> {
    check_shapes(cfg, q, k, v)?;
    let l = cfg.seq_len;
    let dk = cfg.d_k;
    let dv = cfg.d_v;
    let fm = cfg.feature_map;
    if gates.len() != l * dk {
        return Err(MambaError::DimensionMismatch {
            expected: l * dk,
            got: gates.len(),
        });
    }
    for &g in gates {
        if !(g > 0.0 && g <= 1.0) {
            return Err(MambaError::Internal(format!(
                "GLA gate must be in (0, 1], got {g}"
            )));
        }
    }

    let mut s = vec![0.0_f32; dk * dv];
    let mut z = vec![0.0_f32; dk];
    let mut phi_q = vec![0.0_f32; dk];
    let mut phi_k = vec![0.0_f32; dk];
    let mut out = vec![0.0_f32; l * dv];

    for t in 0..l {
        fm.apply_into(&q[t * dk..t * dk + dk], &mut phi_q);
        fm.apply_into(&k[t * dk..t * dk + dk], &mut phi_k);
        let v_t = &v[t * dv..t * dv + dv];
        let alpha = &gates[t * dk..t * dk + dk];

        for i in 0..dk {
            let a = alpha[i];
            // z_i ← α_i z_i + φ(k)_i
            z[i] = a * z[i] + phi_k[i];
            let row = &mut s[i * dv..i * dv + dv];
            let ki = phi_k[i];
            for j in 0..dv {
                row[j] = a * row[j] + ki * v_t[j];
            }
        }
        let mut denom = 0.0_f32;
        for i in 0..dk {
            denom += phi_q[i] * z[i];
        }
        denom += EPS;
        let o_t = &mut out[t * dv..t * dv + dv];
        for i in 0..dk {
            let qi = phi_q[i];
            let row = &s[i * dv..i * dv + dv];
            for j in 0..dv {
                o_t[j] += qi * row[j];
            }
        }
        for o in o_t.iter_mut() {
            *o /= denom;
        }
    }
    Ok(out)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn rand_vec(rng: &mut LcgRng, n: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; n];
        rng.fill_normal(&mut v);
        v
    }

    fn cfg(l: usize, dk: usize, dv: usize) -> LinearAttentionConfig {
        LinearAttentionConfig::new(l, dk, dv).expect("cfg")
    }

    #[test]
    fn config_rejects_bad_dims() {
        assert!(matches!(
            LinearAttentionConfig::new(0, 2, 2),
            Err(MambaError::InvalidSeqLen(0))
        ));
        assert!(matches!(
            LinearAttentionConfig::new(4, 0, 2),
            Err(MambaError::InvalidModelDim(_))
        ));
        assert!(matches!(
            LinearAttentionConfig::new(4, 2, 0),
            Err(MambaError::InvalidModelDim(_))
        ));
    }

    #[test]
    fn feature_map_elu_plus_one() {
        let fm = FeatureMap::EluPlusOne;
        // x>0 → x+1
        assert!((fm.apply_scalar(2.0) - 3.0).abs() < 1e-6);
        // x=0 → e^0 = 1
        assert!((fm.apply_scalar(0.0) - 1.0).abs() < 1e-6);
        // x<0 → e^x ∈ (0,1)
        let v = fm.apply_scalar(-1.0);
        assert!(v > 0.0 && v < 1.0);
    }

    #[test]
    fn feature_map_relu_and_identity() {
        assert_eq!(FeatureMap::Relu.apply_scalar(-3.0), 0.0);
        assert_eq!(FeatureMap::Relu.apply_scalar(3.0), 3.0);
        assert_eq!(FeatureMap::Identity.apply_scalar(-3.0), -3.0);
    }

    #[test]
    fn feature_map_nonnegative_for_elu_relu() {
        let mut rng = LcgRng::new(5);
        for _ in 0..200 {
            let (a, b) = rng.next_normal_pair();
            for &x in &[a, b] {
                assert!(FeatureMap::EluPlusOne.apply_scalar(x) > 0.0);
                assert!(FeatureMap::Relu.apply_scalar(x) >= 0.0);
            }
        }
    }

    #[test]
    fn recurrent_shape_finite() {
        let mut rng = LcgRng::new(1);
        let c = cfg(6, 3, 4);
        let q = rand_vec(&mut rng, 6 * 3);
        let k = rand_vec(&mut rng, 6 * 3);
        let v = rand_vec(&mut rng, 6 * 4);
        let o = linear_attention_recurrent(&c, &q, &k, &v).expect("rec");
        assert_eq!(o.len(), 6 * 4);
        assert!(o.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn recurrent_rejects_bad_shapes() {
        let c = cfg(4, 2, 2);
        assert!(linear_attention_recurrent(&c, &[0.0; 7], &[0.0; 8], &[0.0; 8]).is_err());
        assert!(linear_attention_recurrent(&c, &[0.0; 8], &[0.0; 7], &[0.0; 8]).is_err());
        assert!(linear_attention_recurrent(&c, &[0.0; 8], &[0.0; 8], &[0.0; 7]).is_err());
    }

    #[test]
    fn parallel_equals_recurrent() {
        let mut rng = LcgRng::new(9);
        let c = cfg(10, 4, 3);
        let q = rand_vec(&mut rng, 10 * 4);
        let k = rand_vec(&mut rng, 10 * 4);
        let v = rand_vec(&mut rng, 10 * 3);
        let par = linear_attention_parallel(&c, &q, &k, &v).expect("par");
        let rec = linear_attention_recurrent(&c, &q, &k, &v).expect("rec");
        for (a, b) in par.iter().zip(rec.iter()) {
            assert!((a - b).abs() < 1e-3, "par {a} vs rec {b}");
        }
    }

    #[test]
    fn parallel_equals_recurrent_relu() {
        let mut rng = LcgRng::new(11);
        let c = cfg(8, 3, 3).with_feature_map(FeatureMap::Relu);
        // Use positive-mean inputs so relu features are not all zero.
        let q: Vec<f32> = (0..8 * 3).map(|i| (i % 5) as f32 * 0.3 + 0.1).collect();
        let k: Vec<f32> = (0..8 * 3).map(|i| (i % 7) as f32 * 0.2 + 0.1).collect();
        let v = rand_vec(&mut rng, 8 * 3);
        let par = linear_attention_parallel(&c, &q, &k, &v).expect("par");
        let rec = linear_attention_recurrent(&c, &q, &k, &v).expect("rec");
        for (a, b) in par.iter().zip(rec.iter()) {
            assert!((a - b).abs() < 1e-3);
        }
    }

    #[test]
    fn single_token_is_value() {
        // L=1: output equals v0 exactly (numerator = φ(q)·φ(k)·v, denom = φ(q)·φ(k)).
        let c = cfg(1, 3, 2);
        let q = vec![0.5_f32, -0.2, 0.7];
        let k = vec![0.1_f32, 0.3, -0.4];
        let v = vec![2.0_f32, -5.0];
        let o = linear_attention_recurrent(&c, &q, &k, &v).expect("rec");
        // With a single token the normalizer cancels (up to ε): o ≈ v.
        assert!((o[0] - 2.0).abs() < 1e-2, "o0={}", o[0]);
        assert!((o[1] - (-5.0)).abs() < 1e-2, "o1={}", o[1]);
    }

    #[test]
    fn causality_future_does_not_leak() {
        let c = cfg(3, 2, 2);
        let q = vec![0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let k = vec![0.2, 0.1, 0.3, 0.2, 0.4, 0.3];
        let v0 = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let mut v1 = v0.clone();
        v1[4] = 99.0; // future value at t=2
        v1[5] = 99.0;
        let o0 = linear_attention_recurrent(&c, &q, &k, &v0).expect("o0");
        let o1 = linear_attention_recurrent(&c, &q, &k, &v1).expect("o1");
        assert!((o0[0] - o1[0]).abs() < 1e-9);
        assert!((o0[2] - o1[2]).abs() < 1e-9); // t=1 still unaffected
    }

    #[test]
    fn gla_reduces_to_linear_when_gates_one() {
        let mut rng = LcgRng::new(21);
        let c = cfg(9, 3, 4);
        let q = rand_vec(&mut rng, 9 * 3);
        let k = rand_vec(&mut rng, 9 * 3);
        let v = rand_vec(&mut rng, 9 * 4);
        let ones = vec![1.0_f32; 9 * 3];
        let plain = linear_attention_recurrent(&c, &q, &k, &v).expect("plain");
        let gated = gated_linear_attention(&c, &q, &k, &v, &ones).expect("gated");
        for (a, b) in plain.iter().zip(gated.iter()) {
            assert!((a - b).abs() < 1e-4, "plain {a} vs gated {b}");
        }
    }

    #[test]
    fn gla_rejects_bad_gate() {
        let c = cfg(2, 2, 2);
        let q = vec![0.1; 4];
        let k = vec![0.1; 4];
        let v = vec![0.1; 4];
        let bad = vec![1.5, 0.5, 0.5, 0.5];
        assert!(gated_linear_attention(&c, &q, &k, &v, &bad).is_err());
        let bad2 = vec![0.0, 0.5, 0.5, 0.5];
        assert!(gated_linear_attention(&c, &q, &k, &v, &bad2).is_err());
    }

    #[test]
    fn gla_wrong_gate_length_errors() {
        let c = cfg(2, 2, 2);
        let q = vec![0.1; 4];
        let k = vec![0.1; 4];
        let v = vec![0.1; 4];
        assert!(matches!(
            gated_linear_attention(&c, &q, &k, &v, &[0.5; 3]),
            Err(MambaError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn gla_strong_forget_localizes() {
        // With a tiny gate (≈ strong forgetting) the output depends mainly on the
        // current token, so changing distant past has little effect.
        let c = cfg(4, 2, 2);
        let q = vec![0.5_f32; 8];
        let k = vec![0.5_f32; 8];
        let v_a = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut v_b = v_a.clone();
        v_b[0] = 50.0; // change the distant past (t=0)
        v_b[1] = 50.0;
        let gates = vec![0.01_f32; 8]; // near-total forgetting
        let oa = gated_linear_attention(&c, &q, &k, &v_a, &gates).expect("oa");
        let ob = gated_linear_attention(&c, &q, &k, &v_b, &gates).expect("ob");
        // Last position should be nearly identical despite the past change.
        let last = 3 * 2;
        assert!(
            (oa[last] - ob[last]).abs() < 1e-1,
            "{} vs {}",
            oa[last],
            ob[last]
        );
    }
}
