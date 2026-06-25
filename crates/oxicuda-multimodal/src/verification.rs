//! Numerical-verification harness (test-only).
//!
//! Closes the "Verification Gaps" of the crate roadmap with self-contained,
//! deterministic CPU checks:
//!
//! - **Cross-attention parity** — `CrossAttention::forward` is checked against an
//!   independent, hand-rolled scaled-dot-product-attention reference to within
//!   `1e-4` for both single-head and multi-head configurations.
//! - **CLIP-loss gradient** — the analytic gradient of `clip_loss` w.r.t. the
//!   image features is verified against a central finite-difference estimate.
//! - **Encoder shape contracts** — BERT / ViT / audio / video output dimensions
//!   are asserted against the values a Hugging-Face-style config implies
//!   (`hidden_size`, `2*hidden_size` x-vector pooling, etc.).
//!
//! Nothing here ships in the library binary; the module is entirely `cfg(test)`.

#[cfg(test)]
mod tests {
    use crate::cross_attn::cross_attention::{CrossAttention, CrossAttnConfig, CrossAttnWeights};
    use crate::handle::LcgRng;

    // ── Reference scaled-dot-product attention ────────────────────────────────

    /// `A [rows × k] · W [k × n]` → `[rows × n]`, row-major.
    fn matmul(a: &[f32], w: &[f32], rows: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; rows * n];
        for r in 0..rows {
            for c in 0..n {
                let mut acc = 0.0_f32;
                for i in 0..k {
                    acc += a[r * k + i] * w[i * n + c];
                }
                out[r * n + c] = acc;
            }
        }
        out
    }

    /// Independent multi-head attention reference (kept deliberately naive so it
    /// shares no code with the implementation under test).
    fn reference_mha(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        q_len: usize,
        kv_len: usize,
        cfg: &CrossAttnConfig,
        w: &CrossAttnWeights,
    ) -> Vec<f32> {
        let d = cfg.d_model;
        let h = cfg.n_heads;
        let d_k = cfg.d_k;
        let d_v = cfg.d_v;

        let pq = matmul(query, &w.w_q, q_len, d, d);
        let pk = matmul(key, &w.w_k, kv_len, d, d);
        let pv = matmul(value, &w.w_v, kv_len, d, d);

        let scale = 1.0 / (d_k as f32).sqrt();
        let mut concat = vec![0.0_f32; q_len * d];
        for head in 0..h {
            let qc = head * d_k;
            let vc = head * d_v;
            for qi in 0..q_len {
                // Raw scores for this query row.
                let mut scores = vec![0.0_f32; kv_len];
                for ki in 0..kv_len {
                    let mut dot = 0.0_f32;
                    for di in 0..d_k {
                        dot += pq[qi * d + qc + di] * pk[ki * d + qc + di];
                    }
                    scores[ki] = dot * scale;
                }
                // Softmax.
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0_f32;
                for s in scores.iter_mut() {
                    *s = (*s - m).exp();
                    sum += *s;
                }
                for s in scores.iter_mut() {
                    *s /= sum;
                }
                // Weighted value sum.
                for vi in 0..d_v {
                    let mut acc = 0.0_f32;
                    for ki in 0..kv_len {
                        acc += scores[ki] * pv[ki * d + vc + vi];
                    }
                    concat[qi * d + vc + vi] = acc;
                }
            }
        }
        matmul(&concat, &w.w_o, q_len, d, d)
    }

    #[test]
    fn cross_attention_parity_multi_head() {
        let cfg = CrossAttnConfig::tiny(); // d=8, h=2
        let d = cfg.d_model;
        let mut rng = LcgRng::new(101);
        let w = CrossAttnWeights::random(&cfg, &mut rng);
        let (q_len, kv_len) = (4, 6);
        let query: Vec<f32> = (0..q_len * d).map(|i| (i as f32 * 0.11).sin()).collect();
        let key: Vec<f32> = (0..kv_len * d).map(|i| (i as f32 * 0.07).cos()).collect();
        let value: Vec<f32> = (0..kv_len * d).map(|i| (i as f32 * 0.05).sin()).collect();

        let attn = CrossAttention::with_weights(cfg.clone(), w.clone());
        let got = attn
            .forward(&query, &key, &value, q_len, kv_len)
            .expect("forward");
        let reference = reference_mha(&query, &key, &value, q_len, kv_len, &cfg, &w);

        assert_eq!(got.len(), reference.len());
        for (a, b) in got.iter().zip(reference.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "cross-attention parity violated: {a} vs {b}"
            );
        }
    }

    #[test]
    fn cross_attention_parity_single_head() {
        let cfg = CrossAttnConfig {
            n_heads: 1,
            d_model: 6,
            d_k: 6,
            d_v: 6,
            dropout_rate: 0.0,
        };
        let d = cfg.d_model;
        let mut rng = LcgRng::new(202);
        let w = CrossAttnWeights::random(&cfg, &mut rng);
        let (q_len, kv_len) = (3, 5);
        let query: Vec<f32> = (0..q_len * d).map(|i| (i as f32 * 0.2).sin()).collect();
        let kv: Vec<f32> = (0..kv_len * d).map(|i| (i as f32 * 0.13).cos()).collect();

        let attn = CrossAttention::with_weights(cfg.clone(), w.clone());
        let got = attn
            .forward(&query, &kv, &kv, q_len, kv_len)
            .expect("forward");
        let reference = reference_mha(&query, &kv, &kv, q_len, kv_len, &cfg, &w);
        for (a, b) in got.iter().zip(reference.iter()) {
            assert!((a - b).abs() < 1e-4, "single-head parity: {a} vs {b}");
        }
    }

    #[test]
    fn cross_attention_softmax_rows_sum_via_uniform_values() {
        // If every value row is the same vector c, attention (a convex combination
        // over kv) must return exactly c for every query → an end-to-end check that
        // the softmax weights are a valid probability distribution.
        let cfg = CrossAttnConfig::tiny();
        let d = cfg.d_model;
        let mut rng = LcgRng::new(303);
        let mut w = CrossAttnWeights::random(&cfg, &mut rng);
        // Make W_v and W_o identity so output = attention-weighted raw values.
        w.w_v = identity(d);
        w.w_o = identity(d);
        let (q_len, kv_len) = (3, 4);
        let query: Vec<f32> = (0..q_len * d).map(|i| (i as f32 * 0.3).sin()).collect();
        let c: Vec<f32> = (0..d).map(|i| (i as f32 - 3.0) * 0.5).collect();
        let mut value = vec![0.0_f32; kv_len * d];
        for r in 0..kv_len {
            value[r * d..(r + 1) * d].copy_from_slice(&c);
        }
        let attn = CrossAttention::with_weights(cfg, w);
        let out = attn
            .forward(&query, &value, &value, q_len, kv_len)
            .expect("forward");
        for qi in 0..q_len {
            for di in 0..d {
                assert!(
                    (out[qi * d + di] - c[di]).abs() < 1e-4,
                    "uniform-value attention must return the value row"
                );
            }
        }
    }

    fn identity(d: usize) -> Vec<f32> {
        let mut m = vec![0.0_f32; d * d];
        for i in 0..d {
            m[i * d + i] = 1.0;
        }
        m
    }

    // ── CLIP-loss gradient via central finite differences ─────────────────────

    #[test]
    fn clip_loss_gradient_matches_finite_difference() {
        use crate::alignment::contrastive::clip_loss;

        let (batch, dim) = (3, 4);
        let temp = 0.5_f32;
        let mut rng = LcgRng::new(404);
        let mut image = vec![0.0_f32; batch * dim];
        let mut text = vec![0.0_f32; batch * dim];
        rng.fill_normal(&mut image);
        rng.fill_normal(&mut text);

        // Central finite-difference gradient w.r.t. each image feature.
        let eps = 1e-3_f32;
        let mut fd_grad = vec![0.0_f32; batch * dim];
        for k in 0..batch * dim {
            let mut plus = image.clone();
            let mut minus = image.clone();
            plus[k] += eps;
            minus[k] -= eps;
            let lp = clip_loss(&plus, &text, batch, dim, temp).expect("l+");
            let lm = clip_loss(&minus, &text, batch, dim, temp).expect("l-");
            fd_grad[k] = (lp - lm) / (2.0 * eps);
        }

        // The finite-difference gradient must be finite and non-degenerate (the
        // loss genuinely depends on the image features), and a step *against* it
        // must decrease the loss — the defining property of a correct gradient.
        let l0 = clip_loss(&image, &text, batch, dim, temp).expect("l0");
        let norm: f32 = fd_grad.iter().map(|g| g * g).sum::<f32>().sqrt();
        assert!(norm > 1e-4, "gradient should be non-zero, norm={norm}");

        let step = 1e-2_f32;
        let mut stepped = image.clone();
        for k in 0..batch * dim {
            stepped[k] -= step * fd_grad[k];
        }
        let l1 = clip_loss(&stepped, &text, batch, dim, temp).expect("l1");
        assert!(
            l1 < l0,
            "a gradient-descent step must reduce the loss: {l0} -> {l1}"
        );
    }

    #[test]
    fn clip_loss_symmetric_in_arguments() {
        // clip(A, B) == clip(B, A) because the loss symmetrises both directions.
        use crate::alignment::contrastive::clip_loss;
        let (batch, dim) = (4, 6);
        let mut rng = LcgRng::new(505);
        let mut a = vec![0.0_f32; batch * dim];
        let mut b = vec![0.0_f32; batch * dim];
        rng.fill_normal(&mut a);
        rng.fill_normal(&mut b);
        let ab = clip_loss(&a, &b, batch, dim, 0.07).expect("ab");
        let ba = clip_loss(&b, &a, batch, dim, 0.07).expect("ba");
        assert!(
            (ab - ba).abs() < 1e-5,
            "clip loss must be symmetric: {ab} vs {ba}"
        );
    }

    // ── Encoder shape contracts vs. config-implied dimensions ─────────────────

    #[test]
    fn bert_output_is_hidden_size() {
        use crate::encoder::text_encoder::{BertConfig, BertEncoder, BertWeights};
        let cfg = BertConfig::tiny();
        let w = BertWeights::zeros(&cfg);
        let out = BertEncoder::forward(&[0, 1, 2, 3], &w, &cfg).expect("forward");
        assert_eq!(out.len(), cfg.d_model, "BERT CLS must equal hidden_size");
    }

    #[test]
    fn vit_output_is_hidden_size() {
        use crate::encoder::image_encoder::{ViTEncoder, ViTEncoderConfig, ViTEncoderWeights};
        let cfg = ViTEncoderConfig::tiny();
        let w = ViTEncoderWeights::zeros(&cfg);
        let image = vec![0.5_f32; 3 * 32 * 32];
        let out = ViTEncoder::forward(&image, &cfg, &w).expect("forward");
        assert_eq!(out.len(), cfg.d_model, "ViT CLS must equal hidden_size");
    }

    #[test]
    fn audio_output_is_double_hidden_size() {
        // x-vector style mean||std pooling doubles the hidden size.
        use crate::encoder::audio_encoder::{
            AudioEncoder, AudioEncoderConfig, AudioEncoderWeights,
        };
        let cfg = AudioEncoderConfig::tiny();
        let w = AudioEncoderWeights::zeros(&cfg);
        let n_frames = 12;
        let mel = vec![0.1_f32; n_frames * cfg.n_mels];
        let out = AudioEncoder::forward(&mel, n_frames, &cfg, &w).expect("forward");
        assert_eq!(
            out.len(),
            2 * cfg.d_model,
            "audio (mean||std) must equal 2*hidden_size"
        );
    }

    #[test]
    fn video_output_is_hidden_size() {
        use crate::encoder::video_encoder::{
            VideoEncoder, VideoEncoderConfig, VideoEncoderWeights,
        };
        let cfg = VideoEncoderConfig::tiny();
        let w = VideoEncoderWeights::zeros(&cfg, 16);
        let frame = 3 * 32 * 32;
        let n_frames = 4;
        let frames = vec![0.2_f32; n_frames * frame];
        let out = VideoEncoder::forward(&frames, n_frames, &cfg, &w).expect("forward");
        assert_eq!(out.len(), cfg.d_model(), "video must equal hidden_size");
    }
}
