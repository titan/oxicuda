//! FlashAttention-2 CPU simulation (Dao et al., 2022).
//!
//! Implements the tiled online-softmax algorithm from FlashAttention-2 for a
//! single or multi-head attention computation entirely in CPU memory.  This is
//! **not** optimised for throughput; it serves as a numerically exact reference
//! implementation and test fixture.
//!
//! The key algorithmic insight is that we never materialise the full
//! `seq_len × seq_len` attention matrix.  Instead we process the query
//! sequence in blocks of size `block_size` and accumulate partial softmax
//! statistics (running max `m` and running sum `l`) as we stream over the
//! key/value sequence, also in blocks.  At the end of each query block we
//! divide the accumulated output by `l` to recover the correctly-normalised
//! attention output.

use crate::error::{DnnError, DnnResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`FlashAttention`].
#[derive(Debug, Clone)]
pub struct FlashAttnConfig {
    /// Number of attention heads.
    pub n_heads: usize,
    /// Dimension per head.
    pub d_head: usize,
    /// Tile (block) size along the sequence dimension.
    pub block_size: usize,
    /// Whether to apply a causal (autoregressive) mask.
    pub causal: bool,
}

// ---------------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------------

/// FlashAttention-2 tiling CPU simulation.
pub struct FlashAttention {
    config: FlashAttnConfig,
}

impl FlashAttention {
    /// Construct a new [`FlashAttention`] layer.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if `n_heads`, `d_head`, or
    /// `block_size` is zero.
    pub fn new(config: FlashAttnConfig) -> DnnResult<Self> {
        if config.n_heads == 0 {
            return Err(DnnError::InvalidArgument("n_heads must be > 0".to_owned()));
        }
        if config.d_head == 0 {
            return Err(DnnError::InvalidArgument("d_head must be > 0".to_owned()));
        }
        if config.block_size == 0 {
            return Err(DnnError::InvalidArgument(
                "block_size must be > 0".to_owned(),
            ));
        }
        Ok(Self { config })
    }

    /// Total model dimension: `n_heads × d_head`.
    #[inline]
    pub fn d_model(&self) -> usize {
        self.config.n_heads * self.config.d_head
    }

    // -----------------------------------------------------------------------
    // Single-head forward
    // -----------------------------------------------------------------------

    /// Run FlashAttention-2 tiled algorithm for **one** attention head.
    ///
    /// `q`, `k`, `v` are all `[seq_len × d_head]` in row-major order.
    /// Returns `[seq_len × d_head]`.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidDimension`] when slice lengths do not match
    /// the expected shape.
    pub fn attn_forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
    ) -> DnnResult<Vec<f32>> {
        let d_h = self.config.d_head;
        let bs = self.config.block_size;
        let causal = self.config.causal;

        if q.len() != seq_len * d_h {
            return Err(DnnError::InvalidDimension(format!(
                "q.len() expected {}, got {}",
                seq_len * d_h,
                q.len()
            )));
        }
        if k.len() != seq_len * d_h {
            return Err(DnnError::InvalidDimension(format!(
                "k.len() expected {}, got {}",
                seq_len * d_h,
                k.len()
            )));
        }
        if v.len() != seq_len * d_h {
            return Err(DnnError::InvalidDimension(format!(
                "v.len() expected {}, got {}",
                seq_len * d_h,
                v.len()
            )));
        }

        let scale = 1.0_f32 / (d_h as f32).sqrt();
        let mut output = vec![0.0f32; seq_len * d_h];

        // Iterate over query blocks.
        let mut q_start = 0;
        while q_start < seq_len {
            let q_end = (q_start + bs).min(seq_len);
            let bq = q_end - q_start;

            // Running statistics for this query block.
            let mut m_i = vec![f32::NEG_INFINITY; bq]; // running max per query row
            let mut l_i = vec![0.0f32; bq]; // running denominator
            let mut o_i = vec![0.0f32; bq * d_h]; // running numerator output

            // Iterate over key/value blocks.
            let mut kv_start = 0;
            while kv_start < seq_len {
                let kv_end = (kv_start + bs).min(seq_len);
                let bkv = kv_end - kv_start;

                // Compute raw scores s[i, j] = scale * (q[q_start+i] · k[kv_start+j]).
                // Shape: [bq × bkv].
                let mut s = vec![0.0f32; bq * bkv];
                for i in 0..bq {
                    let qi = &q[(q_start + i) * d_h..(q_start + i + 1) * d_h];
                    for j in 0..bkv {
                        let kj = &k[(kv_start + j) * d_h..(kv_start + j + 1) * d_h];
                        let dot: f32 = qi.iter().zip(kj.iter()).map(|(a, b)| a * b).sum();
                        let val = scale * dot;
                        // Apply causal mask: future positions get -inf.
                        s[i * bkv + j] = if causal && (kv_start + j) > (q_start + i) {
                            f32::NEG_INFINITY
                        } else {
                            val
                        };
                    }
                }

                // Online softmax update for each query row i.
                for i in 0..bq {
                    // New row maximum across this KV block.
                    let s_row_max = s[i * bkv..(i + 1) * bkv]
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let m_new = m_i[i].max(s_row_max);

                    // Rescale existing accumulators by exp(m_old - m_new).
                    let scale_factor = (m_i[i] - m_new).exp();
                    l_i[i] *= scale_factor;
                    for d in 0..d_h {
                        o_i[i * d_h + d] *= scale_factor;
                    }

                    // Accumulate contribution from this KV block.
                    for j in 0..bkv {
                        let exp_s = (s[i * bkv + j] - m_new).exp();
                        l_i[i] += exp_s;
                        let vj = &v[(kv_start + j) * d_h..(kv_start + j + 1) * d_h];
                        for d in 0..d_h {
                            o_i[i * d_h + d] += exp_s * vj[d];
                        }
                    }

                    m_i[i] = m_new;
                }

                kv_start = kv_end;
            }

            // Normalise and write results for this query block.
            for i in 0..bq {
                let row_out = &mut output[(q_start + i) * d_h..(q_start + i + 1) * d_h];
                if l_i[i] > 0.0 {
                    let inv_l = 1.0 / l_i[i];
                    for (d, out_val) in row_out.iter_mut().enumerate() {
                        *out_val = o_i[i * d_h + d] * inv_l;
                    }
                } else {
                    for out_val in row_out.iter_mut() {
                        *out_val = 0.0;
                    }
                }
            }

            q_start = q_end;
        }

        Ok(output)
    }

    // -----------------------------------------------------------------------
    // Multi-head forward
    // -----------------------------------------------------------------------

    /// Run multi-head FlashAttention.
    ///
    /// Input `qkv` layout: `[seq_len × 3 × n_heads × d_head]`.
    /// Returns `[seq_len × n_heads × d_head]`.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidDimension`] when `qkv.len()` does not match
    /// the expected shape.
    pub fn forward(&self, qkv: &[f32], seq_len: usize) -> DnnResult<Vec<f32>> {
        let n_h = self.config.n_heads;
        let d_h = self.config.d_head;
        let qkv_stride = 3 * n_h * d_h; // per-token stride in qkv

        if qkv.len() != seq_len * qkv_stride {
            return Err(DnnError::InvalidDimension(format!(
                "qkv.len() expected {} (seq_len={} × 3 × n_heads={} × d_head={}), got {}",
                seq_len * qkv_stride,
                seq_len,
                n_h,
                d_h,
                qkv.len()
            )));
        }

        let mut output = vec![0.0f32; seq_len * n_h * d_h];

        for h in 0..n_h {
            // Extract q_h, k_h, v_h for this head: shape [seq_len × d_head].
            let mut q_h = vec![0.0f32; seq_len * d_h];
            let mut k_h = vec![0.0f32; seq_len * d_h];
            let mut v_h = vec![0.0f32; seq_len * d_h];

            for t in 0..seq_len {
                let base = t * qkv_stride;
                // Layout: [q_head_0..q_head_{n_h-1} | k_head_0..k_head_{n_h-1} | v_head_0..v_head_{n_h-1}]
                let q_off = base + h * d_h;
                let k_off = base + n_h * d_h + h * d_h;
                let v_off = base + 2 * n_h * d_h + h * d_h;
                q_h[t * d_h..(t + 1) * d_h].copy_from_slice(&qkv[q_off..q_off + d_h]);
                k_h[t * d_h..(t + 1) * d_h].copy_from_slice(&qkv[k_off..k_off + d_h]);
                v_h[t * d_h..(t + 1) * d_h].copy_from_slice(&qkv[v_off..v_off + d_h]);
            }

            let head_out = self.attn_forward(&q_h, &k_h, &v_h, seq_len)?;

            // Write into output: [t, h, d] → output[t * n_h * d_h + h * d_h + d]
            for t in 0..seq_len {
                let src = &head_out[t * d_h..(t + 1) * d_h];
                let dst = &mut output[t * n_h * d_h + h * d_h..t * n_h * d_h + (h + 1) * d_h];
                dst.copy_from_slice(src);
            }
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LcgRng;

    fn make_fa(n_heads: usize, d_head: usize, block_size: usize, causal: bool) -> FlashAttention {
        FlashAttention::new(FlashAttnConfig {
            n_heads,
            d_head,
            block_size,
            causal,
        })
        .expect("valid config")
    }

    fn random_seq(seq_len: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..seq_len * dim)
            .map(|_| (rng.next_f64() as f32 - 0.5) * 2.0)
            .collect()
    }

    /// Naive O(seq_len²) attention reference.
    fn naive_attn(q: &[f32], k: &[f32], v: &[f32], seq_len: usize, d_head: usize) -> Vec<f32> {
        let scale = 1.0_f32 / (d_head as f32).sqrt();
        let mut scores = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let s: f32 = (0..d_head)
                    .map(|d| q[i * d_head + d] * k[j * d_head + d])
                    .sum::<f32>()
                    * scale;
                scores[i * seq_len + j] = s;
            }
        }
        // Softmax per row.
        for i in 0..seq_len {
            let m = scores[i * seq_len..(i + 1) * seq_len]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores[i * seq_len..(i + 1) * seq_len]
                .iter()
                .map(|&s| (s - m).exp())
                .collect();
            let z: f32 = exps.iter().sum();
            for j in 0..seq_len {
                scores[i * seq_len + j] = exps[j] / z;
            }
        }
        // output = scores @ v.
        let mut naive_out = vec![0.0f32; seq_len * d_head];
        for i in 0..seq_len {
            for j in 0..seq_len {
                for d in 0..d_head {
                    naive_out[i * d_head + d] += scores[i * seq_len + j] * v[j * d_head + d];
                }
            }
        }
        naive_out
    }

    // 1. Output shape
    #[test]
    fn output_shape() {
        let fa = make_fa(1, 8, 4, false);
        let seq_len = 12;
        let d_head = 8;
        let q = random_seq(seq_len, d_head, 1);
        let k = random_seq(seq_len, d_head, 2);
        let v = random_seq(seq_len, d_head, 3);
        let out = fa.attn_forward(&q, &k, &v, seq_len).expect("ok");
        assert_eq!(out.len(), seq_len * d_head);
    }

    // 2. All outputs are finite
    #[test]
    fn output_finite() {
        let fa = make_fa(1, 8, 4, false);
        let seq_len = 8;
        let d_head = 8;
        let q = random_seq(seq_len, d_head, 10);
        let k = random_seq(seq_len, d_head, 11);
        let v = random_seq(seq_len, d_head, 12);
        let out = fa.attn_forward(&q, &k, &v, seq_len).expect("ok");
        for (i, val) in out.iter().enumerate() {
            assert!(val.is_finite(), "output[{i}] is not finite: {val}");
        }
    }

    // 3. Causal mask effect
    #[test]
    fn causal_mask_effect() {
        let d_head = 4;
        let seq_len = 4;
        let fa_causal = make_fa(1, d_head, 4, true);
        let fa_full = make_fa(1, d_head, 4, false);
        let q = random_seq(seq_len, d_head, 20);
        let k = random_seq(seq_len, d_head, 21);
        // Use clearly distinct V vectors so causal vs full gives different results.
        // v[0] = [10,0,0,0], v[1] = [0,10,0,0], v[2] = [0,0,10,0], v[3] = [0,0,0,10]
        let mut v = vec![0.0f32; seq_len * d_head];
        for i in 0..seq_len {
            v[i * d_head + i] = 10.0;
        }
        let out_causal = fa_causal
            .attn_forward(&q, &k, &v, seq_len)
            .expect("causal ok");
        let out_full = fa_full.attn_forward(&q, &k, &v, seq_len).expect("full ok");
        // With causal=true, token 0 only attends to itself, so its output
        // must equal v[0] = [10, 0, 0, 0].
        let out0 = &out_causal[..d_head];
        assert!(
            (out0[0] - 10.0).abs() < 1e-3,
            "causal out[0,0]={} should be 10",
            out0[0]
        );
        for (d, item) in out0.iter().enumerate().take(d_head).skip(1) {
            assert!(item.abs() < 1e-3, "causal out[0,{d}]={} should be ~0", item);
        }
        // With full attention, token 0 can see all positions, so it gets a
        // mix of all V vectors.  Out[0, 0] should be < 10 (diluted).
        let full0 = &out_full[..d_head];
        assert!(
            full0[0] < 9.9,
            "full-attn out[0,0]={} should be < 10 (blended with other V rows)",
            full0[0]
        );
    }

    // 4. Single-token sequence
    #[test]
    fn single_token_seq() {
        let fa = make_fa(1, 8, 4, false);
        let d_head = 8;
        let q = random_seq(1, d_head, 30);
        let k = q.clone();
        let v = random_seq(1, d_head, 31);
        let out = fa.attn_forward(&q, &k, &v, 1).expect("ok");
        // With single token, output must equal v (attention weight = 1).
        for d in 0..d_head {
            assert!(
                (out[d] - v[d]).abs() < 1e-5,
                "single token: out[{d}]={} != v[{d}]={}",
                out[d],
                v[d]
            );
        }
    }

    // 5. Matches standard (naive) attention
    #[test]
    fn matches_standard_attn() {
        let d_head = 8;
        let seq_len = 10;
        let fa = make_fa(1, d_head, 3, false);
        let q = random_seq(seq_len, d_head, 40);
        let k = random_seq(seq_len, d_head, 41);
        let v = random_seq(seq_len, d_head, 42);
        let flash_out = fa.attn_forward(&q, &k, &v, seq_len).expect("flash ok");
        let naive_out = naive_attn(&q, &k, &v, seq_len, d_head);
        for i in 0..flash_out.len() {
            assert!(
                (flash_out[i] - naive_out[i]).abs() < 1e-3,
                "mismatch at [{i}]: flash={} naive={}",
                flash_out[i],
                naive_out[i]
            );
        }
    }

    // 6. block_size > seq_len
    #[test]
    fn block_size_gt_seq_len() {
        let fa = make_fa(1, 4, 64, false);
        let q = random_seq(5, 4, 50);
        let k = random_seq(5, 4, 51);
        let v = random_seq(5, 4, 52);
        let out = fa.attn_forward(&q, &k, &v, 5).expect("ok");
        assert_eq!(out.len(), 5 * 4);
    }

    // 7. d_head=0 → error
    #[test]
    fn d_head_0_error() {
        let result = FlashAttention::new(FlashAttnConfig {
            n_heads: 1,
            d_head: 0,
            block_size: 8,
            causal: false,
        });
        assert!(matches!(result, Err(DnnError::InvalidArgument(_))));
    }

    // 8. n_heads=0 → error
    #[test]
    fn n_heads_0_error() {
        let result = FlashAttention::new(FlashAttnConfig {
            n_heads: 0,
            d_head: 8,
            block_size: 8,
            causal: false,
        });
        assert!(matches!(result, Err(DnnError::InvalidArgument(_))));
    }

    // 9. No NaN in output
    #[test]
    fn output_not_nan() {
        let fa = make_fa(2, 4, 3, true);
        let seq_len = 8;
        let d_head = 4;
        let q = random_seq(seq_len, d_head, 60);
        let k = random_seq(seq_len, d_head, 61);
        let v = random_seq(seq_len, d_head, 62);
        let out = fa.attn_forward(&q, &k, &v, seq_len).expect("ok");
        assert!(out.iter().all(|v| !v.is_nan()), "output contains NaN");
    }

    // 10. Running twice gives the same result (determinism)
    #[test]
    fn batch_invariant() {
        let fa = make_fa(1, 8, 4, false);
        let seq_len = 6;
        let d_head = 8;
        let q = random_seq(seq_len, d_head, 70);
        let k = random_seq(seq_len, d_head, 71);
        let v = random_seq(seq_len, d_head, 72);
        let out1 = fa.attn_forward(&q, &k, &v, seq_len).expect("ok");
        let out2 = fa.attn_forward(&q, &k, &v, seq_len).expect("ok");
        for i in 0..out1.len() {
            assert_eq!(out1[i], out2[i], "result differed at [{i}]");
        }
    }
}
