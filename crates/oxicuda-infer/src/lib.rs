//! OxiCUDA Inference Engine — Vol.11.
//!
//! `oxicuda-infer` provides a production-grade GPU inference engine built on
//! the OxiCUDA stack.  It implements the core algorithms required for
//! efficient large-language-model serving:
//!
//! # Architecture
//!
//! ```text
//!  ┌──────────────────────────────────────────────────────┐
//!  │               ContinuousBatcher                      │  ← orchestrator
//!  └────────────────────────┬─────────────────────────────┘
//!         ┌─────────────────┼──────────────────┐
//!  ┌──────▼──────┐  ┌───────▼──────┐  ┌────────▼──────┐
//!  │  Scheduler  │  │ CacheManager │  │ Sampling Suite │
//!  │  (FCFS +    │  │ PagedKvCache │  │  greedy        │
//!  │  preemption)│  │ PrefixCache  │  │  top-k / top-p │
//!  └─────────────┘  └──────────────┘  │  beam search   │
//!                                      │  speculative   │
//!                                      └────────────────┘
//!                                      ┌────────────────┐
//!                                      │  Executor      │
//!                                      │  ModelRunner   │
//!                                      │  PagedAttnCPU  │
//!                                      └────────────────┘
//! ```
//!
//! # Key Algorithms
//!
//! ## PagedAttention (Kwon et al., 2023)
//!
//! KV cache stored in fixed-size physical blocks accessed via a per-sequence
//! block table.  Enables non-contiguous memory allocation and O(1) prefix
//! sharing for prompt caching.
//!
//! ## Continuous Batching (vLLM, Yu et al., 2022)
//!
//! Sequences join and leave the GPU batch at token granularity rather than
//! waiting for the entire batch to finish.  Dramatically improves throughput
//! for variable-length generation.
//!
//! ## Speculative Decoding (Chen et al., 2023)
//!
//! A fast draft model generates `k` candidate tokens; the slow target model
//! verifies all `k+1` positions in a single pass.  Provably correct:
//! the output distribution is identical to target-only sampling.
//!
//! # Quick Start
//!
//! ```rust
//! use oxicuda_infer::batch::{BatcherConfig, ContinuousBatcher, SamplingParams};
//! use oxicuda_infer::cache::kv_cache::PagedKvCache;
//! use oxicuda_infer::error::InferResult;
//!
//! // 1. Build a KV cache (4 layers, 4 kv-heads, head_dim=64, block_size=16, 128 blocks).
//! let kv_cache = PagedKvCache::new(4, 4, 64, 16, 128);
//!
//! // 2. Construct the continuous batcher.
//! let mut batcher = ContinuousBatcher::new(BatcherConfig::default_test(), kv_cache);
//!
//! // 3. Submit a generation request.
//! let params = SamplingParams { max_new_tokens: 4, eos_token_id: Some(1), ..Default::default() };
//! let _seq_id = batcher.add_request(vec![10, 20, 30], params);
//!
//! // 4. Run decode steps until all sequences finish.
//! // (In a real engine, `model_fn` would call the actual GPU model.)
//! let model_fn = |tokens: &[u32], _btables: &[Vec<u32>], _lens: &[usize]| -> InferResult<Vec<Vec<f32>>> {
//!     Ok(tokens.iter().map(|_| {
//!         let mut v = vec![0.0_f32; 256];
//!         v[1] = 10.0;   // always predict EOS
//!         v
//!     }).collect())
//! };
//!
//! let output = batcher.step(model_fn).expect("model_fn is infallible in this example");
//! assert!(!output.is_empty());
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::module_name_repetitions)]

pub mod batch;
pub mod cache;
pub mod decoding;
pub mod error;
pub mod executor;
pub mod handle;
pub mod ptx_kernels;
pub mod quantization;
pub mod sampling;

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests;

// Re-export the most commonly used types.
pub use batch::{
    BatcherConfig, ChunkPlanner, ChunkedPrefillPlan, ContinuousBatcher, FinishReason,
    GenerationOutput, PrefillChunk, SamplingOverride, SamplingOverrideTable, SamplingParams,
    ScheduledBatch, Scheduler, SchedulerConfig, Sequence, SequenceId, SequenceStatus, StepPacking,
};
pub use cache::{
    BlockId, CacheManager, CompactionPlan, KvBlock, KvQuantConfig, MatchResult, PagedKvCache,
    PrefixCache, PrefixEntry, QuantizedToken, RadixCache, SlidingWindowConfig,
    SlidingWindowManager, plan_compaction, quantize_token, rewrite_block_table,
};
pub use decoding::{
    BeamCandidate, BeamConfig, PromptLookupDecoder, beam_search, no_repeat_ngram_banned,
};
pub use error::{InferError, InferResult};
pub use executor::{
    AttentionConfig, MockModelRunner, ModelRunner, RunnerStats, paged_attention_cpu,
};
pub use handle::InferHandle;
pub use quantization::{
    AwqConfig, AwqResult, GroupParams, awq_dequantize, awq_output_mse, awq_quantize,
    dense_output_mse, group_dequantize, group_quantize,
};
pub use sampling::{
    BeamHypothesis, BeamSearchConfig, BeamSearchState, ContrastiveSearchConfig, Dfa, DfaBuilder,
    GrammarConstraint, JsonConstraint, JsonToken, LogitsProcessor, LogitsProcessorConfig,
    MedusaConfig, MedusaDecoder, Mirostat, MirostatConfig, Rng, WatermarkDetection, Watermarker,
    contrastive_search_select, epsilon_filter, epsilon_sample, greedy_sample, greedy_sample_batch,
    speculative_verify, top_k_filter, top_k_sample, top_p_filter, top_p_sample, typical_filter,
    typical_sample,
};

// ─── Integration tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end greedy generation: batcher generates until EOS.
    #[test]
    fn e2e_greedy_until_eos() {
        let vocab = 32_usize;
        let kv = PagedKvCache::new(2, 2, 16, 8, 64);
        let cfg = BatcherConfig {
            vocab_size: vocab,
            ..BatcherConfig::default_test()
        };
        let mut b = ContinuousBatcher::new(cfg, kv);
        let params = SamplingParams {
            eos_token_id: Some(5),
            max_new_tokens: 64,
            ..Default::default()
        };
        b.add_request(vec![1, 2, 3], params);

        // Model: always returns high logit for token 5 (EOS)
        let model_fn =
            |tokens: &[u32], _: &[Vec<u32>], _: &[usize]| -> InferResult<Vec<Vec<f32>>> {
                Ok(tokens
                    .iter()
                    .map(|_| {
                        let mut v = vec![0.0_f32; 32];
                        v[5] = 10.0;
                        v
                    })
                    .collect())
            };

        let outputs = b.step(model_fn).expect("greedy EOS model step succeeds");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].finish_reason, FinishReason::EosToken(5));
        assert!(!b.has_unfinished());
    }

    /// End-to-end: max_new_tokens termination.
    #[test]
    fn e2e_max_tokens_termination() {
        let kv = PagedKvCache::new(2, 2, 16, 8, 64);
        let cfg = BatcherConfig {
            vocab_size: 16,
            ..BatcherConfig::default_test()
        };
        let mut b = ContinuousBatcher::new(cfg, kv);
        let params = SamplingParams {
            max_new_tokens: 1,
            ..Default::default()
        };
        b.add_request(vec![0], params);

        let model_fn =
            |tokens: &[u32], _: &[Vec<u32>], _: &[usize]| -> InferResult<Vec<Vec<f32>>> {
                Ok(tokens.iter().map(|_| vec![1.0_f32; 16]).collect())
            };

        let outputs = b.step(model_fn).expect("uniform logit model step succeeds");
        assert_eq!(outputs[0].finish_reason, FinishReason::MaxLength);
    }

    /// Beam search completes on EOS.
    #[test]
    fn e2e_beam_search_completes() {
        let cfg = BeamSearchConfig {
            beam_width: 2,
            eos_token_id: 0,
            max_new_tokens: 8,
            length_penalty: 0.6,
        };
        let mut state = BeamSearchState::new(cfg);
        // All logits peak at token 0 (EOS) → all beams finish in one step.
        let logits: Vec<Vec<f32>> = (0..2)
            .map(|_| {
                let mut v = vec![0.0_f32; 8];
                v[0] = 10.0;
                v
            })
            .collect();
        let done = state.step(&logits).expect("valid beam search logits");
        assert!(done);
        assert!(!state.completed.is_empty());
    }

    /// Speculative decoding: when draft == target, all drafts accepted.
    #[test]
    fn e2e_speculative_all_accepted() {
        let vocab = 8_usize;
        let k = 3_usize;
        let probs: Vec<Vec<f32>> = (0..k)
            .map(|i| {
                let mut v = vec![0.0_f32; vocab];
                v[i % vocab] = 1.0;
                v
            })
            .collect();
        let mut target = probs.clone();
        target.push({
            let mut v = vec![0.0_f32; vocab];
            v[k % vocab] = 1.0;
            v
        });
        let draft: Vec<u32> = (0..k as u32).map(|i| i % vocab as u32).collect();
        let mut rng = Rng::new(0);
        let (accepted, _bonus) = speculative_verify(&draft, &probs, &target, &mut rng)
            .expect("matching draft and target prob dimensions");
        assert_eq!(accepted.len(), k);
    }

    /// PagedAttention: single-token sequence output equals V.
    #[test]
    fn e2e_paged_attention_single_token() {
        let n_h = 2;
        let hd = 4;
        let bs = 4;
        let mut cache = PagedKvCache::new(1, n_h, hd, bs, 4);
        let id = cache.alloc_block().expect("4-block cache has free blocks");
        let kv = vec![1.0_f32; n_h * hd];
        cache
            .append_token(id, 0, &kv, &kv)
            .expect("layer 0 exists and slot is free");
        let q = vec![1.0_f32; n_h * hd];
        let out = paged_attention_cpu(&q, &cache, &[id], 1, 0, n_h, n_h, hd, bs, 1.0)
            .expect("valid paged attention inputs");
        for &v in &out {
            assert!((v - 1.0_f32).abs() < 1e-5, "expected 1.0, got {v}");
        }
    }

    /// Prefix cache: hit rate is correctly computed across hits and misses.
    #[test]
    fn e2e_prefix_cache_hit_rate() {
        let mut cache = PrefixCache::new(32);
        let t = vec![1_u32, 2, 3, 4];
        cache.insert(&t, vec![BlockId(0), BlockId(1)]);
        cache.lookup(&[99_u32, 88]); // miss
        cache.lookup(&t); // hit
        cache.lookup(&t); // hit
        // 3 total queries, 2 hits → hit_rate = 2/3
        assert!(
            (cache.hit_rate() - 2.0 / 3.0).abs() < 0.01,
            "got {}",
            cache.hit_rate()
        );
    }

    /// MockModelRunner: decode returns correct shape.
    #[test]
    fn e2e_mock_runner_decode() {
        let runner = MockModelRunner::new(64, 0);
        let logits = runner
            .decode(&[1, 2, 3], &[vec![], vec![], vec![]], &[0, 0, 0])
            .expect("valid decode inputs with 3 sequences");
        assert_eq!(logits.len(), 3);
        assert!(logits.iter().all(|row| row.len() == 64));
    }

    /// End-to-end radix prefix reuse: two prompts sharing a system prefix reuse
    /// the prefix's KV blocks.
    #[test]
    fn e2e_radix_prefix_reuse() {
        let mut radix = RadixCache::new(2).expect("valid block size");
        // System prompt "100 101 102 103" → blocks 0,1.
        radix
            .insert(&[100, 101, 102, 103], vec![BlockId(0), BlockId(1)])
            .expect("insert system prefix");
        // A new request with the same prefix then user content reuses both blocks.
        let m = radix.match_prefix(&[100, 101, 102, 103, 200, 201]);
        assert_eq!(m.matched_len, 4, "shared system prefix matched");
        assert_eq!(m.blocks, vec![BlockId(0), BlockId(1)]);
        assert!(radix.hit_rate() > 0.0);
    }

    /// End-to-end grammar-constrained decode: the DFA masks every token that
    /// would break the grammar, and committing the only legal path completes it.
    #[test]
    fn e2e_grammar_constrained_decode() {
        // Accept exactly the bytes "ok".
        let dfa = Dfa::from_literal(b"ok");
        let vocab = vec![b"o".to_vec(), b"k".to_vec(), b"x".to_vec()];
        let mut g = GrammarConstraint::new(dfa, vocab).expect("live start");
        let mut logits = vec![1.0_f32; 3];
        g.mask_logits(&mut logits).expect("a legal token exists");
        // Only "o"(0) is legal from the start; "k"(1) and "x"(2) are masked.
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        g.commit(0).expect("commit 'o'");
        g.commit(1).expect("commit 'k'");
        assert!(g.is_complete(), "'ok' fully formed");
    }

    /// End-to-end chunked prefill: a long prompt drains across steps while a
    /// per-step token budget is respected (Sarathi piggybacking).
    #[test]
    fn e2e_chunked_prefill_drains() {
        let planner = ChunkPlanner::new(8).expect("valid budget");
        let mut prefill = ChunkedPrefillPlan::new(30, 16).expect("valid chunk size");
        let mut total = 0;
        let mut steps = 0;
        while !prefill.is_done() {
            let step = planner.pack_step(2, &mut prefill); // 2 decodes piggyback
            assert!(step.total_tokens <= 8, "budget respected");
            total += step.prefill_chunk.map_or(0, |c| c.len());
            steps += 1;
            assert!(steps < 100, "must terminate");
        }
        assert_eq!(total, 30, "entire 30-token prompt prefilled");
    }
}
