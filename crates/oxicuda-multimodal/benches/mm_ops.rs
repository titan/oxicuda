use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_multimodal::handle::LcgRng;
use oxicuda_multimodal::prelude::*;

// ─── PTX kernel benchmarks ───────────────────────────────────────────────────

fn bench_cross_attn_score_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("cross_attn_score_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(cross_attn_score_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_modal_align_loss_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("modal_align_loss_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(modal_align_loss_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_bilinear_pool_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("bilinear_pool_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(bilinear_pool_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_temporal_pool_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("temporal_pool_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(temporal_pool_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_token_merge_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("token_merge_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(token_merge_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_gate_fusion_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("gate_fusion_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(gate_fusion_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_itm_bce_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("itm_bce_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(itm_bce_ptx(sm)))
        });
    }
    g.finish();
}

// ─── Algorithm benchmarks ────────────────────────────────────────────────────

fn bench_clip_loss_b64_d256(c: &mut Criterion) {
    let n = 64;
    let dim = 256;
    let mut rng = LcgRng::new(0);
    let mut img = vec![0.0_f32; n * dim];
    let mut txt = vec![0.0_f32; n * dim];
    rng.fill_normal(&mut img);
    rng.fill_normal(&mut txt);

    c.bench_function("clip_loss_b64_d256", |b| {
        b.iter(|| {
            std::hint::black_box(
                clip_loss(
                    std::hint::black_box(&img),
                    std::hint::black_box(&txt),
                    n,
                    dim,
                    0.07,
                )
                .expect("ok"),
            )
        })
    });
}

fn bench_mlb_fusion_b32_d512(c: &mut Criterion) {
    let batch = 32;
    let d_v = 512;
    let d_q = 512;
    let d_joint = 256;
    let d_out = 128;
    let fuser = MlbFusion::zeros(d_v, d_q, d_joint, d_out);
    let mut rng = LcgRng::new(1);
    let mut v = vec![0.0_f32; batch * d_v];
    let mut q = vec![0.0_f32; batch * d_q];
    rng.fill_normal(&mut v);
    rng.fill_normal(&mut q);

    c.bench_function("mlb_fusion_b32_d512", |b| {
        b.iter(|| {
            std::hint::black_box(
                fuser
                    .forward(std::hint::black_box(&v), std::hint::black_box(&q), batch)
                    .expect("ok"),
            )
        })
    });
}

fn bench_cross_attn_heads8_d64_len32(c: &mut Criterion) {
    let cfg = CrossAttnConfig::new(8, 64, 0.0).expect("ok");
    let d = cfg.d_model;
    let q_len = 32;
    let kv_len = 32;
    let weights = CrossAttnWeights::identity(&cfg);
    let attn = CrossAttention::with_weights(cfg, weights);
    let query = vec![0.1_f32; q_len * d];
    let kv = vec![0.2_f32; kv_len * d];

    c.bench_function("cross_attn_heads8_d64_len32", |b| {
        b.iter(|| {
            std::hint::black_box(
                attn.forward(
                    std::hint::black_box(&query),
                    std::hint::black_box(&kv),
                    std::hint::black_box(&kv),
                    q_len,
                    kv_len,
                )
                .expect("ok"),
            )
        })
    });
}

fn bench_bert_tiny_forward(c: &mut Criterion) {
    let cfg = BertConfig::tiny();
    let weights = BertWeights::zeros(&cfg);
    let token_ids: Vec<u32> = (0..8).map(|i| i as u32 % cfg.vocab_size as u32).collect();

    c.bench_function("bert_tiny_forward", |b| {
        b.iter(|| {
            std::hint::black_box(
                BertEncoder::forward(
                    std::hint::black_box(&token_ids),
                    std::hint::black_box(&weights),
                    std::hint::black_box(&cfg),
                )
                .expect("ok"),
            )
        })
    });
}

fn bench_vit_tiny_forward(c: &mut Criterion) {
    let cfg = ViTEncoderConfig::tiny();
    let weights = ViTEncoderWeights::zeros(&cfg);
    let image = vec![0.5_f32; 3 * 32 * 32];

    c.bench_function("vit_tiny_forward", |b| {
        b.iter(|| {
            std::hint::black_box(
                ViTEncoder::forward(
                    std::hint::black_box(&image),
                    std::hint::black_box(&cfg),
                    std::hint::black_box(&weights),
                )
                .expect("ok"),
            )
        })
    });
}

criterion_group!(
    benches,
    bench_cross_attn_score_ptx,
    bench_modal_align_loss_ptx,
    bench_bilinear_pool_ptx,
    bench_temporal_pool_ptx,
    bench_token_merge_ptx,
    bench_gate_fusion_ptx,
    bench_itm_bce_ptx,
    bench_clip_loss_b64_d256,
    bench_mlb_fusion_b32_d512,
    bench_cross_attn_heads8_d64_len32,
    bench_bert_tiny_forward,
    bench_vit_tiny_forward,
);
criterion_main!(benches);
