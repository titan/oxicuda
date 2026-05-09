use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxicuda_rlhf::handle::LcgRng;
use oxicuda_rlhf::prelude::*;

fn bench_ptx(c: &mut Criterion) {
    let sm_versions = [80_u32, 90, 100, 120];
    #[allow(clippy::type_complexity)]
    let kernel_fns: &[(&str, fn(u32) -> String)] = &[
        ("bt_reward_loss_ptx", bt_reward_loss_ptx),
        ("dpo_loss_ptx", dpo_loss_ptx),
        ("ipo_loss_ptx", ipo_loss_ptx),
        ("kto_loss_ptx", kto_loss_ptx),
        ("orpo_odds_ptx", orpo_odds_ptx),
        ("rlhf_kl_ptx", rlhf_kl_ptx),
        ("sft_mask_ptx", sft_mask_ptx),
    ];
    let mut g = c.benchmark_group("ptx_generation");
    for (name, gen_fn) in kernel_fns {
        for &sm in &sm_versions {
            g.bench_with_input(BenchmarkId::new(*name, sm), &sm, |b, &sm| {
                b.iter(|| gen_fn(sm))
            });
        }
    }
    g.finish();
}

fn bench_dpo(c: &mut Criterion) {
    let n = 256_usize;
    let mut rng = LcgRng::new(1);
    let chosen_logps: Vec<f32> = (0..n).map(|_| -(rng.next_f32() * 5.0 + 0.1)).collect();
    let rejected_logps: Vec<f32> = (0..n).map(|_| -(rng.next_f32() * 5.0 + 0.1)).collect();
    let ref_chosen_logps: Vec<f32> = (0..n).map(|_| -(rng.next_f32() * 5.0 + 0.1)).collect();
    let ref_rejected_logps: Vec<f32> = (0..n).map(|_| -(rng.next_f32() * 5.0 + 0.1)).collect();
    let batch = PairBatch::new(
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
    )
    .unwrap();
    let cfg = DpoConfig { beta: 0.1 };
    c.bench_function("dpo_loss_n256", |b| {
        b.iter(|| std::hint::black_box(dpo_loss(&batch, &cfg).unwrap()))
    });
}

fn bench_ipo(c: &mut Criterion) {
    let n = 256_usize;
    let mut rng = LcgRng::new(2);
    let chosen_logps: Vec<f32> = (0..n).map(|_| -(rng.next_f32() * 5.0 + 0.1)).collect();
    let rejected_logps: Vec<f32> = (0..n).map(|_| -(rng.next_f32() * 5.0 + 0.1)).collect();
    let ref_chosen_logps: Vec<f32> = (0..n).map(|_| -(rng.next_f32() * 5.0 + 0.1)).collect();
    let ref_rejected_logps: Vec<f32> = (0..n).map(|_| -(rng.next_f32() * 5.0 + 0.1)).collect();
    let batch = PairBatch::new(
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
    )
    .unwrap();
    let cfg = IpoConfig { beta: 0.1 };
    c.bench_function("ipo_loss_n256", |b| {
        b.iter(|| std::hint::black_box(ipo_loss(&batch, &cfg).unwrap()))
    });
}

fn bench_kto(c: &mut Criterion) {
    let n = 256_usize;
    let mut rng = LcgRng::new(3);
    let desirable: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let undesirable: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let cfg = KtoConfig {
        beta: 0.1,
        lambda_d: 1.0,
        lambda_u: 1.0,
    };
    c.bench_function("kto_loss_n256", |b| {
        b.iter(|| std::hint::black_box(kto_loss(&desirable, &undesirable, &cfg).unwrap()))
    });
}

fn bench_sft(c: &mut Criterion) {
    let n_tokens = 512_usize;
    let n_vocab = 32_768_usize;
    let mut rng = LcgRng::new(4);
    let logits: Vec<f32> = (0..n_tokens * n_vocab)
        .map(|_| rng.next_f32() * 4.0 - 2.0)
        .collect();
    let labels: Vec<u32> = (0..n_tokens)
        .map(|_| rng.next_usize(n_vocab) as u32)
        .collect();
    let mask = vec![1_u8; n_tokens];
    c.bench_function("sft_loss_t512_v32k", |b| {
        b.iter(|| std::hint::black_box(sft_loss(&logits, &labels, &mask, n_vocab).unwrap()))
    });
}

fn bench_reward_norm(c: &mut Criterion) {
    let n = 10_000_usize;
    let mut rng = LcgRng::new(5);
    let values: Vec<f32> = (0..n).map(|_| rng.next_f32() * 10.0 - 5.0).collect();
    c.bench_function("reward_normalizer_update_n10k", |b| {
        b.iter(|| {
            let mut norm = RewardNormalizer::new();
            for &v in &values {
                norm.update(std::hint::black_box(v));
            }
            std::hint::black_box(norm)
        })
    });
}

criterion_group!(
    benches,
    bench_ptx,
    bench_dpo,
    bench_ipo,
    bench_kto,
    bench_sft,
    bench_reward_norm
);
criterion_main!(benches);
