use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxicuda_moe::prelude::*;

fn bench_ptx(c: &mut Criterion) {
    let sm_versions = [80_u32, 90, 100, 120];
    #[allow(clippy::type_complexity)]
    let kernel_fns: &[(&str, fn(u32) -> String)] = &[
        ("top_k_gate_ptx", top_k_gate_ptx),
        ("expert_dispatch_ptx", expert_dispatch_ptx),
        ("expert_ffn_ptx", expert_ffn_ptx),
        ("expert_combine_ptx", expert_combine_ptx),
        ("load_balance_loss_ptx", load_balance_loss_ptx),
        ("router_z_loss_ptx", router_z_loss_ptx),
        ("soft_moe_dispatch_ptx", soft_moe_dispatch_ptx),
    ];
    let mut group = c.benchmark_group("ptx_generation");
    for (name, gen_fn) in kernel_fns {
        for &sm in &sm_versions {
            group.bench_with_input(BenchmarkId::new(*name, sm), &sm, |b, &sm_ver| {
                b.iter(|| gen_fn(sm_ver))
            });
        }
    }
    group.finish();
}

fn bench_topk_routing(c: &mut Criterion) {
    let mut rng = LcgRng::new(0);
    let n_tokens = 512_usize;
    let n_experts = 8_usize;
    let input_dim = 256_usize;
    let cfg = TopKConfig {
        k: 2,
        n_experts,
        input_dim,
        noise_std: 0.0,
    };
    let router = TopKRouter::new(cfg, &mut rng).unwrap();
    let x = vec![0.5_f32; n_tokens * input_dim];

    c.bench_function("topk_routing_512tok_8exp_d256", |b| {
        b.iter(|| std::hint::black_box(router.route(std::hint::black_box(&x), n_tokens).unwrap()))
    });
}

fn bench_expert_ffn(c: &mut Criterion) {
    let mut rng = LcgRng::new(1);
    let input_dim = 256_usize;
    let ffn_dim = 1024_usize;
    let batch_size = 64_usize;
    let ffn = ExpertFfn::new(input_dim, ffn_dim, ExpertActivation::Gelu, &mut rng);
    let x = vec![0.5_f32; batch_size * input_dim];

    c.bench_function("expert_ffn_batch64_d256_ffn1024", |b| {
        b.iter(|| {
            std::hint::black_box(
                ffn.forward_batch(std::hint::black_box(&x), batch_size)
                    .unwrap(),
            )
        })
    });
}

fn bench_switch_dispatch(c: &mut Criterion) {
    let n_tokens = 512_usize;
    let n_experts = 8_usize;
    let cfg = SwitchConfig {
        n_experts,
        input_dim: 256,
        capacity_factor: 1.25,
        min_capacity: 1,
        drop_tokens: true,
    };
    let gate_indices: Vec<usize> = (0..n_tokens).map(|t| t % n_experts).collect();

    c.bench_function("switch_dispatch_512tok_8exp", |b| {
        b.iter(|| {
            std::hint::black_box(
                switch_dispatch(
                    std::hint::black_box(&gate_indices),
                    n_tokens,
                    std::hint::black_box(&cfg),
                )
                .unwrap(),
            )
        })
    });
}

fn bench_load_balance(c: &mut Criterion) {
    let n_tokens = 512_usize;
    let n_experts = 8_usize;
    let mut rng = LcgRng::new(2);
    let mut logits = vec![0.0_f32; n_tokens * n_experts];
    rng.fill_normal_scaled(&mut logits, 1.0);
    let assignments: Vec<usize> = (0..n_tokens).map(|t| t % n_experts).collect();

    c.bench_function("load_balance_loss_512tok_8exp", |b| {
        b.iter(|| {
            std::hint::black_box(
                load_balance_loss(
                    std::hint::black_box(&logits),
                    std::hint::black_box(&assignments),
                    n_tokens,
                    n_experts,
                )
                .unwrap(),
            )
        })
    });
}

fn bench_moe_layer(c: &mut Criterion) {
    let mut rng = LcgRng::new(3);
    let input_dim = 64_usize;
    let n_tokens = 128_usize;
    let layer_cfg = MoeLayerConfig {
        input_dim,
        ffn_dim: 256,
        n_experts: 8,
        top_k: 1,
        capacity_factor: 1.25,
        load_balance_coef: 0.01,
        router_z_loss_coef: 0.001,
        activation: ExpertActivation::Gelu,
    };
    let layer = MoeLayer::new(layer_cfg, &mut rng).unwrap();
    let x = vec![0.5_f32; n_tokens * input_dim];

    c.bench_function("moe_layer_forward_128tok_8exp_d64", |b| {
        b.iter(|| std::hint::black_box(layer.forward(std::hint::black_box(&x), n_tokens).unwrap()))
    });
}

criterion_group!(
    benches,
    bench_ptx,
    bench_topk_routing,
    bench_expert_ffn,
    bench_switch_dispatch,
    bench_load_balance,
    bench_moe_layer
);
criterion_main!(benches);
