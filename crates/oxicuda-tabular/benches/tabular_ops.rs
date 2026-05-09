use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxicuda_tabular::prelude::*;

// ─── PTX kernel generation benchmarks ────────────────────────────────────────

fn bench_ptx(c: &mut Criterion) {
    let sm_versions = [80_u32, 90, 100, 120];
    #[allow(clippy::type_complexity)]
    let kernel_fns: &[(&str, fn(u32) -> String)] = &[
        ("sparsemax_ptx", sparsemax_ptx),
        ("feature_tokenize_ptx", feature_tokenize_ptx),
        ("tabnet_step_attn_ptx", tabnet_step_attn_ptx),
        ("intersample_attn_ptx", intersample_attn_ptx),
        ("node_tree_eval_ptx", node_tree_eval_ptx),
        ("quantile_norm_ptx", quantile_norm_ptx),
        ("auc_roc_ptx", auc_roc_ptx),
    ];
    let mut group = c.benchmark_group("ptx_generation");
    for &(name, gen_fn) in kernel_fns {
        for &sm in &sm_versions {
            group.bench_with_input(BenchmarkId::new(name, sm), &sm, |b, &sm_ver| {
                b.iter(|| std::hint::black_box(gen_fn(sm_ver)))
            });
        }
    }
    group.finish();
}

// ─── Algorithm benchmarks ─────────────────────────────────────────────────────

fn bench_sparsemax_d256(c: &mut Criterion) {
    let z = vec![0.01_f32; 256];
    c.bench_function("bench_sparsemax_d256", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                std::hint::black_box(sparsemax(std::hint::black_box(&z)).unwrap());
            }
        })
    });
}

fn bench_ft_transformer_forward(c: &mut Criterion) {
    let mut rng = LcgRng::new(42);
    let cfg = FtConfig {
        n_cont_features: 4,
        cat_n_categories: vec![5, 3],
        embed_dim: 32,
        n_heads: 4,
        n_layers: 3,
        ffn_hidden: 64,
        dropout_rate: 0.0,
        n_classes: 2,
    };
    let model = FtTransformer::new(cfg, &mut rng).unwrap();
    let x_cont = vec![0.5_f32; 4];
    let x_cat = vec![1usize, 0];

    c.bench_function("bench_ft_transformer_forward", |b| {
        b.iter(|| {
            std::hint::black_box(
                model
                    .forward(std::hint::black_box(&x_cont), std::hint::black_box(&x_cat))
                    .unwrap(),
            )
        })
    });
}

fn bench_tabnet_forward(c: &mut Criterion) {
    let mut rng = LcgRng::new(7);
    let cfg = TabNetConfig {
        n_features: 32,
        n_d: 16,
        n_a: 16,
        n_steps: 5,
        gamma: 1.5,
        n_classes: 2,
    };
    let layer = TabNetLayer::new(cfg, &mut rng).unwrap();
    let x = vec![0.3_f32; 32];

    c.bench_function("bench_tabnet_forward", |b| {
        b.iter(|| std::hint::black_box(layer.forward(std::hint::black_box(&x)).unwrap()))
    });
}

fn bench_node_ensemble_forward(c: &mut Criterion) {
    let mut rng = LcgRng::new(11);
    let cfg = NodeConfig {
        n_trees: 20,
        depth: 4,
        input_dim: 32,
        output_dim: 1,
    };
    let ensemble = NodeEnsemble::new(cfg, &mut rng).unwrap();
    let x = vec![0.5_f32; 32];

    c.bench_function("bench_node_ensemble_forward", |b| {
        b.iter(|| std::hint::black_box(ensemble.forward(std::hint::black_box(&x)).unwrap()))
    });
}

fn bench_quantile_normalizer(c: &mut Criterion) {
    let mut rng = LcgRng::new(99);
    let n_samples = 1024;
    let n_features = 32;
    let mut data = vec![0.0_f32; n_samples * n_features];
    rng.fill_normal_scaled(&mut data, 2.0);

    let norm = QuantileNormalizer::fit(&data, n_samples, n_features).unwrap();
    // Build a test batch
    let test_batch: Vec<f32> = data.clone();

    c.bench_function("bench_quantile_normalizer", |b| {
        b.iter(|| {
            for s in 0..n_samples {
                let row = &test_batch[s * n_features..(s + 1) * n_features];
                std::hint::black_box(norm.transform(std::hint::black_box(row)).unwrap());
            }
        })
    });
}

criterion_group!(
    benches,
    bench_ptx,
    bench_sparsemax_d256,
    bench_ft_transformer_forward,
    bench_tabnet_forward,
    bench_node_ensemble_forward,
    bench_quantile_normalizer,
);
criterion_main!(benches);
