use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxicuda_meta::prelude::*;

#[allow(clippy::type_complexity)]
fn bench_ptx(c: &mut Criterion) {
    let sm_versions = [80_u32, 90, 100, 120];
    let kernel_fns: &[(&str, fn(u32) -> String)] = &[
        ("inner_sgd_ptx", inner_sgd_ptx),
        ("reptile_update_ptx", reptile_update_ptx),
        ("proto_distance_ptx", proto_distance_ptx),
        ("cosine_sim_ptx", cosine_sim_ptx),
        ("relation_score_ptx", relation_score_ptx),
        ("meta_grad_accum_ptx", meta_grad_accum_ptx),
        ("episode_sample_ptx", episode_sample_ptx),
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

fn bench_proto_net(c: &mut Criterion) {
    let n_way = 5;
    let k_shot = 5;
    let feat_dim = 64;
    let n_query = 15;

    let mut rng = LcgRng::new(1234);
    let support_feats: Vec<f32> = (0..n_way * k_shot * feat_dim)
        .map(|_| rng.next_f32())
        .collect();
    let support_y: Vec<u32> = (0..n_way)
        .flat_map(|c| std::iter::repeat_n(c as u32, k_shot))
        .collect();
    let query_feats: Vec<f32> = (0..n_way * n_query * feat_dim)
        .map(|_| rng.next_f32())
        .collect();

    let mut g = c.benchmark_group("proto_net");
    g.bench_function("compute_prototypes_5way5shot_64dim", |b| {
        b.iter(|| compute_prototypes(&support_feats, &support_y, n_way, k_shot, feat_dim).unwrap())
    });
    let protos = compute_prototypes(&support_feats, &support_y, n_way, k_shot, feat_dim).unwrap();
    g.bench_function("proto_predict_5way_75query_64dim", |b| {
        b.iter(|| proto_predict(&query_feats, &protos, n_way, feat_dim).unwrap())
    });
    g.bench_function("proto_loss_5way_75query_64dim", |b| {
        let query_y: Vec<u32> = (0..n_way)
            .flat_map(|c| std::iter::repeat_n(c as u32, n_query))
            .collect();
        b.iter(|| proto_loss(&query_feats, &query_y, &protos, n_way, feat_dim).unwrap())
    });
    g.finish();
}

fn bench_matching_net(c: &mut Criterion) {
    let n_way = 5;
    let k_shot = 5;
    let feat_dim = 64;

    let mut rng = LcgRng::new(4321);
    let support_feats: Vec<f32> = (0..n_way * k_shot * feat_dim)
        .map(|_| rng.next_f32())
        .collect();
    let support_y: Vec<u32> = (0..n_way)
        .flat_map(|c| std::iter::repeat_n(c as u32, k_shot))
        .collect();
    let query_feat: Vec<f32> = (0..feat_dim).map(|_| rng.next_f32()).collect();

    let mut g = c.benchmark_group("matching_net");
    g.bench_function("attention_5way5shot_64dim", |b| {
        b.iter(|| {
            matching_net_attention(&query_feat, &support_feats, &support_y, n_way, 1.0).unwrap()
        })
    });
    g.finish();
}

fn bench_maml_adapt(c: &mut Criterion) {
    let n_classes = 5;
    let feat_dim = 16;
    let n_params = n_classes * feat_dim + n_classes;

    let mut rng = LcgRng::new(777);
    let params: Vec<f32> = (0..n_params).map(|_| rng.next_f32() - 0.5).collect();
    let support_x: Vec<f32> = (0..n_classes * feat_dim).map(|_| rng.next_f32()).collect();
    let support_y: Vec<u32> = (0..n_classes as u32).collect();
    let cfg = MamlConfig {
        inner_lr: 0.01,
        n_inner_steps: 3,
    };

    let mut g = c.benchmark_group("maml_adapt");
    g.bench_function("5way_16dim_3steps", |b| {
        b.iter(|| maml_adapt(&params, &support_x, &support_y, n_classes, feat_dim, &cfg).unwrap())
    });
    g.finish();
}

fn bench_reptile_update(c: &mut Criterion) {
    let n_classes = 5;
    let feat_dim = 16;
    let n_params = n_classes * feat_dim + n_classes;

    let mut rng = LcgRng::new(888);
    let params: Vec<f32> = (0..n_params).map(|_| rng.next_f32() - 0.5).collect();
    let task_data: Vec<(Vec<f32>, Vec<u32>)> = (0..4)
        .map(|_| {
            let sx: Vec<f32> = (0..n_classes * feat_dim).map(|_| rng.next_f32()).collect();
            let sy: Vec<u32> = (0..n_classes as u32).collect();
            (sx, sy)
        })
        .collect();
    let cfg = ReptileConfig {
        inner_lr: 0.01,
        n_inner_steps: 3,
        step_size: 0.1,
    };

    let mut g = c.benchmark_group("reptile_update");
    g.bench_function("4tasks_5way_16dim", |b| {
        b.iter(|| reptile_update(&params, &task_data, n_classes, feat_dim, &cfg).unwrap())
    });
    g.finish();
}

fn bench_episode_sampler(c: &mut Criterion) {
    let cfg = EpisodeConfig {
        n_way: 5,
        k_shot: 5,
        n_query: 15,
        feat_dim: 64,
    };
    let sampler = EpisodeSampler::new(cfg.clone()).unwrap();
    let n_classes = 100_usize;
    let n_per_class = cfg.k_shot + cfg.n_query + 10;
    let n_total = n_classes * n_per_class;

    let mut rng = LcgRng::new(999);
    let data: Vec<f32> = (0..n_total * cfg.feat_dim)
        .map(|_| rng.next_f32())
        .collect();
    let labels: Vec<u32> = (0..n_total).map(|i| (i % n_classes) as u32).collect();

    let mut g = c.benchmark_group("episode_sampler");
    g.bench_function("sample_5way5shot15q_100classes_64dim", |b| {
        let mut bench_rng = LcgRng::new(2024);
        b.iter(|| {
            sampler
                .sample(&data, &labels, n_classes, &mut bench_rng)
                .unwrap()
        })
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_ptx,
    bench_proto_net,
    bench_matching_net,
    bench_maml_adapt,
    bench_reptile_update,
    bench_episode_sampler
);
criterion_main!(benches);
