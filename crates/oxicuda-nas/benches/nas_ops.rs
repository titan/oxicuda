use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_nas::handle::LcgRng;
use oxicuda_nas::prelude::*;

fn bench_arch_grad_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("arch_grad_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(arch_grad_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_arch_softmax_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("arch_softmax_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(arch_softmax_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_crossover_uniform_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("crossover_uniform_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(crossover_uniform_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_flops_accumulate_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("flops_accumulate_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(flops_accumulate_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_gumbel_softmax_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("gumbel_softmax_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(gumbel_softmax_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_mixed_op_blend_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("mixed_op_blend_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(mixed_op_blend_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_pareto_dominate_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("pareto_dominate_ptx");
    for &sm in &[80u32, 90, 100, 120] {
        g.bench_function(format!("sm_{sm}"), |b| {
            b.iter(|| std::hint::black_box(pareto_dominate_ptx(sm)))
        });
    }
    g.finish();
}

fn bench_population_random(c: &mut Criterion) {
    c.bench_function("population_random_50_8edges_8ops_2obj", |b| {
        b.iter(|| {
            let mut rng = LcgRng::new(7);
            std::hint::black_box(Population::random(50, 8, 8, 2, &mut rng).expect("ok"))
        })
    });
}

fn bench_nsga2_select(c: &mut Criterion) {
    use oxicuda_nas::evolution::nsga2::Individual;
    let mut rng = LcgRng::new(11);
    let mut individuals: Vec<Individual> = Vec::with_capacity(50);
    for _ in 0..50 {
        let encoding: Vec<usize> = (0..8).map(|_| rng.next_usize(8)).collect();
        let objectives = vec![rng.next_f32(), rng.next_f32()];
        individuals.push(Individual {
            encoding,
            objectives,
            rank: 0,
            crowding_distance: 0.0,
        });
    }
    c.bench_function("nsga2_select_50_to_25", |b| {
        b.iter(|| {
            std::hint::black_box(
                nsga2_select(std::hint::black_box(individuals.clone()), 25).expect("ok"),
            )
        })
    });
}

fn bench_path_sample(c: &mut Criterion) {
    let mut sampler = PathSampler::new(8, 8, SamplingStrategy::Uniform);
    let mut rng = LcgRng::new(13);
    c.bench_function("path_sampler_uniform_8edges_8ops", |b| {
        b.iter(|| std::hint::black_box(sampler.sample(&mut rng).expect("ok")))
    });
}

fn bench_mixed_op_blend(c: &mut Criterion) {
    let mut rng = LcgRng::new(17);
    let op_kinds: Vec<OpKind> = vec![
        OpKind::Identity,
        OpKind::SepConv3x3,
        OpKind::SepConv5x5,
        OpKind::DilConv3x3,
        OpKind::AvgPool3x3,
        OpKind::MaxPool3x3,
    ];
    let n_ops = op_kinds.len();
    let mixed = MixedOp::new(op_kinds, &mut rng);
    let in_ch = 4usize;
    let h = 8usize;
    let w = 8usize;
    let out_ch = 4usize;
    let input = vec![0.1_f32; in_ch * h * w];
    let weights: Vec<OpWeights> = (0..n_ops)
        .map(|_| OpWeights::random(in_ch, out_ch, 3, &mut rng))
        .collect();
    c.bench_function("mixed_op_forward_4chx8x8_6ops", |b| {
        b.iter(|| {
            std::hint::black_box(
                mixed
                    .forward_cpu(
                        std::hint::black_box(&input),
                        in_ch,
                        h,
                        w,
                        out_ch,
                        std::hint::black_box(&weights),
                    )
                    .expect("ok"),
            )
        })
    });
}

criterion_group!(
    benches,
    bench_arch_grad_ptx,
    bench_arch_softmax_ptx,
    bench_crossover_uniform_ptx,
    bench_flops_accumulate_ptx,
    bench_gumbel_softmax_ptx,
    bench_mixed_op_blend_ptx,
    bench_pareto_dominate_ptx,
    bench_population_random,
    bench_nsga2_select,
    bench_path_sample,
    bench_mixed_op_blend,
);
criterion_main!(benches);
