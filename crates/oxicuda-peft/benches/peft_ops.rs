use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_peft::handle::LcgRng;
use oxicuda_peft::lora::lora::{LoraConfig, LoraLinear};
use oxicuda_peft::ptx_kernels::*;

fn bench_ptx(c: &mut Criterion) {
    let mut g = c.benchmark_group("peft_ptx");
    for sm in [75u32, 80, 89, 100] {
        g.bench_function(format!("lora_matmul_sm{sm}"), |b| {
            b.iter(|| lora_matmul_ptx(sm))
        });
        g.bench_function(format!("nf4_dequant_sm{sm}"), |b| {
            b.iter(|| nf4_dequant_ptx(sm))
        });
    }
    g.finish();
}

fn bench_algo(c: &mut Criterion) {
    let mut g = c.benchmark_group("peft_algo");
    let mut rng = LcgRng::new(42);
    let cfg = LoraConfig {
        r: 8,
        alpha: 16.0,
        init_scale: 0.01,
    };
    let lora = LoraLinear::new(64, 64, &cfg, &mut rng);
    let x: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
    g.bench_function("lora_forward_64x64_r8", |b| b.iter(|| lora.forward(&x)));
    g.finish();
}

criterion_group!(benches, bench_ptx, bench_algo);
criterion_main!(benches);
