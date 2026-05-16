use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_sketch::cardinality::HyperLogLog;
use oxicuda_sketch::frequency::CountMinSketch;
use oxicuda_sketch::handle::LcgRng;
use oxicuda_sketch::membership::BloomFilter;
use oxicuda_sketch::ptx_kernels::{
    bloom_insert_ptx, cm_query_ptx, cm_update_ptx, hll_register_ptx, minhash_sketch_ptx,
    reservoir_sample_ptx, tdigest_merge_ptx,
};
use oxicuda_sketch::quantile::KllSketch;
use oxicuda_sketch::quantile::TDigest;
use oxicuda_sketch::sampling::ReservoirSampler;
use oxicuda_sketch::similarity::MinHash;
use oxicuda_sketch::topk::SpaceSaving;

type KernelEntry = (&'static str, fn(u32) -> String);

fn bench_ptx(c: &mut Criterion) {
    let sm_versions = [75u32, 80, 89, 90];
    let kernels: &[KernelEntry] = &[
        ("cm_update", cm_update_ptx),
        ("cm_query", cm_query_ptx),
        ("hll_register", hll_register_ptx),
        ("bloom_insert", bloom_insert_ptx),
        ("minhash_sketch", minhash_sketch_ptx),
        ("tdigest_merge", tdigest_merge_ptx),
        ("reservoir_sample", reservoir_sample_ptx),
    ];
    for &sm in &sm_versions {
        for &(name, f) in kernels {
            c.bench_function(&format!("ptx_{name}_sm{sm}"), |b| b.iter(|| f(sm)));
        }
    }
}

fn bench_hll(c: &mut Criterion) {
    c.bench_function("hll_add_1000_p12", |b| {
        b.iter(|| {
            let mut h = HyperLogLog::new(12, 0).expect("ok");
            for i in 0..1000u64 {
                h.add_u64(i);
            }
            h.estimate()
        })
    });
}

fn bench_cms(c: &mut Criterion) {
    c.bench_function("cms_update_1000", |b| {
        b.iter(|| {
            let mut rng = LcgRng::new(0);
            let mut cms = CountMinSketch::new(5, 256, &mut rng).expect("ok");
            for i in 0..1000u64 {
                cms.add(i % 100);
            }
            cms.query(7)
        })
    });
}

fn bench_bloom(c: &mut Criterion) {
    c.bench_function("bloom_insert_1000", |b| {
        b.iter(|| {
            let mut bf = BloomFilter::new(8192, 5, 0).expect("ok");
            for i in 0..1000u64 {
                bf.insert(i);
            }
            bf.contains(42)
        })
    });
}

fn bench_kll(c: &mut Criterion) {
    c.bench_function("kll_add_1000_k128", |b| {
        b.iter(|| {
            let mut k = KllSketch::new(128, 0).expect("ok");
            for i in 0..1000 {
                k.add(i as f64);
            }
            k.quantile(0.5)
        })
    });
}

fn bench_tdigest(c: &mut Criterion) {
    c.bench_function("tdigest_add_1000_delta100", |b| {
        b.iter(|| {
            let mut td = TDigest::new(100.0).expect("ok");
            for i in 0..1000 {
                td.add(i as f64);
            }
            td.flush();
            td.quantile(0.95)
        })
    });
}

fn bench_minhash(c: &mut Criterion) {
    c.bench_function("minhash_add_500_k64", |b| {
        b.iter(|| {
            let mut rng = LcgRng::new(11);
            let mut mh = MinHash::new(64, &mut rng).expect("ok");
            for i in 0..500u64 {
                mh.add(i);
            }
            mh.signature[0]
        })
    });
}

fn bench_reservoir(c: &mut Criterion) {
    c.bench_function("reservoir_add_1000_k50", |b| {
        b.iter(|| {
            let mut r = ReservoirSampler::new(50, 0).expect("ok");
            for i in 0..1000u64 {
                r.add(i);
            }
            r.sample().len()
        })
    });
}

fn bench_space_saving(c: &mut Criterion) {
    c.bench_function("space_saving_add_1000_k10", |b| {
        b.iter(|| {
            let mut s = SpaceSaving::new(10).expect("ok");
            for i in 0..1000u64 {
                s.add(i % 30);
            }
            s.top_k().len()
        })
    });
}

criterion_group!(
    benches,
    bench_ptx,
    bench_hll,
    bench_cms,
    bench_bloom,
    bench_kll,
    bench_tdigest,
    bench_minhash,
    bench_reservoir,
    bench_space_saving
);
criterion_main!(benches);
