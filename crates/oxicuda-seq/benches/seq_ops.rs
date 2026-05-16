use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_seq::alignment::{ScoringMatrix, hirschberg_align, needleman_wunsch, smith_waterman};
use oxicuda_seq::crf::{LinearChainCrf, viterbi_decode};
use oxicuda_seq::hmm::{HmmDiscrete, viterbi};
use oxicuda_seq::ptx_kernels::{
    beam_topk_ptx, crf_features_ptx, edit_dist_ptx, forward_pass_ptx, kalman_predict_ptx,
    mrf_gibbs_ptx, viterbi_step_ptx,
};

type KernelEntry = (&'static str, fn(u32) -> String);

fn bench_ptx(c: &mut Criterion) {
    let sm_versions = [75u32, 80, 89, 90];
    let kernels: &[KernelEntry] = &[
        ("forward_pass", forward_pass_ptx),
        ("viterbi_step", viterbi_step_ptx),
        ("crf_features", crf_features_ptx),
        ("beam_topk", beam_topk_ptx),
        ("edit_dist", edit_dist_ptx),
        ("kalman_predict", kalman_predict_ptx),
        ("mrf_gibbs", mrf_gibbs_ptx),
    ];
    for &sm in &sm_versions {
        for &(name, f) in kernels {
            c.bench_function(&format!("ptx_{name}_sm{sm}"), |b| b.iter(|| f(sm)));
        }
    }
}

fn bench_viterbi(c: &mut Criterion) {
    let h = HmmDiscrete::new(
        4,
        3,
        vec![0.25; 4],
        {
            let mut a = vec![0.0; 16];
            for i in 0..4 {
                for j in 0..4 {
                    a[i * 4 + j] = if i == j { 0.5 } else { 0.5 / 3.0 };
                }
            }
            a
        },
        {
            let mut b = vec![0.0; 12];
            for i in 0..4 {
                for j in 0..3 {
                    b[i * 3 + j] = 1.0 / 3.0;
                }
            }
            b
        },
    )
    .expect("ok");
    let obs: Vec<usize> = (0..40).map(|i| i % 3).collect();
    c.bench_function("viterbi_4state_40obs", |b| {
        b.iter(|| viterbi(&h, &obs).expect("ok"))
    });
}

fn bench_alignment(c: &mut Criterion) {
    let seq_a: Vec<u8> = (0..50).map(|i| b"ACGT"[i % 4]).collect();
    let seq_b: Vec<u8> = (0..50).map(|i| b"ACGT"[(i + 1) % 4]).collect();
    let sc = ScoringMatrix::default();
    c.bench_function("nw_50x50", |b| {
        b.iter(|| needleman_wunsch(&seq_a, &seq_b, &sc).expect("ok"))
    });
    c.bench_function("sw_50x50", |b| {
        b.iter(|| smith_waterman(&seq_a, &seq_b, &sc).expect("ok"))
    });
    c.bench_function("hirschberg_50x50", |b| {
        b.iter(|| hirschberg_align(&seq_a, &seq_b, &sc).expect("ok"))
    });
}

fn bench_crf_decode(c: &mut Criterion) {
    let mut crf = LinearChainCrf::zeros(4, 6).expect("ok");
    for v in crf.emissions.iter_mut() {
        *v = 0.1;
    }
    for v in crf.transitions.iter_mut() {
        *v = -0.05;
    }
    let x: Vec<f64> = (0..30 * 6).map(|i| (i as f64) * 0.01).collect();
    c.bench_function("crf_viterbi_4lab_30step", |b| {
        b.iter(|| viterbi_decode(&crf, &x).expect("ok"))
    });
}

criterion_group!(
    benches,
    bench_ptx,
    bench_viterbi,
    bench_alignment,
    bench_crf_decode
);
criterion_main!(benches);
