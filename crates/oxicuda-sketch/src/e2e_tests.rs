//! End-to-end integration tests for `oxicuda-sketch`.

use crate::cardinality::HyperLogLog;
use crate::cardinality::HyperLogLogPlus;
use crate::cardinality::LinearCounter;
use crate::frequency::CountMinSketch;
use crate::frequency::CountSketch;
use crate::handle::LcgRng;
use crate::lsh::CosineLsh;
use crate::lsh::JaccardLsh;
use crate::lsh::LshIndex;
use crate::membership::BloomFilter;
use crate::membership::CuckooFilter;
use crate::metrics::{recall_at_k, relative_error};
use crate::moment::{AmsL2Sketch, JlProjection};
use crate::ptx_kernels::{
    bloom_insert_ptx, cm_query_ptx, cm_update_ptx, hll_register_ptx, minhash_sketch_ptx,
    reservoir_sample_ptx, tdigest_merge_ptx,
};
use crate::quantile::KllSketch;
use crate::quantile::TDigest;
use crate::sampling::ReservoirSampler;
use crate::similarity::MinHash;
use crate::similarity::SimHash;
use crate::stream::WelfordOnline;
use crate::topk::MisraGries;
use crate::topk::SpaceSaving;

// 1. HyperLogLog estimates 10000 distinct items to within ±5% (p=14).
#[test]
fn hll_10k_accuracy() {
    let mut h = HyperLogLog::new(14, 0).expect("ok");
    for i in 0..10_000u64 {
        h.add_u64(i);
    }
    let est = h.estimate();
    let rel = (est - 10_000.0).abs() / 10_000.0;
    assert!(rel < 0.05, "HLL relative error = {rel}");
}

// 2. HyperLogLog++ estimates 10000 distinct items to within ±5%.
#[test]
fn hll_plus_10k_accuracy() {
    let mut h = HyperLogLogPlus::new(14, 0).expect("ok");
    for i in 0..10_000u64 {
        h.add_u64(i);
    }
    let est = h.estimate();
    let rel = (est - 10_000.0).abs() / 10_000.0;
    assert!(rel < 0.05, "HLL++ relative error = {rel}");
}

// 3. Linear counting estimates correctly for moderate cardinality.
#[test]
fn lc_distinct_count_accurate() {
    let mut lc = LinearCounter::new(8192, 0).expect("ok");
    for i in 0..2000u64 {
        lc.add_u64(i);
    }
    let rel = relative_error(lc.estimate(), 2000.0);
    assert!(rel < 0.1, "LC relative error {rel}");
}

// 4. Count-Min Sketch never underestimates.
#[test]
fn cms_never_underestimates() {
    let mut rng = LcgRng::new(11);
    let mut cms = CountMinSketch::new(5, 256, &mut rng).expect("ok");
    for i in 0..1000u64 {
        cms.add(i % 50);
    }
    for k in 0..50u64 {
        let est = cms.query(k);
        assert!(est >= 20, "CMS underestimated key {k}: {est}");
    }
}

// 5. Count Sketch produces unbiased estimates close to truth.
#[test]
fn cs_close_to_truth() {
    let mut rng = LcgRng::new(7);
    let mut cs = CountSketch::new(7, 1024, &mut rng).expect("ok");
    for _ in 0..200 {
        cs.add(99);
    }
    let est = cs.query(99);
    assert!((est - 200).abs() < 30, "CS estimate {est}");
}

// 6. Bloom filter never has false negatives.
#[test]
fn bloom_no_false_negatives() {
    let mut bf = BloomFilter::new(8192, 5, 0).expect("ok");
    for i in 0..1000u64 {
        bf.insert(i);
    }
    for i in 0..1000u64 {
        assert!(bf.contains(i), "false negative for inserted item {i}");
    }
}

// 7. Bloom filter false-positive rate is close to predicted.
#[test]
fn bloom_fp_rate_close_to_predicted() {
    let mut bf = BloomFilter::with_expected_fp(2000, 0.01, 11).expect("ok");
    for i in 0..2000u64 {
        bf.insert(i);
    }
    let mut fp = 0usize;
    for i in 100_000..120_000u64 {
        if bf.contains(i) {
            fp += 1;
        }
    }
    let rate = fp as f64 / 20_000.0;
    assert!(rate < 0.05, "Bloom FP rate = {rate}");
}

// 8. Cuckoo filter insert + delete cycle.
#[test]
fn cuckoo_insert_delete_cycle() {
    let mut cf = CuckooFilter::new(512, 4, 12, 0).expect("ok");
    for i in 0..100u64 {
        cf.insert(i).expect("ok");
    }
    for i in 0..100u64 {
        assert!(cf.contains(i));
    }
    cf.delete(0);
    // Probably no longer present (depending on collisions); just ensure no panic and
    // other items remain.
    for i in 1..50u64 {
        assert!(cf.contains(i));
    }
}

// 9. KLL median accurate on uniform integer stream.
#[test]
fn kll_median_uniform() {
    let mut k = KllSketch::new(512, 0).expect("ok");
    for i in 0..10_000 {
        k.add(i as f64);
    }
    let med = k.quantile(0.5);
    // KLL is a randomised sketch; allow generous slack proportional to n / sqrt(k).
    assert!((med - 5000.0).abs() < 2000.0, "KLL median = {med}");
}

// 10. t-Digest p99 is close to truth.
#[test]
fn tdigest_p99_accurate() {
    let mut td = TDigest::new(200.0).expect("ok");
    for i in 0..10_000 {
        td.add(i as f64);
    }
    td.flush();
    let p99 = td.quantile(0.99);
    assert!((p99 - 9_900.0).abs() < 200.0, "t-digest p99 = {p99}");
}

// 11. MinHash Jaccard estimate converges to true value.
#[test]
fn minhash_jaccard_convergence() {
    let rng = LcgRng::new(11);
    let a: Vec<u64> = (0..500).collect();
    let b: Vec<u64> = (250..750).collect();
    let true_j = MinHash::true_jaccard(&a, &b);
    let mh_a = MinHash::from_set(&a, 512, &mut rng.clone()).expect("ok");
    let mh_b = MinHash::from_set(&b, 512, &mut rng.clone()).expect("ok");
    let est = mh_a.jaccard(&mh_b).expect("ok");
    assert!((est - true_j).abs() < 0.08, "true {true_j} vs est {est}");
}

// 12. SimHash cosine similarity = 1 for identical inputs.
#[test]
fn simhash_identical_sim_one() {
    let mut a = SimHash::new(256, 0).expect("ok");
    let mut b = SimHash::new(256, 0).expect("ok");
    for i in 0..50u64 {
        a.add_feature(i, 1.0);
        b.add_feature(i, 1.0);
    }
    let cs = a.cosine_similarity(&b).expect("ok");
    assert!((cs - 1.0).abs() < 1e-6);
}

// 13. Reservoir sampling produces uniform sample (Chi-square style check).
#[test]
fn reservoir_uniform_sample() {
    let trials = 4000usize;
    let n = 25usize;
    let k = 5usize;
    let mut counts = vec![0usize; n];
    for t in 0..trials {
        let seed = (t as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(1);
        let mut r = ReservoirSampler::new(k, seed).expect("ok");
        for i in 0..n as u64 {
            r.add(i);
        }
        for &v in r.sample() {
            counts[v as usize] += 1;
        }
    }
    let expected = (trials * k) / n;
    for &c in &counts {
        let rel = (c as f64 - expected as f64).abs() / expected as f64;
        assert!(rel < 0.25, "reservoir non-uniform count {c}");
    }
}

// 14. Misra-Gries finds heavy hitters with frequency > n/k.
#[test]
fn misra_gries_finds_heavy() {
    let mut m = MisraGries::new(10).expect("ok");
    // Heavy item 7 appears 200 times of 1000 total — frequency 0.2 > 1/10.
    for _ in 0..200 {
        m.add(7);
    }
    for i in 0..800u64 {
        m.add(i + 100);
    }
    let cands: Vec<u64> = m.candidates().iter().map(|(k, _)| *k).collect();
    assert!(cands.contains(&7), "MG failed to track heavy hitter 7");
}

// 15. Welford online mean & variance match direct computation.
#[test]
fn welford_matches_direct() {
    let mut w = WelfordOnline::new();
    let xs = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    for &x in &xs {
        w.add(x);
    }
    let mean_direct: f64 = xs.iter().sum::<f64>() / xs.len() as f64;
    let var_direct: f64 =
        xs.iter().map(|x| (x - mean_direct).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
    assert!((w.mean - mean_direct).abs() < 1e-12);
    assert!((w.sample_variance() - var_direct).abs() < 1e-12);
}

// 16. PTX kernel strings non-empty across 6 SM versions × 7 kernels.
#[test]
fn ptx_kernels_all_sm_versions() {
    type KFn = fn(u32) -> String;
    let kernels: &[(&str, KFn)] = &[
        ("cm_update", cm_update_ptx),
        ("cm_query", cm_query_ptx),
        ("hll_register", hll_register_ptx),
        ("bloom_insert", bloom_insert_ptx),
        ("minhash_sketch", minhash_sketch_ptx),
        ("tdigest_merge", tdigest_merge_ptx),
        ("reservoir_sample", reservoir_sample_ptx),
    ];
    let sms = [75u32, 80, 86, 89, 90, 100];
    for sm in sms {
        for (name, f) in kernels {
            let s = f(sm);
            assert!(!s.is_empty(), "kernel {name} sm={sm} empty");
            assert!(s.contains(".visible .entry"));
            assert!(s.contains("ret"));
        }
    }
}

// 17. Pipeline: build LSH index from MinHash signatures, retrieve nearest.
#[test]
fn lsh_pipeline_retrieves_similar() {
    let rng = LcgRng::new(13);
    let lsh = JaccardLsh::new(4, 16).expect("ok");
    let k = 64;
    let mh1 = MinHash::from_set(&(0..200u64).collect::<Vec<_>>(), k, &mut rng.clone()).expect("ok");
    let mh2 =
        MinHash::from_set(&(100..300u64).collect::<Vec<_>>(), k, &mut rng.clone()).expect("ok");
    let mh3 =
        MinHash::from_set(&(500..700u64).collect::<Vec<_>>(), k, &mut rng.clone()).expect("ok");
    let mut idx = LshIndex::new(16);
    idx.insert(1, &lsh.band_keys(&mh1).expect("ok"))
        .expect("ok");
    idx.insert(2, &lsh.band_keys(&mh2).expect("ok"))
        .expect("ok");
    idx.insert(3, &lsh.band_keys(&mh3).expect("ok"))
        .expect("ok");
    let qmh = MinHash::from_set(&(0..200u64).collect::<Vec<_>>(), k, &mut rng.clone()).expect("ok");
    let cands = idx.query(&lsh.band_keys(&qmh).expect("ok")).expect("ok");
    assert!(cands.contains(&1), "LSH did not retrieve close set");
}

// 18. AMS L2 sketch estimates L2^2 of a known vector.
#[test]
fn ams_l2_known_vector() {
    let mut rng = LcgRng::new(11);
    let mut s = AmsL2Sketch::new(7, 200, &mut rng).expect("ok");
    // x = (1, 1, ..., 1) with 100 ones. ||x||_2^2 = 100.
    for i in 0..100u64 {
        s.update(i, 1.0);
    }
    let est = s.estimate_l2_squared();
    let rel = (est - 100.0).abs() / 100.0;
    assert!(rel < 0.4, "AMS L2^2 rel-err {rel}");
}

// 19. JL projection approximately preserves distance.
#[test]
fn jl_distance_preservation() {
    let mut rng = LcgRng::new(7);
    let d = 100;
    let k = 200;
    let j = JlProjection::new_rademacher(d, k, &mut rng).expect("ok");
    let x: Vec<f64> = (0..d).map(|i| (i as f64) - 50.0).collect();
    let y: Vec<f64> = (0..d).map(|i| ((i + 1) as f64) - 50.0).collect();
    let true_d2: f64 = x.iter().zip(&y).map(|(a, b)| (a - b).powi(2)).sum();
    let px = j.project(&x).expect("ok");
    let py = j.project(&y).expect("ok");
    let proj_d2: f64 = px.iter().zip(&py).map(|(a, b)| (a - b).powi(2)).sum();
    let rel = (proj_d2 / true_d2 - 1.0).abs();
    assert!(rel < 0.5, "JL rel-err = {rel}");
}

// 20. Cosine LSH: similar vectors have small Hamming distance.
#[test]
fn cosine_lsh_similar_close() {
    let mut rng = LcgRng::new(7);
    let dim = 32;
    let lsh = CosineLsh::new(256, dim, &mut rng).expect("ok");
    let x: Vec<f64> = (0..dim).map(|_| rng.next_normal()).collect();
    let y: Vec<f64> = x.iter().map(|v| v + 0.01 * rng.next_normal()).collect();
    let sx = lsh.signature(&x).expect("ok");
    let sy = lsh.signature(&y).expect("ok");
    let d = CosineLsh::hamming_distance(&sx, &sy);
    // Should be much less than half (which would be uncorrelated case).
    assert!(d < 80, "cosine-LSH ham {d} too large for similar vectors");
}

// 21. Space-Saving estimates heavy hitter without undercount.
#[test]
fn space_saving_no_undercount() {
    let mut s = SpaceSaving::new(8).expect("ok");
    for _ in 0..300 {
        s.add(42);
    }
    for i in 0..700u64 {
        s.add(i + 100);
    }
    let e = s.estimate(42);
    assert!(e >= 300, "SS undercounted: {e}");
}

// 22. Top-k recall via Space-Saving.
#[test]
fn space_saving_top_k_recall() {
    let mut s = SpaceSaving::new(5).expect("ok");
    // Heavy hitters: 1 (300x), 2 (200x), 3 (150x).
    for _ in 0..300 {
        s.add(1);
    }
    for _ in 0..200 {
        s.add(2);
    }
    for _ in 0..150 {
        s.add(3);
    }
    for i in 0..350u64 {
        s.add(i + 100);
    }
    let est: Vec<u64> = s.top_k().iter().take(3).map(|(k, _)| *k).collect();
    let truth = vec![1u64, 2, 3];
    let r = recall_at_k(&est, &truth, 3);
    assert!(r >= 2.0 / 3.0, "top-k recall {r}");
}
