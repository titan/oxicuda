//! End-to-end integration tests for oxicuda-hdc.

use crate::classifier::hd_classifier::HdClassifier;
use crate::distance::cosine::{cosine_binary, cosine_complex};
use crate::distance::hamming::hamming_frac;
use crate::encoding::ngram::NgramEncoder;
use crate::encoding::pattern::PatternEncoder;
use crate::encoding::record::RecordEncoder;
use crate::handle::{HdcHandle, LcgRng};
use crate::memory::assoc_memory::AssocMemory;
use crate::memory::item_memory::ItemMemory;
use crate::metrics::metrics::{hopfield_capacity, required_dimension};
use crate::ops::binding::{binary_bind, binary_unbind};
use crate::ops::bundling::bundle_binary;
use crate::ops::permutation::{cyclic_shift, cyclic_shift_right};
use crate::ptx_kernels::{
    bundle_majority_ptx, complex_bind_ptx, cosine_sim_ptx, cyclic_shift_ptx, hamming_dist_ptx,
    hd_classify_ptx, xor_bind_ptx,
};
use crate::vector::binary::random_binary;
use crate::vector::complex::{complex_bind, complex_conjugate, random_complex};

// ─── Test 1 ─────────────────────────────────────────────────────────────────

#[test]
fn random_binary_hv_dimension_correct() {
    let mut rng = LcgRng::new(1);
    let dim = 1000;
    let hv = random_binary(dim, &mut rng).expect("random_binary");
    assert_eq!(hv.len(), dim);
    assert!(
        hv.iter().all(|&v| v == 1 || v == -1),
        "all values must be ±1"
    );
}

// ─── Test 2 ─────────────────────────────────────────────────────────────────

#[test]
fn binary_bind_self_inverse() {
    let mut rng = LcgRng::new(2);
    let a = random_binary(1000, &mut rng).expect("a");
    let bound = binary_bind(&a, &a).expect("bind self");
    // a ⊗ a = all +1 (XOR with self = identity in ±1 domain)
    assert!(bound.iter().all(|&v| v == 1), "a ⊗ a should be all +1");
}

// ─── Test 3 ─────────────────────────────────────────────────────────────────

#[test]
fn bundle_majority_recovers_majority() {
    let mut rng = LcgRng::new(3);
    let dim = 1000;
    let hv_a = random_binary(dim, &mut rng).expect("a");
    // Bundle 3 HVs: 2 copies of hv_a and 1 random — majority should be close to hv_a
    let hv_noise = random_binary(dim, &mut rng).expect("noise");
    let hvs = vec![hv_a.clone(), hv_a.clone(), hv_noise];
    let bundled = bundle_binary(&hvs, &mut rng).expect("bundle");
    let sim = cosine_binary(&hv_a, &bundled).expect("cosine");
    assert!(
        sim > 0.0,
        "bundled majority should align with majority HV, sim={sim}"
    );
}

// ─── Test 4 ─────────────────────────────────────────────────────────────────

#[test]
fn cyclic_shift_left_then_right_recovers() {
    let mut rng = LcgRng::new(4);
    let hv = random_binary(1000, &mut rng).expect("hv");
    let k = 137;
    let shifted = cyclic_shift(&hv, k).expect("shift left");
    let recovered = cyclic_shift_right(&shifted, k).expect("shift right");
    assert_eq!(recovered, hv, "shift left then right must recover original");
}

// ─── Test 5 ─────────────────────────────────────────────────────────────────

#[test]
fn item_memory_query_exact_match() {
    let mut rng = LcgRng::new(5);
    let dim = 1000;
    let mut mem = ItemMemory::new(dim).expect("new");
    // Add 10 random HVs
    for id in 0..10 {
        mem.add_random(id, &mut rng).expect("add_random");
    }
    // For each stored HV, query its exact HV → should return correct id
    for id in 0..10 {
        let stored_hv = mem.get(id).expect("get").to_vec();
        let found_id = mem.query(&stored_hv).expect("query");
        assert_eq!(found_id, id, "exact query should return correct id");
    }
}

// ─── Test 6 ─────────────────────────────────────────────────────────────────

#[test]
fn assoc_memory_retrieval_correct() {
    let mut rng = LcgRng::new(6);
    let dim = 1000;
    let key = random_binary(dim, &mut rng).expect("key");
    let val = random_binary(dim, &mut rng).expect("val");

    let mut mem = AssocMemory::new(dim).expect("new");
    mem.store(&key, &val).expect("store");
    mem.finalize(&mut rng).expect("finalize");

    let retrieved = mem.retrieve(&key).expect("retrieve");
    // Retrieved should correlate highly with val
    let sim = cosine_binary(&retrieved, &val).expect("cosine");
    assert!(
        sim > 0.9,
        "single-pair associative memory should retrieve correctly, sim={sim}"
    );
}

// ─── Test 7 ─────────────────────────────────────────────────────────────────

#[test]
fn hd_classifier_learns_two_classes() {
    let mut rng = LcgRng::new(7);
    let dim = 1000;
    let n_examples = 10;

    // Create two distinct class prototype HVs
    let proto0 = random_binary(dim, &mut rng).expect("proto0");
    let proto1 = random_binary(dim, &mut rng).expect("proto1");

    let mut clf = HdClassifier::new(2, dim).expect("clf");
    // Train: add multiple copies of each prototype
    for _ in 0..n_examples {
        clf.add_example(0, &proto0).expect("class0");
        clf.add_example(1, &proto1).expect("class1");
    }
    clf.build_prototypes(&mut rng).expect("build");

    // Classify each prototype → should get 100% on training HVs
    assert_eq!(clf.classify(&proto0).expect("cls0"), 0);
    assert_eq!(clf.classify(&proto1).expect("cls1"), 1);
}

// ─── Test 8 ─────────────────────────────────────────────────────────────────

#[test]
fn hd_classifier_online_update_reduces_error() {
    let mut rng = LcgRng::new(8);
    let dim = 1000;

    let proto0 = random_binary(dim, &mut rng).expect("proto0");
    let proto1 = random_binary(dim, &mut rng).expect("proto1");

    let mut clf = HdClassifier::new(2, dim).expect("clf");
    clf.add_example(0, &proto0).expect("add0");
    clf.add_example(1, &proto1).expect("add1");
    clf.build_prototypes(&mut rng).expect("build");

    // Force a misclassification scenario by online-updating with a confused example.
    // Use proto0 as "true class 0" query even if classifier got it wrong.
    clf.online_update(&proto0, 0, 1, &mut rng).expect("update");

    // After correction, class 0 prototype should be more aligned with proto0.
    let pred = clf.classify(&proto0).expect("classify");
    assert_eq!(pred, 0, "after online update, proto0 should classify as 0");
}

// ─── Test 9 ─────────────────────────────────────────────────────────────────

#[test]
fn record_encoder_different_records_differ() {
    let mut rng = LcgRng::new(9);
    let enc = RecordEncoder::new(4, 8, 1000, &mut rng).expect("enc");
    let r1 = enc.encode(&[0, 1, 2, 3], &mut rng).expect("r1");
    let r2 = enc.encode(&[4, 5, 6, 7], &mut rng).expect("r2");
    let dist = hamming_frac(&r1, &r2).expect("hamming");
    assert!(
        dist > 0.4,
        "different records should have Hamming distance > 0.4, got {dist:.3}"
    );
}

// ─── Test 10 ────────────────────────────────────────────────────────────────

#[test]
fn ngram_encoder_different_sequences_differ() {
    let mut rng = LcgRng::new(10);
    let enc = NgramEncoder::new(20, 3, 1000, &mut rng).expect("enc");
    let seq1: Vec<usize> = vec![0, 1, 2, 3, 4, 5];
    let seq2: Vec<usize> = vec![10, 11, 12, 13, 14, 15];
    let hv1 = enc.encode(&seq1, &mut rng).expect("hv1");
    let hv2 = enc.encode(&seq2, &mut rng).expect("hv2");
    let dist = hamming_frac(&hv1, &hv2).expect("hamming");
    assert!(
        dist > 0.3,
        "different sequences should have Hamming distance > 0.3, got {dist:.3}"
    );
}

// ─── Test 11 ────────────────────────────────────────────────────────────────

#[test]
fn pattern_encoder_active_pixels_affect_output() {
    let mut rng = LcgRng::new(11);
    let enc = PatternEncoder::new(8, 8, 1000, &mut rng).expect("enc");

    // Pattern A: top-left quadrant active
    let mut pixels_a = vec![0.0f32; 64];
    for r in 0..4 {
        for c in 0..4 {
            pixels_a[r * 8 + c] = 1.0;
        }
    }
    // Pattern B: bottom-right quadrant active
    let mut pixels_b = vec![0.0f32; 64];
    for r in 4..8 {
        for c in 4..8 {
            pixels_b[r * 8 + c] = 1.0;
        }
    }
    let hv_a = enc.encode(&pixels_a, 0.5, &mut rng).expect("hv_a");
    let hv_b = enc.encode(&pixels_b, 0.5, &mut rng).expect("hv_b");
    let dist = hamming_frac(&hv_a, &hv_b).expect("hamming");
    assert!(
        dist > 0.3,
        "different spatial patterns should differ, dist={dist:.3}"
    );
}

// ─── Test 12 ────────────────────────────────────────────────────────────────

#[test]
fn hamming_self_distance_zero() {
    let mut rng = LcgRng::new(12);
    let hv = random_binary(1000, &mut rng).expect("hv");
    let dist = hamming_frac(&hv, &hv).expect("hamming");
    assert!((dist).abs() < 1e-9, "Hamming(a, a) must be 0, got {dist}");
}

// ─── Test 13 ────────────────────────────────────────────────────────────────

#[test]
fn cosine_binary_self_similarity_one() {
    let mut rng = LcgRng::new(13);
    let hv = random_binary(1000, &mut rng).expect("hv");
    let sim = cosine_binary(&hv, &hv).expect("cosine");
    assert!(
        (sim - 1.0_f32).abs() < 1e-5,
        "cosine(a, a) must be 1, got {sim}"
    );
}

// ─── Test 14 ────────────────────────────────────────────────────────────────

#[test]
fn complex_bind_conjugate_recovers() {
    let mut rng = LcgRng::new(14);
    let dim = 500;
    let a = random_complex(dim, &mut rng).expect("a");
    let b = random_complex(dim, &mut rng).expect("b");

    // Bind a with b, then unbind via conjugate of b
    let bound = complex_bind(&a, &b).expect("bind");
    let conj_b = complex_conjugate(&b).expect("conj");
    let recovered = complex_bind(&bound, &conj_b).expect("unbind");

    let sim = cosine_complex(&a, &recovered).expect("cosine");
    assert!(
        sim > 0.99,
        "unbinding via conjugate should recover original HV, sim={sim}"
    );
}

// ─── Test 15 ────────────────────────────────────────────────────────────────

#[test]
fn capacity_estimate_reasonable() {
    let dim = 10_000;
    let cap = hopfield_capacity(dim);
    assert!(
        cap >= 1000,
        "Hopfield capacity for D=10000 should be >= 1000, got {cap}"
    );
}

// ─── Test 16 ────────────────────────────────────────────────────────────────

#[test]
fn required_dimension_scales_with_items() {
    let d100 = required_dimension(100, 0.01).expect("d100");
    let d1000 = required_dimension(1000, 0.01).expect("d1000");
    assert!(
        d1000 > d100,
        "more items should require larger dimension: d100={d100}, d1000={d1000}"
    );
}

// ─── Test 17 ────────────────────────────────────────────────────────────────

#[test]
fn handle_generates_valid_hvs() {
    let mut handle = HdcHandle::new(80, 42);
    let dim = 512;

    let bin_hv = handle.random_binary_hv(dim).expect("binary");
    assert_eq!(bin_hv.len(), dim);
    assert!(bin_hv.iter().all(|&v| v == 1 || v == -1));

    let int_hv = handle.random_integer_hv(dim).expect("integer");
    assert_eq!(int_hv.len(), dim);
    assert!(int_hv.iter().all(|&v| (-1..=1).contains(&v)));

    let cplx_hv = handle.random_complex_hv(dim).expect("complex");
    assert_eq!(cplx_hv.len(), 2 * dim);
    // Each pair should be on the unit circle
    for i in 0..dim {
        let mag = (cplx_hv[2 * i].powi(2) + cplx_hv[2 * i + 1].powi(2)).sqrt();
        assert!(
            (mag - 1.0_f32).abs() < 1e-5,
            "unit circle violated at i={i}, mag={mag}"
        );
    }
}

// ─── Test 18 ────────────────────────────────────────────────────────────────

#[test]
fn ptx_kernels_non_empty_all_sm() {
    for sm in [75u32, 80, 86, 89, 90, 100] {
        for (name, kernel) in [
            ("xor_bind", xor_bind_ptx(sm)),
            ("bundle_majority", bundle_majority_ptx(sm)),
            ("cyclic_shift", cyclic_shift_ptx(sm)),
            ("cosine_sim", cosine_sim_ptx(sm)),
            ("hamming_dist", hamming_dist_ptx(sm)),
            ("complex_bind", complex_bind_ptx(sm)),
            ("hd_classify", hd_classify_ptx(sm)),
        ] {
            assert!(
                kernel.contains(".visible .entry"),
                "kernel {name} for SM={sm} missing .visible .entry"
            );
            assert!(
                kernel.contains(".address_size 64"),
                "kernel {name} for SM={sm} missing .address_size 64"
            );
        }
    }
}

// ─── Additional robustness tests ─────────────────────────────────────────────

#[test]
fn binary_unbind_recovers_original() {
    let mut rng = LcgRng::new(200);
    let a = random_binary(512, &mut rng).expect("a");
    let b = random_binary(512, &mut rng).expect("b");
    let bound = binary_bind(&a, &b).expect("bind");
    let recovered = binary_unbind(&bound, &b).expect("unbind");
    assert_eq!(recovered, a, "unbind should recover original HV");
}

#[test]
fn hd_classifier_multi_class_training() {
    let mut rng = LcgRng::new(300);
    let dim = 1000;
    let n_classes = 5;

    // Generate distinct prototype HVs for each class
    let prototypes: Vec<Vec<i8>> = (0..n_classes)
        .map(|_| random_binary(dim, &mut rng).expect("proto"))
        .collect();

    let mut clf = HdClassifier::new(n_classes, dim).expect("clf");
    for (c, proto) in prototypes.iter().enumerate() {
        for _ in 0..10 {
            clf.add_example(c, proto).expect("add");
        }
    }
    clf.build_prototypes(&mut rng).expect("build");

    // Each prototype should classify correctly
    let mut correct = 0;
    for (c, proto) in prototypes.iter().enumerate() {
        if clf.classify(proto).expect("classify") == c {
            correct += 1;
        }
    }
    // Expect at least 4/5 correct (prototypes are random, so may not always be 5/5)
    assert!(
        correct >= 4,
        "expected at least 4/5 correct, got {correct}/{n_classes}"
    );
}

#[test]
fn associative_memory_capacity_estimate_positive() {
    let mem = AssocMemory::new(1000).expect("new");
    let cap = mem.capacity_estimate();
    assert!(cap > 0, "capacity estimate should be positive");
}

#[test]
fn pattern_encoder_multilevel_works() {
    let mut rng = LcgRng::new(400);
    let enc = PatternEncoder::new(4, 4, 256, &mut rng).expect("enc");
    let pixels = vec![0.3f32; 16];
    let thresholds = vec![0.1, 0.5, 0.8];
    let hv = enc
        .encode_multilevel(&pixels, &thresholds, &mut rng)
        .expect("multilevel");
    assert_eq!(hv.len(), 256);
    assert!(hv.iter().all(|&v| v == 1 || v == -1));
}
