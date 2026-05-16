//! End-to-end integration tests for `oxicuda-seq`.

use crate::alignment::{
    Alignment, ScoringMatrix, hirschberg_align, needleman_wunsch, smith_waterman,
};
use crate::beam::{BeamConfig, BeamSearch};
use crate::crf::{LinearChainCrf, crf_log_likelihood_and_gradient, viterbi_decode};
use crate::handle::LcgRng;
use crate::hmm::{HmmDiscrete, baum_welch_discrete, forward_backward, viterbi};
use crate::kalman::{KalmanFilter, rts_smoother};
use crate::metrics::{bleu_n, edit_distance};
use crate::mrf::{GibbsConfig, IsingModel, ising_gibbs};
use crate::ptx_kernels::{
    beam_topk_ptx, crf_features_ptx, edit_dist_ptx, forward_pass_ptx, kalman_predict_ptx,
    mrf_gibbs_ptx, viterbi_step_ptx,
};

/// Brute-force enumeration of all state paths to compute exact log-likelihood.
fn enumerate_log_likelihood(hmm: &HmmDiscrete, obs: &[usize]) -> f64 {
    let t = obs.len();
    let n = hmm.n_states;
    let mut paths: Vec<Vec<usize>> = vec![vec![]];
    for _ in 0..t {
        let mut new_paths = Vec::with_capacity(paths.len() * n);
        for p in &paths {
            for s in 0..n {
                let mut q = p.clone();
                q.push(s);
                new_paths.push(q);
            }
        }
        paths = new_paths;
    }
    let mut total = 0.0;
    for p in &paths {
        let mut ll = hmm.pi[p[0]] * hmm.b[p[0] * hmm.n_obs + obs[0]];
        for k in 1..t {
            ll *= hmm.a[p[k - 1] * n + p[k]] * hmm.b[p[k] * hmm.n_obs + obs[k]];
        }
        total += ll;
    }
    total.ln()
}

// ──────────────────────────────────────────────
// Test 1: HMM forward-backward matches exhaustive enumeration
// ──────────────────────────────────────────────
#[test]
fn hmm_fb_matches_enumeration() {
    let h = HmmDiscrete::new(
        3,
        2,
        vec![0.5, 0.3, 0.2],
        vec![0.7, 0.2, 0.1, 0.1, 0.6, 0.3, 0.2, 0.3, 0.5],
        vec![0.8, 0.2, 0.4, 0.6, 0.1, 0.9],
    )
    .expect("ok");
    let obs = vec![0, 1, 0, 1];
    let fb = forward_backward(&h, &obs).expect("ok");
    let exact = enumerate_log_likelihood(&h, &obs);
    assert!(
        (fb.log_likelihood - exact).abs() < 1e-9,
        "fb={}, exact={}",
        fb.log_likelihood,
        exact
    );
}

// ──────────────────────────────────────────────
// Test 2: HMM Viterbi recovers deterministic chain
// ──────────────────────────────────────────────
#[test]
fn hmm_viterbi_deterministic() {
    let h = HmmDiscrete::new(
        2,
        2,
        vec![0.99, 0.01],
        vec![0.95, 0.05, 0.05, 0.95],
        vec![0.99, 0.01, 0.01, 0.99],
    )
    .expect("ok");
    let r = viterbi(&h, &[0, 0, 1, 1]).expect("ok");
    assert_eq!(r.path, vec![0, 0, 1, 1]);
}

// ──────────────────────────────────────────────
// Test 3: Baum-Welch is non-decreasing
// ──────────────────────────────────────────────
#[test]
fn baum_welch_nondecreasing() {
    let init = HmmDiscrete::new(
        2,
        2,
        vec![0.55, 0.45],
        vec![0.55, 0.45, 0.45, 0.55],
        vec![0.55, 0.45, 0.45, 0.55],
    )
    .expect("ok");
    let obs = vec![0, 0, 1, 1, 0, 1, 0, 0, 1, 0];
    let r = baum_welch_discrete(&init, &obs, 30, 1e-8).expect("ok");
    for w in r.log_likelihoods.windows(2) {
        assert!(w[1] + 1e-6 >= w[0], "decrease: {} -> {}", w[0], w[1]);
    }
}

// ──────────────────────────────────────────────
// Test 4: CRF Viterbi matches sequence-score argmax (tiny enumeration)
// ──────────────────────────────────────────────
#[test]
fn crf_viterbi_matches_argmax() {
    let mut crf = LinearChainCrf::zeros(2, 2).expect("ok");
    crf.emissions = vec![0.5, -0.2, -0.4, 0.3];
    crf.transitions = vec![0.2, -0.1, -0.3, 0.4];
    let x = vec![1.0, 0.5, 0.0, 1.0, 0.7, 0.2];

    // Enumerate all 2^3 = 8 paths
    let mut best_score = f64::NEG_INFINITY;
    let mut best_path = vec![0usize; 3];
    for a in 0..2 {
        for b in 0..2 {
            for c in 0..2 {
                let p = vec![a, b, c];
                let s = crf.sequence_score(&x, &p).expect("ok");
                if s > best_score {
                    best_score = s;
                    best_path = p;
                }
            }
        }
    }
    let decoded = viterbi_decode(&crf, &x).expect("ok");
    assert_eq!(decoded, best_path);
}

// ──────────────────────────────────────────────
// Test 5: CRF gradient ≈ finite difference (1e-4)
// ──────────────────────────────────────────────
#[test]
fn crf_gradient_matches_fd() {
    let mut crf = LinearChainCrf::zeros(2, 2).expect("ok");
    crf.emissions = vec![0.3, -0.4, 0.1, 0.5];
    crf.transitions = vec![0.2, -0.3, -0.4, 0.6];
    let x = vec![1.0, 0.5, 0.4, 1.0];
    let y = vec![0usize, 1];
    let (_ll, ge, gt) = crf_log_likelihood_and_gradient(&crf, &x, &y).expect("ok");
    let eps = 1e-5;
    for k in 0..crf.emissions.len() {
        let mut p = crf.clone();
        p.emissions[k] += eps;
        let (lp, _, _) = crf_log_likelihood_and_gradient(&p, &x, &y).expect("ok");
        let mut q = crf.clone();
        q.emissions[k] -= eps;
        let (lm, _, _) = crf_log_likelihood_and_gradient(&q, &x, &y).expect("ok");
        let num = (lp - lm) / (2.0 * eps);
        assert!(
            (num - ge[k]).abs() < 1e-3,
            "emit{k}: num={num}, ana={}",
            ge[k]
        );
    }
    for k in 0..crf.transitions.len() {
        let mut p = crf.clone();
        p.transitions[k] += eps;
        let (lp, _, _) = crf_log_likelihood_and_gradient(&p, &x, &y).expect("ok");
        let mut q = crf.clone();
        q.transitions[k] -= eps;
        let (lm, _, _) = crf_log_likelihood_and_gradient(&q, &x, &y).expect("ok");
        let num = (lp - lm) / (2.0 * eps);
        assert!(
            (num - gt[k]).abs() < 1e-3,
            "trans{k}: num={num}, ana={}",
            gt[k]
        );
    }
}

// ──────────────────────────────────────────────
// Test 6: Needleman-Wunsch sane on GATTACA/GCATGCU
// ──────────────────────────────────────────────
#[test]
fn nw_gattaca_gcatgcu() {
    let sc = ScoringMatrix::default();
    let r: Alignment = needleman_wunsch(b"GATTACA", b"GCATGCU", &sc).expect("ok");
    assert!(r.score.abs() <= 3, "score {}", r.score);
    assert!(r.a_aligned.len() == r.b_aligned.len());
}

// ──────────────────────────────────────────────
// Test 7: Smith-Waterman finds embedded common substring
// ──────────────────────────────────────────────
#[test]
fn sw_embedded_substring() {
    let r = smith_waterman(b"XXXACGTYYY", b"ZACGTW", &ScoringMatrix::default()).expect("ok");
    assert!(r.score >= 4);
}

// ──────────────────────────────────────────────
// Test 8: Hirschberg yields the same score as NW
// ──────────────────────────────────────────────
#[test]
fn hirschberg_matches_nw_score() {
    let sc = ScoringMatrix::default();
    let pairs: &[(&[u8], &[u8])] = &[
        (b"GATTACA", b"GCATGCU"),
        (b"ACGTACGT", b"ACGGACGT"),
        (b"AAAAAA", b"AACAAA"),
    ];
    for &(a, b) in pairs {
        let r1 = needleman_wunsch(a, b, &sc).expect("ok");
        let r2 = hirschberg_align(a, b, &sc).expect("ok");
        assert_eq!(r1.score, r2.score, "score mismatch on {a:?}/{b:?}");
    }
}

// ──────────────────────────────────────────────
// Test 9: Edit distance: "kitten" → "sitting" = 3
// ──────────────────────────────────────────────
#[test]
fn edit_distance_kitten_sitting() {
    assert_eq!(edit_distance(b"kitten", b"sitting"), 3);
}

// ──────────────────────────────────────────────
// Test 10: Kalman filter recovers state mean
// ──────────────────────────────────────────────
#[test]
fn kalman_recovers_state() {
    let kf = KalmanFilter::new(
        1,
        1,
        vec![1.0],
        vec![1.0],
        vec![0.01],
        vec![0.05],
        vec![0.0],
        vec![1.0],
    )
    .expect("ok");
    let z = vec![1.0, 1.02, 0.97, 1.01, 0.99, 1.0];
    let r = kf.filter(&z).expect("ok");
    let last = r.means[r.means.len() - 1][0];
    assert!((last - 1.0).abs() < 0.2, "mean drift {last}");
}

// ──────────────────────────────────────────────
// Test 11: RTS smoother variance ≤ filter variance
// ──────────────────────────────────────────────
#[test]
fn rts_variance_le_filter_variance() {
    let kf = KalmanFilter::new(
        1,
        1,
        vec![1.0],
        vec![1.0],
        vec![0.01],
        vec![0.1],
        vec![0.0],
        vec![1.0],
    )
    .expect("ok");
    let z = vec![1.0, 0.95, 1.1, 1.05, 0.9, 1.0];
    let f = kf.filter(&z).expect("ok");
    let s = rts_smoother(&kf, &f).expect("ok");
    for t in 0..z.len() - 1 {
        assert!(
            s.covs[t][0] <= f.covs[t][0] + 1e-9,
            "smoother var {} > filter var {}",
            s.covs[t][0],
            f.covs[t][0]
        );
    }
}

// ──────────────────────────────────────────────
// Test 12: Gibbs Ising recovers magnetisation at low temperature
// ──────────────────────────────────────────────
#[test]
fn ising_gibbs_polarises() {
    let m = IsingModel::new(6, 6, 0.05, 1.0, 2.0).expect("ok");
    let init = vec![1i32; 36];
    let cfg = GibbsConfig {
        n_sweeps: 300,
        burn_in: 100,
        anneal: None,
    };
    let mut rng = LcgRng::new(123);
    let (_, mag) = ising_gibbs(&m, &init, &cfg, &mut rng).expect("ok");
    assert!(mag > 0.4, "magnetisation too low: {mag}");
}

// ──────────────────────────────────────────────
// Test 13: Beam search finds top-1 sequence matching exhaustive enumeration
// ──────────────────────────────────────────────
#[test]
fn beam_matches_exhaustive_top1() {
    // 2 tokens, 3 steps after start. Transition log-prob table.
    let log_probs: [[f64; 2]; 2] = [[-0.05, -2.0], [-1.5, -0.3]];
    let max_steps = 3usize;
    let bs = BeamSearch::new(BeamConfig {
        beam_width: 2,
        max_steps,
        length_alpha: 0.0,
        diversity: 0.0,
    })
    .expect("ok");
    let (path, _score) = bs
        .search(
            0,
            |path| {
                let prev = path.last().copied().unwrap_or(0);
                (0..2).map(|t| (t, log_probs[prev][t])).collect()
            },
            |_t| false,
        )
        .expect("ok");

    // Exhaustive: enumerate 2^3 paths starting from 0
    let mut best_score = f64::NEG_INFINITY;
    let mut best = Vec::new();
    for a in 0..2 {
        for b in 0..2 {
            for c in 0..2 {
                let mut s = log_probs[0][a];
                s += log_probs[a][b];
                s += log_probs[b][c];
                if s > best_score {
                    best_score = s;
                    best = vec![0, a, b, c];
                }
            }
        }
    }
    assert_eq!(path, best);
}

// ──────────────────────────────────────────────
// Test 14: BLEU-1 of identical sentences = 1.0
// ──────────────────────────────────────────────
#[test]
fn bleu1_identical_one() {
    let a = vec![1, 2, 3, 4, 5];
    let s = bleu_n(&a, &a, 1).expect("ok");
    assert!((s - 1.0).abs() < 1e-9, "bleu={s}");
}

// ──────────────────────────────────────────────
// Test 15: PTX kernels non-empty across 6 SM versions × 7 kernels
// ──────────────────────────────────────────────
#[test]
fn ptx_kernels_non_empty() {
    type KernelFn = fn(u32) -> String;
    let kernels: &[(&str, KernelFn)] = &[
        ("forward_pass", forward_pass_ptx),
        ("viterbi_step", viterbi_step_ptx),
        ("crf_features", crf_features_ptx),
        ("beam_topk", beam_topk_ptx),
        ("edit_dist", edit_dist_ptx),
        ("kalman_predict", kalman_predict_ptx),
        ("mrf_gibbs", mrf_gibbs_ptx),
    ];
    let sms = [75u32, 80, 86, 89, 90, 100];
    for &sm in &sms {
        for &(name, f) in kernels {
            let s = f(sm);
            assert!(!s.is_empty(), "{name} sm{sm} empty");
            assert!(
                s.contains(".visible .entry"),
                "{name} sm{sm} missing .visible .entry"
            );
        }
    }
}
