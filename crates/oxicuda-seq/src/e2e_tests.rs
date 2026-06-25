//! End-to-end integration tests for `oxicuda-seq`.

use crate::alignment::{
    Alignment, ScoringMatrix, hirschberg_align, needleman_wunsch, smith_waterman,
};
use crate::beam::{BeamConfig, BeamSearch};
use crate::crf::{
    LbfgsConfig, LinearChainCrf, crf_log_likelihood_and_gradient, train_crf_lbfgs, viterbi_decode,
};
use crate::handle::LcgRng;
use crate::hmm::{HmmDiscrete, baum_welch_discrete, forward_backward, viterbi};
use crate::kalman::{KalmanFilter, rts_smoother};
use crate::metrics::{bleu_n, edit_distance, token_accuracy};
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

// ──────────────────────────────────────────────
// Test 16: Worked example — Kalman 2D constant-velocity position tracking.
//
// Simulate a noisy 2D constant-velocity trajectory (measurement noise drawn from
// the crate's seeded `LcgRng`), run the linear `KalmanFilter` step-by-step, and
// verify the posterior estimate genuinely denoises the raw measurements: the
// filtered position RMSE is clearly below the raw-measurement RMSE, and the
// absolute tracking error stays bounded below one measurement std-dev.
// ──────────────────────────────────────────────
#[test]
fn kalman_2d_position_tracking() {
    // State x = [px, py, vx, vy]; constant-velocity dynamics with dt = 1.
    let dim_x = 4;
    let dim_z = 2;
    // F: position += velocity each step; velocity carried forward.
    let f = vec![
        1.0, 0.0, 1.0, 0.0, //
        0.0, 1.0, 0.0, 1.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0, //
    ];
    // H: observe position only.
    let h = vec![
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
    ];
    // Tiny process noise (true motion is exactly CV); larger measurement noise.
    let q_pos = 1e-4;
    let q_vel = 1e-4;
    let q = vec![
        q_pos, 0.0, 0.0, 0.0, //
        0.0, q_pos, 0.0, 0.0, //
        0.0, 0.0, q_vel, 0.0, //
        0.0, 0.0, 0.0, q_vel, //
    ];
    let sigma_meas = 1.0_f64;
    let r = vec![
        sigma_meas * sigma_meas,
        0.0, //
        0.0,
        sigma_meas * sigma_meas, //
    ];
    // Unknown initial state (velocity must be learned from data), diffuse prior.
    let x0 = vec![0.0, 0.0, 0.0, 0.0];
    let p0 = vec![
        10.0, 0.0, 0.0, 0.0, //
        0.0, 10.0, 0.0, 0.0, //
        0.0, 0.0, 10.0, 0.0, //
        0.0, 0.0, 0.0, 10.0, //
    ];
    let kf = KalmanFilter::new(dim_x, dim_z, f, h, q, r, x0, p0).expect("kf");

    // Ground truth + seeded-noise measurements.
    let t_max = 60usize;
    let (vx, vy) = (1.0_f64, 0.5_f64);
    let mut rng = LcgRng::new(20_260_621);
    let mut true_pos: Vec<(f64, f64)> = Vec::with_capacity(t_max);
    let mut z = Vec::with_capacity(t_max * dim_z);
    for t in 0..t_max {
        let tx = vx * t as f64;
        let ty = vy * t as f64;
        true_pos.push((tx, ty));
        z.push(tx + sigma_meas * rng.next_normal());
        z.push(ty + sigma_meas * rng.next_normal());
    }

    let res = kf.filter(&z).expect("filter");

    // Compare raw-measurement error and filtered-estimate error against truth.
    let mut raw_sq = 0.0;
    let mut filt_sq = 0.0;
    for t in 0..t_max {
        let (tx, ty) = true_pos[t];
        raw_sq += (z[t * dim_z] - tx).powi(2) + (z[t * dim_z + 1] - ty).powi(2);
        filt_sq += (res.means[t][0] - tx).powi(2) + (res.means[t][1] - ty).powi(2);
    }
    let raw_rmse = (raw_sq / t_max as f64).sqrt();
    let filt_rmse = (filt_sq / t_max as f64).sqrt();
    eprintln!(
        "KALMAN2D raw_rmse={raw_rmse:.4} filt_rmse={filt_rmse:.4} ratio={:.3}",
        filt_rmse / raw_rmse
    );

    // The filter must genuinely denoise.
    assert!(
        filt_rmse < raw_rmse,
        "filter did not denoise: filt={filt_rmse}, raw={raw_rmse}"
    );
    // …and by a clear margin, not marginally.
    assert!(
        filt_rmse < 0.7 * raw_rmse,
        "denoising margin too small: filt={filt_rmse}, raw={raw_rmse}"
    );
    // Bounded absolute tracking error: below one measurement std-dev.
    assert!(
        filt_rmse < sigma_meas,
        "filtered RMSE too high: {filt_rmse}"
    );
}

// ──────────────────────────────────────────────
// Test 17: Worked example — train a linear-chain CRF chunker on a small,
// fully in-process synthetic BIO-tagging corpus (NO external corpus) and verify
// it fits the training data: Viterbi per-token accuracy far exceeds the
// majority-class baseline, and the data log-likelihood rises during training.
// ──────────────────────────────────────────────
#[test]
fn crf_chunker_fits_training_data() {
    // Toy noun-phrase chunking. Word categories drive the features; gold BIO tags
    // follow a deterministic rule that needs the *previous* tag — so transitions
    // genuinely matter, not just emissions:
    //   DET                         -> B   (a determiner opens an NP)
    //   NOUN after DET / NOUN        -> I   (continues the NP)
    //   NOUN after VERB / PUNCT / BOS-> B   (opens a fresh NP)
    //   VERB, PUNCT                 -> O
    // Labels: 0 = O, 1 = B, 2 = I.
    const DET: usize = 0;
    const NOUN: usize = 1;
    const VERB: usize = 2;
    const PUNCT: usize = 3;
    const O: usize = 0;
    const B: usize = 1;
    const I: usize = 2;
    let n_cat = 4;
    let n_labels = 3;
    // Features: one-hot(category)[4] + bias[1] + is_bos[1].
    let n_features = n_cat + 2;
    let bias_idx = n_cat;
    let bos_idx = n_cat + 1;

    let mut rng = LcgRng::new(0x00C0_FFEE);
    let n_sent = 60usize;
    let mut examples: Vec<(Vec<f64>, Vec<usize>)> = Vec::with_capacity(n_sent);
    let mut feat_mats: Vec<Vec<f64>> = Vec::with_capacity(n_sent);
    let mut gold_tags: Vec<Vec<usize>> = Vec::with_capacity(n_sent);

    for _ in 0..n_sent {
        let len = 4 + rng.next_usize(5); // length 4..=8
        let mut cats = Vec::with_capacity(len);
        for _ in 0..len {
            // Skewed category mix: nouns common, dets/verbs/punct rarer.
            let u = rng.next_f64();
            let c = if u < 0.25 {
                DET
            } else if u < 0.65 {
                NOUN
            } else if u < 0.85 {
                VERB
            } else {
                PUNCT
            };
            cats.push(c);
        }
        // Gold BIO labels by the deterministic rule.
        let mut tags = Vec::with_capacity(len);
        for t in 0..len {
            let tag = match cats[t] {
                DET => B,
                NOUN => {
                    if t > 0 && (cats[t - 1] == DET || cats[t - 1] == NOUN) {
                        I
                    } else {
                        B
                    }
                }
                _ => O, // VERB, PUNCT
            };
            tags.push(tag);
        }
        // Feature matrix: one-hot category + bias + is_bos.
        let mut x = vec![0.0; len * n_features];
        for t in 0..len {
            x[t * n_features + cats[t]] = 1.0;
            x[t * n_features + bias_idx] = 1.0;
            if t == 0 {
                x[t * n_features + bos_idx] = 1.0;
            }
        }
        examples.push((x.clone(), tags.clone()));
        feat_mats.push(x);
        gold_tags.push(tags);
    }

    // Majority-class baseline: predict the single most frequent label everywhere.
    let mut counts = [0usize; 3];
    let mut n_tok = 0usize;
    for tags in &gold_tags {
        for &g in tags {
            counts[g] += 1;
            n_tok += 1;
        }
    }
    let baseline = *counts.iter().max().expect("nonempty") as f64 / n_tok as f64;

    // Initial (zero-weight) data log-likelihood.
    let mut crf = LinearChainCrf::zeros(n_labels, n_features).expect("crf");
    let mut ll0 = 0.0;
    for (x, y) in &examples {
        let (ll, _, _) = crf_log_likelihood_and_gradient(&crf, x, y).expect("ll0");
        ll0 += ll;
    }

    // Train with L-BFGS.
    let cfg = LbfgsConfig {
        memory: 8,
        max_iter: 200,
        grad_tol: 1e-7,
        backtrack: 0.5,
        max_line_search: 40,
        l2: 1e-3,
    };
    let ll_final = train_crf_lbfgs(&mut crf, &examples, &cfg).expect("train");

    // Decode the training set with Viterbi and measure per-token accuracy.
    let mut preds = Vec::with_capacity(n_sent);
    for x in &feat_mats {
        preds.push(viterbi_decode(&crf, x).expect("decode"));
    }
    let acc = token_accuracy(&preds, &gold_tags).expect("acc");
    eprintln!(
        "CRFCHUNK baseline={baseline:.4} acc={acc:.4} ll0={ll0:.3} ll_final={ll_final:.3} n_tok={n_tok}"
    );

    // Training must raise the (regularised) data log-likelihood.
    assert!(
        ll_final > ll0,
        "log-likelihood did not rise: ll0={ll0}, ll_final={ll_final}"
    );
    // The CRF must fit the data far better than majority-class guessing.
    assert!(
        acc > baseline + 0.2,
        "accuracy {acc} not clearly above baseline {baseline}"
    );
    // The task is deterministic and representable, so accuracy should be high.
    assert!(acc > 0.9, "train-set accuracy too low: {acc}");
}
