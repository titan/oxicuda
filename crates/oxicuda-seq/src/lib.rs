//! `oxicuda-seq` — Sequence Models & Structured Prediction for OxiCUDA.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-seq
//! ├── hmm/          — Hidden Markov Models (discrete/Gaussian), forward-backward, Viterbi, Baum-Welch
//! ├── crf/          — Linear-chain CRF: forward-backward in score space, L-BFGS training, Viterbi decoding
//! ├── memm/         — Maximum-Entropy Markov Models
//! ├── ssvm/         — Structured SVM (linear-chain) with cutting-plane optimisation
//! ├── beam/         — Generic beam search with length normalisation and diversity penalty
//! ├── ctc/          — CTC loss (forward-backward) + greedy / prefix-beam decoding
//! ├── decoders/     — Stochastic decoders: top-k, nucleus (top-p), typical sampling
//! ├── alignment/    — Needleman-Wunsch, Smith-Waterman, Gotoh affine-gap, Hirschberg
//! ├── grid_crf/     — Pairwise 2D CRF + mean-field inference
//! ├── kalman/       — Linear/EKF Kalman filter, RTS smoother, EM parameter learning
//! ├── mrf/          — General MRF + Ising, Gibbs sampler, loopy belief propagation
//! ├── perceptron/   — Structured + averaged perceptron sequence taggers (Collins 2002)
//! ├── matching/     — Aho–Corasick multi-pattern matching automaton
//! ├── string/       — Manacher longest-palindrome, suffix automaton (DAWG),
//! │                   Z-algorithm, suffix array + LCP (SA-IS/Kasai), BWT/FM-index
//! ├── tagging/      — BIO/BIOES conversion + validation, span-based F1 (seqeval)
//! └── metrics/      — Token/sequence accuracy, edit distance, BLEU, perplexity, log-loss
//! ```
//!
//! NOTE on lints: numerical kernels in this crate use straight integer-index
//! loops (`for i in 0..n`) because every body indexes multiple parallel arrays.
//! Rewriting them in iterator form obscures the math and forces awkward `zip`
//! chains, so `clippy::needless_range_loop` is allowed crate-wide.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]

pub mod align;
pub mod alignment;
pub mod beam;
pub mod crf;
pub mod ctc;
pub mod decoders;
pub mod distance;
pub mod error;
pub mod folding;
pub mod grid_crf;
pub mod handle;
pub mod hmm;
pub mod kalman;
pub mod matching;
pub mod memm;
pub mod metrics;
pub mod mrf;
pub mod perceptron;
pub mod ptx_kernels;
pub mod ssvm;
pub mod string;
pub mod tagging;

pub use error::{SeqError, SeqResult};
pub use handle::{LcgRng, SeqHandle, SmVersion};

#[cfg(test)]
mod e2e_tests;
