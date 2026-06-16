//! Hidden Markov Models — Baum-Welch training and Viterbi decoding.

pub mod hmm;

pub use hmm::{
    HmmConfig, HmmDecodeResult, HmmModel, HmmObsType, hmm_decode, hmm_decode_gaussian, hmm_fit,
    hmm_fit_gaussian, hmm_generate, hmm_log_likelihood, hmm_stationary,
};
