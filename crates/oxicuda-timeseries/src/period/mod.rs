//! Period / seasonality detection.
//!
//! [`fft_period`] provides `O(T log T)` FFT-based period detection routed
//! through the pure-CPU real FFT of the sibling `oxicuda-fft` crate, replacing
//! the `O(T²)` time-domain DFT used inside [`crate::timesnet`]. Correctness is
//! pinned to a direct `O(T²)` autocorrelation oracle in the unit tests
//! (elementwise agreement ≤ 1e-8).

pub mod fft_period;

pub use fft_period::{
    PeriodCandidate, PeriodConfig, autocorrelation_fft, detect_period_fft,
    detect_period_fft_ranked, detect_period_fft_with,
};
