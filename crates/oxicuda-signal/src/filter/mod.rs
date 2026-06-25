//! Digital filter design and application (FIR, IIR, Wiener, Adaptive).

pub mod adaptive;
pub mod butterworth;
pub mod deconv;
pub mod fir;
pub mod iir;
pub mod median;
pub mod remez;
pub mod wiener;

pub use adaptive::{AdaptiveLmsConfig, AdaptiveLmsState, lms_filter, nlms_filter, rls_filter};
pub use butterworth::{ButterworthConfig, ButterworthFilter, FilterType};
pub use deconv::{richardson_lucy, wiener_deconvolve};
pub use fir::{
    design_bandpass, design_highpass, design_lowpass, design_raised_cosine, emit_fir_direct_kernel,
    fir_apply, freq_response as fir_freq_response,
};
pub use iir::{Biquad, iir_apply};
pub use median::{median_filter_1d, weighted_median_1d};
pub use remez::{
    RemezBand, magnitude_at as remez_magnitude_at, remez, remez_bandpass, remez_bandstop,
    remez_highpass, remez_lowpass,
};
pub use wiener::{
    apply_wiener_gains, estimate_noise_psd, local_wiener_1d, wiener_filter, wiener_gain,
};
