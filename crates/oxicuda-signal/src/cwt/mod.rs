//! Continuous Wavelet Transform (CWT).
//!
//! Frequency-domain implementation (O(N log N) per scale) supporting Morlet
//! (complex analytic) and Mexican Hat (real) mother wavelets.
pub mod cwt;
pub use cwt::{
    CwtConfig, CwtOutput, CwtWavelet, cwt, cwt_cone_of_influence, cwt_global_power, cwt_ridge,
    cwt_scalogram,
};
