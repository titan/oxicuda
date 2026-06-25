//! Hypervector types: binary {±1}^D, integer Z^D, complex unit (FHRR), phasor-only FHRR,
//! HRR real-valued, HRR with FFT-accelerated binding, and Sparse Block-Codes (SBC).

pub mod binary;
pub mod complex;
pub mod fhrr;
pub mod fpe;
pub mod hrr;
pub mod hrr_fft;
pub mod integer;
pub mod quantize;
pub mod sparse_binary;
pub mod sparse_block_codes;

pub use fhrr::{
    fhrr_bind, fhrr_bundle, fhrr_cosine, fhrr_inverse, fhrr_to_interleaved, fhrr_unbind,
    random_fhrr, validate_fhrr,
};
pub use fpe::{FpeBase, fpe_similarity, kernel_profile};
pub use hrr::{
    HrrItemMemory, hrr_bind, hrr_bind_sequence, hrr_bundle, hrr_bundle_add, hrr_cosine,
    hrr_normalize, hrr_unbind, random_hrr,
};
pub use hrr_fft::{HrrFft, HrrFftConfig};
pub use quantize::{
    binarize_checked, dequantize_levels, quantization_error, quantize_levels, sign_binarize,
    ternarize,
};
pub use sparse_binary::{SparseBinaryConfig, SparseBinaryVec};
pub use sparse_block_codes::{SbcConfig, SbcVec};
