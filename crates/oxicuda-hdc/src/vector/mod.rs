//! Hypervector types: binary {±1}^D, integer Z^D, complex unit (FHRR), HRR real-valued,
//! HRR with FFT-accelerated binding, and Sparse Block-Codes (SBC).

pub mod binary;
pub mod complex;
pub mod hrr;
pub mod hrr_fft;
pub mod integer;
pub mod sparse_block_codes;

pub use hrr::{
    HrrItemMemory, hrr_bind, hrr_bind_sequence, hrr_bundle, hrr_bundle_add, hrr_cosine,
    hrr_normalize, hrr_unbind, random_hrr,
};
pub use hrr_fft::{HrrFft, HrrFftConfig};
pub use sparse_block_codes::{SbcConfig, SbcVec};
