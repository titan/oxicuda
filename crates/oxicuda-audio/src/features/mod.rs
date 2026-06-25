//! Audio feature extraction adapters.

pub mod chroma;
pub mod cmvn;
pub mod companding;
pub mod delta;
pub mod log_mel_adapter;
pub mod log_mel_extractor;
pub mod lpc;
pub mod mel_filterbank;
pub mod mfcc;
pub mod onset;
pub mod spectral;

pub use chroma::{ChromaConfig, ChromaNorm, N_CHROMA, chroma};
pub use cmvn::{CmvnConfig, apply_cmvn, compute_cmvn};
pub use companding::{
    A_LAW_A, MU_LAW_MU, a_law_decode, a_law_decode_sample, a_law_encode, a_law_encode_sample,
    de_emphasis, mu_law_decode, mu_law_decode_sample, mu_law_dequantize, mu_law_encode,
    mu_law_encode_sample, mu_law_quantize, pre_emphasis,
};
pub use delta::{compute_delta, compute_delta_delta, stack_delta_features};
pub use log_mel_adapter::LogMelInput;
pub use log_mel_extractor::{LogMelExtractor, LogMelExtractorConfig};
pub use lpc::{
    Formant, LpcResult, autocorrelation, formants, formants_from_lpc, levinson_durbin, lpc,
};
pub use mel_filterbank::{MelFilterbank, MelFilterbankConfig};
pub use mfcc::{
    MfccConfig, log_mel_spectrogram, mel_filterbank as mfcc_mel_filterbank, mel_spectrogram, mfcc,
};
pub use onset::{
    OnsetConfig, PeakPickConfig, TempoEstimate, detect_onsets, estimate_tempo, onset_strength,
    onset_times, pick_peaks, tempo_from_envelope,
};
pub use spectral::{
    SpectralConfig, rms_energy, spectral_bandwidth, spectral_centroid, spectral_flatness,
    spectral_rolloff, zero_crossing_rate,
};
