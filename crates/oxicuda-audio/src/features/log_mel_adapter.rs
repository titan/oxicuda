//! Adapter type for log-mel spectrogram tensors produced by `oxicuda-signal`.
//!
//! `oxicuda-audio` accepts pre-computed log-mel features as plain `Vec<f32>`.
//! This module provides a validated wrapper so that downstream modules receive
//! a consistent `[T × F]` row-major layout (T time frames, F mel bins).

use crate::error::{AudioError, AudioResult};

/// A validated log-mel spectrogram tensor in row-major `[T, F]` layout.
///
/// Produced by `oxicuda-signal`'s STFT + Mel filterbank + log pipeline:
///
/// ```text
/// raw_pcm  →  stft  →  mel_filterbank  →  log1p  →  LogMelInput
/// ```
///
/// The `data` slice has length `time * mels`; element `[t * mels + f]` is
/// `log(1 + power[t, f])`.
#[derive(Debug, Clone)]
pub struct LogMelInput {
    /// Flattened `[T, F]` buffer — row-major, length `time * mels`.
    pub data: Vec<f32>,
    /// Number of time frames `T`.
    pub time: usize,
    /// Number of mel bins `F`.
    pub mels: usize,
}

impl LogMelInput {
    /// Construct and validate a `LogMelInput` from an existing slice.
    ///
    /// # Errors
    ///
    /// Returns `AudioError::InvalidNumMels` if `mels == 0`,
    /// `AudioError::InvalidSequenceLength` if `time == 0`,
    /// `AudioError::ShapeMismatch` if `data.len() != time * mels`.
    pub fn from_mel(data: &[f32], time: usize, mels: usize) -> AudioResult<Self> {
        if mels == 0 {
            return Err(AudioError::InvalidNumMels(mels));
        }
        if time == 0 {
            return Err(AudioError::InvalidSequenceLength(time));
        }
        let expected = time * mels;
        if data.len() != expected {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "expected data.len() = time*mels = {expected}, got {}",
                    data.len()
                ),
            });
        }
        Ok(Self {
            data: data.to_vec(),
            time,
            mels,
        })
    }

    /// Return the log-mel value at frame `t`, bin `f`.
    #[inline]
    #[must_use]
    pub fn get(&self, t: usize, f: usize) -> f32 {
        self.data[t * self.mels + f]
    }

    /// Return a slice over time frame `t` (length `mels`).
    #[inline]
    #[must_use]
    pub fn frame(&self, t: usize) -> &[f32] {
        &self.data[t * self.mels..(t + 1) * self.mels]
    }

    /// Return the total number of elements (`time * mels`).
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the input is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_mel_ok() {
        let data = vec![1.0f32; 20 * 80];
        let inp = LogMelInput::from_mel(&data, 20, 80).expect("ok");
        assert_eq!(inp.time, 20);
        assert_eq!(inp.mels, 80);
        assert_eq!(inp.len(), 1600);
    }

    #[test]
    fn from_mel_zero_mels() {
        let r = LogMelInput::from_mel(&[], 10, 0);
        assert_eq!(r.unwrap_err(), AudioError::InvalidNumMels(0));
    }

    #[test]
    fn from_mel_zero_time() {
        let r = LogMelInput::from_mel(&[], 0, 80);
        assert_eq!(r.unwrap_err(), AudioError::InvalidSequenceLength(0));
    }

    #[test]
    fn from_mel_shape_mismatch() {
        let data = vec![0.0f32; 5];
        let r = LogMelInput::from_mel(&data, 2, 4);
        assert!(matches!(r.unwrap_err(), AudioError::ShapeMismatch { .. }));
    }

    #[test]
    fn get_and_frame() {
        let mut data = vec![0.0f32; 3 * 4];
        data[4 + 2] = 7.5;
        let inp = LogMelInput::from_mel(&data, 3, 4).expect("ok");
        assert_eq!(inp.get(1, 2), 7.5);
        assert_eq!(inp.frame(1)[2], 7.5);
    }

    #[test]
    fn is_empty_false() {
        let data = vec![0.0f32; 10];
        let inp = LogMelInput::from_mel(&data, 2, 5).expect("ok");
        assert!(!inp.is_empty());
    }

    #[test]
    fn clone_independence() {
        let data = vec![1.0f32; 6];
        let a = LogMelInput::from_mel(&data, 2, 3).expect("ok");
        let mut b = a.clone();
        b.data[0] = 99.0;
        assert_eq!(a.data[0], 1.0);
    }
}
