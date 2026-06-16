//! NeRF positional encoding: γ(p).
//!
//! For each input dimension d and frequency level k:
//!   [sin(2^k · π · p_d), cos(2^k · π · p_d)]
//!
//! With L levels and `input_dim` inputs:
//!   encoded_dim = input_dim * 2 * L
//!   (optionally: encoded_dim += input_dim if `include_input`)

use crate::error::{NerfError, NerfResult};

/// Configuration for positional encoding.
#[derive(Debug, Clone, Copy)]
pub struct PosEncConfig {
    /// L — number of frequency levels.
    pub n_freq: usize,
    /// Whether to prepend the raw (unencoded) input.
    pub include_input: bool,
    /// Input dimensionality (3 for xyz, 3 for view direction).
    pub input_dim: usize,
}

impl PosEncConfig {
    /// Compute the output dimensionality of the encoding.
    #[must_use]
    pub fn output_dim(&self) -> usize {
        let enc = self.input_dim * 2 * self.n_freq;
        if self.include_input {
            enc + self.input_dim
        } else {
            enc
        }
    }
}

/// Encode a flat array of N input vectors of length `cfg.input_dim`.
///
/// Input layout: `[x0_0, x0_1, ..., x0_{D-1}, x1_0, ...]` (N × D).
/// Output layout: `[enc(x0), enc(x1), ...]` (N × output_dim).
///
/// # Errors
///
/// Returns `InvalidFreqLevels` if `n_freq == 0`,
/// `EmptyInput` if `input` is empty,
/// `DimensionMismatch` if `input.len() % input_dim != 0`.
pub fn positional_encode(input: &[f32], cfg: &PosEncConfig) -> NerfResult<Vec<f32>> {
    if cfg.n_freq == 0 {
        return Err(NerfError::InvalidFreqLevels { levels: 0 });
    }
    if input.is_empty() {
        return Err(NerfError::EmptyInput);
    }
    if !input.len().is_multiple_of(cfg.input_dim) {
        return Err(NerfError::DimensionMismatch {
            expected: cfg.input_dim,
            got: input.len() % cfg.input_dim,
        });
    }

    let n = input.len() / cfg.input_dim;
    let out_dim = cfg.output_dim();
    let mut out = vec![0.0_f32; n * out_dim];

    for (pt_idx, out_chunk) in out.chunks_mut(out_dim).enumerate() {
        let in_chunk = &input[pt_idx * cfg.input_dim..(pt_idx + 1) * cfg.input_dim];
        let mut write_pos = 0;

        if cfg.include_input {
            out_chunk[..cfg.input_dim].copy_from_slice(in_chunk);
            write_pos += cfg.input_dim;
        }

        for k in 0..cfg.n_freq {
            let freq = (2_u32.pow(k as u32)) as f32 * std::f32::consts::PI;
            for &val in in_chunk {
                out_chunk[write_pos] = (freq * val).sin();
                write_pos += 1;
                out_chunk[write_pos] = (freq * val).cos();
                write_pos += 1;
            }
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_dim_no_include() {
        let cfg = PosEncConfig {
            n_freq: 4,
            include_input: false,
            input_dim: 3,
        };
        assert_eq!(cfg.output_dim(), 24);
    }

    #[test]
    fn output_dim_with_include() {
        let cfg = PosEncConfig {
            n_freq: 4,
            include_input: true,
            input_dim: 3,
        };
        assert_eq!(cfg.output_dim(), 27);
    }

    #[test]
    fn single_point_shape() {
        let cfg = PosEncConfig {
            n_freq: 2,
            include_input: false,
            input_dim: 3,
        };
        let input = vec![0.1_f32, 0.2, 0.3];
        let out = positional_encode(&input, &cfg).expect("positional_encode should succeed");
        assert_eq!(out.len(), cfg.output_dim());
    }

    #[test]
    fn error_on_zero_freq() {
        let cfg = PosEncConfig {
            n_freq: 0,
            include_input: false,
            input_dim: 3,
        };
        assert!(positional_encode(&[0.0_f32; 3], &cfg).is_err());
    }
}
