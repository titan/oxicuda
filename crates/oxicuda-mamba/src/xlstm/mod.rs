//! xLSTM — Extended LSTM variants (Beck et al. 2024 ICML).
//!
//! Provides two novel LSTM cell types:
//!
//! - **sLSTM** (scalar LSTM): exponential gating with a max-state stabilizer;
//!   scalar cell state with multi-head recurrence.
//! - **mLSTM** (matrix LSTM): replaces the scalar cell state with an outer-product
//!   memory matrix `C ∈ ℝ^{d×d}`, enabling content-addressable associative memory.
//!
//! ## Reference
//!
//! Beck et al. (2024) "xLSTM: Extended Long Short-Term Memory",
//! ICML 2024. <https://arxiv.org/abs/2405.04517>
//!
//! ## Usage
//!
//! ```rust,ignore
//! use oxicuda_mamba::xlstm::{SLstm, SLstmConfig, MLstm, MLstmConfig};
//! use oxicuda_mamba::handle::LcgRng;
//!
//! let mut rng = LcgRng::new(42);
//! let slstm = SLstm::new(SLstmConfig { input_dim: 16, hidden_dim: 8, n_heads: 2, r: 2 }, &mut rng).unwrap();
//! let mlstm = MLstm::new(MLstmConfig { input_dim: 16, head_dim: 8, n_heads: 2 }, &mut rng).unwrap();
//! ```

pub mod mlstm;
pub mod slstm;

pub use mlstm::{MLstm, MLstmConfig, MLstmState, MLstmWeights};
pub use slstm::{SLstm, SLstmConfig, SLstmState, SLstmWeights};

// ─── Integration tests (both types together) ─────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn both_types_stack() {
        // Verifies that sLSTM and mLSTM can be composed in sequence:
        // sLSTM output feeds mLSTM input.
        let mut rng = LcgRng::new(7);

        let slstm_cfg = SLstmConfig {
            input_dim: 8,
            hidden_dim: 4,
            n_heads: 2,
            r: 2,
        };
        let slstm = SLstm::new(slstm_cfg.clone(), &mut rng).expect("SLstm::new");

        // sLSTM output dim = n_heads * hidden_dim = 8
        let mlstm_cfg = MLstmConfig {
            input_dim: slstm_cfg.n_heads * slstm_cfg.hidden_dim,
            head_dim: 4,
            n_heads: 2,
        };
        let mlstm = MLstm::new(mlstm_cfg.clone(), &mut rng).expect("MLstm::new");

        let seq_len = 6;
        let input: Vec<f32> = (0..seq_len * slstm_cfg.input_dim)
            .map(|i| i as f32 * 0.02)
            .collect();

        // Process through sLSTM
        let slstm_out = slstm.forward(&input, seq_len).expect("sLSTM forward");
        assert_eq!(
            slstm_out.len(),
            seq_len * slstm_cfg.n_heads * slstm_cfg.hidden_dim
        );

        // Feed sLSTM output into mLSTM
        let mlstm_out = mlstm.forward(&slstm_out, seq_len).expect("mLSTM forward");
        assert_eq!(
            mlstm_out.len(),
            seq_len * mlstm_cfg.n_heads * mlstm_cfg.head_dim
        );

        assert!(
            mlstm_out.iter().all(|v| v.is_finite()),
            "stacked output must be finite"
        );
    }
}
