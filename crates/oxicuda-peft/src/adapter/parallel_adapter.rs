use crate::adapter::houlsby::gelu;
use crate::handle::LcgRng;

/// Parallel adapter: runs a bottleneck FFN in parallel with the main FFN, then sums.
///
/// Architecture: `out = ffn_out + (x → Linear_down → GELU → Linear_up)`
///
/// Unlike sequential adapters, the parallel adapter branch runs independently from the
/// main FFN sub-layer and its output is summed with the FFN output.
#[derive(Debug, Clone)]
pub struct ParallelAdapter {
    /// Input dimension.
    pub in_dim: usize,
    /// Bottleneck dimension.
    pub bottleneck_dim: usize,
    /// Down-projection weight, shape `[bottleneck_dim × in_dim]`.
    pub down_w: Vec<f32>,
    /// Down-projection bias, shape `[bottleneck_dim]`.
    pub down_b: Vec<f32>,
    /// Up-projection weight, shape `[in_dim × bottleneck_dim]`.
    pub up_w: Vec<f32>,
    /// Up-projection bias, shape `[in_dim]`.
    pub up_b: Vec<f32>,
}

impl ParallelAdapter {
    /// Construct a `ParallelAdapter`.
    ///
    /// `down_w` He-initialised; `up_w` zero-initialised so the adapter initially contributes nothing.
    #[must_use]
    pub fn new(in_dim: usize, bottleneck_dim: usize, rng: &mut LcgRng) -> Self {
        let he_std = (2.0_f32 / in_dim as f32).sqrt();
        let mut down_w = vec![0.0_f32; bottleneck_dim * in_dim];
        rng.fill_normal(&mut down_w);
        for v in down_w.iter_mut() {
            *v *= he_std;
        }
        let down_b = vec![0.0_f32; bottleneck_dim];
        let up_w = vec![0.0_f32; in_dim * bottleneck_dim];
        let up_b = vec![0.0_f32; in_dim];
        Self {
            in_dim,
            bottleneck_dim,
            down_w,
            down_b,
            up_w,
            up_b,
        }
    }

    /// Compute `ffn_out + adapter(x)` where `adapter(x) = Linear_up(GELU(Linear_down(x)))`.
    ///
    /// Both `ffn_out` and `x` must have length `seq_len * in_dim`.
    /// Returns a vector of length `seq_len * in_dim`.
    #[must_use]
    pub fn forward_parallel(&self, ffn_out: &[f32], x: &[f32], seq_len: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(seq_len * self.in_dim);
        for t in 0..seq_len {
            let token = &x[t * self.in_dim..(t + 1) * self.in_dim];
            let ffn_token = &ffn_out[t * self.in_dim..(t + 1) * self.in_dim];
            // Down → GELU
            let hidden: Vec<f32> = (0..self.bottleneck_dim)
                .map(|i| {
                    let acc = self.down_b[i]
                        + self.down_w[i * self.in_dim..(i + 1) * self.in_dim]
                            .iter()
                            .zip(token.iter())
                            .map(|(w, xi)| w * xi)
                            .sum::<f32>();
                    gelu(acc)
                })
                .collect();
            // Up + sum with ffn_out
            for (i, &ffn_val) in ffn_token.iter().enumerate() {
                let acc = self.up_b[i]
                    + hidden
                        .iter()
                        .enumerate()
                        .map(|(j, &h)| self.up_w[i * self.bottleneck_dim + j] * h)
                        .sum::<f32>();
                out.push(ffn_val + acc);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Build a deterministic input vector in `[-1, 1)` of the given length.
    fn make_input(len: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..len).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 1: output length == seq_len * in_dim
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn parallel_adapter_output_shape() {
        let mut rng = LcgRng::new(1);
        let adapter = ParallelAdapter::new(8, 4, &mut rng);
        let ffn_out = make_input(8 * 3, 2);
        let x = make_input(8 * 3, 3);
        let out = adapter.forward_parallel(&ffn_out, &x, 3);
        assert_eq!(out.len(), 24, "output length must be seq_len * in_dim");
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 2: at init (up_w=0, up_b=0), forward_parallel(ffn_out, x) == ffn_out exactly.
    //
    // Proof: acc_i = up_b[i] + Σ_j 0*hidden[j] = 0; out[i] = ffn_out[i] + 0.
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn parallel_adapter_zero_up_w_passes_ffn_out_through() {
        let mut rng = LcgRng::new(7);
        let adapter = ParallelAdapter::new(8, 4, &mut rng);
        assert!(
            adapter.up_w.iter().all(|&v| v == 0.0),
            "up_w must be zero-initialised"
        );
        assert!(
            adapter.up_b.iter().all(|&v| v == 0.0),
            "up_b must be zero-initialised"
        );
        let ffn_out = make_input(8, 9);
        let x = make_input(8, 11);
        let out = adapter.forward_parallel(&ffn_out, &x, 1);
        for (i, (&got, &expected)) in out.iter().zip(ffn_out.iter()).enumerate() {
            assert_eq!(
                got, expected,
                "output[{i}] must equal ffn_out[{i}] when up_w=0 and up_b=0"
            );
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 3: same RNG seed → byte-identical forward output (determinism)
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn parallel_adapter_determinism_fixed_seed() {
        let ffn_out = make_input(16, 88);
        let x = make_input(16, 99);
        let mut rng_a = LcgRng::new(42);
        let mut rng_b = LcgRng::new(42);
        let adapter_a = ParallelAdapter::new(16, 4, &mut rng_a);
        let adapter_b = ParallelAdapter::new(16, 4, &mut rng_b);
        let out_a = adapter_a.forward_parallel(&ffn_out, &x, 1);
        let out_b = adapter_b.forward_parallel(&ffn_out, &x, 1);
        assert_eq!(
            out_a, out_b,
            "same seed must yield identical forward outputs"
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 4: all outputs are finite
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn parallel_adapter_finite_outputs() {
        let mut rng = LcgRng::new(13);
        let adapter = ParallelAdapter::new(8, 4, &mut rng);
        let ffn_out = make_input(8 * 4, 15);
        let x = make_input(8 * 4, 17);
        let out = adapter.forward_parallel(&ffn_out, &x, 4);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "all outputs must be finite"
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 5: non-zero up_w adds the adapter branch contribution → output != ffn_out
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn parallel_adapter_nonzero_up_w_differs_from_ffn_out() {
        let mut rng = LcgRng::new(21);
        let mut adapter = ParallelAdapter::new(8, 4, &mut rng);
        let ffn_out = make_input(8, 23);
        let x = make_input(8, 25);
        for v in adapter.up_w.iter_mut() {
            *v = 1.0;
        }
        let out = adapter.forward_parallel(&ffn_out, &x, 1);
        assert_ne!(
            out, ffn_out,
            "non-zero up_w must add a non-zero adapter contribution to ffn_out"
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // Test 6: analytic — when x = 0 and down_b = 0, gelu(0) = 0, so
    // the adapter branch contributes nothing and output == ffn_out even
    // when up_w is fully non-zero.  Tests the architecture's zero-crossing
    // property of GELU.
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn parallel_adapter_zero_x_with_zero_bias_passes_ffn_out_unchanged() {
        let mut rng = LcgRng::new(33);
        let mut adapter = ParallelAdapter::new(8, 4, &mut rng);
        // Activate the up-projection so we're testing the GELU zero-crossing,
        // not the trivial up_w=0 case.
        for v in adapter.up_w.iter_mut() {
            *v = 1.0;
        }
        // down_b is all zeros at init; x = 0 → down_w@0 + 0 = 0 → gelu(0) = 0
        // → up_proj(0) = 0 → output = ffn_out + 0 = ffn_out
        let ffn_out = make_input(8, 40);
        let x_zero = vec![0.0_f32; 8];
        let out = adapter.forward_parallel(&ffn_out, &x_zero, 1);
        for (i, (&got, &expected)) in out.iter().zip(ffn_out.iter()).enumerate() {
            assert_eq!(
                got, expected,
                "out[{i}] must equal ffn_out[{i}] when x=0 (gelu(0)=0 kills the branch)"
            );
        }
    }
}
