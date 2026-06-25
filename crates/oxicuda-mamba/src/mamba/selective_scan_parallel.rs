//! Parallel (work-efficient Blelloch) Mamba selective scan — CPU model.
//!
//! # Motivation
//!
//! The sequential reference [`crate::mamba::selective_scan::selective_scan`]
//! walks the time axis one step at a time:
//!
//! ```text
//! h[b, t, d, n] = Ā[b, t, d, n] · h[b, t-1, d, n] + B̄[b, t, n] · u[b, t, d]
//! y[b, t, d]    = Σ_n C[b, t, n] · h[b, t, d, n]
//! ```
//!
//! On a GPU this recurrence is instead evaluated with a *parallel associative
//! scan* (the "selective scan" kernel): each `(batch, channel, state)` triple
//! is an independent scalar linear recurrence
//! `h_t = a_t · h_{t-1} + b_t`, which is the prefix scan of the
//! [`crate::ssm::parallel_scan::ScanPair`] `(a_t, b_t)` under the associative
//! operator `(a₁,b₁) ⊕ (a₂,b₂) = (a₂a₁, a₂b₁ + b₂)`.  This module computes the
//! whole forward pass through the **work-efficient Blelloch up-sweep /
//! down-sweep scan** — the exact algorithm the fused GPU kernel realises — so
//! the parallel and sequential paths can be checked for equivalence on the CPU.
//!
//! The fused **GPU** kernel execution itself (PTX launch, warp-shuffle
//! butterfly, shared-memory tiling) is hardware-gated and lives in
//! [`crate::ptx_kernels::parallel_scan_ptx`]; only the *algorithm* is exercised
//! here.
//!
//! All arithmetic is `f32`, matching the sequential reference; the only
//! difference is the order of associative combines, so the two agree to `f32`
//! rounding (asserted by [`verify_selective_scan_equivalence`]).

use crate::error::{MambaError, MambaResult};
use crate::mamba::selective_scan::{SelectiveScanConfig, softplus};
use crate::ssm::parallel_scan::{ScanPair, blelloch_inclusive_scan};

/// Mamba selective scan (S6) computed via the parallel Blelloch prefix scan.
///
/// Drop-in replacement for [`crate::mamba::selective_scan::selective_scan`]
/// with identical inputs, output layout and validation, but the per-element
/// scalar recurrences are evaluated with the work-efficient associative scan
/// instead of a sequential fold.
///
/// # Inputs (all row-major flat `f32` slices)
///
/// * `u`      — `[B, L, D]`  input sequence.
/// * `delta`  — `[B, L, D]`  raw step sizes (passed through `softplus`).
/// * `a_log`  — `[D, N]`     `log(-A)`, so `A = -exp(a_log)`.
/// * `b_proj` — `[B, L, N]`  input-dependent B projection.
/// * `c_proj` — `[B, L, N]`  input-dependent C projection.
///
/// # Output
///
/// `y` — `[B, L, D]`, length `B · L · D`.
///
/// # Errors
///
/// [`MambaError::DimensionMismatch`] if any input slice has the wrong length.
pub fn selective_scan_parallel(
    u: &[f32],
    delta: &[f32],
    a_log: &[f32],
    b_proj: &[f32],
    c_proj: &[f32],
    config: &SelectiveScanConfig,
) -> MambaResult<Vec<f32>> {
    let cfg = config;

    // ── Validate input shapes (mirror the sequential reference). ──────────────
    let expected_u = cfg.u_numel();
    if u.len() != expected_u {
        return Err(MambaError::DimensionMismatch {
            expected: expected_u,
            got: u.len(),
        });
    }
    if delta.len() != expected_u {
        return Err(MambaError::DimensionMismatch {
            expected: expected_u,
            got: delta.len(),
        });
    }
    let expected_a = cfg.d_model * cfg.d_state;
    if a_log.len() != expected_a {
        return Err(MambaError::DimensionMismatch {
            expected: expected_a,
            got: a_log.len(),
        });
    }
    let expected_bc = cfg.bc_numel();
    if b_proj.len() != expected_bc {
        return Err(MambaError::DimensionMismatch {
            expected: expected_bc,
            got: b_proj.len(),
        });
    }
    if c_proj.len() != expected_bc {
        return Err(MambaError::DimensionMismatch {
            expected: expected_bc,
            got: c_proj.len(),
        });
    }

    let mut y = vec![0.0_f32; expected_u];
    let l = cfg.seq_len;

    // Pre-compute softplus(Δ) once per (b, t, d): used to build both Ā and B̄.
    // For every independent (b, d, n) scalar recurrence we assemble the length-L
    // pair sequence and run the work-efficient scan, then fold C into y.
    let mut pairs: Vec<ScanPair> = Vec::with_capacity(l);
    for b in 0..cfg.batch {
        for d in 0..cfg.d_model {
            for n in 0..cfg.d_state {
                let a_val = -(a_log[cfg.a_idx(d, n)].exp());
                pairs.clear();
                for t in 0..l {
                    let dt = softplus(delta[cfg.u_idx(b, t, d)]);
                    let a_bar = (dt * a_val).exp();
                    let b_bar = dt * b_proj[cfg.bc_idx(b, t, n)];
                    let drive = b_bar * u[cfg.u_idx(b, t, d)];
                    pairs.push(ScanPair { a: a_bar, b: drive });
                }
                let scanned = blelloch_inclusive_scan(&pairs);
                for (t, sp) in scanned.iter().enumerate() {
                    // y[b, t, d] += C[b, t, n] · h[b, t, d, n]
                    y[cfg.u_idx(b, t, d)] += c_proj[cfg.bc_idx(b, t, n)] * sp.b;
                }
            }
        }
    }

    Ok(y)
}

/// Check that the parallel (Blelloch) selective scan agrees with the sequential
/// reference within `tol` (max absolute difference).
///
/// Returns `true` when every output element matches; useful in tests and as a
/// numerical guard when validating the GPU kernel against the CPU reference.
///
/// # Errors
///
/// Propagates any shape error from either scan implementation.
pub fn verify_selective_scan_equivalence(
    u: &[f32],
    delta: &[f32],
    a_log: &[f32],
    b_proj: &[f32],
    c_proj: &[f32],
    config: &SelectiveScanConfig,
    tol: f32,
) -> MambaResult<bool> {
    use crate::mamba::selective_scan::selective_scan;
    let seq = selective_scan(u, delta, a_log, b_proj, c_proj, config)?;
    let par = selective_scan_parallel(u, delta, a_log, b_proj, c_proj, config)?;
    let max_diff = seq
        .iter()
        .zip(par.iter())
        .map(|(&s, &p)| (s - p).abs())
        .fold(0.0_f32, f32::max);
    Ok(max_diff <= tol)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::mamba::selective_scan::selective_scan;

    fn randn(rng: &mut LcgRng, n: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; n];
        rng.fill_normal(&mut v);
        v
    }

    /// Parallel == sequential for a representative random configuration.
    #[test]
    fn parallel_matches_sequential_random() {
        let mut rng = LcgRng::new(7);
        let (b, l, d, n) = (2_usize, 16_usize, 4_usize, 8_usize);
        let cfg = SelectiveScanConfig::new(b, l, d, n).expect("config");
        let u = randn(&mut rng, b * l * d);
        let delta = randn(&mut rng, b * l * d);
        let a_log = randn(&mut rng, d * n);
        let b_proj = randn(&mut rng, b * l * n);
        let c_proj = randn(&mut rng, b * l * n);

        let seq = selective_scan(&u, &delta, &a_log, &b_proj, &c_proj, &cfg).expect("seq");
        let par =
            selective_scan_parallel(&u, &delta, &a_log, &b_proj, &c_proj, &cfg).expect("parallel");
        assert_eq!(seq.len(), par.len());
        for (i, (&s, &p)) in seq.iter().zip(par.iter()).enumerate() {
            assert!(
                (s - p).abs() < 1e-3,
                "mismatch at {i}: seq={s}, par={p}, diff={}",
                (s - p).abs()
            );
        }
    }

    /// Equivalence holds across several non-power-of-two sequence lengths.
    #[test]
    fn parallel_matches_sequential_various_lengths() {
        let mut rng = LcgRng::new(13);
        for &l in &[1_usize, 3, 5, 7, 9, 15, 17, 31, 33] {
            let (b, d, n) = (1_usize, 3_usize, 4_usize);
            let cfg = SelectiveScanConfig::new(b, l, d, n).expect("config");
            let u = randn(&mut rng, b * l * d);
            let delta = randn(&mut rng, b * l * d);
            let a_log = randn(&mut rng, d * n);
            let b_proj = randn(&mut rng, b * l * n);
            let c_proj = randn(&mut rng, b * l * n);
            let ok =
                verify_selective_scan_equivalence(&u, &delta, &a_log, &b_proj, &c_proj, &cfg, 1e-3)
                    .expect("verify");
            assert!(ok, "parallel/sequential disagree at L={l}");
        }
    }

    /// Output shape matches `[B, L, D]`.
    #[test]
    fn parallel_output_shape() {
        let mut rng = LcgRng::new(1);
        let (b, l, d, n) = (2_usize, 4_usize, 3_usize, 8_usize);
        let cfg = SelectiveScanConfig::new(b, l, d, n).expect("config");
        let u = randn(&mut rng, b * l * d);
        let delta = randn(&mut rng, b * l * d);
        let a_log = randn(&mut rng, d * n);
        let b_proj = randn(&mut rng, b * l * n);
        let c_proj = randn(&mut rng, b * l * n);
        let y =
            selective_scan_parallel(&u, &delta, &a_log, &b_proj, &c_proj, &cfg).expect("parallel");
        assert_eq!(y.len(), b * l * d);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    /// Zero input drives the state to zero ⇒ zero output.
    #[test]
    fn parallel_zero_input_zero_output() {
        let (b, l, d, n) = (1_usize, 6_usize, 2_usize, 4_usize);
        let cfg = SelectiveScanConfig::new(b, l, d, n).expect("config");
        let u = vec![0.0_f32; b * l * d];
        let delta = vec![0.5_f32; b * l * d];
        let a_log = vec![0.0_f32; d * n];
        let b_proj = vec![0.0_f32; b * l * n];
        let c_proj = vec![1.0_f32; b * l * n];
        let y =
            selective_scan_parallel(&u, &delta, &a_log, &b_proj, &c_proj, &cfg).expect("parallel");
        for (i, &v) in y.iter().enumerate() {
            assert!(v.abs() < 1e-7, "y[{i}]={v} should be zero");
        }
    }

    /// Batch entries are processed independently (replicated input ⇒ identical
    /// per-batch output).
    #[test]
    fn parallel_batch_independence() {
        let mut rng = LcgRng::new(77);
        let (l, d, n) = (8_usize, 3_usize, 4_usize);
        let cfg1 = SelectiveScanConfig::new(1, l, d, n).expect("config1");
        let u1 = randn(&mut rng, l * d);
        let delta1 = randn(&mut rng, l * d);
        let b1 = randn(&mut rng, l * n);
        let c1 = randn(&mut rng, l * n);
        let a_log = vec![0.3_f32; d * n];
        let y1 = selective_scan_parallel(&u1, &delta1, &a_log, &b1, &c1, &cfg1).expect("single");

        let cfg2 = SelectiveScanConfig::new(2, l, d, n).expect("config2");
        let u2: Vec<f32> = u1.iter().chain(u1.iter()).copied().collect();
        let delta2: Vec<f32> = delta1.iter().chain(delta1.iter()).copied().collect();
        let b2: Vec<f32> = b1.iter().chain(b1.iter()).copied().collect();
        let c2: Vec<f32> = c1.iter().chain(c1.iter()).copied().collect();
        let y2 = selective_scan_parallel(&u2, &delta2, &a_log, &b2, &c2, &cfg2).expect("batch");

        let stride = l * d;
        for (i, (&a, &c)) in y2[..stride].iter().zip(y2[stride..].iter()).enumerate() {
            assert!((a - c).abs() < 1e-5, "batch halves differ at {i}");
        }
        for (i, (&s, &b)) in y1.iter().zip(y2[..stride].iter()).enumerate() {
            assert!((s - b).abs() < 1e-5, "single vs batch differ at {i}");
        }
    }

    /// Wrong input lengths surface as `DimensionMismatch`.
    #[test]
    fn parallel_shape_errors() {
        let cfg = SelectiveScanConfig::new(1, 4, 2, 4).expect("config");
        let u = vec![0.0_f32; 5]; // should be 8
        let delta = vec![0.0_f32; 8];
        let a_log = vec![0.0_f32; 8];
        let b_proj = vec![0.0_f32; 16];
        let c_proj = vec![0.0_f32; 16];
        assert!(matches!(
            selective_scan_parallel(&u, &delta, &a_log, &b_proj, &c_proj, &cfg),
            Err(MambaError::DimensionMismatch { .. })
        ));
    }

    /// Long-sequence equivalence (stable poles, 256 steps).
    #[test]
    fn parallel_long_sequence_equivalence() {
        let mut rng = LcgRng::new(99);
        let (b, l, d, n) = (1_usize, 256_usize, 2_usize, 4_usize);
        let cfg = SelectiveScanConfig::new(b, l, d, n).expect("config");
        let u: Vec<f32> = (0..b * l * d).map(|_| rng.next_f32() * 0.2 - 0.1).collect();
        let delta = vec![0.0_f32; b * l * d]; // softplus(0)=ln2
        let a_log = vec![0.0_f32; d * n]; // A = -1, stable
        let b_proj = vec![0.05_f32; b * l * n];
        let c_proj = vec![1.0_f32; b * l * n];
        let ok =
            verify_selective_scan_equivalence(&u, &delta, &a_log, &b_proj, &c_proj, &cfg, 1e-3)
                .expect("verify");
        assert!(ok, "parallel/sequential disagree at L=256");
    }
}
