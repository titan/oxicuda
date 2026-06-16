//! Delay-and-Sum (DAS) and MVDR beamformer implementations.
//!
//! # Overview
//!
//! MVDR (Minimum Variance Distortionless Response), also known as the Capon
//! beamformer, computes optimal weights `w` that minimise output power while
//! maintaining unit response in the look direction:
//!
//! ```text
//! w = R⁻¹ a / (aᵀ R⁻¹ a)
//! ```
//!
//! where `R` is the sample covariance matrix and `a` is the steering vector.
//!
//! The simpler Delay-and-Sum (DAS) beamformer time-aligns each sensor signal
//! by its integer sample delay then sums across sensors.

use crate::error::{SignalError, SignalResult};

/// MVDR beamforming configuration.
///
/// Carries the sensor-array geometry parameters required to set up a
/// Delay-and-Sum or MVDR beamformer pipeline.
#[derive(Debug, Clone)]
pub struct MvdrConfig {
    /// Number of sensors (array elements).
    pub n_sensors: usize,
    /// Number of time snapshots (samples per sensor).
    pub n_snapshots: usize,
}

// ────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ────────────────────────────────────────────────────────────────────────────

/// Invert an `n × n` real matrix `a` (row-major) using Gaussian elimination
/// with partial pivoting, writing the result into `inv` (pre-allocated `n²`).
///
/// Returns `Err(SignalError::InvalidParameter(...))` if a pivot smaller than
/// `1e-12` is encountered (singular or near-singular matrix).
fn gauss_invert(a: &[f64], n: usize, inv: &mut [f64]) -> SignalResult<()> {
    // Build augmented matrix [A | I], stored as a flat Vec of length n × 2n.
    let cols = 2 * n;
    let mut aug = vec![0.0_f64; n * cols];

    for row in 0..n {
        for col in 0..n {
            aug[row * cols + col] = a[row * n + col];
        }
        aug[row * cols + n + row] = 1.0;
    }

    // Forward elimination with partial pivoting.
    for pivot_col in 0..n {
        // Find the row with the largest absolute value in the current column.
        let mut max_val = aug[pivot_col * cols + pivot_col].abs();
        let mut max_row = pivot_col;
        for row in (pivot_col + 1)..n {
            let v = aug[row * cols + pivot_col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_val < 1e-12 {
            return Err(SignalError::InvalidParameter(
                "singular covariance matrix".to_string(),
            ));
        }

        // Swap rows pivot_col and max_row.
        if max_row != pivot_col {
            for col in 0..cols {
                aug.swap(pivot_col * cols + col, max_row * cols + col);
            }
        }

        // Scale the pivot row so the diagonal element becomes 1.
        let pivot = aug[pivot_col * cols + pivot_col];
        for col in 0..cols {
            aug[pivot_col * cols + col] /= pivot;
        }

        // Eliminate all other rows in this column.
        for row in 0..n {
            if row == pivot_col {
                continue;
            }
            let factor = aug[row * cols + pivot_col];
            if factor == 0.0 {
                continue;
            }
            for col in 0..cols {
                let sub = factor * aug[pivot_col * cols + col];
                aug[row * cols + col] -= sub;
            }
        }
    }

    // Extract the right half as the inverse.
    for row in 0..n {
        for col in 0..n {
            inv[row * n + col] = aug[row * cols + n + col];
        }
    }

    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// Public API
// ────────────────────────────────────────────────────────────────────────────

/// Delay-and-Sum (DAS) beamformer.
///
/// Time-aligns each sensor's signal by an integer sample delay then coherently
/// sums across sensors.
///
/// # Arguments
///
/// * `signals` — flattened `[n_sensors × n_snapshots]` row-major array; row `k`
///   holds `n_snapshots` samples from sensor `k`.
/// * `steering` — `[n_sensors]` per-sensor delays as `f64`; each value is
///   floored to a `usize` internally.  Delay `d[k]` means sensor `k`'s sample
///   at output time `t` is `signals[k][t - d[k]]`; indices below 0 contribute
///   zero.
/// * `n_sensors` — number of sensors.
/// * `n_snapshots` — number of time samples per sensor.
///
/// # Returns
///
/// A `Vec<f64>` of length `n_snapshots`.
///
/// # Errors
///
/// * [`SignalError::DimensionMismatch`] — `signals` or `steering` have wrong length.
/// * [`SignalError::InvalidParameter`] — any delay is negative or non-finite.
pub fn delay_and_sum(
    signals: &[f64],
    steering: &[f64],
    n_sensors: usize,
    n_snapshots: usize,
) -> SignalResult<Vec<f64>> {
    // Validate input dimensions.
    let expected_signals = n_sensors * n_snapshots;
    if signals.len() != expected_signals {
        return Err(SignalError::DimensionMismatch {
            expected: format!("n_sensors * n_snapshots = {expected_signals}"),
            got: format!("{}", signals.len()),
        });
    }
    if steering.len() != n_sensors {
        return Err(SignalError::DimensionMismatch {
            expected: format!("n_sensors = {n_sensors}"),
            got: format!("{}", steering.len()),
        });
    }

    // Parse and validate delays.
    let mut delays = Vec::with_capacity(n_sensors);
    for (k, &d) in steering.iter().enumerate() {
        if !d.is_finite() || d < 0.0 {
            return Err(SignalError::InvalidParameter(format!(
                "steering delay for sensor {k} is invalid ({d}); must be a non-negative finite value"
            )));
        }
        delays.push(d as usize);
    }

    // Accumulate the DAS output.
    let mut output = vec![0.0_f64; n_snapshots];
    for (k, &delay) in delays.iter().enumerate() {
        let row_offset = k * n_snapshots;
        for t in 0..n_snapshots {
            if t >= delay {
                output[t] += signals[row_offset + (t - delay)];
            }
            // t < delay → zero contribution (do nothing).
        }
    }

    Ok(output)
}

/// MVDR (Minimum Variance Distortionless Response) beamformer weights.
///
/// Computes the Capon / MVDR optimal weight vector:
///
/// ```text
/// w = R⁻¹ a / (aᵀ R⁻¹ a)
/// ```
///
/// The distortionless constraint `aᵀ w = 1` is satisfied by construction.
///
/// # Arguments
///
/// * `r_xx` — `[n_sensors × n_sensors]` sample covariance matrix (row-major,
///   real-valued).
/// * `a` — `[n_sensors]` real-valued steering vector (look-direction response).
/// * `n_sensors` — number of sensors; must equal `a.len()` and `√(r_xx.len())`.
///
/// # Returns
///
/// `Vec<f64>` of length `n_sensors` containing the MVDR weights `w`.
///
/// # Errors
///
/// * [`SignalError::InvalidParameter`] — `n_sensors == 0`, `R` is singular, or
///   `aᵀ R⁻¹ a ≈ 0`.
/// * [`SignalError::DimensionMismatch`] — slice lengths do not match `n_sensors`.
pub fn mvdr_weights(r_xx: &[f64], a: &[f64], n_sensors: usize) -> SignalResult<Vec<f64>> {
    if n_sensors == 0 {
        return Err(SignalError::InvalidParameter(
            "n_sensors must be > 0".to_string(),
        ));
    }

    // Validate slice dimensions.
    let expected_r = n_sensors * n_sensors;
    if r_xx.len() != expected_r {
        return Err(SignalError::DimensionMismatch {
            expected: format!("n_sensors² = {expected_r}"),
            got: format!("{}", r_xx.len()),
        });
    }
    if a.len() != n_sensors {
        return Err(SignalError::DimensionMismatch {
            expected: format!("n_sensors = {n_sensors}"),
            got: format!("{}", a.len()),
        });
    }

    // Invert R.
    let mut r_inv = vec![0.0_f64; expected_r];
    gauss_invert(r_xx, n_sensors, &mut r_inv)?;

    // Compute u = R⁻¹ a  (matrix-vector product).
    let mut u = vec![0.0_f64; n_sensors];
    for row in 0..n_sensors {
        let mut acc = 0.0_f64;
        for col in 0..n_sensors {
            acc += r_inv[row * n_sensors + col] * a[col];
        }
        u[row] = acc;
    }

    // Compute scalar denominator  d = aᵀ u = aᵀ R⁻¹ a.
    let denom: f64 = a.iter().zip(u.iter()).map(|(&ai, &ui)| ai * ui).sum();

    if denom.abs() < 1e-14 {
        return Err(SignalError::InvalidParameter(
            "singular covariance matrix".to_string(),
        ));
    }

    // w = u / d.
    let weights: Vec<f64> = u.iter().map(|&ui| ui / denom).collect();
    Ok(weights)
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers ----------------------------------------------------------------

    /// Build a flat identity matrix of size n × n.
    fn identity(n: usize) -> Vec<f64> {
        let mut m = vec![0.0_f64; n * n];
        for i in 0..n {
            m[i * n + i] = 1.0;
        }
        m
    }

    /// Build a unit steering vector  a = [1/√N, …, 1/√N].
    fn unit_steering(n: usize) -> Vec<f64> {
        let v = 1.0 / (n as f64).sqrt();
        vec![v; n]
    }

    // 1. DAS output length ---------------------------------------------------
    #[test]
    fn das_output_len() {
        let n_sensors = 4;
        let n_snapshots = 32;
        let signals = vec![0.0_f64; n_sensors * n_snapshots];
        let steering = vec![0.0_f64; n_sensors];
        let out = delay_and_sum(&signals, &steering, n_sensors, n_snapshots)
            .expect("das_output_len failed");
        assert_eq!(out.len(), n_snapshots);
    }

    // 2. DAS all-finite output -----------------------------------------------
    #[test]
    fn das_finite() {
        let n_sensors = 3;
        let n_snapshots = 16;
        // Non-trivial signals: ramp.
        let signals: Vec<f64> = (0..n_sensors * n_snapshots)
            .map(|i| i as f64 * 0.1)
            .collect();
        let steering = vec![0.0, 1.0, 2.0];
        let out =
            delay_and_sum(&signals, &steering, n_sensors, n_snapshots).expect("das_finite failed");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "DAS output contains non-finite values"
        );
    }

    // 3. MVDR weights shape --------------------------------------------------
    #[test]
    fn mvdr_weights_shape() {
        let n = 4;
        let r_xx = identity(n);
        let a = unit_steering(n);
        let w = mvdr_weights(&r_xx, &a, n).expect("mvdr_weights_shape failed");
        assert_eq!(w.len(), n);
    }

    // 4. MVDR weights all finite ---------------------------------------------
    #[test]
    fn mvdr_finite() {
        let n = 5;
        // Diagonal covariance with varying entries (well-conditioned).
        let mut r_xx = vec![0.0_f64; n * n];
        for i in 0..n {
            r_xx[i * n + i] = (i + 1) as f64;
        }
        let a = unit_steering(n);
        let w = mvdr_weights(&r_xx, &a, n).expect("mvdr_finite failed");
        assert!(
            w.iter().all(|v| v.is_finite()),
            "MVDR weights contain non-finite values"
        );
    }

    // 5. Distortionless constraint aᵀ w = 1 ---------------------------------
    #[test]
    fn steering_unit_response() {
        let n = 6;
        let r_xx = identity(n);
        let a = unit_steering(n);
        let w = mvdr_weights(&r_xx, &a, n).expect("steering_unit_response failed");

        // Compute aᵀ w.
        let dot: f64 = a.iter().zip(w.iter()).map(|(&ai, &wi)| ai * wi).sum();
        assert!(
            (dot - 1.0).abs() < 1e-10,
            "distortionless constraint violated: aᵀw = {dot}, expected 1.0"
        );
    }

    // 6. DAS with zero delays sums over all sensors at each time step --------
    #[test]
    fn das_zero_delay_sum() {
        let n_sensors = 3;
        let n_snapshots = 8;
        // signals[k][t] = (k+1) * (t+1) as f64
        let signals: Vec<f64> = (0..n_sensors)
            .flat_map(|k| (0..n_snapshots).map(move |t| ((k + 1) * (t + 1)) as f64))
            .collect();
        let steering = vec![0.0_f64; n_sensors];
        let out = delay_and_sum(&signals, &steering, n_sensors, n_snapshots)
            .expect("das_zero_delay_sum failed");

        // Expected: for each t, sum_{k=0}^{n_sensors-1} (k+1)*(t+1)
        // = (t+1) * sum_{k=0}^{n_sensors-1} (k+1)
        // = (t+1) * (1 + 2 + 3) = (t+1) * 6
        for (t, &out_val) in out.iter().enumerate().take(n_snapshots) {
            let expected = (t + 1) as f64 * 6.0;
            assert!(
                (out_val - expected).abs() < 1e-10,
                "das_zero_delay_sum mismatch at t={t}: got {out_val}, expected {expected}"
            );
        }
    }

    // 7. Singular covariance matrix → InvalidParameter -----------------------
    #[test]
    fn singular_r_error() {
        let n = 3;
        // All-zero matrix is maximally singular.
        let r_xx = vec![0.0_f64; n * n];
        let a = unit_steering(n);
        let result = mvdr_weights(&r_xx, &a, n);
        assert!(
            matches!(result, Err(SignalError::InvalidParameter(_))),
            "expected InvalidParameter for singular R, got: {result:?}"
        );
    }

    // 8. n_sensors = 0 → InvalidParameter -----------------------------------
    #[test]
    fn n_sensors_0_error() {
        let result = mvdr_weights(&[], &[], 0);
        assert!(
            matches!(result, Err(SignalError::InvalidParameter(_))),
            "expected InvalidParameter for n_sensors=0, got: {result:?}"
        );
    }

    // 9. DAS wrong signals length → DimensionMismatch ----------------------
    #[test]
    fn das_dimension_error() {
        let n_sensors = 4;
        let n_snapshots = 16;
        // Provide one fewer element than expected.
        let signals = vec![0.0_f64; n_sensors * n_snapshots - 1];
        let steering = vec![0.0_f64; n_sensors];
        let result = delay_and_sum(&signals, &steering, n_sensors, n_snapshots);
        assert!(
            matches!(result, Err(SignalError::DimensionMismatch { .. })),
            "expected DimensionMismatch for wrong signals length, got: {result:?}"
        );
    }
}
