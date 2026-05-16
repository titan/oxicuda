//! Dormand-Prince RK45 (DOPRI5) — 7-stage embedded RK4(5) with adaptive step.
//!
//! Butcher tableau from Hairer, Nørsett, Wanner. Uses I-controller (simpler than PI).

use crate::error::{NumericError, NumericResult};

const A21: f64 = 1.0 / 5.0;
const A31: f64 = 3.0 / 40.0;
const A32: f64 = 9.0 / 40.0;
const A41: f64 = 44.0 / 45.0;
const A42: f64 = -56.0 / 15.0;
const A43: f64 = 32.0 / 9.0;
const A51: f64 = 19372.0 / 6561.0;
const A52: f64 = -25360.0 / 2187.0;
const A53: f64 = 64448.0 / 6561.0;
const A54: f64 = -212.0 / 729.0;
const A61: f64 = 9017.0 / 3168.0;
const A62: f64 = -355.0 / 33.0;
const A63: f64 = 46732.0 / 5247.0;
const A64: f64 = 49.0 / 176.0;
const A65: f64 = -5103.0 / 18656.0;
const A71: f64 = 35.0 / 384.0;
const A73: f64 = 500.0 / 1113.0;
const A74: f64 = 125.0 / 192.0;
const A75: f64 = -2187.0 / 6784.0;
const A76: f64 = 11.0 / 84.0;

const C2: f64 = 1.0 / 5.0;
const C3: f64 = 3.0 / 10.0;
const C4: f64 = 4.0 / 5.0;
const C5: f64 = 8.0 / 9.0;

// 5th-order weights = A7*
const B1: f64 = A71;
const B3: f64 = A73;
const B4: f64 = A74;
const B5: f64 = A75;
const B6: f64 = A76;
const B7: f64 = 0.0;
// Error coefficients (e = b - b^)
const E1: f64 = 71.0 / 57600.0;
const E3: f64 = -71.0 / 16695.0;
const E4: f64 = 71.0 / 1920.0;
const E5: f64 = -17253.0 / 339200.0;
const E6: f64 = 22.0 / 525.0;
const E7: f64 = -1.0 / 40.0;

/// DOPRI5 adaptive integrator.
pub fn dopri5<F>(
    f: F,
    t0: f64,
    tf: f64,
    y0: &[f64],
    h_init: f64,
    rtol: f64,
    atol: f64,
    max_steps: usize,
) -> NumericResult<(Vec<f64>, Vec<Vec<f64>>)>
where
    F: Fn(f64, &[f64]) -> NumericResult<Vec<f64>>,
{
    if !h_init.is_finite() || h_init <= 0.0 {
        return Err(NumericError::InvalidStepSize { step: h_init });
    }
    if tf <= t0 {
        return Err(NumericError::InvalidParameter("tf must be > t0".into()));
    }
    let dim = y0.len();
    let mut times = vec![t0];
    let mut ys: Vec<Vec<f64>> = vec![y0.to_vec()];
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut h = h_init;
    let safety = 0.9_f64;
    let max_grow = 5.0_f64;
    let min_shrink = 0.1_f64;
    let mut ytmp = vec![0.0_f64; dim];
    let mut yerr = vec![0.0_f64; dim];
    let mut new_y = vec![0.0_f64; dim];
    for _ in 0..max_steps {
        if t + h > tf {
            h = tf - t;
        }
        if h <= 0.0 {
            break;
        }
        let k1 = f(t, &y)?;
        for i in 0..dim {
            ytmp[i] = y[i] + h * A21 * k1[i];
        }
        let k2 = f(t + C2 * h, &ytmp)?;
        for i in 0..dim {
            ytmp[i] = y[i] + h * (A31 * k1[i] + A32 * k2[i]);
        }
        let k3 = f(t + C3 * h, &ytmp)?;
        for i in 0..dim {
            ytmp[i] = y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]);
        }
        let k4 = f(t + C4 * h, &ytmp)?;
        for i in 0..dim {
            ytmp[i] = y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]);
        }
        let k5 = f(t + C5 * h, &ytmp)?;
        for i in 0..dim {
            ytmp[i] =
                y[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]);
        }
        let k6 = f(t + h, &ytmp)?;
        for i in 0..dim {
            new_y[i] = y[i] + h * (B1 * k1[i] + B3 * k3[i] + B4 * k4[i] + B5 * k5[i] + B6 * k6[i]);
        }
        let k7 = f(t + h, &new_y)?;
        for i in 0..dim {
            yerr[i] =
                h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
        }
        let _ = B7;
        // norm
        let mut err_norm = 0.0_f64;
        for i in 0..dim {
            let sc = atol + rtol * y[i].abs().max(new_y[i].abs());
            err_norm += (yerr[i] / sc).powi(2);
        }
        err_norm = (err_norm / dim as f64).sqrt();
        if err_norm <= 1.0 {
            // accept
            t += h;
            y[..dim].copy_from_slice(&new_y[..dim]);
            times.push(t);
            ys.push(y.clone());
            if (t - tf).abs() < 1.0e-14 {
                return Ok((times, ys));
            }
        }
        // step size update
        let factor = if err_norm == 0.0 {
            max_grow
        } else {
            (safety * (1.0 / err_norm).powf(1.0 / 5.0))
                .max(min_shrink)
                .min(max_grow)
        };
        h *= factor;
        if h < 1.0e-12 {
            return Err(NumericError::NumericalInstability(
                "DOPRI5 step shrunk below 1e-12".into(),
            ));
        }
    }
    Err(NumericError::NotConverged {
        iter: max_steps,
        residual: tf - t,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dopri5_exponential_decay() {
        let f = |_t: f64, y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![-y[0]]) };
        let (_t, ys) = dopri5(f, 0.0, 1.0, &[1.0], 0.1, 1.0e-8, 1.0e-10, 10_000).expect("ok");
        let last = ys.last().expect("non-empty");
        assert!((last[0] - (-1.0_f64).exp()).abs() < 1.0e-6);
    }

    #[test]
    fn dopri5_oscillator_conserves_energy() {
        // y'' + y = 0 → energy = y² + y'²  conserved.
        let f = |_t: f64, y: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![y[1], -y[0]]) };
        let (_t, ys) =
            dopri5(f, 0.0, 10.0, &[1.0, 0.0], 0.1, 1.0e-9, 1.0e-12, 100_000).expect("ok");
        let initial_energy = 1.0_f64;
        for yvec in ys.iter() {
            let e = yvec[0] * yvec[0] + yvec[1] * yvec[1];
            assert!((e - initial_energy).abs() < 1.0e-4);
        }
    }
}
