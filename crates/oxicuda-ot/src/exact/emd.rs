//! Earth Mover's Distance (EMD) — exact OT cost.
//!
//! In one dimension EMD reduces to integrating the absolute difference of
//! cumulative distribution functions, `∫ |F_a − F_b| dt`, which can be
//! evaluated in `O((m+n) log(m+n))` with a sort + merge sweep. In higher
//! dimensions we delegate to `network_simplex`.

use crate::error::{OtError, OtResult};
use crate::exact::network_simplex::{NsConfig, network_simplex};

/// Validate balanced marginals and matching lengths for 1D EMD.
fn validate_1d(x: &[f32], y: &[f32], a: &[f32], b: &[f32]) -> OtResult<()> {
    if x.is_empty() || y.is_empty() {
        return Err(OtError::EmptyInput);
    }
    if x.len() != a.len() {
        return Err(OtError::IncompatibleLength {
            a: x.len(),
            b: a.len(),
        });
    }
    if y.len() != b.len() {
        return Err(OtError::IncompatibleLength {
            a: y.len(),
            b: b.len(),
        });
    }
    let mut sa = 0.0_f32;
    for &ai in a {
        if ai < 0.0 || !ai.is_finite() {
            return Err(OtError::NegativeWeight);
        }
        sa += ai;
    }
    let mut sb = 0.0_f32;
    for &bj in b {
        if bj < 0.0 || !bj.is_finite() {
            return Err(OtError::NegativeWeight);
        }
        sb += bj;
    }
    if (sa - sb).abs() > 1e-4 {
        return Err(OtError::MassImbalance {
            sum_a: sa,
            sum_b: sb,
        });
    }
    Ok(())
}

/// EMD between two 1D weighted samples.
///
/// `x[i]` carries mass `a[i]`; `y[j]` carries mass `b[j]`. Returns
/// `∫ |F_a(t) − F_b(t)| dt`, which equals the W1 distance with cost
/// `|x − y|`.
pub fn emd_1d(x: &[f32], y: &[f32], a: &[f32], b: &[f32]) -> OtResult<f32> {
    validate_1d(x, y, a, b)?;

    // Build sorted (position, signed mass) merged events.
    // sign = +1 from a, sign = −1 from b.
    let mut events: Vec<(f32, f32)> = Vec::with_capacity(x.len() + y.len());
    for (xi, ai) in x.iter().zip(a.iter()) {
        events.push((*xi, *ai));
    }
    for (yj, bj) in y.iter().zip(b.iter()) {
        events.push((*yj, -*bj));
    }
    events.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap_or(std::cmp::Ordering::Equal));
    if events.is_empty() {
        return Ok(0.0);
    }

    let mut total = 0.0_f32;
    let mut net = 0.0_f32; // cumulative (F_a − F_b)
    let mut prev_t = events[0].0;
    // Aggregate mass at each unique position before integrating to next.
    let mut idx = 0;
    while idx < events.len() {
        let mut k = idx;
        while k < events.len() && events[k].0 == events[idx].0 {
            net += events[k].1;
            k += 1;
        }
        if k < events.len() {
            let next_t = events[k].0;
            total += net.abs() * (next_t - prev_t);
            prev_t = next_t;
        }
        idx = k;
    }
    Ok(total)
}

/// Generic exact EMD via the network-simplex algorithm.
pub fn emd(c: &[f32], a: &[f32], b: &[f32], m: usize, n: usize, cfg: &NsConfig) -> OtResult<f32> {
    let res = network_simplex(c, a, b, m, n, cfg)?;
    Ok(res.cost)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translation_invariance_1d() {
        let x = vec![0.0_f32, 1.0, 2.0];
        let a = vec![1.0_f32 / 3.0; 3];
        let t = 1.5_f32;
        let y: Vec<f32> = x.iter().map(|v| v + t).collect();
        let b = a.clone();
        let d = emd_1d(&x, &y, &a, &b).expect("ok");
        assert!((d - t).abs() < 1e-3, "d={} expected {}", d, t);
    }

    #[test]
    fn zero_on_equal_distributions_1d() {
        let x = vec![-1.0_f32, 0.5, 2.0];
        let a = vec![0.2_f32, 0.5, 0.3];
        let d = emd_1d(&x, &x, &a, &a).expect("ok");
        assert!(d.abs() < 1e-5);
    }

    #[test]
    fn agrees_with_simplex_1d() {
        let x = vec![0.0_f32, 1.0];
        let y = vec![0.5_f32, 1.5];
        let a = vec![0.5_f32, 0.5];
        let b = vec![0.5_f32, 0.5];
        let d1 = emd_1d(&x, &y, &a, &b).expect("ok");
        let mut c = vec![0.0_f32; 4];
        for i in 0..2 {
            for j in 0..2 {
                c[i * 2 + j] = (x[i] - y[j]).abs();
            }
        }
        let d2 = emd(&c, &a, &b, 2, 2, &NsConfig::default()).expect("ok");
        assert!((d1 - d2).abs() < 1e-4, "1d={} simplex={}", d1, d2);
    }

    #[test]
    fn mass_imbalance_rejected() {
        let x = vec![0.0_f32];
        let a = vec![1.0_f32];
        let y = vec![0.0_f32];
        let b = vec![0.5_f32];
        let res = emd_1d(&x, &y, &a, &b);
        assert!(matches!(res, Err(OtError::MassImbalance { .. })));
    }

    #[test]
    fn empty_rejected() {
        let res = emd_1d(&[], &[], &[], &[]);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }
}
