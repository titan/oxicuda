//! Divergence measures between probability distributions.

/// KL divergence KL(p ‖ q) = Σ p_i · ln(p_i / (q_i + ε)).
///
/// Terms where p_i ≤ 0 are skipped per convention.
#[must_use]
pub fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
    const EPS: f32 = 1e-10;
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            if pi <= 0.0 {
                0.0
            } else {
                pi * (pi / (qi + EPS)).ln()
            }
        })
        .sum()
}

/// Jensen-Shannon divergence: 0.5 · KL(p ‖ m) + 0.5 · KL(q ‖ m), m = (p + q) / 2.
///
/// Always finite and symmetric; range [0, ln 2].
#[must_use]
pub fn js_divergence(p: &[f32], q: &[f32]) -> f32 {
    let m: Vec<f32> = p
        .iter()
        .zip(q.iter())
        .map(|(&a, &b)| (a + b) * 0.5)
        .collect();
    0.5 * kl_divergence(p, &m) + 0.5 * kl_divergence(q, &m)
}

/// 1-D Wasserstein-1 (Earth Mover's) distance.
///
/// Sort both arrays; W₁ = mean |s_sorted`[i]` − t_sorted`[i]`|.
#[must_use]
pub fn wasserstein_1d(s: &[f32], t: &[f32]) -> f32 {
    if s.is_empty() || t.is_empty() {
        return 0.0;
    }
    let mut s_sorted = s.to_vec();
    let mut t_sorted = t.to_vec();
    s_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    t_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s_sorted.len().min(t_sorted.len()) as f32;
    s_sorted
        .iter()
        .zip(t_sorted.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum::<f32>()
        / n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kl_identical_zero() {
        let p = vec![0.3_f32, 0.5, 0.2];
        assert!(kl_divergence(&p, &p) < 1e-5);
    }

    #[test]
    fn js_nonneg_symmetric() {
        let p = vec![0.3_f32, 0.5, 0.2];
        let q = vec![0.2_f32, 0.6, 0.2];
        let js = js_divergence(&p, &q);
        assert!(js >= 0.0 && js.is_finite());
        assert!((js_divergence(&p, &q) - js_divergence(&q, &p)).abs() < 1e-6);
    }

    #[test]
    fn wasserstein_identical_zero() {
        let v = vec![0.1_f32, 0.5, 0.4];
        assert!(wasserstein_1d(&v, &v) < 1e-10);
    }
}
