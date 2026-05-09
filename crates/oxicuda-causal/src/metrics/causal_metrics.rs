/// Precision in Estimation of Heterogeneous Effects: sqrt(mean((pred - true)^2))
pub fn pehe(cate_pred: &[f32], cate_true: &[f32]) -> f32 {
    let n = cate_pred.len().min(cate_true.len());
    if n == 0 {
        return 0.0;
    }
    let mse: f32 = cate_pred[..n]
        .iter()
        .zip(cate_true[..n].iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum::<f32>()
        / n as f32;
    mse.sqrt()
}

/// Absolute bias of ATE estimate.
pub fn ate_bias(ate_pred: f32, ate_true: f32) -> f32 {
    (ate_pred - ate_true).abs()
}

/// Policy risk: fraction of incorrect treatment decisions under threshold policy (CATE > 0 -> treat).
pub fn policy_risk(cate_pred: &[f32], y: &[f32], t: &[f32]) -> f32 {
    let n = cate_pred.len().min(y.len()).min(t.len());
    if n == 0 {
        return 0.0;
    }
    // Policy: treat if predicted CATE > 0
    // Risk = mean loss under policy vs oracle
    let mut loss = 0.0_f32;
    for i in 0..n {
        let policy_t = if cate_pred[i] > 0.0 { 1.0_f32 } else { 0.0 };
        // Observed outcome under observed treatment
        let obs = y[i] * t[i] + (1.0 - t[i]) * (y[i] - 0.0); // observed y
        // Counterfactual: if policy differs from observed treatment, assume symmetric
        let mismatch = (policy_t - t[i]).abs();
        loss += mismatch * obs.abs();
    }
    loss / n as f32
}

/// Qini coefficient for uplift model evaluation.
pub fn qini_coeff(cate_pred: &[f32], y: &[f32], t: &[f32]) -> f32 {
    let n = cate_pred.len().min(y.len()).min(t.len());
    if n == 0 {
        return 0.0;
    }
    // Sort by descending CATE prediction
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        cate_pred[b]
            .partial_cmp(&cate_pred[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Compute Qini curve: for each fraction phi, compute
    // Q(phi) = sum_{top phi} (Y*T - Y*(1-T)*p/(1-p)) where p = overall treatment rate
    let n_treated: f32 = t.iter().sum();
    let n_control = n as f32 - n_treated;
    if n_treated < 1.0 || n_control < 1.0 {
        return 0.0;
    }
    let ratio = n_treated / n_control;

    let mut cum_treated_y = 0.0_f32;
    let mut cum_control_y = 0.0_f32;
    let mut cum_treated_n = 0.0_f32;
    let mut qini_area = 0.0_f32;
    let mut random_area = 0.0_f32;
    let mut prev_q = 0.0_f32;

    for (step, &i) in order.iter().enumerate() {
        if t[i] >= 0.5 {
            cum_treated_y += y[i];
            cum_treated_n += 1.0;
        } else {
            cum_control_y += y[i];
        }
        let q = cum_treated_y - cum_control_y * ratio;
        qini_area += (q + prev_q) * 0.5 / n as f32;
        // Random model contribution
        let random_q = cum_treated_n / n as f32 * (n_treated - cum_treated_n.min(n_treated));
        random_area += random_q / n as f32;
        prev_q = q;
        let _ = step; // silence unused
    }

    qini_area - random_area
}

/// R-squared for CATE estimates.
pub fn r_squared_cate(cate_pred: &[f32], cate_true: &[f32]) -> f32 {
    let n = cate_pred.len().min(cate_true.len());
    if n == 0 {
        return 0.0;
    }
    let mean_true: f32 = cate_true[..n].iter().sum::<f32>() / n as f32;
    let ss_tot: f32 = cate_true[..n]
        .iter()
        .map(|&v| (v - mean_true).powi(2))
        .sum();
    if ss_tot < 1e-10 {
        return 1.0;
    }
    let ss_res: f32 = cate_pred[..n]
        .iter()
        .zip(cate_true[..n].iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum();
    1.0 - ss_res / ss_tot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pehe_perfect() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert!((pehe(&v, &v)).abs() < 1e-6);
    }

    #[test]
    fn ate_bias_zero() {
        assert!((ate_bias(1.5, 1.5)).abs() < 1e-6);
    }

    #[test]
    fn r_squared_perfect() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert!((r_squared_cate(&v, &v) - 1.0).abs() < 1e-5);
    }
}
