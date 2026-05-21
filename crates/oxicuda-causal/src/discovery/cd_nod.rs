//! CD-NOD — Causal Discovery from Heterogeneous / Nonstationary Data.
//!
//! Reference: Huang, Zhang, Zhang, Ramsey, Sanchez-Romero, Glymour, Schölkopf
//! (2020) "Causal Discovery from Heterogeneous/Nonstationary Data", JMLR 21(89).
//!
//! # Core idea
//!
//! Data is collected across `D` domains (or time segments). A **surrogate
//! variable `C`** indexes the domain/time. Constraint-based skeleton discovery
//! is run over the AUGMENTED variable set `{X_1,…,X_p, C}`. Variables that stay
//! adjacent to `C` have **changing causal mechanisms** (nonstationary /
//! heterogeneous). `C` is exogenous (the system index cannot be caused by any
//! `X`), so every edge incident to `C` is oriented `C → X_i`.
//!
//! # Pipeline
//!
//! 1. Build the surrogate column `C[s] = label[s] / max(1, D-1) ∈ [0,1]`.
//! 2. Augment the data matrix with `C` as the last column.
//! 3. PC-style skeleton search using a regression-residual Fisher-Z conditional
//!    independence test (no explicit matrix inverse needed for the partial
//!    correlation — we regress out the conditioning set via ridge-stabilised
//!    OLS and correlate the residuals).
//! 4. `changing_vars` = the `X` nodes still adjacent to `C`.
//! 5. Orientation into a CPDAG: force `C → X_i`, then v-structures, then Meek
//!    rules R1–R3 to a fixpoint.

use crate::error::{CausalError, CausalResult};
use std::collections::HashMap;

/// Configuration for [`CdNod::run`].
#[derive(Debug, Clone)]
pub struct CdNodConfig {
    /// Number of observed variables `p` (excludes the surrogate `C`).
    pub n_vars: usize,
    /// Number of domains / time segments `D` (must be ≥ 2).
    pub n_domains: usize,
    /// Conditional-independence significance level (e.g. `0.05`).
    pub alpha: f64,
    /// Maximum conditioning-set size in the skeleton search.
    pub max_cond_set: usize,
}

/// Result of running CD-NOD.
#[derive(Debug, Clone)]
pub struct CdNodResult {
    /// Number of observed variables `p`.
    pub n_vars: usize,
    /// Symmetric `(p+1)×(p+1)` adjacency of the augmented skeleton.
    /// Index `n_vars` is the surrogate variable `C`.
    pub skeleton: Vec<Vec<bool>>,
    /// `X` indices (`< n_vars`) that remain adjacent to `C` — the variables
    /// whose causal mechanism changes across domains.
    pub changing_vars: Vec<usize>,
    /// CPDAG of the augmented graph: `oriented[i][j] == 1` means `i → j`.
    /// An undirected edge has both directions `0` while `skeleton` is `true`.
    pub oriented: Vec<Vec<i8>>,
    /// Index of the surrogate variable `C` (always `== n_vars`).
    pub c_index: usize,
}

/// CD-NOD entry point.
pub struct CdNod;

/// Standard-normal CDF `Φ(x) = 0.5·(1 + erf(x/√2))`.
///
/// `erf` is the Abramowitz & Stegun 7.1.26 rational approximation
/// (max absolute error ≈ 1.5e-7), which is ample for a CI p-value.
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function via Abramowitz & Stegun 7.1.26.
fn erf(x: f64) -> f64 {
    // erf is odd; compute on |x| and restore the sign.
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let y = 1.0 - poly * (-ax * ax).exp();
    sign * y
}

/// Sample mean of a slice (returns `0.0` for the empty slice).
fn mean_f64(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

/// Pearson correlation of two equal-length residual vectors, clamped to
/// `(-0.999999, 0.999999)` so the Fisher transform stays finite.
fn pearson_clamped(rx: &[f64], ry: &[f64]) -> f64 {
    let n = rx.len().min(ry.len());
    if n < 2 {
        return 0.0;
    }
    let mx = mean_f64(&rx[..n]);
    let my = mean_f64(&ry[..n]);
    let mut num = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for k in 0..n {
        let dx = rx[k] - mx;
        let dy = ry[k] - my;
        num += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx < 1e-12 || syy < 1e-12 {
        return 0.0;
    }
    let rho = num / (sxx.sqrt() * syy.sqrt());
    rho.clamp(-0.999_999, 0.999_999)
}

/// Regress `target` (length `n`) on the columns of `design` (row-major,
/// `n × d`) by ridge-stabilised normal equations (`λ = 1e-6`) and return the
/// residual vector. With `d == 0` the residuals are `target` minus its mean.
fn regress_residuals_f64(design: &[f64], target: &[f64], n: usize, d: usize) -> Vec<f64> {
    if d == 0 {
        let m = mean_f64(target);
        return target.iter().map(|&v| v - m).collect();
    }
    // Augment design with an intercept column so the residual fit is unbiased.
    let dd = d + 1;
    // Normal equations A = Xᵀ X (with intercept), b = Xᵀ y.
    let mut a = vec![0.0_f64; dd * dd];
    let mut b = vec![0.0_f64; dd];
    for row in 0..n {
        // Local design row: [features..., 1].
        for i in 0..dd {
            let xi = if i < d { design[row * d + i] } else { 1.0 };
            for j in 0..dd {
                let xj = if j < d { design[row * d + j] } else { 1.0 };
                a[i * dd + j] += xi * xj;
            }
            b[i] += xi * target[row];
        }
    }
    // Ridge on the slope terms only (do not penalise the intercept).
    for i in 0..d {
        a[i * dd + i] += 1e-6;
    }
    let beta = match solve_linear_system(&mut a, &b, dd) {
        Some(beta) => beta,
        None => {
            // Singular even with ridge: fall back to centred target.
            let m = mean_f64(target);
            return target.iter().map(|&v| v - m).collect();
        }
    };
    let mut residuals = vec![0.0_f64; n];
    for row in 0..n {
        let mut pred = beta[d]; // intercept
        for j in 0..d {
            pred += design[row * d + j] * beta[j];
        }
        residuals[row] = target[row] - pred;
    }
    residuals
}

/// Solve `a · x = b` (`a` is `m×m` row-major) by Gauss-Jordan elimination with
/// partial pivoting. Returns `None` if the matrix is numerically singular.
/// `a` is consumed (mutated) as scratch space.
fn solve_linear_system(a: &mut [f64], b: &[f64], m: usize) -> Option<Vec<f64>> {
    let mut x = b.to_vec();
    for col in 0..m {
        // Partial pivot.
        let mut pivot_row = col;
        let mut pivot_val = a[col * m + col].abs();
        for r in (col + 1)..m {
            let v = a[r * m + col].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = r;
            }
        }
        if pivot_val < 1e-12 {
            return None;
        }
        if pivot_row != col {
            for c in 0..m {
                a.swap(col * m + c, pivot_row * m + c);
            }
            x.swap(col, pivot_row);
        }
        let diag = a[col * m + col];
        for c in 0..m {
            a[col * m + c] /= diag;
        }
        x[col] /= diag;
        for r in 0..m {
            if r == col {
                continue;
            }
            let factor = a[r * m + col];
            if factor == 0.0 {
                continue;
            }
            for c in 0..m {
                a[r * m + c] -= factor * a[col * m + c];
            }
            x[r] -= factor * x[col];
        }
    }
    Some(x)
}

/// Enumerate all size-`k` subsets of `items` (combinations, order preserved).
fn subsets_of_size(items: &[usize], k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![Vec::new()];
    }
    if k > items.len() {
        return Vec::new();
    }
    if k == items.len() {
        return vec![items.to_vec()];
    }
    let mut result = Vec::new();
    for i in 0..=(items.len() - k) {
        let head = items[i];
        for mut tail in subsets_of_size(&items[(i + 1)..], k - 1) {
            tail.insert(0, head);
            result.push(tail);
        }
    }
    result
}

impl CdNod {
    /// Build the normalized surrogate column `C[s] = label[s] / max(1, D-1)`.
    ///
    /// # Errors
    /// Returns [`CausalError::InvalidParameter`] if `n_domains < 2` or any label
    /// is `>= n_domains`.
    pub fn build_surrogate(domain_labels: &[usize], n_domains: usize) -> CausalResult<Vec<f32>> {
        if n_domains < 2 {
            return Err(CausalError::InvalidParameter {
                reason: format!("n_domains must be >= 2, got {n_domains}"),
            });
        }
        let denom = (n_domains - 1).max(1) as f32;
        let mut surrogate = Vec::with_capacity(domain_labels.len());
        for (s, &label) in domain_labels.iter().enumerate() {
            if label >= n_domains {
                return Err(CausalError::InvalidParameter {
                    reason: format!("domain label {label} at sample {s} >= n_domains {n_domains}"),
                });
            }
            surrogate.push(label as f32 / denom);
        }
        Ok(surrogate)
    }

    /// Fisher-Z partial-correlation conditional-independence test.
    ///
    /// Tests `X_i ⫫ X_j | S` on the augmented matrix `data_aug` (row-major,
    /// `n_samples × n_total`). Regresses columns `i` and `j` on the conditioning
    /// columns `cond` (ridge-stabilised OLS), correlates the residuals to obtain
    /// `ρ`, then
    ///
    /// ```text
    /// z         = 0.5·ln((1+ρ)/(1−ρ))
    /// statistic = |z| · sqrt(n − |S| − 3)
    /// p_value   = 2·(1 − Φ(statistic))
    /// ```
    ///
    /// Returns `(statistic, p_value)`; the pair is **independent** when
    /// `p_value > alpha`. If `n − |S| − 3 ≤ 0` the data is insufficient and we
    /// report independence as `(0.0, 1.0)`.
    ///
    /// # Errors
    /// Returns [`CausalError::DimensionMismatch`] if `data_aug.len() !=
    /// n_samples * n_total`, and [`CausalError::InvalidParameter`] if an index is
    /// out of range.
    pub fn fisher_z_test(
        data_aug: &[f32],
        n_samples: usize,
        n_total: usize,
        i: usize,
        j: usize,
        cond: &[usize],
    ) -> CausalResult<(f64, f64)> {
        if data_aug.len() != n_samples * n_total {
            return Err(CausalError::DimensionMismatch {
                expected: n_samples * n_total,
                got: data_aug.len(),
            });
        }
        if i >= n_total || j >= n_total {
            return Err(CausalError::InvalidParameter {
                reason: format!("variable index out of range (i={i}, j={j}, n_total={n_total})"),
            });
        }
        for &c in cond {
            if c >= n_total {
                return Err(CausalError::InvalidParameter {
                    reason: format!("conditioning index {c} out of range (n_total={n_total})"),
                });
            }
        }

        let cond_size = cond.len();
        // Insufficient degrees of freedom → treat as independent.
        if n_samples as i64 - cond_size as i64 - 3 <= 0 {
            return Ok((0.0, 1.0));
        }

        let col_i: Vec<f64> = (0..n_samples)
            .map(|s| data_aug[s * n_total + i] as f64)
            .collect();
        let col_j: Vec<f64> = (0..n_samples)
            .map(|s| data_aug[s * n_total + j] as f64)
            .collect();

        let rho = if cond_size == 0 {
            // No conditioning: centre both columns and correlate directly.
            let ci = {
                let m = mean_f64(&col_i);
                col_i.iter().map(|&v| v - m).collect::<Vec<_>>()
            };
            let cj = {
                let m = mean_f64(&col_j);
                col_j.iter().map(|&v| v - m).collect::<Vec<_>>()
            };
            pearson_clamped(&ci, &cj)
        } else {
            // Build the conditioning design (row-major, n × cond_size).
            let mut design = vec![0.0_f64; n_samples * cond_size];
            for (c_idx, &c) in cond.iter().enumerate() {
                for s in 0..n_samples {
                    design[s * cond_size + c_idx] = data_aug[s * n_total + c] as f64;
                }
            }
            let res_i = regress_residuals_f64(&design, &col_i, n_samples, cond_size);
            let res_j = regress_residuals_f64(&design, &col_j, n_samples, cond_size);
            pearson_clamped(&res_i, &res_j)
        };

        let z = 0.5 * ((1.0 + rho) / (1.0 - rho)).ln();
        let df = n_samples as f64 - cond_size as f64 - 3.0;
        let statistic = z.abs() * df.sqrt();
        // p = 2·(1 − Φ(stat)); clamp into [0, 1] against rounding noise.
        let p_value = (2.0 * (1.0 - normal_cdf(statistic))).clamp(0.0, 1.0);
        Ok((statistic, p_value))
    }

    /// Run CD-NOD on `n_samples × n_vars` row-major data with per-sample
    /// `domain_labels`.
    ///
    /// # Errors
    /// Returns [`CausalError::InvalidParameter`] for `n_domains < 2`,
    /// `n_vars < 1`, `n_samples < 4`, label-length mismatch, or out-of-range
    /// labels; and [`CausalError::DimensionMismatch`] if
    /// `data.len() != n_samples * n_vars`.
    pub fn run(
        data: &[f32],
        n_samples: usize,
        domain_labels: &[usize],
        cfg: &CdNodConfig,
    ) -> CausalResult<CdNodResult> {
        // ---- Validation -----------------------------------------------------
        if cfg.n_domains < 2 {
            return Err(CausalError::InvalidParameter {
                reason: format!("n_domains must be >= 2, got {}", cfg.n_domains),
            });
        }
        if cfg.n_vars < 1 {
            return Err(CausalError::InvalidParameter {
                reason: "n_vars must be >= 1".to_string(),
            });
        }
        if n_samples < 4 {
            return Err(CausalError::InvalidParameter {
                reason: format!("n_samples must be >= 4, got {n_samples}"),
            });
        }
        if domain_labels.len() != n_samples {
            return Err(CausalError::DimensionMismatch {
                expected: n_samples,
                got: domain_labels.len(),
            });
        }
        if data.len() != n_samples * cfg.n_vars {
            return Err(CausalError::DimensionMismatch {
                expected: n_samples * cfg.n_vars,
                got: data.len(),
            });
        }
        for (s, &label) in domain_labels.iter().enumerate() {
            if label >= cfg.n_domains {
                return Err(CausalError::InvalidParameter {
                    reason: format!(
                        "domain label {label} at sample {s} >= n_domains {}",
                        cfg.n_domains
                    ),
                });
            }
        }

        let p = cfg.n_vars;
        let c_index = p; // surrogate column index
        let n_total = p + 1;

        // ---- Augmented matrix (last column = surrogate C) -------------------
        let surrogate = Self::build_surrogate(domain_labels, cfg.n_domains)?;
        let mut data_aug = vec![0.0_f32; n_samples * n_total];
        for s in 0..n_samples {
            for v in 0..p {
                data_aug[s * n_total + v] = data[s * p + v];
            }
            data_aug[s * n_total + c_index] = surrogate[s];
        }

        // ---- PC-style skeleton ---------------------------------------------
        let mut adj: Vec<Vec<bool>> = vec![vec![true; n_total]; n_total];
        for (v, row) in adj.iter_mut().enumerate() {
            row[v] = false;
        }
        let mut sep_sets: HashMap<(usize, usize), Vec<usize>> = HashMap::new();

        let max_l = cfg.max_cond_set.min(n_total.saturating_sub(2));
        for l in 0..=max_l {
            // Stop early once no node has enough neighbours for size-l sets.
            let mut any_eligible = false;

            // Snapshot of remaining ordered adjacent pairs for this level.
            let mut pairs: Vec<(usize, usize)> = Vec::new();
            for (i, adj_row) in adj.iter().enumerate() {
                for (j, &connected) in adj_row.iter().enumerate() {
                    if i != j && connected {
                        pairs.push((i, j));
                    }
                }
            }

            let mut removals: Vec<(usize, usize, Vec<usize>)> = Vec::new();
            for (i, j) in pairs {
                if !adj[i][j] {
                    continue; // possibly removed earlier this level
                }
                let neighbors: Vec<usize> = (0..n_total)
                    .filter(|&v| v != i && v != j && adj[i][v])
                    .collect();
                if neighbors.len() < l {
                    continue;
                }
                any_eligible = true;
                for subset in subsets_of_size(&neighbors, l) {
                    let (_stat, p_value) =
                        Self::fisher_z_test(&data_aug, n_samples, n_total, i, j, &subset)?;
                    if p_value > cfg.alpha {
                        removals.push((i, j, subset));
                        break;
                    }
                }
            }

            for (i, j, sep) in removals {
                if adj[i][j] {
                    adj[i][j] = false;
                    adj[j][i] = false;
                    let key = if i < j { (i, j) } else { (j, i) };
                    sep_sets.insert(key, sep);
                }
            }

            if !any_eligible {
                break;
            }
        }

        // Symmetric skeleton (already symmetric, but build the public matrix).
        let skeleton: Vec<Vec<bool>> = adj.clone();

        // ---- Changing variables = X nodes adjacent to C ---------------------
        let mut changing_vars: Vec<usize> = (0..p).filter(|&v| adj[v][c_index]).collect();
        changing_vars.sort_unstable();

        // ---- Orientation → CPDAG (i8 matrix) --------------------------------
        let mut oriented: Vec<Vec<i8>> = vec![vec![0_i8; n_total]; n_total];
        let sep_of = |a: usize, b: usize| -> Vec<usize> {
            let key = if a < b { (a, b) } else { (b, a) };
            sep_sets.get(&key).cloned().unwrap_or_default()
        };

        // (1) C is exogenous: orient C → X_i for every changing var.
        for &v in &changing_vars {
            oriented[c_index][v] = 1;
            oriented[v][c_index] = 0;
        }

        // (2) v-structures: unshielded triple i—k—j with i,j non-adjacent and
        //     k ∉ sepset(i,j) → i→k and j→k.
        for k in 0..n_total {
            for i in 0..n_total {
                if i == k || !adj[i][k] {
                    continue;
                }
                for j in (i + 1)..n_total {
                    if j == k || !adj[j][k] {
                        continue;
                    }
                    if adj[i][j] {
                        continue; // shielded
                    }
                    if !sep_of(i, j).contains(&k) {
                        // Orient as colliders, but never give C a parent.
                        if i != c_index {
                            oriented[i][k] = 1;
                            oriented[k][i] = 0;
                        }
                        if j != c_index {
                            oriented[j][k] = 1;
                            oriented[k][j] = 0;
                        }
                    }
                }
            }
        }

        // Helper closures over the matrices use explicit functions to avoid
        // borrow conflicts; we operate on `oriented`/`adj` directly below.

        // (3) Meek rules R1–R3 to fixpoint.
        let is_directed = |o: &[Vec<i8>], a: usize, b: usize| o[a][b] == 1 && o[b][a] == 0;
        let is_undirected =
            |o: &[Vec<i8>], a: usize, b: usize| adj[a][b] && o[a][b] == 0 && o[b][a] == 0;

        let mut changed = true;
        while changed {
            changed = false;

            // R1: i → k and k — j with i,j non-adjacent ⇒ k → j.
            for (i, adj_row) in adj.iter().enumerate() {
                for k in 0..n_total {
                    if !is_directed(&oriented, i, k) {
                        continue;
                    }
                    for j in 0..n_total {
                        if j == i || j == k {
                            continue;
                        }
                        if is_undirected(&oriented, k, j) && !adj_row[j] && j != c_index {
                            oriented[k][j] = 1;
                            oriented[j][k] = 0;
                            changed = true;
                        }
                    }
                }
            }

            // R2: i → k → j and i — j ⇒ i → j.
            for i in 0..n_total {
                for j in 0..n_total {
                    if i == j {
                        continue;
                    }
                    if !is_undirected(&oriented, i, j) {
                        continue;
                    }
                    let mut orient = false;
                    for k in 0..n_total {
                        if k == i || k == j {
                            continue;
                        }
                        if is_directed(&oriented, i, k) && is_directed(&oriented, k, j) {
                            orient = true;
                            break;
                        }
                    }
                    if orient && j != c_index {
                        oriented[i][j] = 1;
                        oriented[j][i] = 0;
                        changed = true;
                    }
                }
            }

            // R3: i — j, i — k, i — l, k → j, l → j, k and l non-adjacent ⇒ i → j.
            for i in 0..n_total {
                for j in 0..n_total {
                    if i == j {
                        continue;
                    }
                    if !is_undirected(&oriented, i, j) {
                        continue;
                    }
                    let mut found = false;
                    'outer: for (k, adj_k) in adj.iter().enumerate() {
                        if k == i || k == j {
                            continue;
                        }
                        if !(is_undirected(&oriented, i, k) && is_directed(&oriented, k, j)) {
                            continue;
                        }
                        for (l, &adj_kl) in adj_k.iter().enumerate() {
                            if l == i || l == j || l == k {
                                continue;
                            }
                            if is_undirected(&oriented, i, l)
                                && is_directed(&oriented, l, j)
                                && !adj_kl
                            {
                                found = true;
                                break 'outer;
                            }
                        }
                    }
                    if found && j != c_index {
                        oriented[i][j] = 1;
                        oriented[j][i] = 0;
                        changed = true;
                    }
                }
            }
        }

        Ok(CdNodResult {
            n_vars: p,
            skeleton,
            changing_vars,
            oriented,
            c_index,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Generate a stationary 2-domain linear-Gaussian dataset where the
    /// mechanism is identical in both domains: X0 ~ N, X1 = 0.8·X0 + ε.
    fn stationary_two_domain(n_per: usize, seed: u64) -> (Vec<f32>, Vec<usize>, usize, usize) {
        let p = 2;
        let n = n_per * 2;
        let mut rng = LcgRng::new(seed);
        let mut data = vec![0.0_f32; n * p];
        let mut labels = vec![0_usize; n];
        for s in 0..n {
            let x0 = rng.next_normal();
            let x1 = 0.8 * x0 + 0.3 * rng.next_normal();
            data[s * p] = x0;
            data[s * p + 1] = x1;
            labels[s] = if s < n_per { 0 } else { 1 };
        }
        (data, labels, n, p)
    }

    #[test]
    fn build_surrogate_length_and_values() {
        let labels = vec![0, 1, 2, 3];
        let c = CdNod::build_surrogate(&labels, 4).unwrap();
        assert_eq!(c.len(), 4);
        assert!((c[0] - 0.0).abs() < 1e-6);
        assert!((c[1] - (1.0 / 3.0)).abs() < 1e-6);
        assert!((c[2] - (2.0 / 3.0)).abs() < 1e-6);
        assert!((c[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn build_surrogate_two_domains_endpoints() {
        let labels = vec![0, 0, 1, 1];
        let c = CdNod::build_surrogate(&labels, 2).unwrap();
        assert!((c[0]).abs() < 1e-6);
        assert!((c[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn build_surrogate_rejects_few_domains() {
        let labels = vec![0, 0];
        assert!(matches!(
            CdNod::build_surrogate(&labels, 1),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn build_surrogate_rejects_bad_label() {
        let labels = vec![0, 5];
        assert!(matches!(
            CdNod::build_surrogate(&labels, 2),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn run_rejects_few_domains() {
        let (data, labels, n, p) = stationary_two_domain(20, 1);
        let cfg = CdNodConfig {
            n_vars: p,
            n_domains: 1,
            alpha: 0.05,
            max_cond_set: 2,
        };
        assert!(matches!(
            CdNod::run(&data, n, &labels, &cfg),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn run_rejects_label_out_of_range() {
        let (data, mut labels, n, p) = stationary_two_domain(20, 2);
        labels[0] = 9;
        let cfg = CdNodConfig {
            n_vars: p,
            n_domains: 2,
            alpha: 0.05,
            max_cond_set: 2,
        };
        assert!(matches!(
            CdNod::run(&data, n, &labels, &cfg),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn run_rejects_data_length_mismatch() {
        let (mut data, labels, n, p) = stationary_two_domain(20, 3);
        data.push(0.0); // wrong length
        let cfg = CdNodConfig {
            n_vars: p,
            n_domains: 2,
            alpha: 0.05,
            max_cond_set: 2,
        };
        assert!(matches!(
            CdNod::run(&data, n, &labels, &cfg),
            Err(CausalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn run_rejects_label_length_mismatch() {
        let (data, mut labels, n, p) = stationary_two_domain(20, 4);
        labels.push(0); // wrong length
        let cfg = CdNodConfig {
            n_vars: p,
            n_domains: 2,
            alpha: 0.05,
            max_cond_set: 2,
        };
        assert!(matches!(
            CdNod::run(&data, n, &labels, &cfg),
            Err(CausalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn run_rejects_too_few_samples() {
        let data = vec![0.0_f32; 3 * 2];
        let labels = vec![0, 1, 0];
        let cfg = CdNodConfig {
            n_vars: 2,
            n_domains: 2,
            alpha: 0.05,
            max_cond_set: 2,
        };
        assert!(matches!(
            CdNod::run(&data, 3, &labels, &cfg),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn skeleton_is_symmetric_and_c_index_correct() {
        let (data, labels, n, p) = stationary_two_domain(60, 5);
        let cfg = CdNodConfig {
            n_vars: p,
            n_domains: 2,
            alpha: 0.05,
            max_cond_set: 2,
        };
        let res = CdNod::run(&data, n, &labels, &cfg).unwrap();
        assert_eq!(res.c_index, p);
        assert_eq!(res.skeleton.len(), p + 1);
        for i in 0..(p + 1) {
            assert_eq!(res.skeleton[i].len(), p + 1);
            assert!(!res.skeleton[i][i], "no self-loops");
            for j in 0..(p + 1) {
                assert_eq!(
                    res.skeleton[i][j], res.skeleton[j][i],
                    "symmetric at {i},{j}"
                );
            }
        }
    }

    #[test]
    fn oriented_dims_are_squared() {
        let (data, labels, n, p) = stationary_two_domain(40, 6);
        let cfg = CdNodConfig {
            n_vars: p,
            n_domains: 2,
            alpha: 0.05,
            max_cond_set: 2,
        };
        let res = CdNod::run(&data, n, &labels, &cfg).unwrap();
        let total = (p + 1) * (p + 1);
        let count: usize = res.oriented.iter().map(|r| r.len()).sum();
        assert_eq!(count, total);
        assert_eq!(res.oriented.len(), p + 1);
    }

    #[test]
    fn stationary_data_has_no_changing_vars() {
        // Same linear mechanism in both domains ⇒ C should not be adjacent to
        // any X once we condition appropriately.
        let (data, labels, n, p) = stationary_two_domain(150, 7);
        let cfg = CdNodConfig {
            n_vars: p,
            n_domains: 2,
            alpha: 0.05,
            max_cond_set: 3,
        };
        let res = CdNod::run(&data, n, &labels, &cfg).unwrap();
        assert!(
            res.changing_vars.is_empty(),
            "expected no changing vars, got {:?}",
            res.changing_vars
        );
    }

    #[test]
    fn flipping_mechanism_marks_changing_var() {
        // X0's generating mechanism flips sign across domains: in domain 0 the
        // mean is +shift, in domain 1 it is -shift. This makes X0 depend on C.
        let p = 2;
        let n_per = 150;
        let n = n_per * 2;
        let mut rng = LcgRng::new(8);
        let mut data = vec![0.0_f32; n * p];
        let mut labels = vec![0_usize; n];
        for s in 0..n {
            let domain = if s < n_per { 0 } else { 1 };
            labels[s] = domain;
            let shift = if domain == 0 { 2.0 } else { -2.0 };
            let x0 = shift + rng.next_normal();
            let x1 = 0.5 * x0 + 0.3 * rng.next_normal();
            data[s * p] = x0;
            data[s * p + 1] = x1;
        }
        let cfg = CdNodConfig {
            n_vars: p,
            n_domains: 2,
            alpha: 0.05,
            max_cond_set: 3,
        };
        let res = CdNod::run(&data, n, &labels, &cfg).unwrap();
        assert!(
            res.changing_vars.contains(&0),
            "expected X0 to be changing, got {:?}",
            res.changing_vars
        );
    }

    #[test]
    fn orientation_marks_c_to_changing_var() {
        let p = 2;
        let n_per = 150;
        let n = n_per * 2;
        let mut rng = LcgRng::new(9);
        let mut data = vec![0.0_f32; n * p];
        let mut labels = vec![0_usize; n];
        for s in 0..n {
            let domain = if s < n_per { 0 } else { 1 };
            labels[s] = domain;
            let shift = if domain == 0 { 3.0 } else { -3.0 };
            let x0 = shift + rng.next_normal();
            let x1 = 0.5 * x0 + 0.3 * rng.next_normal();
            data[s * p] = x0;
            data[s * p + 1] = x1;
        }
        let cfg = CdNodConfig {
            n_vars: p,
            n_domains: 2,
            alpha: 0.05,
            max_cond_set: 3,
        };
        let res = CdNod::run(&data, n, &labels, &cfg).unwrap();
        let c = res.c_index;
        for &v in &res.changing_vars {
            assert_eq!(res.oriented[c][v], 1, "expected C->{v}");
            assert_eq!(res.oriented[v][c], 0, "C must not have parent {v}");
        }
        assert!(!res.changing_vars.is_empty());
    }

    #[test]
    fn fisher_z_independent_gaussians_high_p() {
        let n = 400;
        let n_total = 2;
        let mut rng = LcgRng::new(10);
        let mut data = vec![0.0_f32; n * n_total];
        for s in 0..n {
            data[s * n_total] = rng.next_normal();
            data[s * n_total + 1] = rng.next_normal();
        }
        let (_stat, p) = CdNod::fisher_z_test(&data, n, n_total, 0, 1, &[]).unwrap();
        assert!(
            p > 0.05,
            "independent gaussians should have p>0.05, got {p}"
        );
    }

    #[test]
    fn fisher_z_correlated_columns_low_p() {
        let n = 400;
        let n_total = 2;
        let mut rng = LcgRng::new(11);
        let mut data = vec![0.0_f32; n * n_total];
        for s in 0..n {
            let x = rng.next_normal();
            data[s * n_total] = x;
            data[s * n_total + 1] = 0.9 * x + 0.05 * rng.next_normal();
        }
        let (_stat, p) = CdNod::fisher_z_test(&data, n, n_total, 0, 1, &[]).unwrap();
        assert!(p < 0.05, "correlated columns should have p<0.05, got {p}");
    }

    #[test]
    fn fisher_z_rejects_bad_length() {
        let data = vec![0.0_f32; 8];
        assert!(matches!(
            CdNod::fisher_z_test(&data, 5, 2, 0, 1, &[]),
            Err(CausalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn fisher_z_insufficient_data_is_independent() {
        // n - |S| - 3 <= 0 ⇒ report independence (0.0, 1.0).
        let n = 4;
        let n_total = 3;
        let data = vec![0.5_f32; n * n_total];
        let (stat, p) = CdNod::fisher_z_test(&data, n, n_total, 0, 1, &[2]).unwrap();
        assert_eq!(stat, 0.0);
        assert_eq!(p, 1.0);
    }

    #[test]
    fn chain_skeleton_recovers_edges() {
        // Linear-Gaussian chain X0 -> X1 -> X2 across 2 domains (stationary).
        let p = 3;
        let n_per = 200;
        let n = n_per * 2;
        let mut rng = LcgRng::new(12);
        let mut data = vec![0.0_f32; n * p];
        let mut labels = vec![0_usize; n];
        for s in 0..n {
            labels[s] = if s < n_per { 0 } else { 1 };
            let x0 = rng.next_normal();
            let x1 = 0.9 * x0 + 0.25 * rng.next_normal();
            let x2 = 0.9 * x1 + 0.25 * rng.next_normal();
            data[s * p] = x0;
            data[s * p + 1] = x1;
            data[s * p + 2] = x2;
        }
        let cfg = CdNodConfig {
            n_vars: p,
            n_domains: 2,
            alpha: 0.05,
            max_cond_set: 3,
        };
        let res = CdNod::run(&data, n, &labels, &cfg).unwrap();
        assert!(res.skeleton[0][1], "expected edge 0-1");
        assert!(res.skeleton[1][2], "expected edge 1-2");
        assert!(!res.skeleton[0][2], "expected NO edge 0-2 (separated by 1)");
    }

    #[test]
    fn deterministic_given_fixed_seed() {
        let (data, labels, n, p) = stationary_two_domain(80, 13);
        let cfg = CdNodConfig {
            n_vars: p,
            n_domains: 2,
            alpha: 0.05,
            max_cond_set: 2,
        };
        let a = CdNod::run(&data, n, &labels, &cfg).unwrap();
        let b = CdNod::run(&data, n, &labels, &cfg).unwrap();
        assert_eq!(a.skeleton, b.skeleton);
        assert_eq!(a.changing_vars, b.changing_vars);
        assert_eq!(a.oriented, b.oriented);
    }

    #[test]
    fn erf_known_values() {
        // erf(0)=0, erf(1)≈0.8427, erf(-1)≈-0.8427.
        assert!(erf(0.0).abs() < 1e-7);
        assert!((erf(1.0) - 0.842_700_79).abs() < 1e-5);
        assert!((erf(-1.0) + 0.842_700_79).abs() < 1e-5);
    }

    #[test]
    fn normal_cdf_known_values() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-3);
        assert!(normal_cdf(-5.0) < 1e-5);
        assert!(normal_cdf(5.0) > 1.0 - 1e-5);
    }

    #[test]
    fn subsets_enumeration_counts() {
        let items = vec![0, 1, 2, 3];
        assert_eq!(subsets_of_size(&items, 0).len(), 1);
        assert_eq!(subsets_of_size(&items, 1).len(), 4);
        assert_eq!(subsets_of_size(&items, 2).len(), 6);
        assert_eq!(subsets_of_size(&items, 3).len(), 4);
        assert_eq!(subsets_of_size(&items, 4).len(), 1);
        assert_eq!(subsets_of_size(&items, 5).len(), 0);
    }
}
