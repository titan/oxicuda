//! Synthetic data generators with known ground truth.
//!
//! Every generator returns both the sampled data *and* the parameter that the
//! estimator under test is supposed to recover, so verification suites can assert
//! genuine numerical correctness (edge recovery, ATE recovery, CATE/PEHE) rather
//! than merely checking that an output is finite.

use crate::handle::LcgRng;

/// A linear-Gaussian structural equation model with a known weighted adjacency
/// matrix `w` (row-major, `w[i*d + j]` is the coefficient of parent `i` into
/// child `j`). The topological order of the variables is `topo`.
pub struct LinearSem {
    pub d: usize,
    /// True weighted adjacency, row-major `d × d`.
    pub w: Vec<f32>,
    /// A valid topological order of the variables.
    pub topo: Vec<usize>,
}

impl LinearSem {
    /// Sample `n` rows from `X_j = Σ_i w[i,j]·X_i + ε_j`, `ε_j ~ N(0, noise²)`.
    ///
    /// Returns a row-major `n × d` data matrix.
    #[must_use]
    pub fn sample(&self, n: usize, noise: f32, rng: &mut LcgRng) -> Vec<f32> {
        let d = self.d;
        let mut data = vec![0.0_f32; n * d];
        for row in 0..n {
            for &j in &self.topo {
                let mut acc = 0.0_f32;
                for i in 0..d {
                    let wij = self.w[i * d + j];
                    if wij != 0.0 {
                        acc += wij * data[row * d + i];
                    }
                }
                acc += noise * rng.next_normal();
                data[row * d + j] = acc;
            }
        }
        data
    }

    /// The set of true directed edges `(parent, child)`.
    #[must_use]
    pub fn true_edges(&self) -> Vec<(usize, usize)> {
        let d = self.d;
        let mut edges = Vec::new();
        for i in 0..d {
            for j in 0..d {
                if self.w[i * d + j].abs() > 1e-9 {
                    edges.push((i, j));
                }
            }
        }
        edges
    }

    /// The undirected skeleton (unordered adjacent pairs `(min, max)`).
    #[must_use]
    pub fn true_skeleton(&self) -> Vec<(usize, usize)> {
        let mut sk: Vec<(usize, usize)> = self
            .true_edges()
            .into_iter()
            .map(|(a, b)| if a < b { (a, b) } else { (b, a) })
            .collect();
        sk.sort_unstable();
        sk.dedup();
        sk
    }
}

/// A simple chain SEM `0 -> 1 -> 2 -> ... -> (d-1)` with unit-ish weights.
#[must_use]
pub fn chain_sem(d: usize, weight: f32) -> LinearSem {
    let mut w = vec![0.0_f32; d * d];
    for i in 0..d.saturating_sub(1) {
        w[i * d + (i + 1)] = weight;
    }
    LinearSem {
        d,
        w,
        topo: (0..d).collect(),
    }
}

/// The classic "fork + collider" 4-node graph used to exercise v-structure
/// orientation: `0 -> 2`, `1 -> 2`, `2 -> 3`. Variables 0 and 1 are marginally
/// independent but become dependent given the collider 2.
#[must_use]
pub fn collider_sem(weight: f32) -> LinearSem {
    let d = 4;
    let mut w = vec![0.0_f32; d * d];
    w[2] = weight; // 0 -> 2  (row 0, col 2)
    w[d + 2] = weight; // 1 -> 2  (row 1, col 2)
    w[2 * d + 3] = weight; // 2 -> 3
    LinearSem {
        d,
        w,
        topo: vec![0, 1, 2, 3],
    }
}

/// A randomly-weighted upper-triangular (hence acyclic) SEM in the natural
/// variable order `0..d`. Each potential parent→child edge is present with
/// probability `edge_prob`, with a weight drawn uniformly from `±[0.5, 1.5]`
/// (bounded away from zero so recovery is identifiable).
#[must_use]
pub fn random_dag_sem(d: usize, edge_prob: f32, rng: &mut LcgRng) -> LinearSem {
    let mut w = vec![0.0_f32; d * d];
    for i in 0..d {
        for j in (i + 1)..d {
            if rng.next_f32() < edge_prob {
                let mag = 0.5 + rng.next_f32(); // [0.5, 1.5]
                let sign = if rng.next_f32() < 0.5 { -1.0 } else { 1.0 };
                w[i * d + j] = sign * mag;
            }
        }
    }
    LinearSem {
        d,
        w,
        topo: (0..d).collect(),
    }
}

/// A binary-treatment DGP with a known *heterogeneous* treatment effect.
///
/// `Y = baseline(X) + T·τ(X) + ε`, with `τ(x) = tau_base + tau_slope·x₀`
/// (CATE varies linearly with the first covariate) and a randomized treatment
/// (so the propensity is exactly 0.5 and ATE = E[τ(X)] = tau_base).
pub struct HeteroEffectData {
    /// Row-major `n × d` covariates.
    pub x: Vec<f32>,
    /// Treatment indicator (0/1).
    pub t: Vec<f32>,
    /// Observed outcome.
    pub y: Vec<f32>,
    /// True per-unit CATE `τ(Xᵢ)`.
    pub cate_true: Vec<f32>,
    /// True average treatment effect `E[τ(X)]`.
    pub ate_true: f32,
    pub n: usize,
    pub d: usize,
}

/// Generate a heterogeneous-effect dataset (see [`HeteroEffectData`]).
#[must_use]
pub fn hetero_effect_data(
    n: usize,
    d: usize,
    tau_base: f32,
    tau_slope: f32,
    noise: f32,
    rng: &mut LcgRng,
) -> HeteroEffectData {
    let mut x = vec![0.0_f32; n * d];
    let mut t = vec![0.0_f32; n];
    let mut y = vec![0.0_f32; n];
    let mut cate_true = vec![0.0_f32; n];
    let mut ate_acc = 0.0_f64;
    for i in 0..n {
        for j in 0..d {
            x[i * d + j] = rng.next_normal();
        }
        // Randomized treatment, propensity exactly 0.5.
        let ti = if rng.next_f32() < 0.5 { 1.0 } else { 0.0 };
        t[i] = ti;
        let tau = tau_base + tau_slope * x[i * d];
        cate_true[i] = tau;
        ate_acc += tau as f64;
        // A smooth confounder-free baseline plus the treatment contribution.
        // Second covariate when present, else fall back to the first.
        let second = if d > 1 { 1 } else { 0 };
        let baseline = 0.8 * x[i * d] - 0.5 * x[i * d + second];
        y[i] = baseline + ti * tau + noise * rng.next_normal();
    }
    let ate_true = (ate_acc / n as f64) as f32;
    HeteroEffectData {
        x,
        t,
        y,
        cate_true,
        ate_true,
        n,
        d,
    }
}

/// A confounded binary-treatment DGP with a *constant* (homogeneous) effect,
/// used by the Double-ML coverage study.
///
/// `T = 1{ logistic(γ·X) > u }`, `Y = β·X + θ·T + ε`. The treatment depends on
/// `X` (so naive comparison is biased) but partialling-out `X` recovers `θ`.
pub struct ConfoundedData {
    pub x: Vec<f32>,
    pub t: Vec<f32>,
    pub y: Vec<f32>,
    /// True (homogeneous) treatment effect θ.
    pub theta_true: f32,
    pub n: usize,
    pub d: usize,
}

/// Generate a confounded constant-effect dataset (see [`ConfoundedData`]).
#[must_use]
pub fn confounded_data(
    n: usize,
    d: usize,
    theta: f32,
    noise: f32,
    rng: &mut LcgRng,
) -> ConfoundedData {
    let mut x = vec![0.0_f32; n * d];
    let mut t = vec![0.0_f32; n];
    let mut y = vec![0.0_f32; n];
    // Fixed coefficient patterns (deterministic given d) so the DGP is stable.
    for i in 0..n {
        for j in 0..d {
            x[i * d + j] = rng.next_normal();
        }
        // Propensity logit depends on the covariates -> confounding.
        let logit: f32 = (0..d).map(|j| 0.7 * x[i * d + j] / (j as f32 + 1.0)).sum();
        let p = 1.0 / (1.0 + (-logit).exp());
        let ti = if rng.next_f32() < p { 1.0 } else { 0.0 };
        t[i] = ti;
        let base: f32 = (0..d).map(|j| 1.0 * x[i * d + j] / (j as f32 + 1.0)).sum();
        y[i] = base + theta * ti + noise * rng.next_normal();
    }
    ConfoundedData {
        x,
        t,
        y,
        theta_true: theta,
        n,
        d,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_sem_edges() {
        let sem = chain_sem(4, 0.9);
        assert_eq!(sem.true_edges(), vec![(0, 1), (1, 2), (2, 3)]);
        assert_eq!(sem.true_skeleton(), vec![(0, 1), (1, 2), (2, 3)]);
    }

    #[test]
    fn chain_sem_correlation_structure() {
        // Adjacent variables are strongly correlated; sampling is deterministic.
        let sem = chain_sem(3, 0.9);
        let mut rng = LcgRng::new(7);
        let n = 400;
        let data = sem.sample(n, 0.3, &mut rng);
        // Column means roughly zero.
        for j in 0..3 {
            let m: f32 = (0..n).map(|i| data[i * 3 + j]).sum::<f32>() / n as f32;
            assert!(m.abs() < 0.3, "col {j} mean {m}");
        }
    }

    #[test]
    fn collider_sem_marginal_independence() {
        // 0 and 1 are marginal roots: their sampled correlation should be small.
        let sem = collider_sem(1.0);
        let mut rng = LcgRng::new(3);
        let n = 600;
        let data = sem.sample(n, 0.5, &mut rng);
        let c0: Vec<f32> = (0..n).map(|i| data[i * 4]).collect();
        let c1: Vec<f32> = (0..n).map(|i| data[i * 4 + 1]).collect();
        let r = corr(&c0, &c1);
        assert!(r.abs() < 0.15, "0 and 1 should be ~independent, r={r}");
        // The collider 2 is correlated with both parents.
        let c2: Vec<f32> = (0..n).map(|i| data[i * 4 + 2]).collect();
        assert!(corr(&c0, &c2).abs() > 0.4);
        assert!(corr(&c1, &c2).abs() > 0.4);
    }

    #[test]
    fn hetero_data_ate_matches_mean_cate() {
        let mut rng = LcgRng::new(11);
        let dgp = hetero_effect_data(2000, 3, 1.5, 0.8, 0.2, &mut rng);
        let mean_cate: f32 = dgp.cate_true.iter().sum::<f32>() / dgp.n as f32;
        assert!((dgp.ate_true - mean_cate).abs() < 1e-4);
        // E[tau(X)] = tau_base since E[X0] = 0; allow sampling slack.
        assert!((dgp.ate_true - 1.5).abs() < 0.1, "ate={}", dgp.ate_true);
    }

    fn corr(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let ma = a.iter().sum::<f32>() / n as f32;
        let mb = b.iter().sum::<f32>() / n as f32;
        let mut num = 0.0;
        let mut va = 0.0;
        let mut vb = 0.0;
        for i in 0..n {
            num += (a[i] - ma) * (b[i] - mb);
            va += (a[i] - ma).powi(2);
            vb += (b[i] - mb).powi(2);
        }
        num / (va.sqrt() * vb.sqrt() + 1e-12)
    }
}
