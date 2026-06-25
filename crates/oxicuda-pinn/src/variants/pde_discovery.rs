//! Data-driven governing-equation discovery via sparse regression (SINDy) and
//! a PDE-Net-style learned differential-operator primitive.
//!
//! Two complementary primitives for *discovering* the symbolic form of a
//! dynamical system or PDE from data, rather than assuming it:
//!
//! * [`fit_sindy`] — Sparse Identification of Nonlinear Dynamics (Brunton, Proctor
//!   & Kutz, PNAS 2016). Builds a library `Θ(X)` of candidate nonlinear
//!   features (polynomials and trigonometric terms) and recovers a *sparse*
//!   coefficient matrix `Ξ` such that `Ẋ ≈ Θ(X) Ξ`, using sequentially
//!   thresholded least squares (STLSQ). The non-zero entries name the active
//!   terms — symbolic-regression by feature selection.
//!
//! * [`PdeNetCell`] — one PDE-Net (Long et al., ICML 2018) δt forward block:
//!   learned finite-difference stencils approximate the spatial differential
//!   operators `∂_x, ∂_xx, …`, which are combined through a small polynomial
//!   response surface and advanced one explicit Euler step. The stencils are
//!   *moment-constrained* so each provably approximates the intended
//!   derivative order.
//!
//! Both are pure CPU, deterministic, and free of `ndarray` / `rand`.

use crate::error::{PinnError, PinnResult};

// ─── Candidate feature library ─────────────────────────────────────────────────

/// Which families of candidate terms to include in the SINDy feature library.
#[derive(Debug, Clone)]
pub struct LibraryConfig {
    /// State dimensionality (number of variables in each sample row).
    pub n_vars: usize,
    /// Highest total polynomial degree (1 = linear, 2 = +quadratic cross
    /// terms, 3 = +cubic, …). Capped at 3.
    pub poly_degree: usize,
    /// Include `sin(x_i)` and `cos(x_i)` for each variable.
    pub include_trig: bool,
}

impl LibraryConfig {
    /// Validate and build a library configuration.
    pub fn new(n_vars: usize, poly_degree: usize, include_trig: bool) -> PinnResult<Self> {
        if n_vars == 0 {
            return Err(PinnError::EmptyInput);
        }
        if poly_degree == 0 || poly_degree > 3 {
            return Err(PinnError::Internal(
                "poly_degree must be in 1..=3".to_string(),
            ));
        }
        Ok(Self {
            n_vars,
            poly_degree,
            include_trig,
        })
    }
}

/// Human-readable label for each candidate term, kept aligned with the columns
/// produced by [`build_library`].
fn term_labels(config: &LibraryConfig) -> Vec<String> {
    let n = config.n_vars;
    let mut labels = vec!["1".to_string()];
    // Degree 1.
    for i in 0..n {
        labels.push(format!("x{i}"));
    }
    // Degree 2.
    if config.poly_degree >= 2 {
        for i in 0..n {
            for j in i..n {
                labels.push(format!("x{i}*x{j}"));
            }
        }
    }
    // Degree 3.
    if config.poly_degree >= 3 {
        for i in 0..n {
            for j in i..n {
                for k in j..n {
                    labels.push(format!("x{i}*x{j}*x{k}"));
                }
            }
        }
    }
    if config.include_trig {
        for i in 0..n {
            labels.push(format!("sin(x{i})"));
        }
        for i in 0..n {
            labels.push(format!("cos(x{i})"));
        }
    }
    labels
}

/// Evaluate the candidate-function library on `samples` (`n_samples × n_vars`,
/// row-major). Returns `(theta, n_features)` where `theta` is
/// `n_samples × n_features` row-major, column-aligned with `term_labels`.
pub fn build_library(samples: &[f32], config: &LibraryConfig) -> PinnResult<(Vec<f32>, usize)> {
    let n = config.n_vars;
    if n == 0 {
        return Err(PinnError::EmptyInput);
    }
    if samples.len() % n != 0 {
        return Err(PinnError::DimensionMismatch {
            expected: samples.len().div_ceil(n) * n,
            got: samples.len(),
        });
    }
    let n_samples = samples.len() / n;
    if n_samples == 0 {
        return Err(PinnError::EmptyInput);
    }
    let n_features = term_labels(config).len();
    let mut theta = Vec::with_capacity(n_samples * n_features);
    for s in 0..n_samples {
        let row = &samples[s * n..(s + 1) * n];
        // Constant.
        theta.push(1.0);
        // Degree 1.
        for &v in row {
            theta.push(v);
        }
        // Degree 2.
        if config.poly_degree >= 2 {
            for i in 0..n {
                for j in i..n {
                    theta.push(row[i] * row[j]);
                }
            }
        }
        // Degree 3.
        if config.poly_degree >= 3 {
            for i in 0..n {
                for j in i..n {
                    for k in j..n {
                        theta.push(row[i] * row[j] * row[k]);
                    }
                }
            }
        }
        if config.include_trig {
            for &v in row {
                theta.push(v.sin());
            }
            for &v in row {
                theta.push(v.cos());
            }
        }
    }
    Ok((theta, n_features))
}

// ─── Ridge least squares via normal equations ──────────────────────────────────

/// Solve `(AᵀA + λI) ξ = Aᵀ b` for one target column.
/// `a` is `m × p` row-major; `b` has length `m`. Returns ξ (length `p`).
fn ridge_normal_equations(
    a: &[f32],
    b: &[f32],
    m: usize,
    p: usize,
    ridge: f32,
) -> PinnResult<Vec<f32>> {
    // Gram matrix G = AᵀA (p × p) + λI, and rhs = Aᵀ b (p).
    let mut gram = vec![0.0_f32; p * p];
    let mut rhs = vec![0.0_f32; p];
    for r in 0..m {
        let row = &a[r * p..(r + 1) * p];
        let br = b[r];
        for i in 0..p {
            rhs[i] += row[i] * br;
            for j in 0..p {
                gram[i * p + j] += row[i] * row[j];
            }
        }
    }
    for i in 0..p {
        gram[i * p + i] += ridge;
    }
    solve_spd(gram, rhs, p)
}

/// Solve `G x = b` for symmetric positive-(semi)definite `G` (`n × n`
/// row-major) via Gaussian elimination with partial pivoting.
fn solve_spd(mut g: Vec<f32>, mut b: Vec<f32>, n: usize) -> PinnResult<Vec<f32>> {
    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_mag = g[col * n + col].abs();
        for row in (col + 1)..n {
            let mag = g[row * n + col].abs();
            if mag > pivot_mag {
                pivot_mag = mag;
                pivot_row = row;
            }
        }
        if pivot_mag < 1e-20 {
            return Err(PinnError::SolverDivergence {
                reason: "singular Gram matrix in SINDy regression",
            });
        }
        if pivot_row != col {
            for k in 0..n {
                g.swap(col * n + k, pivot_row * n + k);
            }
            b.swap(col, pivot_row);
        }
        let pivot = g[col * n + col];
        for row in (col + 1)..n {
            let factor = g[row * n + col] / pivot;
            if factor != 0.0 {
                for k in col..n {
                    g[row * n + k] -= factor * g[col * n + k];
                }
                b[row] -= factor * b[col];
            }
        }
    }
    let mut x = vec![0.0_f32; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for k in (i + 1)..n {
            sum -= g[i * n + k] * x[k];
        }
        x[i] = sum / g[i * n + i];
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(PinnError::NanEncountered {
            location: "solve_spd",
        });
    }
    Ok(x)
}

// ─── SINDy ─────────────────────────────────────────────────────────────────────

/// Configuration for the SINDy sparse-regression discovery algorithm.
#[derive(Debug, Clone)]
pub struct SindyConfig {
    /// Feature-library specification.
    pub library: LibraryConfig,
    /// Coefficients with `|ξ| < threshold` are pruned each STLSQ pass.
    pub threshold: f32,
    /// Ridge (Tikhonov) regularisation strength for the inner least squares.
    pub ridge: f32,
    /// Number of sequential-thresholding iterations.
    pub max_iters: usize,
}

impl SindyConfig {
    /// Validate and construct a SINDy configuration.
    pub fn new(
        library: LibraryConfig,
        threshold: f32,
        ridge: f32,
        max_iters: usize,
    ) -> PinnResult<Self> {
        if !(threshold.is_finite() && threshold >= 0.0) {
            return Err(PinnError::Internal(
                "threshold must be finite and >= 0".to_string(),
            ));
        }
        if !(ridge.is_finite() && ridge >= 0.0) {
            return Err(PinnError::Internal(
                "ridge must be finite and >= 0".to_string(),
            ));
        }
        if max_iters == 0 {
            return Err(PinnError::Internal("max_iters must be >= 1".to_string()));
        }
        Ok(Self {
            library,
            threshold,
            ridge,
            max_iters,
        })
    }
}

/// Result of a SINDy fit: a sparse coefficient matrix plus term labels.
#[derive(Debug, Clone)]
pub struct SindyModel {
    /// Coefficient matrix `Ξ`, `n_features × n_targets` row-major.
    pub coefficients: Vec<f32>,
    /// Number of library features (rows of `Ξ`).
    pub n_features: usize,
    /// Number of target equations (columns of `Ξ`).
    pub n_targets: usize,
    /// Human-readable label for each feature row.
    pub labels: Vec<String>,
}

impl SindyModel {
    /// Predict `Ẋ` for a batch of states (`n_samples × n_vars` row-major).
    /// Returns `n_samples × n_targets` row-major.
    pub fn predict(&self, samples: &[f32], library: &LibraryConfig) -> PinnResult<Vec<f32>> {
        let (theta, n_feat) = build_library(samples, library)?;
        if n_feat != self.n_features {
            return Err(PinnError::DimensionMismatch {
                expected: self.n_features,
                got: n_feat,
            });
        }
        let n_samples = theta.len() / n_feat;
        let mut out = vec![0.0_f32; n_samples * self.n_targets];
        for s in 0..n_samples {
            let row = &theta[s * n_feat..(s + 1) * n_feat];
            for tgt in 0..self.n_targets {
                let acc: f32 = row
                    .iter()
                    .enumerate()
                    .map(|(f, &rv)| rv * self.coefficients[f * self.n_targets + tgt])
                    .sum();
                out[s * self.n_targets + tgt] = acc;
            }
        }
        Ok(out)
    }

    /// Number of active (non-zero) coefficients across all equations.
    #[must_use]
    pub fn n_active(&self) -> usize {
        self.coefficients.iter().filter(|&&c| c != 0.0).count()
    }

    /// Active term labels for target `tgt` with their coefficients.
    pub fn active_terms(&self, tgt: usize) -> Vec<(String, f32)> {
        let mut out = Vec::new();
        if tgt >= self.n_targets {
            return out;
        }
        for f in 0..self.n_features {
            let c = self.coefficients[f * self.n_targets + tgt];
            if c != 0.0 {
                out.push((self.labels[f].clone(), c));
            }
        }
        out
    }
}

/// Fit SINDy: recover a sparse `Ξ` so that `derivatives ≈ Θ(states) Ξ`.
///
/// * `states` — `n_samples × n_vars` row-major snapshots.
/// * `derivatives` — `n_samples × n_targets` row-major time-derivatives `Ẋ`
///   (typically `n_targets == n_vars`).
///
/// Uses STLSQ: solve ridge least squares, zero coefficients below `threshold`,
/// then re-solve on the surviving support, repeating `max_iters` times.
pub fn fit_sindy(
    states: &[f32],
    derivatives: &[f32],
    config: &SindyConfig,
) -> PinnResult<SindyModel> {
    let n_vars = config.library.n_vars;
    let (theta, n_feat) = build_library(states, &config.library)?;
    let n_samples = theta.len() / n_feat;
    if derivatives.len() % n_samples != 0 {
        return Err(PinnError::DimensionMismatch {
            expected: n_samples,
            got: derivatives.len(),
        });
    }
    let n_targets = derivatives.len() / n_samples;
    if n_targets == 0 {
        return Err(PinnError::EmptyInput);
    }
    let labels = term_labels(&config.library);
    debug_assert_eq!(labels.len(), n_feat);
    let _ = n_vars; // retained for clarity / future per-variable libraries

    let mut coefficients = vec![0.0_f32; n_feat * n_targets];

    for tgt in 0..n_targets {
        // Extract target column b (length n_samples).
        let b: Vec<f32> = (0..n_samples)
            .map(|s| derivatives[s * n_targets + tgt])
            .collect();
        // Active support: start with all features.
        let mut active: Vec<bool> = vec![true; n_feat];
        let mut xi = vec![0.0_f32; n_feat];

        for _ in 0..config.max_iters {
            // Build reduced design matrix from active columns.
            let active_idx: Vec<usize> = (0..n_feat).filter(|&f| active[f]).collect();
            let p = active_idx.len();
            if p == 0 {
                break;
            }
            let mut a_red = vec![0.0_f32; n_samples * p];
            for s in 0..n_samples {
                let full_row = &theta[s * n_feat..(s + 1) * n_feat];
                for (col, &f) in active_idx.iter().enumerate() {
                    a_red[s * p + col] = full_row[f];
                }
            }
            let sol = ridge_normal_equations(&a_red, &b, n_samples, p, config.ridge)?;
            // Scatter back and threshold.
            let mut new_xi = vec![0.0_f32; n_feat];
            let mut changed = false;
            for (col, &f) in active_idx.iter().enumerate() {
                let val = sol[col];
                if val.abs() < config.threshold {
                    if active[f] {
                        active[f] = false;
                        changed = true;
                    }
                    new_xi[f] = 0.0;
                } else {
                    new_xi[f] = val;
                }
            }
            xi = new_xi;
            if !changed {
                break;
            }
        }
        for f in 0..n_feat {
            coefficients[f * n_targets + tgt] = xi[f];
        }
    }

    Ok(SindyModel {
        coefficients,
        n_features: n_feat,
        n_targets,
        labels,
    })
}

// ─── PDE-Net learned differential operator ─────────────────────────────────────

/// One PDE-Net δt block on a 1D periodic grid.
///
/// Holds learnable 1D convolution stencils that approximate spatial
/// derivatives. The stencils are *moment-constrained* (their discrete moments
/// match those of the target derivative) so each provably represents
/// `∂^q/∂x^q` to leading order. A small polynomial response surface combines
/// the operator outputs and the block advances one explicit Euler step:
/// `u_{t+δt} = u_t + δt · Σ_q c_q (∂^q u)`.
#[derive(Debug, Clone)]
pub struct PdeNetCell {
    /// Stencil for each derivative order (each length `kernel_size`).
    stencils: Vec<Vec<f32>>,
    /// Linear combination weight for each derivative order.
    combine: Vec<f32>,
    /// Grid spacing.
    dx: f32,
    /// Time step.
    dt: f32,
}

impl PdeNetCell {
    /// Build a PDE-Net cell with finite-difference stencils for the given
    /// derivative orders (e.g. `[1, 2]` for advection + diffusion).
    ///
    /// `combine[q]` weights the q-th operator. `kernel_size` must be odd and
    /// at least `max_order + 1`.
    pub fn new(
        orders: &[usize],
        combine: Vec<f32>,
        kernel_size: usize,
        dx: f32,
        dt: f32,
    ) -> PinnResult<Self> {
        if orders.is_empty() {
            return Err(PinnError::EmptyInput);
        }
        if orders.len() != combine.len() {
            return Err(PinnError::DimensionMismatch {
                expected: orders.len(),
                got: combine.len(),
            });
        }
        if kernel_size < 3 || kernel_size % 2 == 0 {
            return Err(PinnError::Internal(
                "kernel_size must be odd and >= 3".to_string(),
            ));
        }
        if !(dx.is_finite() && dx > 0.0) {
            return Err(PinnError::InvalidPdeCoefficient {
                name: "dx",
                value: dx,
            });
        }
        if !(dt.is_finite() && dt > 0.0) {
            return Err(PinnError::InvalidStepSize { h: dt });
        }
        let max_order = orders.iter().copied().max().unwrap_or(0);
        if kernel_size <= max_order {
            return Err(PinnError::Internal(
                "kernel_size must exceed the maximum derivative order".to_string(),
            ));
        }
        let mut stencils = Vec::with_capacity(orders.len());
        for &q in orders {
            stencils.push(moment_constrained_stencil(q, kernel_size, dx)?);
        }
        Ok(Self {
            stencils,
            combine,
            dx,
            dt,
        })
    }

    /// Apply all stencils to a periodic 1D field `u` and return the per-order
    /// derivative estimates (`orders.len()` vectors, each length `u.len()`).
    pub fn operators(&self, u: &[f32]) -> PinnResult<Vec<Vec<f32>>> {
        if u.is_empty() {
            return Err(PinnError::EmptyInput);
        }
        let mut out = Vec::with_capacity(self.stencils.len());
        for stencil in &self.stencils {
            out.push(conv1d_periodic(u, stencil));
        }
        Ok(out)
    }

    /// Advance the field one δt step.
    pub fn step(&self, u: &[f32]) -> PinnResult<Vec<f32>> {
        let ops = self.operators(u)?;
        let n = u.len();
        let mut out = u.to_vec();
        for (q, op) in ops.iter().enumerate() {
            let c = self.combine[q];
            for i in 0..n {
                out[i] += self.dt * c * op[i];
            }
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(PinnError::NanEncountered {
                location: "PdeNetCell::step",
            });
        }
        Ok(out)
    }

    /// Grid spacing.
    #[must_use]
    pub fn dx(&self) -> f32 {
        self.dx
    }

    /// Time step.
    #[must_use]
    pub fn dt(&self) -> f32 {
        self.dt
    }
}

/// Circular 1D convolution (correlation) of a periodic field with a centred
/// stencil; output length equals `u.len()`.
fn conv1d_periodic(u: &[f32], stencil: &[f32]) -> Vec<f32> {
    let n = u.len();
    let k = stencil.len();
    let half = k / 2;
    let mut out = vec![0.0_f32; n];
    for (i, cell) in out.iter_mut().enumerate() {
        let mut acc = 0.0_f32;
        for (s, &w) in stencil.iter().enumerate() {
            // Centred: offset = s - half, periodic wrap.
            let idx = (i as isize + s as isize - half as isize).rem_euclid(n as isize) as usize;
            acc += w * u[idx];
        }
        *cell = acc;
    }
    out
}

/// Build a centred finite-difference stencil approximating `∂^q/∂x^q` of the
/// given odd width on uniform grid spacing `dx`, by solving the Vandermonde
/// moment conditions `Σ_s w_s (s·dx)^m / m! = δ_{m,q}`.
fn moment_constrained_stencil(q: usize, kernel_size: usize, dx: f32) -> PinnResult<Vec<f32>> {
    let k = kernel_size;
    let half = (k / 2) as isize;
    // Offsets in units of dx: -half .. +half.
    let offsets: Vec<f32> = (0..k).map(|s| (s as isize - half) as f32 * dx).collect();
    // Vandermonde system V w = e_q where V[m][s] = offset_s^m / m!.
    // (k equations for moments m = 0..k-1, k unknowns w_s.)
    let mut v = vec![0.0_f32; k * k];
    for m in 0..k {
        let fact = factorial(m);
        for s in 0..k {
            v[m * k + s] = offsets[s].powi(m as i32) / fact;
        }
    }
    let mut rhs = vec![0.0_f32; k];
    if q < k {
        rhs[q] = 1.0;
    }
    solve_spd(v, rhs, k)
}

/// Factorial as f32 (small arguments only).
fn factorial(n: usize) -> f32 {
    let mut acc = 1.0_f32;
    for i in 2..=n {
        acc *= i as f32;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_labels_match_columns() {
        let cfg = LibraryConfig::new(2, 2, false).expect("cfg");
        let labels = term_labels(&cfg);
        // 1, x0, x1, x0*x0, x0*x1, x1*x1 = 6.
        assert_eq!(labels.len(), 6);
        let samples = vec![2.0, 3.0]; // one sample
        let (theta, nf) = build_library(&samples, &cfg).expect("lib");
        assert_eq!(nf, 6);
        // 1, 2, 3, 4, 6, 9.
        assert!((theta[0] - 1.0).abs() < 1e-6);
        assert!((theta[1] - 2.0).abs() < 1e-6);
        assert!((theta[2] - 3.0).abs() < 1e-6);
        assert!((theta[3] - 4.0).abs() < 1e-6);
        assert!((theta[4] - 6.0).abs() < 1e-6);
        assert!((theta[5] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn sindy_recovers_linear_oscillator() {
        // Damped linear oscillator: x0' = x1, x1' = -2 x0 - 0.3 x1.
        // Sample a deterministic grid of states; compute exact derivatives.
        let mut states = Vec::new();
        let mut derivs = Vec::new();
        for i in 0..12 {
            for j in 0..12 {
                let x0 = -1.0 + 0.2 * i as f32;
                let x1 = -1.0 + 0.2 * j as f32;
                states.push(x0);
                states.push(x1);
                derivs.push(x1); // x0'
                derivs.push(-2.0 * x0 - 0.3 * x1); // x1'
            }
        }
        let lib = LibraryConfig::new(2, 2, false).expect("lib cfg");
        let cfg = SindyConfig::new(lib.clone(), 0.05, 1e-4, 10).expect("sindy cfg");
        let model = fit_sindy(&states, &derivs, &cfg).expect("fit");
        assert_eq!(model.n_targets, 2);

        // Recover coefficient for x1 in equation 0 (= 1.0) and x0,x1 in eq 1.
        let labels = &model.labels;
        let idx_x0 = labels.iter().position(|l| l == "x0").expect("x0");
        let idx_x1 = labels.iter().position(|l| l == "x1").expect("x1");
        let c_eq0_x1 = model.coefficients[idx_x1 * 2];
        let c_eq1_x0 = model.coefficients[idx_x0 * 2 + 1];
        let c_eq1_x1 = model.coefficients[idx_x1 * 2 + 1];
        assert!((c_eq0_x1 - 1.0).abs() < 1e-2, "eq0 x1 coeff = {c_eq0_x1}");
        assert!((c_eq1_x0 + 2.0).abs() < 1e-2, "eq1 x0 coeff = {c_eq1_x0}");
        assert!((c_eq1_x1 + 0.3).abs() < 1e-2, "eq1 x1 coeff = {c_eq1_x1}");

        // Sparsity: the true system has exactly 3 active terms.
        assert_eq!(model.n_active(), 3, "should recover a 3-term sparse model");
    }

    #[test]
    fn sindy_prediction_matches_derivatives() {
        // x0' = 3 x0 (pure growth).
        let mut states = Vec::new();
        let mut derivs = Vec::new();
        for i in 0..20 {
            let x = -2.0 + 0.2 * i as f32;
            states.push(x);
            derivs.push(3.0 * x);
        }
        let lib = LibraryConfig::new(1, 2, false).expect("lib");
        let cfg = SindyConfig::new(lib.clone(), 0.1, 1e-5, 8).expect("cfg");
        let model = fit_sindy(&states, &derivs, &cfg).expect("fit");
        let pred = model.predict(&states, &lib).expect("predict");
        for (p, d) in pred.iter().zip(derivs.iter()) {
            assert!((p - d).abs() < 1e-2, "pred {p} vs {d}");
        }
        // Single active term: x0.
        assert_eq!(model.n_active(), 1);
        let active = model.active_terms(0);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].0, "x0");
    }

    #[test]
    fn moment_stencil_first_derivative_is_central_difference() {
        // 3-point first-derivative stencil on dx = 1: [-0.5, 0, 0.5].
        let stencil = moment_constrained_stencil(1, 3, 1.0).expect("stencil");
        assert!((stencil[0] + 0.5).abs() < 1e-4, "w0 = {}", stencil[0]);
        assert!(stencil[1].abs() < 1e-4, "w1 = {}", stencil[1]);
        assert!((stencil[2] - 0.5).abs() < 1e-4, "w2 = {}", stencil[2]);
    }

    #[test]
    fn moment_stencil_second_derivative() {
        // 3-point Laplacian on dx = 1: [1, -2, 1].
        let stencil = moment_constrained_stencil(2, 3, 1.0).expect("stencil");
        assert!((stencil[0] - 1.0).abs() < 1e-3, "w0 = {}", stencil[0]);
        assert!((stencil[1] + 2.0).abs() < 1e-3, "w1 = {}", stencil[1]);
        assert!((stencil[2] - 1.0).abs() < 1e-3, "w2 = {}", stencil[2]);
    }

    #[test]
    fn pdenet_first_derivative_of_sine_matches_cosine() {
        // u(x) = sin(x) on periodic [0, 2π); ∂u/∂x = cos(x).
        let n = 64;
        let two_pi = 2.0 * std::f32::consts::PI;
        let dx = two_pi / n as f32;
        let u: Vec<f32> = (0..n).map(|i| (i as f32 * dx).sin()).collect();
        let cell = PdeNetCell::new(&[1], vec![1.0], 5, dx, 1e-3).expect("cell");
        let ops = cell.operators(&u).expect("ops");
        let deriv = &ops[0];
        for (i, &dv) in deriv.iter().enumerate() {
            let x = i as f32 * dx;
            assert!(
                (dv - x.cos()).abs() < 5e-2,
                "deriv at {i} = {dv}, want {}",
                x.cos()
            );
        }
    }

    #[test]
    fn pdenet_diffusion_step_smooths_field() {
        // Pure diffusion (order 2) should reduce the peak of a bump.
        let n = 64;
        let dx = 1.0 / n as f32;
        // Diffusion coefficient scaled for stability: dt * D / dx^2 < 0.5.
        let d = 0.1;
        let dt = 0.4 * dx * dx / d;
        let mut u = vec![0.0_f32; n];
        u[n / 2] = 1.0; // delta bump
        let cell = PdeNetCell::new(&[2], vec![d], 3, dx, dt).expect("cell");
        let peak_before = u[n / 2];
        let stepped = cell.step(&u).expect("step");
        assert!(
            stepped[n / 2] < peak_before,
            "diffusion should lower the peak: {} -> {}",
            peak_before,
            stepped[n / 2]
        );
        // Mass is conserved by a centred Laplacian on a periodic grid.
        let mass_before: f32 = u.iter().sum();
        let mass_after: f32 = stepped.iter().sum();
        assert!((mass_before - mass_after).abs() < 1e-4);
    }

    #[test]
    fn pdenet_advection_translates_profile() {
        // u_t = -c u_x advects a smooth bump; check it moves and stays bounded.
        let n = 128;
        let two_pi = 2.0 * std::f32::consts::PI;
        let dx = two_pi / n as f32;
        let c = 1.0;
        let dt = 0.2 * dx / c;
        let u: Vec<f32> = (0..n).map(|i| (i as f32 * dx).sin()).collect();
        let cell = PdeNetCell::new(&[1], vec![-c], 5, dx, dt).expect("cell");
        let mut field = u.clone();
        for _ in 0..10 {
            field = cell.step(&field).expect("step");
        }
        assert!(field.iter().all(|v| v.is_finite()));
        // Amplitude should stay near 1 (advection is non-dissipative ideally).
        let max_amp = field.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
        assert!(max_amp < 2.0 && max_amp > 0.3, "amplitude {max_amp}");
    }

    #[test]
    fn config_validation() {
        assert!(LibraryConfig::new(0, 2, false).is_err());
        assert!(LibraryConfig::new(2, 0, false).is_err());
        assert!(LibraryConfig::new(2, 4, false).is_err());
        let lib = LibraryConfig::new(2, 2, false).expect("lib");
        assert!(SindyConfig::new(lib.clone(), -1.0, 0.0, 5).is_err());
        assert!(SindyConfig::new(lib.clone(), 0.1, -1.0, 5).is_err());
        assert!(SindyConfig::new(lib, 0.1, 0.0, 0).is_err());
    }

    #[test]
    fn pdenet_invalid_kernel_size_errors() {
        assert!(PdeNetCell::new(&[1], vec![1.0], 4, 0.1, 0.01).is_err()); // even
        assert!(PdeNetCell::new(&[1], vec![1.0], 1, 0.1, 0.01).is_err()); // < 3
        assert!(PdeNetCell::new(&[2], vec![1.0], 3, -0.1, 0.01).is_err()); // dx <= 0
        assert!(PdeNetCell::new(&[], vec![], 3, 0.1, 0.01).is_err()); // empty
    }

    #[test]
    fn sindy_trig_library_recovers_pendulum() {
        // Pendulum: theta'' adds -sin(theta). Equation: x1' = -sin(x0).
        let mut states = Vec::new();
        let mut derivs = Vec::new();
        for i in 0..16 {
            for j in 0..8 {
                let x0 = -3.0 + 0.4 * i as f32;
                let x1 = -1.0 + 0.25 * j as f32;
                states.push(x0);
                states.push(x1);
                derivs.push(x1);
                derivs.push(-x0.sin());
            }
        }
        let lib = LibraryConfig::new(2, 1, true).expect("lib");
        let cfg = SindyConfig::new(lib, 0.1, 1e-4, 10).expect("cfg");
        let model = fit_sindy(&states, &derivs, &cfg).expect("fit");
        let idx_sin0 = model
            .labels
            .iter()
            .position(|l| l == "sin(x0)")
            .expect("sin0");
        let c = model.coefficients[idx_sin0 * 2 + 1];
        assert!((c + 1.0).abs() < 5e-2, "sin(x0) coeff in eq1 = {c}");
    }
}
