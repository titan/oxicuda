//! Deep Gaussian Processes (Damianou & Lawrence 2013) with doubly-stochastic
//! variational inference (Salimbeni & Deisenroth 2017).
//!
//! A deep GP stacks `L` sparse variational GP layers, each mapping the output
//! of the previous layer through its own GP:
//!
//! ```text
//! f^(1) = GP_1(x),   f^(l+1) = GP_{l+1}(f^(l)),   y = f^(L) + ε.
//! ```
//!
//! Every layer carries `m` inducing inputs `Z_l` and an inducing-value
//! posterior `q(u_l) = N(μ_l, S_l)` with prior `p(u_l) = N(0, K_mm^l)`.  The
//! sparse variational GP (SVGP, Hensman et al. 2013) marginal at an input `x`
//! is
//!
//! ```text
//! μ(x) = m(x) + k_xm K_mm⁻¹ μ_u
//! σ²(x) = k_xx − k_xm K_mm⁻¹ k_mx + k_xm K_mm⁻¹ S K_mm⁻¹ k_mx,
//! ```
//!
//! where `m(x)` is a per-layer mean function (identity for hidden layers, zero
//! for the output layer, following Salimbeni & Deisenroth to avoid the
//! mean-collapse pathology).  Doubly-stochastic VI draws one sample per data
//! point at every layer and propagates it forward.
//!
//! The evidence lower bound is
//!
//! ```text
//! ELBO = Σ_n E_q[log p(y_n | f^(L)_n)] − Σ_l KL(q(u_l) ‖ p(u_l)).
//! ```
//!
//! The output layer's variational posterior is set to the closed-form Titsias
//! (2009) optimum given the (mean-propagated) hidden representation; the hidden
//! layers keep the prior posterior `q(u)=p(u)` with an identity mean function,
//! which makes a single-layer deep GP reduce exactly to a sparse variational
//! GP.  End-to-end gradient training of the hidden variational parameters is
//! intentionally out of scope — this is a CPU reference implementation.
//!
//! **References:**
//! - Damianou, A., & Lawrence, N. (2013). Deep Gaussian Processes. *AISTATS*.
//! - Salimbeni, H., & Deisenroth, M. (2017). Doubly Stochastic Variational
//!   Inference for Deep Gaussian Processes. *NeurIPS*.
//! - Titsias, M. (2009). Variational Learning of Inducing Variables in Sparse
//!   Gaussian Processes. *AISTATS*.
//! - Hensman, J., Fusi, N., & Lawrence, N. (2013). Gaussian Processes for Big
//!   Data. *UAI*.

use crate::error::{BayesError, BayesResult};
use crate::handle::LcgRng;

use super::gpr::GprKernel;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for a single deep GP layer.
#[derive(Debug, Clone)]
pub struct DeepGpLayerConfig {
    /// Input dimensionality of this layer.
    pub d_in: usize,
    /// Output dimensionality of this layer.
    pub d_out: usize,
    /// Number of inducing points `m` (clamped to the training-set size in `fit`).
    pub n_inducing: usize,
    /// Kernel function for this layer (shared across its output dimensions).
    pub kernel: GprKernel,
}

/// Configuration for a deep Gaussian Process.
#[derive(Debug, Clone)]
pub struct DeepGpConfig {
    /// Layer specifications, ordered input → output.
    pub layers: Vec<DeepGpLayerConfig>,
    /// Gaussian observation-noise variance `σ²` for the likelihood.
    pub noise_variance: f64,
    /// Numerical stability jitter added to the diagonal before each Cholesky.
    pub jitter: f64,
}

// ─── Layer state ─────────────────────────────────────────────────────────────

/// A fitted / initialised sparse variational GP layer.
#[derive(Debug, Clone)]
pub struct DeepGpLayer {
    /// Input dimensionality.
    pub d_in: usize,
    /// Output dimensionality.
    pub d_out: usize,
    /// Number of inducing points `m`.
    pub n_inducing: usize,
    /// Kernel.
    pub kernel: GprKernel,
    /// Inducing inputs, row-major `[m × d_in]`.
    pub z: Vec<f64>,
    /// Variational means `μ_u`, one length-`m` vector per output dimension
    /// (`d_out × m`, row-major).
    pub q_mu: Vec<f64>,
    /// Variational covariance `S`, shared across output dimensions, `[m × m]`.
    pub s: Vec<f64>,
    /// Whether this layer uses the identity mean function `m(x) = x`
    /// (hidden layers) instead of the zero mean function (output layer).
    pub identity_mean: bool,
}

// ─── Deep GP ─────────────────────────────────────────────────────────────────

/// Deep Gaussian Process with doubly-stochastic variational inference.
#[derive(Debug, Clone)]
pub struct DeepGp {
    /// The stacked SVGP layers.
    pub layers: Vec<DeepGpLayer>,
    /// Observation-noise variance.
    pub noise_variance: f64,
    /// Cholesky jitter.
    pub jitter: f64,
    /// Overall input dimensionality (first layer `d_in`).
    pub d_in: usize,
    /// Overall output dimensionality (last layer `d_out`).
    pub d_out: usize,
}

// ─── Linear algebra helpers ──────────────────────────────────────────────────

/// Lower-triangular Cholesky decomposition (Banachiewicz). `None` if not SPD.
fn cholesky_lower(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0_f64;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let diag = a[i * n + i] - sum;
                if diag <= 0.0 {
                    return None;
                }
                l[i * n + j] = diag.sqrt();
            } else {
                let lj = l[j * n + j];
                if lj == 0.0 {
                    return None;
                }
                l[i * n + j] = (a[i * n + j] - sum) / lj;
            }
        }
    }
    Some(l)
}

/// Cholesky with progressive jitter, up to 7 attempts.
fn cholesky_jitter(a_base: &[f64], n: usize, initial_jitter: f64) -> BayesResult<Vec<f64>> {
    let mut jitter = initial_jitter.max(1e-12);
    for _ in 0..7 {
        let mut a = a_base.to_vec();
        for i in 0..n {
            a[i * n + i] += jitter;
        }
        if let Some(l) = cholesky_lower(&a, n) {
            return Ok(l);
        }
        jitter *= 10.0;
    }
    Err(BayesError::SingularMatrix(
        "deep GP covariance not positive-definite after jitter retries".into(),
    ))
}

/// Forward substitution: solve `L·x = b`.
fn fwd_sub(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * x[j];
        }
        let lii = l[i * n + i];
        x[i] = if lii.abs() < 1e-300 { 0.0 } else { s / lii };
    }
    x
}

/// Backward substitution: solve `Lᵀ·x = b`.
fn bwd_sub_lt(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= l[j * n + i] * x[j];
        }
        let lii = l[i * n + i];
        x[i] = if lii.abs() < 1e-300 { 0.0 } else { s / lii };
    }
    x
}

/// Solve `(L·Lᵀ)·x = b` given the lower-triangular Cholesky factor `L`.
fn solve_chol(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let y = fwd_sub(l, b, n);
    bwd_sub_lt(l, &y, n)
}

/// Matrix-vector product `out = M·v` for a row-major `[n × n]` matrix.
fn matvec(m: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|r| {
            let off = r * n;
            (0..n).map(|c| m[off + c] * v[c]).sum()
        })
        .collect()
}

/// Dot product of two equal-length slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// `log det` of an SPD matrix from its Cholesky factor: `2·Σ log L_ii`.
fn logdet_from_chol(l: &[f64], n: usize) -> f64 {
    (0..n).map(|i| l[i * n + i].ln()).sum::<f64>() * 2.0
}

/// One standard-normal draw (f64) from the LCG via its Box-Muller pair.
fn next_normal(rng: &mut LcgRng) -> f64 {
    let (e, _) = rng.next_normal_pair();
    f64::from(e)
}

/// Sub-sample `m` evenly-strided rows from a row-major `[n × d]` buffer.
fn place_inducing(phi: &[f64], n: usize, d: usize, m: usize) -> Vec<f64> {
    let mut z = vec![0.0_f64; m * d];
    for k in 0..m {
        let idx = if m <= 1 {
            0
        } else {
            ((k * (n - 1)) as f64 / (m - 1) as f64).round() as usize
        };
        let idx = idx.min(n.saturating_sub(1));
        z[k * d..(k + 1) * d].copy_from_slice(&phi[idx * d..(idx + 1) * d]);
    }
    z
}

// ─── Layer operations ────────────────────────────────────────────────────────

impl DeepGpLayer {
    /// Build `K_mm = K(Z, Z)` for this layer.
    fn k_mm(&self) -> Vec<f64> {
        self.kernel.eval_matrix(
            &self.z,
            self.n_inducing,
            &self.z,
            self.n_inducing,
            self.d_in,
        )
    }

    /// SVGP marginal predictive at a single input `x_star`.
    ///
    /// Returns `(means[d_out], variance)`.  The variance is shared across
    /// output dimensions (same kernel and `S`); the means differ via `q_mu`
    /// and the mean function.
    fn marginal(&self, x_star: &[f64], l_mm: &[f64]) -> (Vec<f64>, f64) {
        let m = self.n_inducing;
        // k_*m = [k(x*, z_j)].
        let k_star: Vec<f64> = (0..m)
            .map(|j| {
                self.kernel
                    .eval(x_star, &self.z[j * self.d_in..(j + 1) * self.d_in])
            })
            .collect();
        let k_ss = self.kernel.eval(x_star, x_star);

        // b = K_mm⁻¹ k_*m.
        let b = solve_chol(l_mm, &k_star, m);
        let quad_prior = dot(&k_star, &b);
        // bᵀ S b.
        let sb = matvec(&self.s, &b, m);
        let quad_post = dot(&b, &sb);
        let var = (k_ss - quad_prior + quad_post).max(0.0);

        let means: Vec<f64> = (0..self.d_out)
            .map(|d| {
                let mu_row = &self.q_mu[d * m..(d + 1) * m];
                let gp = dot(&b, mu_row);
                if self.identity_mean {
                    x_star[d] + gp
                } else {
                    gp
                }
            })
            .collect();
        (means, var)
    }

    /// KL(`q(u) ‖ p(u)`) for this layer, summed over output dimensions.
    fn kl(&self, jitter: f64) -> BayesResult<f64> {
        let m = self.n_inducing;
        let k_mm = self.k_mm();
        let l_mm = cholesky_jitter(&k_mm, m, jitter)?;
        let l_s = cholesky_jitter(&self.s, m, jitter)?;

        let logdet_kmm = logdet_from_chol(&l_mm, m);
        let logdet_s = logdet_from_chol(&l_s, m);

        // tr(K_mm⁻¹ S) = Σ_c (K_mm⁻¹ S)[c, c].
        let mut trace = 0.0_f64;
        for c in 0..m {
            let col: Vec<f64> = (0..m).map(|r| self.s[r * m + c]).collect();
            let x = solve_chol(&l_mm, &col, m);
            trace += x[c];
        }

        // Per-output-dim quadratic μᵀ K_mm⁻¹ μ.
        let mut quad = 0.0_f64;
        for d in 0..self.d_out {
            let mu_row = &self.q_mu[d * m..(d + 1) * m];
            let solved = solve_chol(&l_mm, mu_row, m);
            quad += dot(mu_row, &solved);
        }

        let d_out = self.d_out as f64;
        let kl = 0.5 * d_out * (trace + logdet_kmm - logdet_s - m as f64) + 0.5 * quad;
        Ok(kl.max(0.0))
    }
}

impl DeepGp {
    /// Construct a deep GP from a configuration, scattering inducing inputs
    /// uniformly in `[-1, 1]` (data-free; [`Self::fit`] re-places them on data).
    ///
    /// # Errors
    /// - [`BayesError::InvalidConfig`] — no layers, a layer with zero
    ///   dimensionality / inducing count, mismatched layer dimensions, a hidden
    ///   layer whose `d_in != d_out` (identity mean undefined), invalid kernel
    ///   parameters, or negative noise variance.
    pub fn new(config: DeepGpConfig, rng: &mut LcgRng) -> BayesResult<Self> {
        if config.layers.is_empty() {
            return Err(BayesError::InvalidConfig(
                "deep GP requires at least one layer".into(),
            ));
        }
        if config.noise_variance < 0.0 {
            return Err(BayesError::InvalidConfig(
                "noise_variance must be non-negative".into(),
            ));
        }
        let n_layers = config.layers.len();
        let mut layers = Vec::with_capacity(n_layers);

        for (idx, lc) in config.layers.iter().enumerate() {
            if lc.d_in == 0 || lc.d_out == 0 {
                return Err(BayesError::InvalidConfig(
                    "layer dimensions must be >= 1".into(),
                ));
            }
            if lc.n_inducing == 0 {
                return Err(BayesError::InvalidConfig("n_inducing must be >= 1".into()));
            }
            validate_kernel(&lc.kernel)?;
            // Dimensions must chain: previous d_out == this d_in.
            if idx > 0 && config.layers[idx - 1].d_out != lc.d_in {
                return Err(BayesError::DimensionMismatch {
                    expected: config.layers[idx - 1].d_out,
                    got: lc.d_in,
                });
            }
            let is_output = idx == n_layers - 1;
            // Hidden layers use the identity mean function, so they must
            // preserve dimensionality.
            if !is_output && lc.d_in != lc.d_out {
                return Err(BayesError::InvalidConfig(
                    "hidden deep GP layers must have d_in == d_out (identity mean)".into(),
                ));
            }

            let m = lc.n_inducing;
            let mut z = vec![0.0_f64; m * lc.d_in];
            for v in z.iter_mut() {
                *v = f64::from(rng.next_f32()) * 2.0 - 1.0;
            }
            let q_mu = vec![0.0_f64; lc.d_out * m];

            let mut layer = DeepGpLayer {
                d_in: lc.d_in,
                d_out: lc.d_out,
                n_inducing: m,
                kernel: lc.kernel.clone(),
                z,
                q_mu,
                s: vec![0.0_f64; m * m],
                identity_mean: !is_output,
            };
            // Initialise S to the prior K_mm (so q = p, KL = 0).
            layer.s = layer.k_mm();
            layers.push(layer);
        }

        let d_in = config.layers[0].d_in;
        let d_out = config.layers[n_layers - 1].d_out;
        Ok(Self {
            layers,
            noise_variance: config.noise_variance,
            jitter: config.jitter,
            d_in,
            d_out,
        })
    }

    /// Number of layers.
    #[must_use]
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Fit the output layer's variational posterior in closed form (Titsias
    /// 2009) given training data, re-placing every layer's inducing inputs on
    /// the (mean-propagated) data.
    ///
    /// Hidden layers keep the prior posterior with an identity mean function;
    /// only the output layer is variationally fit to the targets.
    ///
    /// # Errors
    /// - [`BayesError::InvalidConfig`] — `n == 0`.
    /// - [`BayesError::DimensionMismatch`] — `x.len() != n·d_in` or
    ///   `y.len() != n·d_out`.
    /// - [`BayesError::SingularMatrix`] — a Cholesky fails after jitter retries.
    pub fn fit(&mut self, x: &[f64], y: &[f64], n: usize) -> BayesResult<()> {
        if n == 0 {
            return Err(BayesError::InvalidConfig(
                "deep GP fit requires at least 1 training point".into(),
            ));
        }
        if x.len() != n * self.d_in {
            return Err(BayesError::DimensionMismatch {
                expected: n * self.d_in,
                got: x.len(),
            });
        }
        if y.len() != n * self.d_out {
            return Err(BayesError::DimensionMismatch {
                expected: n * self.d_out,
                got: y.len(),
            });
        }

        let n_layers = self.layers.len();
        // Φ holds the current layer's input representation (mean-propagated).
        // Its row width equals `self.layers[idx].d_in` at iteration `idx`.
        let mut phi = x.to_vec();

        for idx in 0..n_layers {
            let is_output = idx == n_layers - 1;
            let m = self.layers[idx].n_inducing.min(n);
            let d_in = self.layers[idx].d_in;
            let d_out = self.layers[idx].d_out;

            // Re-place inducing inputs on this layer's data representation.
            let z = place_inducing(&phi, n, d_in, m);
            self.layers[idx].n_inducing = m;
            self.layers[idx].z = z;
            self.layers[idx].q_mu = vec![0.0_f64; d_out * m];
            self.layers[idx].s = self.layers[idx].k_mm();

            if is_output {
                self.fit_output_layer(idx, &phi, n, y)?;
            } else {
                // Mean-propagate Φ through the (identity-mean prior) hidden layer.
                let k_mm = self.layers[idx].k_mm();
                let l_mm = cholesky_jitter(&k_mm, m, self.jitter)?;
                let mut next = vec![0.0_f64; n * d_out];
                for i in 0..n {
                    let x_i = &phi[i * d_in..(i + 1) * d_in];
                    let (means, _var) = self.layers[idx].marginal(x_i, &l_mm);
                    next[i * d_out..(i + 1) * d_out].copy_from_slice(&means);
                }
                phi = next;
            }
        }
        Ok(())
    }

    /// Closed-form Titsias optimal `q(u) = N(μ_u, S)` for the output layer.
    fn fit_output_layer(
        &mut self,
        idx: usize,
        phi: &[f64],
        n: usize,
        y: &[f64],
    ) -> BayesResult<()> {
        let d_in = self.layers[idx].d_in;
        let d_out = self.layers[idx].d_out;
        let m = self.layers[idx].n_inducing;
        let sigma2 = self.noise_variance.max(1e-6);
        let inv_sigma2 = 1.0 / sigma2;

        let k_mm = self.layers[idx].k_mm();
        // K_nm: [n × m] with K_nm[i, a] = k(φ_i, z_a).
        let mut k_nm = vec![0.0_f64; n * m];
        for i in 0..n {
            let x_i = &phi[i * d_in..(i + 1) * d_in];
            for a in 0..m {
                let z_a = &self.layers[idx].z[a * d_in..(a + 1) * d_in];
                k_nm[i * m + a] = self.layers[idx].kernel.eval(x_i, z_a);
            }
        }

        // B = K_mm + σ⁻² K_mn K_nm   (m × m).
        let mut b_mat = k_mm.clone();
        for a in 0..m {
            for c in 0..m {
                let mut s = 0.0_f64;
                for i in 0..n {
                    s += k_nm[i * m + a] * k_nm[i * m + c];
                }
                b_mat[a * m + c] += inv_sigma2 * s;
            }
        }
        let l_b = cholesky_jitter(&b_mat, m, self.jitter)?;

        // Per output dim: μ_u_d = K_mm B⁻¹ (σ⁻² K_mn y_d).
        let mut q_mu = vec![0.0_f64; d_out * m];
        for d in 0..d_out {
            // rhs = σ⁻² K_mn y_d, with (K_mn y_d)[a] = Σ_i K_nm[i, a] y_{i,d}.
            let rhs: Vec<f64> = (0..m)
                .map(|a| {
                    let s: f64 = (0..n).map(|i| k_nm[i * m + a] * y[i * d_out + d]).sum();
                    inv_sigma2 * s
                })
                .collect();
            let b_inv_rhs = solve_chol(&l_b, &rhs, m);
            let mu_u = matvec(&k_mm, &b_inv_rhs, m);
            q_mu[d * m..(d + 1) * m].copy_from_slice(&mu_u);
        }

        // S = K_mm B⁻¹ K_mm.  Compute C = B⁻¹ K_mm column by column, then K_mm·C.
        let mut c_mat = vec![0.0_f64; m * m]; // C = B⁻¹ K_mm
        for col in 0..m {
            let kmm_col: Vec<f64> = (0..m).map(|r| k_mm[r * m + col]).collect();
            let solved = solve_chol(&l_b, &kmm_col, m);
            for r in 0..m {
                c_mat[r * m + col] = solved[r];
            }
        }
        // S = K_mm · C.
        let mut s_mat = vec![0.0_f64; m * m];
        for r in 0..m {
            for col in 0..m {
                let mut acc = 0.0_f64;
                for k in 0..m {
                    acc += k_mm[r * m + k] * c_mat[k * m + col];
                }
                s_mat[r * m + col] = acc;
            }
        }
        // Symmetrise to kill round-off asymmetry before later Cholesky.
        for r in 0..m {
            for col in (r + 1)..m {
                let avg = 0.5 * (s_mat[r * m + col] + s_mat[col * m + r]);
                s_mat[r * m + col] = avg;
                s_mat[col * m + r] = avg;
            }
        }

        self.layers[idx].q_mu = q_mu;
        self.layers[idx].s = s_mat;
        Ok(())
    }

    /// Deterministic posterior prediction: propagate by the marginal mean
    /// through hidden layers and return the output-layer marginal mean and
    /// variance.
    ///
    /// Returns `(means, variances)`, each a row-major `[n_new × d_out]` buffer
    /// (the variance is shared across output dimensions per point but is
    /// repeated for a regular shape).
    ///
    /// # Errors
    /// - [`BayesError::InvalidConfig`] — `n_new == 0`.
    /// - [`BayesError::DimensionMismatch`] — `x_new.len() != n_new·d_in`.
    /// - [`BayesError::SingularMatrix`] — a Cholesky fails.
    pub fn predict(&self, x_new: &[f64], n_new: usize) -> BayesResult<(Vec<f64>, Vec<f64>)> {
        if n_new == 0 {
            return Err(BayesError::InvalidConfig(
                "prediction requires at least 1 test point".into(),
            ));
        }
        if x_new.len() != n_new * self.d_in {
            return Err(BayesError::DimensionMismatch {
                expected: n_new * self.d_in,
                got: x_new.len(),
            });
        }

        let n_layers = self.layers.len();
        let mut phi = x_new.to_vec();
        let mut phi_dim = self.d_in;
        let mut last_var = vec![0.0_f64; n_new];

        for idx in 0..n_layers {
            let layer = &self.layers[idx];
            let m = layer.n_inducing;
            let d_out = layer.d_out;
            let l_mm = cholesky_jitter(&layer.k_mm(), m, self.jitter)?;
            let mut next = vec![0.0_f64; n_new * d_out];
            for i in 0..n_new {
                let x_i = &phi[i * phi_dim..(i + 1) * phi_dim];
                let (means, var) = layer.marginal(x_i, &l_mm);
                next[i * d_out..(i + 1) * d_out].copy_from_slice(&means);
                last_var[i] = var;
            }
            phi = next;
            phi_dim = d_out;
        }

        // Add observation noise to the output-layer marginal variance and
        // broadcast across output dims.
        let mut variances = vec![0.0_f64; n_new * self.d_out];
        for i in 0..n_new {
            let v = last_var[i] + self.noise_variance;
            for d in 0..self.d_out {
                variances[i * self.d_out + d] = v;
            }
        }
        Ok((phi, variances))
    }

    /// Doubly-stochastic forward pass: draw one sample per data point at every
    /// layer and propagate to the output.
    ///
    /// Returns `(samples, total_kl)` where `samples` is a row-major
    /// `[n × d_out]` buffer and `total_kl = Σ_l KL(q(u_l) ‖ p(u_l))`.
    ///
    /// # Errors
    /// - [`BayesError::InvalidConfig`] — `n == 0`.
    /// - [`BayesError::DimensionMismatch`] — `x.len() != n·d_in`.
    /// - [`BayesError::SingularMatrix`] — a Cholesky fails.
    pub fn forward_sample(
        &self,
        x: &[f64],
        n: usize,
        rng: &mut LcgRng,
    ) -> BayesResult<(Vec<f64>, f64)> {
        if n == 0 {
            return Err(BayesError::InvalidConfig(
                "forward_sample requires at least 1 input point".into(),
            ));
        }
        if x.len() != n * self.d_in {
            return Err(BayesError::DimensionMismatch {
                expected: n * self.d_in,
                got: x.len(),
            });
        }

        let mut phi = x.to_vec();
        let mut phi_dim = self.d_in;

        for layer in &self.layers {
            let m = layer.n_inducing;
            let d_out = layer.d_out;
            let l_mm = cholesky_jitter(&layer.k_mm(), m, self.jitter)?;
            let mut next = vec![0.0_f64; n * d_out];
            for i in 0..n {
                let x_i = &phi[i * phi_dim..(i + 1) * phi_dim];
                let (means, var) = layer.marginal(x_i, &l_mm);
                let std = var.sqrt();
                for d in 0..d_out {
                    next[i * d_out + d] = means[d] + std * next_normal(rng);
                }
            }
            phi = next;
            phi_dim = d_out;
        }

        Ok((phi, self.kl_divergence()?))
    }

    /// Total KL divergence `Σ_l KL(q(u_l) ‖ p(u_l))` across all layers.
    ///
    /// # Errors
    /// [`BayesError::SingularMatrix`] — a layer Cholesky fails.
    pub fn kl_divergence(&self) -> BayesResult<f64> {
        let mut total = 0.0_f64;
        for layer in &self.layers {
            total += layer.kl(self.jitter)?;
        }
        Ok(total)
    }

    /// Analytic evidence lower bound on `log p(y | x)`.
    ///
    /// Hidden layers are mean-propagated; the output-layer expected
    /// log-likelihood uses the SVGP closed form
    /// `E_q[log N(y | f, σ²)] = log N(y | μ, σ²) − s²/(2σ²)` (Hensman 2013),
    /// minus the summed per-layer KL.
    ///
    /// # Errors
    /// - [`BayesError::InvalidConfig`] — `n == 0`.
    /// - [`BayesError::DimensionMismatch`] — `x`/`y` length mismatch.
    /// - [`BayesError::SingularMatrix`] — a Cholesky fails.
    pub fn elbo(&self, x: &[f64], y: &[f64], n: usize) -> BayesResult<f64> {
        if n == 0 {
            return Err(BayesError::InvalidConfig(
                "elbo requires at least 1 data point".into(),
            ));
        }
        if x.len() != n * self.d_in {
            return Err(BayesError::DimensionMismatch {
                expected: n * self.d_in,
                got: x.len(),
            });
        }
        if y.len() != n * self.d_out {
            return Err(BayesError::DimensionMismatch {
                expected: n * self.d_out,
                got: y.len(),
            });
        }

        let n_layers = self.layers.len();
        let sigma2 = self.noise_variance.max(1e-6);

        // Mean-propagate through hidden layers to the output-layer input.
        let mut phi = x.to_vec();
        let mut phi_dim = self.d_in;
        for layer in self.layers.iter().take(n_layers - 1) {
            let m = layer.n_inducing;
            let d_out = layer.d_out;
            let l_mm = cholesky_jitter(&layer.k_mm(), m, self.jitter)?;
            let mut next = vec![0.0_f64; n * d_out];
            for i in 0..n {
                let x_i = &phi[i * phi_dim..(i + 1) * phi_dim];
                let (means, _var) = layer.marginal(x_i, &l_mm);
                next[i * d_out..(i + 1) * d_out].copy_from_slice(&means);
            }
            phi = next;
            phi_dim = d_out;
        }

        // Output-layer expected log-likelihood.
        let out = &self.layers[n_layers - 1];
        let m = out.n_inducing;
        let l_mm = cholesky_jitter(&out.k_mm(), m, self.jitter)?;
        let half_log_2pi_sig = 0.5 * (2.0 * std::f64::consts::PI * sigma2).ln();
        let mut ell = 0.0_f64;
        for i in 0..n {
            let x_i = &phi[i * phi_dim..(i + 1) * phi_dim];
            let (means, var) = out.marginal(x_i, &l_mm);
            for d in 0..self.d_out {
                let resid = y[i * self.d_out + d] - means[d];
                ell += -half_log_2pi_sig - (resid * resid) / (2.0 * sigma2) - var / (2.0 * sigma2);
            }
        }

        let kl = self.kl_divergence()?;
        Ok(ell - kl)
    }
}

/// Validate that a kernel's hyper-parameters are strictly positive.
fn validate_kernel(kernel: &GprKernel) -> BayesResult<()> {
    let ok = match kernel {
        GprKernel::Rbf {
            length_scale,
            signal_variance,
        }
        | GprKernel::Matern32 {
            length_scale,
            signal_variance,
        }
        | GprKernel::Matern52 {
            length_scale,
            signal_variance,
        } => *length_scale > 0.0 && *signal_variance > 0.0,
        GprKernel::Periodic {
            length_scale,
            period,
            signal_variance,
        } => *length_scale > 0.0 && *period > 0.0 && *signal_variance > 0.0,
    };
    if ok {
        Ok(())
    } else {
        Err(BayesError::InvalidConfig(
            "deep GP kernel hyper-parameters must be positive".into(),
        ))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp::gpr::{GprConfig, gpr_fit, gpr_predict};

    fn rbf(length_scale: f64, signal_variance: f64) -> GprKernel {
        GprKernel::Rbf {
            length_scale,
            signal_variance,
        }
    }

    /// Deterministic 1-D sin data on `[0, 1]`.
    fn sin_data(n: usize) -> (Vec<f64>, Vec<f64>) {
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| (2.0 * std::f64::consts::PI * x).sin())
            .collect();
        (xs, ys)
    }

    fn one_layer_config(n_inducing: usize, length_scale: f64) -> DeepGpConfig {
        DeepGpConfig {
            layers: vec![DeepGpLayerConfig {
                d_in: 1,
                d_out: 1,
                n_inducing,
                kernel: rbf(length_scale, 1.0),
            }],
            noise_variance: 1e-3,
            jitter: 1e-6,
        }
    }

    fn two_layer_config() -> DeepGpConfig {
        DeepGpConfig {
            layers: vec![
                DeepGpLayerConfig {
                    d_in: 1,
                    d_out: 1,
                    n_inducing: 5,
                    kernel: rbf(0.3, 1.0),
                },
                DeepGpLayerConfig {
                    d_in: 1,
                    d_out: 1,
                    n_inducing: 5,
                    kernel: rbf(0.3, 1.0),
                },
            ],
            noise_variance: 1e-3,
            jitter: 1e-6,
        }
    }

    // (a) predict shapes & finiteness ─────────────────────────────────────────

    #[test]
    fn predict_shapes_and_finite() {
        let mut rng = LcgRng::new(1);
        let (xs, ys) = sin_data(20);
        let mut dgp = DeepGp::new(one_layer_config(8, 0.2), &mut rng).expect("new");
        dgp.fit(&xs, &ys, 20).expect("fit");
        let x_test: Vec<f64> = (0..7).map(|i| i as f64 / 6.0).collect();
        let (means, vars) = dgp.predict(&x_test, 7).expect("predict");
        assert_eq!(means.len(), 7);
        assert_eq!(vars.len(), 7);
        for (&mn, &vr) in means.iter().zip(vars.iter()) {
            assert!(mn.is_finite(), "mean must be finite: {mn}");
            assert!(
                vr.is_finite() && vr >= 0.0,
                "var must be finite & >= 0: {vr}"
            );
        }
    }

    #[test]
    fn two_layer_predict_finite() {
        let mut rng = LcgRng::new(2);
        let (xs, ys) = sin_data(16);
        let mut dgp = DeepGp::new(two_layer_config(), &mut rng).expect("new");
        dgp.fit(&xs, &ys, 16).expect("fit");
        let x_test = vec![0.1_f64, 0.5, 0.9];
        let (means, vars) = dgp.predict(&x_test, 3).expect("predict");
        for (&mn, &vr) in means.iter().zip(vars.iter()) {
            assert!(mn.is_finite() && vr.is_finite());
        }
    }

    // (b) ELBO finite; KL terms >= 0 ──────────────────────────────────────────

    #[test]
    fn elbo_finite_and_kl_non_negative() {
        let mut rng = LcgRng::new(3);
        let (xs, ys) = sin_data(18);
        let mut dgp = DeepGp::new(one_layer_config(6, 0.2), &mut rng).expect("new");
        dgp.fit(&xs, &ys, 18).expect("fit");
        let elbo = dgp.elbo(&xs, &ys, 18).expect("elbo");
        assert!(elbo.is_finite(), "ELBO must be finite, got {elbo}");
        let kl = dgp.kl_divergence().expect("kl");
        assert!(kl >= 0.0, "total KL must be >= 0, got {kl}");
        for layer in &dgp.layers {
            assert!(layer.kl(dgp.jitter).expect("layer kl") >= 0.0);
        }
    }

    #[test]
    fn two_layer_hidden_kl_zero_output_kl_positive() {
        let mut rng = LcgRng::new(4);
        let (xs, ys) = sin_data(16);
        let mut dgp = DeepGp::new(two_layer_config(), &mut rng).expect("new");
        dgp.fit(&xs, &ys, 16).expect("fit");
        // Hidden layer keeps q = p ⇒ KL ≈ 0.
        let hidden_kl = dgp.layers[0].kl(dgp.jitter).expect("kl");
        assert!(hidden_kl < 1e-6, "hidden KL should be ~0, got {hidden_kl}");
        // Output layer is fitted ⇒ KL > 0.
        let out_kl = dgp.layers[1].kl(dgp.jitter).expect("kl");
        assert!(out_kl > 0.0, "output KL should be > 0, got {out_kl}");
    }

    // (c) 1-layer DGP ≈ sparse / exact GP ─────────────────────────────────────

    #[test]
    fn one_layer_reduces_to_gp_on_smooth_data() {
        // With m == n inducing points and small noise, the Titsias-fit single
        // layer should closely match an exact GP regressor.
        let mut rng = LcgRng::new(5);
        let n = 24;
        let (xs, ys) = sin_data(n);
        let length_scale = 0.2;
        let mut dgp = DeepGp::new(one_layer_config(n, length_scale), &mut rng).expect("new");
        dgp.fit(&xs, &ys, n).expect("fit");

        let exact = gpr_fit(
            &xs,
            &ys,
            n,
            1,
            &GprConfig {
                kernel: rbf(length_scale, 1.0),
                noise_variance: 1e-3,
                normalize_y: false,
                jitter: 1e-6,
            },
        )
        .expect("gpr fit");

        let x_test: Vec<f64> = (0..12).map(|i| i as f64 / 11.0).collect();
        let (dgp_means, _) = dgp.predict(&x_test, 12).expect("dgp predict");
        let (exact_means, _) = gpr_predict(&exact, &x_test, 12, false).expect("gpr predict");

        let max_diff = dgp_means
            .iter()
            .zip(exact_means.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 0.1,
            "1-layer DGP vs exact GP max |Δμ| = {max_diff:.4} (should be < 0.1)"
        );
    }

    // (d) propagating through layers preserves dimensionality ──────────────────

    #[test]
    fn forward_sample_preserves_dimensionality() {
        let mut rng = LcgRng::new(6);
        // 2-D hidden(2→2) + output(2→2): every stage stays 2-D.
        let config = DeepGpConfig {
            layers: vec![
                DeepGpLayerConfig {
                    d_in: 2,
                    d_out: 2,
                    n_inducing: 4,
                    kernel: rbf(0.5, 1.0),
                },
                DeepGpLayerConfig {
                    d_in: 2,
                    d_out: 2,
                    n_inducing: 4,
                    kernel: rbf(0.5, 1.0),
                },
            ],
            noise_variance: 1e-2,
            jitter: 1e-6,
        };
        let dgp = DeepGp::new(config, &mut rng).expect("new");
        let n = 5;
        let x: Vec<f64> = (0..n * 2).map(|i| (i as f64) * 0.1 - 0.4).collect();
        let (samples, kl) = dgp.forward_sample(&x, n, &mut rng).expect("forward");
        assert_eq!(samples.len(), n * 2, "output must be [n × 2]");
        assert!(samples.iter().all(|v| v.is_finite()));
        assert!(kl >= 0.0);
    }

    // (e) more inducing points → ELBO data-fit improves (loose) ────────────────

    #[test]
    fn more_inducing_improves_elbo() {
        let mut rng = LcgRng::new(7);
        let n = 40;
        let (xs, ys) = sin_data(n);
        let length_scale = 0.15;

        let mut dgp_few = DeepGp::new(one_layer_config(3, length_scale), &mut rng).expect("new");
        dgp_few.fit(&xs, &ys, n).expect("fit few");
        let elbo_few = dgp_few.elbo(&xs, &ys, n).expect("elbo few");

        let mut dgp_many = DeepGp::new(one_layer_config(12, length_scale), &mut rng).expect("new");
        dgp_many.fit(&xs, &ys, n).expect("fit many");
        let elbo_many = dgp_many.elbo(&xs, &ys, n).expect("elbo many");

        assert!(
            elbo_many >= elbo_few - 1e-3,
            "more inducing should not lower the ELBO: few={elbo_few:.4}, many={elbo_many:.4}"
        );
    }

    // (f) dim / shape errors ──────────────────────────────────────────────────

    #[test]
    fn new_fails_empty_layers() {
        let mut rng = LcgRng::new(8);
        let config = DeepGpConfig {
            layers: vec![],
            noise_variance: 1e-3,
            jitter: 1e-6,
        };
        assert!(matches!(
            DeepGp::new(config, &mut rng),
            Err(BayesError::InvalidConfig(_))
        ));
    }

    #[test]
    fn new_fails_dimension_chain_mismatch() {
        let mut rng = LcgRng::new(9);
        let config = DeepGpConfig {
            layers: vec![
                DeepGpLayerConfig {
                    d_in: 2,
                    d_out: 2,
                    n_inducing: 3,
                    kernel: rbf(0.5, 1.0),
                },
                DeepGpLayerConfig {
                    d_in: 3, // != previous d_out (2)
                    d_out: 1,
                    n_inducing: 3,
                    kernel: rbf(0.5, 1.0),
                },
            ],
            noise_variance: 1e-3,
            jitter: 1e-6,
        };
        assert!(matches!(
            DeepGp::new(config, &mut rng),
            Err(BayesError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn new_fails_hidden_non_square() {
        let mut rng = LcgRng::new(10);
        let config = DeepGpConfig {
            layers: vec![
                DeepGpLayerConfig {
                    d_in: 2,
                    d_out: 3, // hidden layer changes dim ⇒ identity mean undefined
                    n_inducing: 3,
                    kernel: rbf(0.5, 1.0),
                },
                DeepGpLayerConfig {
                    d_in: 3,
                    d_out: 1,
                    n_inducing: 3,
                    kernel: rbf(0.5, 1.0),
                },
            ],
            noise_variance: 1e-3,
            jitter: 1e-6,
        };
        assert!(matches!(
            DeepGp::new(config, &mut rng),
            Err(BayesError::InvalidConfig(_))
        ));
    }

    #[test]
    fn fit_dimension_mismatch_on_x() {
        let mut rng = LcgRng::new(11);
        let (_xs, ys) = sin_data(10);
        let mut dgp = DeepGp::new(one_layer_config(4, 0.2), &mut rng).expect("new");
        let bad_x = vec![0.0_f64; 9]; // not 10 * d_in
        assert!(matches!(
            dgp.fit(&bad_x, &ys, 10),
            Err(BayesError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn predict_zero_n_new_error() {
        let mut rng = LcgRng::new(12);
        let (xs, ys) = sin_data(10);
        let mut dgp = DeepGp::new(one_layer_config(4, 0.2), &mut rng).expect("new");
        dgp.fit(&xs, &ys, 10).expect("fit");
        assert!(matches!(
            dgp.predict(&[], 0),
            Err(BayesError::InvalidConfig(_))
        ));
    }

    #[test]
    fn forward_sample_determinism_same_seed() {
        let mut rng = LcgRng::new(13);
        let (xs, ys) = sin_data(12);
        let mut dgp = DeepGp::new(one_layer_config(5, 0.2), &mut rng).expect("new");
        dgp.fit(&xs, &ys, 12).expect("fit");
        let x_test = vec![0.2_f64, 0.5, 0.8];
        let (s1, _) = dgp
            .forward_sample(&x_test, 3, &mut LcgRng::new(99))
            .expect("fwd");
        let (s2, _) = dgp
            .forward_sample(&x_test, 3, &mut LcgRng::new(99))
            .expect("fwd");
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert!((a - b).abs() < 1e-12, "same seed must be deterministic");
        }
    }
}
