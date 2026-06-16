//! CTGAN: Conditional Tabular GAN.
//!
//! Reference: Xu, Skoularidou, Cuesta-Infante & Veeramachaneni (2019),
//! "Modeling Tabular Data using Conditional GAN", NeurIPS 2019.
//!
//! This is a CPU **reference scaffold**: the networks are randomly initialised
//! and the public API supports forward passes, generation, and a *few*
//! non-converging training micro-steps.  It is not a full end-to-end trainer
//! (mirroring the [`TabDdpm`](crate::diffusion::tabddpm) scaffold in this crate).
//!
//! # Components
//!
//! - **Mode-specific normalisation** ([`ModeNormalizer`]).  Real CTGAN fits a
//!   variational Gaussian mixture (VGM) per continuous column.  Here we use a
//!   lightweight stand-in: each continuous column is clustered into `K` modes by
//!   1-D k-means; a value `v` is encoded as `(mode_one_hot, α)` where
//!   `α = clamp((v − μ_m) / (4·σ_m), −1, 1)` for its assigned mode `m`.  This is
//!   a deliberate simplification of the VGM — it captures the multimodal,
//!   per-mode standardised representation without the full Bayesian mixture.
//! - **Conditional vector** ([`ConditionalSampler`]).  Concatenated one-hot
//!   masks over the discrete columns; training-by-sampling selects a column and
//!   then a category with **log-frequency** weighting (`p ∝ log(count + 1)`),
//!   which up-weights rare categories relative to raw frequency.
//! - **Generator**.  Two residual blocks (`out = concat(in, ReLU(BN-lite(Wx)))`)
//!   followed by a linear head producing the row representation; continuous
//!   scalars use `tanh`, while mode one-hots and categorical logits use the
//!   **Gumbel-softmax** relaxation.
//! - **PacGAN discriminator**.  `pac` rows are packed together before the
//!   discriminator MLP (leaky-ReLU), yielding one score per pack.
//!
//! The generator's "BN-lite" is a per-sample standardisation standing in for the
//! batch-norm used in the paper (single-row forward has no batch statistics).

use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;
use crate::nn::{Dense, leaky_relu, relu, softmax};

/// Cached generator forward output for one row:
/// `(row, head_input, conditioned_column, conditioned_category)`.
type GeneratedRow = (Vec<f32>, Vec<f32>, Option<usize>, Option<usize>);

// ─── Mode-specific normalisation ───────────────────────────────────────────────

/// Per-column mode parameters (mean and standard deviation of each mode).
#[derive(Debug, Clone)]
pub struct ColumnModes {
    /// Mode means (k-means centroids), length `n_modes`.
    pub means: Vec<f32>,
    /// Per-mode standard deviations, length `n_modes` (floored to `1e-3`).
    pub stds: Vec<f32>,
}

/// Mode-specific normaliser for the continuous columns of a table.
///
/// Stands in for CTGAN's variational Gaussian mixture: each continuous column is
/// clustered into `n_modes` modes via 1-D k-means.
#[derive(Debug, Clone)]
pub struct ModeNormalizer {
    columns: Vec<ColumnModes>,
    n_modes: usize,
}

impl ModeNormalizer {
    /// Fit the normaliser from a flat row-major `[n_samples × n_continuous]`
    /// matrix.
    ///
    /// # Errors
    /// - [`TabularError::EmptyInput`] if `n_samples == 0` or `n_continuous == 0`.
    /// - [`TabularError::InvalidParameter`] if `n_modes == 0`.
    /// - [`TabularError::DimensionMismatch`] if the buffer length is wrong.
    pub fn fit(
        data: &[f32],
        n_samples: usize,
        n_continuous: usize,
        n_modes: usize,
    ) -> TabularResult<Self> {
        if n_samples == 0 || n_continuous == 0 {
            return Err(TabularError::EmptyInput);
        }
        if n_modes == 0 {
            return Err(TabularError::InvalidParameter {
                name: "n_modes".into(),
                msg: "must be ≥ 1".into(),
            });
        }
        if data.len() != n_samples * n_continuous {
            return Err(TabularError::DimensionMismatch {
                expected: n_samples * n_continuous,
                got: data.len(),
            });
        }

        let mut columns = Vec::with_capacity(n_continuous);
        for col in 0..n_continuous {
            let vals: Vec<f32> = (0..n_samples)
                .map(|r| data[r * n_continuous + col])
                .collect();
            let (means, stds) = kmeans_1d(&vals, n_modes, 16);
            columns.push(ColumnModes { means, stds });
        }
        Ok(Self { columns, n_modes })
    }

    /// Number of modes per column.
    #[must_use]
    pub fn n_modes(&self) -> usize {
        self.n_modes
    }

    /// Number of continuous columns.
    #[must_use]
    pub fn n_columns(&self) -> usize {
        self.columns.len()
    }

    /// Encode a value of continuous column `col` as `(mode_index, α)`.
    ///
    /// `α = clamp((v − μ_m) / (4·σ_m), −1, 1)` for the nearest mode `m`.
    ///
    /// # Errors
    /// [`TabularError::InvalidParameter`] if `col` is out of range.
    pub fn transform_value(&self, col: usize, value: f32) -> TabularResult<(usize, f32)> {
        let cm = self
            .columns
            .get(col)
            .ok_or_else(|| TabularError::InvalidParameter {
                name: "col".into(),
                msg: format!("column {col} out of range"),
            })?;
        let mode = nearest_mode(&cm.means, value);
        let alpha = ((value - cm.means[mode]) / (4.0 * cm.stds[mode])).clamp(-1.0, 1.0);
        Ok((mode, alpha))
    }

    /// Decode `(mode_index, α)` back into a value of continuous column `col`.
    ///
    /// Inverse of [`transform_value`](Self::transform_value) (exact when `α` was
    /// not clamped).
    ///
    /// # Errors
    /// - [`TabularError::InvalidParameter`] if `col` is out of range.
    /// - [`TabularError::CategoricalOutOfRange`] if `mode` is out of range.
    pub fn inverse_value(&self, col: usize, mode: usize, alpha: f32) -> TabularResult<f32> {
        let cm = self
            .columns
            .get(col)
            .ok_or_else(|| TabularError::InvalidParameter {
                name: "col".into(),
                msg: format!("column {col} out of range"),
            })?;
        if mode >= cm.means.len() {
            return Err(TabularError::CategoricalOutOfRange {
                feat: col,
                val: mode,
                n: cm.means.len(),
            });
        }
        Ok(alpha * 4.0 * cm.stds[mode] + cm.means[mode])
    }
}

/// One-dimensional k-means (Lloyd's algorithm) returning `(means, stds)`.
///
/// Centroids are initialised evenly across `[min, max]` for determinism; empty
/// clusters retain their centroid and receive a floored standard deviation.
fn kmeans_1d(values: &[f32], k: usize, iters: usize) -> (Vec<f32>, Vec<f32>) {
    let n = values.len();
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in values {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    if !min.is_finite() || !max.is_finite() {
        return (vec![0.0; k], vec![1.0; k]);
    }
    let span = (max - min).max(1e-6);

    let mut means: Vec<f32> = (0..k)
        .map(|j| {
            if k == 1 {
                0.5 * (min + max)
            } else {
                min + span * (j as f32) / ((k - 1) as f32)
            }
        })
        .collect();

    let mut assign = vec![0usize; n];
    for _ in 0..iters {
        for (i, &v) in values.iter().enumerate() {
            assign[i] = nearest_mode(&means, v);
        }
        let mut sums = vec![0.0_f32; k];
        let mut counts = vec![0usize; k];
        for (&a, &v) in assign.iter().zip(values.iter()) {
            sums[a] += v;
            counts[a] += 1;
        }
        for (m, (&s, &c)) in means.iter_mut().zip(sums.iter().zip(counts.iter())) {
            if c > 0 {
                *m = s / c as f32;
            }
        }
    }

    for (i, &v) in values.iter().enumerate() {
        assign[i] = nearest_mode(&means, v);
    }
    let mut var = vec![0.0_f32; k];
    let mut counts = vec![0usize; k];
    for (&a, &v) in assign.iter().zip(values.iter()) {
        let d = v - means[a];
        var[a] += d * d;
        counts[a] += 1;
    }
    let stds = var
        .iter()
        .zip(counts.iter())
        .map(|(&s, &c)| {
            if c > 0 {
                (s / c as f32).sqrt().max(1e-3)
            } else {
                1e-3
            }
        })
        .collect();

    (means, stds)
}

/// Index of the nearest mean to `value`.
fn nearest_mode(means: &[f32], value: f32) -> usize {
    means
        .iter()
        .enumerate()
        .min_by(|&(_, &a), &(_, &b)| {
            (a - value)
                .abs()
                .partial_cmp(&(b - value).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

// ─── Conditional sampler ───────────────────────────────────────────────────────

/// Conditional-vector sampler over the discrete columns.
///
/// Holds, per discrete column, a categorical sampling distribution and the
/// offset of its one-hot block inside the concatenated conditional vector.
#[derive(Debug, Clone)]
pub struct ConditionalSampler {
    cardinalities: Vec<usize>,
    offsets: Vec<usize>,
    cond_dim: usize,
    probs: Vec<Vec<f32>>,
}

impl ConditionalSampler {
    fn build(cardinalities: Vec<usize>, probs: Vec<Vec<f32>>) -> Self {
        let mut offsets = Vec::with_capacity(cardinalities.len());
        let mut acc = 0usize;
        for &c in &cardinalities {
            offsets.push(acc);
            acc += c;
        }
        Self {
            cardinalities,
            offsets,
            cond_dim: acc,
            probs,
        }
    }

    /// Build a sampler with a uniform distribution over each column's categories.
    ///
    /// # Errors
    /// [`TabularError::InvalidParameter`] if any cardinality is zero.
    pub fn uniform(cardinalities: &[usize]) -> TabularResult<Self> {
        if cardinalities.contains(&0) {
            return Err(TabularError::InvalidParameter {
                name: "cardinalities".into(),
                msg: "every discrete column must have cardinality ≥ 1".into(),
            });
        }
        let probs = cardinalities
            .iter()
            .map(|&c| vec![1.0 / c as f32; c])
            .collect();
        Ok(Self::build(cardinalities.to_vec(), probs))
    }

    /// Fit per-column **log-frequency** sampling distributions from observed
    /// categorical data laid out row-major `[n_samples × n_columns]`.
    ///
    /// `p(category) ∝ log(count + 1)`.
    ///
    /// # Errors
    /// - [`TabularError::InvalidParameter`] if any cardinality is zero.
    /// - [`TabularError::DimensionMismatch`] if the buffer length is wrong.
    /// - [`TabularError::CategoricalOutOfRange`] for an out-of-range category.
    pub fn fit(
        cat_data: &[usize],
        n_samples: usize,
        cardinalities: &[usize],
    ) -> TabularResult<Self> {
        let n_cols = cardinalities.len();
        if cardinalities.contains(&0) {
            return Err(TabularError::InvalidParameter {
                name: "cardinalities".into(),
                msg: "every discrete column must have cardinality ≥ 1".into(),
            });
        }
        if cat_data.len() != n_samples * n_cols {
            return Err(TabularError::DimensionMismatch {
                expected: n_samples * n_cols,
                got: cat_data.len(),
            });
        }

        let mut probs: Vec<Vec<f32>> = cardinalities.iter().map(|&c| vec![0.0_f32; c]).collect();
        for r in 0..n_samples {
            for col in 0..n_cols {
                let v = cat_data[r * n_cols + col];
                if v >= cardinalities[col] {
                    return Err(TabularError::CategoricalOutOfRange {
                        feat: col,
                        val: v,
                        n: cardinalities[col],
                    });
                }
                probs[col][v] += 1.0;
            }
        }
        for (col, pv) in probs.iter_mut().enumerate() {
            for p in pv.iter_mut() {
                *p = (*p + 1.0).ln();
            }
            let sum: f32 = pv.iter().sum();
            if sum > 0.0 {
                for p in pv.iter_mut() {
                    *p /= sum;
                }
            } else {
                let u = 1.0 / cardinalities[col] as f32;
                for p in pv.iter_mut() {
                    *p = u;
                }
            }
        }
        Ok(Self::build(cardinalities.to_vec(), probs))
    }

    /// Conditional-vector dimension `Σ cardinalities`.
    #[must_use]
    pub fn cond_dim(&self) -> usize {
        self.cond_dim
    }

    /// Number of discrete columns.
    #[must_use]
    pub fn n_columns(&self) -> usize {
        self.cardinalities.len()
    }

    /// Sampling probabilities for the categories of column `col`.
    ///
    /// # Errors
    /// [`TabularError::InvalidParameter`] if `col` is out of range.
    pub fn category_probabilities(&self, col: usize) -> TabularResult<Vec<f32>> {
        self.probs
            .get(col)
            .cloned()
            .ok_or_else(|| TabularError::InvalidParameter {
                name: "col".into(),
                msg: format!("column {col} out of range"),
            })
    }

    /// Sample a `(column, category)` condition, or `None` if there are no
    /// discrete columns.
    pub fn sample(&self, rng: &mut LcgRng) -> Option<(usize, usize)> {
        let n_cols = self.cardinalities.len();
        if n_cols == 0 {
            return None;
        }
        let col = rng.next_usize(n_cols);
        let cat = sample_categorical(&self.probs[col], rng);
        Some((col, cat))
    }

    /// Build the conditional one-hot vector for `(col, cat)`.
    ///
    /// # Errors
    /// - [`TabularError::InvalidParameter`] if `col` is out of range.
    /// - [`TabularError::CategoricalOutOfRange`] if `cat` is out of range.
    pub fn cond_vector(&self, col: usize, cat: usize) -> TabularResult<Vec<f32>> {
        let card = *self
            .cardinalities
            .get(col)
            .ok_or_else(|| TabularError::InvalidParameter {
                name: "col".into(),
                msg: format!("column {col} out of range"),
            })?;
        if cat >= card {
            return Err(TabularError::CategoricalOutOfRange {
                feat: col,
                val: cat,
                n: card,
            });
        }
        let mut v = vec![0.0_f32; self.cond_dim];
        v[self.offsets[col] + cat] = 1.0;
        Ok(v)
    }
}

/// Sample a category index from a probability vector via inverse-CDF.
fn sample_categorical(probs: &[f32], rng: &mut LcgRng) -> usize {
    let r = rng.next_f32();
    let mut acc = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r < acc {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for a [`CtGan`].
#[derive(Debug, Clone)]
pub struct CtganConfig {
    /// Number of continuous columns.
    pub n_continuous: usize,
    /// Cardinality of each categorical column.
    pub categorical_cardinalities: Vec<usize>,
    /// Number of modes per continuous column (VGM stand-in, `K`).
    pub n_modes: usize,
    /// Generator noise (latent) dimension.
    pub noise_dim: usize,
    /// Generator residual-block width.
    pub generator_dim: usize,
    /// Discriminator hidden width.
    pub discriminator_dim: usize,
    /// PacGAN pack size.
    pub pac: usize,
    /// Gumbel-softmax temperature `τ` (must be positive).
    pub gumbel_tau: f32,
}

// ─── CtGan ─────────────────────────────────────────────────────────────────────

/// Conditional Tabular GAN (CPU reference scaffold).
pub struct CtGan {
    /// PacGAN pack size.
    pub pac: usize,
    config: CtganConfig,
    /// Data-row width: `n_continuous·(1 + n_modes) + Σ cardinalities`.
    data_dim: usize,
    /// Conditional-vector width: `Σ cardinalities`.
    cond_dim: usize,
    /// Width of the continuous part of the row: `n_continuous·(1 + n_modes)`.
    cont_block: usize,
    /// Generator input width: `noise_dim + cond_dim`.
    gen_input: usize,
    cond_sampler: ConditionalSampler,
    gen_block1: Dense,
    gen_block2: Dense,
    gen_head: Dense,
    disc1: Dense,
    disc2: Dense,
    disc_head: Dense,
}

impl CtGan {
    /// Construct a new CTGAN scaffold with randomly-initialised weights.
    ///
    /// # Errors
    /// Returns a [`TabularError`] variant if the resulting data width is zero, or
    /// if any of `noise_dim`, `generator_dim`, `discriminator_dim`, `pac`,
    /// `gumbel_tau`, the categorical cardinalities, or `n_modes` (when there are
    /// continuous columns) is invalid.
    pub fn new(config: CtganConfig, rng: &mut LcgRng) -> TabularResult<Self> {
        let cat_sum: usize = config.categorical_cardinalities.iter().sum();
        let cont_block = config.n_continuous * (1 + config.n_modes);
        let data_dim = cont_block + cat_sum;
        if data_dim == 0 {
            return Err(TabularError::InvalidFeatureCount { n: 0 });
        }
        if config.n_continuous > 0 && config.n_modes == 0 {
            return Err(TabularError::InvalidParameter {
                name: "n_modes".into(),
                msg: "must be ≥ 1 when there are continuous columns".into(),
            });
        }
        if config.categorical_cardinalities.contains(&0) {
            return Err(TabularError::InvalidParameter {
                name: "categorical_cardinalities".into(),
                msg: "every categorical column must have cardinality ≥ 1".into(),
            });
        }
        if config.noise_dim == 0 {
            return Err(TabularError::InvalidEmbedDim { dim: 0 });
        }
        if config.generator_dim == 0 {
            return Err(TabularError::InvalidParameter {
                name: "generator_dim".into(),
                msg: "must be ≥ 1".into(),
            });
        }
        if config.discriminator_dim == 0 {
            return Err(TabularError::InvalidParameter {
                name: "discriminator_dim".into(),
                msg: "must be ≥ 1".into(),
            });
        }
        if config.pac == 0 {
            return Err(TabularError::InvalidParameter {
                name: "pac".into(),
                msg: "must be ≥ 1".into(),
            });
        }
        if config.gumbel_tau <= 0.0 || !config.gumbel_tau.is_finite() {
            return Err(TabularError::InvalidParameter {
                name: "gumbel_tau".into(),
                msg: "must be a positive, finite value".into(),
            });
        }

        let cond_dim = cat_sum;
        let gen_input = config.noise_dim + cond_dim;
        let cond_sampler = ConditionalSampler::uniform(&config.categorical_cardinalities)?;

        let gen_block1 = Dense::new(gen_input, config.generator_dim, rng);
        let gen_block2 = Dense::new(gen_input + config.generator_dim, config.generator_dim, rng);
        let gen_head = Dense::new(gen_input + 2 * config.generator_dim, data_dim, rng);
        let disc1 = Dense::new(config.pac * data_dim, config.discriminator_dim, rng);
        let disc2 = Dense::new(config.discriminator_dim, config.discriminator_dim, rng);
        let disc_head = Dense::new(config.discriminator_dim, 1, rng);

        Ok(Self {
            pac: config.pac,
            config,
            data_dim,
            cond_dim,
            cont_block,
            gen_input,
            cond_sampler,
            gen_block1,
            gen_block2,
            gen_head,
            disc1,
            disc2,
            disc_head,
        })
    }

    /// Data-row width.
    #[must_use]
    pub fn data_dim(&self) -> usize {
        self.data_dim
    }

    /// Conditional-vector width.
    #[must_use]
    pub fn cond_dim(&self) -> usize {
        self.cond_dim
    }

    /// Number of packs produced from `n_rows` rows.
    #[must_use]
    pub fn n_packs(&self, n_rows: usize) -> usize {
        n_rows / self.pac
    }

    /// Reference to the internal conditional sampler.
    #[must_use]
    pub fn conditional_sampler(&self) -> &ConditionalSampler {
        &self.cond_sampler
    }

    /// Replace the conditional sampler with one fitted to observed categorical
    /// data (log-frequency weighting).
    ///
    /// # Errors
    /// Propagated from [`ConditionalSampler::fit`]; also
    /// [`TabularError::DimensionMismatch`] if the column count disagrees with the
    /// configuration.
    pub fn fit_conditional(&mut self, cat_data: &[usize], n_samples: usize) -> TabularResult<()> {
        let sampler =
            ConditionalSampler::fit(cat_data, n_samples, &self.config.categorical_cardinalities)?;
        self.cond_sampler = sampler;
        Ok(())
    }

    /// Slice `(start, cardinality)` of categorical column `col` inside the row.
    fn categorical_slice(&self, col: usize) -> (usize, usize) {
        let mut start = self.cont_block;
        for &card in self.config.categorical_cardinalities.iter().take(col) {
            start += card;
        }
        let card = self
            .config
            .categorical_cardinalities
            .get(col)
            .copied()
            .unwrap_or(0);
        (start, card)
    }

    /// Activate a raw generator-head output into a data row: `tanh` for the
    /// continuous scalars, Gumbel-softmax for mode one-hots and categorical
    /// blocks.
    fn activate_row(&self, o: &[f32], rng: &mut LcgRng) -> Vec<f32> {
        let mut row = vec![0.0_f32; self.data_dim];
        let mut pos = 0usize;
        let tau = self.config.gumbel_tau;
        for _ in 0..self.config.n_continuous {
            row[pos] = o[pos].tanh();
            pos += 1;
            let modes = gumbel_softmax(&o[pos..pos + self.config.n_modes], tau, rng);
            row[pos..pos + self.config.n_modes].copy_from_slice(&modes);
            pos += self.config.n_modes;
        }
        for &card in &self.config.categorical_cardinalities {
            let probs = gumbel_softmax(&o[pos..pos + card], tau, rng);
            row[pos..pos + card].copy_from_slice(&probs);
            pos += card;
        }
        row
    }

    /// Generate one row, returning `(row, head_input, cond_col, cond_cat)`.
    ///
    /// The head input and the sampled condition are cached so the training steps
    /// can compute the conditional-matching gradient.
    fn generate_row(&self, rng: &mut LcgRng) -> TabularResult<GeneratedRow> {
        let mut z = vec![0.0_f32; self.config.noise_dim];
        rng.fill_normal(&mut z);

        let cond_opt = self.cond_sampler.sample(rng);
        let cond = match cond_opt {
            Some((c, k)) => self.cond_sampler.cond_vector(c, k)?,
            None => Vec::new(),
        };

        let mut inp = z;
        inp.extend_from_slice(&cond);

        // Residual block 1.
        let pre1 = self.gen_block1.forward(&inp);
        let h1: Vec<f32> = bn_lite(&pre1).into_iter().map(relu).collect();
        let mut r1 = inp;
        r1.extend_from_slice(&h1);

        // Residual block 2.
        let pre2 = self.gen_block2.forward(&r1);
        let h2: Vec<f32> = bn_lite(&pre2).into_iter().map(relu).collect();
        let mut r2 = r1;
        r2.extend_from_slice(&h2);

        let o = self.gen_head.forward(&r2);
        let row = self.activate_row(&o, rng);

        let (col, cat) = match cond_opt {
            Some((c, k)) => (Some(c), Some(k)),
            None => (None, None),
        };
        Ok((row, r2, col, cat))
    }

    /// Generate `n` rows as a flat row-major `[n × data_dim]` buffer.
    ///
    /// # Errors
    /// Propagated from the internal forward pass.
    pub fn generate(&self, n: usize, rng: &mut LcgRng) -> TabularResult<Vec<f32>> {
        let mut out = Vec::with_capacity(n * self.data_dim);
        for _ in 0..n {
            let (row, _r2, _c, _k) = self.generate_row(rng)?;
            out.extend_from_slice(&row);
        }
        Ok(out)
    }

    /// Pack a flat `[n_rows × data_dim]` buffer into `n_rows / pac` packs, each a
    /// `pac · data_dim` slice (PacGAN grouping).
    ///
    /// # Errors
    /// - [`TabularError::DimensionMismatch`] if the buffer length is wrong.
    /// - [`TabularError::InvalidParameter`] if `n_rows` is not a multiple of `pac`.
    pub fn pack(&self, rows: &[f32], n_rows: usize) -> TabularResult<Vec<Vec<f32>>> {
        if rows.len() != n_rows * self.data_dim {
            return Err(TabularError::DimensionMismatch {
                expected: n_rows * self.data_dim,
                got: rows.len(),
            });
        }
        if self.pac == 0 || !n_rows.is_multiple_of(self.pac) {
            return Err(TabularError::InvalidParameter {
                name: "n_rows".into(),
                msg: format!("must be a positive multiple of pac = {}", self.pac),
            });
        }
        let pack_dim = self.pac * self.data_dim;
        let n_packs = n_rows / self.pac;
        let mut packs = Vec::with_capacity(n_packs);
        for p in 0..n_packs {
            packs.push(rows[p * pack_dim..(p + 1) * pack_dim].to_vec());
        }
        Ok(packs)
    }

    /// Discriminator body (the two leaky-ReLU layers before the scalar head).
    fn disc_body(&self, pack: &[f32]) -> Vec<f32> {
        let d1: Vec<f32> = self
            .disc1
            .forward(pack)
            .into_iter()
            .map(|v| leaky_relu(v, 0.2))
            .collect();
        self.disc2
            .forward(&d1)
            .into_iter()
            .map(|v| leaky_relu(v, 0.2))
            .collect()
    }

    /// Discriminator forward pass over a single pack of `pac · data_dim` values,
    /// returning the scalar critic score.
    ///
    /// # Errors
    /// [`TabularError::DimensionMismatch`] if the pack length is wrong.
    pub fn discriminator_forward(&self, pack: &[f32]) -> TabularResult<f32> {
        if pack.len() != self.disc1.in_dim {
            return Err(TabularError::DimensionMismatch {
                expected: self.disc1.in_dim,
                got: pack.len(),
            });
        }
        let h = self.disc_body(pack);
        Ok(self.disc_head.forward(&h)[0])
    }

    /// One discriminator training micro-step on the Wasserstein objective
    /// `L_D = mean(D(fake)) − mean(D(real))`.
    ///
    /// The two-layer body is held fixed (a random feature extractor) and the
    /// scalar head is updated by the exact gradient of `L_D`; this is a genuine,
    /// non-converging step, not full end-to-end training.  Returns the
    /// pre-update loss.
    ///
    /// # Errors
    /// - [`TabularError::DimensionMismatch`] if `real_rows` length is wrong.
    /// - [`TabularError::InvalidParameter`] if `n_real` is not a positive
    ///   multiple of `pac`.
    pub fn discriminator_step(
        &mut self,
        real_rows: &[f32],
        n_real: usize,
        rng: &mut LcgRng,
        lr: f32,
    ) -> TabularResult<f32> {
        if real_rows.len() != n_real * self.data_dim {
            return Err(TabularError::DimensionMismatch {
                expected: n_real * self.data_dim,
                got: real_rows.len(),
            });
        }
        if self.pac == 0 || n_real == 0 || !n_real.is_multiple_of(self.pac) {
            return Err(TabularError::InvalidParameter {
                name: "n_real".into(),
                msg: format!("must be a positive multiple of pac = {}", self.pac),
            });
        }

        let fake_rows = self.generate(n_real, rng)?;
        let pack_dim = self.pac * self.data_dim;
        let n_packs = n_real / self.pac;

        let mut sum_real = 0.0_f32;
        let mut sum_fake = 0.0_f32;
        let mut grad_w = vec![0.0_f32; self.disc_head.w.len()];
        for p in 0..n_packs {
            let rp = &real_rows[p * pack_dim..(p + 1) * pack_dim];
            let fp = &fake_rows[p * pack_dim..(p + 1) * pack_dim];
            let hr = self.disc_body(rp);
            let hf = self.disc_body(fp);
            sum_real += self.disc_head.forward(&hr)[0];
            sum_fake += self.disc_head.forward(&hf)[0];
            for (g, (&a, &b)) in grad_w.iter_mut().zip(hf.iter().zip(hr.iter())) {
                *g += a - b;
            }
        }
        let inv = 1.0 / n_packs as f32;
        let loss = (sum_fake - sum_real) * inv;

        for (w, &g) in self.disc_head.w.iter_mut().zip(grad_w.iter()) {
            *w -= lr * g * inv;
        }
        Ok(loss)
    }

    /// One generator training micro-step.
    ///
    /// Returns `L_G = −mean(D(fake)) + L_cond`, where `L_cond` is the conditional
    /// cross-entropy enforcing that the generated discrete column matches the
    /// sampled condition.  The generator head is updated by the exact gradient of
    /// `L_cond` (the body is held fixed and the discriminator is frozen, as in
    /// alternating GAN training); the adversarial term is reported for
    /// monitoring.  A genuine, non-converging step — not full training.
    ///
    /// # Errors
    /// - [`TabularError::InvalidParameter`] if `batch` is not a positive multiple
    ///   of `pac`.
    /// - Propagated from the internal forward / discriminator passes.
    pub fn generator_step(
        &mut self,
        batch: usize,
        rng: &mut LcgRng,
        lr: f32,
    ) -> TabularResult<f32> {
        if self.pac == 0 || batch == 0 || !batch.is_multiple_of(self.pac) {
            return Err(TabularError::InvalidParameter {
                name: "batch".into(),
                msg: format!("must be a positive multiple of pac = {}", self.pac),
            });
        }

        let mut rows = Vec::with_capacity(batch * self.data_dim);
        let mut caches: Vec<(Vec<f32>, Option<usize>, Option<usize>)> = Vec::with_capacity(batch);
        for _ in 0..batch {
            let (row, r2, c, k) = self.generate_row(rng)?;
            rows.extend_from_slice(&row);
            caches.push((r2, c, k));
        }

        // Adversarial term: −mean over packs of the critic score.
        let pack_dim = self.pac * self.data_dim;
        let n_packs = batch / self.pac;
        let mut sum_d = 0.0_f32;
        for p in 0..n_packs {
            let fp = &rows[p * pack_dim..(p + 1) * pack_dim];
            sum_d += self.discriminator_forward(fp)?;
        }
        let adv = -(sum_d / n_packs as f32);

        // Conditional cross-entropy term and its exact gradient on the head.
        let head_in = self.gen_input + 2 * self.config.generator_dim;
        let mut grad_w = vec![0.0_f32; self.gen_head.w.len()];
        let mut grad_b = vec![0.0_f32; self.data_dim];
        let mut cond_loss = 0.0_f32;
        let mut count = 0usize;
        for (r2, c, k) in &caches {
            if let (Some(col), Some(cat)) = (*c, *k) {
                let (start, card) = self.categorical_slice(col);
                let o = self.gen_head.forward(r2);
                let sm = softmax(&o[start..start + card]);
                cond_loss += -sm[cat].max(1e-30).ln();
                count += 1;
                for (j, &prob) in sm.iter().enumerate() {
                    let grad = prob - if j == cat { 1.0 } else { 0.0 };
                    let out_unit = start + j;
                    grad_b[out_unit] += grad;
                    let base = out_unit * head_in;
                    for (gw, &rv) in grad_w[base..base + head_in].iter_mut().zip(r2.iter()) {
                        *gw += grad * rv;
                    }
                }
            }
        }

        if count > 0 {
            let inv = 1.0 / count as f32;
            for (w, &g) in self.gen_head.w.iter_mut().zip(grad_w.iter()) {
                *w -= lr * g * inv;
            }
            for (b, &g) in self.gen_head.b.iter_mut().zip(grad_b.iter()) {
                *b -= lr * g * inv;
            }
            cond_loss *= inv;
        }

        Ok(adv + cond_loss)
    }
}

// ─── Internal helpers ──────────────────────────────────────────────────────────

/// Per-sample standardisation standing in for batch normalisation in the
/// single-row generator forward pass.
fn bn_lite(v: &[f32]) -> Vec<f32> {
    let n = v.len();
    if n == 0 {
        return Vec::new();
    }
    let nf = n as f32;
    let mean = v.iter().sum::<f32>() / nf;
    let var = v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / nf;
    let inv = 1.0 / (var + 1e-5).sqrt();
    v.iter().map(|&x| (x - mean) * inv).collect()
}

/// Gumbel-softmax relaxation: `softmax((logits + g) / τ)`, `g_i = −log(−log u_i)`.
fn gumbel_softmax(logits: &[f32], tau: f32, rng: &mut LcgRng) -> Vec<f32> {
    let inv_tau = 1.0 / tau.max(1e-6);
    let noisy: Vec<f32> = logits
        .iter()
        .map(|&l| {
            let u = rng.next_f32().clamp(1e-10, 1.0 - 1e-10);
            let g = -(-(u.ln())).ln();
            (l + g) * inv_tau
        })
        .collect();
    softmax(&noisy)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> CtganConfig {
        CtganConfig {
            n_continuous: 2,
            categorical_cardinalities: vec![4, 2],
            n_modes: 3,
            noise_dim: 8,
            generator_dim: 16,
            discriminator_dim: 16,
            pac: 2,
            gumbel_tau: 0.5,
        }
    }

    fn make_model() -> CtGan {
        let mut rng = LcgRng::new(42);
        CtGan::new(small_cfg(), &mut rng).expect("value should be present")
    }

    // ── 1. dims correct ──────────────────────────────────────────────────────
    #[test]
    fn dims_correct() {
        let m = make_model();
        // 2 continuous × (1 + 3 modes) = 8, plus 4 + 2 categorical = 6 → 14.
        assert_eq!(m.data_dim(), 14);
        assert_eq!(m.cond_dim(), 6);
        assert_eq!(m.pac, 2);
    }

    // ── 2. generate shape and finite ─────────────────────────────────────────
    #[test]
    fn generate_shape_finite() {
        let m = make_model();
        let mut rng = LcgRng::new(1);
        let out = m.generate(6, &mut rng).expect("generate should succeed");
        assert_eq!(out.len(), 6 * m.data_dim());
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── 3. gumbel blocks sum to 1, scalars in (-1, 1) ────────────────────────
    #[test]
    fn generated_row_structure() {
        let m = make_model();
        let mut rng = LcgRng::new(2);
        let out = m.generate(1, &mut rng).expect("generate should succeed");
        // layout: [scalar0, mode0(3), scalar1, mode1(3), cat0(4), cat1(2)]
        assert!(out[0].abs() <= 1.0 + 1e-6);
        assert!(out[4].abs() <= 1.0 + 1e-6);
        let mode0: f32 = out[1..4].iter().sum();
        let mode1: f32 = out[5..8].iter().sum();
        let cat0: f32 = out[8..12].iter().sum();
        let cat1: f32 = out[12..14].iter().sum();
        for (name, s) in [
            ("mode0", mode0),
            ("mode1", mode1),
            ("cat0", cat0),
            ("cat1", cat1),
        ] {
            assert!((s - 1.0).abs() < 1e-4, "{name} sums to {s}");
        }
    }

    // ── 4. generation is deterministic given the seed ────────────────────────
    #[test]
    fn generate_deterministic() {
        let m = make_model();
        let mut r1 = LcgRng::new(99);
        let mut r2 = LcgRng::new(99);
        assert_eq!(
            m.generate(4, &mut r1).expect("generate should succeed"),
            m.generate(4, &mut r2).expect("generate should succeed")
        );
    }

    // ── 5. mode normalisation round-trips within a mode ──────────────────────
    #[test]
    fn mode_normalizer_round_trip() {
        let data = vec![0.0_f32, 0.1, 0.2, 5.0, 5.1, 5.2];
        let mn = ModeNormalizer::fit(&data, 6, 1, 2).expect("fit should succeed");
        for &v in &[0.0_f32, 0.1, 0.2, 5.0, 5.1, 5.2] {
            let (mode, alpha) = mn
                .transform_value(0, v)
                .expect("transform_value should succeed");
            let back = mn
                .inverse_value(0, mode, alpha)
                .expect("inverse_value should succeed");
            assert!((back - v).abs() < 1e-3, "v={v} round-trips to {back}");
        }
    }

    // ── 6. mode assignment picks the nearest centroid ────────────────────────
    #[test]
    fn mode_normalizer_assignment() {
        let data = vec![0.0_f32, 0.1, 0.2, 5.0, 5.1, 5.2];
        let mn = ModeNormalizer::fit(&data, 6, 1, 2).expect("fit should succeed");
        let (low_mode, _) = mn
            .transform_value(0, 0.05)
            .expect("transform_value should succeed");
        let (high_mode, _) = mn
            .transform_value(0, 5.05)
            .expect("transform_value should succeed");
        assert_ne!(low_mode, high_mode);
        assert_eq!(mn.n_modes(), 2);
        assert_eq!(mn.n_columns(), 1);
    }

    // ── 7. conditional vector is a correct one-hot ───────────────────────────
    #[test]
    fn conditional_vector_one_hot() {
        let sampler = ConditionalSampler::uniform(&[4, 2]).expect("uniform should succeed");
        assert_eq!(sampler.cond_dim(), 6);
        let v = sampler
            .cond_vector(1, 1)
            .expect("cond_vector should succeed");
        assert_eq!(v.len(), 6);
        // column 1 starts at offset 4; category 1 → index 5.
        for (i, &val) in v.iter().enumerate() {
            let expected = if i == 5 { 1.0 } else { 0.0 };
            assert!((val - expected).abs() < 1e-9, "index {i} = {val}");
        }
        assert!(sampler.cond_vector(0, 9).is_err());
        assert!(sampler.cond_vector(5, 0).is_err());
    }

    // ── 8. conditional sampler respects category ranges ──────────────────────
    #[test]
    fn conditional_sampler_in_range_and_deterministic() {
        let sampler = ConditionalSampler::uniform(&[4, 2]).expect("uniform should succeed");
        let mut r1 = LcgRng::new(7);
        let mut r2 = LcgRng::new(7);
        for _ in 0..50 {
            let (c1, k1) = sampler.sample(&mut r1).expect("sample should succeed");
            let (c2, k2) = sampler.sample(&mut r2).expect("sample should succeed");
            assert_eq!((c1, k1), (c2, k2));
            assert!(c1 < 2);
            let card = [4, 2][c1];
            assert!(k1 < card, "category {k1} out of range for column {c1}");
        }
    }

    // ── 9. log-frequency weighting up-weights frequent categories ────────────
    #[test]
    fn conditional_sampler_log_frequency() {
        // column 0 (cardinality 3): category 0 very frequent, 1 and 2 rare.
        let mut cat_data = vec![0usize; 102];
        cat_data[100] = 1;
        cat_data[101] = 2;
        let sampler = ConditionalSampler::fit(&cat_data, 102, &[3]).expect("fit should succeed");
        let probs = sampler
            .category_probabilities(0)
            .expect("category_probabilities should succeed");
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[0] > probs[1], "frequent category should dominate");
        assert!((probs[1] - probs[2]).abs() < 1e-6);
    }

    // ── 10. packing groups pac rows ──────────────────────────────────────────
    #[test]
    fn pack_groups_rows() {
        let m = make_model();
        let mut rng = LcgRng::new(3);
        let rows = m.generate(4, &mut rng).expect("generate should succeed");
        let packs = m.pack(&rows, 4).expect("pack should succeed");
        assert_eq!(packs.len(), 2);
        assert_eq!(m.n_packs(4), 2);
        for p in &packs {
            assert_eq!(p.len(), m.pac * m.data_dim());
        }
        // first pack is the first two rows concatenated
        assert_eq!(packs[0], rows[0..2 * m.data_dim()].to_vec());
        assert!(m.pack(&rows, 3).is_err()); // 3 not a multiple of pac=2
    }

    // ── 11. discriminator forward finite, wrong length errors ────────────────
    #[test]
    fn discriminator_forward_finite() {
        let m = make_model();
        let mut rng = LcgRng::new(4);
        let rows = m.generate(2, &mut rng).expect("generate should succeed");
        let packs = m.pack(&rows, 2).expect("pack should succeed");
        let score = m
            .discriminator_forward(&packs[0])
            .expect("discriminator_forward should succeed");
        assert!(score.is_finite());
        assert!(m.discriminator_forward(&[0.0; 3]).is_err());
    }

    // ── 12. discriminator step returns finite loss ───────────────────────────
    #[test]
    fn discriminator_step_finite() {
        let mut m = make_model();
        let mut rng = LcgRng::new(5);
        let real = m.generate(4, &mut rng).expect("generate should succeed");
        let loss = m
            .discriminator_step(&real, 4, &mut rng, 0.01)
            .expect("discriminator_step should succeed");
        assert!(loss.is_finite(), "loss = {loss}");
    }

    // ── 13. generator step returns finite loss ───────────────────────────────
    #[test]
    fn generator_step_finite() {
        let mut m = make_model();
        let mut rng = LcgRng::new(6);
        let loss = m
            .generator_step(4, &mut rng, 0.01)
            .expect("generator_step should succeed");
        assert!(loss.is_finite(), "loss = {loss}");
    }

    // ── 14. all-continuous configuration (no conditioning) works ─────────────
    #[test]
    fn all_continuous_config() {
        let cfg = CtganConfig {
            n_continuous: 3,
            categorical_cardinalities: vec![],
            n_modes: 2,
            noise_dim: 6,
            generator_dim: 12,
            discriminator_dim: 12,
            pac: 2,
            gumbel_tau: 0.5,
        };
        let mut rng = LcgRng::new(8);
        let mut m = CtGan::new(cfg, &mut rng).expect("new should succeed");
        assert_eq!(m.cond_dim(), 0);
        let out = m.generate(2, &mut rng).expect("generate should succeed");
        assert!(out.iter().all(|v| v.is_finite()));
        // generator step with no categorical columns is still finite (adv only)
        assert!(
            m.generator_step(2, &mut rng, 0.01)
                .expect("generator_step should succeed")
                .is_finite()
        );
    }

    // ── 15. fit_conditional updates the sampler ──────────────────────────────
    #[test]
    fn fit_conditional_updates_sampler() {
        let mut m = make_model();
        // 4 samples × 2 categorical columns (cardinalities 4 and 2)
        let cat_data = vec![0usize, 0, 0, 0, 0, 1, 0, 1];
        m.fit_conditional(&cat_data, 4)
            .expect("fit_conditional should succeed");
        let probs = m
            .conditional_sampler()
            .category_probabilities(0)
            .expect("category_probabilities should succeed");
        // category 0 appears in every row of column 0 → highest probability
        assert!(probs[0] > probs[1]);
    }

    // ── 16. constructor validation ───────────────────────────────────────────
    #[test]
    fn new_rejects_bad_config() {
        let mut rng = LcgRng::new(1);

        let mut cfg = small_cfg();
        cfg.n_continuous = 0;
        cfg.categorical_cardinalities = vec![];
        assert!(matches!(
            CtGan::new(cfg, &mut rng),
            Err(TabularError::InvalidFeatureCount { .. })
        ));

        let mut cfg = small_cfg();
        cfg.pac = 0;
        assert!(CtGan::new(cfg, &mut rng).is_err());

        let mut cfg = small_cfg();
        cfg.gumbel_tau = 0.0;
        assert!(CtGan::new(cfg, &mut rng).is_err());

        let mut cfg = small_cfg();
        cfg.noise_dim = 0;
        assert!(CtGan::new(cfg, &mut rng).is_err());

        let mut cfg = small_cfg();
        cfg.categorical_cardinalities = vec![0];
        assert!(CtGan::new(cfg, &mut rng).is_err());
    }
}
