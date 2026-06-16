//! Additive Quantization (AQ).
//!
//! Babenko & Lempitsky, "Additive Quantization for Extreme Vector Compression",
//! CVPR 2014.
//!
//! AQ approximates each database vector `x` as a **sum** of `m` codewords, one
//! drawn from each of `m` separate codebooks:
//!
//! ```text
//! x ≈ x̂ = c_{1, a_1} + c_{2, a_2} + … + c_{m, a_m}
//! ```
//!
//! Unlike Product Quantization — where the `m` codebooks live in *disjoint*
//! sub-spaces and the codewords are concatenated — every AQ codeword spans the
//! **full** dimension and the reconstruction is their vector sum.  This strictly
//! generalises PQ (PQ is the special case where codebook `t` is zero outside of
//! sub-space `t`) and yields lower distortion at the same bit-rate, at the cost
//! of a harder, combinatorial encoding problem.
//!
//! ## Encoding (beam search)
//!
//! Choosing the codes that minimise `‖x − x̂‖²` is NP-hard, so encoding is done
//! with a fixed-width **beam search** over codebooks: starting from the empty
//! selection we add one codebook at a time, keeping the `beam_width` partial
//! assignments with the smallest residual energy.  This is the procedure
//! proposed in the original paper (their "Beam search" encoding).
//!
//! ## Training (alternating)
//!
//! Training alternates two steps until the epoch budget is exhausted:
//!   1. **Encode** every sample with the current codebooks (beam search).
//!   2. **Re-fit** the codebooks by least squares: holding the integer codes
//!      fixed, the stacked codeword matrix that minimises total reconstruction
//!      error solves a linear system whose normal equations are
//!      `Bᵀ B · C = Bᵀ X`, where `B` is the `n × (m·k)` binary selection matrix
//!      and `C` is the `(m·k) × dim` matrix of all codewords.  We solve the
//!      `(m·k) × (m·k)` system per output dimension with a small ridge term for
//!      numerical stability (the system is singular whenever a codeword is
//!      unused, which a plain inverse could not handle).
use crate::error::{AnnError, AnnResult};
use crate::handle::LcgRng;
use crate::kmeans::kmeans::KMeans;

/// Configuration for Additive Quantization training.
#[derive(Debug, Clone, Copy)]
pub struct AqConfig {
    /// Number of codebooks `m` (codewords summed per vector). Must be ≥ 1.
    pub m: usize,
    /// Codewords per codebook `k`. Must be in `[1, 256]` so a code fits in `u8`.
    pub k: usize,
    /// Vector dimensionality. Must be ≥ 1.
    pub dim: usize,
    /// Beam width used by the encoder. Must be ≥ 1.
    pub beam_width: usize,
    /// Number of alternating train epochs (encode ↔ refit). Must be ≥ 1.
    pub n_iter: usize,
    /// Ridge regularisation added to the codebook least-squares solve.
    /// Must be ≥ 0; a tiny positive value keeps the normal equations invertible.
    pub ridge: f32,
}

impl AqConfig {
    fn validate(&self) -> AnnResult<()> {
        if self.m == 0 {
            return Err(AnnError::InvalidLayerCount { n: 0 });
        }
        if self.dim == 0 {
            return Err(AnnError::InvalidVectorDim { dim: 0 });
        }
        if self.k == 0 || self.k > 256 {
            return Err(AnnError::InvalidK { k: self.k, n: 256 });
        }
        if self.beam_width == 0 {
            return Err(AnnError::Internal {
                msg: "aq: beam_width must be ≥ 1".to_string(),
            });
        }
        if self.n_iter == 0 {
            return Err(AnnError::Internal {
                msg: "aq: n_iter must be ≥ 1".to_string(),
            });
        }
        if self.ridge < 0.0 || !self.ridge.is_finite() {
            return Err(AnnError::Internal {
                msg: format!("aq: ridge must be finite and ≥ 0, got {}", self.ridge),
            });
        }
        Ok(())
    }
}

/// A trained Additive-Quantization model: `m` codebooks of `k` codewords each,
/// every codeword being a full `dim`-dimensional vector.
#[derive(Debug, Clone)]
pub struct AdditiveQuantizer {
    /// `codebooks[t]` is a row-major `k × dim` matrix of codewords.
    codebooks: Vec<Vec<f32>>,
    m: usize,
    k: usize,
    dim: usize,
}

#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[inline]
fn squared_norm(a: &[f32]) -> f32 {
    a.iter().map(|v| v * v).sum()
}

impl AdditiveQuantizer {
    /// Number of codebooks `m`.
    #[must_use]
    pub fn m(&self) -> usize {
        self.m
    }

    /// Codewords per codebook `k`.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Vector dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Row-major `k × dim` codeword matrix of codebook `t`.
    ///
    /// # Errors
    /// Returns [`AnnError::IdOutOfRange`] if `t >= m`.
    pub fn codebook(&self, t: usize) -> AnnResult<&[f32]> {
        self.codebooks
            .get(t)
            .map(Vec::as_slice)
            .ok_or(AnnError::IdOutOfRange { id: t, n: self.m })
    }

    /// Codeword `idx` of codebook `t` (a `dim`-length slice).
    fn codeword(&self, t: usize, idx: usize) -> AnnResult<&[f32]> {
        let cb = self.codebook(t)?;
        let off = idx * self.dim;
        let end = off + self.dim;
        cb.get(off..end)
            .ok_or(AnnError::IdOutOfRange { id: idx, n: self.k })
    }

    /// Train an additive quantizer on `n` row-major vectors `[n, dim]`.
    ///
    /// Codebooks are seeded by running `m` independent k-means passes on the
    /// running residual (a residual-quantization warm start), after which the
    /// alternating encode/refit optimisation refines them jointly.
    ///
    /// # Errors
    /// - [`AnnError::EmptyInput`] when `n == 0`.
    /// - [`AnnError::DimensionMismatch`] when `data.len() != n * dim`.
    /// - configuration errors from [`AqConfig`] validation.
    pub fn train(
        data: &[f32],
        n: usize,
        cfg: &AqConfig,
        rng: &mut LcgRng,
    ) -> AnnResult<AdditiveQuantizer> {
        cfg.validate()?;
        if n == 0 {
            return Err(AnnError::EmptyInput);
        }
        let dim = cfg.dim;
        if data.len() != n * dim {
            return Err(AnnError::DimensionMismatch {
                expected: n * dim,
                got: data.len(),
            });
        }

        // ── Warm start: residual k-means per codebook ───────────────────────
        let mut codebooks: Vec<Vec<f32>> = Vec::with_capacity(cfg.m);
        let mut residual = data.to_vec();
        for _ in 0..cfg.m {
            let k = cfg.k.min(n);
            let km = KMeans::fit(&residual, n, dim, k, cfg.n_iter.max(10), rng)?;
            let mut cb = vec![0.0_f32; cfg.k * dim];
            // Copy the (possibly fewer than k) learned centroids; pad the rest
            // by cycling through the learned ones so every slot is valid.
            let learned = km.centroids();
            for c in 0..cfg.k {
                let src = (c % k) * dim;
                cb[c * dim..(c + 1) * dim].copy_from_slice(&learned[src..src + dim]);
            }
            // Subtract the assigned centroid from the residual for the next book.
            let assign = km.predict(&residual, n, dim)?;
            for (i, &a) in assign.iter().enumerate() {
                let src = (a % k) * dim;
                for d in 0..dim {
                    residual[i * dim + d] -= learned[src + d];
                }
            }
            codebooks.push(cb);
        }

        let mut quant = AdditiveQuantizer {
            codebooks,
            m: cfg.m,
            k: cfg.k,
            dim,
        };

        // ── Alternating optimisation ────────────────────────────────────────
        for _ in 0..cfg.n_iter {
            // Encode all samples with the current codebooks.
            let mut codes = vec![0u8; n * cfg.m];
            for i in 0..n {
                let x = &data[i * dim..(i + 1) * dim];
                let assign = quant.encode_with_beam(x, cfg.beam_width)?;
                for t in 0..cfg.m {
                    codes[i * cfg.m + t] = assign[t];
                }
            }
            // Re-fit codebooks by ridge least squares against fixed codes.
            quant.refit_codebooks(data, n, &codes, cfg.ridge)?;
        }

        Ok(quant)
    }

    /// Encode one `dim`-length vector into `m` byte codes via beam search.
    ///
    /// # Errors
    /// [`AnnError::DimensionMismatch`] when `x.len() != dim`.
    pub fn encode(&self, x: &[f32]) -> AnnResult<Vec<u8>> {
        if x.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        self.encode_with_beam(x, self.k.max(1))
    }

    /// Beam-search encoder with explicit beam width.
    ///
    /// Maintains up to `beam_width` partial assignments ordered by residual
    /// energy `‖x − Σ chosen‖²`.  At each codebook it expands every surviving
    /// candidate with every codeword, then truncates back to `beam_width`.
    fn encode_with_beam(&self, x: &[f32], beam_width: usize) -> AnnResult<Vec<u8>> {
        let dim = self.dim;

        /// A partial assignment plus its running reconstruction and energy.
        struct Candidate {
            codes: Vec<u8>,
            recon: Vec<f32>,
            energy: f32,
        }

        let mut beam: Vec<Candidate> = vec![Candidate {
            codes: Vec::with_capacity(self.m),
            recon: vec![0.0_f32; dim],
            energy: squared_norm(x),
        }];

        for t in 0..self.m {
            let mut next: Vec<Candidate> = Vec::with_capacity(beam.len() * self.k);
            for cand in &beam {
                for idx in 0..self.k {
                    let cw = self.codeword(t, idx)?;
                    let mut recon = cand.recon.clone();
                    for d in 0..dim {
                        recon[d] += cw[d];
                    }
                    let energy = l2_sq(x, &recon);
                    let mut codes = cand.codes.clone();
                    codes.push(idx as u8);
                    next.push(Candidate {
                        codes,
                        recon,
                        energy,
                    });
                }
            }
            // Keep the `beam_width` lowest-energy partial assignments.
            next.sort_by(|a, b| {
                a.energy
                    .partial_cmp(&b.energy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            next.truncate(beam_width.max(1));
            beam = next;
        }

        let best = beam.into_iter().next().ok_or(AnnError::Internal {
            msg: "aq: beam search produced no candidates".to_string(),
        })?;
        Ok(best.codes)
    }

    /// Reconstruct `x̂` from `m` byte codes.
    ///
    /// # Errors
    /// - [`AnnError::DimensionMismatch`] when `codes.len() != m`.
    /// - [`AnnError::IdOutOfRange`] when any code ≥ `k`.
    pub fn decode(&self, codes: &[u8]) -> AnnResult<Vec<f32>> {
        if codes.len() != self.m {
            return Err(AnnError::DimensionMismatch {
                expected: self.m,
                got: codes.len(),
            });
        }
        let mut out = vec![0.0_f32; self.dim];
        for (t, &c) in codes.iter().enumerate() {
            let cw = self.codeword(t, c as usize)?;
            for d in 0..self.dim {
                out[d] += cw[d];
            }
        }
        Ok(out)
    }

    /// Reconstruction error `‖x − decode(encode(x))‖²` for a single vector.
    ///
    /// # Errors
    /// Propagates [`Self::encode`] / [`Self::decode`] errors.
    pub fn reconstruction_error(&self, x: &[f32]) -> AnnResult<f32> {
        let codes = self.encode(x)?;
        let recon = self.decode(&codes)?;
        Ok(l2_sq(x, &recon))
    }

    /// Build an Asymmetric Distance Computation (ADC) lookup table for a query.
    ///
    /// Entry `[t * k + idx]` holds `‖q − c_{t, idx}‖²`-style partials usable for
    /// fast distance estimation.  The exact L2² between a query `q` and a coded
    /// vector with codes `(a_1, …, a_m)` is
    ///
    /// ```text
    /// ‖q − x̂‖² = ‖q‖² − 2 q·x̂ + ‖x̂‖²
    /// ```
    ///
    /// The cross term `q·x̂ = Σ_t q·c_{t, a_t}` decomposes additively, so the
    /// table stores `dot[t * k + idx] = q · c_{t, idx}`.  The `‖x̂‖²` term does
    /// **not** decompose (cross-codebook products appear), so callers that need
    /// exact distances should combine this table with [`Self::self_dot_table`].
    ///
    /// # Errors
    /// [`AnnError::DimensionMismatch`] when `q.len() != dim`.
    pub fn build_dot_table(&self, q: &[f32]) -> AnnResult<Vec<f32>> {
        if q.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: q.len(),
            });
        }
        let mut table = vec![0.0_f32; self.m * self.k];
        for t in 0..self.m {
            for idx in 0..self.k {
                let cw = self.codeword(t, idx)?;
                table[t * self.k + idx] = q.iter().zip(cw.iter()).map(|(a, b)| a * b).sum();
            }
        }
        Ok(table)
    }

    /// Precompute the codeword cross-product table used for the `‖x̂‖²` term.
    ///
    /// Returns a flat `(m·k) × (m·k)` row-major matrix `G` where
    /// `G[(t·k+i) * (m·k) + (u·k+j)] = c_{t,i} · c_{u,j}`.  Given codes
    /// `(a_1, …, a_m)`, `‖x̂‖² = Σ_{t,u} G[(t·k+a_t), (u·k+a_u)]`.
    #[must_use]
    pub fn self_dot_table(&self) -> Vec<f32> {
        let mk = self.m * self.k;
        let mut g = vec![0.0_f32; mk * mk];
        for t in 0..self.m {
            for i in 0..self.k {
                let ci = &self.codebooks[t][i * self.dim..(i + 1) * self.dim];
                let row = t * self.k + i;
                for u in 0..self.m {
                    for j in 0..self.k {
                        let cj = &self.codebooks[u][j * self.dim..(j + 1) * self.dim];
                        let dot: f32 = ci.iter().zip(cj.iter()).map(|(a, b)| a * b).sum();
                        g[row * mk + (u * self.k + j)] = dot;
                    }
                }
            }
        }
        g
    }

    /// Estimate `‖q − x̂‖²` from precomputed tables and the codes of `x`.
    ///
    /// `q_norm_sq = ‖q‖²`, `dot_table` from [`Self::build_dot_table`], and
    /// `self_dot` from [`Self::self_dot_table`].  This reproduces the exact L2²
    /// distance (up to floating-point error) without ever materialising `x̂`.
    ///
    /// # Errors
    /// [`AnnError::DimensionMismatch`] when `codes.len() != m`.
    pub fn adc_distance(
        &self,
        q_norm_sq: f32,
        dot_table: &[f32],
        self_dot: &[f32],
        codes: &[u8],
    ) -> AnnResult<f32> {
        if codes.len() != self.m {
            return Err(AnnError::DimensionMismatch {
                expected: self.m,
                got: codes.len(),
            });
        }
        let mk = self.m * self.k;
        let mut cross = 0.0_f32; // q · x̂
        let mut self_term = 0.0_f32; // ‖x̂‖²
        for t in 0..self.m {
            let at = codes[t] as usize;
            cross += dot_table[t * self.k + at];
            let row = t * self.k + at;
            for u in 0..self.m {
                let au = codes[u] as usize;
                self_term += self_dot[row * mk + (u * self.k + au)];
            }
        }
        Ok(q_norm_sq - 2.0 * cross + self_term)
    }

    /// Re-fit all codebooks by ridge least squares with the codes held fixed.
    ///
    /// Solves, independently for each output dimension `d`, the normal equations
    /// `(BᵀB + λI) c_d = Bᵀ x_d` where `B` is the `n × (m·k)` selection matrix
    /// (`B[i, t·k + a_{i,t}] = 1`).  Because `B` has exactly `m` ones per row,
    /// `BᵀB` is the `(m·k) × (m·k)` codeword co-occurrence matrix, which we form
    /// directly without instantiating `B`.
    fn refit_codebooks(
        &mut self,
        data: &[f32],
        n: usize,
        codes: &[u8],
        ridge: f32,
    ) -> AnnResult<()> {
        let dim = self.dim;
        let mk = self.m * self.k;

        // Gram matrix BᵀB (mk × mk) and RHS Bᵀ X (mk × dim).
        let mut gram = vec![0.0_f32; mk * mk];
        let mut rhs = vec![0.0_f32; mk * dim];
        for i in 0..n {
            // Indices of the `m` active columns for sample i.
            let mut active = [0usize; 64];
            let active = if self.m <= 64 {
                let slot = &mut active[..self.m];
                for t in 0..self.m {
                    slot[t] = t * self.k + codes[i * self.m + t] as usize;
                }
                slot.to_vec()
            } else {
                (0..self.m)
                    .map(|t| t * self.k + codes[i * self.m + t] as usize)
                    .collect::<Vec<_>>()
            };
            for &r in &active {
                for &c in &active {
                    gram[r * mk + c] += 1.0;
                }
                let x = &data[i * dim..(i + 1) * dim];
                for d in 0..dim {
                    rhs[r * dim + d] += x[d];
                }
            }
        }
        // Add ridge to the diagonal so the system stays invertible even when a
        // codeword is never selected (its row/col would otherwise be all zero).
        for r in 0..mk {
            gram[r * mk + r] += ridge.max(1e-6);
        }

        // Solve gram · C = rhs for C (mk × dim) via Gauss-Jordan on the
        // augmented [gram | rhs] system (one factorisation, `dim` right sides).
        let solution = solve_spd_multi(&gram, &rhs, mk, dim)?;

        // Scatter the solution back into the per-codebook codeword storage.
        for t in 0..self.m {
            for idx in 0..self.k {
                let row = t * self.k + idx;
                let dst = &mut self.codebooks[t][idx * dim..(idx + 1) * dim];
                dst.copy_from_slice(&solution[row * dim..(row + 1) * dim]);
            }
        }
        Ok(())
    }
}

/// Solve `A · X = B` for `X`, where `A` is `n × n` (row-major, symmetric
/// positive-definite after ridge) and `B` is `n × rhs` (row-major), returning
/// `X` as `n × rhs` row-major.  Uses Gauss-Jordan elimination with partial
/// pivoting on the augmented matrix.
fn solve_spd_multi(a: &[f32], b: &[f32], n: usize, rhs: usize) -> AnnResult<Vec<f32>> {
    // Augmented matrix [A | B] with width n + rhs.
    let w = n + rhs;
    let mut aug = vec![0.0_f32; n * w];
    for r in 0..n {
        for c in 0..n {
            aug[r * w + c] = a[r * n + c];
        }
        for c in 0..rhs {
            aug[r * w + n + c] = b[r * rhs + c];
        }
    }

    for col in 0..n {
        // Partial pivot: largest |value| in this column at or below the diagonal.
        let mut pivot = col;
        let mut best = aug[col * w + col].abs();
        for r in (col + 1)..n {
            let v = aug[r * w + col].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best < 1e-20 {
            return Err(AnnError::Internal {
                msg: format!("aq: singular codebook system at column {col}"),
            });
        }
        if pivot != col {
            for c in 0..w {
                aug.swap(col * w + c, pivot * w + c);
            }
        }
        // Normalise pivot row.
        let inv = 1.0 / aug[col * w + col];
        for c in 0..w {
            aug[col * w + c] *= inv;
        }
        // Eliminate this column from every other row.
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = aug[r * w + col];
            if factor != 0.0 {
                for c in 0..w {
                    aug[r * w + c] -= factor * aug[col * w + c];
                }
            }
        }
    }

    // Extract the solution block.
    let mut x = vec![0.0_f32; n * rhs];
    for r in 0..n {
        for c in 0..rhs {
            x[r * rhs + c] = aug[r * w + n + c];
        }
    }
    Ok(x)
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..n * dim).map(|_| rng.next_normal_pair().0).collect()
    }

    fn small_cfg(dim: usize) -> AqConfig {
        AqConfig {
            m: 2,
            k: 8,
            dim,
            beam_width: 4,
            n_iter: 5,
            ridge: 1e-4,
        }
    }

    #[test]
    fn aq_train_shapes() {
        let dim = 6;
        let n = 80;
        let data = rand_data(n, dim, 1);
        let mut rng = LcgRng::new(7);
        let q = AdditiveQuantizer::train(&data, n, &small_cfg(dim), &mut rng)
            .expect("valid training configuration");
        assert_eq!(q.m(), 2);
        assert_eq!(q.k(), 8);
        assert_eq!(q.dim(), dim);
        assert_eq!(q.codebook(0).expect("valid codebook index").len(), 8 * dim);
        assert_eq!(q.codebook(1).expect("valid codebook index").len(), 8 * dim);
    }

    #[test]
    fn aq_encode_len_equals_m() {
        let dim = 5;
        let n = 60;
        let data = rand_data(n, dim, 2);
        let mut rng = LcgRng::new(8);
        let q = AdditiveQuantizer::train(&data, n, &small_cfg(dim), &mut rng)
            .expect("valid training configuration");
        let codes = q.encode(&data[0..dim]).expect("valid vector dimension");
        assert_eq!(codes.len(), 2);
        assert!(codes.iter().all(|&c| (c as usize) < q.k()));
    }

    #[test]
    fn aq_decode_roundtrip_shape() {
        let dim = 4;
        let n = 50;
        let data = rand_data(n, dim, 3);
        let mut rng = LcgRng::new(9);
        let q = AdditiveQuantizer::train(&data, n, &small_cfg(dim), &mut rng)
            .expect("valid training configuration");
        let codes = q.encode(&data[0..dim]).expect("valid vector dimension");
        let recon = q.decode(&codes).expect("valid code length");
        assert_eq!(recon.len(), dim);
        assert!(recon.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn aq_reconstruction_finite_nonneg() {
        let dim = 6;
        let n = 70;
        let data = rand_data(n, dim, 4);
        let mut rng = LcgRng::new(10);
        let q = AdditiveQuantizer::train(&data, n, &small_cfg(dim), &mut rng)
            .expect("valid training configuration");
        for i in 0..n {
            let err = q
                .reconstruction_error(&data[i * dim..(i + 1) * dim])
                .expect("valid vector dimension");
            assert!(err.is_finite() && err >= 0.0, "err={err}");
        }
    }

    #[test]
    fn aq_distortion_below_input_energy() {
        // Average reconstruction error should be far below the average vector
        // energy: the quantizer must actually capture structure.
        let dim = 8;
        let n = 120;
        let data = rand_data(n, dim, 5);
        let mut rng = LcgRng::new(11);
        let cfg = AqConfig {
            m: 3,
            k: 16,
            dim,
            beam_width: 8,
            n_iter: 8,
            ridge: 1e-4,
        };
        let q = AdditiveQuantizer::train(&data, n, &cfg, &mut rng)
            .expect("valid training configuration");
        let mut total_err = 0.0_f32;
        let mut total_energy = 0.0_f32;
        for i in 0..n {
            let x = &data[i * dim..(i + 1) * dim];
            total_err += q.reconstruction_error(x).expect("valid vector dimension");
            total_energy += squared_norm(x);
        }
        assert!(
            total_err < 0.6 * total_energy,
            "distortion not reduced: err={total_err} energy={total_energy}"
        );
    }

    #[test]
    fn aq_more_codebooks_lower_error() {
        // Increasing m must not increase the (training) reconstruction error
        // by much — additive quantization is monotone-ish with the alternating
        // refit. We test that m=4 is at least as good as m=1 on average.
        let dim = 6;
        let n = 100;
        let data = rand_data(n, dim, 6);

        let avg_err = |m: usize| -> f32 {
            let mut rng = LcgRng::new(123);
            let cfg = AqConfig {
                m,
                k: 16,
                dim,
                beam_width: 6,
                n_iter: 8,
                ridge: 1e-4,
            };
            let q = AdditiveQuantizer::train(&data, n, &cfg, &mut rng)
                .expect("valid training configuration");
            (0..n)
                .map(|i| {
                    q.reconstruction_error(&data[i * dim..(i + 1) * dim])
                        .expect("valid vector dimension")
                })
                .sum::<f32>()
                / n as f32
        };

        let e1 = avg_err(1);
        let e4 = avg_err(4);
        assert!(e4 <= e1 + 1e-3, "m=4 err {e4} should be ≤ m=1 err {e1}");
    }

    #[test]
    fn aq_adc_matches_exact_distance() {
        let dim = 5;
        let n = 60;
        let data = rand_data(n, dim, 7);
        let mut rng = LcgRng::new(12);
        let q = AdditiveQuantizer::train(&data, n, &small_cfg(dim), &mut rng)
            .expect("valid training configuration");

        let query = rand_data(1, dim, 99);
        let q_norm_sq = squared_norm(&query);
        let dot_table = q.build_dot_table(&query).expect("valid query dimension");
        let self_dot = q.self_dot_table();

        for i in 0..10 {
            let x = &data[i * dim..(i + 1) * dim];
            let codes = q.encode(x).expect("valid vector dimension");
            let recon = q.decode(&codes).expect("valid code length");
            let exact = l2_sq(&query, &recon);
            let adc = q
                .adc_distance(q_norm_sq, &dot_table, &self_dot, &codes)
                .expect("valid code length and tables");
            assert!((exact - adc).abs() < 1e-2, "exact={exact} adc={adc}");
        }
    }

    #[test]
    fn aq_dot_table_dimensions() {
        let dim = 4;
        let n = 40;
        let data = rand_data(n, dim, 8);
        let mut rng = LcgRng::new(13);
        let q = AdditiveQuantizer::train(&data, n, &small_cfg(dim), &mut rng)
            .expect("valid training configuration");
        let table = q
            .build_dot_table(&data[0..dim])
            .expect("valid query dimension");
        assert_eq!(table.len(), q.m() * q.k());
        let g = q.self_dot_table();
        assert_eq!(g.len(), (q.m() * q.k()).pow(2));
    }

    #[test]
    fn aq_beam_width_one_still_works() {
        let dim = 5;
        let n = 50;
        let data = rand_data(n, dim, 9);
        let mut rng = LcgRng::new(14);
        let cfg = AqConfig {
            m: 2,
            k: 8,
            dim,
            beam_width: 1,
            n_iter: 5,
            ridge: 1e-4,
        };
        let q = AdditiveQuantizer::train(&data, n, &cfg, &mut rng)
            .expect("valid training configuration");
        let codes = q.encode(&data[0..dim]).expect("valid vector dimension");
        assert_eq!(codes.len(), 2);
    }

    #[test]
    fn aq_wider_beam_not_worse() {
        // A wider beam should find an encoding with error ≤ a narrow beam for
        // the same trained codebooks.
        let dim = 6;
        let n = 80;
        let data = rand_data(n, dim, 10);
        let mut rng = LcgRng::new(15);
        let cfg = AqConfig {
            m: 3,
            k: 16,
            dim,
            beam_width: 8,
            n_iter: 6,
            ridge: 1e-4,
        };
        let q = AdditiveQuantizer::train(&data, n, &cfg, &mut rng)
            .expect("valid training configuration");
        let x = &data[0..dim];
        let narrow = q
            .encode_with_beam(x, 1)
            .expect("valid beam encoding parameters");
        let wide = q
            .encode_with_beam(x, 16)
            .expect("valid beam encoding parameters");
        let e_narrow = l2_sq(x, &q.decode(&narrow).expect("valid code length"));
        let e_wide = l2_sq(x, &q.decode(&wide).expect("valid code length"));
        assert!(e_wide <= e_narrow + 1e-5, "wide={e_wide} narrow={e_narrow}");
    }

    #[test]
    fn aq_err_empty_input() {
        let mut rng = LcgRng::new(16);
        let cfg = small_cfg(4);
        let err = AdditiveQuantizer::train(&[], 0, &cfg, &mut rng).unwrap_err();
        assert!(matches!(err, AnnError::EmptyInput));
    }

    #[test]
    fn aq_err_dim_mismatch() {
        let mut rng = LcgRng::new(17);
        let cfg = small_cfg(4);
        let err = AdditiveQuantizer::train(&[1.0, 2.0, 3.0], 5, &cfg, &mut rng).unwrap_err();
        assert!(matches!(err, AnnError::DimensionMismatch { .. }));
    }

    #[test]
    fn aq_err_bad_k() {
        let mut rng = LcgRng::new(18);
        let cfg = AqConfig {
            m: 2,
            k: 512,
            dim: 4,
            beam_width: 2,
            n_iter: 3,
            ridge: 1e-4,
        };
        let data = rand_data(20, 4, 1);
        let err = AdditiveQuantizer::train(&data, 20, &cfg, &mut rng).unwrap_err();
        assert!(matches!(err, AnnError::InvalidK { .. }));
    }

    #[test]
    fn aq_err_zero_m() {
        let mut rng = LcgRng::new(19);
        let cfg = AqConfig {
            m: 0,
            k: 8,
            dim: 4,
            beam_width: 2,
            n_iter: 3,
            ridge: 1e-4,
        };
        let data = rand_data(20, 4, 1);
        let err = AdditiveQuantizer::train(&data, 20, &cfg, &mut rng).unwrap_err();
        assert!(matches!(err, AnnError::InvalidLayerCount { .. }));
    }

    #[test]
    fn aq_encode_dim_mismatch_errors() {
        let dim = 4;
        let n = 40;
        let data = rand_data(n, dim, 20);
        let mut rng = LcgRng::new(21);
        let q = AdditiveQuantizer::train(&data, n, &small_cfg(dim), &mut rng)
            .expect("valid training configuration");
        let err = q.encode(&[1.0, 2.0]).unwrap_err();
        assert!(matches!(err, AnnError::DimensionMismatch { .. }));
    }

    #[test]
    fn aq_solve_identity() {
        // Sanity check the linear solver on a known 2×2 system.
        let a = vec![2.0_f32, 0.0, 0.0, 4.0];
        let b = vec![6.0_f32, 8.0];
        let x = solve_spd_multi(&a, &b, 2, 1).expect("valid linear system parameters");
        assert!((x[0] - 3.0).abs() < 1e-5);
        assert!((x[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn aq_decode_known_codes() {
        // After training, decode must equal the sum of the chosen codewords.
        let dim = 4;
        let n = 40;
        let data = rand_data(n, dim, 22);
        let mut rng = LcgRng::new(23);
        let q = AdditiveQuantizer::train(&data, n, &small_cfg(dim), &mut rng)
            .expect("valid training configuration");
        let codes = vec![1u8, 3u8];
        let recon = q.decode(&codes).expect("valid code length");
        let mut expected = vec![0.0_f32; dim];
        for (t, &c) in codes.iter().enumerate() {
            let cw = q
                .codeword(t, c as usize)
                .expect("valid codebook and codeword indices");
            for d in 0..dim {
                expected[d] += cw[d];
            }
        }
        for d in 0..dim {
            assert!((recon[d] - expected[d]).abs() < 1e-6);
        }
    }
}
