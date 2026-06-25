//! Residual Vector Quantization (RVQ) — the neural-codec core shared by
//! SoundStream (Zeghidour 2021), EnCodec (Défossez 2022) and Bark-style
//! acoustic tokenisers.
//!
//! # Algorithm
//!
//! An RVQ encodes a vector `x ∈ ℝ^dim` into `n_quantizers` discrete indices.
//! Each stage owns an independent codebook of `codebook_size` vectors.  Encoding
//! is **greedy and sequential**: stage `s` quantises the *running residual* to
//! its codebook's Euclidean nearest entry, subtracts that entry, and passes the
//! new residual to stage `s+1`:
//!
//! ```text
//! r₀ = x
//! for s in 0..n_quantizers:
//!     code_s = argmin_j ‖ r_s − codebook_s[j] ‖
//!     r_{s+1} = r_s − codebook_s[code_s]
//! x̂ = Σ_s codebook_s[code_s]            (decode = sum of the chosen entries)
//! ```
//!
//! # Monotone residual descent (the defining property)
//!
//! For the reconstruction error to be **non-increasing in the number of
//! stages** for *every* input (not merely on average), a stage must always be
//! able to *add nothing*.  [`ResidualVectorQuantizer::new`] therefore reserves
//! **index 0 of every codebook as the zero vector**: since `‖r − 0‖ = ‖r‖` is
//! always a candidate, the chosen entry satisfies `‖r − c*‖ ≤ ‖r‖`, so
//! `‖r_{s+1}‖ ≤ ‖r_s‖` holds *exactly* at every stage.  This is what makes the
//! classic "more quantizers ⇒ lower error" guarantee true per-input rather than
//! statistical.  [`ResidualVectorQuantizer::fit`] preserves the pinned zero row.
//!
//! Codebooks supplied verbatim via [`ResidualVectorQuantizer::from_codebooks`]
//! are *not* assumed to contain the zero vector; monotonicity then holds only to
//! the extent the supplied codebooks provide a near-zero option.
//!
//! # Scope
//!
//! This is the rigorously CPU-verifiable codec core (round-trip, nestedness,
//! exact nearest-neighbour, k-means adaptation).  The *trained* autoregressive
//! transformers that generate Bark semantic / coarse / fine tokens from text are
//! **out of scope** — they require training-scale data and are not unit
//! verifiable.  See [`crate::codec::bark`] for the token-layout wrapper and
//! `TODO.md`.

use crate::error::{AudioError, AudioResult};
use crate::handle::LcgRng;

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Squared Euclidean distance between two equal-length slices.
#[inline]
fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Euclidean (L2) distance between two equal-length slices.
#[inline]
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    squared_distance(a, b).sqrt()
}

/// Index of the codebook row (one of `k` rows of length `dim`) nearest to `v`
/// under Euclidean distance.  Ties resolve to the lowest index (strict `<`).
///
/// This is the canonical brute-force `argmin` the encoder uses at every stage.
#[inline]
fn nearest_row(codebook: &[f32], v: &[f32], k: usize, dim: usize) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for c in 0..k {
        let row = &codebook[c * dim..(c + 1) * dim];
        let d = squared_distance(v, row);
        if d < best_d {
            best_d = d;
            best = c;
        }
    }
    best
}

// ─── RvqFitReport ────────────────────────────────────────────────────────────

/// Outcome of a [`ResidualVectorQuantizer::fit`] pass.
///
/// `error_before` / `error_after` are the **mean per-vector squared
/// reconstruction error** over the training batch, before and after the
/// adaptation.  By construction `error_after ≤ error_before` (a pass that would
/// increase the batch error is rejected and the prior codebooks are restored).
#[derive(Debug, Clone, Copy)]
pub struct RvqFitReport {
    /// Mean squared reconstruction error over the batch before fitting.
    pub error_before: f32,
    /// Mean squared reconstruction error over the batch after fitting
    /// (guaranteed `≤ error_before`).
    pub error_after: f32,
}

// ─── ResidualVectorQuantizer ─────────────────────────────────────────────────

/// Multi-stage residual vector quantizer (RVQ neural-codec core).
///
/// Holds `n_quantizers` independent codebooks, each a row-major
/// `[codebook_size, dim]` table.  See the module docs for the algorithm and the
/// reserved-zero-code monotonicity guarantee.
#[derive(Debug, Clone)]
pub struct ResidualVectorQuantizer {
    /// Per-stage codebooks; `codebooks[s]` is row-major `[codebook_size, dim]`.
    codebooks: Vec<Vec<f32>>,
    /// Number of quantization stages.
    n_quantizers: usize,
    /// Entries per stage codebook.
    codebook_size: usize,
    /// Vector dimensionality.
    dim: usize,
    /// Whether index 0 of every codebook is pinned to the zero vector
    /// (guarantees the monotone residual-descent property).
    reserve_zero: bool,
}

impl ResidualVectorQuantizer {
    /// Build an RVQ with random codebooks (deterministic given `rng`).
    ///
    /// Rows `1..codebook_size` of each stage are initialised with N(0, 1)
    /// samples scaled by `1/√dim`; **row 0 of every stage is the zero vector**,
    /// which guarantees the monotone residual-descent property (see module
    /// docs).  Use [`ResidualVectorQuantizer::fit`] to adapt the codebooks to
    /// data.
    ///
    /// # Errors
    ///
    /// - [`AudioError::EmptyInput`] if `n_quantizers == 0`.
    /// - [`AudioError::InvalidVocabSize`] if `codebook_size == 0`.
    /// - [`AudioError::InvalidEmbedDim`] if `dim == 0`.
    pub fn new(
        n_quantizers: usize,
        codebook_size: usize,
        dim: usize,
        rng: &mut LcgRng,
    ) -> AudioResult<Self> {
        if n_quantizers == 0 {
            return Err(AudioError::EmptyInput {
                msg: "rvq: n_quantizers must be > 0".into(),
            });
        }
        if codebook_size == 0 {
            return Err(AudioError::InvalidVocabSize(0));
        }
        if dim == 0 {
            return Err(AudioError::InvalidEmbedDim(0));
        }

        let scale = 1.0 / (dim as f32).sqrt();
        let mut codebooks = Vec::with_capacity(n_quantizers);
        for _ in 0..n_quantizers {
            let mut cb = vec![0.0_f32; codebook_size * dim];
            // Reserve row 0 as the zero vector; randomise the remaining rows.
            let start = dim.min(cb.len());
            rng.fill_normal(&mut cb[start..]);
            for v in &mut cb[start..] {
                *v *= scale;
            }
            codebooks.push(cb);
        }

        Ok(Self {
            codebooks,
            n_quantizers,
            codebook_size,
            dim,
            reserve_zero: true,
        })
    }

    /// Build an RVQ from explicit per-stage codebooks (taken verbatim).
    ///
    /// Each `codebooks[s]` must be a row-major `[codebook_size, dim]` table.
    /// The zero vector is **not** forced, so the monotone-descent guarantee
    /// holds only insofar as the supplied codebooks provide a near-zero option.
    ///
    /// # Errors
    ///
    /// - [`AudioError::EmptyInput`] if `codebooks` is empty.
    /// - [`AudioError::InvalidVocabSize`] if `codebook_size == 0`.
    /// - [`AudioError::InvalidEmbedDim`] if `dim == 0`.
    /// - [`AudioError::WeightShapeMismatch`] if any stage's length is not
    ///   `codebook_size * dim`.
    pub fn from_codebooks(
        codebooks: Vec<Vec<f32>>,
        codebook_size: usize,
        dim: usize,
    ) -> AudioResult<Self> {
        if codebooks.is_empty() {
            return Err(AudioError::EmptyInput {
                msg: "rvq from_codebooks: no stages".into(),
            });
        }
        if codebook_size == 0 {
            return Err(AudioError::InvalidVocabSize(0));
        }
        if dim == 0 {
            return Err(AudioError::InvalidEmbedDim(0));
        }
        for (s, cb) in codebooks.iter().enumerate() {
            if cb.len() != codebook_size * dim {
                return Err(AudioError::WeightShapeMismatch {
                    msg: format!(
                        "rvq stage {s}: codebook len {} != codebook_size*dim {}",
                        cb.len(),
                        codebook_size * dim
                    ),
                });
            }
        }
        let n_quantizers = codebooks.len();
        Ok(Self {
            codebooks,
            n_quantizers,
            codebook_size,
            dim,
            reserve_zero: false,
        })
    }

    /// Encode `x` (length `dim`) into one code index per stage.
    ///
    /// # Errors
    ///
    /// [`AudioError::ShapeMismatch`] if `x.len() != dim`.
    pub fn encode(&self, x: &[f32]) -> AudioResult<Vec<usize>> {
        self.encode_stages(x, self.n_quantizers)
    }

    /// Encode `x` using only the first `n_active` stages.
    ///
    /// Because the first `n_active` codes are identical to the prefix of a full
    /// encode, this drives the nestedness sweep (`decode(&codes[..m])`).
    ///
    /// # Errors
    ///
    /// - [`AudioError::ShapeMismatch`] if `x.len() != dim`.
    /// - [`AudioError::ShapeMismatch`] if `n_active > n_quantizers`.
    pub fn encode_stages(&self, x: &[f32], n_active: usize) -> AudioResult<Vec<usize>> {
        if x.len() != self.dim {
            return Err(AudioError::ShapeMismatch {
                msg: format!("rvq encode: x.len()={} != dim={}", x.len(), self.dim),
            });
        }
        if n_active > self.n_quantizers {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "rvq encode: n_active={} > n_quantizers={}",
                    n_active, self.n_quantizers
                ),
            });
        }
        let mut residual = x.to_vec();
        let mut codes = Vec::with_capacity(n_active);
        for s in 0..n_active {
            let cb = &self.codebooks[s];
            let idx = nearest_row(cb, &residual, self.codebook_size, self.dim);
            let row = &cb[idx * self.dim..(idx + 1) * self.dim];
            for (r, &e) in residual.iter_mut().zip(row) {
                *r -= e;
            }
            codes.push(idx);
        }
        Ok(codes)
    }

    /// Decode `codes` back to `x̂` (length `dim`) as the sum of the selected
    /// codebook entries.
    ///
    /// `codes` may be shorter than `n_quantizers` (a partial / lower-fidelity
    /// reconstruction using the first `codes.len()` stages).
    ///
    /// # Errors
    ///
    /// - [`AudioError::ShapeMismatch`] if `codes.len() > n_quantizers`.
    /// - [`AudioError::ShapeMismatch`] if any code is `>= codebook_size`.
    pub fn decode(&self, codes: &[usize]) -> AudioResult<Vec<f32>> {
        if codes.len() > self.n_quantizers {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "rvq decode: {} codes > n_quantizers {}",
                    codes.len(),
                    self.n_quantizers
                ),
            });
        }
        let mut x_hat = vec![0.0_f32; self.dim];
        for (s, &idx) in codes.iter().enumerate() {
            if idx >= self.codebook_size {
                return Err(AudioError::ShapeMismatch {
                    msg: format!(
                        "rvq decode: code {idx} >= codebook_size {} at stage {s}",
                        self.codebook_size
                    ),
                });
            }
            let row = &self.codebooks[s][idx * self.dim..(idx + 1) * self.dim];
            for (h, &e) in x_hat.iter_mut().zip(row) {
                *h += e;
            }
        }
        Ok(x_hat)
    }

    /// Encode then decode `x`, returning `(x̂, codes, residual_norm)` where
    /// `residual_norm = ‖x − x̂‖₂`.
    ///
    /// # Errors
    ///
    /// [`AudioError::ShapeMismatch`] if `x.len() != dim`.
    pub fn quantize(&self, x: &[f32]) -> AudioResult<(Vec<f32>, Vec<usize>, f32)> {
        let codes = self.encode(x)?;
        let x_hat = self.decode(&codes)?;
        let residual_norm = l2_distance(x, &x_hat);
        Ok((x_hat, codes, residual_norm))
    }

    /// Full-stage reconstruction error `‖x − decode(encode(x))‖₂`.
    ///
    /// # Errors
    ///
    /// [`AudioError::ShapeMismatch`] if `x.len() != dim`.
    pub fn reconstruction_error(&self, x: &[f32]) -> AudioResult<f32> {
        let codes = self.encode(x)?;
        let x_hat = self.decode(&codes)?;
        Ok(l2_distance(x, &x_hat))
    }

    /// Mean per-vector **squared** reconstruction error over a `[n, dim]` batch.
    ///
    /// # Errors
    ///
    /// - [`AudioError::DimensionMismatch`] if `dim` differs from the codec's.
    /// - [`AudioError::ShapeMismatch`] if `data.len() != n * dim`.
    /// - [`AudioError::EmptyInput`] if `n == 0`.
    pub fn batch_mean_sq_error(&self, data: &[f32], n: usize, dim: usize) -> AudioResult<f32> {
        self.check_matrix(data, n, dim)?;
        if n == 0 {
            return Err(AudioError::EmptyInput {
                msg: "rvq batch error: n == 0".into(),
            });
        }
        let mut total = 0.0_f32;
        for i in 0..n {
            let x = &data[i * dim..(i + 1) * dim];
            let codes = self.encode(x)?;
            let x_hat = self.decode(&codes)?;
            total += squared_distance(x, &x_hat);
        }
        Ok(total / n as f32)
    }

    /// Adapt the codebooks to a `[n, dim]` batch by **stage-wise residual
    /// k-means** (Lloyd iterations on each stage's running residuals — exactly
    /// how SoundStream / EnCodec codebooks are learned offline).
    ///
    /// The pass is *guarded*: if it would increase the batch reconstruction
    /// error it is rejected and the prior codebooks are restored, so the
    /// returned `error_after ≤ error_before` unconditionally.  On clusterable
    /// data the error strictly decreases.  A pinned zero row (from
    /// [`ResidualVectorQuantizer::new`]) is preserved, keeping the monotone
    /// residual-descent guarantee intact after fitting.
    ///
    /// # Errors
    ///
    /// - [`AudioError::DimensionMismatch`] if `dim` differs from the codec's.
    /// - [`AudioError::ShapeMismatch`] if `data.len() != n * dim`.
    /// - [`AudioError::EmptyInput`] if `n == 0`.
    pub fn fit(
        &mut self,
        data: &[f32],
        n: usize,
        dim: usize,
        n_iter: usize,
        rng: &mut LcgRng,
    ) -> AudioResult<RvqFitReport> {
        self.check_matrix(data, n, dim)?;
        if n == 0 {
            return Err(AudioError::EmptyInput {
                msg: "rvq fit: no data".into(),
            });
        }

        let error_before = self.batch_mean_sq_error(data, n, dim)?;
        let backup = self.codebooks.clone();

        let k = self.codebook_size;
        let d = self.dim;
        let mut residuals = data.to_vec();
        let mut sums = vec![0.0_f32; k * d];
        let mut counts = vec![0_usize; k];

        for s in 0..self.n_quantizers {
            for _ in 0..n_iter {
                sums.fill(0.0);
                counts.fill(0);

                // Assignment + accumulation over the current residual stream.
                for i in 0..n {
                    let r = &residuals[i * d..(i + 1) * d];
                    let cid = nearest_row(&self.codebooks[s], r, k, d);
                    counts[cid] += 1;
                    let acc = &mut sums[cid * d..(cid + 1) * d];
                    for (a, &val) in acc.iter_mut().zip(r) {
                        *a += val;
                    }
                }

                // Centroid update (mean of assigned residuals).
                for c in 0..k {
                    if self.reserve_zero && c == 0 {
                        // Keep the reserved zero row pinned at the origin.
                        for v in &mut self.codebooks[s][0..d] {
                            *v = 0.0;
                        }
                        continue;
                    }
                    if counts[c] == 0 {
                        // Re-seed an empty cluster from a random residual.
                        let ri = rng.next_usize(n);
                        self.codebooks[s][c * d..(c + 1) * d]
                            .copy_from_slice(&residuals[ri * d..(ri + 1) * d]);
                    } else {
                        let inv = 1.0 / counts[c] as f32;
                        let src = &sums[c * d..(c + 1) * d];
                        let dst = &mut self.codebooks[s][c * d..(c + 1) * d];
                        for (out, &acc) in dst.iter_mut().zip(src) {
                            *out = acc * inv;
                        }
                    }
                }
            }

            // Advance the residual stream through the freshly fitted stage.
            for i in 0..n {
                let cid = nearest_row(&self.codebooks[s], &residuals[i * d..(i + 1) * d], k, d);
                let row = self.codebooks[s][cid * d..(cid + 1) * d].to_vec();
                let r = &mut residuals[i * d..(i + 1) * d];
                for (rr, &e) in r.iter_mut().zip(&row) {
                    *rr -= e;
                }
            }
        }

        let error_after = self.batch_mean_sq_error(data, n, dim)?;
        if error_after > error_before {
            // Guard: never let a fit pass increase the batch error.
            self.codebooks = backup;
            Ok(RvqFitReport {
                error_before,
                error_after: error_before,
            })
        } else {
            Ok(RvqFitReport {
                error_before,
                error_after,
            })
        }
    }

    /// Number of quantization stages.
    #[must_use]
    pub fn n_quantizers(&self) -> usize {
        self.n_quantizers
    }

    /// Entries per stage codebook.
    #[must_use]
    pub fn codebook_size(&self) -> usize {
        self.codebook_size
    }

    /// Vector dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Whether index 0 of every codebook is the pinned zero vector.
    #[must_use]
    pub fn reserves_zero_code(&self) -> bool {
        self.reserve_zero
    }

    /// Borrow stage `stage`'s codebook as a row-major `[codebook_size, dim]`
    /// slice.
    ///
    /// # Errors
    ///
    /// [`AudioError::ShapeMismatch`] if `stage >= n_quantizers`.
    pub fn codebook(&self, stage: usize) -> AudioResult<&[f32]> {
        if stage >= self.n_quantizers {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "rvq codebook: stage {stage} >= n_quantizers {}",
                    self.n_quantizers
                ),
            });
        }
        Ok(&self.codebooks[stage])
    }

    /// Validate a `[n, dim]` matrix against the codec dimensionality.
    fn check_matrix(&self, data: &[f32], n: usize, dim: usize) -> AudioResult<()> {
        if dim != self.dim {
            return Err(AudioError::DimensionMismatch {
                expected: self.dim,
                got: dim,
            });
        }
        if data.len() != n * dim {
            return Err(AudioError::ShapeMismatch {
                msg: format!("rvq: data.len()={} != n*dim={}", data.len(), n * dim),
            });
        }
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Euclidean distance between two equal-length slices (test helper).
    fn dist(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum::<f32>()
            .sqrt()
    }

    /// TEST 1 — round-trip / nestedness: reconstruction error is monotone
    /// non-increasing as the number of active stages grows.
    #[test]
    fn round_trip_nestedness_monotone() {
        let mut rng = LcgRng::new(20_240_617);
        let n_q = 6usize;
        let k = 8usize;
        let dim = 8usize;
        let rvq = ResidualVectorQuantizer::new(n_q, k, dim, &mut rng).expect("new ok");

        let mut x = vec![0.0_f32; dim];
        LcgRng::new(7).fill_normal(&mut x);
        let codes = rvq.encode(&x).expect("encode ok");
        assert_eq!(codes.len(), n_q);

        let x_norm = dist(&x, &vec![0.0; dim]);
        let mut prev = f32::INFINITY;
        let mut errs = Vec::with_capacity(n_q + 1);
        for m in 0..=n_q {
            let x_hat = rvq.decode(&codes[..m]).expect("decode ok");
            assert_eq!(x_hat.len(), dim);
            let e = dist(&x, &x_hat);
            assert!(e.is_finite(), "stage {m}: non-finite error");
            assert!(
                e <= prev + 1e-6,
                "non-monotone: stage {m} err {e} > prev {prev}"
            );
            prev = e;
            errs.push(e);
        }
        // Zero stages ⇒ error is exactly ‖x‖.
        assert!(
            (errs[0] - x_norm).abs() < 1e-5,
            "errs[0]={} ‖x‖={x_norm}",
            errs[0]
        );
        // Full encode does not exceed the no-op reconstruction.
        assert!(errs[n_q] <= errs[0]);
    }

    /// TEST 2 — exactness: if `x` is an exact sum of one entry per codebook,
    /// encode recovers those indices and decode reconstructs `x`.
    #[test]
    fn exact_sum_recovered() {
        // Geometric-scale orthonormal codebooks: stage s, row j = α^s · e_j.
        // Distinct stage scales + orthonormal anchors make greedy decoding exact.
        let dim = 4usize;
        let k = 4usize;
        let n_q = 3usize;
        let alpha = 0.2_f32;

        let mut codebooks = Vec::with_capacity(n_q);
        for s in 0..n_q {
            let scale = alpha.powi(s as i32);
            let mut cb = vec![0.0_f32; k * dim];
            for j in 0..k.min(dim) {
                cb[j * dim + j] = scale;
            }
            codebooks.push(cb);
        }
        let rvq = ResidualVectorQuantizer::from_codebooks(codebooks, k, dim).expect("build ok");

        let target = [1usize, 2, 3];
        let mut x = vec![0.0_f32; dim];
        for (s, &idx) in target.iter().enumerate() {
            x[idx] += alpha.powi(s as i32);
        }

        let codes = rvq.encode(&x).expect("encode ok");
        assert_eq!(codes, target.to_vec(), "indices not exactly recovered");
        let x_hat = rvq.decode(&codes).expect("decode ok");
        let err = dist(&x, &x_hat);
        assert!(err <= 1e-5, "exact reconstruction error {err} > 1e-5");
    }

    /// TEST 3 — NN correctness: each stage's chosen code is the true Euclidean
    /// nearest neighbour of the residual (cross-checked against brute force).
    #[test]
    fn each_stage_is_true_nearest_neighbour() {
        let mut rng = LcgRng::new(123);
        let n_q = 5usize;
        let k = 7usize;
        let dim = 6usize;
        let rvq = ResidualVectorQuantizer::new(n_q, k, dim, &mut rng).expect("new ok");

        let mut x = vec![0.0_f32; dim];
        LcgRng::new(999).fill_normal(&mut x);
        let codes = rvq.encode(&x).expect("encode ok");

        for s in 0..n_q {
            // Residual entering stage s = x − decode(first s stages).
            let prefix = rvq.decode(&codes[..s]).expect("decode prefix ok");
            let residual: Vec<f32> = x.iter().zip(&prefix).map(|(a, b)| a - b).collect();

            // Independent brute-force argmin over stage s's codebook.
            let cb = rvq.codebook(s).expect("codebook ok");
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let row = &cb[c * dim..(c + 1) * dim];
                let d: f32 = residual
                    .iter()
                    .zip(row)
                    .map(|(a, b)| {
                        let e = a - b;
                        e * e
                    })
                    .sum();
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            assert_eq!(best, codes[s], "stage {s}: encoder code != brute-force NN");
        }
    }

    /// TEST 4 — fit (k-means) does not increase mean reconstruction error, and
    /// strictly improves it on clusterable data.
    #[test]
    fn fit_does_not_increase_error() {
        let dim = 5usize;
        let n_blobs = 4usize;
        let per = 20usize;
        let n = n_blobs * per;

        // Cluster centres, well separated.
        let mut centers = vec![0.0_f32; n_blobs * dim];
        LcgRng::new(1).fill_normal(&mut centers);
        for v in &mut centers {
            *v *= 3.0;
        }
        // Tight Gaussian blobs around the centres.
        let mut noise = vec![0.0_f32; n * dim];
        LcgRng::new(2).fill_normal(&mut noise);
        let mut data = vec![0.0_f32; n * dim];
        for i in 0..n {
            let b = i % n_blobs;
            for c in 0..dim {
                data[i * dim + c] = centers[b * dim + c] + 0.2 * noise[i * dim + c];
            }
        }

        let mut rng = LcgRng::new(42);
        let mut rvq = ResidualVectorQuantizer::new(4, 6, dim, &mut rng).expect("new ok");
        let before = rvq.batch_mean_sq_error(&data, n, dim).expect("err ok");
        let report = rvq.fit(&data, n, dim, 15, &mut rng).expect("fit ok");
        let after = rvq.batch_mean_sq_error(&data, n, dim).expect("err ok");

        assert!((report.error_before - before).abs() < 1e-4);
        assert!((report.error_after - after).abs() < 1e-4);
        assert!(
            after <= before + 1e-6,
            "fit increased error: {before} -> {after}"
        );
        assert!(
            after < before,
            "fit should improve clustered data: {before} -> {after}"
        );
        assert!(after.is_finite());

        // The monotone-descent guarantee still holds after fitting (zero row).
        assert!(rvq.reserves_zero_code());
        let mut x = vec![0.0_f32; dim];
        LcgRng::new(3).fill_normal(&mut x);
        let codes = rvq.encode(&x).expect("encode ok");
        let mut prev = f32::INFINITY;
        for m in 0..=4 {
            let e = dist(&x, &rvq.decode(&codes[..m]).expect("decode ok"));
            assert!(e <= prev + 1e-6, "post-fit non-monotone at {m}");
            prev = e;
        }
    }

    /// TEST 5 — determinism, shapes, finiteness.
    #[test]
    fn deterministic_shapes_finite() {
        let mut a = LcgRng::new(2024);
        let mut b = LcgRng::new(2024);
        let r1 = ResidualVectorQuantizer::new(4, 8, 6, &mut a).expect("new ok");
        let r2 = ResidualVectorQuantizer::new(4, 8, 6, &mut b).expect("new ok");

        let mut x = vec![0.0_f32; 6];
        LcgRng::new(5).fill_normal(&mut x);

        let (xh1, c1, n1) = r1.quantize(&x).expect("quantize ok");
        let (xh2, c2, n2) = r2.quantize(&x).expect("quantize ok");
        assert_eq!(c1, c2, "non-deterministic codes");
        assert_eq!(xh1, xh2, "non-deterministic reconstruction");
        assert!((n1 - n2).abs() < 1e-9);
        assert_eq!(xh1.len(), 6, "wrong reconstruction shape");
        assert_eq!(c1.len(), 4, "wrong code count");
        assert!(xh1.iter().all(|v| v.is_finite()) && n1.is_finite());
        for s in 0..4 {
            assert_eq!(r1.codebook(s).unwrap(), r2.codebook(s).unwrap());
        }

        // residual_norm matches reconstruction_error.
        let re = r1.reconstruction_error(&x).expect("re ok");
        assert!((re - n1).abs() < 1e-6);
    }

    /// Constructor + method validation paths.
    #[test]
    fn validation_errors() {
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            ResidualVectorQuantizer::new(0, 4, 4, &mut rng).unwrap_err(),
            AudioError::EmptyInput { .. }
        ));
        assert!(matches!(
            ResidualVectorQuantizer::new(2, 0, 4, &mut rng).unwrap_err(),
            AudioError::InvalidVocabSize(0)
        ));
        assert!(matches!(
            ResidualVectorQuantizer::new(2, 4, 0, &mut rng).unwrap_err(),
            AudioError::InvalidEmbedDim(0)
        ));

        let rvq = ResidualVectorQuantizer::new(3, 4, 5, &mut rng).expect("new ok");
        assert!(matches!(
            rvq.encode(&[0.0; 4]).unwrap_err(),
            AudioError::ShapeMismatch { .. }
        ));
        assert!(matches!(
            rvq.encode_stages(&[0.0; 5], 99).unwrap_err(),
            AudioError::ShapeMismatch { .. }
        ));
        assert!(matches!(
            rvq.decode(&[0, 1, 2, 3]).unwrap_err(),
            AudioError::ShapeMismatch { .. }
        ));
        assert!(matches!(
            rvq.decode(&[99]).unwrap_err(),
            AudioError::ShapeMismatch { .. }
        ));
        assert!(matches!(
            rvq.codebook(99).unwrap_err(),
            AudioError::ShapeMismatch { .. }
        ));
        assert!(matches!(
            rvq.batch_mean_sq_error(&[0.0; 4], 1, 4).unwrap_err(),
            AudioError::DimensionMismatch { .. }
        ));
        assert!(matches!(
            ResidualVectorQuantizer::from_codebooks(vec![vec![0.0; 3]], 2, 4).unwrap_err(),
            AudioError::WeightShapeMismatch { .. }
        ));
    }
}
