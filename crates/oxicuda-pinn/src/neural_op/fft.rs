//! Pure-Rust CPU fast Fourier transform for the Fourier Neural Operator.
//!
//! This module replaces the `O(N²)` brute-force DFT used by [`super::fno`] with
//! an `O(N log N)` transform that is numerically identical (to `f32` round-off)
//! to the textbook DFT defined there. No external FFT crate is used — every line
//! below is hand-written Cooley-Tukey / Bluestein math.
//!
//! # Complex representation
//!
//! To match the rest of [`super::fno`] exactly, complex sequences are passed as
//! *two parallel slices* `(real, imag)` of `f32`, and a real input is just the
//! `real` half with an implicit zero `imag`. The forward transform uses the same
//! sign convention as [`super::fno::dft_1d`]:
//!
//! ```text
//!            N-1
//!   X[k]  =  Σ   x[j] · exp(-i·2π·k·j / N)        (forward,  e^{-iθ})
//!            j=0
//!
//!            N-1
//!   x[j]  =  Σ   X[k] · exp(+i·2π·k·j / N) / N    (inverse,  e^{+iθ})
//!            k=0
//! ```
//!
//! The inverse [`ifft_1d`] returns only the **real part** of `x[j]`, exactly as
//! [`super::fno::idft_1d`] does (the imaginary part of the inverse of a spectrum
//! that came from a real signal is zero up to round-off).
//!
//! # Algorithms
//!
//! * **Radix-2 iterative Cooley-Tukey** (`fft_radix2`) — used when `N` is a
//!   power of two. The input is permuted by bit-reversal, then `log₂N` stages of
//!   butterflies combine length-`L` sub-transforms into length-`2L` ones. Stage
//!   `L` applies twiddle factors `Wₗ^k = exp(±i·2π·k/L)`. Each twiddle is
//!   recomputed from its angle (rather than by incremental complex rotation) so
//!   that rounding error does **not** accumulate across a stage.
//!
//! * **Bluestein chirp-z** (`bluestein`) — used for arbitrary (non-power-of-two)
//!   `N`. It rewrites the DFT as a convolution using the identity
//!   `k·j = (k² + j² − (k−j)²)/2`, so that
//!
//!   ```text
//!     X[k] = w[k] · Σ_j ( x[j]·w[j] ) · conj(w[k−j]) ,   w[m] = exp(-i·π·m²/N)
//!   ```
//!
//!   The inner sum is a linear convolution of the *chirp-premultiplied* signal
//!   `a[j] = x[j]·w[j]` with the symmetric chirp kernel `b[m] = conj(w[m])`. That
//!   convolution is evaluated by zero-padding both sequences to the next power of
//!   two `M ≥ 2N−1` and using the radix-2 transform above:
//!   `c = IFFT_M( FFT_M(a) ⊙ FFT_M(b) )`. Because the chirp kernel is even in
//!   `m` (`b[-m] = b[m]`), the negative-index half is placed at the wrap-around
//!   positions `M−m`, turning the linear convolution into a circular one with no
//!   aliasing for `M ≥ 2N−1`.
//!
//! Internally the working buffers are `f64`: the data carried by the FNO is `f32`,
//! but accumulating the butterflies and computing twiddle/chirp factors in double
//! precision keeps the result far inside the `f32` round-off of the reference DFT
//! (the dominant error term is the `O(N·εf32)` cancellation in the *reference*,
//! not in this transform). All public entry points take and return `f32`.

use std::f64::consts::PI;

/// In-place iterative radix-2 Cooley-Tukey FFT for power-of-two length.
///
/// Operates on the complex buffer `(re, im)` (both of length `n`, a power of two).
/// `inverse == false` applies the `e^{-iθ}` forward kernel; `inverse == true`
/// applies the `e^{+iθ}` kernel **without** the `1/N` normalization (callers that
/// need a true inverse divide afterwards).
///
/// # Panics
///
/// Never — the function is only reached with `re.len() == im.len()` a power of two
/// (guaranteed by [`fft_dispatch`]).
fn fft_radix2(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    if n <= 1 {
        return;
    }

    // ── Bit-reversal permutation ─────────────────────────────────────────────
    // Reorder the input so that decimation-in-time butterflies read contiguous
    // pairs. `j` walks the bit-reversed counter alongside the natural index `i`.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // ── Butterfly stages ─────────────────────────────────────────────────────
    // Stage combines sub-transforms of length `half` into length `len = 2*half`.
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        // Base angle step for this stage; twiddle k uses angle = base * k.
        let base = sign * 2.0 * PI / len as f64;
        let mut group = 0usize;
        while group < n {
            for k in 0..half {
                let angle = base * k as f64;
                let w_re = angle.cos();
                let w_im = angle.sin();

                let a_re = re[group + k];
                let a_im = im[group + k];
                // t = W^k · b
                let b_re = re[group + k + half];
                let b_im = im[group + k + half];
                let t_re = w_re * b_re - w_im * b_im;
                let t_im = w_re * b_im + w_im * b_re;

                re[group + k] = a_re + t_re;
                im[group + k] = a_im + t_im;
                re[group + k + half] = a_re - t_re;
                im[group + k + half] = a_im - t_im;
            }
            group += len;
        }
        len <<= 1;
    }
}

/// Chirp factor `w[m] = exp(sign · i · π · m² / N)` evaluated stably.
///
/// `m²` is reduced modulo `2N` before forming the angle: `exp(i·π·m²/N)` has
/// period `2N` in `m²`, so this keeps the trig argument small and accurate even
/// for large `m` (avoiding catastrophic loss of significance in `m²`).
#[inline]
fn chirp(m: usize, n: usize, sign: f64) -> (f64, f64) {
    let two_n = 2 * n as u64;
    let mm = (m as u64 * m as u64) % two_n;
    let angle = sign * PI * mm as f64 / n as f64;
    (angle.cos(), angle.sin())
}

/// Bluestein chirp-z transform for arbitrary (non-power-of-two) length.
///
/// Computes the same forward/inverse DFT as [`fft_radix2`] but for any `n ≥ 1`,
/// by expressing the DFT as a convolution and evaluating that convolution with a
/// power-of-two radix-2 FFT of length `M ≥ 2N−1`. The inverse variant
/// (`inverse == true`) is left **unnormalized**, matching [`fft_radix2`].
fn bluestein(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    if n <= 1 {
        return;
    }
    // Sign of the *outer* chirp: forward uses w[m]=e^{-iπm²/N}, inverse e^{+iπm²/N}.
    let sign = if inverse { 1.0 } else { -1.0 };

    // Convolution length: smallest power of two ≥ 2N-1.
    let mut m = 1usize;
    while m < 2 * n - 1 {
        m <<= 1;
    }

    // Precompute the chirp w[k] = exp(sign·iπk²/N) for k = 0..N.
    let mut w_re = vec![0.0_f64; n];
    let mut w_im = vec![0.0_f64; n];
    for (k, (wr, wi)) in w_re.iter_mut().zip(w_im.iter_mut()).enumerate() {
        let (c, s) = chirp(k, n, sign);
        *wr = c;
        *wi = s;
    }

    // a[j] = x[j] · w[j], zero-padded to length M.
    let mut a_re = vec![0.0_f64; m];
    let mut a_im = vec![0.0_f64; m];
    for j in 0..n {
        let xr = re[j];
        let xi = im[j];
        a_re[j] = xr * w_re[j] - xi * w_im[j];
        a_im[j] = xr * w_im[j] + xi * w_re[j];
    }

    // b[t] = conj(w[t]) for the symmetric (even) chirp kernel; negative indices
    // wrap to M−t so the circular convolution reproduces the linear one.
    let mut b_re = vec![0.0_f64; m];
    let mut b_im = vec![0.0_f64; m];
    b_re[0] = w_re[0];
    b_im[0] = -w_im[0];
    for t in 1..n {
        let cr = w_re[t];
        let ci = -w_im[t];
        b_re[t] = cr;
        b_im[t] = ci;
        b_re[m - t] = cr;
        b_im[m - t] = ci;
    }

    // c = IFFT_M( FFT_M(a) ⊙ FFT_M(b) ) — circular convolution of a and b.
    fft_radix2(&mut a_re, &mut a_im, false);
    fft_radix2(&mut b_re, &mut b_im, false);
    for t in 0..m {
        let pr = a_re[t] * b_re[t] - a_im[t] * b_im[t];
        let pi = a_re[t] * b_im[t] + a_im[t] * b_re[t];
        a_re[t] = pr;
        a_im[t] = pi;
    }
    fft_radix2(&mut a_re, &mut a_im, true);
    let inv_m = 1.0 / m as f64;

    // X[k] = w[k] · c[k].
    for k in 0..n {
        let cr = a_re[k] * inv_m;
        let ci = a_im[k] * inv_m;
        re[k] = cr * w_re[k] - ci * w_im[k];
        im[k] = cr * w_im[k] + ci * w_re[k];
    }
}

/// Dispatch to the radix-2 path for power-of-two `n`, else to Bluestein.
///
/// `inverse` selects the kernel sign; the result is **unnormalized** in both
/// cases (the `1/N` factor for an inverse transform is applied by the public
/// `ifft_*` wrappers).
fn fft_dispatch(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    if n <= 1 {
        return;
    }
    if n.is_power_of_two() {
        fft_radix2(re, im, inverse);
    } else {
        bluestein(re, im, inverse);
    }
}

/// Forward FFT of a **real** signal — drop-in fast replacement for
/// [`super::fno::dft_1d`].
///
/// Returns `(real, imag)` of length `N` using the `e^{-i·2π·k·j/N}` kernel,
/// matching the reference DFT element-wise to `f32` round-off.
pub fn fft_1d(x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = x.len();
    let mut re: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let mut im = vec![0.0_f64; n];
    fft_dispatch(&mut re, &mut im, false);
    (
        re.iter().map(|&v| v as f32).collect(),
        im.iter().map(|&v| v as f32).collect(),
    )
}

/// Inverse FFT returning the **real part** — drop-in fast replacement for
/// [`super::fno::idft_1d`].
///
/// Applies the `e^{+i·2π·k·j/N}` kernel, divides by `N`, and returns `Re{x[j]}`.
pub fn ifft_1d(real: &[f32], imag: &[f32]) -> Vec<f32> {
    let n = real.len();
    let mut re: Vec<f64> = real.iter().map(|&v| v as f64).collect();
    let mut im: Vec<f64> = imag.iter().map(|&v| v as f64).collect();
    fft_dispatch(&mut re, &mut im, true);
    let inv_n = 1.0 / n.max(1) as f64;
    re.iter().map(|&v| (v * inv_n) as f32).collect()
}

/// Forward FFT of a **complex** signal `(real, imag)` → `(real, imag)`.
///
/// Used by the separable 2D transform for the second (column) pass, where the
/// row-pass output is already complex. Unnormalized forward kernel `e^{-iθ}`.
pub fn fft_complex_1d(real: &[f32], imag: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut re: Vec<f64> = real.iter().map(|&v| v as f64).collect();
    let mut im: Vec<f64> = imag.iter().map(|&v| v as f64).collect();
    fft_dispatch(&mut re, &mut im, false);
    (
        re.iter().map(|&v| v as f32).collect(),
        im.iter().map(|&v| v as f32).collect(),
    )
}

/// Inverse FFT of a **complex** spectrum `(real, imag)` → `(real, imag)`, with
/// the `1/N` normalization applied.
///
/// Used by the separable 2D inverse transform for the column pass, which must
/// retain the imaginary part before the row pass collapses it to the real output.
pub fn ifft_complex_1d(real: &[f32], imag: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = real.len();
    let mut re: Vec<f64> = real.iter().map(|&v| v as f64).collect();
    let mut im: Vec<f64> = imag.iter().map(|&v| v as f64).collect();
    fft_dispatch(&mut re, &mut im, true);
    let inv_n = 1.0 / n.max(1) as f64;
    (
        re.iter().map(|&v| (v * inv_n) as f32).collect(),
        im.iter().map(|&v| (v * inv_n) as f32).collect(),
    )
}

/// Forward 2D FFT via separable 1D FFTs (row-wise then column-wise).
///
/// Mirrors `dft_2d` exactly: input is the real field `x` stored
/// row-major as `[nx × ny]`; the result `(real, imag)` is the 2D spectrum in the
/// same layout. Rows (length `ny`) are real→complex; columns (length `nx`) are
/// complex→complex.
pub fn fft_2d(x: &[f32], nx: usize, ny: usize) -> (Vec<f32>, Vec<f32>) {
    // Row-wise forward FFT (real input).
    let mut r1 = vec![0.0_f32; nx * ny];
    let mut i1 = vec![0.0_f32; nx * ny];
    for row in 0..nx {
        let row_data: Vec<f32> = (0..ny).map(|col| x[row * ny + col]).collect();
        let (rr, ri) = fft_1d(&row_data);
        for col in 0..ny {
            r1[row * ny + col] = rr[col];
            i1[row * ny + col] = ri[col];
        }
    }
    // Column-wise forward FFT on the (now complex) row-transform result.
    let mut r2 = vec![0.0_f32; nx * ny];
    let mut i2 = vec![0.0_f32; nx * ny];
    for col in 0..ny {
        let col_r: Vec<f32> = (0..nx).map(|row| r1[row * ny + col]).collect();
        let col_i: Vec<f32> = (0..nx).map(|row| i1[row * ny + col]).collect();
        let (cr, ci) = fft_complex_1d(&col_r, &col_i);
        for k in 0..nx {
            r2[k * ny + col] = cr[k];
            i2[k * ny + col] = ci[k];
        }
    }
    (r2, i2)
}

/// Inverse 2D FFT via separable column-wise then row-wise inverse FFTs.
///
/// Mirrors `idft_2d` exactly: the column pass inverts along `nx`
/// (keeping the complex result, normalized by `1/nx`); the row pass inverts along
/// `ny` (normalized by `1/ny`) and returns the real field.
pub fn ifft_2d(real: &[f32], imag: &[f32], nx: usize, ny: usize) -> Vec<f32> {
    // Column-wise inverse FFT first (complex → complex, /nx).
    let mut r1 = vec![0.0_f32; nx * ny];
    let mut i1 = vec![0.0_f32; nx * ny];
    for col in 0..ny {
        let col_r: Vec<f32> = (0..nx).map(|k| real[k * ny + col]).collect();
        let col_i: Vec<f32> = (0..nx).map(|k| imag[k * ny + col]).collect();
        let (cr, ci) = ifft_complex_1d(&col_r, &col_i);
        for n in 0..nx {
            r1[n * ny + col] = cr[n];
            i1[n * ny + col] = ci[n];
        }
    }
    // Row-wise inverse FFT (complex → real, /ny).
    let mut out = vec![0.0_f32; nx * ny];
    for row in 0..nx {
        let row_r: Vec<f32> = (0..ny).map(|col| r1[row * ny + col]).collect();
        let row_i: Vec<f32> = (0..ny).map(|col| i1[row * ny + col]).collect();
        let spatial = ifft_1d(&row_r, &row_i);
        for col in 0..ny {
            out[row * ny + col] = spatial[col];
        }
    }
    out
}
