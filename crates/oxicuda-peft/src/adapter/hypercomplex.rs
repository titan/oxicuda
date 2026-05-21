//! Quaternionic (hypercomplex) bottleneck adapter using Hamilton product weight sharing.
//!
//! References:
//! - Hamilton W R (1843) "On a new Species of Imaginary Quantities Connected with a Theory of
//!   Quaternions", *Proc. R. Irish Acad.* 2: 424–434.
//! - Parcollet T, Morchid M, Linares G (2019) "Quaternion Recurrent Neural Networks",
//!   *ICLR 2019*. <https://arxiv.org/abs/1806.04418>
//! - Zhang Y, Wang Y (2022) "Beyond Real: 2.5D Visual Sound Source Localization",
//!   *arXiv 2022*. (Hypercomplex adapter design.)
//!
//! ## Design
//!
//! A quaternion number `q = r + i·î + j·ĵ + k·k̂` lives in **H** with basis
//! `{1, i, j, k}` satisfying `i²=j²=k²=ijk=−1`. The **Hamilton product** captures
//! all cross-component interactions in a non-commutative algebra (`i×j=k` but `j×i=−k`).
//!
//! The adapter stores its weights as four real matrices `(W_r, W_i, W_j, W_k)` (one per
//! quaternion component) giving 4× parameter reduction vs. a real adapter of the same
//! dimension: each "quaternion weight" at position `(a,b)` covers 4 real connections with
//! shared multiplicative structure.
//!
//! ### Forward pass
//! ```text
//! q_in  = pack(x)            : ℝ^{in_dim}    → Q^{in_dim/4}
//! h     = down.matvec(q_in)  : Q^{in_dim/4}  → Q^{bottleneck/4}
//! h     = split_gelu(h)      : GELU on real part only
//! o     = up.matvec(h)       : Q^{bottleneck/4} → Q^{in_dim/4}
//! out   = unpack(o) + x      : residual connection
//! ```

use crate::error::{PeftError, PeftResult};
use crate::handle::{LcgRng, PeftHandle};

// ---------------------------------------------------------------------------
// GELU activation (matches houlsby.rs definition)
// ---------------------------------------------------------------------------

/// GELU activation: `0.5 · x · (1 + tanh(sqrt(2/π) · (x + 0.044715 · x³)))`.
#[inline]
fn gelu(x: f32) -> f32 {
    const C0: f32 = 0.797_884_56; // sqrt(2/π)
    const C1: f32 = 0.044_715;
    let inner = C0 * (x + C1 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

// ---------------------------------------------------------------------------
// Quaternion scalar type
// ---------------------------------------------------------------------------

/// One quaternion `q = r + i·î + j·ĵ + k·k̂`.
#[derive(Clone, Debug, PartialEq)]
pub struct Quat {
    /// Real (scalar) component.
    pub r: f32,
    /// `î` component.
    pub i: f32,
    /// `ĵ` component.
    pub j: f32,
    /// `k̂` component.
    pub k: f32,
}

impl Quat {
    /// Additive identity `0 + 0î + 0ĵ + 0k̂`.
    #[inline]
    #[must_use]
    pub fn zero() -> Self {
        Self {
            r: 0.0,
            i: 0.0,
            j: 0.0,
            k: 0.0,
        }
    }

    /// Multiplicative identity `1 + 0î + 0ĵ + 0k̂`.
    #[inline]
    #[must_use]
    pub fn one() -> Self {
        Self {
            r: 1.0,
            i: 0.0,
            j: 0.0,
            k: 0.0,
        }
    }

    /// Squared norm `|q|² = r² + i² + j² + k²`.
    #[inline]
    #[must_use]
    pub fn norm_sq(&self) -> f32 {
        self.r * self.r + self.i * self.i + self.j * self.j + self.k * self.k
    }

    /// Quaternion conjugate `q* = r − î·i − ĵ·j − k̂·k`.
    #[inline]
    #[must_use]
    pub fn conjugate(&self) -> Quat {
        Quat {
            r: self.r,
            i: -self.i,
            j: -self.j,
            k: -self.k,
        }
    }

    /// **Hamilton product** `p ⊗ q` (non-commutative: `î×ĵ = k̂` but `ĵ×î = −k̂`).
    ///
    /// Multiplication table:
    /// ```text
    ///   i·j = k,  j·k = i,  k·i = j
    ///   j·i = -k, k·j = -i, i·k = -j
    /// ```
    #[inline]
    #[must_use]
    pub fn hamilton(p: &Quat, q: &Quat) -> Quat {
        Quat {
            r: p.r * q.r - p.i * q.i - p.j * q.j - p.k * q.k,
            i: p.r * q.i + p.i * q.r + p.j * q.k - p.k * q.j,
            j: p.r * q.j - p.i * q.k + p.j * q.r + p.k * q.i,
            k: p.r * q.k + p.i * q.j - p.j * q.i + p.k * q.r,
        }
    }

    /// Add two quaternions component-wise.
    #[inline]
    #[must_use]
    fn add(a: &Quat, b: &Quat) -> Quat {
        Quat {
            r: a.r + b.r,
            i: a.i + b.i,
            j: a.j + b.j,
            k: a.k + b.k,
        }
    }
}

// ---------------------------------------------------------------------------
// Quaternion matrix
// ---------------------------------------------------------------------------

/// A matrix of quaternions `W ∈ Q^{rows × cols}`.
///
/// Stored as four real matrices `(wr, wi, wj, wk)` each of size `rows × cols`
/// in row-major order. The quaternion at `(a, b)` is:
/// `W[a,b] = Quat { r: wr[a*cols+b], i: wi[a*cols+b], j: wj[a*cols+b], k: wk[a*cols+b] }`.
#[derive(Debug)]
pub struct QuatMatrix {
    /// Real component of each quaternion weight.
    pub wr: Vec<f32>,
    /// `î` component of each quaternion weight.
    pub wi: Vec<f32>,
    /// `ĵ` component of each quaternion weight.
    pub wj: Vec<f32>,
    /// `k̂` component of each quaternion weight.
    pub wk: Vec<f32>,
    /// Number of quaternion rows.
    pub rows: usize,
    /// Number of quaternion columns.
    pub cols: usize,
}

impl QuatMatrix {
    /// Allocate a zero-valued quaternion matrix of shape `rows × cols`.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let sz = rows * cols;
        Self {
            wr: vec![0.0; sz],
            wi: vec![0.0; sz],
            wj: vec![0.0; sz],
            wk: vec![0.0; sz],
            rows,
            cols,
        }
    }

    /// He (Kaiming) uniform initialisation of the **real component only**;
    /// imaginary parts `wi`, `wj`, `wk` remain zero.
    ///
    /// This starts the adapter in a near-real regime so the Hamilton product
    /// approximates a real matrix–vector product during early training.
    ///
    /// `limit = sqrt(6 / cols)` (fan-in uniform bound).
    #[must_use]
    pub fn kaiming_real(rows: usize, cols: usize, rng: &mut LcgRng) -> Self {
        let limit = (6.0_f32 / cols as f32).sqrt();
        let sz = rows * cols;
        let wr = (0..sz)
            .map(|_| rng.next_f32() * 2.0 * limit - limit)
            .collect();
        Self {
            wr,
            wi: vec![0.0; sz],
            wj: vec![0.0; sz],
            wk: vec![0.0; sz],
            rows,
            cols,
        }
    }

    /// Hamilton-product matrix–vector product:
    /// `y[a] = Σ_{b=0}^{cols-1} hamilton(W[a,b], x[b])`.
    ///
    /// `x` must have length `cols`. Returns a `Vec<Quat>` of length `rows`.
    #[must_use]
    pub fn matvec(&self, x: &[Quat]) -> Vec<Quat> {
        (0..self.rows)
            .map(|a| {
                let base = a * self.cols;
                x.iter().enumerate().fold(Quat::zero(), |acc, (b, xb)| {
                    let w_ab = Quat {
                        r: self.wr[base + b],
                        i: self.wi[base + b],
                        j: self.wj[base + b],
                        k: self.wk[base + b],
                    };
                    let prod = Quat::hamilton(&w_ab, xb);
                    Quat::add(&acc, &prod)
                })
            })
            .collect()
    }

    /// Total number of real scalar parameters: `4 * rows * cols`.
    #[inline]
    #[must_use]
    pub fn total_params(&self) -> usize {
        4 * self.rows * self.cols
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for a [`QuaternionAdapter`].
#[derive(Clone, Debug, PartialEq)]
pub struct QuaternionAdapterConfig {
    /// Input (and output) dimension; **must be divisible by 4**.
    pub in_dim: usize,
    /// Bottleneck dimension; **must be divisible by 4**.
    pub bottleneck: usize,
}

// ---------------------------------------------------------------------------
// Adapter
// ---------------------------------------------------------------------------

/// Quaternionic bottleneck adapter.
///
/// Architecture:
/// ```text
/// pack(x) → down (Q^{bt/4 × in/4}, Kaiming-real) → split-GELU → up (Q^{in/4 × bt/4}, zeros) → unpack + x
/// ```
///
/// **Parameter count** (vs. real adapter `in_dim × bottleneck`):
/// ```text
/// total = 4·(bt/4)·(in/4) + 4·(in/4)·(bt/4) = in·bt/4 + in·bt/4 = in·bt/2
/// ```
/// This gives a 2× reduction compared to a standard adapter (which would have
/// `2 × in_dim × bottleneck` parameters) — or equivalently a 4× reduction in
/// quaternion parameters vs. scalar parameters of the same connection structure.
#[derive(Debug)]
pub struct QuaternionAdapter {
    /// Down-projection: `Q^{bottleneck/4 × in_dim/4}`, Kaiming-real init.
    pub down: QuatMatrix,
    /// Up-projection: `Q^{in_dim/4 × bottleneck/4}`, zero-init (residual identity).
    pub up: QuatMatrix,
    /// Adapter configuration.
    pub cfg: QuaternionAdapterConfig,
}

impl QuaternionAdapter {
    /// Construct a new quaternion adapter.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::UnalignedDimension`] if `in_dim` or `bottleneck`
    /// is not divisible by 4.
    pub fn new(cfg: QuaternionAdapterConfig, handle: &mut PeftHandle) -> PeftResult<Self> {
        if !cfg.in_dim.is_multiple_of(4) {
            return Err(PeftError::UnalignedDimension {
                bot: cfg.bottleneck,
                in_dim: cfg.in_dim,
            });
        }
        if !cfg.bottleneck.is_multiple_of(4) {
            return Err(PeftError::UnalignedDimension {
                bot: cfg.bottleneck,
                in_dim: cfg.in_dim,
            });
        }
        let q_rows_down = cfg.bottleneck / 4;
        let q_cols_down = cfg.in_dim / 4;
        let q_rows_up = cfg.in_dim / 4;
        let q_cols_up = cfg.bottleneck / 4;
        let down = QuatMatrix::kaiming_real(q_rows_down, q_cols_down, &mut handle.rng);
        let up = QuatMatrix::zeros(q_rows_up, q_cols_up);
        Ok(Self { down, up, cfg })
    }

    /// Apply the quaternion adapter to an input of shape `[seq_len × in_dim]`.
    ///
    /// The forward pass packs real-valued tokens into quaternions, applies the
    /// Hamilton-product down-projection, a split-GELU (GELU on the real part only),
    /// the zero-initialized up-projection, unpacks back to real, and adds the
    /// residual `x`.
    ///
    /// Returns a `Vec<f32>` of length `seq_len × in_dim`.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::DimensionMismatch`] if `x.len() ≠ seq_len × in_dim`.
    pub fn forward(&self, x: &[f32], seq_len: usize) -> PeftResult<Vec<f32>> {
        let in_dim = self.cfg.in_dim;
        let expected = seq_len * in_dim;
        if x.len() != expected {
            return Err(PeftError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }
        let q_in_len = in_dim / 4;
        let mut out = Vec::with_capacity(expected);

        for t in 0..seq_len {
            let x_t = &x[t * in_dim..(t + 1) * in_dim];

            // Pack: x_t[4j..4j+4] → Quat
            let q_in: Vec<Quat> = (0..q_in_len)
                .map(|j| Quat {
                    r: x_t[4 * j],
                    i: x_t[4 * j + 1],
                    j: x_t[4 * j + 2],
                    k: x_t[4 * j + 3],
                })
                .collect();

            // Down projection
            let mut h = self.down.matvec(&q_in);

            // Split-GELU: apply GELU to real part only
            for q in h.iter_mut() {
                q.r = gelu(q.r);
            }

            // Up projection
            let o = self.up.matvec(&h);

            // Unpack + residual
            for (j, oj) in o.iter().enumerate() {
                out.push(x_t[4 * j] + oj.r);
                out.push(x_t[4 * j + 1] + oj.i);
                out.push(x_t[4 * j + 2] + oj.j);
                out.push(x_t[4 * j + 3] + oj.k);
            }
        }

        Ok(out)
    }

    /// Total number of real scalar parameters:
    /// `4·(bottleneck/4)·(in_dim/4) × 2 = in_dim·bottleneck/2`.
    #[inline]
    #[must_use]
    pub fn total_params(&self) -> usize {
        self.down.total_params() + self.up.total_params()
    }
}
