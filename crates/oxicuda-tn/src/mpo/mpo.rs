//! Single-site MPO tensor and the [`Mpo`] container.

use crate::{TnError, TnResult};

/// One site of an MPO with shape `(W_l, d_out, d_in, W_r)` row-major.
///
/// The element `[w_l, p_out, p_in, w_r]` lives at index
/// `((w_l * d_out + p_out) * d_in + p_in) * W_r + w_r`.
#[derive(Debug, Clone)]
pub struct MpoTensor {
    pub w_l: usize,
    pub d_out: usize,
    pub d_in: usize,
    pub w_r: usize,
    pub data: Vec<f64>,
}

impl MpoTensor {
    /// Construct an MPO tensor with the given shape and row-major data.
    pub fn new(
        w_l: usize,
        d_out: usize,
        d_in: usize,
        w_r: usize,
        data: Vec<f64>,
    ) -> TnResult<Self> {
        if w_l == 0 || d_out == 0 || d_in == 0 || w_r == 0 {
            return Err(TnError::InvalidBondDimension(0));
        }
        if data.len() != w_l * d_out * d_in * w_r {
            return Err(TnError::ShapeMismatch {
                expected: vec![w_l, d_out, d_in, w_r],
                got: vec![data.len()],
            });
        }
        Ok(Self {
            w_l,
            d_out,
            d_in,
            w_r,
            data,
        })
    }

    /// Construct a zero tensor of the given shape.
    pub fn zeros(w_l: usize, d_out: usize, d_in: usize, w_r: usize) -> TnResult<Self> {
        Self::new(w_l, d_out, d_in, w_r, vec![0.0; w_l * d_out * d_in * w_r])
    }

    /// Row-major access.
    pub fn get(&self, w_l: usize, p_out: usize, p_in: usize, w_r: usize) -> TnResult<f64> {
        if w_l >= self.w_l || p_out >= self.d_out || p_in >= self.d_in || w_r >= self.w_r {
            return Err(TnError::IndexOutOfBounds {
                index: w_l,
                len: self.w_l,
            });
        }
        Ok(self.data[((w_l * self.d_out + p_out) * self.d_in + p_in) * self.w_r + w_r])
    }

    /// Row-major mutator.
    pub fn set(
        &mut self,
        w_l: usize,
        p_out: usize,
        p_in: usize,
        w_r: usize,
        v: f64,
    ) -> TnResult<()> {
        if w_l >= self.w_l || p_out >= self.d_out || p_in >= self.d_in || w_r >= self.w_r {
            return Err(TnError::IndexOutOfBounds {
                index: w_l,
                len: self.w_l,
            });
        }
        self.data[((w_l * self.d_out + p_out) * self.d_in + p_in) * self.w_r + w_r] = v;
        Ok(())
    }

    /// Shape tuple.
    pub fn shape(&self) -> (usize, usize, usize, usize) {
        (self.w_l, self.d_out, self.d_in, self.w_r)
    }
}

/// MPO container.
#[derive(Debug, Clone)]
pub struct Mpo {
    pub site_tensors: Vec<MpoTensor>,
}

impl Mpo {
    /// Construct from a vector of MPO tensors. Validates virtual bond compatibility and
    /// that the boundary virtual bonds equal 1.
    pub fn from_tensors(site_tensors: Vec<MpoTensor>) -> TnResult<Self> {
        if site_tensors.is_empty() {
            return Err(TnError::EmptyInput);
        }
        if site_tensors[0].w_l != 1 {
            return Err(TnError::InvalidBondDimension(site_tensors[0].w_l));
        }
        let last_wr = site_tensors.last().ok_or(TnError::EmptyInput)?.w_r;
        if last_wr != 1 {
            return Err(TnError::InvalidBondDimension(last_wr));
        }
        for i in 0..site_tensors.len() - 1 {
            if site_tensors[i].w_r != site_tensors[i + 1].w_l {
                return Err(TnError::DimensionMismatch {
                    a: site_tensors[i].w_r,
                    b: site_tensors[i + 1].w_l,
                });
            }
        }
        Ok(Self { site_tensors })
    }

    /// Number of sites.
    pub fn n_sites(&self) -> usize {
        self.site_tensors.len()
    }

    /// Build the identity MPO acting on `n` sites of physical dimension `d`.
    pub fn identity(n_sites: usize, d: usize) -> TnResult<Self> {
        if n_sites == 0 || d == 0 {
            return Err(TnError::EmptyInput);
        }
        let mut tensors = Vec::with_capacity(n_sites);
        for _ in 0..n_sites {
            // W_l = W_r = 1, just identity on physical legs
            let mut data = vec![0.0; d * d];
            for p in 0..d {
                data[p * d + p] = 1.0;
            }
            tensors.push(MpoTensor::new(1, d, d, 1, data)?);
        }
        Self::from_tensors(tensors)
    }

    /// Build the isotropic 1D Heisenberg XXX MPO
    /// `H = sum_i (S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + S^z_i S^z_{i+1})`
    /// on `n_sites` spin-1/2 qubits (open boundary).
    ///
    /// Equivalent to [`heisenberg_xxx_with_couplings`](Self::heisenberg_xxx_with_couplings)`(n_sites, 1.0, 1.0)`.
    pub fn heisenberg_xxx(n_sites: usize) -> TnResult<Self> {
        Self::heisenberg_xxx_with_couplings(n_sites, 1.0, 1.0)
    }

    /// Build the 1D Heisenberg XXZ MPO
    /// `H = sum_i [ jxy/2 (S^+_i S^-_{i+1} + S^-_i S^+_{i+1}) + jz S^z_i S^z_{i+1} ]`
    /// on `n_sites` spin-1/2 qubits (open boundary).
    ///
    /// The fully isotropic Heisenberg XXX point is `jxy == jz`.
    ///
    /// W matrix (canonical, bond dimension 5):
    /// ```text
    /// W = [ I            0           0           0           0 ;
    ///       S^+          0           0           0           0 ;
    ///       S^-          0           0           0           0 ;
    ///       S^z          0           0           0           0 ;
    ///       0      jxy/2 S^-   jxy/2 S^+    jz S^z        I ]
    /// ```
    ///
    /// The first-site MPO tensor (shape `(1, d, d, 5)`) holds the LAST ROW of `W`
    /// only; the last-site MPO tensor (shape `(5, d, d, 1)`) holds the FIRST COLUMN
    /// of `W` only; interior tensors (shape `(5, d, d, 5)`) carry all 8 non-zero
    /// `W` entries.
    pub fn heisenberg_xxx_with_couplings(n_sites: usize, jxy: f64, jz: f64) -> TnResult<Self> {
        if n_sites < 2 {
            return Err(TnError::InvalidConfiguration("n_sites < 2".into()));
        }
        if !jxy.is_finite() || !jz.is_finite() {
            return Err(TnError::InvalidConfiguration("non-finite coupling".into()));
        }
        let d = 2usize;
        // Spin-1/2 operators in the basis |↑⟩=0, |↓⟩=1 (row-major 2×2).
        let sz: [f64; 4] = [0.5, 0.0, 0.0, -0.5];
        let sp: [f64; 4] = [0.0, 1.0, 0.0, 0.0];
        let sm: [f64; 4] = [0.0, 0.0, 1.0, 0.0];
        let id: [f64; 4] = [1.0, 0.0, 0.0, 1.0];
        // Pre-scaled coupling forms.
        let half_jxy = 0.5 * jxy;
        let scaled_jxy_sm: [f64; 4] = [
            half_jxy * sm[0],
            half_jxy * sm[1],
            half_jxy * sm[2],
            half_jxy * sm[3],
        ];
        let scaled_jxy_sp: [f64; 4] = [
            half_jxy * sp[0],
            half_jxy * sp[1],
            half_jxy * sp[2],
            half_jxy * sp[3],
        ];
        let scaled_jz_sz: [f64; 4] = [jz * sz[0], jz * sz[1], jz * sz[2], jz * sz[3]];

        // Construct an MPO tensor of shape `(w_l, d, d, w_r)` from a sparse list
        // of `(row_in_W, col_in_W, &local_2x2_op)` entries. Indexing matches
        // `((w_l * d + p_out) * d + p_in) * w_r + w_r_idx` (row-major).
        let build_tensor =
            |w_l: usize, w_r: usize, entries: &[(usize, usize, &[f64])]| -> Vec<f64> {
                let mut data = vec![0.0; w_l * d * d * w_r];
                for (row, col, mat) in entries {
                    for p_out in 0..d {
                        for p_in in 0..d {
                            data[((row * d + p_out) * d + p_in) * w_r + col] +=
                                mat[p_out * d + p_in];
                        }
                    }
                }
                data
            };

        let mut tensors = Vec::with_capacity(n_sites);
        // First site: shape (1, d, d, 5) — last row of W.
        let first_data = build_tensor(
            1,
            5,
            &[
                // (0, 0) is 0 → omitted.
                (0, 1, &scaled_jxy_sm),
                (0, 2, &scaled_jxy_sp),
                (0, 3, &scaled_jz_sz),
                (0, 4, &id),
            ],
        );
        tensors.push(MpoTensor::new(1, d, d, 5, first_data)?);

        // Interior sites: shape (5, d, d, 5) — full W.
        for _ in 1..n_sites - 1 {
            let mid = build_tensor(
                5,
                5,
                &[
                    (0, 0, &id),
                    (1, 0, &sp),
                    (2, 0, &sm),
                    (3, 0, &sz),
                    (4, 1, &scaled_jxy_sm),
                    (4, 2, &scaled_jxy_sp),
                    (4, 3, &scaled_jz_sz),
                    (4, 4, &id),
                ],
            );
            tensors.push(MpoTensor::new(5, d, d, 5, mid)?);
        }

        // Last site: shape (5, d, d, 1) — first column of W.
        let last = build_tensor(
            5,
            1,
            &[
                (0, 0, &id),
                (1, 0, &sp),
                (2, 0, &sm),
                (3, 0, &sz),
                // (4, 0) is 0 → omitted.
            ],
        );
        tensors.push(MpoTensor::new(5, d, d, 1, last)?);
        Self::from_tensors(tensors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dmrg::two_site_excited::mps_inner_product;
    use crate::mpo::contraction::apply_mpo_to_mps;
    use crate::mps::mps::Mps;
    use crate::mps::tensor::MpsTensor;

    // ─────────────────────────────────────────────────────────────────────────
    // Test helpers
    // ─────────────────────────────────────────────────────────────────────────
    //
    // Approach taken for ⟨ψ|H|ψ⟩ in the value-checked tests below:
    //
    //   1. Build the test state as an MPS (product state via
    //      `Mps::from_product_state`, or rank-2 MPS for singlet/triplet-zero
    //      using `MpsTensor` directly).
    //   2. Apply the Heisenberg MPO with `apply_mpo_to_mps(.., chi_max=64,
    //      tol=1e-14)` to obtain `H|ψ⟩` as an MPS.
    //   3. Contract ⟨ψ|H|ψ⟩ via `mps_inner_product`.
    //
    // No new production APIs are introduced for testing; only existing public
    // contraction primitives are reused.

    const SQRT_HALF: f64 = std::f64::consts::FRAC_1_SQRT_2; // = 1/√2.

    /// Build a product state MPS with each site amplitude vector of length 2.
    fn product_mps(states: &[[f64; 2]]) -> Mps {
        let local: Vec<Vec<f64>> = states.iter().map(|s| vec![s[0], s[1]]).collect();
        Mps::from_product_state(&local).expect("product MPS")
    }

    /// Build the singlet `|s⟩ = (|↑↓⟩ − |↓↑⟩)/√2` as a rank-2 MPS.
    ///
    /// Site 0 (shape (1, 2, 2)): A[0,p,α], A[0,↑,0]=1, A[0,↓,1]=1.
    /// Site 1 (shape (2, 2, 1)): B[α,p,0], B[0,↓,0]=SQRT_HALF, B[1,↑,0]=-SQRT_HALF.
    fn singlet_mps() -> Mps {
        // A row-major layout: ((d_l * d_p + p) * d_r + α) actually for shape
        // (d_l, d_p, d_r) the indexing is `(a * d_p + p) * d_r + b`.
        let a_data = vec![
            // a=0
            1.0, 0.0, // p=↑ → (α=0, α=1)
            0.0, 1.0, // p=↓ → (α=0, α=1)
        ];
        let a = MpsTensor::new(1, 2, 2, a_data).expect("site 0");
        let b_data = vec![
            // α=0
            0.0,       // p=↑
            SQRT_HALF, // p=↓
            // α=1
            -SQRT_HALF, // p=↑
            0.0,        // p=↓
        ];
        let b = MpsTensor::new(2, 2, 1, b_data).expect("site 1");
        Mps::from_tensors(vec![a, b]).expect("singlet mps")
    }

    /// Build the triplet-zero `|t_0⟩ = (|↑↓⟩ + |↓↑⟩)/√2` as a rank-2 MPS.
    fn triplet_zero_mps() -> Mps {
        let a_data = vec![1.0, 0.0, 0.0, 1.0];
        let a = MpsTensor::new(1, 2, 2, a_data).expect("site 0");
        let b_data = vec![
            // α=0
            0.0, SQRT_HALF, // α=1
            SQRT_HALF, 0.0,
        ];
        let b = MpsTensor::new(2, 2, 1, b_data).expect("site 1");
        Mps::from_tensors(vec![a, b]).expect("triplet zero mps")
    }

    /// Compute ⟨ψ|H|ψ⟩ for the given MPS and MPO via apply-then-inner-product.
    fn energy(mpo: &Mpo, psi: &Mps) -> f64 {
        let h_psi = apply_mpo_to_mps(mpo, psi, 64, 1e-14).expect("apply MPO");
        mps_inner_product(psi, &h_psi).expect("inner product")
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Existing structural tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn identity_mpo_shape() {
        let mpo = Mpo::identity(3, 2).expect("ok");
        assert_eq!(mpo.n_sites(), 3);
        for t in &mpo.site_tensors {
            assert_eq!(t.shape(), (1, 2, 2, 1));
        }
    }

    #[test]
    fn heisenberg_constructs() {
        let mpo = Mpo::heisenberg_xxx(4).expect("ok");
        assert_eq!(mpo.n_sites(), 4);
        assert_eq!(mpo.site_tensors[0].shape(), (1, 2, 2, 5));
        assert_eq!(mpo.site_tensors[1].shape(), (5, 2, 2, 5));
        assert_eq!(mpo.site_tensors[3].shape(), (5, 2, 2, 1));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Heisenberg correctness tests
    // ─────────────────────────────────────────────────────────────────────────

    /// For the all-up state `|↑↑…↑⟩`, only `S^z·S^z` survives (`S^+|↑⟩ = 0`),
    /// so `⟨H⟩ = (n - 1) · jz · (1/2) · (1/2) = (n - 1) · jz / 4`.
    #[test]
    fn heisenberg_polarized_diagonal() {
        for &n in &[2usize, 3, 4, 5] {
            let up: Vec<[f64; 2]> = (0..n).map(|_| [1.0, 0.0]).collect();
            let psi = product_mps(&up);
            let mpo = Mpo::heisenberg_xxx(n).expect("mpo");
            let e = energy(&mpo, &psi);
            let expected = (n as f64 - 1.0) * 1.0 / 4.0;
            assert!(
                (e - expected).abs() < 1e-9,
                "polarized n={n}: ⟨H⟩ = {e}, expected {expected}",
            );
        }
    }

    /// For `n = 2`, the singlet `|s⟩ = (|↑↓⟩ − |↓↑⟩)/√2` is the unique
    /// eigenstate of `S²` with eigenvalue 0; for isotropic Heisenberg
    /// `H = S_1·S_2 = (S²_tot − S²_1 − S²_2)/2 = S²_tot/2 − 3/4`, hence
    /// `⟨s|H|s⟩ = 0 − 3/4 = -3/4`.
    #[test]
    fn heisenberg_two_site_singlet() {
        let mpo = Mpo::heisenberg_xxx(2).expect("mpo");
        let psi = singlet_mps();
        // Sanity-check normalisation first.
        let n2 = psi.norm_squared().expect("norm");
        assert!((n2 - 1.0).abs() < 1e-12, "singlet norm² = {n2}");
        let e = energy(&mpo, &psi);
        assert!(
            (e - (-0.75)).abs() < 1e-9,
            "singlet ⟨H⟩ = {e}, expected -0.75",
        );
    }

    /// The three triplet states `|t_+⟩=|↑↑⟩`, `|t_0⟩=(|↑↓⟩+|↓↑⟩)/√2`,
    /// `|t_-⟩=|↓↓⟩` are eigenstates of isotropic Heisenberg with eigenvalue
    /// `+1/4` for `n = 2`.
    #[test]
    fn heisenberg_two_site_triplet() {
        let mpo = Mpo::heisenberg_xxx(2).expect("mpo");

        let psi_plus = product_mps(&[[1.0, 0.0], [1.0, 0.0]]);
        let e_plus = energy(&mpo, &psi_plus);
        assert!(
            (e_plus - 0.25).abs() < 1e-9,
            "|t_+⟩ ⟨H⟩ = {e_plus}, expected 0.25",
        );

        let psi_zero = triplet_zero_mps();
        let n2 = psi_zero.norm_squared().expect("norm");
        assert!((n2 - 1.0).abs() < 1e-12, "triplet0 norm² = {n2}");
        let e_zero = energy(&mpo, &psi_zero);
        assert!(
            (e_zero - 0.25).abs() < 1e-9,
            "|t_0⟩ ⟨H⟩ = {e_zero}, expected 0.25",
        );

        let psi_minus = product_mps(&[[0.0, 1.0], [0.0, 1.0]]);
        let e_minus = energy(&mpo, &psi_minus);
        assert!(
            (e_minus - 0.25).abs() < 1e-9,
            "|t_-⟩ ⟨H⟩ = {e_minus}, expected 0.25",
        );
    }

    /// Verify that `jxy` and `jz` scale the energy correctly.
    ///
    /// `n = 2` isotropic with `J = 2`: triplet eigenvalue 2 · 1/4 = 0.5,
    /// singlet eigenvalue 2 · -3/4 = -1.5.
    ///
    /// `n = 2` XXZ with `jxy = 2, jz = 0`: H acts only as the XY exchange
    /// `(jxy/2)(S^+ S^- + S^- S^+)`. On `|↑↑⟩` this is 0; on the singlet,
    /// `(jxy/2)(S^+ S^- + S^- S^+)|s⟩ = -(jxy/2) |s⟩` so `⟨s|H|s⟩ = -1`.
    ///
    /// `n = 2` XXZ with `jxy = 0, jz = 2`: only `S^z S^z` survives. The
    /// singlet is a superposition of `|↑↓⟩` and `|↓↑⟩` so `S^z S^z` gives
    /// (1/2)(-1/2) = -1/4 on each component → `⟨s|H|s⟩ = -1/2`. For `|t_+⟩`
    /// we get `2 · (1/2)(1/2) = 1/2`.
    #[test]
    fn heisenberg_couplings_propagate() {
        let psi_plus = product_mps(&[[1.0, 0.0], [1.0, 0.0]]);
        let psi_singlet = singlet_mps();

        // Sub-case 1: isotropic J = 2.
        let mpo = Mpo::heisenberg_xxx_with_couplings(2, 2.0, 2.0).expect("mpo");
        let e_plus = energy(&mpo, &psi_plus);
        let e_singlet = energy(&mpo, &psi_singlet);
        assert!(
            (e_plus - 0.5).abs() < 1e-9,
            "J=2 triplet ⟨H⟩ = {e_plus}, expected 0.5",
        );
        assert!(
            (e_singlet - (-1.5)).abs() < 1e-9,
            "J=2 singlet ⟨H⟩ = {e_singlet}, expected -1.5",
        );

        // Sub-case 2: XY only (jxy=2, jz=0).
        let mpo = Mpo::heisenberg_xxx_with_couplings(2, 2.0, 0.0).expect("mpo");
        let e_plus = energy(&mpo, &psi_plus);
        let e_singlet = energy(&mpo, &psi_singlet);
        assert!(
            e_plus.abs() < 1e-9,
            "XY-only triplet ⟨H⟩ = {e_plus}, expected 0",
        );
        assert!(
            (e_singlet - (-1.0)).abs() < 1e-9,
            "XY-only singlet ⟨H⟩ = {e_singlet}, expected -1.0",
        );

        // Sub-case 3: Ising-Z only (jxy=0, jz=2).
        let mpo = Mpo::heisenberg_xxx_with_couplings(2, 0.0, 2.0).expect("mpo");
        let e_plus = energy(&mpo, &psi_plus);
        let e_singlet = energy(&mpo, &psi_singlet);
        assert!(
            (e_plus - 0.5).abs() < 1e-9,
            "Z-only triplet ⟨H⟩ = {e_plus}, expected 0.5",
        );
        assert!(
            (e_singlet - (-0.5)).abs() < 1e-9,
            "Z-only singlet ⟨H⟩ = {e_singlet}, expected -0.5",
        );
    }

    /// Error paths for `heisenberg_xxx_with_couplings`.
    #[test]
    fn heisenberg_xxx_finite_couplings() {
        assert!(matches!(
            Mpo::heisenberg_xxx_with_couplings(1, 1.0, 1.0),
            Err(TnError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            Mpo::heisenberg_xxx_with_couplings(4, f64::NAN, 1.0),
            Err(TnError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            Mpo::heisenberg_xxx_with_couplings(4, 1.0, f64::INFINITY),
            Err(TnError::InvalidConfiguration(_))
        ));
    }

    /// `heisenberg_xxx(n)` must produce byte-identical site tensors to
    /// `heisenberg_xxx_with_couplings(n, 1.0, 1.0)`.
    #[test]
    fn heisenberg_xxx_matches_default() {
        let n = 5;
        let a = Mpo::heisenberg_xxx(n).expect("default");
        let b = Mpo::heisenberg_xxx_with_couplings(n, 1.0, 1.0).expect("explicit");
        assert_eq!(a.n_sites(), b.n_sites());
        for s in 0..a.n_sites() {
            let ta = &a.site_tensors[s];
            let tb = &b.site_tensors[s];
            assert_eq!(ta.shape(), tb.shape(), "site {s} shape mismatch");
            assert_eq!(ta.data, tb.data, "site {s} data mismatch");
        }
    }
}
