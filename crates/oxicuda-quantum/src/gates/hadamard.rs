use num_complex::Complex;

type Complex32 = Complex<f32>;

#[inline]
fn c(re: f32, im: f32) -> Complex32 {
    Complex32::new(re, im)
}

const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Hadamard gate H = (X+Z)/√2.
#[must_use]
pub fn gate_h() -> [[Complex32; 2]; 2] {
    [
        [c(INV_SQRT2, 0.0), c(INV_SQRT2, 0.0)],
        [c(INV_SQRT2, 0.0), c(-INV_SQRT2, 0.0)],
    ]
}

/// Phase gate S = diag(1, i).
#[must_use]
pub fn gate_s() -> [[Complex32; 2]; 2] {
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(0.0, 1.0)]]
}

/// T gate = diag(1, e^{iπ/4}).
#[must_use]
pub fn gate_t() -> [[Complex32; 2]; 2] {
    let phase = Complex32::new(INV_SQRT2, INV_SQRT2);
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), phase]]
}

/// S-dagger gate = diag(1, -i).
#[must_use]
pub fn gate_sdg() -> [[Complex32; 2]; 2] {
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(0.0, -1.0)]]
}

/// T-dagger gate = diag(1, e^{-iπ/4}).
#[must_use]
pub fn gate_tdg() -> [[Complex32; 2]; 2] {
    let phase = Complex32::new(INV_SQRT2, -INV_SQRT2);
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), phase]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hadamard_squared_is_identity() {
        let h = gate_h();
        // H*H should be I (Hermitian unitary)
        let hh00 = h[0][0] * h[0][0] + h[0][1] * h[1][0];
        let hh01 = h[0][0] * h[0][1] + h[0][1] * h[1][1];
        assert!((hh00 - c(1.0, 0.0)).norm() < 1e-5, "hh00={hh00:?}");
        assert!(hh01.norm() < 1e-5, "hh01={hh01:?}");
    }
}
