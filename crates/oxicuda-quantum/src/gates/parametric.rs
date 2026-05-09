use num_complex::Complex;

type Complex32 = Complex<f32>;

#[inline]
fn c(re: f32, im: f32) -> Complex32 {
    Complex32::new(re, im)
}

/// Rotation around X axis: Rx(θ) = cos(θ/2)I - i·sin(θ/2)X.
#[must_use]
pub fn gate_rx(theta: f32) -> [[Complex32; 2]; 2] {
    let half = theta * 0.5;
    let cos = half.cos();
    let sin = half.sin();
    [[c(cos, 0.0), c(0.0, -sin)], [c(0.0, -sin), c(cos, 0.0)]]
}

/// Rotation around Y axis: Ry(θ) = cos(θ/2)I - i·sin(θ/2)Y.
#[must_use]
pub fn gate_ry(theta: f32) -> [[Complex32; 2]; 2] {
    let half = theta * 0.5;
    let cos = half.cos();
    let sin = half.sin();
    [[c(cos, 0.0), c(-sin, 0.0)], [c(sin, 0.0), c(cos, 0.0)]]
}

/// Rotation around Z axis: Rz(θ) = diag(e^{-iθ/2}, e^{+iθ/2}).
#[must_use]
pub fn gate_rz(theta: f32) -> [[Complex32; 2]; 2] {
    let half = theta * 0.5;
    [
        [Complex32::new(half.cos(), -half.sin()), c(0.0, 0.0)],
        [c(0.0, 0.0), Complex32::new(half.cos(), half.sin())],
    ]
}

/// General U3 gate: U3(θ,φ,λ).
///
/// U3 = [[cos(θ/2), -e^{iλ}·sin(θ/2)],
///        [e^{iφ}·sin(θ/2), e^{i(φ+λ)}·cos(θ/2)]]
#[must_use]
pub fn gate_u3(theta: f32, phi: f32, lam: f32) -> [[Complex32; 2]; 2] {
    let half = theta * 0.5;
    let cos = half.cos();
    let sin = half.sin();
    let e_phi = Complex32::new(phi.cos(), phi.sin());
    let e_lam = Complex32::new(lam.cos(), lam.sin());
    let e_sum = Complex32::new((phi + lam).cos(), (phi + lam).sin());
    [[c(cos, 0.0), -e_lam * sin], [e_phi * sin, e_sum * cos]]
}

/// Phase gate: P(θ) = diag(1, e^{iθ}).
#[must_use]
pub fn gate_phase(theta: f32) -> [[Complex32; 2]; 2] {
    let phase = Complex32::new(theta.cos(), theta.sin());
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), phase]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rx_pi_is_x_up_to_phase() {
        use std::f32::consts::PI;
        let rx = gate_rx(PI);
        // Rx(π) = -i*X, so |rx[0][1]| = 1
        assert!((rx[0][1].norm() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rz_preserves_norm() {
        use std::f32::consts::PI;
        let rz = gate_rz(PI / 3.0);
        let col0_norm = rz[0][0].norm_sqr() + rz[1][0].norm_sqr();
        assert!((col0_norm - 1.0).abs() < 1e-5);
    }
}
