use num_complex::Complex;

type Complex32 = Complex<f32>;

#[inline]
fn c(re: f32, im: f32) -> Complex32 {
    Complex32::new(re, im)
}

/// Identity gate I.
#[must_use]
pub fn gate_i() -> [[Complex32; 2]; 2] {
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]]
}

/// Pauli-X (NOT) gate.
#[must_use]
pub fn gate_x() -> [[Complex32; 2]; 2] {
    [[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)]]
}

/// Pauli-Y gate.
#[must_use]
pub fn gate_y() -> [[Complex32; 2]; 2] {
    [[c(0.0, 0.0), c(0.0, -1.0)], [c(0.0, 1.0), c(0.0, 0.0)]]
}

/// Pauli-Z gate.
#[must_use]
pub fn gate_z() -> [[Complex32; 2]; 2] {
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(-1.0, 0.0)]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pauli_gates_unitary() {
        // X^2 = I
        let x = gate_x();
        let xx00 = x[0][0] * x[0][0] + x[0][1] * x[1][0];
        assert!((xx00 - c(1.0, 0.0)).norm() < 1e-6);
    }
}
