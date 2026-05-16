//! Dirichlet (essential) boundary condition: `u = g(x)` on the boundary.

/// Dirichlet BC value.
#[derive(Debug, Clone, Copy)]
pub struct DirichletBc {
    pub value: f64,
}

impl DirichletBc {
    /// New constant Dirichlet condition `u = v`.
    pub const fn new(v: f64) -> Self {
        Self { value: v }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_const() {
        let bc = DirichletBc::new(2.5);
        assert!((bc.value - 2.5).abs() < 1.0e-12);
    }
}
