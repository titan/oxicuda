//! Tape-based reverse-mode automatic differentiation (Wengert list).

use crate::error::{PinnError, PinnResult};

/// A single node in the Wengert tape.
struct TapeNode {
    value: f32,
    /// Up to 2 parent (dependency) indices and their local partial derivatives.
    deps: [(usize, f32); 2],
    n_deps: u8,
}

/// Tape / Wengert list for reverse-mode AD.
///
/// Build a computation graph by calling methods on `Tape`, then call
/// `gradient(output)` for a reverse pass that returns all partial derivatives.
pub struct Tape {
    nodes: Vec<TapeNode>,
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

impl Tape {
    /// Create an empty tape.
    #[must_use]
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Push a new node and return its index.
    fn push(&mut self, value: f32, deps: [(usize, f32); 2], n_deps: u8) -> Var {
        let idx = self.nodes.len();
        self.nodes.push(TapeNode {
            value,
            deps,
            n_deps,
        });
        Var {
            idx,
            tape_len: idx + 1,
        }
    }

    /// Register a leaf input variable.
    pub fn variable(&mut self, value: f32) -> Var {
        self.push(value, [(0, 0.0); 2], 0)
    }

    /// Register a constant (leaf with no gradient flow).
    pub fn constant(&mut self, value: f32) -> Var {
        self.push(value, [(0, 0.0); 2], 0)
    }

    /// Get the primal value of a `Var`.
    #[must_use]
    pub fn value(&self, v: Var) -> f32 {
        self.nodes[v.idx].value
    }

    /// `z = a + b`; `∂z/∂a = 1`, `∂z/∂b = 1`.
    pub fn add(&mut self, a: Var, b: Var) -> Var {
        let val = self.nodes[a.idx].value + self.nodes[b.idx].value;
        self.push(val, [(a.idx, 1.0), (b.idx, 1.0)], 2)
    }

    /// `z = a - b`; `∂z/∂a = 1`, `∂z/∂b = -1`.
    pub fn sub(&mut self, a: Var, b: Var) -> Var {
        let val = self.nodes[a.idx].value - self.nodes[b.idx].value;
        self.push(val, [(a.idx, 1.0), (b.idx, -1.0)], 2)
    }

    /// `z = a * b`; product rule.
    pub fn mul(&mut self, a: Var, b: Var) -> Var {
        let av = self.nodes[a.idx].value;
        let bv = self.nodes[b.idx].value;
        self.push(av * bv, [(a.idx, bv), (b.idx, av)], 2)
    }

    /// `z = a / b`; quotient rule.
    pub fn div(&mut self, a: Var, b: Var) -> PinnResult<Var> {
        let av = self.nodes[a.idx].value;
        let bv = self.nodes[b.idx].value;
        if bv == 0.0 {
            return Err(PinnError::NanEncountered {
                location: "tape::div by zero",
            });
        }
        let val = av / bv;
        let da = 1.0 / bv;
        let db = -av / (bv * bv);
        Ok(self.push(val, [(a.idx, da), (b.idx, db)], 2))
    }

    /// `z = sin(a)`; `∂z/∂a = cos(a)`.
    pub fn sin(&mut self, a: Var) -> Var {
        let av = self.nodes[a.idx].value;
        self.push(av.sin(), [(a.idx, av.cos()), (0, 0.0)], 1)
    }

    /// `z = cos(a)`; `∂z/∂a = -sin(a)`.
    pub fn cos(&mut self, a: Var) -> Var {
        let av = self.nodes[a.idx].value;
        self.push(av.cos(), [(a.idx, -av.sin()), (0, 0.0)], 1)
    }

    /// `z = exp(a)`; `∂z/∂a = exp(a)`.
    pub fn exp(&mut self, a: Var) -> Var {
        let ev = self.nodes[a.idx].value.exp();
        self.push(ev, [(a.idx, ev), (0, 0.0)], 1)
    }

    /// `z = ln(a)`; `∂z/∂a = 1/a`.
    pub fn ln(&mut self, a: Var) -> PinnResult<Var> {
        let av = self.nodes[a.idx].value;
        if av <= 0.0 {
            return Err(PinnError::NanEncountered {
                location: "tape::ln of non-positive",
            });
        }
        Ok(self.push(av.ln(), [(a.idx, 1.0 / av), (0, 0.0)], 1))
    }

    /// `z = tanh(a)`; `∂z/∂a = 1 - tanh²(a)`.
    pub fn tanh(&mut self, a: Var) -> Var {
        let tv = self.nodes[a.idx].value.tanh();
        self.push(tv, [(a.idx, 1.0 - tv * tv), (0, 0.0)], 1)
    }

    /// `z = a²`; `∂z/∂a = 2a`.
    pub fn sq(&mut self, a: Var) -> Var {
        let av = self.nodes[a.idx].value;
        self.push(av * av, [(a.idx, 2.0 * av), (0, 0.0)], 1)
    }

    /// Negate: `z = -a`.
    pub fn neg(&mut self, a: Var) -> Var {
        let av = self.nodes[a.idx].value;
        self.push(-av, [(a.idx, -1.0), (0, 0.0)], 1)
    }

    /// Scalar multiply: `z = s * a`.
    pub fn scale(&mut self, a: Var, s: f32) -> Var {
        let av = self.nodes[a.idx].value;
        self.push(s * av, [(a.idx, s), (0, 0.0)], 1)
    }

    /// Reverse pass: returns gradient vector indexed by node index.
    ///
    /// Sets `grads[output.idx] = 1.0` then propagates backward via chain rule.
    pub fn gradient(&self, output: Var) -> PinnResult<Vec<f32>> {
        let n = self.nodes.len();
        if output.idx >= n {
            return Err(PinnError::TapeIndexOutOfRange {
                idx: output.idx,
                size: n,
            });
        }
        let mut grads = vec![0.0_f32; n];
        grads[output.idx] = 1.0;

        for node_idx in (0..n).rev() {
            let g = grads[node_idx];
            if g == 0.0 {
                continue;
            }
            let node = &self.nodes[node_idx];
            for dep_i in 0..node.n_deps as usize {
                let (parent_idx, partial) = node.deps[dep_i];
                if parent_idx < n {
                    grads[parent_idx] += g * partial;
                }
            }
        }
        Ok(grads)
    }
}

/// A reference to a node in the tape, carrying node index and tape length
/// at creation time (for validation).
#[derive(Clone, Copy, Debug)]
pub struct Var {
    pub idx: usize,
    /// Tape length when this `Var` was created; used for staleness checks.
    pub tape_len: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tape_x_squared_grad() {
        // f(x) = x^2, grad = 2x for x=3
        let mut tape = Tape::new();
        let x = tape.variable(3.0);
        let f = tape.sq(x);
        let grads = tape.gradient(f).unwrap();
        assert!(
            (grads[x.idx] - 6.0).abs() < 1e-6,
            "grad f(x)=x² at x=3 should be 6, got {}",
            grads[x.idx]
        );
    }

    #[test]
    fn tape_xy_partial_derivatives() {
        // f(x, y) = x*y; ∂f/∂x = y, ∂f/∂y = x
        let mut tape = Tape::new();
        let x = tape.variable(2.0);
        let y = tape.variable(5.0);
        let f = tape.mul(x, y);
        let grads = tape.gradient(f).unwrap();
        assert!((grads[x.idx] - 5.0).abs() < 1e-6);
        assert!((grads[y.idx] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn tape_sin_grad() {
        // f(x) = sin(x), grad = cos(x)
        let x_val = 1.2_f32;
        let mut tape = Tape::new();
        let x = tape.variable(x_val);
        let f = tape.sin(x);
        let grads = tape.gradient(f).unwrap();
        assert!((grads[x.idx] - x_val.cos()).abs() < 1e-5);
    }

    #[test]
    fn tape_exp_grad() {
        // f(x) = exp(x), grad = exp(x)
        let x_val = 0.8_f32;
        let mut tape = Tape::new();
        let x = tape.variable(x_val);
        let f = tape.exp(x);
        let grads = tape.gradient(f).unwrap();
        assert!((grads[x.idx] - x_val.exp()).abs() < 1e-5);
    }

    #[test]
    fn tape_chain_rule_exp_x_squared() {
        // f(x) = exp(x^2), f'(x) = 2x * exp(x^2)
        let x_val = 0.5_f32;
        let mut tape = Tape::new();
        let x = tape.variable(x_val);
        let x2 = tape.sq(x);
        let f = tape.exp(x2);
        let grads = tape.gradient(f).unwrap();
        let expected = 2.0 * x_val * (x_val * x_val).exp();
        assert!((grads[x.idx] - expected).abs() < 1e-5);
    }

    #[test]
    fn tape_add_sub_grads() {
        let mut tape = Tape::new();
        let x = tape.variable(3.0);
        let y = tape.variable(2.0);
        let s = tape.add(x, y); // x + y
        let grads_s = tape.gradient(s).unwrap();
        assert!((grads_s[x.idx] - 1.0).abs() < 1e-6);
        assert!((grads_s[y.idx] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn tape_div_grad() {
        // f(x, y) = x/y; ∂f/∂x = 1/y, ∂f/∂y = -x/y²
        let x_val = 6.0_f32;
        let y_val = 2.0_f32;
        let mut tape = Tape::new();
        let x = tape.variable(x_val);
        let y = tape.variable(y_val);
        let f = tape.div(x, y).unwrap();
        let grads = tape.gradient(f).unwrap();
        assert!((grads[x.idx] - 0.5).abs() < 1e-6);
        assert!((grads[y.idx] - (-x_val / y_val.powi(2))).abs() < 1e-6);
    }

    #[test]
    fn tape_tanh_grad() {
        let x_val = 0.6_f32;
        let mut tape = Tape::new();
        let x = tape.variable(x_val);
        let f = tape.tanh(x);
        let grads = tape.gradient(f).unwrap();
        let t = x_val.tanh();
        assert!((grads[x.idx] - (1.0 - t * t)).abs() < 1e-5);
    }

    #[test]
    fn tape_neg_grad() {
        let mut tape = Tape::new();
        let x = tape.variable(3.0);
        let f = tape.neg(x);
        let grads = tape.gradient(f).unwrap();
        assert!((grads[x.idx] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn tape_scale_grad() {
        let mut tape = Tape::new();
        let x = tape.variable(4.0);
        let f = tape.scale(x, 3.0);
        let grads = tape.gradient(f).unwrap();
        assert!((grads[x.idx] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn tape_out_of_range_error() {
        let tape = Tape::new();
        let fake_var = Var {
            idx: 99,
            tape_len: 100,
        };
        assert!(tape.gradient(fake_var).is_err());
    }

    #[test]
    fn tape_multivar_quadratic() {
        // f(x, y) = x^2 + y^2; ∂f/∂x = 2x, ∂f/∂y = 2y
        let x_val = 3.0_f32;
        let y_val = 4.0_f32;
        let mut tape = Tape::new();
        let x = tape.variable(x_val);
        let y = tape.variable(y_val);
        let x2 = tape.sq(x);
        let y2 = tape.sq(y);
        let f = tape.add(x2, y2);
        let grads = tape.gradient(f).unwrap();
        assert!((grads[x.idx] - 2.0 * x_val).abs() < 1e-5);
        assert!((grads[y.idx] - 2.0 * y_val).abs() < 1e-5);
    }
}
