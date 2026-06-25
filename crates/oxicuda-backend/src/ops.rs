//! Operation descriptor enums shared by every backend.

use std::fmt;

/// Transpose mode for matrix operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendTranspose {
    /// No transpose — use the matrix as-is.
    NoTrans,
    /// Transpose (swap rows and columns).
    Trans,
    /// Conjugate transpose (Hermitian).
    ConjTrans,
}

impl fmt::Display for BackendTranspose {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoTrans => write!(f, "N"),
            Self::Trans => write!(f, "T"),
            Self::ConjTrans => write!(f, "C"),
        }
    }
}

/// Reduction operation applied along an axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    /// Summation.
    Sum,
    /// Maximum value.
    Max,
    /// Minimum value.
    Min,
    /// Arithmetic mean.
    Mean,
}

impl fmt::Display for ReduceOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sum => write!(f, "sum"),
            Self::Max => write!(f, "max"),
            Self::Min => write!(f, "min"),
            Self::Mean => write!(f, "mean"),
        }
    }
}

/// Element-wise unary operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    /// Rectified linear unit: max(0, x).
    Relu,
    /// Sigmoid: 1 / (1 + exp(-x)).
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// Exponential.
    Exp,
    /// Natural logarithm.
    Log,
    /// Square root.
    Sqrt,
    /// Absolute value.
    Abs,
    /// Negation.
    Neg,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Relu => write!(f, "relu"),
            Self::Sigmoid => write!(f, "sigmoid"),
            Self::Tanh => write!(f, "tanh"),
            Self::Exp => write!(f, "exp"),
            Self::Log => write!(f, "log"),
            Self::Sqrt => write!(f, "sqrt"),
            Self::Abs => write!(f, "abs"),
            Self::Neg => write!(f, "neg"),
        }
    }
}

/// Reduced-precision **input storage** format for a mixed-precision GEMM.
///
/// Mixed-precision GEMM stores its `A`/`B` operands in a 16-bit format but
/// accumulates the dot products in `f32` (the Tensor-Core / WMMA contract).
/// This enum names the input storage format; the accumulator and output are
/// always `f32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MixedPrecision {
    /// IEEE-754 binary16 (half): 1 sign, 5 exponent, 10 mantissa bits.
    F16,
    /// Google bfloat16: 1 sign, 8 exponent, 7 mantissa bits (the top 16 bits
    /// of an `f32`).
    Bf16,
}

impl fmt::Display for MixedPrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F16 => write!(f, "f16"),
            Self::Bf16 => write!(f, "bf16"),
        }
    }
}

/// Element-wise binary operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
    /// Element-wise maximum.
    Max,
    /// Element-wise minimum.
    Min,
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Add => write!(f, "add"),
            Self::Sub => write!(f, "sub"),
            Self::Mul => write!(f, "mul"),
            Self::Div => write!(f, "div"),
            Self::Max => write!(f, "max"),
            Self::Min => write!(f, "min"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_transpose_display_and_values() {
        assert_eq!(BackendTranspose::NoTrans.to_string(), "N");
        assert_eq!(BackendTranspose::Trans.to_string(), "T");
        assert_eq!(BackendTranspose::ConjTrans.to_string(), "C");

        assert_eq!(BackendTranspose::NoTrans, BackendTranspose::NoTrans);
        assert_ne!(BackendTranspose::NoTrans, BackendTranspose::Trans);
    }

    #[test]
    fn reduce_op_display_and_coverage() {
        let ops = [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min, ReduceOp::Mean];
        let names = ["sum", "max", "min", "mean"];
        for (op, name) in ops.iter().zip(names.iter()) {
            assert_eq!(op.to_string(), *name);
        }
    }

    #[test]
    fn unary_op_display_and_coverage() {
        let ops = [
            UnaryOp::Relu,
            UnaryOp::Sigmoid,
            UnaryOp::Tanh,
            UnaryOp::Exp,
            UnaryOp::Log,
            UnaryOp::Sqrt,
            UnaryOp::Abs,
            UnaryOp::Neg,
        ];
        let names = [
            "relu", "sigmoid", "tanh", "exp", "log", "sqrt", "abs", "neg",
        ];
        for (op, name) in ops.iter().zip(names.iter()) {
            assert_eq!(op.to_string(), *name);
        }
    }

    #[test]
    fn mixed_precision_display_and_coverage() {
        assert_eq!(MixedPrecision::F16.to_string(), "f16");
        assert_eq!(MixedPrecision::Bf16.to_string(), "bf16");
        assert_ne!(MixedPrecision::F16, MixedPrecision::Bf16);
        // Participates in a HashSet like the other op descriptors.
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(MixedPrecision::F16);
        assert!(set.contains(&MixedPrecision::F16));
        assert!(!set.contains(&MixedPrecision::Bf16));
    }

    #[test]
    fn binary_op_display_and_coverage() {
        let ops = [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Max,
            BinaryOp::Min,
        ];
        let names = ["add", "sub", "mul", "div", "max", "min"];
        for (op, name) in ops.iter().zip(names.iter()) {
            assert_eq!(op.to_string(), *name);
        }
    }

    #[test]
    fn enum_clone_and_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(ReduceOp::Sum);
        set.insert(ReduceOp::Max);
        assert!(set.contains(&ReduceOp::Sum));
        assert!(!set.contains(&ReduceOp::Min));

        let op = UnaryOp::Relu;
        let cloned = op;
        assert_eq!(op, cloned);

        let bop = BinaryOp::Add;
        let bcloned = bop;
        assert_eq!(bop, bcloned);

        let trans = BackendTranspose::ConjTrans;
        let tcloned = trans;
        assert_eq!(trans, tcloned);
    }
}
