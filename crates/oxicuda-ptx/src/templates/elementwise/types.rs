//! Operation enum and classifiers for elementwise kernel templates.
//!
//! Refactored with [SplitRS](https://github.com/cool-japan/splitrs).

/// Elementwise operation type.
///
/// Covers binary arithmetic, unary activations, unary math, and fused operations.
/// Each variant determines the kernel signature (number of input/output pointers)
/// and the PTX instruction sequence emitted in the kernel body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementwiseOp {
    /// Element-wise addition: `c[i] = a[i] + b[i]`.
    Add,
    /// Element-wise subtraction: `c[i] = a[i] - b[i]`.
    Sub,
    /// Element-wise multiplication: `c[i] = a[i] * b[i]`.
    Mul,
    /// Element-wise division: `c[i] = a[i] / b[i]`.
    Div,
    /// Rectified linear unit: `b[i] = max(0, a[i])`.
    Relu,
    /// Gaussian error linear unit (tanh approximation):
    /// `b[i] = 0.5 * a[i] * (1 + tanh(sqrt(2/pi) * (a[i] + 0.044715 * a[i]^3)))`.
    Gelu,
    /// Sigmoid activation: `b[i] = 1 / (1 + exp(-a[i]))`.
    Sigmoid,
    /// Sigmoid linear unit: `b[i] = a[i] * sigmoid(a[i])`.
    Silu,
    /// Hyperbolic tangent: `b[i] = tanh(a[i])`.
    Tanh,
    /// Arithmetic negation: `b[i] = -a[i]`.
    Neg,
    /// Absolute value: `b[i] = |a[i]|`.
    Abs,
    /// Square root: `b[i] = sqrt(a[i])`.
    Sqrt,
    /// Reciprocal square root: `b[i] = 1 / sqrt(a[i])`.
    Rsqrt,
    /// Exponential: `b[i] = exp(a[i])`.
    Exp,
    /// Natural logarithm: `b[i] = ln(a[i])`.
    Log,
    /// Scalar scaling: `b[i] = alpha * a[i]`.
    Scale,
    /// Add scalar: `b[i] = a[i] + scalar`.
    AddScalar,
    /// Ceiling (round toward +inf): `b[i] = ceil(a[i])`.
    Ceil,
    /// Floor (round toward -inf): `b[i] = floor(a[i])`.
    Floor,
    /// Hard sigmoid: `b[i] = max(0, min(1, 0.2*a[i] + 0.5))`.
    HardSigmoid,
    /// Hard swish: `b[i] = a[i] * max(0, min(6, a[i]+3)) / 6`.
    HardSwish,
    /// Softplus: `b[i] = ln(1 + exp(a[i]))`.
    Softplus,
    /// Leaky relu: `b[i] = a[i] >= 0 ? a[i] : 0.01 * a[i]`.
    LeakyRelu,
    /// One-minus: `b[i] = 1 - a[i]`.
    OneMinus,
    /// Fused add-relu: `c[i] = relu(a[i] + b[i])`.
    FusedAddRelu,
    /// Fused scale-add: `c[i] = alpha * a[i] + beta * b[i]`.
    FusedScaleAdd,
    /// Element-wise power: `c[i] = a[i]^b[i]` (lg2+mul+ex2 approximation).
    Pow,
    /// Element-wise minimum: `c[i] = min(a[i], b[i])`.
    Min,
    /// Element-wise maximum: `c[i] = max(a[i], b[i])`.
    Max,
    /// Comparison equal: `c[i] = (a[i] == b[i]) ? 1.0 : 0.0`.
    CmpEq,
    /// Comparison not-equal: `c[i] = (a[i] != b[i]) ? 1.0 : 0.0`.
    CmpNe,
    /// Comparison less-than: `c[i] = (a[i] < b[i]) ? 1.0 : 0.0`.
    CmpLt,
    /// Comparison greater-than: `c[i] = (a[i] > b[i]) ? 1.0 : 0.0`.
    CmpGt,
    /// Comparison less-or-equal: `c[i] = (a[i] <= b[i]) ? 1.0 : 0.0`.
    CmpLe,
    /// Comparison greater-or-equal: `c[i] = (a[i] >= b[i]) ? 1.0 : 0.0`.
    CmpGe,
    /// Fuzzy OR via max (same PTX as Max, distinct semantic label).
    OrMax,
    /// Probabilistic OR: `c[i] = a[i] + b[i] - a[i]*b[i]`.
    OrProbSum,
    /// Fuzzy NAND: `c[i] = 1 - a[i]*b[i]`.
    Nand,
    /// Fuzzy NOR: `c[i] = 1 - (a[i] + b[i] - a[i]*b[i])`.
    Nor,
    /// Fuzzy XOR: `c[i] = a[i] + b[i] - 2*a[i]*b[i]`.
    Xor,
    /// Fill: write a scalar to every element: `dst[i] = value`.
    Fill,
}
impl ElementwiseOp {
    /// Returns a short lowercase name suitable for kernel naming.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Relu => "relu",
            Self::Gelu => "gelu",
            Self::Sigmoid => "sigmoid",
            Self::Silu => "silu",
            Self::Tanh => "tanh",
            Self::Neg => "neg",
            Self::Abs => "abs",
            Self::Sqrt => "sqrt",
            Self::Rsqrt => "rsqrt",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Ceil => "ceil",
            Self::Floor => "floor",
            Self::HardSigmoid => "hard_sigmoid",
            Self::HardSwish => "hard_swish",
            Self::Softplus => "softplus",
            Self::LeakyRelu => "leaky_relu",
            Self::OneMinus => "one_minus",
            Self::Pow => "pow",
            Self::Min => "min",
            Self::Max => "max",
            Self::CmpEq => "cmp_eq",
            Self::CmpNe => "cmp_ne",
            Self::CmpLt => "cmp_lt",
            Self::CmpGt => "cmp_gt",
            Self::CmpLe => "cmp_le",
            Self::CmpGe => "cmp_ge",
            Self::OrMax => "or_max",
            Self::OrProbSum => "or_prob_sum",
            Self::Nand => "nand",
            Self::Nor => "nor",
            Self::Xor => "xor",
            Self::Scale => "scale",
            Self::AddScalar => "add_scalar",
            Self::FusedAddRelu => "fused_add_relu",
            Self::FusedScaleAdd => "fused_scale_add",
            Self::Fill => "fill",
        }
    }
    /// Returns `true` if this is a binary operation requiring two input arrays.
    #[must_use]
    pub const fn is_binary(self) -> bool {
        matches!(
            self,
            Self::Add
                | Self::Sub
                | Self::Mul
                | Self::Div
                | Self::FusedAddRelu
                | Self::FusedScaleAdd
                | Self::Pow
                | Self::Min
                | Self::Max
                | Self::CmpEq
                | Self::CmpNe
                | Self::CmpLt
                | Self::CmpGt
                | Self::CmpLe
                | Self::CmpGe
                | Self::OrMax
                | Self::OrProbSum
                | Self::Nand
                | Self::Nor
                | Self::Xor
        )
    }
    /// Returns `true` if this operation requires scalar parameter(s).
    #[must_use]
    pub const fn needs_scalar(self) -> bool {
        matches!(
            self,
            Self::Scale | Self::AddScalar | Self::FusedScaleAdd | Self::Fill
        )
    }
    /// Returns `true` if this operation is implemented with the special
    /// function unit (SFU) approximation instructions `ex2.approx` /
    /// `lg2.approx`, which PTX only defines for `.f32`.
    ///
    /// Such kernels can therefore only be generated at [`crate::ir::PtxType::F32`]
    /// precision; requesting F16/BF16/F64 for them yields PTX that `ptxas`
    /// rejects (`Unexpected instruction types specified for 'ex2'`), so the
    /// template refuses those precisions up front with a clear error rather
    /// than emitting a module that fails at load time. Note that `rsqrt`,
    /// `sqrt`, and the `hard_*`/`leaky_relu` activations are intentionally
    /// excluded: they use only instructions (`rsqrt.approx.f64`, `sqrt.rn`,
    /// `min`/`max`/`mul`/`add`/`selp`) that PTX defines for every supported
    /// precision.
    #[must_use]
    pub const fn requires_f32_sfu(self) -> bool {
        matches!(
            self,
            Self::Exp
                | Self::Log
                | Self::Sigmoid
                | Self::Gelu
                | Self::Silu
                | Self::Tanh
                | Self::Softplus
                | Self::Pow
        )
    }
}
