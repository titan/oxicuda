//! Type definition for the elementwise PTX kernel template.
//!
//! Refactored with [SplitRS](https://github.com/cool-japan/splitrs).

use crate::arch::SmVersion;
use crate::ir::PtxType;

use super::types::ElementwiseOp;

/// Template for generating elementwise PTX kernels.
///
/// Combines an [`ElementwiseOp`], a precision ([`PtxType`]), and a target
/// architecture ([`SmVersion`]) to produce a complete PTX module string.
///
/// The generated kernel handles global thread indexing and bounds checking.
/// For complex activations (GELU, sigmoid, `SiLU`), the template emits
/// approximate PTX instruction sequences using `ex2.approx` and `rcp.approx`.
pub struct ElementwiseTemplate {
    /// The elementwise operation to generate.
    pub op: ElementwiseOp,
    /// The data precision for computation (e.g., `PtxType::F32`).
    pub precision: PtxType,
    /// The target GPU architecture.
    pub target: SmVersion,
}
