//! Error types for `oxicuda-geometry3d`.

use thiserror::Error;

/// All errors that can be returned from `oxicuda-geometry3d`.
#[derive(Debug, Error, PartialEq)]
pub enum Geom3dError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("empty point cloud")]
    EmptyPointCloud,

    #[error("points must be 3D, got dim {dim}")]
    InvalidPointDim { dim: usize },

    #[error("k={k} exceeds n={n} points")]
    InvalidK { k: usize, n: usize },

    #[error("invalid radius {radius}: must be > 0 and finite")]
    InvalidRadius { radius: f32 },

    #[error("invalid voxel size {voxel_size}: must be > 0")]
    InvalidVoxelSize { voxel_size: f32 },

    #[error("invalid sample count: requested {requested} but only {available} available")]
    InvalidSampleCount { requested: usize, available: usize },

    #[error("invalid SH coefficients: expected {expected}, got {got}")]
    InvalidShCoefficients { expected: usize, got: usize },

    #[error("invalid quaternion: norm {norm} is near zero")]
    InvalidQuaternion { norm: f32 },

    #[error("ICP did not converge after {iterations} iterations (residual {residual})")]
    IcpDidNotConverge { iterations: usize, residual: f32 },

    #[error("EMD/Sinkhorn did not converge after {iterations} iterations")]
    EmdDidNotConverge { iterations: usize },

    #[error("invalid topology: {reason}")]
    InvalidTopology { reason: &'static str },

    #[error("invalid covariance: {reason}")]
    InvalidCovariance { reason: &'static str },

    #[error("NaN encountered at: {location}")]
    NanEncountered { location: &'static str },

    #[error("batch size mismatch: lhs={lhs}, rhs={rhs}")]
    BatchSizeMismatch { lhs: usize, rhs: usize },

    #[error("internal error: {0}")]
    Internal(String),
}

/// Convenience result alias.
pub type Geom3dResult<T> = Result<T, Geom3dError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_dimension_mismatch() {
        let e = Geom3dError::DimensionMismatch {
            expected: 3,
            got: 4,
        };
        assert!(e.to_string().contains("3") && e.to_string().contains("4"));
    }

    #[test]
    fn error_display_empty_point_cloud() {
        let e = Geom3dError::EmptyPointCloud;
        assert!(e.to_string().contains("empty"));
    }

    #[test]
    fn error_display_invalid_point_dim() {
        let e = Geom3dError::InvalidPointDim { dim: 2 };
        assert!(e.to_string().contains("2"));
        assert!(e.to_string().contains("3D"));
    }

    #[test]
    fn error_display_invalid_k() {
        let e = Geom3dError::InvalidK { k: 10, n: 5 };
        assert!(e.to_string().contains("10") && e.to_string().contains("5"));
    }

    #[test]
    fn error_display_invalid_radius() {
        let e = Geom3dError::InvalidRadius { radius: -1.0 };
        assert!(e.to_string().contains("-1"));
    }

    #[test]
    fn error_display_invalid_voxel_size() {
        let e = Geom3dError::InvalidVoxelSize { voxel_size: 0.0 };
        assert!(e.to_string().contains("voxel"));
    }

    #[test]
    fn error_display_invalid_sample_count() {
        let e = Geom3dError::InvalidSampleCount {
            requested: 20,
            available: 10,
        };
        assert!(e.to_string().contains("20") && e.to_string().contains("10"));
    }

    #[test]
    fn error_display_invalid_sh_coefficients() {
        let e = Geom3dError::InvalidShCoefficients {
            expected: 27,
            got: 9,
        };
        assert!(e.to_string().contains("27") && e.to_string().contains("9"));
    }

    #[test]
    fn error_display_invalid_quaternion() {
        let e = Geom3dError::InvalidQuaternion { norm: 0.0 };
        assert!(e.to_string().contains("norm") || e.to_string().contains("quaternion"));
    }

    #[test]
    fn error_display_icp_did_not_converge() {
        let e = Geom3dError::IcpDidNotConverge {
            iterations: 100,
            residual: 0.5,
        };
        assert!(e.to_string().contains("100"));
    }

    #[test]
    fn error_display_emd_did_not_converge() {
        let e = Geom3dError::EmdDidNotConverge { iterations: 50 };
        assert!(e.to_string().contains("50"));
    }

    #[test]
    fn error_display_invalid_topology() {
        let e = Geom3dError::InvalidTopology {
            reason: "degenerate face",
        };
        assert!(e.to_string().contains("degenerate face"));
    }

    #[test]
    fn error_display_invalid_covariance() {
        let e = Geom3dError::InvalidCovariance {
            reason: "not positive-definite",
        };
        assert!(e.to_string().contains("not positive-definite"));
        assert!(e.to_string().contains("covariance"));
    }

    #[test]
    fn error_display_nan_encountered() {
        let e = Geom3dError::NanEncountered {
            location: "chamfer",
        };
        assert!(e.to_string().contains("chamfer"));
    }

    #[test]
    fn error_display_batch_size_mismatch() {
        let e = Geom3dError::BatchSizeMismatch { lhs: 8, rhs: 16 };
        assert!(e.to_string().contains("8") && e.to_string().contains("16"));
    }

    #[test]
    fn error_display_internal() {
        let e = Geom3dError::Internal("unexpected shape".into());
        assert!(e.to_string().contains("unexpected shape"));
    }

    #[test]
    fn error_partial_eq() {
        assert_eq!(Geom3dError::EmptyPointCloud, Geom3dError::EmptyPointCloud);
        assert_ne!(
            Geom3dError::DimensionMismatch {
                expected: 3,
                got: 4
            },
            Geom3dError::DimensionMismatch {
                expected: 5,
                got: 4
            }
        );
    }

    #[test]
    fn result_type_alias_ok() {
        let r: Geom3dResult<i32> = Ok(42);
        assert!(r.is_ok());
    }

    #[test]
    fn result_type_alias_err() {
        let r: Geom3dResult<i32> = Err(Geom3dError::EmptyPointCloud);
        assert!(r.is_err());
    }
}
