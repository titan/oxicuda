//! Error types for the `oxicuda-gnn` crate.

use thiserror::Error;

/// All errors that can arise from GNN operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum GnnError {
    /// Node feature matrix row count does not match graph node count.
    #[error("graph has {0} nodes but feature matrix has {1} rows")]
    NodeFeatureMismatch(usize, usize),

    /// Edge feature row count does not match graph edge count.
    #[error("graph has {0} edges but edge features have {1} rows")]
    EdgeFeatureMismatch(usize, usize),

    /// A node index is out of bounds.
    #[error("node index {idx} out of range [0, {n_nodes})")]
    NodeIndexOutOfRange { idx: usize, n_nodes: usize },

    /// Graph has no nodes.
    #[error("empty graph: must have at least one node")]
    EmptyGraph,

    /// Aggregation type or parameters are invalid.
    #[error("invalid aggregation: {0}")]
    InvalidAggregation(&'static str),

    /// Layer configuration is invalid.
    #[error("invalid layer config: {0}")]
    InvalidLayerConfig(String),

    /// Dimension mismatch between tensors.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Output tensor contains NaN or Inf values.
    #[error("invalid GNN output: {0} contains non-finite values")]
    NonFiniteOutput(&'static str),

    /// Top-K pool k exceeds graph node count.
    #[error("Top-K pool k={k} exceeds graph size n={n}")]
    TopKExceedsGraphSize { k: usize, n: usize },

    /// Heterogeneous graph has no edges between given node types.
    #[error("heterogeneous graph: node type '{src}' has no edges to '{dst}'")]
    NoEdgeType { src: String, dst: String },

    /// Neighborhood sampling depth exceeds allowed maximum.
    #[error("sampling depth {0} exceeds maximum {1}")]
    SamplingDepthExceeded(usize, usize),

    /// Weight matrix shape is incompatible with input dimension.
    #[error("weight matrix shape [{r}×{c}] incompatible with input dim {d}")]
    WeightShapeMismatch { r: usize, c: usize, d: usize },

    /// GAT head count does not evenly divide embed_dim.
    #[error("invalid GAT heads: embed_dim {dim} must be divisible by num_heads {heads}")]
    InvalidAttentionHeads { dim: usize, heads: usize },

    /// Internal invariant violation.
    #[error("internal error: {0}")]
    Internal(String),
}

/// Convenience `Result` alias.
pub type GnnResult<T> = Result<T, GnnError>;

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_node_feature_mismatch() {
        let e = GnnError::NodeFeatureMismatch(10, 8);
        let s = e.to_string();
        assert!(s.contains("10"));
        assert!(s.contains("8"));
    }

    #[test]
    fn error_display_edge_feature_mismatch() {
        let e = GnnError::EdgeFeatureMismatch(20, 15);
        let s = e.to_string();
        assert!(s.contains("20"));
        assert!(s.contains("15"));
    }

    #[test]
    fn error_display_node_index_out_of_range() {
        let e = GnnError::NodeIndexOutOfRange { idx: 5, n_nodes: 4 };
        let s = e.to_string();
        assert!(s.contains("5"));
        assert!(s.contains("4"));
    }

    #[test]
    fn error_display_empty_graph() {
        let e = GnnError::EmptyGraph;
        assert!(e.to_string().contains("empty graph"));
    }

    #[test]
    fn error_display_invalid_aggregation() {
        let e = GnnError::InvalidAggregation("unknown type");
        assert!(e.to_string().contains("unknown type"));
    }

    #[test]
    fn error_display_invalid_layer_config() {
        let e = GnnError::InvalidLayerConfig("in_features must be > 0".to_string());
        assert!(e.to_string().contains("in_features must be > 0"));
    }

    #[test]
    fn error_display_dimension_mismatch() {
        let e = GnnError::DimensionMismatch {
            expected: 64,
            got: 32,
        };
        let s = e.to_string();
        assert!(s.contains("64"));
        assert!(s.contains("32"));
    }

    #[test]
    fn error_display_non_finite_output() {
        let e = GnnError::NonFiniteOutput("GCN forward");
        assert!(e.to_string().contains("GCN forward"));
    }

    #[test]
    fn error_display_topk_exceeds_graph_size() {
        let e = GnnError::TopKExceedsGraphSize { k: 10, n: 5 };
        let s = e.to_string();
        assert!(s.contains("10"));
        assert!(s.contains("5"));
    }

    #[test]
    fn error_display_no_edge_type() {
        let e = GnnError::NoEdgeType {
            src: "paper".into(),
            dst: "author".into(),
        };
        let s = e.to_string();
        assert!(s.contains("paper"));
        assert!(s.contains("author"));
    }

    #[test]
    fn error_display_sampling_depth_exceeded() {
        let e = GnnError::SamplingDepthExceeded(5, 3);
        let s = e.to_string();
        assert!(s.contains("5"));
        assert!(s.contains("3"));
    }

    #[test]
    fn error_display_weight_shape_mismatch() {
        let e = GnnError::WeightShapeMismatch {
            r: 64,
            c: 32,
            d: 16,
        };
        let s = e.to_string();
        assert!(s.contains("64"));
        assert!(s.contains("32"));
        assert!(s.contains("16"));
    }

    #[test]
    fn error_display_invalid_attention_heads() {
        let e = GnnError::InvalidAttentionHeads { dim: 64, heads: 3 };
        let s = e.to_string();
        assert!(s.contains("64"));
        assert!(s.contains("3"));
    }

    #[test]
    fn error_display_internal() {
        let e = GnnError::Internal("unexpected state".to_string());
        assert!(e.to_string().contains("unexpected state"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(GnnError::Internal("test".to_string()));
        assert!(e.to_string().contains("test"));
    }

    #[test]
    fn error_clone_and_eq() {
        let a = GnnError::EmptyGraph;
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn error_result_alias() {
        fn produce_ok() -> GnnResult<usize> {
            Ok(42)
        }
        fn produce_err() -> GnnResult<usize> {
            Err(GnnError::EmptyGraph)
        }
        let val = produce_ok().expect("test invariant: value must be valid");
        assert_eq!(val, 42);
        assert!(produce_err().is_err());
    }
}
