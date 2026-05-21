//! GNN layer implementations.

pub mod appnp;
pub mod edgeconv;
pub mod gat;
pub mod gat_v2;
pub mod gcn;
pub mod gin;
pub mod graph_transformer;
pub mod jk_net;
pub mod pna;
pub mod rgcn;
pub mod sage;
pub mod sgc;

pub use appnp::{AppnpConfig, AppnpLayer};
pub use edgeconv::{EdgeConvConfig, EdgeConvLayer, EdgeConvMode, edge_feature};
pub use graph_transformer::{
    GraphTransformerConfig, GraphTransformerLayer, GraphTransformerWeights,
};
pub use jk_net::{JkMode, JkNet, JkNetConfig};
pub use pna::{PnaAggregator, PnaConfig, PnaLayer, PnaScaler, aggregate, scale};
pub use rgcn::{RgcnConfig, RgcnLayer};
pub use sgc::{sgc_forward, sgc_linear, sgc_propagate};
