//! Transformer-based tabular models: FT-Transformer, AutoInt, DCN V2,
//! TabTransformer, RoPE FT-Transformer, and TabPFN in-context classifier.

pub mod autoint;
pub mod dcn_v2;
pub mod ft_rope;
pub mod ft_transformer;
pub mod ft_transformer_grad;
pub mod ft_transformer_v2;
pub mod tab_transformer;
pub mod tabpfn;
pub mod unified_encoder;

pub use autoint::{AutoInt, AutoIntConfig, AutoIntLayerWeights, AutoIntWeights, layer_norm};
pub use dcn_v2::{
    CrossLayerWeights, DcnV2, DcnV2Config, DcnV2Mode, DcnV2Weights, DeepLayerWeights,
};
// The compact scalar-head FT-Transformer is re-exported under aliased names so
// it coexists with the multi-class `ft_transformer::FtTransformer`.
pub use ft_rope::{FtRopeConfig, FtRopeTransformer};
pub use ft_transformer_v2::{
    FtTransformer as FtTransformerV2, FtTransformerConfig as FtTransformerV2Config,
};
pub use tab_transformer::{TabTransformer, TabTransformerConfig, TabTransformerWeights};
pub use tabpfn::{TabPfn, TabPfnConfig};
