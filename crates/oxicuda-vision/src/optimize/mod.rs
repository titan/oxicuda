//! Inference-time optimization passes for vision models.

pub mod bn_folding;

pub use bn_folding::{BnParams, fold_bn_into_conv, fold_bn_into_linear, verify_bn_fold};
