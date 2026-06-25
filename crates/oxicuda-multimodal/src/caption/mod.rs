//! Caption generation and VQA heads.

pub mod prefix_lm;
pub mod sampling;
pub mod vqa_head;

pub use sampling::{
    Beam, SamplingConfig, beam_search, nucleus_filter, sample_categorical, sample_token,
    temperature_softmax, top_k_filter,
};
