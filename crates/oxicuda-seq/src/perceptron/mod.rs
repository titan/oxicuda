//! Structured-perceptron sequence taggers (Collins 2002), including the
//! averaged-perceptron variant.

pub mod structured_perceptron;

pub use structured_perceptron::{
    PerceptronConfig, PerceptronExample, PerceptronTrainResult, StructuredPerceptron,
    train_perceptron,
};
