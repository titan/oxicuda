//! Streaming statistics: online mean/variance, exponential decay, sliding window counts.

pub mod exponential_decay;
pub mod online_mean_var;
pub mod sliding_window;

pub use exponential_decay::ExponentialDecay;
pub use online_mean_var::WelfordOnline;
pub use sliding_window::SlidingWindowCount;
