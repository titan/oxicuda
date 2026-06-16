//! Classical image-processing primitives operating on flat `f32` buffers.
//!
//! Provides:
//! - **`edges`**: Sobel gradient operator and the full Canny edge detector
//!   (non-maximum suppression + hysteresis double thresholding).
//! - **`morphology`**: binary / grayscale mathematical morphology — erosion,
//!   dilation, opening, closing, and the morphological gradient with arbitrary
//!   structuring elements.
//! - **`connected_components`**: union-find connected-component labelling of
//!   binary images (4- or 8-connectivity) with per-component sizes and boxes.
//! - **`hough`**: Hough transform for straight-line detection (ρ–θ accumulator
//!   + non-maximum-suppressed peak extraction).

pub mod connected_components;
pub mod edges;
pub mod hough;
pub mod morphology;

pub use connected_components::{ComponentLabels, Connectivity, connected_components};
pub use edges::{SOBEL_GX, SOBEL_GY, SobelOutput, canny, sobel_gradients};
pub use hough::{HoughAccumulator, HoughConfig, HoughLine, hough_accumulate, hough_lines};
pub use morphology::{StructuringElement, close, dilate, erode, morphological_gradient, open};
