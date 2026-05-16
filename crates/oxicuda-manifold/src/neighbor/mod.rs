//! Nearest-neighbour search data structures.
//!
//! - [`knn_brute()`] O(n^2) brute force pairwise distances.
//! - [`kd_tree`] axis-aligned KD-tree (median split).
//! - [`ball_tree`] ball tree with centroid + radius bounding.

pub mod ball_tree;
pub mod kd_tree;
pub mod knn_brute;

pub use ball_tree::{BallTree, BallTreeNode};
pub use kd_tree::{KdTree, KdTreeNode};
pub use knn_brute::{knn_brute, knn_brute_from_distance_matrix};
