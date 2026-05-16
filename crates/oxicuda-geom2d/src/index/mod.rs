//! 2D spatial indices: KD-tree, R-tree, Quadtree.

pub mod kd_tree_2d;
pub mod quadtree;
pub mod rtree_2d;

pub use kd_tree_2d::KdTree2d;
pub use quadtree::Quadtree;
pub use rtree_2d::Rtree2d;
