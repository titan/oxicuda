//! Closest pair of points: brute force O(n^2) and divide-and-conquer O(n log n).

pub mod brute_force;
pub mod divide_conquer;

pub use brute_force::closest_pair_brute;
pub use divide_conquer::closest_pair_dc;
