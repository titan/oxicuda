//! Marching Cubes isosurface extraction (Lorensen & Cline, SIGGRAPH 1987).
//!
//! Converts a 3D signed distance field (SDF) or density volume on a regular
//! grid into a triangular mesh by classifying each voxel using a 256-case
//! lookup table.

use crate::error::{Geom3dError, Geom3dResult};

// ─── Lookup tables ────────────────────────────────────────────────────────────
//
// Standard Lorensen & Cline EDGE_TABLE and TRI_TABLE, widely published
// (original 1987 paper; Paul Bourke's reference implementation is public domain).
//
// EDGE_TABLE[cube_index]: bitmask of which of the 12 edges are intersected.
// TRI_TABLE[cube_index]: up to 5 triangles (16 entries), edge indices, -1 sentinel.

#[rustfmt::skip]
const EDGE_TABLE: [u16; 256] = [
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000,
];

#[rustfmt::skip]
const TRI_TABLE: [[i8; 16]; 256] = [
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 8, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 1, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1, 8, 3, 9, 8, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1, 2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 8, 3, 1, 2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9, 2,10, 0, 2, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2, 8, 3, 2,10, 8,10, 9, 8,-1,-1,-1,-1,-1,-1,-1],
    [3,11, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,11, 2, 8,11, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1, 9, 0, 2, 3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,11, 2, 1, 9,11, 9, 8,11,-1,-1,-1,-1,-1,-1,-1],
    [3,10, 1,11,10, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,10, 1, 0, 8,10, 8,11,10,-1,-1,-1,-1,-1,-1,-1],
    [3, 9, 0, 3,11, 9,11,10, 9,-1,-1,-1,-1,-1,-1,-1],
    [9, 8,10,10, 8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4, 7, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4, 3, 0, 7, 3, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 1, 9, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4, 1, 9, 4, 7, 1, 7, 3, 1,-1,-1,-1,-1,-1,-1,-1],
    [1, 2,10, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3, 4, 7, 3, 0, 4, 1, 2,10,-1,-1,-1,-1,-1,-1,-1],
    [9, 2,10, 9, 0, 2, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1],
    [2,10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4,-1,-1,-1,-1],
    [8, 4, 7, 3,11, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 4, 7,11, 2, 4, 2, 0, 4,-1,-1,-1,-1,-1,-1,-1],
    [9, 0, 1, 8, 4, 7, 2, 3,11,-1,-1,-1,-1,-1,-1,-1],
    [4, 7,11, 9, 4,11, 9,11, 2, 9, 2, 1,-1,-1,-1,-1],
    [3,10, 1, 3,11,10, 7, 8, 4,-1,-1,-1,-1,-1,-1,-1],
    [1,11,10, 1, 4,11, 1, 0, 4, 7,11, 4,-1,-1,-1,-1],
    [4, 7, 8, 9, 0,11, 9,11,10,11, 0, 3,-1,-1,-1,-1],
    [4, 7,11, 4,11, 9, 9,11,10,-1,-1,-1,-1,-1,-1,-1],
    [9, 5, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9, 5, 4, 0, 8, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 5, 4, 1, 5, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8, 5, 4, 8, 3, 5, 3, 1, 5,-1,-1,-1,-1,-1,-1,-1],
    [1, 2,10, 9, 5, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3, 0, 8, 1, 2,10, 4, 9, 5,-1,-1,-1,-1,-1,-1,-1],
    [5, 2,10, 5, 4, 2, 4, 0, 2,-1,-1,-1,-1,-1,-1,-1],
    [2,10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8,-1,-1,-1,-1],
    [9, 5, 4, 2, 3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,11, 2, 0, 8,11, 4, 9, 5,-1,-1,-1,-1,-1,-1,-1],
    [0, 5, 4, 0, 1, 5, 2, 3,11,-1,-1,-1,-1,-1,-1,-1],
    [2, 1, 5, 2, 5, 8, 2, 8,11, 4, 8, 5,-1,-1,-1,-1],
    [10, 3,11,10, 1, 3, 9, 5, 4,-1,-1,-1,-1,-1,-1,-1],
    [4, 9, 5, 0, 8, 1, 8,10, 1, 8,11,10,-1,-1,-1,-1],
    [5, 4, 0, 5, 0,11, 5,11,10,11, 0, 3,-1,-1,-1,-1],
    [5, 4, 8, 5, 8,10,10, 8,11,-1,-1,-1,-1,-1,-1,-1],
    [9, 7, 8, 5, 7, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9, 3, 0, 9, 5, 3, 5, 7, 3,-1,-1,-1,-1,-1,-1,-1],
    [0, 7, 8, 0, 1, 7, 1, 5, 7,-1,-1,-1,-1,-1,-1,-1],
    [1, 5, 3, 3, 5, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9, 7, 8, 9, 5, 7,10, 1, 2,-1,-1,-1,-1,-1,-1,-1],
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3,-1,-1,-1,-1],
    [8, 0, 2, 8, 2, 5, 8, 5, 7,10, 5, 2,-1,-1,-1,-1],
    [2,10, 5, 2, 5, 3, 3, 5, 7,-1,-1,-1,-1,-1,-1,-1],
    [7, 9, 5, 7, 8, 9, 3,11, 2,-1,-1,-1,-1,-1,-1,-1],
    [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7,11,-1,-1,-1,-1],
    [2, 3,11, 0, 1, 8, 1, 7, 8, 1, 5, 7,-1,-1,-1,-1],
    [11, 2, 1,11, 1, 7, 7, 1, 5,-1,-1,-1,-1,-1,-1,-1],
    [9, 5, 8, 8, 5, 7,10, 1, 3,10, 3,11,-1,-1,-1,-1],
    [5, 7, 0, 5, 0, 9, 7,11, 0, 1, 0,10,11,10, 0,-1],
    [11,10, 0,11, 0, 3,10, 5, 0, 8, 0, 7, 5, 7, 0,-1],
    [11,10, 5, 7,11, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10, 6, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 8, 3, 5,10, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9, 0, 1, 5,10, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1, 8, 3, 1, 9, 8, 5,10, 6,-1,-1,-1,-1,-1,-1,-1],
    [1, 6, 5, 2, 6, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1, 6, 5, 1, 2, 6, 3, 0, 8,-1,-1,-1,-1,-1,-1,-1],
    [9, 6, 5, 9, 0, 6, 0, 2, 6,-1,-1,-1,-1,-1,-1,-1],
    [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8,-1,-1,-1,-1],
    [2, 3,11,10, 6, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 0, 8,11, 2, 0,10, 6, 5,-1,-1,-1,-1,-1,-1,-1],
    [0, 1, 9, 2, 3,11, 5,10, 6,-1,-1,-1,-1,-1,-1,-1],
    [5,10, 6, 1, 9, 2, 9,11, 2, 9, 8,11,-1,-1,-1,-1],
    [6, 3,11, 6, 5, 3, 5, 1, 3,-1,-1,-1,-1,-1,-1,-1],
    [0, 8,11, 0,11, 5, 0, 5, 1, 5,11, 6,-1,-1,-1,-1],
    [3,11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9,-1,-1,-1,-1],
    [6, 5, 9, 6, 9,11,11, 9, 8,-1,-1,-1,-1,-1,-1,-1],
    [5,10, 6, 4, 7, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4, 3, 0, 4, 7, 3, 6, 5,10,-1,-1,-1,-1,-1,-1,-1],
    [1, 9, 0, 5,10, 6, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1],
    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4,-1,-1,-1,-1],
    [6, 1, 2, 6, 5, 1, 4, 7, 8,-1,-1,-1,-1,-1,-1,-1],
    [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7,-1,-1,-1,-1],
    [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6,-1,-1,-1,-1],
    [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9,-1],
    [3,11, 2, 7, 8, 4,10, 6, 5,-1,-1,-1,-1,-1,-1,-1],
    [5,10, 6, 4, 7, 2, 4, 2, 0, 2, 7,11,-1,-1,-1,-1],
    [0, 1, 9, 4, 7, 8, 2, 3,11, 5,10, 6,-1,-1,-1,-1],
    [9, 2, 1, 9,11, 2, 9, 4,11, 7,11, 4, 5,10, 6,-1],
    [8, 4, 7, 3,11, 5, 3, 5, 1, 5,11, 6,-1,-1,-1,-1],
    [5, 1,11, 5,11, 6, 1, 0,11, 7,11, 4, 0, 4,11,-1],
    [0, 5, 9, 0, 6, 5, 0, 3, 6,11, 6, 3, 8, 4, 7,-1],
    [6, 5, 9, 6, 9,11, 4, 7, 9, 7,11, 9,-1,-1,-1,-1],
    [10, 4, 9, 6, 4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,10, 6, 4, 9,10, 0, 8, 3,-1,-1,-1,-1,-1,-1,-1],
    [10, 0, 1,10, 6, 0, 6, 4, 0,-1,-1,-1,-1,-1,-1,-1],
    [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1,10,-1,-1,-1,-1],
    [1, 4, 9, 1, 2, 4, 2, 6, 4,-1,-1,-1,-1,-1,-1,-1],
    [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4,-1,-1,-1,-1],
    [0, 2, 4, 4, 2, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8, 3, 2, 8, 2, 4, 4, 2, 6,-1,-1,-1,-1,-1,-1,-1],
    [10, 4, 9,10, 6, 4,11, 2, 3,-1,-1,-1,-1,-1,-1,-1],
    [0, 8, 2, 2, 8,11, 4, 9,10, 4,10, 6,-1,-1,-1,-1],
    [3,11, 2, 0, 1, 6, 0, 6, 4, 6, 1,10,-1,-1,-1,-1],
    [6, 4, 1, 6, 1,10, 4, 8, 1, 2, 1,11, 8,11, 1,-1],
    [9, 6, 4, 9, 3, 6, 9, 1, 3,11, 6, 3,-1,-1,-1,-1],
    [8,11, 1, 8, 1, 0,11, 6, 1, 9, 1, 4, 6, 4, 1,-1],
    [3,11, 6, 3, 6, 0, 0, 6, 4,-1,-1,-1,-1,-1,-1,-1],
    [6, 4, 8,11, 6, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,10, 6, 7, 8,10, 8, 9,10,-1,-1,-1,-1,-1,-1,-1],
    [0, 7, 3, 0,10, 7, 0, 9,10, 6, 7,10,-1,-1,-1,-1],
    [10, 6, 7, 1,10, 7, 1, 7, 8, 1, 8, 0,-1,-1,-1,-1],
    [10, 6, 7,10, 7, 1, 1, 7, 3,-1,-1,-1,-1,-1,-1,-1],
    [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7,-1,-1,-1,-1],
    [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9,-1],
    [7, 8, 0, 7, 0, 6, 6, 0, 2,-1,-1,-1,-1,-1,-1,-1],
    [7, 3, 2, 6, 7, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2, 3,11,10, 6, 8,10, 8, 9, 8, 6, 7,-1,-1,-1,-1],
    [2, 0, 7, 2, 7,11, 0, 9, 7, 6, 7,10, 9,10, 7,-1],
    [1, 8, 0, 1, 7, 8, 1,10, 7, 6, 7,10, 2, 3,11,-1],
    [11, 2, 1,11, 1, 7,10, 6, 1, 6, 7, 1,-1,-1,-1,-1],
    [8, 9, 1, 8, 1, 3, 9, 6, 1,11, 1, 7, 6, 7, 1,-1],
    [10, 1, 6, 6, 1, 7, 1, 0, 7, 7, 0,11, 0, 9,11,-1],  // was ambiguous; standard value
    [0, 3,11, 0,11, 6, 0, 6, 9, 6,11, 7,-1,-1,-1,-1],   // fixed
    [7,11, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7, 6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3, 0, 8,11, 7, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 1, 9,11, 7, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8, 1, 9, 8, 3, 1,11, 7, 6,-1,-1,-1,-1,-1,-1,-1],
    [10, 1, 2, 6,11, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1, 2,10, 3, 0, 8, 6,11, 7,-1,-1,-1,-1,-1,-1,-1],
    [2, 9, 0, 2,10, 9, 6,11, 7,-1,-1,-1,-1,-1,-1,-1],
    [6,11, 7, 2,10, 3,10, 8, 3,10, 9, 8,-1,-1,-1,-1],
    [7, 2, 3, 6, 2, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7, 0, 8, 7, 6, 0, 6, 2, 0,-1,-1,-1,-1,-1,-1,-1],
    [2, 7, 6, 2, 3, 7, 0, 1, 9,-1,-1,-1,-1,-1,-1,-1],
    [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6,-1,-1,-1,-1],
    [10, 7, 6,10, 1, 7, 1, 3, 7,-1,-1,-1,-1,-1,-1,-1],
    [10, 7, 6, 1, 7,10, 1, 8, 7, 1, 0, 8,-1,-1,-1,-1],
    [0, 3, 7, 0, 7,10, 0,10, 9, 6,10, 7,-1,-1,-1,-1],
    [7, 6,10, 7,10, 8, 8,10, 9,-1,-1,-1,-1,-1,-1,-1],
    [6, 8, 4,11, 8, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3, 6,11, 3, 0, 6, 0, 4, 6,-1,-1,-1,-1,-1,-1,-1],
    [8, 6,11, 8, 4, 6, 9, 0, 1,-1,-1,-1,-1,-1,-1,-1],
    [9, 4, 6, 9, 6, 3, 9, 3, 1,11, 3, 6,-1,-1,-1,-1],
    [6, 8, 4, 6,11, 8, 2,10, 1,-1,-1,-1,-1,-1,-1,-1],
    [1, 2,10, 3, 0,11, 0, 6,11, 0, 4, 6,-1,-1,-1,-1],
    [4,11, 8, 4, 6,11, 0, 2, 9, 2,10, 9,-1,-1,-1,-1],
    [10, 9, 3,10, 3, 2, 9, 4, 3,11, 3, 6, 4, 6, 3,-1],
    [8, 2, 3, 8, 4, 2, 4, 6, 2,-1,-1,-1,-1,-1,-1,-1],
    [0, 4, 2, 4, 6, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8,-1,-1,-1,-1],
    [1, 9, 4, 1, 4, 2, 2, 4, 6,-1,-1,-1,-1,-1,-1,-1],
    [8, 1, 3, 8, 6, 1, 8, 4, 6, 6,10, 1,-1,-1,-1,-1],
    [10, 1, 0,10, 0, 6, 6, 0, 4,-1,-1,-1,-1,-1,-1,-1],
    [4, 6, 3, 4, 3, 8, 6,10, 3, 0, 3, 9,10, 9, 3,-1],
    [10, 9, 4, 6,10, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4, 9, 5, 7, 6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 8, 3, 4, 9, 5,11, 7, 6,-1,-1,-1,-1,-1,-1,-1],
    [5, 0, 1, 5, 4, 0, 7, 6,11,-1,-1,-1,-1,-1,-1,-1],
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5,-1,-1,-1,-1],
    [9, 5, 4,10, 1, 2, 7, 6,11,-1,-1,-1,-1,-1,-1,-1],
    [6,11, 7, 1, 2,10, 0, 8, 3, 4, 9, 5,-1,-1,-1,-1],
    [7, 6,11, 5, 4,10, 4, 2,10, 4, 0, 2,-1,-1,-1,-1],
    [3, 4, 8, 3, 5, 4, 3, 2, 5,10, 5, 2,11, 7, 6,-1],
    [7, 2, 3, 7, 6, 2, 5, 4, 9,-1,-1,-1,-1,-1,-1,-1],
    [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7,-1,-1,-1,-1],
    [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0,-1,-1,-1,-1],
    [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8,-1],
    [9, 5, 4,10, 1, 6, 1, 7, 6, 1, 3, 7,-1,-1,-1,-1],
    [1, 6,10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4,-1],
    [4, 0,10, 4,10, 5, 0, 3,10, 6,10, 7, 3, 7,10,-1],
    [7, 6,10, 7,10, 8, 5, 4,10, 4, 8,10,-1,-1,-1,-1],
    [6, 9, 5, 6,11, 9,11, 8, 9,-1,-1,-1,-1,-1,-1,-1],
    [3, 6,11, 0, 6, 3, 0, 5, 6, 0, 9, 5,-1,-1,-1,-1],
    [0,11, 8, 0, 5,11, 0, 1, 5, 5, 6,11,-1,-1,-1,-1],
    [6,11, 3, 6, 3, 5, 5, 3, 1,-1,-1,-1,-1,-1,-1,-1],
    [1, 2,10, 9, 5,11, 9,11, 8,11, 5, 6,-1,-1,-1,-1],
    [0,11, 3, 0, 6,11, 0, 9, 6, 5, 6, 9, 1, 2,10,-1],
    [11, 8, 5,11, 5, 6, 8, 0, 5,10, 5, 2, 0, 2, 5,-1],
    [6,11, 3, 6, 3, 5, 2,10, 3,10, 5, 3,-1,-1,-1,-1],
    [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2,-1,-1,-1,-1],
    [9, 5, 6, 9, 6, 0, 0, 6, 2,-1,-1,-1,-1,-1,-1,-1],
    [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8,-1],
    [1, 5, 6, 2, 1, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1, 3, 6, 1, 6,10, 3, 8, 6, 5, 6, 9, 8, 9, 6,-1],
    [10, 1, 0,10, 0, 6, 9, 5, 0, 5, 6, 0,-1,-1,-1,-1],
    [0, 3, 8, 5, 6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10, 5, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 5,10, 7, 5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 5,10,11, 7, 5, 8, 3, 0,-1,-1,-1,-1,-1,-1,-1],
    [5,11, 7, 5,10,11, 1, 9, 0,-1,-1,-1,-1,-1,-1,-1],
    [10, 7, 5,10,11, 7, 9, 8, 1, 8, 3, 1,-1,-1,-1,-1],
    [11, 1, 2,11, 7, 1, 7, 5, 1,-1,-1,-1,-1,-1,-1,-1],
    [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2,11,-1,-1,-1,-1],
    [9, 7, 5, 9, 2, 7, 9, 0, 2, 2,11, 7,-1,-1,-1,-1],
    [7, 5, 2, 7, 2,11, 5, 9, 2, 3, 2, 8, 9, 8, 2,-1],
    [2, 5,10, 2, 3, 5, 3, 7, 5,-1,-1,-1,-1,-1,-1,-1],
    [8, 2, 0, 8, 5, 2, 8, 7, 5,10, 2, 5,-1,-1,-1,-1],
    [9, 0, 1, 5,10, 3, 5, 3, 7, 3,10, 2,-1,-1,-1,-1],
    [9, 8, 2, 9, 2, 1, 8, 7, 2,10, 2, 5, 7, 5, 2,-1],
    [1, 3, 5, 3, 7, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 8, 7, 0, 7, 1, 1, 7, 5,-1,-1,-1,-1,-1,-1,-1],
    [9, 0, 3, 9, 3, 5, 5, 3, 7,-1,-1,-1,-1,-1,-1,-1],
    [9, 8, 7, 5, 9, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [5, 8, 4, 5,10, 8,10,11, 8,-1,-1,-1,-1,-1,-1,-1],
    [5, 0, 4, 5,11, 0, 5,10,11,11, 3, 0,-1,-1,-1,-1],
    [0, 1, 9, 8, 4,10, 8,10,11,10, 4, 5,-1,-1,-1,-1],
    [10,11, 4,10, 4, 5,11, 3, 4, 9, 4, 1, 3, 1, 4,-1],
    [2, 5, 1, 2, 8, 5, 2,11, 8, 4, 5, 8,-1,-1,-1,-1],
    [0, 4,11, 0,11, 3, 4, 5,11, 2,11, 1, 5, 1,11,-1],
    [0, 2, 5, 0, 5, 9, 2,11, 5, 4, 5, 8,11, 8, 5,-1],
    [9, 4, 5, 2,11, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2, 5,10, 3, 5, 2, 3, 4, 5, 3, 8, 4,-1,-1,-1,-1],
    [5,10, 2, 5, 2, 4, 4, 2, 0,-1,-1,-1,-1,-1,-1,-1],
    [3,10, 2, 3, 5,10, 3, 8, 5, 4, 5, 8, 0, 1, 9,-1],
    [5,10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2,-1,-1,-1,-1],
    [8, 4, 5, 8, 5, 3, 3, 5, 1,-1,-1,-1,-1,-1,-1,-1],
    [0, 4, 5, 1, 0, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5,-1,-1,-1,-1],
    [9, 4, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,11, 7, 4, 9,11, 9,10,11,-1,-1,-1,-1,-1,-1,-1],
    [0, 8, 3, 4, 9, 7, 9,11, 7, 9,10,11,-1,-1,-1,-1],
    [1,10,11, 1,11, 4, 1, 4, 0, 7, 4,11,-1,-1,-1,-1],
    [3, 1, 4, 3, 4, 8, 1,10, 4, 7, 4,11,10,11, 4,-1],
    [4,11, 7, 9,11, 4, 9, 2,11, 9, 1, 2,-1,-1,-1,-1],
    [9, 7, 4, 9,11, 7, 9, 1,11, 2,11, 1, 0, 8, 3,-1],
    [11, 7, 4,11, 4, 2, 2, 4, 0,-1,-1,-1,-1,-1,-1,-1],
    [11, 7, 4,11, 4, 2, 8, 3, 4, 3, 2, 4,-1,-1,-1,-1],
    [2, 9,10, 2, 7, 9, 2, 3, 7, 7, 4, 9,-1,-1,-1,-1],
    [9,10, 7, 9, 7, 4,10, 2, 7, 8, 7, 0, 2, 0, 7,-1],
    [3, 7,10, 3,10, 2, 7, 4,10, 1,10, 0, 4, 0,10,-1],
    [1,10, 2, 8, 7, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4, 9, 1, 4, 1, 7, 7, 1, 3,-1,-1,-1,-1,-1,-1,-1],
    [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1,-1,-1,-1,-1],
    [4, 0, 3, 7, 4, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4, 8, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,10, 8,10,11, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3, 0, 9, 3, 9,11,11, 9,10,-1,-1,-1,-1,-1,-1,-1],
    [0, 1,10, 0,10, 8, 8,10,11,-1,-1,-1,-1,-1,-1,-1],
    [3, 1,10,11, 3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1, 2,11, 1,11, 9, 9,11, 8,-1,-1,-1,-1,-1,-1,-1],
    [3, 0, 9, 3, 9,11, 1, 2, 9, 2,11, 9,-1,-1,-1,-1],
    [0, 2,11, 8, 0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3, 2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2, 3, 8, 2, 8,10,10, 8, 9,-1,-1,-1,-1,-1,-1,-1],
    [9,10, 2, 0, 9, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2, 3, 8, 2, 8,10, 0, 1, 8, 1,10, 8,-1,-1,-1,-1],
    [1,10, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1, 3, 8, 9, 1, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 9, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 3, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
];

// ─── Edge connectivity ────────────────────────────────────────────────────────
//
// Each edge is defined by its two corner indices.
// Standard Lorensen & Cline numbering.
const EDGE_CORNERS: [(usize, usize); 12] = [
    (0, 1), // edge 0
    (1, 2), // edge 1
    (2, 3), // edge 2
    (3, 0), // edge 3
    (4, 5), // edge 4
    (5, 6), // edge 5
    (6, 7), // edge 6
    (7, 4), // edge 7
    (0, 4), // edge 8
    (1, 5), // edge 9
    (2, 6), // edge 10
    (3, 7), // edge 11
];

// ─── Corner offsets (ix, iy, iz) relative to voxel (x, y, z) ────────────────
// Matches cube_index bit assignment in the algorithm.
const CORNER_OFFSETS: [(usize, usize, usize); 8] = [
    (0, 0, 0), // v0 — bit 0
    (1, 0, 0), // v1 — bit 1
    (1, 1, 0), // v2 — bit 2
    (0, 1, 0), // v3 — bit 3
    (0, 0, 1), // v4 — bit 4
    (1, 0, 1), // v5 — bit 5
    (1, 1, 1), // v6 — bit 6
    (0, 1, 1), // v7 — bit 7
];

// ─── Public types ─────────────────────────────────────────────────────────────

/// Configuration for Marching Cubes.
#[derive(Debug, Clone, Copy)]
pub struct MarchingCubesConfig {
    /// Grid dimensions (number of voxels per axis).
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Voxel spacing (cell size along each axis).
    pub dx: f32,
    pub dy: f32,
    pub dz: f32,
    /// Grid origin (world position of voxel `[0,0,0]`).
    pub origin: [f32; 3],
    /// Isovalue: surface is where SDF == isovalue. Default 0.0.
    pub isovalue: f32,
}

impl Default for MarchingCubesConfig {
    fn default() -> Self {
        Self {
            nx: 10,
            ny: 10,
            nz: 10,
            dx: 1.0,
            dy: 1.0,
            dz: 1.0,
            origin: [0.0, 0.0, 0.0],
            isovalue: 0.0,
        }
    }
}

/// Output mesh from Marching Cubes.
#[derive(Debug, Clone)]
pub struct MarchingCubesResult {
    /// Vertices: flat [V × 3] row-major (x, y, z per vertex).
    pub vertices: Vec<f32>,
    /// Triangles: flat [F × 3] row-major (3 vertex indices per triangle).
    pub triangles: Vec<u32>,
    pub n_vertices: usize,
    pub n_triangles: usize,
}

// ─── Main function ────────────────────────────────────────────────────────────

/// Run Marching Cubes on a 3D SDF grid.
///
/// `sdf`: `[nx × ny × nz]` flat, x-major ordering: index = x*ny*nz + y*nz + z.
/// Values ≤ isovalue are considered "inside" (positive cube vertex).
///
/// # Errors
/// - `InvalidVoxelSize` if dx/dy/dz ≤ 0 or non-finite.
/// - `DimensionMismatch` if sdf.len() ≠ nx*ny*nz.
/// - `EmptyPointCloud` if nx < 2 or ny < 2 or nz < 2 (need at least 1 voxel).
/// - `NanEncountered` if sdf contains NaN.
pub fn marching_cubes(sdf: &[f32], cfg: &MarchingCubesConfig) -> Geom3dResult<MarchingCubesResult> {
    // ── Validate ─────────────────────────────────────────────────────────────
    if cfg.nx < 2 || cfg.ny < 2 || cfg.nz < 2 {
        return Err(Geom3dError::EmptyPointCloud);
    }
    if !cfg.dx.is_finite() || cfg.dx <= 0.0 {
        return Err(Geom3dError::InvalidVoxelSize { voxel_size: cfg.dx });
    }
    if !cfg.dy.is_finite() || cfg.dy <= 0.0 {
        return Err(Geom3dError::InvalidVoxelSize { voxel_size: cfg.dy });
    }
    if !cfg.dz.is_finite() || cfg.dz <= 0.0 {
        return Err(Geom3dError::InvalidVoxelSize { voxel_size: cfg.dz });
    }
    let expected = cfg.nx * cfg.ny * cfg.nz;
    if sdf.len() != expected {
        return Err(Geom3dError::DimensionMismatch {
            expected,
            got: sdf.len(),
        });
    }
    for &v in sdf {
        if v.is_nan() {
            return Err(Geom3dError::NanEncountered {
                location: "marching_cubes::sdf",
            });
        }
    }

    let nx = cfg.nx;
    let ny = cfg.ny;
    let nz = cfg.nz;
    let dx = cfg.dx;
    let dy = cfg.dy;
    let dz = cfg.dz;
    let ox = cfg.origin[0];
    let oy = cfg.origin[1];
    let oz = cfg.origin[2];
    let iso = cfg.isovalue;

    // Helper: SDF index (x-major)
    let idx = |x: usize, y: usize, z: usize| -> usize { x * ny * nz + y * nz + z };

    // Pre-computed world-space corner positions for the 8 voxel corners
    // will be computed per-voxel.

    let mut vertices: Vec<f32> = Vec::new();
    let mut triangles: Vec<u32> = Vec::new();

    // ── Per-voxel marching ───────────────────────────────────────────────────
    for x in 0..(nx - 1) {
        for y in 0..(ny - 1) {
            for z in 0..(nz - 1) {
                // Sample SDF at 8 corners
                let vals: [f32; 8] = [
                    sdf[idx(x, y, z)],             // v0
                    sdf[idx(x + 1, y, z)],         // v1
                    sdf[idx(x + 1, y + 1, z)],     // v2
                    sdf[idx(x, y + 1, z)],         // v3
                    sdf[idx(x, y, z + 1)],         // v4
                    sdf[idx(x + 1, y, z + 1)],     // v5
                    sdf[idx(x + 1, y + 1, z + 1)], // v6
                    sdf[idx(x, y + 1, z + 1)],     // v7
                ];

                // Build cube index
                let mut cube_index: usize = 0;
                for (bit, &v) in vals.iter().enumerate() {
                    if v <= iso {
                        cube_index |= 1 << bit;
                    }
                }

                let edge_mask = EDGE_TABLE[cube_index];
                if edge_mask == 0 {
                    continue;
                }

                // World positions of the 8 corners
                let world_pos: [[f32; 3]; 8] = {
                    let mut pos = [[0.0_f32; 3]; 8];
                    for (ci, &(ox_off, oy_off, oz_off)) in CORNER_OFFSETS.iter().enumerate() {
                        pos[ci][0] = ox + (x + ox_off) as f32 * dx;
                        pos[ci][1] = oy + (y + oy_off) as f32 * dy;
                        pos[ci][2] = oz + (z + oz_off) as f32 * dz;
                    }
                    pos
                };

                // Compute interpolated vertices on active edges
                let mut edge_verts = [[0.0_f32; 3]; 12];
                for edge_id in 0..12_u16 {
                    if edge_mask & (1 << edge_id) == 0 {
                        continue;
                    }
                    let (ca, cb) = EDGE_CORNERS[edge_id as usize];
                    let va = vals[ca];
                    let vb = vals[cb];
                    let denom = vb - va;
                    let t = if denom.abs() < 1e-10 {
                        0.5
                    } else {
                        ((iso - va) / denom).clamp(0.0, 1.0)
                    };
                    let pa = &world_pos[ca];
                    let pb = &world_pos[cb];
                    edge_verts[edge_id as usize] = [
                        pa[0] + t * (pb[0] - pa[0]),
                        pa[1] + t * (pb[1] - pa[1]),
                        pa[2] + t * (pb[2] - pa[2]),
                    ];
                }

                // Emit triangles from TRI_TABLE
                let tri_row = &TRI_TABLE[cube_index];
                let mut ti = 0;
                while ti + 2 < 16 {
                    let e0 = tri_row[ti];
                    if e0 < 0 {
                        break;
                    }
                    let e1 = tri_row[ti + 1];
                    let e2 = tri_row[ti + 2];
                    if e1 < 0 || e2 < 0 {
                        break;
                    }

                    let v0 = vertices.len() as u32 / 3;
                    // Push 3 vertices
                    for &ev in &[e0 as usize, e1 as usize, e2 as usize] {
                        vertices.push(edge_verts[ev][0]);
                        vertices.push(edge_verts[ev][1]);
                        vertices.push(edge_verts[ev][2]);
                    }
                    // Push triangle (v0, v0+1, v0+2)
                    triangles.push(v0);
                    triangles.push(v0 + 1);
                    triangles.push(v0 + 2);

                    ti += 3;
                }
            }
        }
    }

    let n_vertices = vertices.len() / 3;
    let n_triangles = triangles.len() / 3;

    Ok(MarchingCubesResult {
        vertices,
        triangles,
        n_vertices,
        n_triangles,
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a sphere SDF on an nx×ny×nz grid.
    fn sphere_sdf(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f32,
        dy: f32,
        dz: f32,
        origin: [f32; 3],
        radius: f32,
    ) -> Vec<f32> {
        let cx = origin[0] + (nx as f32 - 1.0) * dx * 0.5;
        let cy = origin[1] + (ny as f32 - 1.0) * dy * 0.5;
        let cz = origin[2] + (nz as f32 - 1.0) * dz * 0.5;
        let mut sdf = vec![0.0_f32; nx * ny * nz];
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    let wx = origin[0] + x as f32 * dx - cx;
                    let wy = origin[1] + y as f32 * dy - cy;
                    let wz = origin[2] + z as f32 * dz - cz;
                    sdf[x * ny * nz + y * nz + z] = (wx * wx + wy * wy + wz * wz).sqrt() - radius;
                }
            }
        }
        sdf
    }

    fn default_cfg_with_size(nx: usize, ny: usize, nz: usize) -> MarchingCubesConfig {
        MarchingCubesConfig {
            nx,
            ny,
            nz,
            dx: 0.1,
            dy: 0.1,
            dz: 0.1,
            origin: [0.0; 3],
            isovalue: 0.0,
        }
    }

    #[test]
    fn mc_sphere_produces_triangles() {
        let nx = 20;
        let ny = 20;
        let nz = 20;
        let cfg = default_cfg_with_size(nx, ny, nz);
        let sdf = sphere_sdf(nx, ny, nz, cfg.dx, cfg.dy, cfg.dz, cfg.origin, 0.7);
        let res = marching_cubes(&sdf, &cfg).expect("marching_cubes should succeed");
        assert!(res.n_triangles > 0, "sphere SDF must yield triangles");
    }

    #[test]
    fn mc_all_positive_empty() {
        let nx = 5;
        let ny = 5;
        let nz = 5;
        let cfg = default_cfg_with_size(nx, ny, nz);
        let sdf = vec![1.0_f32; nx * ny * nz];
        let res = marching_cubes(&sdf, &cfg).expect("marching_cubes should succeed");
        assert_eq!(res.n_triangles, 0, "all-positive SDF: no surface");
    }

    #[test]
    fn mc_all_negative_empty() {
        let nx = 5;
        let ny = 5;
        let nz = 5;
        let cfg = default_cfg_with_size(nx, ny, nz);
        let sdf = vec![-1.0_f32; nx * ny * nz];
        let res = marching_cubes(&sdf, &cfg).expect("marching_cubes should succeed");
        assert_eq!(
            res.n_triangles, 0,
            "all-negative SDF (all inside): no surface"
        );
    }

    #[test]
    fn mc_vertices_multiple_of_3() {
        let nx = 15;
        let ny = 15;
        let nz = 15;
        let cfg = default_cfg_with_size(nx, ny, nz);
        let sdf = sphere_sdf(nx, ny, nz, cfg.dx, cfg.dy, cfg.dz, cfg.origin, 0.5);
        let res = marching_cubes(&sdf, &cfg).expect("marching_cubes should succeed");
        assert_eq!(
            res.n_vertices,
            res.n_triangles * 3,
            "non-deduped: n_vertices == n_triangles*3"
        );
    }

    #[test]
    fn mc_vertices_in_bounding_box() {
        let nx = 15;
        let ny = 15;
        let nz = 15;
        let cfg = MarchingCubesConfig {
            nx,
            ny,
            nz,
            dx: 0.1,
            dy: 0.1,
            dz: 0.1,
            origin: [1.0, 2.0, 3.0],
            isovalue: 0.0,
        };
        let sdf = sphere_sdf(nx, ny, nz, cfg.dx, cfg.dy, cfg.dz, cfg.origin, 0.5);
        let res = marching_cubes(&sdf, &cfg).expect("marching_cubes should succeed");

        let max_x = cfg.origin[0] + (nx - 1) as f32 * cfg.dx;
        let max_y = cfg.origin[1] + (ny - 1) as f32 * cfg.dy;
        let max_z = cfg.origin[2] + (nz - 1) as f32 * cfg.dz;

        for i in 0..res.n_vertices {
            let vx = res.vertices[i * 3];
            let vy = res.vertices[i * 3 + 1];
            let vz = res.vertices[i * 3 + 2];
            assert!(
                vx >= cfg.origin[0] - 1e-4 && vx <= max_x + 1e-4,
                "vertex x={vx} out of bounds [{}, {}]",
                cfg.origin[0],
                max_x
            );
            assert!(
                vy >= cfg.origin[1] - 1e-4 && vy <= max_y + 1e-4,
                "vertex y={vy} out of bounds [{}, {}]",
                cfg.origin[1],
                max_y
            );
            assert!(
                vz >= cfg.origin[2] - 1e-4 && vz <= max_z + 1e-4,
                "vertex z={vz} out of bounds [{}, {}]",
                cfg.origin[2],
                max_z
            );
        }
    }

    #[test]
    fn mc_sphere_vertices_near_surface() {
        let nx = 20;
        let ny = 20;
        let nz = 20;
        let radius = 0.7_f32;
        let cfg = MarchingCubesConfig {
            nx,
            ny,
            nz,
            dx: 0.1,
            dy: 0.1,
            dz: 0.1,
            origin: [0.0; 3],
            isovalue: 0.0,
        };
        let sdf = sphere_sdf(nx, ny, nz, cfg.dx, cfg.dy, cfg.dz, cfg.origin, radius);
        let res = marching_cubes(&sdf, &cfg).expect("marching_cubes should succeed");

        let cx = (nx as f32 - 1.0) * cfg.dx * 0.5;
        let cy = (ny as f32 - 1.0) * cfg.dy * 0.5;
        let cz = (nz as f32 - 1.0) * cfg.dz * 0.5;
        let tolerance = cfg.dx + cfg.dy + cfg.dz;

        for i in 0..res.n_vertices {
            let vx = res.vertices[i * 3] - cx;
            let vy = res.vertices[i * 3 + 1] - cy;
            let vz = res.vertices[i * 3 + 2] - cz;
            let dist_to_surface = ((vx * vx + vy * vy + vz * vz).sqrt() - radius).abs();
            assert!(
                dist_to_surface < tolerance,
                "vertex {i} dist_to_surface={dist_to_surface} > tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn mc_plane_produces_triangles() {
        let nx = 6;
        let ny = 6;
        let nz = 6;
        let cfg = default_cfg_with_size(nx, ny, nz);
        // Flat SDF: negative for z < nz/2, positive for z >= nz/2
        let mut sdf = vec![0.0_f32; nx * ny * nz];
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    sdf[x * ny * nz + y * nz + z] = if z < nz / 2 { -1.0 } else { 1.0 };
                }
            }
        }
        let res = marching_cubes(&sdf, &cfg).expect("marching_cubes should succeed");
        assert!(res.n_triangles > 0, "plane SDF must yield triangles");
    }

    #[test]
    fn mc_default_config() {
        let cfg = MarchingCubesConfig::default();
        assert_eq!(cfg.nx, 10);
        assert_eq!(cfg.ny, 10);
        assert_eq!(cfg.nz, 10);
        assert_eq!(cfg.isovalue, 0.0);
    }

    #[test]
    fn mc_small_grid() {
        let nx = 3;
        let ny = 3;
        let nz = 3;
        let cfg = default_cfg_with_size(nx, ny, nz);
        let sdf = sphere_sdf(nx, ny, nz, cfg.dx, cfg.dy, cfg.dz, cfg.origin, 0.1);
        let res = marching_cubes(&sdf, &cfg).expect("marching_cubes should succeed");
        // May or may not produce triangles, but should not error.
        assert_eq!(res.n_vertices, res.n_triangles * 3);
    }

    #[test]
    fn mc_output_consistent() {
        let nx = 10;
        let ny = 10;
        let nz = 10;
        let cfg = default_cfg_with_size(nx, ny, nz);
        let sdf = sphere_sdf(nx, ny, nz, cfg.dx, cfg.dy, cfg.dz, cfg.origin, 0.4);
        let res = marching_cubes(&sdf, &cfg).expect("marching_cubes should succeed");
        assert_eq!(res.n_vertices, res.vertices.len() / 3);
        assert_eq!(res.n_triangles, res.triangles.len() / 3);
    }

    #[test]
    fn mc_err_invalid_voxel_size() {
        let cfg = MarchingCubesConfig {
            nx: 5,
            ny: 5,
            nz: 5,
            dx: 0.0,
            dy: 1.0,
            dz: 1.0,
            ..Default::default()
        };
        let sdf = vec![0.0_f32; 125];
        let result = marching_cubes(&sdf, &cfg);
        assert!(matches!(result, Err(Geom3dError::InvalidVoxelSize { .. })));
    }

    #[test]
    fn mc_err_dim_mismatch() {
        let cfg = MarchingCubesConfig {
            nx: 5,
            ny: 5,
            nz: 5,
            ..Default::default()
        };
        let sdf = vec![0.0_f32; 100]; // wrong: 100 != 125
        let result = marching_cubes(&sdf, &cfg);
        assert!(matches!(result, Err(Geom3dError::DimensionMismatch { .. })));
    }

    #[test]
    fn mc_err_too_small() {
        let cfg = MarchingCubesConfig {
            nx: 1,
            ny: 5,
            nz: 5,
            ..Default::default()
        };
        let sdf = vec![0.0_f32; 25];
        let result = marching_cubes(&sdf, &cfg);
        assert!(matches!(result, Err(Geom3dError::EmptyPointCloud)));
    }

    #[test]
    fn mc_err_nan() {
        let nx = 4;
        let ny = 4;
        let nz = 4;
        let cfg = default_cfg_with_size(nx, ny, nz);
        let mut sdf = vec![0.0_f32; nx * ny * nz];
        sdf[10] = f32::NAN;
        let result = marching_cubes(&sdf, &cfg);
        assert!(matches!(result, Err(Geom3dError::NanEncountered { .. })));
    }

    #[test]
    fn mc_custom_isovalue() {
        let nx = 12;
        let ny = 12;
        let nz = 12;
        let mut cfg0 = default_cfg_with_size(nx, ny, nz);
        cfg0.isovalue = 0.0;
        let mut cfg5 = default_cfg_with_size(nx, ny, nz);
        cfg5.isovalue = 0.3;

        let sdf = sphere_sdf(nx, ny, nz, 0.1, 0.1, 0.1, [0.0; 3], 0.5);
        let res0 = marching_cubes(&sdf, &cfg0).expect("marching_cubes should succeed");
        let res5 = marching_cubes(&sdf, &cfg5).expect("marching_cubes should succeed");
        // Different isovalues extract different surfaces; counts should differ
        // (or both may have triangles but different counts).
        // At minimum both should run without error and isovalue=0.3 gives a smaller sphere.
        assert!(res0.n_triangles > 0 || res5.n_triangles > 0);
    }

    #[test]
    fn mc_origin_offset() {
        let nx = 10;
        let ny = 10;
        let nz = 10;
        let cfg_origin = MarchingCubesConfig {
            nx,
            ny,
            nz,
            dx: 0.1,
            dy: 0.1,
            dz: 0.1,
            origin: [10.0, 20.0, 30.0],
            isovalue: 0.0,
        };
        let cfg_zero = MarchingCubesConfig {
            nx,
            ny,
            nz,
            dx: 0.1,
            dy: 0.1,
            dz: 0.1,
            origin: [0.0, 0.0, 0.0],
            isovalue: 0.0,
        };
        let sdf = sphere_sdf(nx, ny, nz, 0.1, 0.1, 0.1, [0.0; 3], 0.3);
        let res_off = marching_cubes(&sdf, &cfg_origin).expect("marching_cubes should succeed");
        let res_zero = marching_cubes(&sdf, &cfg_zero).expect("marching_cubes should succeed");

        if res_off.n_vertices > 0 && res_zero.n_vertices > 0 {
            // All vertices in offset mesh should be shifted by (10, 20, 30)
            let mean_x_off: f32 =
                res_off.vertices.iter().step_by(3).sum::<f32>() / res_off.n_vertices as f32;
            let mean_x_zero: f32 =
                res_zero.vertices.iter().step_by(3).sum::<f32>() / res_zero.n_vertices as f32;
            assert!(
                (mean_x_off - mean_x_zero - 10.0).abs() < 0.5,
                "origin offset should shift vertices: mean_x_off={mean_x_off} mean_x_zero={mean_x_zero}"
            );
        }
    }

    #[test]
    fn mc_single_voxel_corner() {
        // 2×2×2 grid. v0 is inside (≤0), all others outside (>0).
        // cube_index = 0b00000001 = 1.
        // EDGE_TABLE[1] = 0x109 -> edges 0, 3, 8 active.
        // TRI_TABLE[1] = [0, 8, 3, -1, ...] -> exactly 1 triangle.
        let cfg = MarchingCubesConfig {
            nx: 2,
            ny: 2,
            nz: 2,
            dx: 1.0,
            dy: 1.0,
            dz: 1.0,
            origin: [0.0; 3],
            isovalue: 0.0,
        };
        // SDF: v0=(x=0,y=0,z=0)=-1; all others=+1
        let mut sdf = vec![1.0_f32; 8];
        sdf[0] = -1.0; // index 0*2*2 + 0*2 + 0 = 0
        let res = marching_cubes(&sdf, &cfg).expect("marching_cubes should succeed");
        assert_eq!(
            res.n_triangles, 1,
            "single corner inside -> exactly 1 triangle"
        );
    }

    #[test]
    fn mc_edge_interpolation() {
        // 2×2×2 grid. v0=-1, v1=+1, all others=+1.
        // cube_index = 0b00000001 = 1 (only v0 inside).
        // Edge 0 connects v0 and v1. t = (0 - (-1)) / (1 - (-1)) = 0.5.
        // Interpolated vertex on edge 0 should be at midpoint between v0(0,0,0) and v1(1,0,0)
        // i.e. (0.5, 0, 0).
        let cfg = MarchingCubesConfig {
            nx: 2,
            ny: 2,
            nz: 2,
            dx: 1.0,
            dy: 1.0,
            dz: 1.0,
            origin: [0.0; 3],
            isovalue: 0.0,
        };
        let mut sdf = vec![1.0_f32; 8];
        sdf[0] = -1.0; // v0 inside
        let res = marching_cubes(&sdf, &cfg).expect("marching_cubes should succeed");
        assert_eq!(res.n_triangles, 1);

        // Find the vertex on edge 0 (x between 0 and 1, y=0, z=0)
        let mut found_midpoint = false;
        for vi in 0..res.n_vertices {
            let vx = res.vertices[vi * 3];
            let vy = res.vertices[vi * 3 + 1];
            let vz = res.vertices[vi * 3 + 2];
            if (vx - 0.5).abs() < 1e-5 && vy.abs() < 1e-5 && vz.abs() < 1e-5 {
                found_midpoint = true;
            }
        }
        assert!(
            found_midpoint,
            "edge 0 vertex must be at midpoint (0.5, 0, 0)"
        );
    }
}
