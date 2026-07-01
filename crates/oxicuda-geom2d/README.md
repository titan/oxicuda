# oxicuda-geom2d

2D computational geometry -- convex hulls, Delaunay/Voronoi, sweep-line, clipping, and spatial indices in pure Rust.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-geom2d` is a self-contained 2D computational geometry library
covering the canonical primitives, predicates, and algorithms that appear
across CAD, GIS, mesh generation, motion planning, and graphics. It is
designed as a dependable building block for higher-level OxiCUDA crates
that need robust planar geometry without pulling in a C/C++ toolchain.

GPU PTX kernels for batched primitives are generated and dispatched entirely
from Rust via the OxiCUDA driver stack. There is no C/CUDA toolchain at
build time and no external linear-algebra dependency beyond `thiserror`;
random sampling uses the workspace `LcgRng` (MMIX LCG with the bit-32
boolean trick).

Algorithm coverage spans all five major convex-hull algorithms (Graham
scan, Andrew's monotone chain, QuickHull, Jarvis march, Chan), both major
Delaunay triangulation families (ear clipping, Bowyer-Watson incremental,
constrained Delaunay), Voronoi diagrams (Fortune sweep-line and Delaunay
dual), the four standard polygon clipping algorithms (Sutherland-Hodgman,
Weiler-Atherton, Cohen-Sutherland, Liang-Barsky), Bentley-Ottmann
segment-intersection sweep, Welzl's smallest enclosing circle, rotating
calipers, slab and trapezoidal-map point location, and three spatial
indices (KD-tree, R-tree with STR bulk load, quadtree).

## Modules

| Module | Description |
|--------|-------------|
| `primitives` | Point, Vector, Line, Segment, Ray, Circle, Aabb, Polygon |
| `predicate` | Orientation, in-circle, dot/cross, robust signs |
| `intersection` | Segment-segment, line-line, segment-polygon, circle intersections |
| `containment` | Point-in-polygon (winding / ray-cast), in convex polygon, in circle |
| `hull` | Graham scan, Andrew monotone chain, QuickHull, Jarvis march, Chan |
| `triangulation` | Ear clipping, Bowyer-Watson Delaunay, constrained Delaunay |
| `voronoi` | Fortune sweep-line, Voronoi from Delaunay dual |
| `clipping` | Sutherland-Hodgman, Weiler-Atherton, Cohen-Sutherland, Liang-Barsky |
| `polygon_ops` | Shoelace area, centroid, perimeter, convexity, offset, Minkowski sum |
| `closest_pair` | Brute force O(n^2), divide-and-conquer O(n log n) |
| `enclosing` | Welzl smallest circle, AABB, rotating calipers diameter / width |
| `sweepline` | Bentley-Ottmann segment intersection sweep |
| `point_location` | Slab method, trapezoidal map |
| `index` | 2D KD-tree, R-tree (STR bulk load), quadtree |
| `metrics` | Euclidean, Manhattan, Chebyshev, angle, signed area |
| `handle` | `Geom2dHandle`, `SmVersion`, `LcgRng` (MMIX LCG) |
| `error` | `Geom2dError` / `Geom2dResult` |
| `ptx_kernels` | GPU PTX kernel templates per SM target |

## Quick Start

```rust,no_run
use oxicuda_geom2d::hull::graham_scan;
use oxicuda_geom2d::primitives::point::Point;

fn main() -> oxicuda_geom2d::Geom2dResult<()> {
    // Build a small point set and compute the convex hull (counter-clockwise).
    let pts = vec![
        Point::new(0.0, 0.0),
        Point::new(1.0, 0.0),
        Point::new(1.0, 1.0),
        Point::new(0.0, 1.0),
        Point::new(0.5, 0.5), // interior point, will be discarded
    ];
    let hull = graham_scan(&pts)?;
    assert_eq!(hull.len(), 4);
    Ok(())
}
```

## Status

**Alpha** -- 10,028 SLoC, 301 passing tests. API may evolve before v1.0.

## License

Apache-2.0
