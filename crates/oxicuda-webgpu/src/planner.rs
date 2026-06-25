//! Host-side dispatch and workgroup planning for WebGPU compute.
//!
//! Everything in this module is pure arithmetic over adapter limits and
//! problem shapes — no `wgpu` device, queue, or adapter is required, so the
//! logic is fully unit-testable on CPU.  The planners answer questions a
//! WebGPU backend must resolve *before* recording a compute pass:
//!
//! * how large a workgroup may be, given the adapter's
//!   `max_compute_invocations_per_workgroup` / per-axis caps (auto-tuning);
//! * how many workgroups to dispatch for a 1-D / 2-D / 3-D problem while
//!   staying below WebGPU's 65 535-per-axis limit;
//! * how a linear problem must be folded into a 2-D grid when it would
//!   otherwise exceed that per-axis cap;
//! * texture upload row pitch under WebGPU's 256-byte `bytes_per_row`
//!   alignment rule.
//!
//! These mirror the constants in [`crate::shader`] (e.g. the reductions use a
//! 256-thread workgroup and a 2-D dispatch decode `wgid.y * grid_x + wgid.x`).

/// WebGPU's hard cap on workgroup count per dispatch axis.
///
/// The spec guarantees `maxComputeWorkgroupsPerDimension >= 65535`; we treat
/// it as the conservative portable maximum.
pub const MAX_WORKGROUPS_PER_DIM: u32 = 65_535;

/// WebGPU's required alignment (in bytes) for the `bytes_per_row` field of a
/// texture copy (`COPY_BYTES_PER_ROW_ALIGNMENT`).
pub const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

/// A conservative, portable subset of the WebGPU compute limits a planner
/// needs.  Field names follow the WebGPU / `wgpu` limit names.
///
/// [`Limits::portable_default`] reflects the **guaranteed** lower bounds every
/// conformant WebGPU adapter must expose, so planning against it yields
/// dispatches that run everywhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// `maxComputeWorkgroupSizeX`.
    pub max_workgroup_size_x: u32,
    /// `maxComputeWorkgroupSizeY`.
    pub max_workgroup_size_y: u32,
    /// `maxComputeWorkgroupSizeZ`.
    pub max_workgroup_size_z: u32,
    /// `maxComputeInvocationsPerWorkgroup` — product of the three axes must
    /// not exceed this.
    pub max_invocations_per_workgroup: u32,
    /// `maxComputeWorkgroupsPerDimension`.
    pub max_workgroups_per_dim: u32,
}

impl Limits {
    /// The WebGPU **guaranteed** lower bounds (Chrome / Firefox / Safari all
    /// expose at least these).
    ///
    /// `256` invocations, `(256, 256, 64)` per-axis sizes, `65535` workgroups
    /// per dimension.
    #[must_use]
    pub fn portable_default() -> Self {
        Self {
            max_workgroup_size_x: 256,
            max_workgroup_size_y: 256,
            max_workgroup_size_z: 64,
            max_invocations_per_workgroup: 256,
            max_workgroups_per_dim: MAX_WORKGROUPS_PER_DIM,
        }
    }

    /// A higher-end desktop profile (e.g. discrete NVIDIA / AMD via Vulkan)
    /// exposing `1024` invocations per workgroup.
    #[must_use]
    pub fn desktop_default() -> Self {
        Self {
            max_workgroup_size_x: 1024,
            max_workgroup_size_y: 1024,
            max_workgroup_size_z: 64,
            max_invocations_per_workgroup: 1024,
            max_workgroups_per_dim: MAX_WORKGROUPS_PER_DIM,
        }
    }
}

/// Integer ceil-division `ceil(a / b)`.  Returns `0` when `b == 0`.
#[inline]
#[must_use]
pub fn div_ceil_u32(a: u32, b: u32) -> u32 {
    match a.checked_div(b) {
        Some(q) => q + u32::from(a % b != 0),
        None => 0,
    }
}

/// Round `value` **up** to the next multiple of `align`.
///
/// `align` must be non-zero; when it is, the result is the smallest multiple
/// of `align` that is `>= value`.  Returns `value` unchanged if `align == 0`.
#[inline]
#[must_use]
pub fn align_up_u32(value: u32, align: u32) -> u32 {
    if align == 0 {
        return value;
    }
    div_ceil_u32(value, align).saturating_mul(align)
}

/// A planned 1-D workgroup size: a single `@workgroup_size(x)` value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Workgroup1D {
    /// Thread count along X (`@workgroup_size(x)`).
    pub x: u32,
}

/// A planned 2-D workgroup size for tiled kernels (`@workgroup_size(x, y)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Workgroup2D {
    /// Thread count along X.
    pub x: u32,
    /// Thread count along Y.
    pub y: u32,
}

impl Workgroup2D {
    /// Total invocations (`x * y`).
    #[must_use]
    pub fn total(self) -> u32 {
        self.x.saturating_mul(self.y)
    }
}

/// Auto-tune a 1-D workgroup size: the largest power of two `<= preferred`
/// that fits both `max_workgroup_size_x` and `max_invocations_per_workgroup`.
///
/// Power-of-two sizing keeps tree reductions (which halve a stride each step)
/// exact.  A non-power-of-two `preferred` is rounded **down** to a power of
/// two first.  The result is always `>= 1`.
#[must_use]
pub fn plan_workgroup_1d(limits: &Limits, preferred: u32) -> Workgroup1D {
    let ceiling = limits
        .max_workgroup_size_x
        .min(limits.max_invocations_per_workgroup);
    let want = preferred.max(1).min(ceiling.max(1));
    Workgroup1D {
        x: floor_pow2(want).max(1),
    }
}

/// Auto-tune a square 2-D workgroup tile for GEMM-style kernels.
///
/// Returns the largest power-of-two tile `t` such that:
/// * `t <= preferred_tile`,
/// * `t <= max_workgroup_size_x` and `t <= max_workgroup_size_y`, and
/// * `t * t <= max_invocations_per_workgroup`.
///
/// For the portable default (`256` invocations) this caps the tile at `16`
/// (`16 * 16 == 256`); a desktop profile (`1024`) allows `32`.
#[must_use]
pub fn plan_workgroup_square(limits: &Limits, preferred_tile: u32) -> Workgroup2D {
    let axis_cap = limits.max_workgroup_size_x.min(limits.max_workgroup_size_y);
    let mut t = floor_pow2(preferred_tile.max(1)).max(1);
    t = t.min(floor_pow2(axis_cap.max(1)).max(1));
    // Shrink until t*t fits the invocation budget.
    while t > 1 && t.saturating_mul(t) > limits.max_invocations_per_workgroup {
        t >>= 1;
    }
    Workgroup2D { x: t, y: t }
}

/// Largest power of two `<= v` (for `v >= 1`).  `floor_pow2(0) == 0`.
#[inline]
#[must_use]
pub fn floor_pow2(v: u32) -> u32 {
    if v == 0 {
        0
    } else {
        1u32 << (31 - v.leading_zeros())
    }
}

/// A concrete dispatch grid: the `(x, y, z)` argument to
/// `dispatch_workgroups`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchGrid {
    /// Workgroup count along X.
    pub x: u32,
    /// Workgroup count along Y.
    pub y: u32,
    /// Workgroup count along Z.
    pub z: u32,
}

impl DispatchGrid {
    /// Total workgroup count (`x * y * z`).
    #[must_use]
    pub fn total(self) -> u64 {
        u64::from(self.x) * u64::from(self.y) * u64::from(self.z)
    }
}

/// Plan a 1-D dispatch covering `total_threads` items with `workgroup_x`
/// threads each, automatically folding into a 2-D grid when the required
/// workgroup count would exceed the per-axis limit.
///
/// The companion shader must decode its linear slot as
/// `wgid.y * grid_x + wgid.x` (exactly as [`crate::shader::reduction_nd_wgsl`]
/// does) and early-return when the slot is out of range.
///
/// `grid_x` (the second return value) is the fold width to embed as the
/// `grid_x` uniform; for a non-folded dispatch it equals the X workgroup
/// count and the decode is still correct (`wgid.y == 0`).
///
/// Returns an error string if `total_threads` is so large that even a square
/// 2-D grid would exceed the per-axis cap on both axes.
#[allow(clippy::missing_errors_doc)]
pub fn plan_dispatch_1d(
    limits: &Limits,
    total_threads: u64,
    workgroup_x: u32,
) -> Result<(DispatchGrid, u32), String> {
    let wg = workgroup_x.max(1);
    let groups = div_ceil_u64(total_threads, u64::from(wg));
    let max_dim = u64::from(limits.max_workgroups_per_dim.max(1));

    if groups <= max_dim {
        let gx = u32::try_from(groups).unwrap_or(limits.max_workgroups_per_dim);
        return Ok((DispatchGrid { x: gx, y: 1, z: 1 }, gx));
    }

    // Fold into a 2-D grid: grid_x columns, ceil(groups / grid_x) rows.
    let grid_x = max_dim;
    let grid_y = div_ceil_u64(groups, grid_x);
    if grid_y > max_dim {
        return Err(format!(
            "dispatch of {groups} workgroups exceeds 2-D capacity ({max_dim} x {max_dim})"
        ));
    }
    let gx = u32::try_from(grid_x).unwrap_or(limits.max_workgroups_per_dim);
    let gy = u32::try_from(grid_y).unwrap_or(limits.max_workgroups_per_dim);
    Ok((DispatchGrid { x: gx, y: gy, z: 1 }, gx))
}

/// Plan a 2-D tiled dispatch for an `m x n` output covered by a
/// `tile.x * tile.y` workgroup, optionally with a `batch` Z-dimension.
///
/// Used by GEMM (`grid.z == 1`) and batched GEMM (`grid.z == batch`).  Returns
/// an error string if any axis would exceed the per-axis workgroup cap.
#[allow(clippy::missing_errors_doc)]
pub fn plan_dispatch_2d(
    limits: &Limits,
    rows_m: u32,
    cols_n: u32,
    tile: Workgroup2D,
    batch: u32,
) -> Result<DispatchGrid, String> {
    let gx = div_ceil_u32(cols_n, tile.x.max(1));
    let gy = div_ceil_u32(rows_m, tile.y.max(1));
    let gz = batch.max(1);
    let cap = limits.max_workgroups_per_dim.max(1);
    if gx > cap || gy > cap || gz > cap {
        return Err(format!(
            "2-D dispatch ({gx}, {gy}, {gz}) exceeds per-axis cap {cap}"
        ));
    }
    Ok(DispatchGrid {
        x: gx,
        y: gy,
        z: gz,
    })
}

/// Integer ceil-division for `u64`.  Returns `0` when `b == 0`.
#[inline]
#[must_use]
pub fn div_ceil_u64(a: u64, b: u64) -> u64 {
    match a.checked_div(b) {
        Some(q) => q + u64::from(a % b != 0),
        None => 0,
    }
}

/// Texture upload layout under WebGPU's 256-byte row-alignment rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureRowLayout {
    /// Unpadded bytes of actual pixel data per row (`width * bytes_per_texel`).
    pub unpadded_bytes_per_row: u32,
    /// Padded `bytes_per_row` (a multiple of 256) to pass to wgpu.
    pub padded_bytes_per_row: u32,
    /// Total bytes for the whole image (`padded_bytes_per_row * height`).
    pub total_bytes: u64,
}

/// Compute the 256-aligned row pitch for a texture upload/readback.
///
/// WebGPU requires the `bytes_per_row` of a buffer↔texture copy to be a
/// multiple of [`COPY_BYTES_PER_ROW_ALIGNMENT`] (256).  Callers must allocate
/// `padded_bytes_per_row * height` bytes of staging and copy each row into its
/// padded slot.
#[must_use]
pub fn texture_row_layout(width: u32, height: u32, bytes_per_texel: u32) -> TextureRowLayout {
    let unpadded = width.saturating_mul(bytes_per_texel);
    let padded = align_up_u32(unpadded, COPY_BYTES_PER_ROW_ALIGNMENT);
    TextureRowLayout {
        unpadded_bytes_per_row: unpadded,
        padded_bytes_per_row: padded,
        total_bytes: u64::from(padded) * u64::from(height),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn div_ceil_u32_basic() {
        assert_eq!(div_ceil_u32(0, 4), 0);
        assert_eq!(div_ceil_u32(1, 4), 1);
        assert_eq!(div_ceil_u32(4, 4), 1);
        assert_eq!(div_ceil_u32(5, 4), 2);
        assert_eq!(div_ceil_u32(8, 4), 2);
        // Division by zero yields 0 (no panic).
        assert_eq!(div_ceil_u32(10, 0), 0);
    }

    #[test]
    fn align_up_u32_to_256() {
        assert_eq!(align_up_u32(0, 256), 0);
        assert_eq!(align_up_u32(1, 256), 256);
        assert_eq!(align_up_u32(256, 256), 256);
        assert_eq!(align_up_u32(257, 256), 512);
        assert_eq!(align_up_u32(513, 256), 768);
        // align 0 is identity.
        assert_eq!(align_up_u32(123, 0), 123);
    }

    #[test]
    fn floor_pow2_values() {
        assert_eq!(floor_pow2(0), 0);
        assert_eq!(floor_pow2(1), 1);
        assert_eq!(floor_pow2(2), 2);
        assert_eq!(floor_pow2(3), 2);
        assert_eq!(floor_pow2(255), 128);
        assert_eq!(floor_pow2(256), 256);
        assert_eq!(floor_pow2(257), 256);
        assert_eq!(floor_pow2(1024), 1024);
    }

    #[test]
    fn plan_workgroup_1d_respects_invocation_cap() {
        let lim = Limits::portable_default(); // 256 invocations
        // Preferred 256 fits exactly.
        assert_eq!(plan_workgroup_1d(&lim, 256).x, 256);
        // Preferred 1024 is clamped down to 256.
        assert_eq!(plan_workgroup_1d(&lim, 1024).x, 256);
        // Non-power-of-two rounds down.
        assert_eq!(plan_workgroup_1d(&lim, 200).x, 128);
        // Zero is bumped to 1.
        assert_eq!(plan_workgroup_1d(&lim, 0).x, 1);
    }

    #[test]
    fn plan_workgroup_1d_desktop_allows_larger() {
        let lim = Limits::desktop_default(); // 1024 invocations
        assert_eq!(plan_workgroup_1d(&lim, 1024).x, 1024);
        assert_eq!(plan_workgroup_1d(&lim, 2048).x, 1024);
    }

    #[test]
    fn plan_workgroup_square_caps_at_16_on_portable() {
        let lim = Limits::portable_default(); // 256 invocations -> 16x16
        let wg = plan_workgroup_square(&lim, 32);
        assert_eq!(wg, Workgroup2D { x: 16, y: 16 });
        assert!(wg.total() <= lim.max_invocations_per_workgroup);
    }

    #[test]
    fn plan_workgroup_square_allows_32_on_desktop() {
        let lim = Limits::desktop_default(); // 1024 invocations -> 32x32
        let wg = plan_workgroup_square(&lim, 32);
        assert_eq!(wg, Workgroup2D { x: 32, y: 32 });
        assert_eq!(wg.total(), 1024);
    }

    #[test]
    fn plan_workgroup_square_respects_preferred_smaller() {
        let lim = Limits::desktop_default();
        // Prefer 8 even though 32 would fit.
        let wg = plan_workgroup_square(&lim, 8);
        assert_eq!(wg, Workgroup2D { x: 8, y: 8 });
    }

    #[test]
    fn plan_dispatch_1d_simple() {
        let lim = Limits::portable_default();
        let (grid, grid_x) = plan_dispatch_1d(&lim, 1000, 256).expect("plan");
        // ceil(1000 / 256) = 4 workgroups.
        assert_eq!(grid, DispatchGrid { x: 4, y: 1, z: 1 });
        assert_eq!(grid_x, 4);
    }

    #[test]
    fn plan_dispatch_1d_exact_multiple() {
        let lim = Limits::portable_default();
        let (grid, _) = plan_dispatch_1d(&lim, 512, 256).expect("plan");
        assert_eq!(grid.x, 2);
        assert_eq!(grid.total(), 2);
    }

    #[test]
    fn plan_dispatch_1d_folds_into_2d_when_too_wide() {
        let lim = Limits::portable_default(); // 65535 per dim
        // Need 65535 * 2 + 1 workgroups -> must fold to 2-D.
        let total_groups = u64::from(MAX_WORKGROUPS_PER_DIM) * 2 + 1;
        let total_threads = total_groups; // workgroup_x = 1
        let (grid, grid_x) = plan_dispatch_1d(&lim, total_threads, 1).expect("plan");
        assert!(grid.y > 1, "expected 2-D fold, got {grid:?}");
        assert_eq!(grid.x, MAX_WORKGROUPS_PER_DIM);
        assert_eq!(grid_x, MAX_WORKGROUPS_PER_DIM);
        // The grid must cover every workgroup.
        assert!(grid.total() >= total_groups);
        // Decode must be able to reach the last slot.
        assert!(u64::from(grid.x) * u64::from(grid.y) >= total_groups);
    }

    #[test]
    fn plan_dispatch_1d_errors_when_beyond_2d_capacity() {
        let lim = Limits::portable_default();
        // Beyond 65535 * 65535 workgroups (one thread each) cannot fit a 2-D grid.
        let beyond = u64::from(MAX_WORKGROUPS_PER_DIM) * u64::from(MAX_WORKGROUPS_PER_DIM) + 1;
        let err = plan_dispatch_1d(&lim, beyond, 1).unwrap_err();
        assert!(err.contains("exceeds 2-D capacity"));
    }

    #[test]
    fn plan_dispatch_2d_gemm() {
        let lim = Limits::portable_default();
        let tile = Workgroup2D { x: 16, y: 16 };
        // 100x200 output, no batch.
        let grid = plan_dispatch_2d(&lim, 100, 200, tile, 1).expect("plan");
        assert_eq!(grid.x, div_ceil_u32(200, 16)); // cols -> x
        assert_eq!(grid.y, div_ceil_u32(100, 16)); // rows -> y
        assert_eq!(grid.z, 1);
    }

    #[test]
    fn plan_dispatch_2d_batched() {
        let lim = Limits::portable_default();
        let tile = Workgroup2D { x: 8, y: 8 };
        let grid = plan_dispatch_2d(&lim, 32, 32, tile, 7).expect("plan");
        assert_eq!(grid.z, 7);
        assert_eq!(grid.x, 4);
        assert_eq!(grid.y, 4);
    }

    #[test]
    fn plan_dispatch_2d_errors_when_axis_overflows() {
        let lim = Limits::portable_default();
        let tile = Workgroup2D { x: 1, y: 1 };
        // cols way beyond the per-axis cap with tile 1.
        let huge = MAX_WORKGROUPS_PER_DIM + 10;
        let err = plan_dispatch_2d(&lim, 4, huge, tile, 1).unwrap_err();
        assert!(err.contains("exceeds per-axis cap"));
    }

    #[test]
    fn texture_row_layout_padding() {
        // RGBA8: 4 bytes/texel, width 100 -> 400 unpadded -> 512 padded.
        let layout = texture_row_layout(100, 50, 4);
        assert_eq!(layout.unpadded_bytes_per_row, 400);
        assert_eq!(layout.padded_bytes_per_row, 512);
        assert_eq!(layout.total_bytes, 512 * 50);
        // Padded row must be a multiple of 256.
        assert_eq!(
            layout.padded_bytes_per_row % COPY_BYTES_PER_ROW_ALIGNMENT,
            0
        );
    }

    #[test]
    fn texture_row_layout_already_aligned() {
        // width 64, 4 bytes -> 256, already aligned.
        let layout = texture_row_layout(64, 10, 4);
        assert_eq!(layout.unpadded_bytes_per_row, 256);
        assert_eq!(layout.padded_bytes_per_row, 256);
        assert_eq!(layout.total_bytes, 2560);
    }

    #[test]
    fn limits_portable_default_is_conservative() {
        let lim = Limits::portable_default();
        assert_eq!(lim.max_invocations_per_workgroup, 256);
        assert_eq!(lim.max_workgroups_per_dim, MAX_WORKGROUPS_PER_DIM);
        // The desktop profile must be at least as permissive.
        let desk = Limits::desktop_default();
        assert!(desk.max_invocations_per_workgroup >= lim.max_invocations_per_workgroup);
    }
}
