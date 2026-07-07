//! CSR5 SpMV kernel.
//!
//! Computes `y = alpha * A * x + beta * y` where `A` is in CSR5 format.
//!
//! CSR5 achieves load-balanced SpMV by dividing non-zeros into fixed-width
//! tiles (32 elements each, matching warp width). Each warp processes one
//! tile, using tile descriptors to determine row boundaries.
//!
//! The SpMV proceeds in two phases:
//! 1. **Tile phase**: Each warp computes partial sums for its tile, using
//!    warp shuffle to reduce within rows. Results are written to `y` for
//!    rows fully contained within a tile, or to the calibrator for rows
//!    that span tile boundaries.
//! 2. **Calibrate phase**: A separate kernel merges cross-tile partial
//!    sums from the calibrator into the final `y` vector.

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{SparseError, SparseResult};
use crate::format::csr5::Csr5Matrix;
use crate::handle::SparseHandle;
use crate::ptx_helpers::{
    add_float, emit_shfl_float, load_float_imm, load_global_float, mul_float,
    reinterpret_bits_to_float, store_global_float,
};

/// Block size for CSR5 tile kernel (should be a multiple of 32).
const CSR5_TILE_BLOCK: u32 = 256;

/// Block size for the calibration kernel.
const CSR5_CALIBRATE_BLOCK: u32 = 256;

/// CSR5 SpMV: `y = alpha * A * x + beta * y`.
///
/// Performs load-balanced SpMV using the CSR5 tile-based format.
///
/// # Arguments
///
/// * `handle` -- Sparse handle providing stream and device context.
/// * `csr5` -- Sparse CSR5 matrix `A`.
/// * `x` -- Dense input vector of length `A.cols()`.
/// * `y` -- Dense output vector of length `A.rows()`.
/// * `alpha` -- Scalar multiplier for `A * x`.
/// * `beta` -- Scalar multiplier for existing `y`.
///
/// # Errors
///
/// Returns [`SparseError::PtxGeneration`] if kernel generation fails.
/// Returns [`SparseError::Cuda`] on kernel launch failure.
/// Returns [`SparseError::DimensionMismatch`] if vector lengths are wrong.
pub fn csr5_spmv<T: GpuFloat>(
    handle: &SparseHandle,
    csr5: &Csr5Matrix<T>,
    x: &DeviceBuffer<T>,
    y: &mut DeviceBuffer<T>,
    alpha: T,
    beta: T,
) -> SparseResult<()> {
    if csr5.rows() == 0 || csr5.cols() == 0 {
        return Ok(());
    }

    if x.len() < csr5.cols() as usize {
        return Err(SparseError::DimensionMismatch(format!(
            "x length ({}) must be >= cols ({})",
            x.len(),
            csr5.cols()
        )));
    }
    if y.len() < csr5.rows() as usize {
        return Err(SparseError::DimensionMismatch(format!(
            "y length ({}) must be >= rows ({})",
            y.len(),
            csr5.rows()
        )));
    }

    // Phase 1: Tile kernel -- each warp processes one tile
    let tile_ptx = emit_csr5_tile_kernel::<T>(handle.sm_version())?;
    let tile_module = Arc::new(Module::from_ptx(&tile_ptx)?);
    let tile_kernel = Kernel::from_module(tile_module, "csr5_tile")?;

    // One warp per tile; warps_per_block = block / 32
    let warps_per_block = CSR5_TILE_BLOCK / 32;
    let tile_grid = grid_size_for(csr5.num_tiles(), warps_per_block);

    tile_kernel.launch(
        &LaunchParams::new(tile_grid, CSR5_TILE_BLOCK),
        handle.stream(),
        &(
            csr5.row_ptr().as_device_ptr(),
            csr5.col_idx().as_device_ptr(),
            csr5.values().as_device_ptr(),
            csr5.tile_ptr().as_device_ptr(),
            csr5.tile_desc().as_device_ptr(),
            x.as_device_ptr(),
            y.as_device_ptr(),
            csr5.calibrator().as_device_ptr(),
            alpha.to_bits_u64(),
            beta.to_bits_u64(),
            csr5.rows(),
            csr5.num_tiles(),
            csr5.nnz(),
        ),
    )?;

    // Phase 2: Calibration kernel -- merge cross-tile partial sums
    let cal_ptx = emit_csr5_calibrate_kernel::<T>(handle.sm_version())?;
    let cal_module = Arc::new(Module::from_ptx(&cal_ptx)?);
    let cal_kernel = Kernel::from_module(cal_module, "csr5_calibrate")?;

    let cal_grid = grid_size_for(csr5.rows(), CSR5_CALIBRATE_BLOCK);
    cal_kernel.launch(
        &LaunchParams::new(cal_grid, CSR5_CALIBRATE_BLOCK),
        handle.stream(),
        &(
            y.as_device_ptr(),
            csr5.calibrator().as_device_ptr(),
            beta.to_bits_u64(),
            csr5.rows(),
        ),
    )?;

    Ok(())
}

/// Generates PTX for the CSR5 tile kernel.
///
/// Each warp processes one tile of 32 non-zero elements. The kernel:
/// 1. Loads the tile descriptor to determine row boundaries
/// 2. Each lane loads one element, computes `val * x[col]`
/// 3. Uses warp shuffle to reduce partial sums within rows
/// 4. Lane 0 of each row segment writes to `y` or calibrator
fn emit_csr5_tile_kernel<T: GpuFloat>(sm: SmVersion) -> SparseResult<String> {
    let elem_bytes = T::size_u32();
    let is_f64 = T::SIZE == 8;
    let mov_suffix = if is_f64 { "f64" } else { "f32" };

    KernelBuilder::new("csr5_tile")
        .target(sm)
        .param("row_ptr", PtxType::U64)
        .param("col_idx", PtxType::U64)
        .param("values_ptr", PtxType::U64)
        .param("tile_ptr", PtxType::U64)
        .param("tile_desc", PtxType::U64)
        .param("x_ptr", PtxType::U64)
        .param("y_ptr", PtxType::U64)
        .param("calibrator_ptr", PtxType::U64)
        .param("alpha_bits", PtxType::U64)
        .param("beta_bits", PtxType::U64)
        .param("num_rows", PtxType::U32)
        .param("num_tiles", PtxType::U32)
        .param("nnz", PtxType::U32)
        .body(move |b| {
            // Warp ID = global_tid / 32
            let tid_global = b.global_thread_id_x();
            let num_tiles = b.load_param_u32("num_tiles");

            let lane = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("and.b32 {lane}, {tid_global}, 31;"));

            let tile_id = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("shr.u32 {tile_id}, {tid_global}, 5;"));

            let tile_id_inner = tile_id.clone();
            let lane_inner = lane.clone();
            b.if_lt_u32(tile_id, num_tiles, move |b| {
                let tile_id = tile_id_inner;
                let lane = lane_inner;

                let col_idx_base = b.load_param_u64("col_idx");
                let values_base = b.load_param_u64("values_ptr");
                let tile_ptr_base = b.load_param_u64("tile_ptr");
                let tile_desc_base = b.load_param_u64("tile_desc");
                let x_ptr = b.load_param_u64("x_ptr");
                let _y_ptr = b.load_param_u64("y_ptr");
                let calibrator_ptr = b.load_param_u64("calibrator_ptr");
                let alpha_bits = b.load_param_u64("alpha_bits");
                let beta_bits = b.load_param_u64("beta_bits");
                let num_rows_reg = b.load_param_u32("num_rows");
                let nnz_reg = b.load_param_u32("nnz");

                let alpha = reinterpret_bits_to_float::<T>(b, alpha_bits);
                // beta is used in the calibrate kernel, not here
                let _beta = reinterpret_bits_to_float::<T>(b, beta_bits);

                // Load tile_ptr[tile_id] to get the starting element index
                let tp_addr = b.byte_offset_addr(tile_ptr_base.clone(), tile_id.clone(), 4);
                let tile_start = b.load_global_u32(tp_addr);

                // This lane's element index
                let elem_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {elem_idx}, {tile_start}, {lane};"));

                // Check bounds: skip the load when elem_idx >= nnz. Inverted
                // skip-branch (`setp.lo` -> `setp.hs`) via the structured
                // `branch_if` so the target matches the `$`-prefixed label.
                let oob = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.hs.u32 {oob}, {elem_idx}, {nnz_reg};"));

                // Load value and column, compute product (zero if out of bounds)
                let product = load_float_imm::<T>(b, 0.0);

                let compute_label = b.fresh_label("csr5_compute");
                let after_compute = b.fresh_label("csr5_after_compute");

                b.branch_if(oob, &after_compute);
                b.label(&compute_label);

                // Load col_idx[elem_idx]
                let ci_addr = b.byte_offset_addr(col_idx_base, elem_idx.clone(), 4);
                let col_i32 = b.load_global_i32(ci_addr);
                let col_u32 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {col_u32}, {col_i32};"));

                // Load values[elem_idx]
                let v_addr = b.byte_offset_addr(values_base, elem_idx, elem_bytes);
                let val = load_global_float::<T>(b, v_addr);

                // Load x[col]
                let x_addr = b.byte_offset_addr(x_ptr, col_u32, elem_bytes);
                let x_val = load_global_float::<T>(b, x_addr);

                // product = val * x_val
                let prod = mul_float::<T>(b, val, x_val);
                b.raw_ptx(&format!("mov.{mov_suffix} {product}, {prod};"));

                b.label(&after_compute);

                // Load tile descriptor for this tile:
                // TileDescriptor has 2 u32 fields = 8 bytes per descriptor
                let desc_addr = b.byte_offset_addr(tile_desc_base, tile_id.clone(), 8);
                let seg_mask = b.load_global_u32(desc_addr.clone());

                // Load first_row (at offset +4 from desc_addr)
                let desc_addr_plus4 = b.alloc_reg(PtxType::U64);
                b.raw_ptx(&format!("add.u64 {desc_addr_plus4}, {desc_addr}, 4;"));
                let first_row = b.load_global_u32(desc_addr_plus4);

                // Determine which row this lane belongs to within the tile.
                // Count the number of set bits in seg_mask at positions <= lane.
                // This gives the row offset from first_row.
                //
                // We use a mask: (1 << (lane + 1)) - 1 to isolate bits 0..lane
                let lane_plus_1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {lane_plus_1}, {lane}, 1;"));

                let lane_mask = b.alloc_reg(PtxType::U32);
                let one = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {one}, 1;"));
                b.raw_ptx(&format!("shl.b32 {lane_mask}, {one}, {lane_plus_1};"));
                let lane_mask_sub = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("sub.u32 {lane_mask_sub}, {lane_mask}, 1;"));

                // Count bits in seg_mask & lane_mask
                let masked_seg = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "and.b32 {masked_seg}, {seg_mask}, {lane_mask_sub};"
                ));
                let row_offset = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("popc.b32 {row_offset}, {masked_seg};"));

                // This lane's row = first_row + row_offset
                let my_row = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {my_row}, {first_row}, {row_offset};"));

                // Warp-level segmented reduction:
                // Use inclusive scan to sum products within each row segment.
                // A lane starts a new segment if its bit in seg_mask is set.
                //
                // We do a simple approach: for each shuffle offset, check if
                // the source lane is in the same segment (same row).
                let acc = b.alloc_reg(T::PTX_TYPE);
                b.raw_ptx(&format!("mov.{mov_suffix} {acc}, {product};"));

                for offset in [1u32, 2, 4, 8, 16] {
                    // Segmented up-shuffle. `emit_shfl_float` handles the f64
                    // unpack/repack so no `.b64` shfl (rejected by ptxas) is
                    // emitted; for f32 it is a single `.b32` shuffle.
                    let shuffled =
                        emit_shfl_float::<T>(b, "up", acc.clone(), &offset.to_string(), "0");
                    // Only add if source lane (lane - offset) is in the same row
                    let src_lane = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("sub.u32 {src_lane}, {lane}, {offset};"));
                    // Check that lane >= offset (otherwise src is invalid)
                    let valid = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.ge.u32 {valid}, {lane}, {offset};"));
                    // Check that src is in same row segment by checking no
                    // segment boundary bits between src_lane+1 and lane
                    // For simplicity, check that my_row of src == my_row
                    // We re-compute src_row = first_row + popc(seg_mask & ((1<<src_lane+1)-1))
                    // But this is expensive in PTX. Instead, use the seg_mask directly:
                    // between lanes (src_lane, lane], if any bit is set in seg_mask,
                    // they are in different segments.
                    //
                    // Mask for bits in range (src_lane, lane]:
                    // range_mask = lane_mask_sub & ~((1 << (src_lane+1)) - 1)
                    // But if lane < offset, this is invalid.
                    //
                    // Simpler: use selp to conditionally add
                    let sum = b.alloc_reg(T::PTX_TYPE);
                    b.raw_ptx(&format!("add.{mov_suffix} {sum}, {acc}, {shuffled};"));
                    // We need to check if the shuffle source is the same row.
                    // Compute src_row via popc approach
                    let src_lane_p1 = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("add.u32 {src_lane_p1}, {src_lane}, 1;"));
                    let src_mask = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("shl.b32 {src_mask}, {one}, {src_lane_p1};"));
                    let src_mask_sub = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("sub.u32 {src_mask_sub}, {src_mask}, 1;"));
                    let src_masked = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "and.b32 {src_masked}, {seg_mask}, {src_mask_sub};"
                    ));
                    let src_row_off = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("popc.b32 {src_row_off}, {src_masked};"));
                    let src_row = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("add.u32 {src_row}, {first_row}, {src_row_off};"));
                    let same_row = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.eq.u32 {same_row}, {src_row}, {my_row};"));
                    // Combine: valid AND same_row
                    let do_add = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("and.pred {do_add}, {valid}, {same_row};"));
                    b.raw_ptx(&format!("selp.{mov_suffix} {acc}, {sum}, {acc}, {do_add};"));
                }

                // Now `acc` contains the inclusive segmented prefix sum.
                // The last lane for each row segment holds the row's total.
                //
                // A lane is the "last" for its segment if:
                //   lane == 31 OR the next lane starts a new segment
                let is_last = b.alloc_reg(PtxType::Pred);
                let is_lane_31 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.eq.u32 {is_lane_31}, {lane}, 31;"));

                // Check if next lane starts a new segment
                let next_lane = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {next_lane}, {lane}, 1;"));
                let next_bit = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("shr.b32 {next_bit}, {seg_mask}, {next_lane};"));
                let next_bit_masked = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("and.b32 {next_bit_masked}, {next_bit}, 1;"));
                let next_is_new_seg = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!(
                    "setp.ne.u32 {next_is_new_seg}, {next_bit_masked}, 0;"
                ));
                b.raw_ptx(&format!(
                    "or.pred {is_last}, {is_lane_31}, {next_is_new_seg};"
                ));

                // Write result if this lane is the last for its row
                // Only the last lane of each row segment writes. Branch past the
                // write when this lane is not the last (inverted: `is_last`'s
                // complement). We invert by testing `is_last == false` directly.
                let not_last = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("not.pred {not_last}, {is_last};"));
                let write_label = b.fresh_label("csr5_write");
                let skip_write = b.fresh_label("csr5_skip_write");
                b.branch_if(not_last, &skip_write);
                b.label(&write_label);

                // Check row is valid: skip the write when my_row >= num_rows
                // (inverted `setp.lo` -> `setp.hs`).
                let row_oob = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.hs.u32 {row_oob}, {my_row}, {num_rows_reg};"));
                let row_skip = b.fresh_label("csr5_row_skip");
                b.branch_if(row_oob, &row_skip);

                // For the first tile (tile_id == 0) and the row that starts at
                // the tile boundary, we can write directly to y with beta scaling.
                // For other tiles contributing to a cross-boundary row, write to
                // calibrator to be merged later.
                //
                // Simplified approach: use atomic add to y for partial rows,
                // or direct write. For correctness with beta, the first tile's
                // first write applies beta; subsequent writes add.
                //
                // For a clean implementation: all tiles write alpha*partial to
                // calibrator[my_row], then calibrate kernel merges.
                // But this would double-count rows fully within a tile.
                //
                // Better approach: check if this row started in the current tile
                // and ended in the current tile (fully contained). If so, write
                // directly to y. Otherwise, accumulate via calibrator.
                //
                // For now, use a simpler strategy:
                //   - Scale by alpha
                //   - If tile_id == 0 and lane covers the row from the start:
                //     y[row] = alpha*partial + beta*y[row]
                //   - Otherwise: use atomic add of alpha*partial to y[row]
                //
                // This works because:
                //   - Phase 1 scales y by beta only once (first tile touching
                //     each row)
                //   - Phase 2 calibration adds remaining partials

                let scaled_acc = mul_float::<T>(b, alpha.clone(), acc);

                // Write to calibrator and let the calibrate kernel handle it.
                // The calibrator accumulates partial sums per row.
                let cal_addr = b.byte_offset_addr(calibrator_ptr, my_row.clone(), elem_bytes);
                // Use atomic add for thread-safe accumulation
                let _old = b.alloc_reg(T::PTX_TYPE);
                b.raw_ptx(&format!(
                    "atom.global.add.{mov_suffix} {_old}, [{cal_addr}], {scaled_acc};"
                ));

                b.label(&row_skip);
                b.label(&skip_write);
            });

            b.ret();
        })
        .build()
        .map_err(|e| SparseError::PtxGeneration(e.to_string()))
}

/// Generates PTX for the CSR5 calibration kernel.
///
/// This kernel merges the partial sums from the calibrator into the final
/// `y` vector: `y[row] = calibrator[row] + beta * y[row]`.
fn emit_csr5_calibrate_kernel<T: GpuFloat>(sm: SmVersion) -> SparseResult<String> {
    let elem_bytes = T::size_u32();

    KernelBuilder::new("csr5_calibrate")
        .target(sm)
        .param("y_ptr", PtxType::U64)
        .param("calibrator_ptr", PtxType::U64)
        .param("beta_bits", PtxType::U64)
        .param("num_rows", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let num_rows = b.load_param_u32("num_rows");

            let gid_inner = gid.clone();
            b.if_lt_u32(gid, num_rows, move |b| {
                let row = gid_inner;
                let y_ptr = b.load_param_u64("y_ptr");
                let cal_ptr = b.load_param_u64("calibrator_ptr");
                let beta_bits = b.load_param_u64("beta_bits");
                let beta = reinterpret_bits_to_float::<T>(b, beta_bits);

                // Load calibrator[row] = alpha * (A*x)[row] (accumulated in phase 1).
                let cal_addr = b.byte_offset_addr(cal_ptr, row.clone(), elem_bytes);
                let cal_val = load_global_float::<T>(b, cal_addr);

                // Load y[row]
                let y_addr = b.byte_offset_addr(y_ptr, row, elem_bytes);
                let y_val = load_global_float::<T>(b, y_addr.clone());

                // y[row] = calibrator[row] + beta * y[row]
                //        = alpha * (A*x)[row] + beta * y_old[row]
                let beta_y = mul_float::<T>(b, beta, y_val);
                let result = add_float::<T>(b, cal_val, beta_y);
                store_global_float::<T>(b, y_addr, result);
            });

            b.ret();
        })
        .build()
        .map_err(|e| SparseError::PtxGeneration(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx_helpers::test_support::assert_assembles_and_clean;

    /// The CSR5 tile + calibrate kernels must assemble for sm_86 in both
    /// precisions. The f64 tile kernel exercises the segmented warp up-shuffle:
    /// it must split into `.b32` halves (no `shfl.sync.up.b64`).
    #[test]
    fn csr5_tile_calibrate_f32_f64_assemble_sm86() {
        let tile_f32 = emit_csr5_tile_kernel::<f32>(SmVersion::Sm86).expect("f32 tile");
        assert_assembles_and_clean("csr5_tile_f32", &tile_f32);
        let tile_f64 = emit_csr5_tile_kernel::<f64>(SmVersion::Sm86).expect("f64 tile");
        assert_assembles_and_clean("csr5_tile_f64", &tile_f64);
        assert!(
            !tile_f64.contains("shfl.sync.up.b64"),
            "f64 CSR5 tile kernel must not emit shfl.sync.up.b64:\n{tile_f64}"
        );
        assert!(
            tile_f64.contains("shfl.sync.up.b32"),
            "f64 CSR5 tile kernel must reduce via paired b32 up-shuffles:\n{tile_f64}"
        );
        assert!(
            !tile_f64.contains("0F00000000"),
            "f64 CSR5 tile kernel must not materialize an f32 0.0 immediate:\n{tile_f64}"
        );

        let cal_f32 = emit_csr5_calibrate_kernel::<f32>(SmVersion::Sm86).expect("f32 cal");
        assert_assembles_and_clean("csr5_calibrate_f32", &cal_f32);
        let cal_f64 = emit_csr5_calibrate_kernel::<f64>(SmVersion::Sm86).expect("f64 cal");
        assert_assembles_and_clean("csr5_calibrate_f64", &cal_f64);
    }

    #[test]
    fn csr5_tile_ptx_generates_f32() {
        let ptx = emit_csr5_tile_kernel::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx_text = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_text.contains(".entry csr5_tile"));
        assert!(ptx_text.contains(".target sm_80"));
    }

    #[test]
    fn csr5_tile_ptx_generates_f64() {
        let ptx = emit_csr5_tile_kernel::<f64>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx_text = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_text.contains(".entry csr5_tile"));
    }

    #[test]
    fn csr5_calibrate_ptx_generates_f32() {
        let ptx = emit_csr5_calibrate_kernel::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx_text = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_text.contains(".entry csr5_calibrate"));
    }

    #[test]
    fn csr5_calibrate_ptx_generates_f64() {
        let ptx = emit_csr5_calibrate_kernel::<f64>(SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    #[test]
    fn csr5_tile_ptx_contains_segmented_reduction() {
        let ptx = emit_csr5_tile_kernel::<f32>(SmVersion::Sm80);
        let ptx_text = ptx.expect("test: PTX gen should succeed");
        // Should contain warp shuffle instructions
        assert!(ptx_text.contains("shfl.sync.up"));
        // Should contain popcount for segment detection
        assert!(ptx_text.contains("popc.b32"));
    }

    #[test]
    fn csr5_tile_ptx_contains_atomic_add() {
        let ptx = emit_csr5_tile_kernel::<f32>(SmVersion::Sm80);
        let ptx_text = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_text.contains("atom.global.add"));
    }

    #[test]
    fn csr5_block_sizes_are_warp_aligned() {
        assert_eq!(CSR5_TILE_BLOCK % 32, 0);
        assert_eq!(CSR5_CALIBRATE_BLOCK % 32, 0);
    }
}

// ---------------------------------------------------------------------------
// On-device numeric validation (feature = "gpu-tests")
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_device_tests {
    use super::*;
    use crate::format::CsrMatrix;
    use crate::gpu_test_support::{assert_close, gpu_handle};
    use crate::host_csr::{f64_to_gpu, gpu_to_f64};
    use oxicuda_memory::DeviceBuffer;

    /// CPU oracle for `y = alpha * A * x + beta * y0` over a CSR matrix
    /// (row count is derived from `row_ptr`).
    fn cpu_csr_spmv(
        row_ptr: &[i32],
        col_idx: &[i32],
        values: &[f64],
        x: &[f64],
        y0: &[f64],
        alpha: f64,
        beta: f64,
    ) -> Vec<f64> {
        let rows = row_ptr.len() - 1;
        let mut y = vec![0.0_f64; rows];
        for (i, slot) in y.iter_mut().enumerate() {
            let mut acc = 0.0_f64;
            for k in row_ptr[i] as usize..row_ptr[i + 1] as usize {
                acc += values[k] * x[col_idx[k] as usize];
            }
            *slot = alpha * acc + beta * y0[i];
        }
        y
    }

    /// Drive the production `csr5_spmv` op and compare to the CPU oracle.
    #[allow(clippy::too_many_arguments)]
    fn run_csr5<T: GpuFloat>(
        rows: u32,
        cols: u32,
        row_ptr: &[i32],
        col_idx: &[i32],
        values: &[f64],
        x: &[f64],
        y0: &[f64],
        alpha: f64,
        beta: f64,
        tol: f64,
        tag: &str,
    ) {
        let Some(handle) = gpu_handle() else {
            return;
        };
        let dev_values: Vec<T> = values.iter().map(|&v| f64_to_gpu::<T>(v)).collect();
        // Build via CSR -> CSR5 so the tile metadata is produced by the real
        // conversion path.
        let csr = CsrMatrix::<T>::from_host(rows, cols, row_ptr, col_idx, &dev_values)
            .expect("test: build CSR");
        let csr5 = Csr5Matrix::<T>::from_csr(&csr).expect("test: build CSR5");

        let dev_x: Vec<T> = x.iter().map(|&v| f64_to_gpu::<T>(v)).collect();
        let dev_y: Vec<T> = y0.iter().map(|&v| f64_to_gpu::<T>(v)).collect();
        let x_buf = DeviceBuffer::from_host(&dev_x).expect("test: upload x");
        let mut y_buf = DeviceBuffer::from_host(&dev_y).expect("test: upload y");

        csr5_spmv::<T>(
            &handle,
            &csr5,
            &x_buf,
            &mut y_buf,
            f64_to_gpu::<T>(alpha),
            f64_to_gpu::<T>(beta),
        )
        .expect("test: csr5_spmv launch");
        handle.stream().synchronize().expect("test: sync");

        let mut out = vec![T::gpu_zero(); rows as usize];
        y_buf.copy_to_host(&mut out).expect("test: download y");
        let got: Vec<f64> = out.iter().map(|&v| gpu_to_f64(v)).collect();
        let want = cpu_csr_spmv(row_ptr, col_idx, values, x, y0, alpha, beta);
        assert_close(&got, &want, tol, tag);
    }

    /// Tridiagonal SPD-ish matrix of order `n` (3*n - 2 non-zeros). For `n = 20`
    /// this is 58 non-zeros => 2 CSR5 tiles, so rows straddle the tile boundary
    /// at element 32 and exercise the cross-tile calibrator merge.
    fn tridiagonal(n: usize) -> (u32, u32, Vec<i32>, Vec<i32>, Vec<f64>) {
        let mut row_ptr = vec![0i32];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            if i > 0 {
                col_idx.push((i - 1) as i32);
                values.push(-1.0);
            }
            col_idx.push(i as i32);
            values.push(4.0 + 0.01 * (i as f64));
            if i + 1 < n {
                col_idx.push((i + 1) as i32);
                values.push(-1.0);
            }
            row_ptr.push(col_idx.len() as i32);
        }
        (n as u32, n as u32, row_ptr, col_idx, values)
    }

    #[test]
    fn csr5_single_tile_f64_beta_one() {
        // 10 rows => 28 nnz => single tile.
        let (r, c, rp, ci, v) = tridiagonal(10);
        let x: Vec<f64> = (0..r as usize).map(|i| 1.0 + i as f64).collect();
        let y0 = vec![0.0_f64; r as usize];
        run_csr5::<f64>(
            r,
            c,
            &rp,
            &ci,
            &v,
            &x,
            &y0,
            1.0,
            1.0,
            1e-10,
            "csr5_f64_single",
        );
    }

    #[test]
    fn csr5_cross_tile_f64_beta_nonunit() {
        // 20 rows => 58 nnz => 2 tiles. beta != 1 exercises the (previously
        // dropped) beta scale in the calibration kernel.
        let (r, c, rp, ci, v) = tridiagonal(20);
        let x: Vec<f64> = (0..r as usize).map(|i| 0.5 + 0.25 * i as f64).collect();
        let y0: Vec<f64> = (0..r as usize).map(|i| 100.0 - i as f64).collect();
        run_csr5::<f64>(
            r,
            c,
            &rp,
            &ci,
            &v,
            &x,
            &y0,
            2.0,
            -0.5,
            1e-10,
            "csr5_f64_cross",
        );
    }

    #[test]
    fn csr5_cross_tile_f32_alpha_beta() {
        let (r, c, rp, ci, v) = tridiagonal(20);
        let x: Vec<f64> = (0..r as usize).map(|i| 1.0 + 0.1 * i as f64).collect();
        let y0: Vec<f64> = (0..r as usize).map(|i| 5.0 + i as f64).collect();
        run_csr5::<f32>(
            r,
            c,
            &rp,
            &ci,
            &v,
            &x,
            &y0,
            1.5,
            0.25,
            1e-4,
            "csr5_f32_cross",
        );
    }

    #[test]
    fn csr5_beta_zero_overwrites() {
        // beta = 0 must fully overwrite the prior y.
        let (r, c, rp, ci, v) = tridiagonal(12);
        let x = vec![1.0_f64; r as usize];
        let y0 = vec![1e9_f64; r as usize];
        run_csr5::<f64>(r, c, &rp, &ci, &v, &x, &y0, 1.0, 0.0, 1e-10, "csr5_beta0");
    }
}
