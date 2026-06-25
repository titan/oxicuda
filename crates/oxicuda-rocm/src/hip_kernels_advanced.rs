//! Advanced HIP C++ kernel source generators for AMD ROCm GPUs.
//!
//! Complements [`crate::hip_kernels`] with kernels that exploit AMD-specific
//! wavefront primitives and LDS (Local Data Share) tiling:
//!
//! - **Wavefront reductions** using `__shfl_down` / `__ballot` cross-lane ops.
//! - **LDS-tiled GEMM** with a 4-byte skew to avoid 32-bank LDS conflicts.
//! - **Softmax**, **LayerNorm**, **inclusive scan (prefix sum)**, and
//!   **transpose** kernels with `__shared__` staging.
//!
//! Each generator returns a complete, structurally-valid HIP translation unit;
//! the unit tests assert the presence of the right intrinsics, bounds guards,
//! arithmetic, and LDS declarations.  No GPU is required.

// ─── Wavefront reduction ──────────────────────────────────────────────────────

/// HIP C++ source for a single-pass reduction kernel that uses AMD wavefront
/// shuffle intrinsics (`__shfl_down`) for the intra-wave phase and `__shared__`
/// LDS for the inter-wave phase.
///
/// `wave_width` must be 32 (RDNA) or 64 (CDNA).  Supported `op`: `"sum"`,
/// `"max"`, `"min"`.  Returns an empty string for an unknown op.
pub fn wavefront_reduce_hip(op: &str, wave_width: u32) -> String {
    let (identity, combine, name) = match op {
        "sum" => ("0.0f", "a + b", "wave_reduce_sum_f32"),
        "max" => ("-HUGE_VALF", "fmaxf(a, b)", "wave_reduce_max_f32"),
        "min" => ("HUGE_VALF", "fminf(a, b)", "wave_reduce_min_f32"),
        _ => return String::new(),
    };
    format!(
        r#"
#include <math.h>

__device__ inline float wave_combine(float a, float b) {{ return {combine}; }}

extern "C" __global__ void {name}(
    const float* __restrict__ input,
    float*       __restrict__ output,
    unsigned int n
) {{
    const unsigned int WAVE = {wave};
    __shared__ float lds[64 / 1]; // one slot per wave in the block (<= 64)

    unsigned int gid  = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    unsigned int lane = hipThreadIdx_x % WAVE;
    unsigned int wid  = hipThreadIdx_x / WAVE;

    // 1. Each thread loads (or identity if out of range).
    float v = (gid < n) ? input[gid] : {identity};

    // 2. Intra-wave reduction via cross-lane shuffle.
    for (unsigned int offset = WAVE / 2; offset > 0; offset >>= 1) {{
        float other = __shfl_down(v, offset, WAVE);
        v = wave_combine(v, other);
    }}

    // 3. Lane 0 of each wave writes its partial into LDS.
    if (lane == 0) {{
        lds[wid] = v;
    }}
    __syncthreads();

    // 4. The first wave reduces the per-wave partials.
    unsigned int waves_in_block = (hipBlockDim_x + WAVE - 1) / WAVE;
    if (wid == 0) {{
        float w = (lane < waves_in_block) ? lds[lane] : {identity};
        for (unsigned int offset = WAVE / 2; offset > 0; offset >>= 1) {{
            float other = __shfl_down(w, offset, WAVE);
            w = wave_combine(w, other);
        }}
        if (lane == 0) {{
            output[hipBlockIdx_x] = w;
        }}
    }}
}}
"#,
        combine = combine,
        name = name,
        wave = wave_width,
        identity = identity,
    )
}

/// HIP C++ source for an active-lane count kernel using `__ballot`.
///
/// Counts, per block, how many input elements satisfy `value > threshold`
/// using the AMD 64-bit `__ballot` wavefront vote intrinsic and
/// `__popcll` population count.
pub fn ballot_count_hip() -> &'static str {
    r#"
extern "C" __global__ void ballot_count_gt(
    const float* __restrict__ input,
    unsigned int* __restrict__ counts,
    float threshold,
    unsigned int n
) {
    unsigned int gid  = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    unsigned int lane = hipThreadIdx_x % warpSize;

    int pred = (gid < n) && (input[gid] > threshold);
    // 64-bit wavefront ballot: bit l set iff lane l's predicate is true.
    unsigned long long mask = __ballot(pred);
    unsigned int active = __popcll(mask);

    if (lane == 0) {
        atomicAdd(&counts[hipBlockIdx_x], active);
    }
}
"#
}

// ─── LDS-tiled GEMM with bank-conflict skew ───────────────────────────────────

/// HIP C++ source for an LDS-tiled single-precision GEMM that pads each shared
/// tile row by one float (4 bytes) to eliminate 32-way LDS bank conflicts.
///
/// The A and B tiles are staged in `__shared__` memory of size
/// `tile x (tile + 1)`; the `+1` skew ensures consecutive rows map to
/// different LDS banks.
///
/// Grid: `dim3((n+ts-1)/ts, (m+ts-1)/ts)`, Block: `dim3(ts, ts)`.
pub fn gemm_lds_tiled_hip(tile_size: u32) -> String {
    format!(
        r#"
extern "C" __global__ void gemm_lds_tiled_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float*       __restrict__ c,
    unsigned int m,
    unsigned int n,
    unsigned int k,
    float alpha,
    float beta
) {{
    const unsigned int TS = {ts};
    // +1 column skew avoids 32-bank LDS conflicts on consecutive rows.
    __shared__ float a_tile[{ts}][{ts} + 1];
    __shared__ float b_tile[{ts}][{ts} + 1];

    unsigned int ty = hipThreadIdx_y;
    unsigned int tx = hipThreadIdx_x;
    unsigned int row = hipBlockIdx_y * TS + ty;
    unsigned int col = hipBlockIdx_x * TS + tx;

    float acc = 0.0f;
    unsigned int num_tiles = (k + TS - 1) / TS;
    for (unsigned int t = 0; t < num_tiles; ++t) {{
        unsigned int a_col = t * TS + tx;
        unsigned int b_row = t * TS + ty;
        a_tile[ty][tx] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
        b_tile[ty][tx] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (unsigned int i = 0; i < TS; ++i) {{
            acc += a_tile[ty][i] * b_tile[i][tx];
        }}
        __syncthreads();
    }}

    if (row < m && col < n) {{
        unsigned int idx = row * n + col;
        c[idx] = alpha * acc + beta * c[idx];
    }}
}}
"#,
        ts = tile_size
    )
}

// ─── Softmax ──────────────────────────────────────────────────────────────────

/// HIP C++ source for a numerically-stable row-wise softmax kernel.
///
/// Each block handles one row of an `[rows, cols]` matrix using a three-pass
/// LDS reduction: max, exp-sum, then normalise.
///
/// Grid: `(rows)`, Block: `block_size` (with `block_size` floats of dynamic LDS).
pub fn softmax_hip() -> &'static str {
    r#"
#include <math.h>

extern "C" __global__ void softmax_rows_f32(
    const float* __restrict__ input,
    float*       __restrict__ output,
    unsigned int rows,
    unsigned int cols
) {
    extern __shared__ float sdata[];
    unsigned int row = hipBlockIdx_x;
    unsigned int lid = hipThreadIdx_x;
    unsigned int bs  = hipBlockDim_x;
    if (row >= rows) return;

    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;

    // Pass 1: row maximum.
    float local_max = -HUGE_VALF;
    for (unsigned int c = lid; c < cols; c += bs) {
        local_max = fmaxf(local_max, in_row[c]);
    }
    sdata[lid] = local_max;
    __syncthreads();
    for (unsigned int s = bs / 2; s > 0; s >>= 1) {
        if (lid < s) sdata[lid] = fmaxf(sdata[lid], sdata[lid + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // Pass 2: exp-sum.
    float local_sum = 0.0f;
    for (unsigned int c = lid; c < cols; c += bs) {
        local_sum += expf(in_row[c] - row_max);
    }
    sdata[lid] = local_sum;
    __syncthreads();
    for (unsigned int s = bs / 2; s > 0; s >>= 1) {
        if (lid < s) sdata[lid] = sdata[lid] + sdata[lid + s];
        __syncthreads();
    }
    float row_sum = sdata[0];
    float inv = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

    // Pass 3: normalise.
    for (unsigned int c = lid; c < cols; c += bs) {
        out_row[c] = expf(in_row[c] - row_max) * inv;
    }
}
"#
}

// ─── LayerNorm ────────────────────────────────────────────────────────────────

/// HIP C++ source for a row-wise LayerNorm kernel with affine `gamma`/`beta`.
///
/// Computes `y = (x - mean) / sqrt(var + eps) * gamma + beta` per row of an
/// `[rows, cols]` matrix using a two-pass LDS reduction (mean, then variance).
///
/// Grid: `(rows)`, Block: `block_size` (with `block_size` floats of dynamic LDS).
pub fn layernorm_hip() -> &'static str {
    r#"
#include <math.h>

extern "C" __global__ void layernorm_rows_f32(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float*       __restrict__ output,
    unsigned int rows,
    unsigned int cols,
    float eps
) {
    extern __shared__ float sdata[];
    unsigned int row = hipBlockIdx_x;
    unsigned int lid = hipThreadIdx_x;
    unsigned int bs  = hipBlockDim_x;
    if (row >= rows) return;

    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;

    // Pass 1: mean.
    float local = 0.0f;
    for (unsigned int c = lid; c < cols; c += bs) local += in_row[c];
    sdata[lid] = local;
    __syncthreads();
    for (unsigned int s = bs / 2; s > 0; s >>= 1) {
        if (lid < s) sdata[lid] += sdata[lid + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)cols;
    __syncthreads();

    // Pass 2: variance.
    float local_var = 0.0f;
    for (unsigned int c = lid; c < cols; c += bs) {
        float d = in_row[c] - mean;
        local_var += d * d;
    }
    sdata[lid] = local_var;
    __syncthreads();
    for (unsigned int s = bs / 2; s > 0; s >>= 1) {
        if (lid < s) sdata[lid] += sdata[lid + s];
        __syncthreads();
    }
    float var = sdata[0] / (float)cols;
    float inv_std = rsqrtf(var + eps);

    // Apply normalisation with affine transform.
    for (unsigned int c = lid; c < cols; c += bs) {
        float norm = (in_row[c] - mean) * inv_std;
        out_row[c] = norm * gamma[c] + beta[c];
    }
}
"#
}

// ─── Inclusive scan (prefix sum) ──────────────────────────────────────────────

/// HIP C++ source for a per-block inclusive scan (prefix sum) using the
/// work-efficient Blelloch up/down-sweep over `__shared__` LDS.
///
/// Each block scans up to `2 * block_size` elements.  The grid is sized so each
/// block owns a contiguous segment; the per-block totals are written to
/// `block_sums` for an optional second-level scan.
///
/// Block: `block_size`, dynamic LDS: `2 * block_size` floats.
pub fn inclusive_scan_hip() -> &'static str {
    r#"
extern "C" __global__ void inclusive_scan_f32(
    const float* __restrict__ input,
    float*       __restrict__ output,
    float*       __restrict__ block_sums,
    unsigned int n
) {
    extern __shared__ float temp[];
    unsigned int tid = hipThreadIdx_x;
    unsigned int bs  = hipBlockDim_x;
    unsigned int base = hipBlockIdx_x * (2 * bs);

    unsigned int ai = base + tid;
    unsigned int bi = base + tid + bs;
    temp[tid]      = (ai < n) ? input[ai] : 0.0f;
    temp[tid + bs] = (bi < n) ? input[bi] : 0.0f;

    // Up-sweep (reduce) phase.
    unsigned int offset = 1;
    for (unsigned int d = bs; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            unsigned int a = offset * (2 * tid + 1) - 1;
            unsigned int b = offset * (2 * tid + 2) - 1;
            temp[b] += temp[a];
        }
        offset *= 2;
    }

    // Save block total, then clear the last element for the down-sweep.
    if (tid == 0) {
        if (block_sums) block_sums[hipBlockIdx_x] = temp[2 * bs - 1];
        temp[2 * bs - 1] = 0.0f;
    }

    // Down-sweep phase.
    for (unsigned int d = 1; d < 2 * bs; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            unsigned int a = offset * (2 * tid + 1) - 1;
            unsigned int b = offset * (2 * tid + 2) - 1;
            float t = temp[a];
            temp[a] = temp[b];
            temp[b] += t;
        }
    }
    __syncthreads();

    // Convert exclusive to inclusive by adding the input back.
    if (ai < n) output[ai] = temp[tid]      + ((ai < n) ? input[ai] : 0.0f);
    if (bi < n) output[bi] = temp[tid + bs] + ((bi < n) ? input[bi] : 0.0f);
}
"#
}

// ─── Transpose ────────────────────────────────────────────────────────────────

/// HIP C++ source for a tiled matrix transpose with a bank-conflict-avoiding
/// `+1` LDS skew.  Transposes an `[rows, cols]` matrix into `[cols, rows]`.
///
/// Grid: `dim3((cols+ts-1)/ts, (rows+ts-1)/ts)`, Block: `dim3(ts, ts)`.
pub fn transpose_tiled_hip(tile_size: u32) -> String {
    format!(
        r#"
extern "C" __global__ void transpose_tiled_f32(
    const float* __restrict__ input,
    float*       __restrict__ output,
    unsigned int rows,
    unsigned int cols
) {{
    const unsigned int TS = {ts};
    __shared__ float tile[{ts}][{ts} + 1]; // +1 avoids LDS bank conflicts

    unsigned int x = hipBlockIdx_x * TS + hipThreadIdx_x;
    unsigned int y = hipBlockIdx_y * TS + hipThreadIdx_y;
    if (x < cols && y < rows) {{
        tile[hipThreadIdx_y][hipThreadIdx_x] = input[y * cols + x];
    }}
    __syncthreads();

    unsigned int tx = hipBlockIdx_y * TS + hipThreadIdx_x;
    unsigned int ty = hipBlockIdx_x * TS + hipThreadIdx_y;
    if (tx < rows && ty < cols) {{
        output[ty * rows + tx] = tile[hipThreadIdx_x][hipThreadIdx_y];
    }}
}}
"#,
        ts = tile_size
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wavefront_reduce_sum_uses_shfl() {
        let src = wavefront_reduce_hip("sum", 64);
        assert!(src.contains("wave_reduce_sum_f32"));
        assert!(src.contains("__shfl_down"));
        assert!(src.contains("__shared__"));
        assert!(src.contains("a + b"));
        assert!(src.contains("const unsigned int WAVE = 64;"));
        assert!(src.contains("__global__"));
    }

    #[test]
    fn wavefront_reduce_all_ops() {
        assert!(wavefront_reduce_hip("max", 32).contains("fmaxf(a, b)"));
        assert!(wavefront_reduce_hip("max", 32).contains("-HUGE_VALF"));
        assert!(wavefront_reduce_hip("min", 64).contains("fminf(a, b)"));
        assert!(wavefront_reduce_hip("min", 64).contains("HUGE_VALF"));
        assert!(wavefront_reduce_hip("unknown", 64).is_empty());
    }

    #[test]
    fn wavefront_reduce_wave_width_embedded() {
        assert!(wavefront_reduce_hip("sum", 32).contains("WAVE = 32;"));
        assert!(wavefront_reduce_hip("sum", 64).contains("WAVE = 64;"));
    }

    #[test]
    fn ballot_count_uses_ballot_and_popcll() {
        let src = ballot_count_hip();
        assert!(src.contains("__ballot"));
        assert!(src.contains("__popcll"));
        assert!(src.contains("warpSize"));
        assert!(src.contains("atomicAdd"));
        assert!(src.contains("unsigned long long mask"));
    }

    #[test]
    fn gemm_lds_tiled_has_skew_and_sync() {
        let src = gemm_lds_tiled_hip(16);
        assert!(src.contains("gemm_lds_tiled_f32"));
        assert!(src.contains("__shared__"));
        // +1 skew on both tiles.
        assert!(src.contains("a_tile[16][16 + 1]"));
        assert!(src.contains("b_tile[16][16 + 1]"));
        assert!(src.contains("__syncthreads"));
        assert!(src.contains("alpha * acc + beta"));
    }

    #[test]
    fn gemm_lds_tiled_tile_size_embedded() {
        assert!(gemm_lds_tiled_hip(32).contains("a_tile[32][32 + 1]"));
        assert!(gemm_lds_tiled_hip(8).contains("const unsigned int TS = 8;"));
    }

    #[test]
    fn softmax_is_numerically_stable() {
        let src = softmax_hip();
        assert!(src.contains("softmax_rows_f32"));
        assert!(src.contains("__global__"));
        assert!(src.contains("fmaxf")); // max pass
        assert!(src.contains("expf(in_row[c] - row_max)")); // shifted exp
        assert!(src.contains("__syncthreads"));
        assert!(src.contains("extern __shared__ float sdata[]"));
    }

    #[test]
    fn layernorm_has_mean_var_affine() {
        let src = layernorm_hip();
        assert!(src.contains("layernorm_rows_f32"));
        assert!(src.contains("rsqrtf(var + eps)"));
        assert!(src.contains("norm * gamma[c] + beta[c]"));
        assert!(src.contains("(float)cols"));
        assert!(src.contains("eps"));
    }

    #[test]
    fn inclusive_scan_has_up_and_down_sweep() {
        let src = inclusive_scan_hip();
        assert!(src.contains("inclusive_scan_f32"));
        assert!(src.contains("extern __shared__ float temp[]"));
        assert!(src.contains("block_sums"));
        // exclusive→inclusive conversion adds input back.
        assert!(src.contains("temp[tid]      + "));
        assert!(src.contains("__syncthreads"));
    }

    #[test]
    fn transpose_has_bank_conflict_skew() {
        let src = transpose_tiled_hip(16);
        assert!(src.contains("transpose_tiled_f32"));
        assert!(src.contains("tile[16][16 + 1]"));
        assert!(src.contains("__syncthreads"));
        assert!(src.contains("output[ty * rows + tx]"));
    }

    #[test]
    fn transpose_tile_size_embedded() {
        assert!(transpose_tiled_hip(32).contains("tile[32][32 + 1]"));
        assert!(transpose_tiled_hip(8).contains("const unsigned int TS = 8;"));
    }
}
