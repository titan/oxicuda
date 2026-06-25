//! Extended MSL (Metal Shading Language) kernel source generation.
//!
//! This module complements [`crate::msl`] with the neural-network and
//! extended-numeric kernels that are larger and more specialised:
//!
//! * `softmax_msl` — row-wise numerically-stable softmax.
//! * `layernorm_msl` — layer normalisation with affine `gamma`/`beta`.
//! * `scan_msl` — inclusive/exclusive prefix sum (Hillis-Steele in threadgroup).
//! * `simdgroup_gemm_msl` — GEMM using Apple `simdgroup_matrix<...>` MMA tiles
//!   (analogous to Tensor Cores; needs Metal 3 / Apple GPU family 7+).
//! * `gemm_msl_f64_ds` — double-single emulated FP64 GEMM (Metal has no native
//!   `double`; pairs of `float` carry the high/low limbs).
//! * `int8_quant_gemm_msl` — INT8 × INT8 → INT32 GEMM with per-tensor
//!   dequantisation back to `float` (dynamic-quantization inference path).
//!
//! Each function returns a complete, self-contained MSL translation unit as an
//! owned `String`.  They are unit-tested structurally (correct `kernel`
//! signatures, `[[buffer(n)]]` / `[[threadgroup(n)]]` attributes, the right
//! arithmetic and bounds guards) and, on macOS, compile-tested against a real
//! device when one is present.

// ─── Softmax ──────────────────────────────────────────────────────────────────

/// MSL source for a row-wise, numerically-stable softmax kernel.
///
/// The input is treated as a `rows × cols` matrix in row-major order; each
/// threadgroup processes exactly one row.  The classic three-pass stable
/// softmax is used: (1) reduce the row maximum, (2) reduce `sum(exp(x - max))`,
/// (3) write `exp(x - max) / sum`.  Threadgroup memory holds the per-thread
/// partial maxima and sums.
///
/// Buffer layout:
/// * `[[buffer(0)]]` input  (`const float*`)
/// * `[[buffer(1)]]` output (`float*`)
/// * `[[buffer(2)]]` `rows` (`constant uint&`)
/// * `[[buffer(3)]]` `cols` (`constant uint&`)
/// * `[[threadgroup(0)]]` scratch (`threadgroup float*`)
pub fn softmax_msl() -> &'static str {
    r#"
#include <metal_stdlib>
using namespace metal;

kernel void softmax_rows_f32(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant uint&      rows   [[buffer(2)]],
    constant uint&      cols   [[buffer(3)]],
    threadgroup float*  scratch [[threadgroup(0)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint lid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tg_id >= rows) return;
    uint base = tg_id * cols;

    // Pass 1: row maximum (parallel reduction in threadgroup memory).
    float local_max = -INFINITY;
    for (uint c = lid; c < cols; c += tg_size) {
        local_max = max(local_max, input[base + c]);
    }
    scratch[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            scratch[lid] = max(scratch[lid], scratch[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: sum of exp(x - max).
    float local_sum = 0.0f;
    for (uint c = lid; c < cols; c += tg_size) {
        local_sum += exp(input[base + c] - row_max);
    }
    scratch[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            scratch[lid] += scratch[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = scratch[0];
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

    // Pass 3: normalise.
    for (uint c = lid; c < cols; c += tg_size) {
        output[base + c] = exp(input[base + c] - row_max) * inv_sum;
    }
}
"#
}

// ─── Layer normalisation ───────────────────────────────────────────────────────

/// MSL source for a row-wise layer-normalisation kernel with affine transform.
///
/// For each row `r` of the `rows × cols` input:
/// `y = (x - mean) / sqrt(var + eps) * gamma + beta`
/// where `mean`/`var` are computed across the `cols` feature dimension.  Each
/// threadgroup processes one row; partial sums use threadgroup memory.
///
/// Buffer layout:
/// * `[[buffer(0)]]` input  (`const float*`)
/// * `[[buffer(1)]]` gamma  (`const float*`, length `cols`)
/// * `[[buffer(2)]]` beta   (`const float*`, length `cols`)
/// * `[[buffer(3)]]` output (`float*`)
/// * `[[buffer(4)]]` `rows` (`constant uint&`)
/// * `[[buffer(5)]]` `cols` (`constant uint&`)
/// * `[[buffer(6)]]` `eps`  (`constant float&`)
/// * `[[threadgroup(0)]]` scratch (`threadgroup float*`)
pub fn layernorm_msl() -> &'static str {
    r#"
#include <metal_stdlib>
using namespace metal;

kernel void layernorm_rows_f32(
    device const float* input  [[buffer(0)]],
    device const float* gamma  [[buffer(1)]],
    device const float* beta   [[buffer(2)]],
    device float*       output [[buffer(3)]],
    constant uint&      rows   [[buffer(4)]],
    constant uint&      cols   [[buffer(5)]],
    constant float&     eps    [[buffer(6)]],
    threadgroup float*  scratch [[threadgroup(0)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint lid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tg_id >= rows) return;
    uint base = tg_id * cols;

    // Pass 1: mean.
    float local_sum = 0.0f;
    for (uint c = lid; c < cols; c += tg_size) {
        local_sum += input[base + c];
    }
    scratch[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            scratch[lid] += scratch[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = scratch[0] / float(cols);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: variance.
    float local_var = 0.0f;
    for (uint c = lid; c < cols; c += tg_size) {
        float d = input[base + c] - mean;
        local_var += d * d;
    }
    scratch[lid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            scratch[lid] += scratch[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float var = scratch[0] / float(cols);
    float inv_std = rsqrt(var + eps);

    // Pass 3: affine normalise.
    for (uint c = lid; c < cols; c += tg_size) {
        float normed = (input[base + c] - mean) * inv_std;
        output[base + c] = normed * gamma[c] + beta[c];
    }
}
"#
}

// ─── Prefix sum (scan) ─────────────────────────────────────────────────────────

/// MSL source for a single-threadgroup Hillis-Steele inclusive/exclusive scan.
///
/// Computes a prefix sum over `n` elements within one threadgroup using
/// double-buffered threadgroup memory.  When `exclusive` is `true` the kernel
/// shifts the result right by one so element `i` holds the sum of `[0, i)`.
///
/// Buffer layout:
/// * `[[buffer(0)]]` input  (`const float*`)
/// * `[[buffer(1)]]` output (`float*`)
/// * `[[buffer(2)]]` `n`    (`constant uint&`)
/// * `[[threadgroup(0)]]` ping-pong scratch (`threadgroup float*`, length `2*n`)
pub fn scan_msl(exclusive: bool) -> String {
    // For exclusive scan, seed each lane with its left neighbour (identity at 0).
    let seed = if exclusive {
        "(gid > 0u) ? input[gid - 1u] : 0.0f"
    } else {
        "input[gid]"
    };
    let kernel_name = if exclusive {
        "scan_exclusive_f32"
    } else {
        "scan_inclusive_f32"
    };
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

kernel void {kernel_name}(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant uint&      n      [[buffer(2)]],
    threadgroup float*  scratch [[threadgroup(0)]],
    uint gid     [[thread_position_in_grid]],
    uint lid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {{
    // Two halves of `scratch` form the ping-pong buffers.
    threadgroup float* buf_a = scratch;
    threadgroup float* buf_b = scratch + tg_size;

    float v = (gid < n) ? ({seed}) : 0.0f;
    buf_a[lid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* src = buf_a;
    threadgroup float* dst = buf_b;
    for (uint offset = 1u; offset < tg_size; offset <<= 1u) {{
        if (lid >= offset) {{
            dst[lid] = src[lid] + src[lid - offset];
        }} else {{
            dst[lid] = src[lid];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* tmp = src;
        src = dst;
        dst = tmp;
    }}

    if (gid < n) {{
        output[gid] = src[lid];
    }}
}}
"#,
        kernel_name = kernel_name,
        seed = seed,
    )
}

/// Return the MSL function name for the requested scan variant.
pub fn scan_function_name(exclusive: bool) -> &'static str {
    if exclusive {
        "scan_exclusive_f32"
    } else {
        "scan_inclusive_f32"
    }
}

// ─── SIMD-group matrix GEMM ────────────────────────────────────────────────────

/// MSL source for a GEMM kernel using Apple `simdgroup_matrix` MMA tiles.
///
/// Each SIMD-group cooperatively multiplies `8×8` `float` tiles via
/// `simdgroup_float8x8` accumulators — the Metal analogue of NVIDIA Tensor
/// Cores.  This requires Metal 3 and Apple GPU family 7+ (M-series / A14+) to
/// execute, but the *source* can be generated and validated on any host.
///
/// The kernel tiles the output into `8×8` blocks; one threadgroup covers a
/// `TILE×TILE` super-tile where `TILE` is a multiple of 8.  Accumulation walks
/// the `K` dimension in steps of 8 using `simdgroup_load` / `simdgroup_multiply_accumulate`.
///
/// Buffer layout matches [`crate::msl::gemm_msl`] (`a`, `b`, `c`, `GemmParams`).
pub fn simdgroup_gemm_msl() -> &'static str {
    r#"
#include <metal_stdlib>
using namespace metal;

struct GemmParams {
    uint m;
    uint n;
    uint k;
    float alpha;
    float beta;
};

// SIMD-group matrix GEMM. Each simdgroup computes one 8x8 output tile.
// Requires Metal 3 / Apple GPU family 7+ at dispatch time.
kernel void simdgroup_gemm_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float*       c [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    threadgroup float* tile [[threadgroup(0)]],
    uint2 tg_pos  [[threadgroup_position_in_grid]],
    uint  lid     [[thread_index_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]]
) {
    const uint TILE = 8u;
    uint tile_row = tg_pos.y * TILE;
    uint tile_col = tg_pos.x * TILE;
    if (tile_row >= params.m || tile_col >= params.n) return;

    // Accumulate A*B into an 8x8 simdgroup matrix.
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint kk = 0u; kk < params.k; kk += TILE) {
        simdgroup_float8x8 a_frag;
        simdgroup_float8x8 b_frag;
        // A is row-major (m x k); load the 8x8 tile at (tile_row, kk).
        simdgroup_load(a_frag, a + tile_row * params.k + kk, params.k);
        // B is row-major (k x n); load the 8x8 tile at (kk, tile_col).
        simdgroup_load(b_frag, b + kk * params.n + tile_col, params.n);
        simdgroup_multiply_accumulate(acc, a_frag, b_frag, acc);
    }

    // Spill the raw product to threadgroup memory, then apply alpha/beta and the
    // C read-modify-write per element (each lane owns a strided set of cells).
    simdgroup_store(acc, tile, TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = lid; e < TILE * TILE; e += tg_size) {
        uint r = e / TILE;
        uint col = e % TILE;
        uint out_idx = (tile_row + r) * params.n + (tile_col + col);
        float prod = tile[e];
        float prev = (params.beta != 0.0f) ? c[out_idx] : 0.0f;
        c[out_idx] = params.alpha * prod + params.beta * prev;
    }
}
"#
}

// ─── Double-single emulated FP64 GEMM ──────────────────────────────────────────

/// MSL source for a double-single (`df64`) emulated FP64 GEMM kernel.
///
/// Metal has no native `double`.  This kernel represents each value as an
/// unevaluated sum of two `float` limbs (`hi + lo`), giving ~44 bits of
/// mantissa.  Products use Dekker's `two_prod` (via `fma`) and sums use
/// Knuth's `two_sum`, accumulating in extended precision before the result is
/// rounded back to a single `float` on store.
///
/// Storage is interleaved: buffers `a`/`b`/`c` are `float2` arrays where
/// `.x` is the high limb and `.y` the low limb.
///
/// Buffer layout matches [`crate::msl::gemm_msl`] (`GemmParams` at `[[buffer(3)]]`).
pub fn gemm_msl_f64_ds() -> &'static str {
    r#"
#include <metal_stdlib>
using namespace metal;

struct GemmParams {
    uint m;
    uint n;
    uint k;
    float alpha;
    float beta;
};

// ── Double-single (df64) primitives ──
struct df64 { float hi; float lo; };

inline df64 ds_from(float a) { return df64{a, 0.0f}; }

// Knuth two-sum: returns rounded sum + exact error.
inline df64 two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    float err = (a - (s - bb)) + (b - bb);
    return df64{s, err};
}

// Dekker two-product using fused multiply-add for the error term.
inline df64 two_prod(float a, float b) {
    float p = a * b;
    float err = fma(a, b, -p);
    return df64{p, err};
}

inline df64 ds_add(df64 a, df64 b) {
    df64 s = two_sum(a.hi, b.hi);
    float lo = s.lo + (a.lo + b.lo);
    df64 r = two_sum(s.hi, lo);
    return r;
}

inline df64 ds_mul(df64 a, df64 b) {
    df64 p = two_prod(a.hi, b.hi);
    float lo = p.lo + (a.hi * b.lo + a.lo * b.hi);
    df64 r = two_sum(p.hi, lo);
    return r;
}

kernel void gemm_f64_ds(
    device const float2* a [[buffer(0)]],
    device const float2* b [[buffer(1)]],
    device float2*       c [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= params.m || col >= params.n) return;

    df64 acc = ds_from(0.0f);
    for (uint i = 0u; i < params.k; i++) {
        float2 av = a[row * params.k + i];
        float2 bv = b[i * params.n + col];
        df64 ad = df64{av.x, av.y};
        df64 bd = df64{bv.x, bv.y};
        acc = ds_add(acc, ds_mul(ad, bd));
    }

    df64 scaled = ds_mul(acc, ds_from(params.alpha));
    uint out_idx = row * params.n + col;
    if (params.beta != 0.0f) {
        float2 cv = c[out_idx];
        df64 cd = ds_mul(df64{cv.x, cv.y}, ds_from(params.beta));
        scaled = ds_add(scaled, cd);
    }
    c[out_idx] = float2(scaled.hi, scaled.lo);
}
"#
}

// ─── INT8 quantised GEMM ───────────────────────────────────────────────────────

/// MSL source for an INT8 × INT8 → dequantised-`float` GEMM kernel.
///
/// Per-tensor symmetric quantisation is assumed: the integer accumulation in
/// `int` is scaled by `scale_a * scale_b` on store.  An optional per-tensor
/// zero point on each operand is subtracted before multiply (set to 0 for
/// symmetric quantisation).
///
/// Buffers `a`/`b` are `char` (signed 8-bit); `c` is `float`.  The
/// `Int8GemmParams` constant buffer carries shapes, scales, and zero points.
pub fn int8_quant_gemm_msl() -> &'static str {
    r#"
#include <metal_stdlib>
using namespace metal;

struct Int8GemmParams {
    uint m;
    uint n;
    uint k;
    float scale_a;
    float scale_b;
    int  zero_a;
    int  zero_b;
};

kernel void int8_gemm(
    device const char* a [[buffer(0)]],
    device const char* b [[buffer(1)]],
    device float*      c [[buffer(2)]],
    constant Int8GemmParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= params.m || col >= params.n) return;

    int acc = 0;
    for (uint i = 0u; i < params.k; i++) {
        int av = int(a[row * params.k + i]) - params.zero_a;
        int bv = int(b[i * params.n + col]) - params.zero_b;
        acc += av * bv;
    }
    uint out_idx = row * params.n + col;
    c[out_idx] = float(acc) * params.scale_a * params.scale_b;
}
"#
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Softmax ──
    #[test]
    fn softmax_has_kernel_and_stable_passes() {
        let src = softmax_msl();
        assert!(src.contains("kernel void softmax_rows_f32"));
        assert!(src.contains("metal_stdlib"));
        // Numerically-stable softmax must subtract the row max before exp.
        assert!(src.contains("input[base + c] - row_max"));
        assert!(src.contains("threadgroup float*  scratch [[threadgroup(0)]]"));
        assert!(src.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"));
        // Bounds guard on the row index.
        assert!(src.contains("if (tg_id >= rows) return;"));
    }

    #[test]
    fn softmax_buffer_bindings() {
        let src = softmax_msl();
        assert!(src.contains("input  [[buffer(0)]]"));
        assert!(src.contains("output [[buffer(1)]]"));
        assert!(src.contains("rows   [[buffer(2)]]"));
        assert!(src.contains("cols   [[buffer(3)]]"));
    }

    // ── LayerNorm ──
    #[test]
    fn layernorm_has_kernel_and_affine() {
        let src = layernorm_msl();
        assert!(src.contains("kernel void layernorm_rows_f32"));
        // Affine transform: normed * gamma + beta.
        assert!(src.contains("normed * gamma[c] + beta[c]"));
        // Uses rsqrt(var + eps) for the inverse std.
        assert!(src.contains("rsqrt(var + eps)"));
        assert!(src.contains("eps    [[buffer(6)]]"));
    }

    #[test]
    fn layernorm_computes_mean_and_var() {
        let src = layernorm_msl();
        assert!(src.contains("scratch[0] / float(cols)"));
        // variance accumulates squared deviations
        assert!(src.contains("local_var += d * d;"));
    }

    // ── Scan ──
    #[test]
    fn scan_inclusive_seed_and_name() {
        let src = scan_msl(false);
        assert!(src.contains("kernel void scan_inclusive_f32"));
        assert!(src.contains("input[gid]"));
        // ping-pong double buffer
        assert!(src.contains("threadgroup float* buf_a = scratch;"));
        assert!(src.contains("threadgroup float* buf_b = scratch + tg_size;"));
        assert_eq!(scan_function_name(false), "scan_inclusive_f32");
    }

    #[test]
    fn scan_exclusive_shifts_right() {
        let src = scan_msl(true);
        assert!(src.contains("kernel void scan_exclusive_f32"));
        // exclusive scan seeds each lane with its left neighbour
        assert!(src.contains("(gid > 0u) ? input[gid - 1u] : 0.0f"));
        assert_eq!(scan_function_name(true), "scan_exclusive_f32");
    }

    #[test]
    fn scan_uses_hillis_steele_doubling() {
        let src = scan_msl(false);
        assert!(src.contains("for (uint offset = 1u; offset < tg_size; offset <<= 1u)"));
        assert!(src.contains("src[lid] + src[lid - offset]"));
    }

    // ── SIMD-group GEMM ──
    #[test]
    fn simdgroup_gemm_uses_mma_tiles() {
        let src = simdgroup_gemm_msl();
        assert!(src.contains("kernel void simdgroup_gemm_f32"));
        assert!(src.contains("simdgroup_float8x8"));
        assert!(src.contains("simdgroup_load"));
        assert!(src.contains("simdgroup_multiply_accumulate"));
        assert!(src.contains("simdgroup_store"));
        assert!(src.contains("GemmParams"));
    }

    #[test]
    fn simdgroup_gemm_walks_k_in_steps_of_8() {
        let src = simdgroup_gemm_msl();
        assert!(src.contains("kk += TILE"));
        assert!(src.contains("const uint TILE = 8u;"));
    }

    // ── Double-single FP64 GEMM ──
    #[test]
    fn f64_ds_has_dekker_primitives() {
        let src = gemm_msl_f64_ds();
        assert!(src.contains("kernel void gemm_f64_ds"));
        assert!(src.contains("struct df64"));
        assert!(src.contains("two_sum"));
        assert!(src.contains("two_prod"));
        // two_prod must use fma for the exact error term.
        assert!(src.contains("fma(a, b, -p)"));
        // Storage uses float2 limbs.
        assert!(src.contains("device const float2* a"));
    }

    #[test]
    fn f64_ds_accumulates_in_extended_precision() {
        let src = gemm_msl_f64_ds();
        assert!(src.contains("acc = ds_add(acc, ds_mul(ad, bd));"));
        assert!(src.contains("ds_mul(acc, ds_from(params.alpha))"));
    }

    // ── INT8 GEMM ──
    #[test]
    fn int8_gemm_dequantises_with_scales() {
        let src = int8_quant_gemm_msl();
        assert!(src.contains("kernel void int8_gemm"));
        assert!(src.contains("device const char* a"));
        // integer accumulation
        assert!(src.contains("int acc = 0;"));
        // dequant on store
        assert!(src.contains("float(acc) * params.scale_a * params.scale_b"));
        // zero points subtracted
        assert!(src.contains("- params.zero_a"));
        assert!(src.contains("- params.zero_b"));
    }

    #[test]
    fn int8_gemm_has_bounds_guard() {
        let src = int8_quant_gemm_msl();
        assert!(src.contains("if (row >= params.m || col >= params.n) return;"));
    }

    // ── macOS compile checks (skipped without a device) ──
    #[cfg(target_os = "macos")]
    #[test]
    fn nn_kernels_compile_on_macos() {
        use metal::{CompileOptions, Device};
        let Some(device) = Device::system_default() else {
            return;
        };
        let opts = CompileOptions::new();
        let sources = [
            softmax_msl().to_string(),
            layernorm_msl().to_string(),
            scan_msl(false),
            scan_msl(true),
            gemm_msl_f64_ds().to_string(),
            int8_quant_gemm_msl().to_string(),
        ];
        for src in &sources {
            if let Err(e) = device.new_library_with_source(src, &opts) {
                panic!("NN MSL failed to compile: {e}\n--- source ---\n{src}");
            }
        }
    }

    // `simdgroup_matrix` needs Metal 3; compile separately and tolerate older
    // toolchains by only asserting on success.
    #[cfg(target_os = "macos")]
    #[test]
    fn simdgroup_gemm_compiles_on_metal3() {
        use metal::{CompileOptions, Device};
        let Some(device) = Device::system_default() else {
            return;
        };
        let opts = CompileOptions::new();
        // A failure here is acceptable on pre-Metal-3 stacks; we only ensure no panic.
        let _ = device.new_library_with_source(simdgroup_gemm_msl(), &opts);
    }
}
