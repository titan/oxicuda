//! On-device GPU validation for the `rnn_misc` subsystem of `oxicuda-dnn`.
//!
//! Covers the four hand-written PTX kernels emitted under `src/rnn/`:
//! * `dnn_lstm_fused_{f32,f64}` — fused 4-gate LSTM cell (one thread per
//!   `(batch, hidden)` pair).
//! * `dnn_gru_fused_{f32,f64}`  — fused 3-gate GRU cell.
//! * `dnn_copy_{f32,f64}` / `dnn_gru_copy_{f32,f64}` — trivial `dst[i]=src[i]`
//!   kernels used to extract the final hidden state after a sequence pass.
//!
//! ## Strategy
//! The fused cells are driven through the **public op API**
//! (`lstm_cell_forward` / `gru_cell_forward` / `*_sequence_forward`), which is
//! the production path (it JIT-compiles the kernel PTX, so a numeric pass also
//! proves the PTX assembles on the live device). Each is checked against an
//! independent CPU re-derivation of the standard LSTM/GRU equations.
//!
//! The copy kernels have no standalone public op, so they are launched
//! directly from their `pub(crate)` generators and checked for bit-exact
//! `dst == src`.
//!
//! ## Precision
//! The gate non-linearities are built from `ex2.approx.f32`
//! (`sigmoid(x)=1/(1+exp(-x))`, `tanh(x)=2*sigmoid(2x)-1`). The base-2 hardware
//! exponential is accurate but not bit-exact, and the `f64` path down-converts
//! the exponent argument to `f32` — so the fused-cell oracles compare with a
//! loose tolerance. The copy kernels are pure load/store and are bit-exact.

use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;

use super::*;

use crate::rnn::gru::{
    GruWeights, generate_copy_kernel_ptx_gru, generate_gru_fused_ptx, gru_cell_forward,
    gru_sequence_forward,
};
use crate::rnn::lstm::{
    LstmWeights, generate_copy_kernel_ptx, generate_lstm_fused_ptx, lstm_cell_forward,
    lstm_sequence_forward,
};

// ---------------------------------------------------------------------------
// CPU oracles (independent re-derivation of the standard cell equations)
// ---------------------------------------------------------------------------

/// Logistic sigmoid, matching the kernel's `1/(1+exp(-x))` formulation.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `tanh` via `2*sigmoid(2x)-1`, exactly the algebra the kernels use.
fn tanh_via_sigmoid(x: f64) -> f64 {
    2.0 * sigmoid(2.0 * x) - 1.0
}

/// One LSTM cell step. Gate order in the concatenated weights is `[i, f, g, o]`.
///
/// * `x`      : `[batch * input]`
/// * `h_prev` : `[batch * hidden]`
/// * `c_prev` : `[batch * hidden]`
/// * `w_x`    : `[4*hidden, input]`  row-major
/// * `w_h`    : `[4*hidden, hidden]` row-major
/// * `bias`   : `[4*hidden]`
///
/// Returns `(h_out, c_out)`, each `[batch * hidden]`.
#[allow(clippy::too_many_arguments)]
fn lstm_cell_oracle(
    batch: usize,
    hidden: usize,
    input: usize,
    x: &[f64],
    h_prev: &[f64],
    c_prev: &[f64],
    w_x: &[f64],
    w_h: &[f64],
    bias: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut h_out = vec![0.0_f64; batch * hidden];
    let mut c_out = vec![0.0_f64; batch * hidden];
    for b in 0..batch {
        for j in 0..hidden {
            // Gate accumulators seeded with bias[gate*H + j].
            let mut g = [
                bias[j],
                bias[hidden + j],
                bias[2 * hidden + j],
                bias[3 * hidden + j],
            ];
            // W_x . x
            for k in 0..input {
                let xv = x[b * input + k];
                for (gate, acc) in g.iter_mut().enumerate() {
                    let row = gate * hidden + j;
                    *acc += w_x[row * input + k] * xv;
                }
            }
            // W_h . h_prev
            for kh in 0..hidden {
                let hv = h_prev[b * hidden + kh];
                for (gate, acc) in g.iter_mut().enumerate() {
                    let row = gate * hidden + j;
                    *acc += w_h[row * hidden + kh] * hv;
                }
            }
            let i_gate = sigmoid(g[0]);
            let f_gate = sigmoid(g[1]);
            let g_gate = tanh_via_sigmoid(g[2]);
            let o_gate = sigmoid(g[3]);
            let c_new = f_gate * c_prev[b * hidden + j] + i_gate * g_gate;
            let h_new = o_gate * tanh_via_sigmoid(c_new);
            c_out[b * hidden + j] = c_new;
            h_out[b * hidden + j] = h_new;
        }
    }
    (h_out, c_out)
}

/// One GRU cell step. Gate order in the concatenated weights is `[z, r, h]`.
///
/// * `w_x`  : `[3*hidden, input]`,  `w_h` : `[3*hidden, hidden]`,
/// * `bias` : `[3*hidden]`.
///
/// Returns `h_out` of shape `[batch * hidden]`.
#[allow(clippy::too_many_arguments)]
fn gru_cell_oracle(
    batch: usize,
    hidden: usize,
    input: usize,
    x: &[f64],
    h_prev: &[f64],
    w_x: &[f64],
    w_h: &[f64],
    bias: &[f64],
) -> Vec<f64> {
    let mut h_out = vec![0.0_f64; batch * hidden];
    for b in 0..batch {
        for j in 0..hidden {
            // gate accumulators seeded with bias[gate*H + j], gates z=0,r=1,cand=2
            let mut g = [bias[j], bias[hidden + j], bias[2 * hidden + j]];
            for k in 0..input {
                let xv = x[b * input + k];
                for (gate, acc) in g.iter_mut().enumerate() {
                    let row = gate * hidden + j;
                    *acc += w_x[row * input + k] * xv;
                }
            }
            // W_h . h_prev: z & r fold into their gate accumulators; the
            // candidate's W_hh.h is kept separate (gated by r).
            let mut wh_cand = 0.0_f64;
            for kh in 0..hidden {
                let hv = h_prev[b * hidden + kh];
                for (gate, acc) in g.iter_mut().enumerate().take(2) {
                    let row = gate * hidden + j;
                    *acc += w_h[row * hidden + kh] * hv;
                }
                let row = 2 * hidden + j;
                wh_cand += w_h[row * hidden + kh] * hv;
            }
            let z = sigmoid(g[0]);
            let r = sigmoid(g[1]);
            let cand_pre = g[2] + r * wh_cand;
            let h_cand = tanh_via_sigmoid(cand_pre);
            let h_prev_v = h_prev[b * hidden + j];
            h_out[b * hidden + j] = (1.0 - z) * h_prev_v + z * h_cand;
        }
    }
    h_out
}

// ---------------------------------------------------------------------------
// Host data generation
// ---------------------------------------------------------------------------

/// Generates `n` `f32` samples in `[-r, r]` from the shared LCG.
fn gen_f32(lcg: &mut Lcg, n: usize, r: f64) -> Vec<f32> {
    (0..n).map(|_| lcg.range_f32(-r, r)).collect()
}

/// Generates `n` `f64` samples in `[-r, r]` from the shared LCG.
fn gen_f64(lcg: &mut Lcg, n: usize, r: f64) -> Vec<f64> {
    (0..n).map(|_| lcg.range_f64(-r, r)).collect()
}

/// A weight magnitude that keeps gate pre-activations in a well-conditioned
/// (non-saturated) band for a given fan-in.
fn weight_radius(input: usize, hidden: usize) -> f64 {
    0.7 / ((input + hidden) as f64).sqrt()
}

// ---------------------------------------------------------------------------
// LSTM fused cell — numeric oracle
// ---------------------------------------------------------------------------

#[test]
fn lstm_cell_forward_f32_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // Two configurations: a tiny single-block case and a multi-block case
    // (batch*hidden = 320 > block 256) to exercise the grid loop.
    for &(batch, hidden, input) in &[(2_usize, 3_usize, 4_usize), (4, 80, 8)] {
        let wr = weight_radius(input, hidden);
        let mut lcg = Lcg::new(0x15A7_u64 ^ ((hidden as u64) << 8));
        let four_h = 4 * hidden;

        let x = gen_f32(&mut lcg, batch * input, 0.6);
        let h_prev = gen_f32(&mut lcg, batch * hidden, 0.6);
        let c_prev = gen_f32(&mut lcg, batch * hidden, 0.6);
        let w_x = gen_f32(&mut lcg, four_h * input, wr);
        let w_h = gen_f32(&mut lcg, four_h * hidden, wr);
        let bias = gen_f32(&mut lcg, four_h, 0.2);

        let x_d = DeviceBuffer::from_host(&x).expect("x upload");
        let h_d = DeviceBuffer::from_host(&h_prev).expect("h upload");
        let c_d = DeviceBuffer::from_host(&c_prev).expect("c upload");
        let wx_d = DeviceBuffer::from_host(&w_x).expect("w_x upload");
        let wh_d = DeviceBuffer::from_host(&w_h).expect("w_h upload");
        let bias_d = DeviceBuffer::from_host(&bias).expect("bias upload");
        let mut h_out_d = DeviceBuffer::<f32>::zeroed(batch * hidden).expect("h_out alloc");
        let mut c_out_d = DeviceBuffer::<f32>::zeroed(batch * hidden).expect("c_out alloc");

        let weights = LstmWeights {
            w_x: &wx_d,
            w_h: &wh_d,
            bias: &bias_d,
            input_size: input,
            hidden_size: hidden,
        };
        lstm_cell_forward(
            &fx.handle,
            &weights,
            batch,
            &x_d,
            &h_d,
            &c_d,
            &mut h_out_d,
            &mut c_out_d,
        )
        .expect("lstm_cell_forward");
        fx.stream().synchronize().expect("sync");

        let mut h_gpu = vec![0.0_f32; batch * hidden];
        let mut c_gpu = vec![0.0_f32; batch * hidden];
        h_out_d.copy_to_host(&mut h_gpu).expect("h_out d2h");
        c_out_d.copy_to_host(&mut c_gpu).expect("c_out d2h");

        let (h_ref, c_ref) = lstm_cell_oracle(
            batch,
            hidden,
            input,
            &to_f64(&x),
            &to_f64(&h_prev),
            &to_f64(&c_prev),
            &to_f64(&w_x),
            &to_f64(&w_h),
            &to_f64(&bias),
        );

        assert_close_f32(&h_gpu, &to_f32(&h_ref), 2e-3, 2e-3, "lstm h_out f32");
        assert_close_f32(&c_gpu, &to_f32(&c_ref), 2e-3, 2e-3, "lstm c_out f32");
    }
}

// f64 fused LSTM cell — numeric oracle. Previously blocked by an upstream
// `oxicuda-ptx` register-allocator defect (f32 SFU scratch in an f64 kernel was
// declared `.b64` and ptxas rejected the `ex2.approx.f32` uses); that allocator
// now declares heterogeneous banks per-register, so the f64 fused kernel
// assembles and is validated here. The exp/sigmoid still run through
// `ex2.approx.f32` (down-converted), so the tolerance matches the f32 path.
#[test]
fn lstm_cell_forward_f64_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    for &(batch, hidden, input) in &[(2_usize, 3_usize, 4_usize), (4, 80, 8)] {
        let wr = weight_radius(input, hidden);
        let mut lcg = Lcg::new(0x6D_F64A_u64 ^ ((hidden as u64) << 8));
        let four_h = 4 * hidden;

        let x = gen_f64(&mut lcg, batch * input, 0.6);
        let h_prev = gen_f64(&mut lcg, batch * hidden, 0.6);
        let c_prev = gen_f64(&mut lcg, batch * hidden, 0.6);
        let w_x = gen_f64(&mut lcg, four_h * input, wr);
        let w_h = gen_f64(&mut lcg, four_h * hidden, wr);
        let bias = gen_f64(&mut lcg, four_h, 0.2);

        let x_d = DeviceBuffer::from_host(&x).expect("x upload");
        let h_d = DeviceBuffer::from_host(&h_prev).expect("h upload");
        let c_d = DeviceBuffer::from_host(&c_prev).expect("c upload");
        let wx_d = DeviceBuffer::from_host(&w_x).expect("w_x upload");
        let wh_d = DeviceBuffer::from_host(&w_h).expect("w_h upload");
        let bias_d = DeviceBuffer::from_host(&bias).expect("bias upload");
        let mut h_out_d = DeviceBuffer::<f64>::zeroed(batch * hidden).expect("h_out alloc");
        let mut c_out_d = DeviceBuffer::<f64>::zeroed(batch * hidden).expect("c_out alloc");

        let weights = LstmWeights {
            w_x: &wx_d,
            w_h: &wh_d,
            bias: &bias_d,
            input_size: input,
            hidden_size: hidden,
        };
        lstm_cell_forward(
            &fx.handle,
            &weights,
            batch,
            &x_d,
            &h_d,
            &c_d,
            &mut h_out_d,
            &mut c_out_d,
        )
        .expect("lstm_cell_forward f64");
        fx.stream().synchronize().expect("sync");

        let mut h_gpu = vec![0.0_f64; batch * hidden];
        let mut c_gpu = vec![0.0_f64; batch * hidden];
        h_out_d.copy_to_host(&mut h_gpu).expect("h_out d2h");
        c_out_d.copy_to_host(&mut c_gpu).expect("c_out d2h");

        let (h_ref, c_ref) = lstm_cell_oracle(
            batch, hidden, input, &x, &h_prev, &c_prev, &w_x, &w_h, &bias,
        );
        assert_close_f64(&h_gpu, &h_ref, 2e-3, 2e-3, "lstm h_out f64");
        assert_close_f64(&c_gpu, &c_ref, 2e-3, 2e-3, "lstm c_out f64");
    }
}

// ---------------------------------------------------------------------------
// GRU fused cell — numeric oracle
// ---------------------------------------------------------------------------

#[test]
fn gru_cell_forward_f32_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    for &(batch, hidden, input) in &[(2_usize, 3_usize, 4_usize), (3, 96, 5)] {
        let wr = weight_radius(input, hidden);
        let mut lcg = Lcg::new(0x6701_u64.wrapping_add((hidden as u64) << 7));
        let three_h = 3 * hidden;

        let x = gen_f32(&mut lcg, batch * input, 0.6);
        let h_prev = gen_f32(&mut lcg, batch * hidden, 0.6);
        let w_x = gen_f32(&mut lcg, three_h * input, wr);
        let w_h = gen_f32(&mut lcg, three_h * hidden, wr);
        let bias = gen_f32(&mut lcg, three_h, 0.2);

        let x_d = DeviceBuffer::from_host(&x).expect("x upload");
        let h_d = DeviceBuffer::from_host(&h_prev).expect("h upload");
        let wx_d = DeviceBuffer::from_host(&w_x).expect("w_x upload");
        let wh_d = DeviceBuffer::from_host(&w_h).expect("w_h upload");
        let bias_d = DeviceBuffer::from_host(&bias).expect("bias upload");
        let mut h_out_d = DeviceBuffer::<f32>::zeroed(batch * hidden).expect("h_out alloc");

        let weights = GruWeights {
            w_x: &wx_d,
            w_h: &wh_d,
            bias: &bias_d,
            input_size: input,
            hidden_size: hidden,
        };
        gru_cell_forward(&fx.handle, &weights, batch, &x_d, &h_d, &mut h_out_d)
            .expect("gru_cell_forward");
        fx.stream().synchronize().expect("sync");

        let mut h_gpu = vec![0.0_f32; batch * hidden];
        h_out_d.copy_to_host(&mut h_gpu).expect("h_out d2h");

        let h_ref = gru_cell_oracle(
            batch,
            hidden,
            input,
            &to_f64(&x),
            &to_f64(&h_prev),
            &to_f64(&w_x),
            &to_f64(&w_h),
            &to_f64(&bias),
        );
        assert_close_f32(&h_gpu, &to_f32(&h_ref), 2e-3, 2e-3, "gru h_out f32");
    }
}

// f64 fused GRU cell — numeric oracle. Enabled by the same `oxicuda-ptx`
// heterogeneous-bank allocator fix as `lstm_cell_forward_f64`.
#[test]
fn gru_cell_forward_f64_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    for &(batch, hidden, input) in &[(2_usize, 3_usize, 4_usize), (3, 96, 5)] {
        let wr = weight_radius(input, hidden);
        let mut lcg = Lcg::new(0x6701_F64A_u64.wrapping_add((hidden as u64) << 7));
        let three_h = 3 * hidden;

        let x = gen_f64(&mut lcg, batch * input, 0.6);
        let h_prev = gen_f64(&mut lcg, batch * hidden, 0.6);
        let w_x = gen_f64(&mut lcg, three_h * input, wr);
        let w_h = gen_f64(&mut lcg, three_h * hidden, wr);
        let bias = gen_f64(&mut lcg, three_h, 0.2);

        let x_d = DeviceBuffer::from_host(&x).expect("x upload");
        let h_d = DeviceBuffer::from_host(&h_prev).expect("h upload");
        let wx_d = DeviceBuffer::from_host(&w_x).expect("w_x upload");
        let wh_d = DeviceBuffer::from_host(&w_h).expect("w_h upload");
        let bias_d = DeviceBuffer::from_host(&bias).expect("bias upload");
        let mut h_out_d = DeviceBuffer::<f64>::zeroed(batch * hidden).expect("h_out alloc");

        let weights = GruWeights {
            w_x: &wx_d,
            w_h: &wh_d,
            bias: &bias_d,
            input_size: input,
            hidden_size: hidden,
        };
        gru_cell_forward(&fx.handle, &weights, batch, &x_d, &h_d, &mut h_out_d)
            .expect("gru_cell_forward f64");
        fx.stream().synchronize().expect("sync");

        let mut h_gpu = vec![0.0_f64; batch * hidden];
        h_out_d.copy_to_host(&mut h_gpu).expect("h_out d2h");

        let h_ref = gru_cell_oracle(batch, hidden, input, &x, &h_prev, &w_x, &w_h, &bias);
        assert_close_f64(&h_gpu, &h_ref, 2e-3, 2e-3, "gru h_out f64");
    }
}

// ---------------------------------------------------------------------------
// Sequence forward — exercises the ping-pong recurrence AND the copy kernel
// (dnn_copy_f32 / dnn_gru_copy_f32 extract the final hidden state).
// ---------------------------------------------------------------------------

#[test]
fn lstm_sequence_forward_f32_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // seq_len chosen >2 so the c-state ping-pong (and the final-c copy branch)
    // is exercised; the last step lands on an odd index for seq_len=4.
    let (seq_len, batch, hidden, input) = (4_usize, 2_usize, 3_usize, 4_usize);
    let wr = weight_radius(input, hidden);
    let mut lcg = Lcg::new(0x5EED5_u64);
    let four_h = 4 * hidden;
    let bh = batch * hidden;
    let bi = batch * input;

    let x = gen_f32(&mut lcg, seq_len * bi, 0.6);
    let h0 = gen_f32(&mut lcg, bh, 0.5);
    let c0 = gen_f32(&mut lcg, bh, 0.5);
    let w_x = gen_f32(&mut lcg, four_h * input, wr);
    let w_h = gen_f32(&mut lcg, four_h * hidden, wr);
    let bias = gen_f32(&mut lcg, four_h, 0.2);

    let x_d = DeviceBuffer::from_host(&x).expect("x upload");
    let h0_d = DeviceBuffer::from_host(&h0).expect("h0 upload");
    let c0_d = DeviceBuffer::from_host(&c0).expect("c0 upload");
    let wx_d = DeviceBuffer::from_host(&w_x).expect("w_x upload");
    let wh_d = DeviceBuffer::from_host(&w_h).expect("w_h upload");
    let bias_d = DeviceBuffer::from_host(&bias).expect("bias upload");
    let mut hseq_d = DeviceBuffer::<f32>::zeroed(seq_len * bh).expect("h_seq alloc");
    let mut hn_d = DeviceBuffer::<f32>::zeroed(bh).expect("h_n alloc");
    let mut cn_d = DeviceBuffer::<f32>::zeroed(bh).expect("c_n alloc");

    let weights = LstmWeights {
        w_x: &wx_d,
        w_h: &wh_d,
        bias: &bias_d,
        input_size: input,
        hidden_size: hidden,
    };
    lstm_sequence_forward(
        &fx.handle,
        &weights,
        seq_len,
        batch,
        &x_d,
        &h0_d,
        &c0_d,
        &mut hseq_d,
        &mut hn_d,
        &mut cn_d,
    )
    .expect("lstm_sequence_forward");
    fx.stream().synchronize().expect("sync");

    let mut hseq_gpu = vec![0.0_f32; seq_len * bh];
    let mut hn_gpu = vec![0.0_f32; bh];
    let mut cn_gpu = vec![0.0_f32; bh];
    hseq_d.copy_to_host(&mut hseq_gpu).expect("h_seq d2h");
    hn_d.copy_to_host(&mut hn_gpu).expect("h_n d2h");
    cn_d.copy_to_host(&mut cn_gpu).expect("c_n d2h");

    // CPU recurrence.
    let xf = to_f64(&x);
    let wxf = to_f64(&w_x);
    let whf = to_f64(&w_h);
    let bf = to_f64(&bias);
    let mut h_prev = to_f64(&h0);
    let mut c_prev = to_f64(&c0);
    let mut hseq_ref = vec![0.0_f64; seq_len * bh];
    let mut cn_ref = vec![0.0_f64; bh];
    for t in 0..seq_len {
        let xt = &xf[t * bi..(t + 1) * bi];
        let (h_t, c_t) =
            lstm_cell_oracle(batch, hidden, input, xt, &h_prev, &c_prev, &wxf, &whf, &bf);
        hseq_ref[t * bh..(t + 1) * bh].copy_from_slice(&h_t);
        h_prev = h_t;
        c_prev = c_t.clone();
        cn_ref = c_t;
    }
    let hn_ref = hseq_ref[(seq_len - 1) * bh..].to_vec();

    assert_close_f32(&hseq_gpu, &to_f32(&hseq_ref), 3e-3, 3e-3, "lstm seq h_seq");
    assert_close_f32(&hn_gpu, &to_f32(&hn_ref), 3e-3, 3e-3, "lstm seq h_n (copy)");
    assert_close_f32(&cn_gpu, &to_f32(&cn_ref), 3e-3, 3e-3, "lstm seq c_n");
}

#[test]
fn gru_sequence_forward_f32_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (seq_len, batch, hidden, input) = (5_usize, 2_usize, 3_usize, 4_usize);
    let wr = weight_radius(input, hidden);
    let mut lcg = Lcg::new(0x5EED6_u64);
    let three_h = 3 * hidden;
    let bh = batch * hidden;
    let bi = batch * input;

    let x = gen_f32(&mut lcg, seq_len * bi, 0.6);
    let h0 = gen_f32(&mut lcg, bh, 0.5);
    let w_x = gen_f32(&mut lcg, three_h * input, wr);
    let w_h = gen_f32(&mut lcg, three_h * hidden, wr);
    let bias = gen_f32(&mut lcg, three_h, 0.2);

    let x_d = DeviceBuffer::from_host(&x).expect("x upload");
    let h0_d = DeviceBuffer::from_host(&h0).expect("h0 upload");
    let wx_d = DeviceBuffer::from_host(&w_x).expect("w_x upload");
    let wh_d = DeviceBuffer::from_host(&w_h).expect("w_h upload");
    let bias_d = DeviceBuffer::from_host(&bias).expect("bias upload");
    let mut hseq_d = DeviceBuffer::<f32>::zeroed(seq_len * bh).expect("h_seq alloc");
    let mut hn_d = DeviceBuffer::<f32>::zeroed(bh).expect("h_n alloc");

    let weights = GruWeights {
        w_x: &wx_d,
        w_h: &wh_d,
        bias: &bias_d,
        input_size: input,
        hidden_size: hidden,
    };
    gru_sequence_forward(
        &fx.handle,
        &weights,
        seq_len,
        batch,
        &x_d,
        &h0_d,
        &mut hseq_d,
        &mut hn_d,
    )
    .expect("gru_sequence_forward");
    fx.stream().synchronize().expect("sync");

    let mut hseq_gpu = vec![0.0_f32; seq_len * bh];
    let mut hn_gpu = vec![0.0_f32; bh];
    hseq_d.copy_to_host(&mut hseq_gpu).expect("h_seq d2h");
    hn_d.copy_to_host(&mut hn_gpu).expect("h_n d2h");

    let xf = to_f64(&x);
    let wxf = to_f64(&w_x);
    let whf = to_f64(&w_h);
    let bf = to_f64(&bias);
    let mut h_prev = to_f64(&h0);
    let mut hseq_ref = vec![0.0_f64; seq_len * bh];
    for t in 0..seq_len {
        let xt = &xf[t * bi..(t + 1) * bi];
        let h_t = gru_cell_oracle(batch, hidden, input, xt, &h_prev, &wxf, &whf, &bf);
        hseq_ref[t * bh..(t + 1) * bh].copy_from_slice(&h_t);
        h_prev = h_t;
    }
    let hn_ref = hseq_ref[(seq_len - 1) * bh..].to_vec();

    assert_close_f32(&hseq_gpu, &to_f32(&hseq_ref), 3e-3, 3e-3, "gru seq h_seq");
    assert_close_f32(&hn_gpu, &to_f32(&hn_ref), 3e-3, 3e-3, "gru seq h_n (copy)");
}

// ---------------------------------------------------------------------------
// Copy kernels — direct PTX launch, bit-exact dst == src
// ---------------------------------------------------------------------------

/// Drives one of the `dst[i]=src[i]` copy kernels and asserts bit-exact f32.
fn check_copy_f32(fx: &GpuFixture, ptx: &str, entry: &str, n: usize) {
    let kernel: Kernel = load_kernel(ptx, entry);
    let mut lcg = Lcg::new(0xC0_u64 ^ n as u64);
    let src: Vec<f32> = (0..n).map(|_| lcg.range_f32(-100.0, 100.0)).collect();
    let src_d = DeviceBuffer::from_host(&src).expect("src upload");
    let dst_d = DeviceBuffer::<f32>::zeroed(n).expect("dst alloc");

    let grid = ceil_div(n as u32, 256);
    let params = LaunchParams::new(grid, 256_u32);
    let args = (src_d.as_device_ptr(), dst_d.as_device_ptr(), n as u32);
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("copy launch");
    fx.stream().synchronize().expect("sync");

    let mut dst = vec![0.0_f32; n];
    dst_d.copy_to_host(&mut dst).expect("dst d2h");
    assert_eq!(dst, src, "{entry}: copy must be bit-exact");
}

/// Drives one of the copy kernels and asserts bit-exact f64.
fn check_copy_f64(fx: &GpuFixture, ptx: &str, entry: &str, n: usize) {
    let kernel: Kernel = load_kernel(ptx, entry);
    let mut lcg = Lcg::new(0xD0_u64 ^ n as u64);
    let src: Vec<f64> = (0..n).map(|_| lcg.range_f64(-100.0, 100.0)).collect();
    let src_d = DeviceBuffer::from_host(&src).expect("src upload");
    let dst_d = DeviceBuffer::<f64>::zeroed(n).expect("dst alloc");

    let grid = ceil_div(n as u32, 256);
    let params = LaunchParams::new(grid, 256_u32);
    let args = (src_d.as_device_ptr(), dst_d.as_device_ptr(), n as u32);
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("copy launch");
    fx.stream().synchronize().expect("sync");

    let mut dst = vec![0.0_f64; n];
    dst_d.copy_to_host(&mut dst).expect("dst d2h");
    assert_eq!(dst, src, "{entry}: copy must be bit-exact");
}

#[test]
fn lstm_copy_kernel_f32_exact() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let ptx = generate_copy_kernel_ptx::<f32>(fx.sm).expect("gen copy f32");
    // n = 300 spans more than one 256-wide block (grid bounds check exercised).
    check_copy_f32(&fx, &ptx, "dnn_copy_f32", 300);
}

#[test]
fn lstm_copy_kernel_f64_exact() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let ptx = generate_copy_kernel_ptx::<f64>(fx.sm).expect("gen copy f64");
    check_copy_f64(&fx, &ptx, "dnn_copy_f64", 257);
}

#[test]
fn gru_copy_kernel_f32_exact() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let ptx = generate_copy_kernel_ptx_gru::<f32>(fx.sm).expect("gen gru copy f32");
    check_copy_f32(&fx, &ptx, "dnn_gru_copy_f32", 300);
}

#[test]
fn gru_copy_kernel_f64_exact() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let ptx = generate_copy_kernel_ptx_gru::<f64>(fx.sm).expect("gen gru copy f64");
    check_copy_f64(&fx, &ptx, "dnn_gru_copy_f64", 257);
}

// ---------------------------------------------------------------------------
// ptxas pre-screen — the f32 fused kernels emit both the live-device (sm_86)
// and a Hopper (sm_90) target path; confirm both assemble. The f64 fused path
// is a documented tripwire for an upstream defect (see below).
// ---------------------------------------------------------------------------

#[test]
fn lstm_fused_ptx_assembles() {
    if gpu_fixture().is_none() {
        return;
    }
    // f32 fused kernel assembles for the live target and the Hopper target.
    for sm in [SmVersion::Sm86, SmVersion::Sm90] {
        let ptx = generate_lstm_fused_ptx::<f32>(sm).expect("gen lstm f32");
        ptxas_assembles(&ptx, "lstm_fused_f32").expect("lstm f32 must assemble");
    }
    // The f64 fused kernel down-converts its sigmoid/tanh through
    // `ex2.approx.f32` (no f64 form), mixing f32 (`.b32`) scratch with f64
    // (`.b64`) registers in the shared `%f` bank. Since the `oxicuda-ptx`
    // allocator declares heterogeneous banks per-register, this now assembles
    // for sm_86 (and is numerically validated by `lstm_cell_forward_f64`).
    let ptx64 = generate_lstm_fused_ptx::<f64>(SmVersion::Sm86).expect("gen lstm f64");
    ptxas_assembles(&ptx64, "lstm_fused_f64").expect("lstm f64 fused must assemble");
}

#[test]
fn gru_fused_ptx_assembles() {
    if gpu_fixture().is_none() {
        return;
    }
    for sm in [SmVersion::Sm86, SmVersion::Sm90] {
        let ptx = generate_gru_fused_ptx::<f32>(sm).expect("gen gru f32");
        ptxas_assembles(&ptx, "gru_fused_f32").expect("gru f32 must assemble");
    }
    // The f64 fused kernel now assembles (heterogeneous-bank allocator fix);
    // numerically validated by `gru_cell_forward_f64`.
    let ptx64 = generate_gru_fused_ptx::<f64>(SmVersion::Sm86).expect("gen gru f64");
    ptxas_assembles(&ptx64, "gru_fused_f64").expect("gru f64 fused must assemble");
}

// ---------------------------------------------------------------------------
// small conversion helpers
// ---------------------------------------------------------------------------

fn to_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

fn to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}
