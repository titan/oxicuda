//! Conv2D forward pass — target: P1 (ResNet-50 layer3, 256-channel 14x14, 3x3 stride-1 pad-1).
//!
//! Measures the throughput of [`oxicuda_dnn::conv::conv_forward`] using the
//! ResNet-50 layer3 problem shape:
//!
//! - input  `[N=1, C=256, H=14, W=14]`
//! - filter `[K=256, C=256, R=3, S=3]`
//! - output `[N=1, K=256, P=14, Q=14]` (stride 1, pad 1)
//!
//! The bench reports **elements/sec** as throughput. Effective GFLOPS is
//! recovered offline by multiplying by the per-output FLOP cost:
//! `2 * C_in * Kh * Kw = 2 * 256 * 9 = 4608` FLOPs per output element.
//!
//! On macOS / no-GPU systems the bench function returns immediately because
//! `init()` reports `UnsupportedPlatform`. On Linux+NVIDIA the harness calls
//! the real `conv_forward` API once per `iter()`, with all device buffers
//! and tensor descriptors built outside the timing loop.

use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use oxicuda_dnn::DnnHandle;
use oxicuda_dnn::conv::conv_forward;
use oxicuda_dnn::types::{ConvolutionDescriptor, TensorDesc, TensorDescMut};
use oxicuda_driver::{Context, Device};
use oxicuda_memory::DeviceBuffer;

// ResNet-50 layer3 (Stage 3) middle 3x3 conv parameters.
const N: u32 = 1;
const C: u32 = 256;
const H: u32 = 14;
const W: u32 = 14;
const K: u32 = 256;
const KH: u32 = 3;
const KW: u32 = 3;
const STRIDE: u32 = 1;
const PAD: u32 = 1;

fn bench_conv2d_resnet50_layer3(c: &mut Criterion) {
    // Skip on hosts without an NVIDIA driver / GPU. macOS hits
    // UnsupportedPlatform; Linux without a GPU returns DeviceNotFound.
    if oxicuda_driver::init().is_err() {
        eprintln!("skip: no GPU (driver init failed)");
        return;
    }
    if !matches!(Device::count(), Ok(n) if n > 0) {
        eprintln!("skip: no GPU (Device::count <= 0)");
        return;
    }
    let device = match Device::get(0) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skip: no GPU (Device::get(0) failed)");
            return;
        }
    };
    let ctx = match Context::new(&device) {
        Ok(c) => Arc::new(c),
        Err(_) => {
            eprintln!("skip: no GPU (context creation failed)");
            return;
        }
    };
    let handle = match DnnHandle::new(&ctx) {
        Ok(h) => h,
        Err(_) => {
            eprintln!("skip: no GPU (DnnHandle init failed)");
            return;
        }
    };

    // Allocate device buffers (NCHW) for input, filter, output.
    let in_elems = (N * C * H * W) as usize;
    let filt_elems = (K * C * KH * KW) as usize;
    let out_elems = (N * K * H * W) as usize;

    let in_buf = match DeviceBuffer::<f32>::zeroed(in_elems) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (input)");
            return;
        }
    };
    let filt_buf = match DeviceBuffer::<f32>::zeroed(filt_elems) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (filter)");
            return;
        }
    };
    let mut out_buf = match DeviceBuffer::<f32>::zeroed(out_elems) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (output)");
            return;
        }
    };

    let input = match TensorDesc::<f32>::nchw(&in_buf, N, C, H, W) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skip: input tensor desc failed");
            return;
        }
    };
    let filter = match TensorDesc::<f32>::nchw(&filt_buf, K, C, KH, KW) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skip: filter tensor desc failed");
            return;
        }
    };
    let mut output = match TensorDescMut::<f32>::nchw(&mut out_buf, N, K, H, W) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skip: output tensor desc failed");
            return;
        }
    };
    let conv_desc = match ConvolutionDescriptor::conv2d(PAD, PAD, STRIDE, STRIDE, 1, 1, 1) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skip: conv descriptor failed");
            return;
        }
    };

    let work = u64::from(out_elems as u32);

    let mut group = c.benchmark_group("dnn_p1_conv2d_resnet50_layer3");
    // Throughput: number of output elements per second.
    // Multiply by 2*C*KH*KW = 2*256*9 = 4608 FLOPs/element offline for GFLOPS.
    group.throughput(Throughput::Elements(work));
    group.bench_function("oxicuda_f32_nchw_3x3_s1p1", |b| {
        b.iter(|| {
            let _ = conv_forward(&handle, &input, &filter, &mut output, &conv_desc, None);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_conv2d_resnet50_layer3);
criterion_main!(benches);
