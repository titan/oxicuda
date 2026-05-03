use super::*;

// -- PackedFp6 pack/unpack roundtrip --

#[test]
fn packed_fp6_roundtrip() {
    let packed = PackedFp6::pack(0, 31, 63).expect("pack should succeed");
    let (v0, v1, v2) = packed.unpack();
    assert_eq!((v0, v1, v2), (0, 31, 63));
}

#[test]
fn packed_fp6_all_zeros() {
    let packed = PackedFp6::pack(0, 0, 0).expect("pack should succeed");
    let (v0, v1, v2) = packed.unpack();
    assert_eq!((v0, v1, v2), (0, 0, 0));
}

#[test]
fn packed_fp6_overflow_rejected() {
    assert!(PackedFp6::pack(64, 0, 0).is_err());
    assert!(PackedFp6::pack(0, 255, 0).is_err());
}

// -- PackedFp4 pack/unpack roundtrip --

#[test]
fn packed_fp4_roundtrip() {
    let packed = PackedFp4::pack(0, 15).expect("pack should succeed");
    let (v0, v1) = packed.unpack();
    assert_eq!((v0, v1), (0, 15));
}

#[test]
fn packed_fp4_all_values() {
    for a in 0..16u8 {
        for b in 0..16u8 {
            let packed = PackedFp4::pack(a, b).expect("pack should succeed");
            let (v0, v1) = packed.unpack();
            assert_eq!((v0, v1), (a, b));
        }
    }
}

#[test]
fn packed_fp4_overflow_rejected() {
    assert!(PackedFp4::pack(16, 0).is_err());
    assert!(PackedFp4::pack(0, 16).is_err());
}

// -- FP6 quantize/dequantize roundtrip --

#[test]
fn fp6_e3m2_roundtrip_accuracy() {
    let quantizer = Fp6Quantizer::new(Fp6Format::E3M2);
    let values = [0.0f32, 1.0, 2.0, -1.0, -2.0, 4.0];
    let packed = quantizer
        .quantize(&values)
        .expect("quantize should succeed");
    let recovered = quantizer.dequantize(&packed);

    assert_eq!(recovered.len(), values.len());
    for (orig, rec) in values.iter().zip(recovered.iter()) {
        // FP6 E3M2 has limited precision, allow reasonable tolerance
        let tol = orig.abs() * 0.35 + 0.01;
        assert!(
            (orig - rec).abs() <= tol,
            "E3M2 roundtrip: orig={orig}, recovered={rec}, tol={tol}"
        );
    }
}

#[test]
fn fp6_e2m3_roundtrip_accuracy() {
    let quantizer = Fp6Quantizer::new(Fp6Format::E2M3);
    let values = [0.0f32, 0.5, 1.0, -0.5, -1.0, 2.0];
    let packed = quantizer
        .quantize(&values)
        .expect("quantize should succeed");
    let recovered = quantizer.dequantize(&packed);

    assert_eq!(recovered.len(), values.len());
    for (orig, rec) in values.iter().zip(recovered.iter()) {
        let tol = orig.abs() * 0.35 + 0.01;
        assert!(
            (orig - rec).abs() <= tol,
            "E2M3 roundtrip: orig={orig}, recovered={rec}, tol={tol}"
        );
    }
}

// -- FP4 quantize/dequantize roundtrip --

#[test]
fn fp4_e2m1_roundtrip_accuracy() {
    let quantizer = Fp4Quantizer::new(Fp4Format::E2M1);
    let values = [0.0f32, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 6.0];
    let packed = quantizer
        .quantize(&values)
        .expect("quantize should succeed");
    let recovered = quantizer.dequantize(&packed);

    assert_eq!(recovered.len(), values.len());
    for (orig, rec) in values.iter().zip(recovered.iter()) {
        let tol = orig.abs() * 0.6 + 0.01;
        assert!(
            (orig - rec).abs() <= tol,
            "E2M1 roundtrip: orig={orig}, recovered={rec}, tol={tol}"
        );
    }
}

#[test]
fn fp4_int4_roundtrip() {
    let quantizer = Fp4Quantizer::new(Fp4Format::Int4);
    let values = [0.0f32, 1.0, -1.0, 7.0, -8.0, 3.0, -5.0, 2.0];
    let packed = quantizer
        .quantize(&values)
        .expect("quantize should succeed");
    let recovered = quantizer.dequantize(&packed);

    for (orig, rec) in values.iter().zip(recovered.iter()) {
        assert!(
            (orig - rec).abs() < 0.5,
            "INT4 roundtrip: orig={orig}, recovered={rec}"
        );
    }
}

// -- Micro-scaling quantize/dequantize --

#[test]
fn micro_scaling_fp6_roundtrip() {
    let config = MicroScalingConfig {
        block_size: 32,
        scaling_format: ScalingFormat::Fp32,
        scaling_granularity: ScalingGranularity::PerBlock,
    };
    let msq = MicroScalingQuantizer::new(config);

    // Create 33 values (>1 block) to test multi-block
    let mut values = vec![0.0f32; 33];
    for (i, v) in values.iter_mut().enumerate() {
        *v = (i as f32) * 0.5 - 8.0;
    }
    // Pad to multiple of 3 for FP6
    values.push(0.0);
    values.push(0.0);
    values.push(0.0);
    let original_len = 33;

    let (packed, scales) = msq
        .quantize_fp6(&values[..original_len], Fp6Format::E3M2)
        .expect("ms quantize should succeed");
    let recovered = msq.dequantize_fp6(&packed, &scales, Fp6Format::E3M2, original_len);

    assert_eq!(recovered.len(), original_len);
    for (orig, rec) in values.iter().take(original_len).zip(recovered.iter()) {
        let tol = orig.abs() * 0.5 + 0.5;
        assert!(
            (orig - rec).abs() <= tol,
            "MS FP6 roundtrip: orig={orig}, recovered={rec}"
        );
    }
}

// -- Tile selection --

#[test]
fn fp6_tile_blackwell_large() {
    let tile = select_fp6_tile(2048, 2048, 2048, SmVersion::Sm100);
    assert!(tile.use_tensor_core);
    assert_eq!(tile.tile_m, 256);
    assert_eq!(tile.tile_n, 256);
    assert_eq!(tile.tile_k, 96);
}

#[test]
fn fp6_tile_pre_blackwell_fallback() {
    let tile = select_fp6_tile(2048, 2048, 2048, SmVersion::Sm90);
    assert!(!tile.use_tensor_core);
    assert_eq!(tile.stages, 1);
}

#[test]
fn fp4_tile_blackwell_skinny_m() {
    let tile = select_fp4_tile(64, 512, 256, SmVersion::Sm100);
    assert!(tile.use_tensor_core);
    assert_eq!(tile.tile_m, 64);
    assert_eq!(tile.tile_n, 256);
}

#[test]
fn fp4_tile_blackwell_large() {
    let tile = select_fp4_tile(4096, 4096, 4096, SmVersion::Sm120);
    assert!(tile.use_tensor_core);
    assert_eq!(tile.tile_m, 256);
    assert_eq!(tile.tile_n, 256);
    assert_eq!(tile.tile_k, 128);
}

// -- PTX generation --

#[test]
fn fp6_gemm_ptx_valid() {
    let config = Fp6GemmConfig {
        m: 128,
        n: 128,
        k: 96,
        format: Fp6Format::E3M2,
        accumulator: SubByteAccumulator::F32,
        micro_scaling: MicroScalingConfig {
            block_size: 32,
            scaling_format: ScalingFormat::Fp32,
            scaling_granularity: ScalingGranularity::PerBlock,
        },
        sm_version: SmVersion::Sm100,
    };
    let ptx = generate_fp6_gemm_ptx(&config).expect("PTX generation should succeed");
    assert!(ptx.contains(".visible .entry fp6_e3m2_gemm_"));
    assert!(ptx.contains(".target sm_100"));
    assert!(ptx.contains("FP6_K_LOOP"));
    assert!(ptx.contains("fma.rn.f32"));
}

#[test]
fn fp4_gemm_ptx_valid() {
    let config = Fp4GemmConfig {
        m: 256,
        n: 256,
        k: 128,
        format: Fp4Format::E2M1,
        accumulator: SubByteAccumulator::F32,
        micro_scaling: MicroScalingConfig {
            block_size: 64,
            scaling_format: ScalingFormat::Fp16,
            scaling_granularity: ScalingGranularity::PerBlock,
        },
        sm_version: SmVersion::Sm100,
    };
    let ptx = generate_fp4_gemm_ptx(&config).expect("PTX generation should succeed");
    assert!(ptx.contains(".visible .entry fp4_e2m1_gemm_"));
    assert!(ptx.contains("FP4_K_LOOP"));
    assert!(ptx.contains("fma.rn.f32"));
}

#[test]
fn fp6_dequantize_ptx_valid() {
    let ptx = generate_fp6_dequantize_ptx(Fp6Format::E2M3, 64).expect("dequant PTX should succeed");
    assert!(ptx.contains("fp6_e2m3_dequantize_bs64"));
    assert!(ptx.contains("DEQUANT6_DONE"));
}

#[test]
fn fp4_dequantize_ptx_valid() {
    let ptx =
        generate_fp4_dequantize_ptx(Fp4Format::Int4, 128).expect("dequant PTX should succeed");
    assert!(ptx.contains("fp4_int4_dequantize_bs128"));
    assert!(ptx.contains("DEQUANT4_DONE"));
}

// -- Edge cases --

#[test]
fn fp6_zero_values() {
    let quantizer = Fp6Quantizer::new(Fp6Format::E3M2);
    let values = [0.0f32, 0.0, 0.0];
    let packed = quantizer.quantize(&values).expect("quantize zeros");
    let recovered = quantizer.dequantize(&packed);
    for v in &recovered {
        assert!(v.abs() < 1e-6, "zero should roundtrip: got {v}");
    }
}

#[test]
fn fp6_max_range_clamping() {
    let quantizer = Fp6Quantizer::new(Fp6Format::E3M2);
    // Values beyond FP6 E3M2 range should be clamped
    let raw = quantizer.quantize_one(100.0);
    let recovered = quantizer.dequantize_one(raw);
    assert!(
        recovered <= Fp6Format::E3M2.max_value() + 0.01,
        "should clamp to max: got {recovered}"
    );
}

#[test]
fn fp6_subnormals() {
    let quantizer = Fp6Quantizer::new(Fp6Format::E3M2);
    // Very small values should produce subnormals or zero
    let raw = quantizer.quantize_one(0.01);
    let recovered = quantizer.dequantize_one(raw);
    // Should be close to the original or clamped to nearest representable
    assert!(recovered.abs() < 1.0, "subnormal range: got {recovered}");
}

// -- Config validation --

#[test]
fn fp6_config_rejects_pre_blackwell() {
    let config = Fp6GemmConfig {
        m: 128,
        n: 128,
        k: 96,
        format: Fp6Format::E3M2,
        accumulator: SubByteAccumulator::F32,
        micro_scaling: MicroScalingConfig {
            block_size: 32,
            scaling_format: ScalingFormat::Fp32,
            scaling_granularity: ScalingGranularity::PerBlock,
        },
        sm_version: SmVersion::Sm90,
    };
    assert!(config.validate().is_err());
}

#[test]
fn fp4_config_rejects_pre_blackwell() {
    let config = Fp4GemmConfig {
        m: 128,
        n: 128,
        k: 128,
        format: Fp4Format::E2M1,
        accumulator: SubByteAccumulator::F32,
        micro_scaling: MicroScalingConfig {
            block_size: 32,
            scaling_format: ScalingFormat::Fp32,
            scaling_granularity: ScalingGranularity::PerBlock,
        },
        sm_version: SmVersion::Sm89,
    };
    assert!(config.validate().is_err());
}

#[test]
fn fp6_config_rejects_zero_dims() {
    let config = Fp6GemmConfig {
        m: 0,
        n: 128,
        k: 96,
        format: Fp6Format::E3M2,
        accumulator: SubByteAccumulator::F32,
        micro_scaling: MicroScalingConfig {
            block_size: 32,
            scaling_format: ScalingFormat::Fp32,
            scaling_granularity: ScalingGranularity::PerBlock,
        },
        sm_version: SmVersion::Sm100,
    };
    assert!(config.validate().is_err());
}

#[test]
fn invalid_block_size_rejected() {
    let config = MicroScalingConfig {
        block_size: 17,
        scaling_format: ScalingFormat::Fp32,
        scaling_granularity: ScalingGranularity::PerBlock,
    };
    assert!(config.validate().is_err());
}

#[test]
fn dequantize_ptx_invalid_block_size() {
    assert!(generate_fp6_dequantize_ptx(Fp6Format::E3M2, 17).is_err());
    assert!(generate_fp4_dequantize_ptx(Fp4Format::E2M1, 99).is_err());
}
