# oxicuda-signal TODO

GPU-accelerated signal, audio, and image processing primitives — pure-Rust replacements for cuSignal / cuFFT-DCT / cuDNN image kernels. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.6).

(C) 2026 COOLJAPAN OU (Team KitaSan)

## Implementation Status

**Actual: 12,276 SLoC across 55 files**

Vol.6 covers DCT/DWT transforms, audio analysis (STFT, mel, MFCC, spectrogram metrics),
window functions and their analysis metrics, FIR/IIR/Wiener filters, correlation
(auto, cross, GCC-PHAT, phase), and image primitives (Gaussian blur, Sobel,
morphology, NMS). Every operation ships with a CPU reference and, where it
makes sense on GPU, a PTX kernel generator. The crate is `[COMPLETE]` —
all features in the Vol.6 roadmap are implemented; outstanding work is
GPU-hardware verification.

### Completed [x]

#### Core scaffolding
- [x] `lib.rs` -- module roots + `prelude` re-exporting the most common types
- [x] `error.rs` -- `SignalError` (6 variants), `SignalResult<T>` alias
- [x] `handle.rs` -- `SignalHandle` carrying `SmVersion` + stream binding
- [x] `types.rs` -- `WaveletFamily`, `WindowType`, `SignalPrecision`, `PadMode`, `NormMode`, `StructuringElement`, `TransformDirection`
- [x] `ptx_helpers.rs` -- shared PTX preamble: `ptx_header`, `global_tid_1d`, `bounds_check`, `next_pow2`
- [x] `window.rs` -- standard window catalog + analysis metrics (coherent gain, ENBW, process gain, peak sidelobe level) + PTX window-apply kernel

#### DCT transforms (`dct/`)
- [x] `dct/dct2.rs` -- DCT-II CPU reference + twiddle PTX kernel + `Dct2Plan` + ortho scaling
- [x] `dct/dct3.rs` -- DCT-III (IDCT) CPU reference + pre-twiddle/un-permute PTX kernels
- [x] `dct/dct4.rs` -- DCT-IV CPU reference + PTX pre/post twiddle + lookup tables
- [x] `dct/mdct.rs` -- MDCT / IMDCT + sine window + KBD window + `MdctPlan`

#### DWT wavelets (`dwt/`)
- [x] `dwt/haar.rs` -- Haar forward/inverse CPU + PTX kernels
- [x] `dwt/daubechies.rs` -- Daubechies db2–db10 forward/inverse CPU, filter tables, `conv_downsample`
- [x] `dwt/sym.rs` -- Symlet sym2–sym10 forward + `sym_lowpass`
- [x] `dwt/coiflet.rs` -- Coiflet `coif_forward` / `coif_inverse` / `coif_lowpass`
- [x] `dwt/biorthogonal.rs` -- Biorthogonal `bior_forward` / `bior_inverse` / `bior_lowpass_pair`
- [x] `dwt/multilevel.rs` -- Multi-level DWT forward/inverse + `WaveletDecomposition` + soft/hard/universal denoising thresholds

#### Audio processing (`audio/`)
- [x] `audio/stft.rs` -- STFT / windowed DFT + `StftConfig` + Hann/Hamming/Blackman/Blackman-Harris/Kaiser/Bartlett/Gaussian/FlatTop/Dolph-Chebyshev windows + magnitude / power spectrogram
- [x] `audio/mel.rs` -- Mel filterbank generation + `MelFilterbankConfig` + `apply_filterbank` + `hz_to_mel` / `mel_to_hz`
- [x] `audio/mfcc.rs` -- MFCC + `MfccConfig` + delta/delta-delta coefficients
- [x] `audio/spectrogram.rs` -- spectrogram metrics: SNR, peak, LUFS, spectral centroid/rolloff/flatness, MFCC distance, chroma

#### FIR/IIR filters (`filter/`)
- [x] `filter/fir.rs` -- FIR design: lowpass/highpass/bandpass/bandstop via windowed sinc; raised cosine / RRC; direct-form apply with zero / circular / reflect / replicate padding; freq response; PTX direct-form kernel (≤ 64 taps)
- [x] `filter/iir.rs` -- IIR Biquad sections: lowpass/highpass/bandpass/peaking EQ + freq response; general-order IIR apply (Direct Form II Transposed); Butterworth pole design + SOS cascade
- [x] `filter/remez.rs` -- Parks-McClellan / Remez exchange equiripple FIR design (type-I): `remez` + `RemezBand` + lowpass/highpass/bandpass/bandstop constructors; barycentric Lagrange interpolation, alternation-theorem extremum exchange, DCT-I impulse-response recovery
- [x] `filter/wiener.rs` -- Wiener filter: spectral noise PSD estimation + gain computation + batch apply + local 1D Wiener

#### Spectral estimation (`spectral/`)
- [x] `spectral/welch.rs` -- power spectral density estimators: `periodogram`, Welch's averaged periodogram (`welch`, overlap + mean-detrend + density/spectrum scaling), Bartlett (`bartlett_psd`), sine-taper multitaper (`multitaper_psd`); one-sided Parseval-calibrated output via self-contained radix-2 FFT

#### Resampling (`resample/`)
- [x] `resample/polyphase.rs` -- rational sample-rate conversion (`resample_poly`, `resample_rate`): gcd ratio reduction, Kaiser-windowed-sinc anti-alias prototype (reusing `filter::design_lowpass`), efficient polyphase commutator (no zero-stuffing), group-delay compensation, `ceil(N·up/down)` output length

#### Correlation (`correlation/`)
- [x] `correlation/autocorr.rs` -- biased / unbiased / normalised autocorrelation, autocovariance, partial autocorrelation (PACF), Ljung-Box Q-statistic
- [x] `correlation/crosscorr.rs` -- cross-correlation, normalised correlation coefficient, convolution (linear + circular), `find_delay`, phase correlation, GCC-PHAT

#### Image processing (`image/`)
- [x] `image/gaussian_blur.rs` -- separable Gaussian blur: 1D kernel generation, H/V passes, full 2D + radius-from-sigma + PTX H/V kernels
- [x] `image/sobel.rs` -- Sobel Gx/Gy + magnitude + orientation (angle) + PTX kernels
- [x] `image/morphology.rs` -- dilate / erode / open / close / top-hat / black-hat / morphological gradient + `StructuringElement` mask generation + PTX kernels
- [x] `image/nms.rs` -- `BBox`, IoU, greedy NMS, Soft-NMS (`SoftNmsDecay`), heatmap (2D) NMS

### Future Enhancements [ ]

#### P0 — Critical (Already Implemented)
- [x] Full Daubechies family (db2–db10) — filter tables + forward/inverse
- [x] Full Symlet family (sym2–sym10) — forward path
- [x] Coiflet and Biorthogonal wavelet families
- [x] Multi-level DWT decomposition + denoising thresholds
- [x] Dolph-Chebyshev and Kaiser windows for high-dynamic-range spectral analysis
- [x] MDCT/IMDCT with both sine and KBD windows for audio codec parity (MP3/AAC/Vorbis/Opus)

#### P1 — Important (Already Implemented)
- [x] Direct-form FIR PTX kernel (≤ 64-tap fast path)
- [x] General-order IIR via Direct Form II Transposed
- [x] Butterworth SOS cascade
- [x] Parks-McClellan / Remez exchange equiripple FIR design (type-I, minimax-optimal)
- [x] Welch / Bartlett / sine-taper multitaper PSD estimation
- [x] Polyphase rational resampling (anti-aliased `up/down` sample-rate conversion)
- [x] Wiener filter (spectral + local 1D)
- [x] GCC-PHAT for time-of-arrival estimation
- [x] Spectral feature metrics (centroid, rolloff, flatness, LUFS, MFCC distance)
- [x] Chroma feature from power spectrogram

#### P2 — Nice-to-have (Already Implemented)
- [x] Morphological black-hat and top-hat
- [x] Soft-NMS and 2D heatmap NMS for keypoint / detection post-processing
- [x] Sobel angle (gradient orientation)
- [x] Padding modes (zero, circular, reflect, replicate) across FIR / correlation

#### P2 — Nice-to-Have (Algorithmic Extensions)
- [x] Savitzky-Golay polynomial smoothing (`filter/savgol.rs`) — Savitzky-Golay 1964; least-squares polynomial fitting over sliding window with arbitrary derivative order; `SavgolFilter`
- [x] Continuous Wavelet Transform (`cwt/cwt.rs`) — scalogram via convolution with Morlet/Ricker/Paul wavelets across log-scale bank; `CwtPlan`
- [x] Kalman filter + RTS smoother (`filter/kalman.rs`) — linear Kalman predict-update cycle + Rauch-Tung-Striebel backward smoother for optimal linear state estimation; `KalmanFilter`
- [ ] MP3/AAC aligned DCT basis (`dct/mp3_dct.rs`) — 36-point and 18-point MDCT variants matching ISO 11172-3 short/long block window switching; `Mp3MdctPlan`
- [x] `beamform/mvdr.rs` / `MVDR` — Minimum Variance Distortionless Response beamformer: spatial covariance matrix R estimation from multichannel snapshots; MVDR weight w = R⁻¹d / (dᴴR⁻¹d); array gain and beam-pattern metrics; `MvdrBeamformer { n_mics, freq_hz }`
- [x] `beamform/delay_and_sum.rs` — Delay-and-Sum beamformer: per-microphone fractional-sample delay via polyphase FIR interpolation; steering vector from far-field direction-of-arrival; coherent summation; `DelayAndSumBeamformer`

#### Outstanding — Hardware Verification
- [ ] (P0) All DCT/DWT/STFT/MFCC PTX kernels round-trip-verified on Linux + NVIDIA hardware
- [ ] (P1) FIR / Gaussian blur / Sobel / morphology PTX kernels benchmarked against cuSignal / cuDNN reference
- [ ] (P1) NMS PTX kernel for very large detection counts (current implementation is CPU-side for portability)
- [ ] (P2) Multi-channel STFT batch PTX kernel (per-frame parallelism)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA Driver API wrapper (libloading) | Yes (runtime FFI only) |
| oxicuda-memory | Device / host memory management | Yes |
| oxicuda-launch | Type-safe kernel launch | Yes |
| oxicuda-ptx | PTX code generation DSL | Yes |
| oxicuda-fft | FFT primitives (used by STFT, Wiener, GCC-PHAT, phase correlation) | Yes |
| num-complex | Complex number types for spectral ops | Yes |
| num-traits | Generic numeric trait bounds | Yes |
| thiserror | Error derive macros | Yes |
| half (optional, `f16`) | FP16 support | Yes |
| serde (optional, `serde`) | (De)serialisation of configs and plans | Yes |

## Quality Status

- Warnings: 0 (clippy + rustdoc clean)
- Tests: 414 passing
- unwrap() calls: 0 (production code)
- All public functions return `SignalResult<T>` for fallible paths
- macOS: compiles, PTX-generation tests run; GPU-execution tests gated behind `feature = "gpu-tests"` and return `UnsupportedPlatform`

## Functional Coverage (derived from `lib.rs`)

| # | Capability | Module | Status |
|---|------------|--------|--------|
| F1 | DCT-II / DCT-III / DCT-IV / MDCT | `dct/` | [x] |
| F2 | Haar / Daubechies / Symlet / Coiflet / Biorthogonal DWT | `dwt/` | [x] |
| F3 | Multi-level wavelet decomposition + denoising | `dwt/multilevel.rs` | [x] |
| F4 | STFT + window catalog | `audio/stft.rs`, `window.rs` | [x] |
| F5 | Mel filterbank + MFCC + delta features | `audio/mel.rs`, `audio/mfcc.rs` | [x] |
| F6 | Spectrogram metrics (SNR, LUFS, centroid, rolloff, flatness) | `audio/spectrogram.rs` | [x] |
| F7 | FIR design + apply (windowed sinc, raised cosine, RRC) | `filter/fir.rs` | [x] |
| F8 | IIR Biquad / SOS / Butterworth | `filter/iir.rs` | [x] |
| F9 | Wiener filtering (spectral + local) | `filter/wiener.rs` | [x] |
| F16 | Parks-McClellan / Remez equiripple FIR design | `filter/remez.rs` | [x] |
| F17 | Welch / Bartlett / multitaper PSD estimation | `spectral/welch.rs` | [x] |
| F18 | Polyphase rational resampling | `resample/polyphase.rs` | [x] |
| F10 | Auto- / cross-correlation, convolution | `correlation/` | [x] |
| F11 | GCC-PHAT, phase correlation | `correlation/crosscorr.rs` | [x] |
| F12 | Gaussian blur (separable) | `image/gaussian_blur.rs` | [x] |
| F13 | Sobel edge detection (Gx/Gy/mag/angle) | `image/sobel.rs` | [x] |
| F14 | Morphology (erode/dilate/open/close/top-hat/black-hat/gradient) | `image/morphology.rs` | [x] |
| F15 | NMS (greedy / soft / heatmap) | `image/nms.rs` | [x] |

## Numerical Accuracy Targets

| Operation | Precision | Acceptable Error |
|-----------|-----------|------------------|
| DCT-II round-trip (DCT3 ∘ DCT2) | FP32 | `< N · ε_machine` per element |
| DWT round-trip (inverse ∘ forward) | FP32 | `< 10 · N · ε_machine` (filter cascade) |
| STFT inverse (OLA round-trip on Hann/Hamming) | FP32 | `< 1e-5` mean abs error |
| MFCC stability (mel → DCT path) | FP32 | reproducible bit-for-bit on identical input |
| FIR direct-form apply | FP32 | `< 1e-5` vs. CPU reference |
| IIR DF-II-T apply | FP32 | `< 1e-4` (recursive accumulation) |

where ε_machine = 1.19e-7 for FP32, 2.22e-16 for FP64.

## Architecture-Specific Deepening Opportunities

### Ampere (sm_80) / Ada (sm_89)
- [x] PTX kernels emit `cp.async`-friendly memory access patterns where shared memory is used
- [ ] Verify Gaussian blur H/V passes saturate Ampere shared-memory bandwidth

### Hopper (sm_90)
- [x] PTX `ptx_header` selects appropriate `.target sm_NN` based on `SignalHandle::sm_version`
- [ ] TMA-based loads for very wide separable-blur stencils (potential P2)

## Deepening Opportunities

> Items marked `[x]` above represent API-surface coverage. The remaining gaps are
> hardware-verification and high-end GPU-architecture-specific optimisation work.

### Verification Gaps
- [x] CPU-reference parity tests for every public transform (DCT/DWT/STFT/MFCC/FIR/IIR/correlation/morphology)
- [x] PTX-generation tests exercise every `emit_*_kernel` path (string-content assertions)
- [x] Round-trip accuracy for DCT2↔DCT3, MDCT↔IMDCT, DWT forward↔inverse
- [ ] All PTX kernels verified by `cuModuleLoadData` + execution on Linux + NVIDIA
- [ ] Benchmark vs. cuSignal / scipy.signal across representative signal lengths (N ∈ {1024, 4096, 16384, 65536})
- [ ] Image-kernel benchmarks vs. cuDNN reference (Gaussian blur σ ∈ {1, 2, 4}, Sobel, morphology with 3×3 / 5×5 SE)

### Coverage
- [x] All wavelet families documented in `dwt/mod.rs`
- [x] `prelude` re-exports cover typical user pipelines
- [ ] Multi-GPU spectrogram batching (post-Vol.6 enhancement, depends on oxicuda-driver `multi_gpu`)
- [ ] FP16 fast paths for image kernels (requires `f16` feature wiring)
