# oxicuda-distill

Knowledge distillation primitives -- logit, feature, relation, attention, online, and data-free methods in pure Rust.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-distill` implements the major knowledge-distillation techniques used
to compress, transfer, or co-train deep networks. It is organised by where
the distillation signal is taken from -- output logits, intermediate
features, pairwise relations, attention maps -- and by who supplies the
teacher (offline, online, born-again, or no teacher at all in the data-free
setting).

GPU PTX kernel templates are generated and launched entirely from Rust via
the OxiCUDA driver stack, with no C/CUDA toolchain at build time. All loss
functions, similarity primitives, and normalisation utilities are
implemented in plain Rust and operate on host-side `f32` slices, so they can
be used both as building blocks for GPU kernels and as standalone host code
in tests and training loops.

Logit-level coverage includes Hinton soft-label KD with temperature scaling,
decoupled KD (TCKD/NCKD), DIST (Pearson-correlation based inter-/intra-class
relations), and adaptive temperature scheduling.

Feature-level methods include FitNets hints with regressor projection, AT
(attention transfer), PKT (probabilistic knowledge transfer), MGD masked
generative distillation, and TinyBert-style attention / hidden / embedding
matching. Relation-based methods include RKD (distance and angle), CRD
(contrastive representation distillation) with EMA memory bank, and CC
(Gram-matrix correlation congruence).

Online and self methods include DML (deep mutual learning), BYOT (be your
own teacher), and EMA self-distillation. Data-free methods include DAFL
generators and ZSKD (zero-shot KD) via class impressions. Born-again chains
include BAN, TAS, and progressive distillation. Diagnostic metrics cover
top-k agreement, KL/JS divergence, and compression accounting.

## Modules

| Module | Description |
|--------|-------------|
| `logit` | Hinton KD, decoupled KD (DKD/TCKD/NCKD), DIST, adaptive temperature, SKD |
| `feature` | FitNets, AT, PKT, MGD, CRD-multi, self-KD, projection-free, TinyBert-style |
| `relation` | RKD, CRD with memory bank, CC (Gram matrix), graph distillation |
| `attention` | Attention transfer, multi-head attention, value-only (MiniLM-style) |
| `online` | DML, BYOT, EMA self-distillation |
| `born_again` | BAN, TAS, progressive distillation |
| `data_free` | DAFL, ZSKD |
| `metrics` | Top-k agreement, divergence (KL/JS), compression metrics |
| `handle` | `DistillHandle`, `SmVersion`, `LcgRng` (MMIX LCG) |
| `error` | `DistillError` / `DistillResult` |
| `ptx_kernels` | GPU PTX kernel templates per SM target |

## Quick Start

```rust,no_run
use oxicuda_distill::logit::hinton_kd::{HintonKdConfig, kd_loss};

fn main() -> oxicuda_distill::error::DistillResult<()> {
    // Student and teacher logits over C classes for a single example.
    let student_logits: Vec<f32> = unimplemented!();
    let teacher_logits: Vec<f32> = unimplemented!();
    let label: usize = unimplemented!();

    // Soften with temperature T = 4 and mix soft (alpha) and hard (1 - alpha) losses.
    let cfg = HintonKdConfig {
        temperature: 4.0,
        alpha: 0.7,
    };
    let _loss = kd_loss(&student_logits, &teacher_logits, label, &cfg)?;
    Ok(())
}
```

## Status

**Alpha** -- 11,889 SLoC, 447 passing tests. API may evolve before v1.0.

## License

Apache-2.0
