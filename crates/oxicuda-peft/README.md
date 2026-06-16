# oxicuda-peft

Parameter-Efficient Fine-Tuning primitives -- LoRA family, adapters, prompt-based methods, sparse fine-tuning, and model merging in pure Rust.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-peft` is a pure-Rust CPU implementation of the parameter-efficient
fine-tuning (PEFT) primitives that have become standard in large-model
adaptation. It covers the full spectrum from low-rank adapters (LoRA and its
many derivatives) through prompt-based methods, bottleneck adapters,
sparse-difference learning, and model merging arithmetic.

The algorithms run on CPU; GPU PTX kernel strings are emitted by the
`ptx_kernels` module for the operations whose hot paths are amenable to a
direct kernel mapping (low-rank matmul, IA scaling, prefix expansion,
adapter forward, NF4 dequant, LoRA merge, prompt concatenation), parameterised
on SM compute capability from Turing through Blackwell.

The crate aims for breadth across the published PEFT literature: the `lora`
module alone contains roughly two dozen variants, and the `merge` module
implements both arithmetic merges and the Fisher / RegMean / AdaMerging /
Model-Soup families used in multi-task model fusion.

## Modules

| Module | Description |
|--------|-------------|
| `lora` | Low-rank adapter family: LoRA, QLoRA, AdaLoRA, DoRA, LoHa, LoKr, VeRA, PiSSA, MoLoRA, OLoRA, LoRA-FA, LoRA+, QA-LoRA, plus AWQ / GPTQ / HQQ quantisation |
| `ia3` | IA (Infused Adapter by Inhibiting and Amplifying Inner Activations) scaling vectors |
| `prefix` | Prefix-Tuning, P-Tuning v2, Prompt-Tuning, APrompt, ATTEMPT, prompt pool, SPoT |
| `adapter` | Houlsby, Pfeiffer, Parallel, Compacter (PHM), hypercomplex, LST, AdapterFusion |
| `bitfit` | BitFit -- bias-only fine-tuning |
| `diff_pruning` | Diff-Pruning with Hard Concrete L0 regularisation |
| `merge` | Linear merge, TIES, DARE-TIES, task arithmetic, Fisher merging, RegMean, AdaMerging, Model Soup |
| `metrics` | Parameter efficiency metrics and merge quality tests |
| `handle` | `PeftHandle`, `SmVersion`, `LcgRng` |
| `ptx_kernels` | 7 GPU PTX kernel strings across 6 SM versions |
| `error` | `PeftError` / `PeftResult` |

## Method Coverage

### Low-rank adapters (`lora`)
- LoRA, LoRA+, LoRA-FA -- baseline low-rank decomposition variants
- QLoRA -- 4-bit NF4-quantised base weights with LoRA delta
- AdaLoRA -- importance-driven adaptive rank allocation
- DoRA -- weight-decomposed magnitude / direction adaptation
- LoHa, LoKr -- Hadamard / Kronecker-product low-rank forms
- VeRA -- shared random projection with per-layer scaling
- PiSSA, MoLoRA, OLoRA -- variant initialisations and mixtures
- QA-LoRA -- quantisation-aware LoRA
- AWQ, GPTQ, HQQ -- post-training weight quantisers

### Prompt-based methods (`prefix`)
- Prefix-Tuning, P-Tuning v2 -- learnable key/value prefixes per layer
- Prompt-Tuning -- soft prompt tokens prepended to the input embedding
- APrompt, ATTEMPT, prompt pool, SPoT -- multi-task / transfer extensions

### Adapter modules (`adapter`)
- Houlsby, Pfeiffer, Parallel adapters -- canonical bottleneck designs
- Compacter -- parameterised hypercomplex multiplication bottleneck
- Hypercomplex, LST -- ladder side-tuning and PHM variants
- AdapterFusion -- attention-based composition of multiple adapters

### Model merging (`merge`)
- Linear merge, task arithmetic
- TIES, DARE-TIES -- magnitude trimming and rescaled drop merging
- Fisher merging, RegMean -- second-order weighted merges
- AdaMerging, Model Soup -- learned and uniform soup-style fusion

## Quick Start

```rust,no_run
use oxicuda_peft::handle::LcgRng;
use oxicuda_peft::lora::lora::{LoraConfig, LoraLinear};

fn main() {
    let mut rng = LcgRng::new(0);

    // Configure a rank-r LoRA adapter for a single linear layer.
    let cfg = LoraConfig {
        r: 8,
        alpha: 16.0,
        init_scale: 0.01,
    };

    // Wrap an (in_features, out_features) linear layer with the adapter.
    let mut lora = LoraLinear::new(768, 768, &cfg, &mut rng);

    // Forward an input vector through W + scale * B * A.
    let x: Vec<f32> = unimplemented!();
    let _y = lora.forward(&x);

    // Merge the low-rank delta into the base weight (and unmerge later).
    lora.merge_into_w();
    lora.unmerge_from_w();
}
```

## Status

**Alpha** -- 19,975 SLoC, 643 passing tests. API may evolve before v1.0.

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
