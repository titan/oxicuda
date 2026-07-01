# oxicuda-evol

Evolutionary and genetic algorithms -- CMA-ES, NSGA-II/III, MOEA/D, NEAT, DE, PSO, ACO, and more in pure Rust.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-evol` is a broad black-box optimisation library covering canonical
genetic algorithms, modern evolution strategies, multi-objective evolutionary
algorithms, neuroevolution, and swarm intelligence. It is built for problems
where gradients are unavailable, expensive, or unreliable -- combinatorial
search, mixed-integer design, multi-objective trade-off curves, controller
tuning, and topology / architecture search.

GPU PTX kernel templates are generated and dispatched entirely from Rust via
the OxiCUDA driver stack, with no C/CUDA toolchain at build time. All
selection, crossover, mutation, sampling, and ranking operators are
implemented in plain Rust against the workspace `LcgRng` (MMIX LCG with the
bit-32 boolean trick), so the same code runs identically on host and GPU.

Single-objective coverage includes the canonical GA (binary, real-valued,
and permutation encodings; tournament, roulette, and rank selection;
one-point, two-point, uniform, and SBX crossover; Gaussian, polynomial,
and swap mutation), CMA-ES (vanilla, active, IPOP-restart, BIPOP-restart),
Differential Evolution (DE/rand/1, DE/best/1, jDE/SaDE adaptive), and
parallel topologies (island model with ring/star/torus, cellular GA,
master-slave, coevolution, memetic).

Multi-objective coverage includes NSGA-II, NSGA-III with reference points,
MOEA/D with Tchebycheff scalarisation, SMS-EMOA, MOPSO, R-NSGA-II, and
preference-based MOEA/D. Neuroevolution includes NEAT (innovation tracking
and speciation), HyperNEAT, and ES-HyperNEAT. Swarm methods include PSO
with inertia weight, ACO for TSP, cuckoo search, firefly algorithm, and
artificial bee colony. A BBOB-style benchmark suite (sphere, ellipsoid,
rastrigin, rosenbrock, griewank, schwefel, ackley, ZDT1/2, DTLZ1) and
hypervolume / IGD / GD / spacing metrics are bundled for evaluation.

## Modules

| Module | Description |
|--------|-------------|
| `genetic` | Canonical GA: individuals, population, selection, crossover, mutation, parallel |
| `evolution` | CMA-ES (vanilla / active / restart), Differential Evolution, coevolution, island, memetic |
| `multiobjective` | NSGA-II, NSGA-III, MOEA/D, SMS-EMOA, MOPSO, preference-based variants |
| `neuroevolution` | NEAT, HyperNEAT, ES-HyperNEAT (topology evolution, speciation) |
| `swarm` | PSO, ACO, cuckoo, firefly, artificial bee colony |
| `benchmarks` | BBOB-style test problems (sphere, rastrigin, rosenbrock, ZDT1/2, DTLZ1, ...) |
| `metrics` | Hypervolume (2D and N-D), IGD, GD, spacing, Pareto front extraction |
| `handle` | `EvolHandle`, `SmVersion`, `LcgRng` (MMIX LCG) |
| `error` | `EvolError` / `EvolResult` |
| `ptx_kernels` | GPU PTX kernel templates per SM target |

## Quick Start

```rust,no_run
use oxicuda_evol::evolution::cmaes::{CmaEsConfig, CmaEsState};
use oxicuda_evol::handle::LcgRng;

fn main() -> oxicuda_evol::EvolResult<()> {
    // Minimise sphere(x) = sum x_i^2 in 10 dimensions.
    let n_dims = 10usize;
    let cfg = CmaEsConfig::new(n_dims)?;
    let mean_init = vec![1.0_f64; n_dims];
    let mut state = CmaEsState::new(mean_init, &cfg)?;
    let mut rng = LcgRng::new(42);

    let sphere = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
    let (_best_x, _best_fit) = state.run(sphere, &cfg, &mut rng)?;
    Ok(())
}
```

## Status

**Alpha** -- 17,544 SLoC, 612 passing tests. API may evolve before v1.0.

## License

Apache-2.0
