# oxicuda-causal

Causal-inference primitives for OxiCUDA — NOTEARS, PC/GES, IPW, DML, DragonNet, causal forest, do-calculus, counterfactuals.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement.

## Features

- **Causal structure learning**: NOTEARS gradient-based DAG learning, PC algorithm for skeleton discovery and orientation, cycle-free DAG with topological sort
- **Causal effect estimation**: IPW (Inverse Probability Weighting), Double Machine Learning (DML) with cross-fitting, DragonNet (joint propensity + outcome network)
- **Non-parametric methods**: Causal forest for heterogeneous treatment effect (CATE) estimation
- **Do-calculus tools**: Backdoor admissibility checking, d-separation queries on DAGs, counterfactual prediction
- **PTX kernels**: 7 GPU kernels (partial correlation, NOTEARS loss, matrix exponential Padé, propensity logit, IPW estimator, DML residual, causal split score) × 6 SM versions

## Usage

```rust
use oxicuda_causal::{
    dag::dag::Dag,
    do_calculus::identification::backdoor_admissible,
    effect::ipw::ipw_ate,
};

// Build a causal DAG: X -> Z -> Y, X -> Y
let mut dag = Dag::new(3); // nodes: X=0, Z=2, Y=1
dag.add_edge(0, 2).unwrap();
dag.add_edge(2, 1).unwrap();
dag.add_edge(0, 1).unwrap();

// Check backdoor admissibility of X -> Y controlling for {}
let admissible = backdoor_admissible(&dag, 0, 1, &[]);
println!("Backdoor admissible (no controls): {admissible}");

// IPW average treatment effect
let y  = vec![0.5_f32, 0.8, 0.3, 0.6];
let t  = vec![1.0_f32, 1.0, 0.0, 0.0];
let pi = vec![0.7_f32, 0.7, 0.3, 0.3];
let ate = ipw_ate(&y, &t, &pi).unwrap();
println!("ATE (IPW): {ate}");
```

## Documentation

- [API Documentation](https://docs.rs/oxicuda-causal)
- [OxiCUDA Project](https://github.com/cool-japan/oxicuda)

## License

Apache-2.0 — Copyright 2026 COOLJAPAN OU (Team Kitasan)
