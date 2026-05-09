# oxicuda-adversarial

Pure-Rust adversarial robustness primitives for the OxiCUDA ecosystem.

Vol.27 of OxiCUDA. Provides:

- **Attacks** — FGSM, PGD (L∞ / L2), MIM (Momentum Iterative), CW (Carlini-Wagner), AutoPGD (Croce 2020)
- **Defenses** — TRADES (Zhang 2019), MART (Wang 2020), randomized smoothing
  (Cohen 2019), interval bound propagation (Gowal 2018), Lipschitz-based
  certified radius
- **Threat models** — L∞ / L2 / L1 ball constraint helpers, ε-budget tracking
- **Metrics** — robust accuracy, attack success rate, certified accuracy

Plus PTX kernel string generators for per-batch numerics: FGSM step, PGD
projection, smoothing noise, certified-radius reduction, etc.

Zero C/CUDA toolchain required.
