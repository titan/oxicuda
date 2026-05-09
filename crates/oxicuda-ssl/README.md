# oxicuda-ssl

Pure-Rust self-supervised learning primitives for the OxiCUDA ecosystem.

Vol.26 of OxiCUDA. Provides the four canonical SSL families:

- **Contrastive** — SimCLR (NT-Xent), MoCo (memory-bank InfoNCE)
- **Non-contrastive** — BYOL (cosine target), Barlow Twins (cross-correlation),
  VICReg (variance + invariance + covariance)
- **Masked** — MAE (random patch dropping + reconstruction MSE)
- **Clustering** — SwAV (Sinkhorn-Knopp), DINO (centred + sharpened CE)

Plus shared infrastructure: EMA momentum encoder, MLP projection / predictor heads,
SSL-style data augmentation (color jitter, multi-crop), and PTX kernel string
generators for the per-batch numerics.

Zero C/CUDA toolchain required.
