use crate::error::{MetaError, MetaResult};
use crate::handle::LcgRng;

#[derive(Debug, Clone)]
pub struct CnpConfig {
    pub x_dim: usize,
    pub y_dim: usize,
    pub r_dim: usize,
    pub encoder_hidden: usize,
    pub decoder_hidden: usize,
}

fn validate_config(cfg: &CnpConfig) -> MetaResult<()> {
    if cfg.x_dim == 0 {
        return Err(MetaError::InvalidEpisodeConfig {
            msg: "x_dim must be > 0".into(),
        });
    }
    if cfg.y_dim == 0 {
        return Err(MetaError::InvalidEpisodeConfig {
            msg: "y_dim must be > 0".into(),
        });
    }
    if cfg.r_dim == 0 {
        return Err(MetaError::InvalidEpisodeConfig {
            msg: "r_dim must be > 0".into(),
        });
    }
    if cfg.encoder_hidden == 0 {
        return Err(MetaError::InvalidEpisodeConfig {
            msg: "encoder_hidden must be > 0".into(),
        });
    }
    if cfg.decoder_hidden == 0 {
        return Err(MetaError::InvalidEpisodeConfig {
            msg: "decoder_hidden must be > 0".into(),
        });
    }
    Ok(())
}

/// Kaiming uniform init: W ~ Uniform(-sqrt(6/fan_in), +sqrt(6/fan_in)), zero biases.
fn kaiming_uniform_layer(fan_in: usize, out_dim: usize, rng: &mut LcgRng) -> (Vec<f32>, Vec<f32>) {
    let limit = (6.0_f32 / fan_in as f32).sqrt();
    let w_len = out_dim * fan_in;
    let mut w = vec![0.0_f32; w_len];
    for v in w.iter_mut() {
        *v = (rng.next_f32() * 2.0 - 1.0) * limit;
    }
    let b = vec![0.0_f32; out_dim];
    (w, b)
}

/// Two-layer MLP forward: ReLU(W1@x + b1), W2@h + b2.
/// W1: hidden_size × in_dim (row-major), W2: out_dim × hidden_size (row-major).
fn mlp_two_layer(
    input: &[f32],
    in_dim: usize,
    w1: &[f32],
    b1: &[f32],
    hidden_size: usize,
    w2: &[f32],
    b2: &[f32],
    out_dim: usize,
) -> Vec<f32> {
    // Layer 1: hidden = ReLU(W1 @ input + b1)
    let mut hidden = vec![0.0_f32; hidden_size];
    for (i, (h, &bi)) in hidden.iter_mut().zip(b1.iter()).enumerate() {
        let row = &w1[i * in_dim..(i + 1) * in_dim];
        *h = row
            .iter()
            .zip(input.iter())
            .map(|(&wi, &xi)| wi * xi)
            .sum::<f32>()
            + bi;
        if *h < 0.0 {
            *h = 0.0;
        }
    }

    // Layer 2: out = W2 @ hidden + b2
    let mut out = vec![0.0_f32; out_dim];
    for (k, (o, &bk)) in out.iter_mut().zip(b2.iter()).enumerate() {
        let row = &w2[k * hidden_size..(k + 1) * hidden_size];
        *o = row
            .iter()
            .zip(hidden.iter())
            .map(|(&wi, &hi)| wi * hi)
            .sum::<f32>()
            + bk;
    }
    out
}

/// Two-layer MLP encoder: [x_dim + y_dim] → encoder_hidden → r_dim
/// Architecture: `ReLU(W1@[x,y]+b1)`, `W2@h+b2`
#[derive(Debug, Clone)]
pub struct CnpEncoder {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub x_dim: usize,
    pub y_dim: usize,
    pub r_dim: usize,
    pub encoder_hidden: usize,
}

impl CnpEncoder {
    pub fn new(cfg: &CnpConfig, rng: &mut LcgRng) -> MetaResult<Self> {
        validate_config(cfg)?;
        let in_dim = cfg.x_dim + cfg.y_dim;
        let (w1, b1) = kaiming_uniform_layer(in_dim, cfg.encoder_hidden, rng);
        let (w2, b2) = kaiming_uniform_layer(cfg.encoder_hidden, cfg.r_dim, rng);
        Ok(Self {
            w1,
            b1,
            w2,
            b2,
            x_dim: cfg.x_dim,
            y_dim: cfg.y_dim,
            r_dim: cfg.r_dim,
            encoder_hidden: cfg.encoder_hidden,
        })
    }

    /// Encode single (x, y) pair → r (length r_dim).
    pub fn encode_pair(&self, x: &[f32], y: &[f32]) -> MetaResult<Vec<f32>> {
        if x.len() != self.x_dim {
            return Err(MetaError::DimensionMismatch {
                expected: self.x_dim,
                got: x.len(),
            });
        }
        if y.len() != self.y_dim {
            return Err(MetaError::DimensionMismatch {
                expected: self.y_dim,
                got: y.len(),
            });
        }

        let mut input = Vec::with_capacity(self.x_dim + self.y_dim);
        input.extend_from_slice(x);
        input.extend_from_slice(y);

        let r = mlp_two_layer(
            &input,
            self.x_dim + self.y_dim,
            &self.w1,
            &self.b1,
            self.encoder_hidden,
            &self.w2,
            &self.b2,
            self.r_dim,
        );

        Ok(r)
    }

    /// Encode context set: xs [n_ctx × x_dim], ys [n_ctx × y_dim].
    /// Returns `r = mean(encode_pair(xs[i], ys[i]))` for i in `0..n_ctx`.
    pub fn encode_context(&self, xs: &[f32], ys: &[f32], n_ctx: usize) -> MetaResult<Vec<f32>> {
        if n_ctx == 0 {
            return Err(MetaError::EmptySupport);
        }
        if xs.len() != n_ctx * self.x_dim {
            return Err(MetaError::DimensionMismatch {
                expected: n_ctx * self.x_dim,
                got: xs.len(),
            });
        }
        if ys.len() != n_ctx * self.y_dim {
            return Err(MetaError::DimensionMismatch {
                expected: n_ctx * self.y_dim,
                got: ys.len(),
            });
        }

        let mut r_sum = vec![0.0_f32; self.r_dim];
        for i in 0..n_ctx {
            let x_i = &xs[i * self.x_dim..(i + 1) * self.x_dim];
            let y_i = &ys[i * self.y_dim..(i + 1) * self.y_dim];
            let r_i = self.encode_pair(x_i, y_i)?;
            for (s, &v) in r_sum.iter_mut().zip(r_i.iter()) {
                *s += v;
            }
        }

        let n_ctx_f = n_ctx as f32;
        for s in r_sum.iter_mut() {
            *s /= n_ctx_f;
        }

        Ok(r_sum)
    }
}

/// Two-layer MLP decoder: [x_dim + r_dim] → decoder_hidden → 2*y_dim (mean + log_sigma)
/// Architecture: `ReLU(W1@[x_tgt,r]+b1)`, `W2@h+b2`
#[derive(Debug, Clone)]
pub struct CnpDecoder {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub x_dim: usize,
    pub r_dim: usize,
    pub y_dim: usize,
    pub decoder_hidden: usize,
}

impl CnpDecoder {
    pub fn new(cfg: &CnpConfig, rng: &mut LcgRng) -> MetaResult<Self> {
        validate_config(cfg)?;
        let in_dim = cfg.x_dim + cfg.r_dim;
        let out_dim = 2 * cfg.y_dim;
        let (w1, b1) = kaiming_uniform_layer(in_dim, cfg.decoder_hidden, rng);
        let (w2, b2) = kaiming_uniform_layer(cfg.decoder_hidden, out_dim, rng);
        Ok(Self {
            w1,
            b1,
            w2,
            b2,
            x_dim: cfg.x_dim,
            r_dim: cfg.r_dim,
            y_dim: cfg.y_dim,
            decoder_hidden: cfg.decoder_hidden,
        })
    }

    /// Decode: input = `concat(x_target [x_dim], r [r_dim])` → `(mu [y_dim], log_sigma [y_dim])`.
    pub fn decode(&self, x_target: &[f32], r: &[f32]) -> MetaResult<(Vec<f32>, Vec<f32>)> {
        if x_target.len() != self.x_dim {
            return Err(MetaError::DimensionMismatch {
                expected: self.x_dim,
                got: x_target.len(),
            });
        }
        if r.len() != self.r_dim {
            return Err(MetaError::DimensionMismatch {
                expected: self.r_dim,
                got: r.len(),
            });
        }

        let mut input = Vec::with_capacity(self.x_dim + self.r_dim);
        input.extend_from_slice(x_target);
        input.extend_from_slice(r);

        let out = mlp_two_layer(
            &input,
            self.x_dim + self.r_dim,
            &self.w1,
            &self.b1,
            self.decoder_hidden,
            &self.w2,
            &self.b2,
            2 * self.y_dim,
        );

        let mu = out[..self.y_dim].to_vec();
        let log_sigma = out[self.y_dim..].to_vec();

        Ok((mu, log_sigma))
    }
}

#[derive(Debug, Clone)]
pub struct Cnp {
    pub encoder: CnpEncoder,
    pub decoder: CnpDecoder,
    pub cfg: CnpConfig,
}

impl Cnp {
    pub fn new(cfg: CnpConfig, rng: &mut LcgRng) -> MetaResult<Self> {
        validate_config(&cfg)?;
        let encoder = CnpEncoder::new(&cfg, rng)?;
        let decoder = CnpDecoder::new(&cfg, rng)?;
        Ok(Self {
            encoder,
            decoder,
            cfg,
        })
    }

    /// Forward pass.
    /// ctx_x: [n_ctx × x_dim], ctx_y: [n_ctx × y_dim]
    /// target_x: [n_tgt × x_dim]
    /// Returns (mu [n_tgt × y_dim], log_sigma [n_tgt × y_dim]).
    pub fn forward(
        &self,
        ctx_x: &[f32],
        ctx_y: &[f32],
        n_ctx: usize,
        target_x: &[f32],
        n_tgt: usize,
    ) -> MetaResult<(Vec<f32>, Vec<f32>)> {
        if ctx_x.len() != n_ctx * self.cfg.x_dim {
            return Err(MetaError::DimensionMismatch {
                expected: n_ctx * self.cfg.x_dim,
                got: ctx_x.len(),
            });
        }
        if ctx_y.len() != n_ctx * self.cfg.y_dim {
            return Err(MetaError::DimensionMismatch {
                expected: n_ctx * self.cfg.y_dim,
                got: ctx_y.len(),
            });
        }
        if target_x.len() != n_tgt * self.cfg.x_dim {
            return Err(MetaError::DimensionMismatch {
                expected: n_tgt * self.cfg.x_dim,
                got: target_x.len(),
            });
        }

        // Encode context → aggregate representation r
        let r = self.encoder.encode_context(ctx_x, ctx_y, n_ctx)?;

        // Decode each target
        let y_dim = self.cfg.y_dim;
        let mut mu_all = Vec::with_capacity(n_tgt * y_dim);
        let mut log_sigma_all = Vec::with_capacity(n_tgt * y_dim);

        for i in 0..n_tgt {
            let x_t = &target_x[i * self.cfg.x_dim..(i + 1) * self.cfg.x_dim];
            let (mu_i, ls_i) = self.decoder.decode(x_t, &r)?;
            mu_all.extend_from_slice(&mu_i);
            log_sigma_all.extend_from_slice(&ls_i);
        }

        Ok((mu_all, log_sigma_all))
    }

    /// Gaussian NLL loss:
    /// `L = (1/n_tgt) * Σ_i Σ_d [0.5*log(2π) + log_sigma[i,d] + 0.5*(y[i,d]-mu[i,d])²/exp(2*log_sigma[i,d])]`
    /// Clamp log_sigma to [-10, 10] for stability.
    pub fn nll_loss(&self, mu: &[f32], log_sigma: &[f32], y_target: &[f32]) -> MetaResult<f32> {
        if mu.len() != log_sigma.len() {
            return Err(MetaError::DimensionMismatch {
                expected: mu.len(),
                got: log_sigma.len(),
            });
        }
        if mu.len() != y_target.len() {
            return Err(MetaError::DimensionMismatch {
                expected: mu.len(),
                got: y_target.len(),
            });
        }

        let n = mu.len();
        if n == 0 {
            return Err(MetaError::EmptySupport);
        }

        let log2pi_half = 0.5_f32 * (2.0_f32 * std::f32::consts::PI).ln();

        let mut total = 0.0_f32;
        for i in 0..n {
            let ls = log_sigma[i].clamp(-10.0, 10.0);
            let diff = y_target[i] - mu[i];
            let var_inv = (-2.0 * ls).exp();
            total += log2pi_half + ls + 0.5 * diff * diff * var_inv;
        }

        Ok(total / n as f32)
    }

    /// Number of parameters: encoder_params + decoder_params.
    pub fn n_params(&self) -> usize {
        let enc = self.encoder.w1.len()
            + self.encoder.b1.len()
            + self.encoder.w2.len()
            + self.encoder.b2.len();
        let dec = self.decoder.w1.len()
            + self.decoder.b1.len()
            + self.decoder.w2.len()
            + self.decoder.b2.len();
        enc + dec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> CnpConfig {
        CnpConfig {
            x_dim: 1,
            y_dim: 1,
            r_dim: 4,
            encoder_hidden: 8,
            decoder_hidden: 8,
        }
    }

    #[test]
    fn encoder_pair_output_len() {
        let cfg = default_cfg();
        let mut rng = LcgRng::new(1);
        let enc = CnpEncoder::new(&cfg, &mut rng).unwrap();
        let x = vec![0.5_f32];
        let y = vec![1.0_f32];
        let r = enc.encode_pair(&x, &y).unwrap();
        assert_eq!(r.len(), cfg.r_dim);
    }

    #[test]
    fn encoder_context_output_len() {
        let cfg = default_cfg();
        let mut rng = LcgRng::new(2);
        let enc = CnpEncoder::new(&cfg, &mut rng).unwrap();
        let xs = vec![0.0_f32, 1.0_f32];
        let ys = vec![0.5_f32, 1.5_f32];
        let r = enc.encode_context(&xs, &ys, 2).unwrap();
        assert_eq!(r.len(), cfg.r_dim);
    }

    #[test]
    fn encoder_context_single() {
        let cfg = default_cfg();
        let mut rng1 = LcgRng::new(3);
        let enc = CnpEncoder::new(&cfg, &mut rng1).unwrap();
        let x = vec![0.7_f32];
        let y = vec![0.3_f32];
        let r_pair = enc.encode_pair(&x, &y).unwrap();
        let r_ctx = enc.encode_context(&x, &y, 1).unwrap();
        for (a, b) in r_pair.iter().zip(r_ctx.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn decoder_output_len() {
        let cfg = default_cfg();
        let mut rng = LcgRng::new(4);
        let dec = CnpDecoder::new(&cfg, &mut rng).unwrap();
        let x = vec![0.5_f32];
        let r = vec![0.1_f32, 0.2_f32, 0.3_f32, 0.4_f32];
        let (mu, ls) = dec.decode(&x, &r).unwrap();
        assert_eq!(mu.len(), cfg.y_dim);
        assert_eq!(ls.len(), cfg.y_dim);
    }

    #[test]
    fn cnp_new_runs() {
        let cfg = default_cfg();
        let mut rng = LcgRng::new(5);
        let cnp = Cnp::new(cfg, &mut rng);
        assert!(cnp.is_ok());
    }

    #[test]
    fn cnp_forward_mu_len() {
        let cfg = default_cfg();
        let mut rng = LcgRng::new(6);
        let cnp = Cnp::new(cfg.clone(), &mut rng).unwrap();
        let ctx_x = vec![0.0_f32, 1.0_f32];
        let ctx_y = vec![0.5_f32, 1.5_f32];
        let tgt_x = vec![0.3_f32, 0.7_f32, 0.9_f32];
        let (mu, _) = cnp.forward(&ctx_x, &ctx_y, 2, &tgt_x, 3).unwrap();
        assert_eq!(mu.len(), 3 * cfg.y_dim);
    }

    #[test]
    fn cnp_forward_log_sigma_len() {
        let cfg = default_cfg();
        let mut rng = LcgRng::new(7);
        let cnp = Cnp::new(cfg.clone(), &mut rng).unwrap();
        let ctx_x = vec![0.0_f32, 1.0_f32];
        let ctx_y = vec![0.5_f32, 1.5_f32];
        let tgt_x = vec![0.3_f32, 0.7_f32, 0.9_f32];
        let (_, log_sigma) = cnp.forward(&ctx_x, &ctx_y, 2, &tgt_x, 3).unwrap();
        assert_eq!(log_sigma.len(), 3 * cfg.y_dim);
    }

    #[test]
    fn cnp_forward_2ctx_3tgt() {
        let cfg = default_cfg();
        let mut rng = LcgRng::new(8);
        let cnp = Cnp::new(cfg.clone(), &mut rng).unwrap();
        let ctx_x = vec![0.1_f32, 0.9_f32];
        let ctx_y = vec![0.2_f32, 0.8_f32];
        let tgt_x = vec![0.3_f32, 0.5_f32, 0.7_f32];
        let result = cnp.forward(&ctx_x, &ctx_y, 2, &tgt_x, 3);
        assert!(result.is_ok());
        let (mu, ls) = result.unwrap();
        assert_eq!(mu.len(), 3);
        assert_eq!(ls.len(), 3);
    }

    #[test]
    fn nll_loss_finite() {
        let cfg = default_cfg();
        let mut rng = LcgRng::new(9);
        let cnp = Cnp::new(cfg, &mut rng).unwrap();
        let mu = vec![0.5_f32, 1.0_f32, 1.5_f32];
        let ls = vec![0.0_f32, 0.0_f32, 0.0_f32];
        let y = vec![0.6_f32, 0.9_f32, 1.4_f32];
        let loss = cnp.nll_loss(&mu, &ls, &y).unwrap();
        assert!(loss.is_finite());
    }

    #[test]
    fn nll_loss_same_pred() {
        // When mu == y_target and log_sigma very negative (tight), loss approaches 0.5*log(2π) + ls
        // With ls=0 and mu==y, loss = 0.5*log(2π) + 0 + 0 = 0.5*log(2π) ≈ 0.919
        // With small log_sigma (negative), we get smaller than with random mu
        let cfg = default_cfg();
        let mut rng = LcgRng::new(10);
        let cnp = Cnp::new(cfg, &mut rng).unwrap();
        let mu = vec![1.0_f32, 2.0_f32];
        let ls_small = vec![-3.0_f32, -3.0_f32]; // tight prediction
        let ls_large = vec![3.0_f32, 3.0_f32]; // broad prediction
        let y = vec![1.0_f32, 2.0_f32]; // matches mu exactly
        let loss_small = cnp.nll_loss(&mu, &ls_small, &y).unwrap();
        let loss_large = cnp.nll_loss(&mu, &ls_large, &y).unwrap();
        // Tighter sigma (smaller log_sigma) yields lower NLL when prediction is accurate
        assert!(loss_small < loss_large);
    }

    #[test]
    fn nll_loss_dim_mismatch() {
        let cfg = default_cfg();
        let mut rng = LcgRng::new(11);
        let cnp = Cnp::new(cfg, &mut rng).unwrap();
        let mu = vec![0.5_f32, 1.0_f32];
        let ls = vec![0.0_f32];
        let y = vec![0.6_f32, 0.9_f32];
        let result = cnp.nll_loss(&mu, &ls, &y);
        assert!(matches!(result, Err(MetaError::DimensionMismatch { .. })));
    }

    #[test]
    fn n_params_formula() {
        let cfg = CnpConfig {
            x_dim: 2,
            y_dim: 1,
            r_dim: 4,
            encoder_hidden: 8,
            decoder_hidden: 8,
        };
        let mut rng = LcgRng::new(12);
        let cnp = Cnp::new(cfg.clone(), &mut rng).unwrap();

        // Encoder: W1: enc_hidden × (x+y), b1: enc_hidden, W2: r_dim × enc_hidden, b2: r_dim
        let enc_w1 = cfg.encoder_hidden * (cfg.x_dim + cfg.y_dim);
        let enc_b1 = cfg.encoder_hidden;
        let enc_w2 = cfg.r_dim * cfg.encoder_hidden;
        let enc_b2 = cfg.r_dim;

        // Decoder: W1: dec_hidden × (x+r), b1: dec_hidden, W2: 2*y_dim × dec_hidden, b2: 2*y_dim
        let dec_w1 = cfg.decoder_hidden * (cfg.x_dim + cfg.r_dim);
        let dec_b1 = cfg.decoder_hidden;
        let dec_w2 = (2 * cfg.y_dim) * cfg.decoder_hidden;
        let dec_b2 = 2 * cfg.y_dim;

        let expected = enc_w1 + enc_b1 + enc_w2 + enc_b2 + dec_w1 + dec_b1 + dec_w2 + dec_b2;
        assert_eq!(cnp.n_params(), expected);
    }

    #[test]
    fn zero_x_dim_err() {
        let cfg = CnpConfig {
            x_dim: 0,
            y_dim: 1,
            r_dim: 4,
            encoder_hidden: 8,
            decoder_hidden: 8,
        };
        let mut rng = LcgRng::new(13);
        let result = Cnp::new(cfg, &mut rng);
        assert!(matches!(
            result,
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
    }

    #[test]
    fn zero_context_err() {
        let cfg = default_cfg();
        let mut rng = LcgRng::new(14);
        let enc = CnpEncoder::new(&cfg, &mut rng).unwrap();
        let result = enc.encode_context(&[], &[], 0);
        assert!(matches!(result, Err(MetaError::EmptySupport)));
    }

    #[test]
    fn ctx_x_dim_mismatch() {
        let cfg = default_cfg();
        let mut rng = LcgRng::new(15);
        let cnp = Cnp::new(cfg.clone(), &mut rng).unwrap();
        // ctx_x has wrong length (3 instead of 2*x_dim=2)
        let ctx_x = vec![0.0_f32, 0.5_f32, 1.0_f32];
        let ctx_y = vec![0.0_f32, 1.0_f32];
        let tgt_x = vec![0.5_f32];
        let result = cnp.forward(&ctx_x, &ctx_y, 2, &tgt_x, 1);
        assert!(matches!(result, Err(MetaError::DimensionMismatch { .. })));
    }

    #[test]
    fn target_x_dim_mismatch() {
        let cfg = default_cfg();
        let mut rng = LcgRng::new(16);
        let cnp = Cnp::new(cfg.clone(), &mut rng).unwrap();
        let ctx_x = vec![0.0_f32, 1.0_f32];
        let ctx_y = vec![0.5_f32, 1.5_f32];
        // target_x wrong length (3 instead of 2*x_dim=2)
        let tgt_x = vec![0.1_f32, 0.5_f32, 0.9_f32];
        let result = cnp.forward(&ctx_x, &ctx_y, 2, &tgt_x, 2);
        assert!(matches!(result, Err(MetaError::DimensionMismatch { .. })));
    }

    #[test]
    fn deterministic_same_seed() {
        let cfg = default_cfg();
        let mut rng1 = LcgRng::new(42);
        let mut rng2 = LcgRng::new(42);
        let cnp1 = Cnp::new(cfg.clone(), &mut rng1).unwrap();
        let cnp2 = Cnp::new(cfg, &mut rng2).unwrap();

        let ctx_x = vec![0.3_f32];
        let ctx_y = vec![0.7_f32];
        let tgt_x = vec![0.5_f32, 0.8_f32];

        let (mu1, ls1) = cnp1.forward(&ctx_x, &ctx_y, 1, &tgt_x, 2).unwrap();
        let (mu2, ls2) = cnp2.forward(&ctx_x, &ctx_y, 1, &tgt_x, 2).unwrap();

        for (a, b) in mu1.iter().zip(mu2.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
        for (a, b) in ls1.iter().zip(ls2.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn multi_dim_xy() {
        let cfg = CnpConfig {
            x_dim: 3,
            y_dim: 2,
            r_dim: 8,
            encoder_hidden: 16,
            decoder_hidden: 16,
        };
        let mut rng = LcgRng::new(99);
        let cnp = Cnp::new(cfg.clone(), &mut rng).unwrap();

        // n_ctx=2, n_tgt=3
        let ctx_x = vec![0.1_f32; 2 * cfg.x_dim];
        let ctx_y = vec![0.5_f32; 2 * cfg.y_dim];
        let tgt_x = vec![0.2_f32; 3 * cfg.x_dim];

        let result = cnp.forward(&ctx_x, &ctx_y, 2, &tgt_x, 3);
        assert!(result.is_ok());
        let (mu, ls) = result.unwrap();
        assert_eq!(mu.len(), 3 * cfg.y_dim);
        assert_eq!(ls.len(), 3 * cfg.y_dim);
    }
}
