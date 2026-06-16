//! DFAD — Data-Free Adversarial Distillation (Fang et al. 2022).
//!
//! "Data-Free Adversarial Distillation" trains a student to imitate a frozen
//! teacher **without any real data**. A generator `G` synthesises inputs and is
//! trained *adversarially*: it seeks inputs that **maximise** the disagreement
//! between teacher and student, while the student is trained to **minimise** the
//! very same disagreement. The min-max game
//!
//! ```text
//! min_S max_G  E_{z}[ d( T(G(z)), S(G(z)) ) ],   d = mean L1 over logits
//! ```
//!
//! drives the generator to surface the student's current weaknesses (a hard,
//! curriculum-style stream of synthetic examples) and the student to close them,
//! so the student progressively matches the teacher on the support the teacher
//! actually responds to. Because the teacher is a black box with no usable
//! gradient, the disagreement is back-propagated through the *student* to obtain
//! a direction in input space for the generator (the teacher output is treated
//! as a fixed local target), exactly as in the original method.
//!
//! This module is a compact, faithful CPU reference: the generator and student
//! are two-layer ReLU MLPs, the loss is the mean absolute logit difference, and
//! each [`Dfad::train_step`] performs one SGD descent step for the student and
//! one SGD ascent step for the generator.

use crate::error::{DistillError, DistillResult};
use crate::handle::LcgRng;

/// Sub-gradient sign of `x` (`+1`, `−1`, `0` at zero) — the `d|x|/dx` used by
/// the L1 disagreement.
#[inline]
fn sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// `y = W·x + b` for a row-major weight matrix `W` of shape `[out_dim × in_dim]`.
fn linear_forward(w: &[f32], b: &[f32], x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    (0..out_dim)
        .map(|o| {
            let row = &w[o * in_dim..(o + 1) * in_dim];
            let dot: f32 = row.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi).sum();
            dot + b[o]
        })
        .collect()
}

/// Back-propagate `dy` (= ∂L/∂y) through `y = W·x + b`, accumulating the weight
/// and bias gradients and returning `dx = ∂L/∂x`.
fn linear_backward(
    w: &[f32],
    x: &[f32],
    dy: &[f32],
    in_dim: usize,
    out_dim: usize,
    dw: &mut [f32],
    db: &mut [f32],
) -> Vec<f32> {
    let mut dx = vec![0.0_f32; in_dim];
    for o in 0..out_dim {
        let dyo = dy[o];
        db[o] += dyo;
        let w_row = &w[o * in_dim..(o + 1) * in_dim];
        let dw_row = &mut dw[o * in_dim..(o + 1) * in_dim];
        for ((dwi, &wi), (&xi, dxi)) in dw_row
            .iter_mut()
            .zip(w_row.iter())
            .zip(x.iter().zip(dx.iter_mut()))
        {
            *dwi += dyo * xi;
            *dxi += dyo * wi;
        }
    }
    dx
}

/// Back-propagate `dy` through `y = W·x + b` returning only `dx`, without
/// touching any parameter gradient (used to route a signal *through* a frozen
/// student into the generator).
fn linear_dx(w: &[f32], dy: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut dx = vec![0.0_f32; in_dim];
    for o in 0..out_dim {
        let dyo = dy[o];
        let w_row = &w[o * in_dim..(o + 1) * in_dim];
        for (dxi, &wi) in dx.iter_mut().zip(w_row.iter()) {
            *dxi += dyo * wi;
        }
    }
    dx
}

/// Route a gradient back through a ReLU given its pre-activation.
fn relu_backward(dh: &[f32], pre_h: &[f32]) -> Vec<f32> {
    dh.iter()
        .zip(pre_h.iter())
        .map(|(&d, &p)| if p > 0.0 { d } else { 0.0 })
        .collect()
}

/// Two-layer ReLU MLP: `in → hidden (ReLU) → out` (linear output).
#[derive(Debug, Clone)]
struct Mlp {
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

impl Mlp {
    /// He-initialised MLP (zero biases).
    fn new(in_dim: usize, hidden: usize, out_dim: usize, rng: &mut LcgRng) -> Self {
        let scale1 = if in_dim == 0 {
            1.0
        } else {
            (2.0_f32 / in_dim as f32).sqrt()
        };
        let scale2 = if hidden == 0 {
            1.0
        } else {
            (2.0_f32 / hidden as f32).sqrt()
        };
        let mut w1 = vec![0.0_f32; hidden * in_dim];
        for w in &mut w1 {
            *w = rng.next_normal() * scale1;
        }
        let mut w2 = vec![0.0_f32; out_dim * hidden];
        for w in &mut w2 {
            *w = rng.next_normal() * scale2;
        }
        Self {
            in_dim,
            hidden,
            out_dim,
            w1,
            b1: vec![0.0_f32; hidden],
            w2,
            b2: vec![0.0_f32; out_dim],
        }
    }

    /// Forward pass returning `(pre_hidden, hidden, output)` for back-prop.
    fn forward(&self, x: &[f32]) -> DistillResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        if x.len() != self.in_dim {
            return Err(DistillError::DimensionMismatch {
                expected: self.in_dim,
                got: x.len(),
            });
        }
        let pre_h = linear_forward(&self.w1, &self.b1, x, self.in_dim, self.hidden);
        let h: Vec<f32> = pre_h.iter().map(|&v| v.max(0.0)).collect();
        let out = linear_forward(&self.w2, &self.b2, &h, self.hidden, self.out_dim);
        Ok((pre_h, h, out))
    }
}

/// Zeroed parameter-gradient accumulator mirroring an [`Mlp`].
struct MlpGrad {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

impl MlpGrad {
    fn zeros(mlp: &Mlp) -> Self {
        Self {
            w1: vec![0.0_f32; mlp.w1.len()],
            b1: vec![0.0_f32; mlp.b1.len()],
            w2: vec![0.0_f32; mlp.w2.len()],
            b2: vec![0.0_f32; mlp.b2.len()],
        }
    }
}

/// Apply `param += step · grad` element-wise (descent uses a negative `step`,
/// ascent a positive one).
fn apply_step(params: &mut [f32], grads: &[f32], step: f32) {
    for (p, &g) in params.iter_mut().zip(grads.iter()) {
        *p += step * g;
    }
}

/// Layer dimensions for a [`Dfad`] generator/student pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DfadDims {
    /// Generator latent (noise) dimension.
    pub latent_dim: usize,
    /// Generator hidden width.
    pub gen_hidden: usize,
    /// Synthetic-input dimension (generator output = student input).
    pub input_dim: usize,
    /// Student hidden width.
    pub stu_hidden: usize,
    /// Number of output classes (teacher and student logit width).
    pub n_classes: usize,
}

/// Learning rates for the adversarial [`Dfad`] game.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DfadConfig {
    /// Generator (ascent) learning rate.
    pub gen_lr: f32,
    /// Student (descent) learning rate.
    pub stu_lr: f32,
}

/// Data-free adversarial distillation state: generator + student.
#[derive(Debug, Clone)]
pub struct Dfad {
    config: DfadConfig,
    dims: DfadDims,
    generator: Mlp,
    student: Mlp,
    /// Number of synthetic samples per [`Dfad::train_step`].
    pub batch_size: usize,
}

impl Dfad {
    /// Construct a new [`Dfad`] with He-initialised generator and student.
    ///
    /// # Errors
    /// - [`DistillError::InvalidConfig`] if either learning rate is `≤ 0` or
    ///   non-finite, or if any dimension is `0`.
    pub fn new(config: DfadConfig, dims: DfadDims, rng: &mut LcgRng) -> DistillResult<Self> {
        if !config.gen_lr.is_finite() || config.gen_lr <= 0.0 {
            return Err(DistillError::InvalidConfig {
                msg: format!("gen_lr must be finite and > 0, got {}", config.gen_lr),
            });
        }
        if !config.stu_lr.is_finite() || config.stu_lr <= 0.0 {
            return Err(DistillError::InvalidConfig {
                msg: format!("stu_lr must be finite and > 0, got {}", config.stu_lr),
            });
        }
        if dims.latent_dim == 0
            || dims.gen_hidden == 0
            || dims.input_dim == 0
            || dims.stu_hidden == 0
            || dims.n_classes == 0
        {
            return Err(DistillError::InvalidConfig {
                msg: "all DfadDims must be > 0".into(),
            });
        }
        let generator = Mlp::new(dims.latent_dim, dims.gen_hidden, dims.input_dim, rng);
        let student = Mlp::new(dims.input_dim, dims.stu_hidden, dims.n_classes, rng);
        Ok(Self {
            config,
            dims,
            generator,
            student,
            batch_size: 16,
        })
    }

    /// The configured dimensions.
    #[must_use]
    pub fn dims(&self) -> DfadDims {
        self.dims
    }

    /// Mean absolute (L1) disagreement between student and teacher logits.
    ///
    /// Non-negative, and exactly `0` when the two outputs coincide. The shorter
    /// length governs the mean; empty inputs give `0`.
    #[must_use]
    pub fn disagreement_loss(student_out: &[f32], teacher_out: &[f32]) -> f32 {
        let count = student_out.len().min(teacher_out.len());
        if count == 0 {
            return 0.0;
        }
        let sum: f32 = student_out
            .iter()
            .zip(teacher_out.iter())
            .map(|(&s, &t)| (s - t).abs())
            .sum();
        sum / count as f32
    }

    /// Draw `n` latent vectors of i.i.d. standard-normal noise.
    fn sample_latents(&self, rng: &mut LcgRng, n: usize) -> Vec<Vec<f32>> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let mut z = Vec::with_capacity(self.dims.latent_dim);
            for _ in 0..self.dims.latent_dim {
                z.push(rng.next_normal());
            }
            out.push(z);
        }
        out
    }

    /// Forward the generator over a set of latents, returning synthetic inputs.
    ///
    /// # Errors
    /// - [`DistillError::DimensionMismatch`] if any latent length is not
    ///   `latent_dim`.
    pub fn generate_from_latents(&self, latents: &[Vec<f32>]) -> DistillResult<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(latents.len());
        for z in latents {
            let (_pre, _h, x) = self.generator.forward(z)?;
            out.push(x);
        }
        Ok(out)
    }

    /// Sample `batch` latents and synthesise a batch of inputs.
    ///
    /// # Errors
    /// - propagates [`Dfad::generate_from_latents`].
    pub fn generate_batch(&self, rng: &mut LcgRng, batch: usize) -> DistillResult<Vec<Vec<f32>>> {
        let latents = self.sample_latents(rng, batch);
        self.generate_from_latents(&latents)
    }

    /// Evaluate the frozen teacher on `x`, checking the output width.
    fn teacher_eval(
        &self,
        teacher: &dyn Fn(&[f32]) -> Vec<f32>,
        x: &[f32],
    ) -> DistillResult<Vec<f32>> {
        let t_out = teacher(x);
        if t_out.len() != self.dims.n_classes {
            return Err(DistillError::DimensionMismatch {
                expected: self.dims.n_classes,
                got: t_out.len(),
            });
        }
        Ok(t_out)
    }

    /// Mean disagreement of the current student against the teacher over a set
    /// of (already-synthesised) inputs.
    ///
    /// # Errors
    /// - propagates the student forward and teacher-width checks.
    pub fn disagreement_on_inputs(
        &self,
        inputs: &[Vec<f32>],
        teacher: &dyn Fn(&[f32]) -> Vec<f32>,
    ) -> DistillResult<f32> {
        if inputs.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        let mut total = 0.0_f32;
        for x in inputs {
            let (_pre, _h, s_out) = self.student.forward(x)?;
            let t_out = self.teacher_eval(teacher, x)?;
            total += Self::disagreement_loss(&s_out, &t_out);
        }
        Ok(total / inputs.len() as f32)
    }

    /// One student-only **descent** step on a fixed batch of inputs.
    ///
    /// Returns the mean disagreement measured *before* the update, so a sequence
    /// of calls on the same batch yields a decreasing curve as the student
    /// learns to mimic the teacher.
    ///
    /// # Errors
    /// - [`DistillError::EmptyInput`] if `inputs` is empty.
    /// - propagates the student forward and teacher-width checks.
    pub fn student_step(
        &mut self,
        inputs: &[Vec<f32>],
        teacher: &dyn Fn(&[f32]) -> Vec<f32>,
    ) -> DistillResult<f32> {
        if inputs.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        let c = self.dims.n_classes;
        let mut grad = MlpGrad::zeros(&self.student);
        let mut total = 0.0_f32;
        for x in inputs {
            let (pre_h, h, s_out) = self.student.forward(x)?;
            let t_out = self.teacher_eval(teacher, x)?;
            total += Self::disagreement_loss(&s_out, &t_out);
            // ∂d/∂s_k = sign(s_k − t_k) / C.
            let inv_c = 1.0 / c as f32;
            let d_out: Vec<f32> = s_out
                .iter()
                .zip(t_out.iter())
                .map(|(&s, &t)| sign(s - t) * inv_c)
                .collect();
            let d_h = linear_backward(
                &self.student.w2,
                &h,
                &d_out,
                self.student.hidden,
                self.student.out_dim,
                &mut grad.w2,
                &mut grad.b2,
            );
            let d_pre = relu_backward(&d_h, &pre_h);
            let _d_x = linear_backward(
                &self.student.w1,
                x,
                &d_pre,
                self.student.in_dim,
                self.student.hidden,
                &mut grad.w1,
                &mut grad.b1,
            );
        }
        // Descent: average over batch and subtract.
        let step = -self.config.stu_lr / inputs.len() as f32;
        apply_step(&mut self.student.w1, &grad.w1, step);
        apply_step(&mut self.student.b1, &grad.b1, step);
        apply_step(&mut self.student.w2, &grad.w2, step);
        apply_step(&mut self.student.b2, &grad.b2, step);
        Ok(total / inputs.len() as f32)
    }

    /// Accumulate the generator gradient of the disagreement over one latent's
    /// forward pass, routing ∂d/∂x through the (frozen) student. Returns the
    /// sample disagreement.
    fn accumulate_generator_grad(
        &self,
        z: &[f32],
        teacher: &dyn Fn(&[f32]) -> Vec<f32>,
        grad: &mut MlpGrad,
    ) -> DistillResult<f32> {
        let c = self.dims.n_classes;
        let (pre_hg, hg, x) = self.generator.forward(z)?;
        let (pre_h, _h, s_out) = self.student.forward(&x)?;
        let t_out = self.teacher_eval(teacher, &x)?;
        let disagreement = Self::disagreement_loss(&s_out, &t_out);

        // ∂d/∂s with the teacher held fixed, then route to ∂d/∂x via the student.
        let inv_c = 1.0 / c as f32;
        let d_out: Vec<f32> = s_out
            .iter()
            .zip(t_out.iter())
            .map(|(&s, &t)| sign(s - t) * inv_c)
            .collect();
        let d_h = linear_dx(
            &self.student.w2,
            &d_out,
            self.student.hidden,
            self.student.out_dim,
        );
        let d_pre = relu_backward(&d_h, &pre_h);
        let d_x = linear_dx(
            &self.student.w1,
            &d_pre,
            self.student.in_dim,
            self.student.hidden,
        );

        // Back-prop ∂d/∂x through the generator, accumulating its gradients.
        let d_hg = linear_backward(
            &self.generator.w2,
            &hg,
            &d_x,
            self.generator.hidden,
            self.generator.out_dim,
            &mut grad.w2,
            &mut grad.b2,
        );
        let d_pre_g = relu_backward(&d_hg, &pre_hg);
        let _d_z = linear_backward(
            &self.generator.w1,
            z,
            &d_pre_g,
            self.generator.in_dim,
            self.generator.hidden,
            &mut grad.w1,
            &mut grad.b1,
        );
        Ok(disagreement)
    }

    /// One generator-only **ascent** step on a fixed batch of latents.
    ///
    /// Returns the mean disagreement measured *before* the update.
    ///
    /// # Errors
    /// - [`DistillError::EmptyInput`] if `latents` is empty.
    /// - propagates the forward and teacher-width checks.
    pub fn generator_step(
        &mut self,
        latents: &[Vec<f32>],
        teacher: &dyn Fn(&[f32]) -> Vec<f32>,
    ) -> DistillResult<f32> {
        if latents.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        let mut grad = MlpGrad::zeros(&self.generator);
        let mut total = 0.0_f32;
        for z in latents {
            total += self.accumulate_generator_grad(z, teacher, &mut grad)?;
        }
        // Ascent: average over batch and add (maximise disagreement).
        let step = self.config.gen_lr / latents.len() as f32;
        apply_step(&mut self.generator.w1, &grad.w1, step);
        apply_step(&mut self.generator.b1, &grad.b1, step);
        apply_step(&mut self.generator.w2, &grad.w2, step);
        apply_step(&mut self.generator.b2, &grad.b2, step);
        Ok(total / latents.len() as f32)
    }

    /// One full adversarial step: synthesise a batch, then take one student
    /// descent step and one generator ascent step on it.
    ///
    /// Returns the disagreement measured before the updates.
    ///
    /// # Errors
    /// - propagates the forward, teacher-width, and step checks.
    pub fn train_step(
        &mut self,
        teacher: &dyn Fn(&[f32]) -> Vec<f32>,
        rng: &mut LcgRng,
    ) -> DistillResult<f32> {
        let latents = self.sample_latents(rng, self.batch_size);
        let inputs = self.generate_from_latents(&latents)?;
        let disagreement = self.student_step(&inputs, teacher)?;
        let _ = self.generator_step(&latents, teacher)?;
        Ok(disagreement)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dims() -> DfadDims {
        DfadDims {
            latent_dim: 4,
            gen_hidden: 8,
            input_dim: 5,
            stu_hidden: 8,
            n_classes: 3,
        }
    }

    fn config() -> DfadConfig {
        DfadConfig {
            gen_lr: 0.05,
            stu_lr: 0.05,
        }
    }

    /// Fixed linear teacher `R^5 → R^3` (deterministic, gradient-free black box).
    fn linear_teacher(x: &[f32]) -> Vec<f32> {
        (0..3)
            .map(|c| {
                x.iter()
                    .enumerate()
                    .map(|(i, &xi)| xi * ((i + c) as f32 * 0.1 - 0.2))
                    .sum::<f32>()
                    + 0.05 * c as f32
            })
            .collect()
    }

    #[test]
    fn shapes_finite_and_correct() {
        let mut rng = LcgRng::new(1);
        let dfad = Dfad::new(config(), dims(), &mut rng).expect("valid");
        let batch = dfad.generate_batch(&mut rng, 6).expect("ok");
        assert_eq!(batch.len(), 6);
        for x in &batch {
            assert_eq!(x.len(), dims().input_dim);
            assert!(x.iter().all(|v| v.is_finite()), "input not finite: {x:?}");
        }
    }

    #[test]
    fn disagreement_nonneg_and_zero_when_equal() {
        let a = vec![1.0_f32, -2.0, 0.5];
        let b = vec![0.5_f32, 0.0, -1.0];
        let d = Dfad::disagreement_loss(&a, &b);
        assert!(d > 0.0 && d.is_finite(), "d={d}");
        let same = Dfad::disagreement_loss(&a, &a);
        assert!(same.abs() < 1e-7, "equal outputs must give 0, got {same}");
    }

    #[test]
    fn student_minimisation_decreases_disagreement() {
        let mut rng = LcgRng::new(42);
        let mut dfad = Dfad::new(config(), dims(), &mut rng).expect("valid");
        // Fixed generator batch.
        let inputs = dfad.generate_batch(&mut rng, 12).expect("ok");
        let start = dfad
            .disagreement_on_inputs(&inputs, &linear_teacher)
            .expect("ok");
        for _ in 0..60 {
            dfad.student_step(&inputs, &linear_teacher).expect("ok");
        }
        let end = dfad
            .disagreement_on_inputs(&inputs, &linear_teacher)
            .expect("ok");
        assert!(
            end < start,
            "student should reduce disagreement on a fixed batch: start={start} end={end}"
        );
    }

    #[test]
    fn generator_step_does_not_decrease_disagreement() {
        // Use a CONSTANT teacher so its output does not move when the generator
        // shifts the inputs; ascent on |S(x) − const| then cannot decrease.
        fn const_teacher(_x: &[f32]) -> Vec<f32> {
            vec![0.0_f32, 0.0, 0.0]
        }
        let mut rng = LcgRng::new(7);
        let mut dfad = Dfad::new(config(), dims(), &mut rng).expect("valid");
        let latents: Vec<Vec<f32>> = {
            let mut l = Vec::new();
            for _ in 0..10 {
                let mut z = Vec::new();
                for _ in 0..dims().latent_dim {
                    z.push(rng.next_normal());
                }
                l.push(z);
            }
            l
        };
        let inputs_before = dfad.generate_from_latents(&latents).expect("ok");
        let before = dfad
            .disagreement_on_inputs(&inputs_before, &const_teacher)
            .expect("ok");
        dfad.generator_step(&latents, &const_teacher).expect("ok");
        let inputs_after = dfad.generate_from_latents(&latents).expect("ok");
        let after = dfad
            .disagreement_on_inputs(&inputs_after, &const_teacher)
            .expect("ok");
        assert!(
            after >= before - 1e-5,
            "generator ascent must not decrease disagreement: before={before} after={after}"
        );
    }

    #[test]
    fn train_step_runs_and_is_finite() {
        let mut rng = LcgRng::new(99);
        let mut dfad = Dfad::new(config(), dims(), &mut rng).expect("valid");
        dfad.batch_size = 8;
        for _ in 0..5 {
            let d = dfad.train_step(&linear_teacher, &mut rng).expect("ok");
            assert!(d >= 0.0 && d.is_finite(), "disagreement={d}");
        }
    }

    #[test]
    fn invalid_lr_error() {
        let mut rng = LcgRng::new(1);
        let bad_gen = DfadConfig {
            gen_lr: 0.0,
            stu_lr: 0.05,
        };
        assert!(matches!(
            Dfad::new(bad_gen, dims(), &mut rng),
            Err(DistillError::InvalidConfig { .. })
        ));
        let bad_stu = DfadConfig {
            gen_lr: 0.05,
            stu_lr: -1.0,
        };
        assert!(matches!(
            Dfad::new(bad_stu, dims(), &mut rng),
            Err(DistillError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn invalid_dims_error() {
        let mut rng = LcgRng::new(1);
        let bad = DfadDims {
            latent_dim: 0,
            gen_hidden: 8,
            input_dim: 5,
            stu_hidden: 8,
            n_classes: 3,
        };
        assert!(matches!(
            Dfad::new(config(), bad, &mut rng),
            Err(DistillError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn teacher_width_mismatch_error() {
        fn wrong_teacher(_x: &[f32]) -> Vec<f32> {
            vec![0.0_f32, 1.0] // width 2 != n_classes 3
        }
        let mut rng = LcgRng::new(3);
        let mut dfad = Dfad::new(config(), dims(), &mut rng).expect("valid");
        let inputs = dfad.generate_batch(&mut rng, 2).expect("ok");
        assert!(matches!(
            dfad.student_step(&inputs, &wrong_teacher),
            Err(DistillError::DimensionMismatch { .. })
        ));
    }
}
