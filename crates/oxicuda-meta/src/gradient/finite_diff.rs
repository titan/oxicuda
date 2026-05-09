pub fn fd_gradient<F>(params: &[f32], f: &F, eps: f32) -> Vec<f32>
where
    F: Fn(&[f32]) -> f32,
{
    let n = params.len();
    let mut grad = vec![0.0_f32; n];
    let mut p = params.to_vec();

    for i in 0..n {
        let orig = p[i];
        p[i] = orig + eps;
        let f_plus = f(&p);
        p[i] = orig - eps;
        let f_minus = f(&p);
        p[i] = orig;
        grad[i] = (f_plus - f_minus) / (2.0 * eps);
    }

    grad
}
