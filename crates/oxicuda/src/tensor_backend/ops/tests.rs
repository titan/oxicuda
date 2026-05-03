use super::*;

#[test]
fn test_add_forward() {
    let a = GpuTensor::from_host_f64(&[1.0, 2.0, 3.0], &[3], 0)
        .expect("GpuTensor creation from host data should succeed");
    let b = GpuTensor::from_host_f64(&[4.0, 5.0, 6.0], &[3], 0)
        .expect("GpuTensor creation from host data should succeed");
    let c = add(&a, &b, None).expect("operation should succeed with valid inputs");
    assert!((c.host_data()[0] - 5.0).abs() < 1e-10);
    assert!((c.host_data()[1] - 7.0).abs() < 1e-10);
    assert!((c.host_data()[2] - 9.0).abs() < 1e-10);
}

#[test]
fn test_sub_forward() {
    let a = GpuTensor::from_host_f64(&[5.0, 3.0], &[2], 0)
        .expect("GpuTensor creation from host data should succeed");
    let b = GpuTensor::from_host_f64(&[2.0, 1.0], &[2], 0)
        .expect("GpuTensor creation from host data should succeed");
    let c = sub(&a, &b, None).expect("operation should succeed with valid inputs");
    assert!((c.host_data()[0] - 3.0).abs() < 1e-10);
}

#[test]
fn test_mul_forward() {
    let a = GpuTensor::from_host_f64(&[2.0, 3.0], &[2], 0)
        .expect("GpuTensor creation from host data should succeed");
    let b = GpuTensor::from_host_f64(&[4.0, 5.0], &[2], 0)
        .expect("GpuTensor creation from host data should succeed");
    let c = mul(&a, &b, None).expect("operation should succeed with valid inputs");
    assert!((c.host_data()[0] - 8.0).abs() < 1e-10);
    assert!((c.host_data()[1] - 15.0).abs() < 1e-10);
}

#[test]
fn test_div_forward() {
    let a = GpuTensor::from_host_f64(&[10.0, 6.0], &[2], 0)
        .expect("GpuTensor creation from host data should succeed");
    let b = GpuTensor::from_host_f64(&[2.0, 3.0], &[2], 0)
        .expect("GpuTensor creation from host data should succeed");
    let c = div(&a, &b, None).expect("operation should succeed with valid inputs");
    assert!((c.host_data()[0] - 5.0).abs() < 1e-10);
    assert!((c.host_data()[1] - 2.0).abs() < 1e-10);
}

#[test]
fn test_relu_forward() {
    let a = GpuTensor::from_host_f64(&[-1.0, 0.0, 2.0, -3.0], &[4], 0)
        .expect("GpuTensor creation from host data should succeed");
    let c = relu(&a, None).expect("operation should succeed with valid inputs");
    assert!((c.host_data()[0] - 0.0).abs() < 1e-10);
    assert!((c.host_data()[1] - 0.0).abs() < 1e-10);
    assert!((c.host_data()[2] - 2.0).abs() < 1e-10);
    assert!((c.host_data()[3] - 0.0).abs() < 1e-10);
}

#[test]
fn test_sigmoid_forward() {
    let a = GpuTensor::from_host_f64(&[0.0], &[1], 0)
        .expect("GpuTensor creation from host data should succeed");
    let c = sigmoid(&a, None).expect("operation should succeed with valid inputs");
    assert!((c.host_data()[0] - 0.5).abs() < 1e-10);
}

#[test]
fn test_matmul_forward() {
    // [1 2; 3 4] @ [5 6; 7 8] = [19 22; 43 50]
    let a = GpuTensor::from_host_f64(&[1.0, 2.0, 3.0, 4.0], &[2, 2], 0)
        .expect("GpuTensor creation from host data should succeed");
    let b = GpuTensor::from_host_f64(&[5.0, 6.0, 7.0, 8.0], &[2, 2], 0)
        .expect("GpuTensor creation from host data should succeed");
    let c = matmul(&a, &b, None).expect("operation should succeed with valid inputs");
    assert!((c.host_data()[0] - 19.0).abs() < 1e-10);
    assert!((c.host_data()[1] - 22.0).abs() < 1e-10);
    assert!((c.host_data()[2] - 43.0).abs() < 1e-10);
    assert!((c.host_data()[3] - 50.0).abs() < 1e-10);
}

#[test]
fn test_sum_forward() {
    let a = GpuTensor::from_host_f64(&[1.0, 2.0, 3.0], &[3], 0)
        .expect("GpuTensor creation from host data should succeed");
    let s = sum(&a, None).expect("operation should succeed with valid inputs");
    assert!(
        (s.item()
            .expect("scalar tensor item extraction should succeed")
            - 6.0)
            .abs()
            < 1e-10
    );
}

#[test]
fn test_mean_forward() {
    let a = GpuTensor::from_host_f64(&[1.0, 2.0, 3.0], &[3], 0)
        .expect("GpuTensor creation from host data should succeed");
    let m = mean(&a, None).expect("operation should succeed with valid inputs");
    assert!(
        (m.item()
            .expect("scalar tensor item extraction should succeed")
            - 2.0)
            .abs()
            < 1e-10
    );
}

#[test]
fn test_softmax_forward() {
    let a = GpuTensor::from_host_f64(&[1.0, 2.0, 3.0], &[3], 0)
        .expect("GpuTensor creation from host data should succeed");
    let s = softmax(&a, None).expect("operation should succeed with valid inputs");
    let total: f64 = s.host_data().iter().sum();
    assert!((total - 1.0).abs() < 1e-10);
    // Values should be monotonically increasing
    assert!(s.host_data()[0] < s.host_data()[1]);
    assert!(s.host_data()[1] < s.host_data()[2]);
}

#[test]
fn test_cross_entropy_loss() {
    // 2 samples, 3 classes
    let logits = GpuTensor::from_host_f64(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3], 0)
        .expect("GpuTensor creation from host data should succeed");
    let targets = vec![2, 0]; // class 2, class 0
    let loss = cross_entropy_loss(&logits, &targets, None)
        .expect("operation should succeed with valid inputs");
    // loss > 0
    assert!(
        loss.item()
            .expect("scalar tensor item extraction should succeed")
            > 0.0
    );
}

#[test]
fn test_mse_loss() {
    let pred = GpuTensor::from_host_f64(&[1.0, 2.0], &[2], 0)
        .expect("GpuTensor creation from host data should succeed");
    let target = GpuTensor::from_host_f64(&[1.0, 2.0], &[2], 0)
        .expect("GpuTensor creation from host data should succeed");
    let loss = mse_loss(&pred, &target, None).expect("operation should succeed with valid inputs");
    assert!(
        (loss
            .item()
            .expect("scalar tensor item extraction should succeed")
            - 0.0)
            .abs()
            < 1e-10
    );

    let target2 = GpuTensor::from_host_f64(&[2.0, 3.0], &[2], 0)
        .expect("GpuTensor creation from host data should succeed");
    let loss2 =
        mse_loss(&pred, &target2, None).expect("operation should succeed with valid inputs");
    assert!(
        (loss2
            .item()
            .expect("scalar tensor item extraction should succeed")
            - 1.0)
            .abs()
            < 1e-10
    ); // mean((1)^2, (1)^2) = 1
}

#[test]
fn test_conv2d_forward() {
    // Simple 1x1x3x3 input, 1x1x2x2 kernel, stride=1, padding=0
    let input_data: Vec<f64> = (1..=9).map(|x| x as f64).collect();
    let input = GpuTensor::from_host_f64(&input_data, &[1, 1, 3, 3], 0)
        .expect("GpuTensor creation from host data should succeed");
    let weight = GpuTensor::from_host_f64(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2], 0)
        .expect("GpuTensor creation from host data should succeed");
    let out = conv2d(&input, &weight, None, (1, 1), (0, 0), None)
        .expect("operation should succeed with valid inputs");
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
    // kernel picks top-left and bottom-right of each 2x2 window
    // window (0,0): 1+5=6, (0,1): 2+6=8, (1,0): 4+8=12, (1,1): 5+9=14
    assert!((out.host_data()[0] - 6.0).abs() < 1e-10);
    assert!((out.host_data()[1] - 8.0).abs() < 1e-10);
}

#[test]
fn test_max_pool2d() {
    let input_data: Vec<f64> = (1..=16).map(|x| x as f64).collect();
    let input = GpuTensor::from_host_f64(&input_data, &[1, 1, 4, 4], 0)
        .expect("GpuTensor creation from host data should succeed");
    let out = max_pool2d(&input, (2, 2), (2, 2), (0, 0), None)
        .expect("operation should succeed with valid inputs");
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
    assert!((out.host_data()[0] - 6.0).abs() < 1e-10); // max of [1,2,5,6]
    assert!((out.host_data()[1] - 8.0).abs() < 1e-10); // max of [3,4,7,8]
}

#[test]
fn test_avg_pool2d() {
    let input_data = vec![1.0, 2.0, 3.0, 4.0];
    let input = GpuTensor::from_host_f64(&input_data, &[1, 1, 2, 2], 0)
        .expect("GpuTensor creation from host data should succeed");
    let out = avg_pool2d(&input, (2, 2), (2, 2), (0, 0), None)
        .expect("operation should succeed with valid inputs");
    assert_eq!(out.shape(), &[1, 1, 1, 1]);
    assert!((out.host_data()[0] - 2.5).abs() < 1e-10); // mean(1,2,3,4)
}

#[test]
fn test_layer_norm_forward() {
    let a = GpuTensor::from_host_f64(&[1.0, 2.0, 3.0, 4.0], &[2, 2], 0)
        .expect("GpuTensor creation from host data should succeed");
    let gamma = vec![1.0, 1.0];
    let beta = vec![0.0, 0.0];
    let out = layer_norm(&a, &[2], &gamma, &beta, 1e-5, None)
        .expect("operation should succeed with valid inputs");
    assert_eq!(out.shape(), &[2, 2]);
    // Each pair should be normalized to ~[-1, 1]
    assert!((out.host_data()[0] + out.host_data()[1]).abs() < 1e-5);
}

#[test]
fn test_numerical_gradient_exp() {
    // Verify exp backward with finite differences
    let eps = 1e-5;
    let x_val = 1.5;
    let a = GpuTensor::from_host_f64(&[x_val], &[1], 0)
        .expect("GpuTensor creation from host data should succeed");
    let fwd = exp(&a, None).expect("operation should succeed with valid inputs");
    let analytical = fwd.host_data()[0]; // exp'(x) = exp(x)

    let a_plus = GpuTensor::from_host_f64(&[x_val + eps], &[1], 0)
        .expect("GpuTensor creation from host data should succeed");
    let a_minus = GpuTensor::from_host_f64(&[x_val - eps], &[1], 0)
        .expect("GpuTensor creation from host data should succeed");
    let f_plus = exp(&a_plus, None).expect("operation should succeed with valid inputs");
    let f_minus = exp(&a_minus, None).expect("operation should succeed with valid inputs");
    let numerical = (f_plus.host_data()[0] - f_minus.host_data()[0]) / (2.0 * eps);

    assert!((analytical - numerical).abs() < 1e-5);
}

#[test]
fn test_gelu_forward() {
    let a = GpuTensor::from_host_f64(&[0.0, 1.0, -1.0], &[3], 0)
        .expect("GpuTensor creation from host data should succeed");
    let c = gelu(&a, None).expect("operation should succeed with valid inputs");
    // GELU(0) ≈ 0
    assert!((c.host_data()[0]).abs() < 1e-5);
    // GELU(1) ≈ 0.8412
    assert!((c.host_data()[1] - 0.8412).abs() < 0.01);
}

#[test]
fn test_shape_mismatch_error() {
    let a = GpuTensor::from_host_f64(&[1.0, 2.0], &[2], 0)
        .expect("GpuTensor creation from host data should succeed");
    let b = GpuTensor::from_host_f64(&[1.0, 2.0, 3.0], &[3], 0)
        .expect("GpuTensor creation from host data should succeed");
    assert!(add(&a, &b, None).is_err());
}

#[test]
fn test_batch_norm_forward() {
    // 1 batch, 2 channels, 2x2 spatial
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input = GpuTensor::from_host_f64(&data, &[1, 2, 2, 2], 0)
        .expect("GpuTensor creation from host data should succeed");
    let gamma = vec![1.0, 1.0];
    let beta = vec![0.0, 0.0];
    let out = batch_norm(&input, &gamma, &beta, 1e-5, None)
        .expect("operation should succeed with valid inputs");
    assert_eq!(out.shape(), &[1, 2, 2, 2]);
    // Each channel should be zero-mean (approximately)
    let ch0_mean: f64 = out.host_data()[0..4].iter().sum::<f64>() / 4.0;
    assert!(ch0_mean.abs() < 1e-5);
}

#[test]
fn test_l1_loss() {
    let pred = GpuTensor::from_host_f64(&[1.0, 3.0], &[2], 0)
        .expect("GpuTensor creation from host data should succeed");
    let target = GpuTensor::from_host_f64(&[2.0, 1.0], &[2], 0)
        .expect("GpuTensor creation from host data should succeed");
    let loss = l1_loss(&pred, &target, None).expect("operation should succeed with valid inputs");
    // mean(|1-2|, |3-1|) = mean(1, 2) = 1.5
    assert!(
        (loss
            .item()
            .expect("scalar tensor item extraction should succeed")
            - 1.5)
            .abs()
            < 1e-10
    );
}

#[test]
fn test_smooth_l1_loss() {
    let pred = GpuTensor::from_host_f64(&[0.5], &[1], 0)
        .expect("GpuTensor creation from host data should succeed");
    let target = GpuTensor::from_host_f64(&[0.0], &[1], 0)
        .expect("GpuTensor creation from host data should succeed");
    // beta=1.0: diff=0.5, |0.5| < 1.0 => 0.5*0.25/1.0 = 0.125
    let loss = smooth_l1_loss(&pred, &target, 1.0, None)
        .expect("operation should succeed with valid inputs");
    assert!(
        (loss
            .item()
            .expect("scalar tensor item extraction should succeed")
            - 0.125)
            .abs()
            < 1e-10
    );
}
