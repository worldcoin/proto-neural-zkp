use ndarray::arr3;

use neural_zkp::relu::*;

#[test]

fn relu_test() {
    let input = arr3(&[[[1.2, -4.3], [-2.1, 4.3]], [[5.2, 6.1], [7.6, -1.8]], [
        [9.3, 0.0],
        [1.2, 3.4],
    ]]);

    let ReLU::<f32> {
        output: x,
        n_params,
        n_multiplications,
        name,
    } = relu_layer(input);

    assert_eq!(
        x,
        arr3(&[[[1.2, 0.0], [0.0, 4.3]], [[5.2, 6.1], [7.6, 0.0]], [
            [9.3, 0.0],
            [1.2, 3.4],
        ]])
    );
    println!(
        "
        {} 
        # of parameters: {}
        output dim: {}x1
        # of ops: {}
        output:\n{}",
        name, n_params, n_params, n_multiplications, x
    );
}
