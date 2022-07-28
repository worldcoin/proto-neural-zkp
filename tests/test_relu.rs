use ndarray::{arr1, arr3, Ix1, Ix3};

use neural_zkp::relu::*;

#[test]

fn relu_test() {
    // Array3 ReLU
    let input = arr3(&[[[1.2, -4.3], [-2.1, 4.3]], [[5.2, 6.1], [7.6, -1.8]], [
        [9.3, 0.0],
        [1.2, 3.4],
    ]])
    .into_dyn();

    let ReLU::<f32> {
        output: x,
        n_params,
        n_multiplications,
        name,
    } = relu_layer(input);

    let result = x.clone();

    assert_eq!(
        result,
        arr3(&[[[1.2, 0.0], [0.0, 4.3]], [[5.2, 6.1], [7.6, 0.0]], [
            [9.3, 0.0],
            [1.2, 3.4],
        ]])
        .into_dyn()
    );

    let size: (usize, usize, usize) = (3, 2, 2);

    assert_eq!(result.into_dimensionality::<Ix3>().unwrap().dim(), size);

    println!(
        "
        {} 
        # of parameters: {}
        output dim: {}x1
        # of ops: {}
        output:\n{}",
        name, n_params, n_params, n_multiplications, x
    );

    // Array1 ReLU

    let input = arr1(&[-4., -3.4, 6., 7., 1., -3.]).into_dyn();

    let ReLU::<f32> {
        output: x,
        n_params,
        n_multiplications,
        name,
    } = relu_layer(input);

    let result = x.clone();

    assert_eq!(result, arr1(&[0., 0., 6., 7., 1., 0.]).into_dyn());

    assert_eq!(result.into_dimensionality::<Ix1>().unwrap().len(), 6);

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
