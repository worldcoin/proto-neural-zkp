use ndarray::{Array1, Array3};

pub fn flatten_layer(input: Array3<f32>) -> (Array1<f32>, i32, i32, String) {
    let n_params = 0;
    let n_multiplications = 0;
    let output = input.into_iter().collect();

    return (output, n_params, n_multiplications, String::from("flatten"));
}
