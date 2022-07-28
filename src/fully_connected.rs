use ndarray::{Array1, Array2};

pub struct FCLayer<T> {
    pub output:            Array1<T>,
    pub n_params:          i32,
    pub n_multiplications: i32,
    pub name:              String,
}

pub fn fully_connected(
    input: Array1<f32>,
    weights: Array2<f32>,
    biases: Array1<f32>,
) -> FCLayer<f32> {
    assert!(input.ndim() == 1, "Input must be a flattenened array!");
    assert!(
        weights.shape()[0] == input.shape()[0],
        "Input shapes must match (for the dot product to work)!"
    );
    assert!(
        weights.shape()[1] == biases.shape()[0],
        "Output shapes must match!"
    );

    let output = input.dot(&weights) + &biases;

    let n_params = (weights.len() + biases.len()) as i32;

    let n_multiplications = weights.len() as i32;

    FCLayer {
        output,
        n_params,
        n_multiplications,
        name: String::from("normalize"),
    }
}
