use ndarray::{ArrayD, IxDyn};

pub struct ReLU<T> {
    pub output:            ArrayD<T>,
    pub n_params:          i32,
    pub n_multiplications: i32,
    pub name:              String,
}

fn relu(v: f32) -> f32 {
    if v >= 0.0 {
        v
    } else {
        0.0
    }
}

pub fn relu_layer(input: ArrayD<f32>) -> ReLU<f32> {
    let output = input.mapv(relu);
    let n_params = output.len() as i32;
    let n_multiplications = 0;

    ReLU {
        output,
        n_params,
        n_multiplications,
        name: String::from("ReLU"),
    }
}
