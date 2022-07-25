use ndarray::Array3;

fn relu(v: f64) -> f64 {
    if v >= 0.0 {
        v
    } else {
        0.0
    }
}

pub fn relu_layer(input: Array3<f64>) -> Array3<f64> {
    let output = input.mapv(relu);
    output
}
