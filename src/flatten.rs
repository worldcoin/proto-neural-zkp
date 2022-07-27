use ndarray::{Array1, Array3};

pub struct Flatten<T> {
    pub output:            Array1<T>,
    pub n_params:          i32,
    pub n_multiplications: i32,
    pub name:              String,
}

pub fn flatten_layer(input: Array3<f32>) -> Flatten<f32> {
    let n_params = 0;
    let n_multiplications = 0;
    let output = input.into_iter().collect();

    Flatten {
        output,
        n_params,
        n_multiplications,
        name: String::from("flatten"),
    }
}
