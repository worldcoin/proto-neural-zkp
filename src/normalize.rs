use ndarray::Array1;

pub struct Normalize<T> {
    pub output:            Array1<T>,
    pub n_params:          i32,
    pub n_multiplications: i32,
    pub name:              String,
}

pub fn normalize(input: Array1<i128>) -> Normalize<f64> {
    let n_params = 0;
    let n_multiplications = 1 + input.len() as i32;
    let norm = f64::sqrt(input.mapv(|x| x.pow(2)).sum() as f64);
    let output = input.mapv(|x| x as f64 / norm);

    Normalize {
        output,
        n_params,
        n_multiplications,
        name: String::from("normalize"),
    }
}
