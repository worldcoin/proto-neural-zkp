use ndarray::Array1;

struct Normalize<T> {
    output:            Array1<T>,
    n_params:          i32,
    n_multiplications: i32,
    name:              String,
}
