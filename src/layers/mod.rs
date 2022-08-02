pub mod conv;
pub mod flatten;
pub mod fully_connected;
pub mod maxpool;
pub mod normalize;
pub mod relu;

pub struct OutputWrapper<T> {
    pub output:            T,
    pub n_params:          usize,
    pub n_multiplications: usize,
    pub name:              String,
}
