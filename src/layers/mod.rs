use ndarray::{Array3, ArrayView3};

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

trait Layer {
    #[must_use]
    fn apply(&self, input: &ArrayView3<f32>) -> Array3<f32>;

    #[must_use]
    fn name(&self) -> &str;

    #[must_use]
    fn num_params(&self) -> usize;

    #[must_use]
    fn num_muls(&self, input: &ArrayView3<f32>) -> usize;
}

pub struct NeuralNetwork {
    layers: Vec<Box<dyn Layer>>,
}

impl NeuralNetwork {
    pub fn apply(&self, input: &ArrayView3<f32>) -> Array3<f32> {
        let mut output = input.clone();
        for layer in &self.layers {
            todo!(); // output = layer.apply(&output);
        }
        todo!()
    }
}
