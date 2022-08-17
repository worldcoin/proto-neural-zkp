use ndarray::{Array3, ArrayD, ArrayView3, ArrayViewD};

pub mod conv;
pub mod flatten;
pub mod fully_connected;
pub mod maxpool;
pub mod normalize;
pub mod relu;

pub trait Layer {
    #[must_use]
    fn apply(&self, input: &ArrayViewD<f32>) -> ArrayD<f32>;

    #[must_use]
    fn name(&self) -> &str;

    #[must_use]
    fn num_params(&self) -> usize;

    #[must_use]
    fn num_muls(&self, input: &ArrayViewD<f32>) -> usize;

    fn output_shape(&self, input: &ArrayViewD<f32>, dim: usize) -> Option<Vec<usize>>;
}

pub struct NeuralNetwork {
    layers: Vec<Box<dyn Layer>>,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Self { layers }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        // TODO: add dimensionality sanity checks
        self.layers.push(layer);
    }

    pub fn apply(&self, input: &ArrayViewD<f32>) -> ArrayD<f32> {
        let mut output = input.to_owned();
        for layer in &self.layers {
            output = layer.apply(&output.view());
        }
        output
    }
}
