use std::fmt::Display;

use ndarray::{Array3, ArrayD, ArrayView3, ArrayViewD, Ix3};

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

// TODO
impl Display for Box<dyn Layer> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

pub struct NeuralNetwork {
    layers: Vec<Box<dyn Layer>>,
}

impl NeuralNetwork {
    pub fn new() -> Self {
        Self { layers: vec![] }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn apply(&self, input: &ArrayViewD<f32>, dim: usize) -> Option<ArrayD<f32>> {
        if dim == 3 {
            let mut output = input.view().into_owned();
            for layer in &self.layers {
                // TODO: add dimensionality sanity checks
                output = layer.apply(&output.view());
            }
            Some(output)
        } else {
            None
        }
    }
}
