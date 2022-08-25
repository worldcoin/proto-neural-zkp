use std::fmt::{Display, Formatter, Result};

use ndarray::{ArrayD, ArrayViewD};

pub mod conv;
pub mod flatten;
pub mod fully_connected;
pub mod maxpool;
pub mod normalize;
pub mod relu;

pub trait Layer {
    #[must_use]
    fn apply(&self, input: &ArrayViewD<f32>) -> ArrayD<f32>;

    fn input_shape(&self) -> Vec<usize>;

    #[must_use]
    fn name(&self) -> &str;

    #[must_use]
    fn num_params(&self) -> usize;

    #[must_use]
    fn num_muls(&self) -> usize;

    fn output_shape(&self) -> Vec<usize>;
}

impl Display for Box<dyn Layer> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "{:<20} | {:?}{:<5} | {:<5} | {:<5}",
            self.name(),
            self.output_shape(),
            "",
            self.num_params(),
            self.num_muls(),
        )
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
                println!("{}", layer);
            }
            Some(output)
        } else {
            None
        }
    }
}

impl Default for NeuralNetwork {
    fn default() -> Self {
        Self::new()
    }
}
