use std::fmt::{Display, Formatter, Result};

use erased_serde::serialize_trait_object;
use ndarray::{ArcArray, ArrayD, ArrayViewD, Ix1, Ix2, Ix4};
use serde::{Deserialize, Serialize};

pub mod conv;
pub mod flatten;
pub mod fully_connected;
pub mod maxpool;
pub mod normalize;
pub mod relu;

pub trait Layer: erased_serde::Serialize {
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

    #[must_use]
    fn to_json(&self) -> LayerJson;

    fn box_clone(&self) -> Box<dyn Layer>;
}

serialize_trait_object!(Layer);

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

impl Clone for Box<dyn Layer> {
    fn clone(&self) -> Self {
        //    self -> &Box<dyn Layer>
        //   *self ->  Box<dyn Layer>
        //  **self ->      dyn Layer
        // &**self ->     &dyn Layer
        Layer::box_clone(&**self)
    }
}

#[derive(Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "layer_type")]
pub enum LayerJson {
    Convolution {
        kernel:      ArcArray<f32, Ix4>,
        input_shape: Vec<usize>,
    },
    MaxPool {
        window:      usize,
        input_shape: Vec<usize>,
    },
    FullyConnected {
        weights: ArcArray<f32, Ix2>,
        biases:  ArcArray<f32, Ix1>,
    },
    Relu {
        input_shape: Vec<usize>,
    },
    Flatten {
        input_shape: Vec<usize>,
    },
    Normalize {
        input_shape: Vec<usize>,
    },
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct NNJson {
    pub layers: Vec<LayerJson>,
}

impl TryFrom<LayerJson> for Box<dyn Layer> {
    type Error = ();

    fn try_from(value: LayerJson) -> std::result::Result<Self, ()> {
        Ok(match value {
            LayerJson::Convolution {
                kernel,
                input_shape,
            } => Box::new(conv::Convolution::new(kernel.to_owned(), input_shape)),
            LayerJson::MaxPool {
                window,
                input_shape,
            } => Box::new(maxpool::MaxPool::new(window.to_owned(), input_shape)),
            LayerJson::FullyConnected { weights, biases } => Box::new(
                fully_connected::FullyConnected::new(weights.to_owned(), biases.to_owned()),
            ),
            LayerJson::Flatten { input_shape } => Box::new(flatten::Flatten::new(input_shape)),
            LayerJson::Relu { input_shape } => Box::new(relu::Relu::new(input_shape)),
            LayerJson::Normalize { input_shape } => {
                Box::new(normalize::Normalize::new(input_shape))
            }
        })
    }
}

impl FromIterator<LayerJson> for NNJson {
    fn from_iter<T: IntoIterator<Item = LayerJson>>(iter: T) -> Self {
        let mut nnvec = vec![];

        for i in iter {
            nnvec.push(i);
        }

        Self { layers: nnvec }
    }
}

impl From<NeuralNetwork> for NNJson {
    fn from(nn: NeuralNetwork) -> Self {
        nn.layers.into_iter().map(|l| l.to_json()).collect()
    }
}

impl TryFrom<NNJson> for NeuralNetwork {
    type Error = ();

    fn try_from(value: NNJson) -> std::result::Result<Self, ()> {
        Ok(Self {
            layers: value
                .layers
                .into_iter()
                .map(|i| i.try_into().unwrap())
                .collect(),
        })
    }
}

#[derive(Clone, Serialize)]
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
