#[cfg(test)]
pub mod tests {
    use crate::layers::{
        conv::Convolution, flatten::Flatten, fully_connected::FullyConnected, maxpool::MaxPool,
        normalize::Normalize, relu::Relu, Layer, NNJson, NeuralNetwork,
    };
    use ndarray::{ArcArray, Array1, Array2, Array3, Array4, Ix1, Ix2, Ix3, Ix4};
    use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
    use rand::rngs::StdRng;
    use serde_json;
    use std::fs;

    #[test]
    fn serialize_model_json() {}
    #[test]
    fn deserialize_model_json() {
        println!(
            "{:<20} | {:<15} | {:<15} | {:<15}",
            "layer", "output shape", "#parameters", "#ops"
        );
        println!("{:-<77}", "");

        let input_json =
            fs::read_to_string("./src/json/initial.json").expect("Unable to read file");
        let input = serde_json::from_str::<ArcArray<f32, Ix3>>(&input_json).unwrap();

        let model_data = fs::read_to_string("./src/json/model.json").expect("Unable to read file");
        let model_json = serde_json::from_str::<NNJson>(&model_data).unwrap();

        let neural_net: NeuralNetwork = model_json.try_into().unwrap();

        let output = neural_net.apply(&input.into_dyn().view(), 3);

        if output.is_some() {
            println!("final output (normalized):\n{}", output.unwrap());
        } else {
            print!("Unsupported dimensionality of input Array");
        }
    }
}
