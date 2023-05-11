use std::{fs, io::Write};

use crate::layers::{NNJson, NeuralNetwork};

/// path: &str -> e.g. deserialize_model_json("./src/json/model.json")
/// Deserializes a model from JSON from the NNJson Struct in mod.rs
pub fn deserialize_model_json(path: &str) -> NeuralNetwork {
    let model_data = fs::read_to_string(path).expect("Unable to read file");
    let model_json = serde_json::from_str::<NNJson>(&model_data).unwrap();

    let neural_net: NeuralNetwork = model_json.try_into().unwrap();

    neural_net
}

/// path: &str - path for the creation of a json file containing serialized
/// model representation
/// model: NeuralNetwork - model to be serialized (dummy
/// creation with nn::create_neural_net())
/// example usage:
/// serialize_model_json("./src/json/nn.json", create_neural_net())
pub fn serialize_model_json(path: &str, model: NeuralNetwork) {
    let mut file = fs::File::create(path).expect("Error encountered while creating file!");

    let model_json: NNJson = model.into();

    file.write_all(serde_json::to_string(&model_json).unwrap().as_bytes())
        .expect("Unable to write data");
}

#[cfg(test)]
pub mod tests {
    use ndarray::{ArcArray, Ix3};
    use serde_json;
    use std::fs;

    extern crate test;
    use test::Bencher;

    use super::*;
    use crate::nn::{create_neural_net, log_nn_table};

    #[test]
    fn serialize_nn_json() {
        serialize_model_json("./src/json/nn.json", create_neural_net())
    }

    #[test]
    fn deserialize_nn_json() {
        // initial log
        log_nn_table();

        // deserialize input JSON file into an ArcArray
        let input_json =
            fs::read_to_string("./src/json/initial.json").expect("Unable to read file");
        let input = serde_json::from_str::<ArcArray<f32, Ix3>>(&input_json).unwrap();

        // deserialize model
        let neural_net = deserialize_model_json("./src/json/model.json");

        // run inference
        let output = neural_net.apply(&input.into_dyn().view(), 3);

        if output.is_some() {
            println!("final output (normalized):\n{}", output.unwrap());
        } else {
            print!("Unsupported dimensionality of input Array");
        }
    }

    #[test]
    fn serde_full_circle() {
        serialize_model_json("./src/json/nn.json", create_neural_net());

        // initial log
        log_nn_table();

        // deserialize input JSON file into an ArcArray
        let input_json =
            fs::read_to_string("./src/json/initial.json").expect("Unable to read file");
        let input = serde_json::from_str::<ArcArray<f32, Ix3>>(&input_json).unwrap();

        // deserialize model
        let neural_net = deserialize_model_json("./src/json/nn.json");

        // run inference
        let output = neural_net.apply(&input.into_dyn().view(), 3);

        if output.is_some() {
            println!("final output (normalized):\n{}", output.unwrap());
        } else {
            print!("Unsupported dimensionality of input Array");
        }
    }

    #[bench]
    fn bench_deserialize_neural_net(b: &mut Bencher) {
        b.iter(deserialize_nn_json);
        // full deserialization benchmark times (M1 Max Macbook Pro)
        // cargo bench - 565,564,850 ns/iter (+/- 61,387,641)
    }

    #[bench]
    fn bench_serialize_neural_net(b: &mut Bencher) {
        b.iter(serialize_nn_json);
        // full serialization benchmark times (M1 Max Macbook Pro)
        // cargo bench - 579,057,637 ns/iter (+/- 20,202,535)
    }
}
