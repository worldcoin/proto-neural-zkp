#[cfg(test)]
pub mod tests {
    use ndarray::{ArcArray, Ix3};
    use serde_json;
    use std::{fs, io::prelude::*};

    extern crate test;
    use test::Bencher;

    use crate::layers::{NNJson, NeuralNetwork};

    use crate::nn::create_neural_net;

    #[test]
    fn serialize_model_json() {
        let mut file =
            fs::File::create("./src/json/nn.json").expect("Error encountered while creating file!");

        let model = create_neural_net();

        let model_json: NNJson = model.into();

        file.write_all(serde_json::to_string(&model_json).unwrap().as_bytes())
            .expect("Unable to write data");
    }

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

    #[bench]
    fn bench_serde_neural_net(b: &mut Bencher) {
        b.iter(deserialize_model_json);
        // full deserialization benchmark times
        // cargo bench - 565,564,850 ns/iter (+/- 61,387,641)
    }
}
