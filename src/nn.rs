#[cfg(test)]
pub mod tests {
    extern crate test;
    use crate::layers::{
        conv::Convolution, flatten::Flatten, fully_connected::FullyConnected, maxpool::MaxPool,
        normalize::Normalize, relu::Relu, Layer, NeuralNetwork,
    };
    use ndarray::{ArcArray, Array1, Array2, Array3, Array4, Ix1, Ix2, Ix3, Ix4};
    use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
    use rand::rngs::StdRng;
    use std::fs;
    use test::Bencher;

    #[test]
    fn neural_net() {
        // neural net layers:
        // conv
        // maxpool
        // relu
        // conv
        // max pool
        // relu
        // flatten
        // fully connected
        // relu
        // fully connected
        // normalization

        let seed = 694201337;
        let mut rng = StdRng::seed_from_u64(seed);

        println!(
            "{:<20} | {:<15} | {:<15} | {:<15}",
            "layer", "output shape", "#parameters", "#ops"
        );
        println!("{:-<77}", "");

        let input = Array3::random_using((120, 80, 3), Uniform::<f32>::new(-5., 5.), &mut rng);

        let mut neural_net = NeuralNetwork::new();

        let kernel = Array4::random_using((32, 5, 5, 3), Uniform::<f32>::new(-10., 10.), &mut rng);

        neural_net.add_layer(Box::new(Convolution::new(kernel, vec![120, 80, 3])));

        neural_net.add_layer(Box::new(MaxPool::new(2, vec![116, 76, 32])));

        neural_net.add_layer(Box::new(Relu::new(vec![58, 38, 32])));

        let kernel = Array4::random_using((32, 5, 5, 32), Uniform::<f32>::new(-10., 10.), &mut rng);

        neural_net.add_layer(Box::new(Convolution::new(kernel, vec![58, 38, 32])));

        neural_net.add_layer(Box::new(MaxPool::new(2, vec![54, 34, 32])));
        neural_net.add_layer(Box::new(Relu::new(vec![27, 17, 32])));
        neural_net.add_layer(Box::new(Flatten::new(vec![27, 17, 32])));
        let weights =
            Array2::random_using((1000, 14688), Uniform::<f32>::new(-10.0, 10.0), &mut rng);
        let biases = Array1::random_using(1000, Uniform::<f32>::new(-10.0, 10.0), &mut rng);
        neural_net.add_layer(Box::new(FullyConnected::new(weights, biases)));
        neural_net.add_layer(Box::new(Relu::new(vec![1000])));

        let weights = Array2::random_using((5, 1000), Uniform::<f32>::new(-10.0, 10.0), &mut rng);
        let biases = Array1::random_using(5, Uniform::<f32>::new(-10.0, 10.0), &mut rng);

        neural_net.add_layer(Box::new(FullyConnected::new(weights, biases)));
        neural_net.add_layer(Box::new(Normalize::new(vec![5])));

        let output = neural_net.apply(&input.into_dyn().view(), 3);

        if output.is_some() {
            println!("final output (normalized):\n{}", output.unwrap());
        } else {
            print!("Unsupported dimensionality of input Array");
        }
    }

    #[bench]
    fn bench_neural_net(b: &mut Bencher) {
        println!(
            "{:<20} | {:<15} | {:<15} | {:<15}",
            "layer", "output shape", "#parameters", "#ops"
        );
        println!("{:-<77}", "");

        let data = fs::read_to_string("./src/json/initial.json").expect("Unable to read file");
        let res = serde_json::from_str::<ArcArray<f32, Ix3>>(&data);

        let input = res.unwrap();

        let mut neural_net = NeuralNetwork::new();

        let data = fs::read_to_string("./src/json/conv1.json").expect("Unable to read file");
        let res = serde_json::from_str::<ArcArray<f32, Ix4>>(&data);

        let kernel = res.unwrap().into_owned();

        neural_net.add_layer(Box::new(Convolution::new(kernel, vec![120, 80, 3])));
        neural_net.add_layer(Box::new(MaxPool::new(2, vec![116, 76, 32])));
        neural_net.add_layer(Box::new(Relu::new(vec![58, 38, 32])));

        let data = fs::read_to_string("./src/json/conv2.json").expect("Unable to read file");
        let res = serde_json::from_str::<ArcArray<f32, Ix4>>(&data);

        let kernel = res.unwrap().into_owned();

        neural_net.add_layer(Box::new(Convolution::new(kernel, vec![58, 38, 32])));

        neural_net.add_layer(Box::new(MaxPool::new(2, vec![54, 34, 32])));
        neural_net.add_layer(Box::new(Relu::new(vec![27, 17, 32])));
        neural_net.add_layer(Box::new(Flatten::new(vec![27, 17, 32])));

        let data = fs::read_to_string("./src/json/weights1.json").expect("Unable to read file");
        let res = serde_json::from_str::<ArcArray<f32, Ix2>>(&data);

        let weights = res.unwrap().into_owned();

        let data = fs::read_to_string("./src/json/biases1.json").expect("Unable to read file");
        let res = serde_json::from_str::<ArcArray<f32, Ix1>>(&data);

        let biases = res.unwrap().into_owned();

        neural_net.add_layer(Box::new(FullyConnected::new(weights, biases)));
        neural_net.add_layer(Box::new(Relu::new(vec![1000])));

        let data = fs::read_to_string("./src/json/weights2.json").expect("Unable to read file");
        let res = serde_json::from_str::<ArcArray<f32, Ix2>>(&data);

        let weights = res.unwrap().into_owned();

        let data = fs::read_to_string("./src/json/biases2.json").expect("Unable to read file");
        let res = serde_json::from_str::<ArcArray<f32, Ix1>>(&data);

        let biases = res.unwrap().into_owned();

        neural_net.add_layer(Box::new(FullyConnected::new(weights, biases)));
        neural_net.add_layer(Box::new(Normalize::new(vec![5])));

        // Python Vanilla CNN implementation, run time: 0.8297840171150046 (average over
        // 1000 runs on M1 Max Macbook Pro)
        b.iter(|| neural_net.apply(&input.clone().into_dyn().view(), 3));
        // cargo bench - 0.151693329s +- 0.002147193s - (about 5.8x faster)
    }
}
