#[cfg(test)]
pub mod test {
    use crate::layers::{
        conv::Convolution, flatten::Flatten, fully_connected::FullyConnected, maxpool::MaxPooling,
        normalize::Normalize, relu::Relu, Layer, NeuralNetwork,
    };
    use ndarray::{Array1, Array2, Array3, Array4, Ix3};
    use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
    use rand::rngs::StdRng;
    #[test]
    fn neural_net() {
        let seed = 694201337;
        let mut rng = StdRng::seed_from_u64(seed);

        println!(
            "{:<20} | {:<15} | {:<15} | {:<15}",
            "layer", "output shape", "#parameters", "#ops"
        );
        println!("{:-<77}", "");

        // input
        let input = Array3::random_using((120, 80, 3), Uniform::<f32>::new(-5., 5.), &mut rng);

        let kernel = Array4::random_using((32, 5, 5, 3), Uniform::<f32>::new(-10., 10.), &mut rng);

        let conv = Convolution::new(kernel);

        let n_multiplications = conv.num_muls(&input.clone().into_dyn().view());

        let x = conv
            .apply(&input.into_dyn().view())
            .into_dimensionality::<Ix3>()
            .unwrap();

        // let Conv2D::<f32> {
        //     output: x,
        //     n_params,
        //     n_multiplications,
        //     name,
        // } = convolution(&x.view(), &f.view());

        let (dim_x, dim_y, dim_z) = &x.dim();

        assert_eq!(&x.dim(), &(116, 76, 32));

        println!(
            "{} |  ({}, {}, {}) | {} |  {}",
            conv.name(),
            dim_x,
            dim_y,
            dim_z,
            conv.num_params(),
            n_multiplications
        );

        // max pooling
        // kernel side

        let maxpool = MaxPooling::new(2);

        let n_multiplications = maxpool.num_muls(&x.clone().into_dyn().view());

        let x = maxpool
            .apply(&x.into_dyn().view())
            .into_dimensionality::<Ix3>()
            .unwrap();

        // let MaxPool::<f32> {
        //     output: x,
        //     n_params,
        //     n_multiplications,
        //     name,
        // } = max_pooling_layer(&x, s);

        assert_eq!(x.dim(), (58, 38, 32));

        let (dim_x, dim_y, dim_z) = x.dim();

        println!(
            "{} |  ({}, {}, {}) | {} |  {}",
            maxpool.name(),
            dim_x,
            dim_y,
            dim_z,
            maxpool.num_params(),
            n_multiplications
        );

        // relu layer

        let relu = Relu::new();

        let n_params = relu.num_params();

        let n_multiplications = relu.num_muls(&x.clone().into_dyn().view());

        let x = relu.apply(&x.into_dyn().view());

        // let ReLU::<f32> {
        //     output: x,
        //     n_params,
        //     n_multiplications,
        //     name,
        // } = relu_layer(&x.into_dyn());

        let x = x.into_dimensionality::<Ix3>().unwrap();

        assert_eq!(x.dim(), (58, 38, 32));

        println!(
            "{} |  ({}, {}, {}) | {} |  {}",
            relu.name(),
            dim_x,
            dim_y,
            dim_z,
            n_params,
            n_multiplications
        );

        // conv layer

        // kernel
        let f = Array4::random_using((32, 5, 5, 32), Uniform::<f32>::new(-10., 10.), &mut rng);

        let (c_out, hf, wf, c_in) = &f.dim();

        dbg!(f.dim());
        dbg!(x.dim());
        let conv = Convolution::new(f);

        let x = conv
            .apply(&x.into_dyn().view())
            .into_dimensionality::<Ix3>()
            .unwrap();
        // let Conv2D::<f32> {
        //     output: x,
        //     n_params,
        //     n_multiplications,
        //     name,
        // } = convolution(&x.into_dimensionality::<Ix3>().unwrap().view(), &f.view());

        let (dim_x, dim_y, dim_z) = &x.dim();

        assert_eq!(&x.dim(), &(54, 34, 32));

        println!(
            "{} |  ({}, {}, {}) | {} |  {}",
            conv.name(),
            dim_x,
            dim_y,
            dim_z,
            n_params,
            n_multiplications
        );

        // max pooling

        let maxpool = MaxPooling::new(2);

        let n_multiplications = maxpool.num_muls(&x.clone().into_dyn().view());

        let x = maxpool
            .apply(&x.into_dyn().view())
            .into_dimensionality::<Ix3>()
            .unwrap();
        // let MaxPool::<f32> {
        //     output: x,
        //     n_params,
        //     n_multiplications,
        //     name,
        // } = max_pooling_layer(&x, 2);

        assert_eq!(x.dim(), (27, 17, 32));

        let (dim_x, dim_y, dim_z) = x.dim();

        println!(
            "{} |  ({}, {}, {}) | {} |  {}",
            maxpool.name(),
            dim_x,
            dim_y,
            dim_z,
            n_params,
            n_multiplications
        );

        // relu layer

        let relu = Relu::new();

        let n_params = relu.num_params();

        let n_multiplications = relu.num_muls(&x.clone().into_dyn().view());

        let x = relu
            .apply(&x.into_dyn().view())
            .into_dimensionality::<Ix3>()
            .unwrap();

        // let ReLU::<f32> {
        //     output: x,
        //     n_params,
        //     n_multiplications,
        //     name,
        // } = relu_layer(&x.into_dyn());

        assert_eq!(&x.dim(), &(27, 17, 32));

        println!(
            "{} |  ({}, {}, {}) | {} |  {}",
            relu.name(),
            dim_x,
            dim_y,
            dim_z,
            n_params,
            n_multiplications
        );

        // flatten

        let flatten = Flatten::new();

        let n_params = flatten.num_params();
        let n_multiplications = flatten.num_muls(&x.clone().into_dyn().view().clone());

        let x = flatten.apply(&x.into_dyn().view());
        // let Flatten::<f32> {
        //     output: x,
        //     n_params,
        //     n_multiplications,
        //     name,
        // } = flatten_layer(&x.into_dyn().view());

        assert_eq!(x.len(), 14688);

        println!(
            "{} |  ({}x1) | {} |  {}",
            flatten.name(),
            x.len(),
            n_params,
            n_multiplications
        );

        // fully connected

        let weights =
            Array2::random_using((1000, 14688), Uniform::<f32>::new(-10.0, 10.0), &mut rng);
        let biases = Array1::random_using(1000, Uniform::<f32>::new(-10.0, 10.0), &mut rng);

        let fully_connected = FullyConnected::new(weights, biases);

        let n_params = fully_connected.num_params();

        let n_multiplications = fully_connected.num_muls(&x.clone().into_dyn().view());

        let x = fully_connected.apply(&x.into_dyn().view());

        // let FCLayer::<f32> {
        //     output: x,
        //     n_params,
        //     n_multiplications,
        //     name,
        // } = fully_connected(&x, &weights, &biases);

        println!(
            "{} |  ({}x1) | {} |  {}",
            fully_connected.name(),
            x.len(),
            n_params,
            n_multiplications
        );

        // relu layer
        let relu = Relu::new();

        let n_params = relu.num_params();

        let n_multiplications = relu.num_muls(&x.clone().into_dyn().view());

        let x = relu.apply(&x.into_dyn().view());
        // let ReLU::<f32> {
        //     output: x,
        //     n_params,
        //     n_multiplications,
        //     name,
        // } = relu_layer(&x.into_dyn());

        assert_eq!(x.len(), 1000);

        println!(
            "{} |  ({}) | {} |  {}",
            relu.name(),
            x.len(),
            n_params,
            n_multiplications
        );

        // fully connected

        let weights = Array2::random_using((5, 1000), Uniform::<f32>::new(-10.0, 10.0), &mut rng);
        let biases = Array1::random_using(5, Uniform::<f32>::new(-10.0, 10.0), &mut rng);

        let fully_connected = FullyConnected::new(weights, biases);

        let n_params = fully_connected.num_params();

        let n_multiplications = fully_connected.num_muls(&x.clone().into_dyn().view());

        let x = fully_connected.apply(&x.into_dyn().view());

        // let FCLayer::<f32> {
        //     output: x,
        //     n_params,
        //     n_multiplications,
        //     name,
        // } = fully_connected(&x, &weights, &biases);

        println!(
            "{} |  ({}x1) | {} |  {} \n final output: \n{}",
            fully_connected.name(),
            x.len(),
            n_params,
            n_multiplications,
            x
        );

        // normalization

        let normalize = Normalize::new();

        let x = normalize.apply(&x.into_dyn().view());
        // let Normalize::<f64> {
        //     output: x,
        //     n_params,
        //     n_multiplications,
        //     name,
        // } = normalize(&x);

        println!("final output (normalized):\n{}", x);
    }

    #[test]
    fn neural_net2() {
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

        neural_net.add_layer(Box::new(Convolution::new(kernel)));

        neural_net.add_layer(Box::new(MaxPooling::new(2)));

        neural_net.add_layer(Box::new(Relu::new()));

        let kernel = Array4::random_using((32, 5, 5, 32), Uniform::<f32>::new(-10., 10.), &mut rng);

        neural_net.add_layer(Box::new(Convolution::new(kernel)));

        neural_net.add_layer(Box::new(MaxPooling::new(2)));
        neural_net.add_layer(Box::new(Relu::new()));
        neural_net.add_layer(Box::new(Flatten::new()));
        let weights =
            Array2::random_using((1000, 14688), Uniform::<f32>::new(-10.0, 10.0), &mut rng);
        let biases = Array1::random_using(1000, Uniform::<f32>::new(-10.0, 10.0), &mut rng);
        neural_net.add_layer(Box::new(FullyConnected::new(weights, biases)));
        neural_net.add_layer(Box::new(Relu::new()));

        let weights = Array2::random_using((5, 1000), Uniform::<f32>::new(-10.0, 10.0), &mut rng);
        let biases = Array1::random_using(5, Uniform::<f32>::new(-10.0, 10.0), &mut rng);

        neural_net.add_layer(Box::new(FullyConnected::new(weights, biases)));
        neural_net.add_layer(Box::new(Normalize::new()));

        let output = neural_net.apply(&input.into_dyn().view(), 3);

        if output.is_some() {
            println!("final output (normalized):\n{}", output.unwrap());
        } else {
            print!("Unsupported dimensionality of input Array");
        }
    }
}
