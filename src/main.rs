use nalgebra::{Matrix3, matrix, vector};
use std::iter::zip;
use std::process::Output;
fn main() {
    // let inputs = dvector![1.0, 2.0, 3.0, 2.5];
    // let weights = dvector![0.2, 0.8, -0.5, 1.0];
    // let bias = 2.0;
    // let output = inputs.dot(&weights) + bias;
    // println!("{}", output);

    // let a = dvector![1.0, 2.0, 3.0];
    // let b = dvector![4.0, 5.0, 6.0];
    // let c = a.dot(&b);
    // println!("{}", c);
    let inputs = matrix![
        1.0, 2.0, 3.0, 2.5;
        2.0,5.0,-1.0,2.0;
        -1.5,2.7,3.3,-0.8;
    ];
    let weights = matrix! [
        0.2, 0.8, -0.5, 1.0;
        0.5, -0.91, 0.26, -0.5;
        -0.26, -0.27, 0.17, 0.87;
    ];
    let weights = weights.transpose();
    let weights2 = matrix! [
        0.1, -0.14, 0.5;
        -0.5, 0.12, -0.33;
        -0.44, 0.73, -0.13;
    ];
    let weights2 = weights2.transpose();
    let biases = vector![2.0, 3.0, 0.5];
    let biases2 = vector![-1.0, 2.0, -0.5];
    let m = Matrix3::from_columns(&[biases, biases, biases]);
    let m = m.transpose();
    let m2 = Matrix3::from_columns(&[biases2, biases2, biases2]);
    let m2 = m2.transpose();
    let layer_outputs1 = inputs * weights + m;
    let layer_outputs2 = layer_outputs1 * weights2 + m2;

    // layer_outputs= inputs.dot(&weights)
    // for (neuron_weights, neuron_bias) in zip(weights, biases) {
    //     let mut neuron_output = 0.0;
    //     for (n_input, weight) in zip(inputs, neuron_weights) {
    //         neuron_output += n_input * weight;
    //     }
    //     neuron_output += neuron_bias;
    //     layer_outputs.push(neuron_output);
    // }
    // let output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias;
    println!("{}, {}", layer_outputs1, layer_outputs2);
}
