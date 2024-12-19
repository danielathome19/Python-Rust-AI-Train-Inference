/* Run with: cargo run --bin perceptron */
/* Inlay hints with Ctrl + Alt */

// use std::fs::File;
// use std::io::{BufReader, Read};
use ndarray::Array1;
use ndarray_npy::read_npy;
use python_rust_ai::dot;

// Define a Perceptron struct
struct Perceptron {
    weights: Array1<f64>,
    bias: f64,
}

impl Perceptron {
    fn predict(&self, inputs: &[f64]) -> i32 {
        let weighted_sum: f64 = dot(&self.weights, inputs) + self.bias;
        if weighted_sum >= 0.0 { 1 } else { 0 }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the weights and bias from files
    let weights: Array1<f64> = read_npy("models/perceptron_weights.npy")?;
    let bias_array: Array1<f64> = read_npy("models/perceptron_bias.npy")?;
    let bias: f64 = bias_array[0];

    // Initialize the Perceptron model
    let perceptron = Perceptron { weights, bias };

    // Define input data for inference (AND gate examples)
    let input_data = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    // Perform inference and collect results
    let results: Vec<i32> = input_data
        .iter()
        .map(|inputs| perceptron.predict(inputs))
        .collect();

    // Print the results
    for (inputs, output) in input_data.iter().zip(results.iter()) {
        println!("Input: {:?}, Output: {}", inputs, output);
    }

    Ok(())
}
