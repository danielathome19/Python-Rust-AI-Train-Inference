use std::fs::File;
use std::io::{BufReader, Read};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the trained model
    let file = File::open("perceptron_weights.npy")?;
    let mut buf_reader = BufReader::new(file);
    let mut contents = Vec::new();
    buf_reader.read_to_end(&mut contents)?;
    let weights: Array1<f64> = ndarray::Array::from_shape_vec((3,), contents)?;

    // Define the activation function
    let activation_fn = |x: f64| -> i32 {
        if x >= 0.0 { 1 } else { 0 }
    };

    // Perform inference on new data
    let input_data = vec![1.0, 0.0, 1.0]; // Example input data
    let z = weights.dot(&Array1::from(input_data));
    let prediction = activation_fn(z);

    // Print the inference result
    println!("Prediction: {}", prediction);

    Ok(())
}
