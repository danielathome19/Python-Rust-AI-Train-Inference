use ort::{Environment, SessionBuilder, GraphOptimizationLevel, Value, LoggingLevel};
use std::{path::Path, sync::Arc};
use ndarray::{Array2, CowArray};
use python_rust_ai::{read_numeric_sample, ort_argmax};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the environment
    let env = Arc::new(Environment::builder()
        .with_name("fnn_inference")
        .with_log_level(LoggingLevel::Warning)
        .build()?);

    // Load the ONNX model
    let model_path = Path::new("models/lt_fnn_model.onnx");
    let session = SessionBuilder::new(&env)?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_model_from_file(model_path)?;

    // Prepare sample input data
    let (input_data, input_label) = read_numeric_sample("data/mnist_sample_row.csv")?;
    let input_array = CowArray::from(Array2::from_shape_vec((1, 28*28), input_data)?.into_dyn());
    let input_tensor_values = vec![Value::from_array(session.allocator(), &input_array)?];

    // Perform inference
    let outputs = session.run(input_tensor_values)?;
    for output in outputs {
        let output_array = output.try_extract::<f32>()?;
        let predicted_label = ort_argmax(output_array);
        println!("Predicted label: {}", predicted_label);
        println!("Actual label: {}", input_label);
    }

    Ok(())
}