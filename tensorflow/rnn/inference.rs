use onnxruntime::{environment::Environment, tensor::OrtOwnedTensor, GraphOptimizationLevel, LoggingLevel, session::Session};
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the ONNX model
    let environment = Environment::builder()
        .with_name("rnn_inference")
        .with_log_level(LoggingLevel::Warning)
        .build()?;
    let session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Basic)?
        .with_model_from_file("rnn_model.onnx")?;

    // Prepare input data for inference
    let input_data = array![[[0.0; 28]; 28]];
    let input_tensor_values = vec![input_data.into_dyn()];

    // Perform inference
    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor_values)?;

    // Print the inference result
    for output in outputs {
        println!("{:?}", output);
    }

    Ok(())
}
