use onnxruntime::{environment::Environment, tensor::OrtOwnedTensor, GraphOptimizationLevel, LoggingLevel, session::Session};
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the ONNX model
    let environment = Environment::builder()
        .with_name("regression_inference")
        .with_log_level(LoggingLevel::Warning)
        .build()?;
    let session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Basic)?
        .with_model_from_file("regression_model.onnx")?;

    // Prepare input data for inference
    let input_data = array![[8.3252, 41.0, 6.984127, 1.02381, 322.0, 2.555556, 37.88, 1.0, 0.0, 0.0, 0.0, 0.0]];
    let input_tensor_values = vec![input_data.into_dyn()];

    // Perform inference
    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor_values)?;

    // Print the inference result
    for output in outputs {
        println!("{:?}", output);
    }

    Ok(())
}
