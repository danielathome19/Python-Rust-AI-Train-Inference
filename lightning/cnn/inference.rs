use python_rust_ai::{create_onnx_session, read_numeric_sample, ort_argmax, predict_outputs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let session = create_onnx_session("models/lt_cnn_model.onnx")?;
    let (input_data, input_label) = read_numeric_sample("data/cifar_sample_row.csv")?;
    let outputs = predict_outputs(&session, input_data, &[1, 3, 32, 32])?;
    
    for output in outputs {
        let output_array = output.try_extract::<f32>()?;
        let predicted_label = ort_argmax(output_array);
        println!("Predicted label: {}", predicted_label);
        println!("Actual label: {}", input_label);
    }

    Ok(())
}