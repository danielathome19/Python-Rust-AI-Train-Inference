use python_rust_ai::{create_onnx_session, read_numeric_sample, predict_outputs, get_output_names};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let session = create_onnx_session("models/regression_model.onnx")?;
    let (input_data, _) = read_numeric_sample("data/housing_sample_row.csv")?;
    let input_size = input_data.len();
    let outputs = predict_outputs(&session, input_data, &[1, input_size])?;
    let output_names = get_output_names(&session);
    
    for (i, output) in outputs.iter().enumerate() {
        let output_name = &output_names[i];
        if output_name == "variable" {
            let variables_array = output.try_extract::<f32>()?;
            println!("Predicted value: {:?}", variables_array.view().as_slice().unwrap());
        }
    }

    Ok(())
}
