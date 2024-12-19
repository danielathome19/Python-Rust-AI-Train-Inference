use python_rust_ai::{create_onnx_session, read_tabular_sample, predict_outputs, get_output_names};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let session = create_onnx_session("models/dimensionality_reduction_model.onnx")?;
    let input_data_2d = read_tabular_sample("data/digits.csv")?;
    let input_shape = &[input_data_2d.len(), input_data_2d[0].len()];
    let input_flat = input_data_2d.into_iter().flatten().map(|x| x as f32).collect();  // Flatten 1D Vec<f32>
    let outputs = predict_outputs(&session, input_flat, input_shape)?;
    let output_names = get_output_names(&session);
    
    for (i, output) in outputs.iter().enumerate() {
        let output_name = &output_names[i];
        if output_name == "variable" {
            let variables_array = output.try_extract::<f32>()?;
            println!("Transformed data shape: {:?}", variables_array.view().shape());
            println!("Original data shape: {:?}", input_shape);
        }
    }
    
    Ok(())
}
