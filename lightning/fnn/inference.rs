use python_rust_ai::{create_onnx_session, read_numeric_sample, ort_argmax, predict_outputs};
// use rand::{Rng, thread_rng};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // let mut rng = thread_rng();
    // let input_data: Vec<f32> = (0..28*28).map(|_| rng.gen_range(0.0..1.0)).collect::<Vec<f32>>();
    let session = create_onnx_session("models/lt_fnn_model.onnx")?;
    let (input_data, input_label) = read_numeric_sample("data/mnist_sample_row.csv")?;
    let outputs = predict_outputs(&session, input_data, &[1, 28*28])?;
    for output in outputs {
        let output_array = output.try_extract::<f32>()?;
        let predicted_label = ort_argmax(output_array);
        println!("Predicted label: {}", predicted_label);
        println!("Actual label: {}", input_label);
    }

    Ok(())
}