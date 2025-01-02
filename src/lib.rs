use csv::ReaderBuilder;
use std::{path::Path, sync::Arc, error::Error};
use ndarray::{Array1, ArrayD, CowArray, Dim, IxDynImpl, IxDyn};
use ort::{Environment, Session, SessionBuilder, Value, GraphOptimizationLevel, LoggingLevel, tensor::OrtOwnedTensor};

pub fn create_onnx_session(model_path: &str) -> Result<Session, Box<dyn Error>> {
    // Initialize the environment
    let env = Arc::new(Environment::builder()
        .with_name("model_inference")
        .with_log_level(LoggingLevel::Verbose)  // Warning
        .build()?);

    // Load the ONNX model and create a session
    let model_file_path = Path::new(model_path);
    let session = SessionBuilder::new(&env)?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_model_from_file(model_file_path)?;
    
    Ok(session)
}

pub fn predict_outputs(session: &Session, input_data: Vec<f32>, input_shape: &[usize]) -> Result<Vec<Value<'static>>, Box<dyn Error>> {
    let input_array = CowArray::from(ArrayD::from_shape_vec(IxDyn(input_shape), input_data)?);
    let input_tensor_values = vec![Value::from_array(session.allocator(), &input_array)?];
    let outputs = session.run(input_tensor_values)?;
    Ok(outputs)
}

pub fn get_output_names(session: &Session) -> Vec<String> {
    session
        .outputs
        .iter()
        .map(|output| output.name.to_string())
        .collect()
}

pub fn ort_argmax(array: OrtOwnedTensor<'_, f32, Dim<IxDynImpl>>) -> usize {
    array
        .view()
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}

pub fn read_numeric_sample(file_path: &str) -> Result<(Vec<f32>, f32), Box<dyn Error>> {
    // Create a CSV reader
    let mut reader = ReaderBuilder::new()
        .has_headers(false)  // Disable header parsing
        .from_path(file_path)?;

    // Read the first row
    if let Some(result) = reader.records().next() {
        let record = result?;
        // Convert all but the last column to f32
        let numbers: Vec<f32> = record
            .iter()
            .take(record.len() - 1)  // All but the last column
            .filter_map(|s| s.trim().parse::<f32>().ok())
            .collect();

        // Parse the last column (the label)
        if let Some(last_value) = record.iter().last() {
            let label = last_value.trim().parse::<f32>()?;
            return Ok((numbers, label));
        }
    }

    Err("File is empty or invalid format".into())
}

pub fn read_tabular_sample(file_path: &str) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    // Read in CSV file with tabular data; return a 2D array of floats
    let mut reader = ReaderBuilder::new()
        .has_headers(false)   // Disable header parsing
        .from_path(file_path)?;

    let mut data = Vec::new();
    for result in reader.records() {
        let record = result?;
        let numbers: Vec<f64> = record
            .iter()
            .filter_map(|s| s.trim().parse::<f64>().ok())
            .collect();
        data.push(numbers);
    }

    Ok(data)
}

pub fn dot(v1: &Array1<f64>, v2: &[f64]) -> f64 {
    v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
}
