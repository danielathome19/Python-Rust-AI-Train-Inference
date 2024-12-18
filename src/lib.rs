use std::error::Error;
use csv::ReaderBuilder;
use ort::tensor::OrtOwnedTensor;
use ndarray::{Dim, IxDynImpl};

pub fn ort_argmax(array: OrtOwnedTensor<'_, f32, Dim<IxDynImpl>>) -> usize {
    array
        .view()
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}

pub fn read_numeric_sample(file_path: &str) -> Result<(Vec<f32>, i32), Box<dyn Error>> {
    // Create a CSV reader
    let mut reader = ReaderBuilder::new()
        .has_headers(false) // Disable header parsing
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

        // Parse the last column as i32 (the label)
        if let Some(last_value) = record.iter().last() {
            let label = last_value.trim().parse::<i32>()?;
            return Ok((numbers, label));
        }
    }

    Err("File is empty or invalid format".into())
}
