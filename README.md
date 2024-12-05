# Various Python train/Rust inference examples for different ML/DL libraries

This repository demonstrates various machine learning and deep learning models using different libraries. Each example includes a Python script to define and train the model, and a Rust file to demonstrate inference. The models are saved in ONNX format to ensure they can be reloaded and used for inference in Rust.

## Repository Structure

The repository is organized as follows:

- `scikit-learn/`
  - `regression/`
  - `classification/`
  - `clustering/`
  - `dimensionality-reduction/`
- `tensorflow/`
  - `mlp/`
  - `cnn/`
  - `rnn/`
  - `lstm/`
  - `gan/`
- `lightning/`
  - `fnn/`
  - `cnn/`
  - `rnn/`
- `perceptron/`

## Instructions

### Running the Python Training Scripts

1. Navigate to the desired example directory (e.g., `scikit-learn/regression/`).
2. Run the Python training script to train the model and save it in ONNX format:
   ```bash
   python train.py
   ```

### Running the Rust Inference Scripts

1. Navigate to the desired example directory (e.g., `scikit-learn/regression/`).
2. Ensure you have Rust and the `onnxruntime` crate installed.
3. Run the Rust inference script to load the ONNX model and perform inference on new data:
   ```bash
   cargo run --release
   ```
