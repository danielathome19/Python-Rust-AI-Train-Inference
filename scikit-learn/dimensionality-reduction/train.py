import onnx
import skl2onnx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from skl2onnx.common.data_types import FloatTensorType

# Load a suitable dataset
digits = load_digits()
data = digits.data

# Write the dataset to a CSV file for later inference testing
df = pd.DataFrame(data)
df.to_csv("data/digits.csv", index=False)

# Define and train a dimensionality reduction model
model = PCA(n_components=2)
model.fit(data)

# Save the trained model in ONNX format
initial_type = [('float_input', FloatTensorType([None, data.shape[1]]))]
onnx_model = skl2onnx.convert_sklearn(model, initial_types=initial_type, target_opset=10)
with open("models/dimensionality_reduction_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Evaluate the model
transformed_data = model.transform(data)
print(f"Original data shape: {data.shape}")
print(f"Transformed data shape: {transformed_data.shape}")
