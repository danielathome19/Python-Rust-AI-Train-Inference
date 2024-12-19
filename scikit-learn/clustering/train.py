import onnx
import skl2onnx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from skl2onnx.common.data_types import FloatTensorType

# Load the iris dataset
iris = load_iris()
data = iris.data

# Define and train a clustering model
model = KMeans(n_clusters=3, random_state=42)
model.fit(data)

# Save the trained model in ONNX format
initial_type = [('float_input', FloatTensorType([None, data.shape[1]]))]
onnx_model = skl2onnx.convert_sklearn(model, initial_types=initial_type, target_opset=10)
with open("models/clustering_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Test the model on the first sample
sample = data[:1]
print(f"Sample: {sample}")
print(f"Prediction: {model.predict(sample)}")
