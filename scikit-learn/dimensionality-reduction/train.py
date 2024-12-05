import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import onnx
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType

# Load a suitable dataset
digits = load_digits()
data = digits.data

# Define and train a dimensionality reduction model
model = PCA(n_components=2)
model.fit(data)

# Save the trained model in ONNX format
initial_type = [('float_input', FloatTensorType([None, data.shape[1]]))]
onnx_model = skl2onnx.convert_sklearn(model, initial_types=initial_type)
with open("dimensionality_reduction_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
