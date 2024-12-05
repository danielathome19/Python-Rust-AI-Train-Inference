import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import onnx
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType

# Load the iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
iris = pd.read_csv(url, header=None, names=column_names)

# Preprocess the data
iris["class"] = iris["class"].astype('category').cat.codes

# Split the data into training and testing sets
train_set, test_set = train_test_split(iris, test_size=0.2, random_state=42)
train_labels = train_set["class"].copy()
train_set = train_set.drop("class", axis=1)
test_labels = test_set["class"].copy()
test_set = test_set.drop("class", axis=1)

# Define and train a classification model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_set, train_labels)

# Save the trained model in ONNX format
initial_type = [('float_input', FloatTensorType([None, train_set.shape[1]]))]
onnx_model = skl2onnx.convert_sklearn(model, initial_types=initial_type)
with open("classification_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())