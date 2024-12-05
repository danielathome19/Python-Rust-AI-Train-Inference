import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import onnx
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType

# Load the housing dataset
url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
housing = pd.read_csv(url)

# Preprocess the data
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
housing = pd.get_dummies(housing, columns=["ocean_proximity"])

# Split the data into training and testing sets
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
train_labels = train_set["median_house_value"].copy()
train_set = train_set.drop("median_house_value", axis=1)
test_labels = test_set["median_house_value"].copy()
test_set = test_set.drop("median_house_value", axis=1)

# Define and train a regression model
model = LinearRegression()
model.fit(train_set, train_labels)

# Save the trained model in ONNX format
initial_type = [('float_input', FloatTensorType([None, train_set.shape[1]]))]
onnx_model = skl2onnx.convert_sklearn(model, initial_types=initial_type)
with open("regression_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
