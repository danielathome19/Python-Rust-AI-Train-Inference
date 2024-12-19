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
train_set, test_set = train_test_split(iris, test_size=0.2)  #, random_state=42)
train_labels = train_set["class"].copy()
train_set = train_set.drop("class", axis=1)
test_labels = test_set["class"].copy()
test_set = test_set.drop("class", axis=1)

# Grab the first sample from the dataset
with open("data/iris_sample_row.csv", "w") as f:
    f.write(", ".join(f"{x:.1f}" for x in train_set.iloc[0].tolist()) + 
            ", " + str(train_labels.iloc[0]) + "\n")

# Define and train a classification model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_set, train_labels)

# Save the trained model in ONNX format
target_opset = 10
initial_type = [('float_input', FloatTensorType([None, train_set.shape[1]]))]
onnx_model = skl2onnx.convert_sklearn(
    model, 
    initial_types=initial_type, 
    target_opset=target_opset,
    options={"zipmap": False, "output_class_labels": False}  # Disable class label output
)
with open("models/classification_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Evaluate the model
accuracy = model.score(test_set, test_labels)
print(f"Model accuracy: {accuracy:.2f}")

# Verify model
model = onnx.load("models/classification_model.onnx")
onnx.checker.check_model(model)
print("The model is valid.")

"""
# Inspect inputs
print("Inputs:")
for input_tensor in model.graph.input:
    print(f"Name: {input_tensor.name}")
    print(f"Type: {input_tensor.type.tensor_type.elem_type}")  # Check the element type
    print(f"Shape: {[(d.dim_value if d.dim_value > 0 else 'None') for d in input_tensor.type.tensor_type.shape.dim]}")

# Inspect outputs
print("\nOutputs:")
for output_tensor in model.graph.output:
    print(f"Name: {output_tensor.name}")
    print(f"Type: {output_tensor.type.tensor_type.elem_type}")  # Check the element type
    print(f"Shape: {[(d.dim_value if d.dim_value > 0 else 'None') for d in output_tensor.type.tensor_type.shape.dim]}")
"""
# Test the model after loading it
import onnxruntime as rt
sess = rt.InferenceSession("models/classification_model.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
predictions = sess.run([label_name], {input_name: test_set.values.astype(np.float32)})[0]
accuracy = np.mean(predictions == test_labels)
print(f"Model accuracy: {accuracy:.2f}")
