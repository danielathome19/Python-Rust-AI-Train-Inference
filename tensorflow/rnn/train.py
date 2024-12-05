import tensorflow as tf
import numpy as np
import onnx
import tf2onnx

# Load a suitable dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define and train an RNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(128, input_shape=(28, 28)),
    tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

# Save the trained model in ONNX format
spec = (tf.TensorSpec((None, 28, 28), tf.float32, name="input"),)
output_path = "rnn_model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())
