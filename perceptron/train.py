import numpy as np

class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=10):
        self.W = np.zeros(input_size)
        self.b = 0
        self.alpha = lr
        self.epochs = epochs

    def activate(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = self.W.T.dot(x) + self.b
        return self.activate(z)

    def fit(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                y = self.activate(self.W.T.dot(X[i]) + self.b)
                e = d[i] - y
                self.W = self.W + self.alpha * e * X[i]
                self.b = self.b + self.alpha * e

# Training data for AND gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
d = np.array([0, 0, 0, 1])

# Create and train the Perceptron model
perceptron = Perceptron(input_size=2)
perceptron.fit(X, d)

# Save the trained model
np.save("models/perceptron_weights.npy", perceptron.W)
np.save("models/perceptron_bias.npy", np.array([perceptron.b]))

# Test the trained model
outputs = [perceptron.predict(x) for x in X]
print(outputs)