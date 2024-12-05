import numpy as np

class Perceptron:
    def __init__(self, input_size, lr=1, epochs=10):
        self.W = np.zeros(input_size + 1)
        self.lr = lr
        self.epochs = epochs

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = self.W.T.dot(np.insert(x, 0, 1))
        return self.activation_fn(z)

    def fit(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)
                y = self.activation_fn(self.W.T.dot(x))
                e = d[i] - y
                self.W = self.W + self.lr * e * x

# Training data for AND gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
d = np.array([0, 0, 0, 1])

# Create and train the Perceptron model
perceptron = Perceptron(input_size=2)
perceptron.fit(X, d)

# Save the trained model
np.save("perceptron_weights.npy", perceptron.W)
