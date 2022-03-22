import numpy as np
import math

e = math.e

np.random.seed(0)
inputs = [[1, 2, 3, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]]

def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y 

import matplotlib.pyplot as plt

print("here")
# X, y = create_data(100,3)

# plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.Spectral)
# plt.show()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1,n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples_length = len(y_pred)
        # clip to avoid doing lof of zer as it is infinity
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            # picking the values from the output according to the y-true... that will  not be multiplied by zero
            correct_confidences = y_pred_clipped[range(samples_length), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


X, y = create_data(100,3)
dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()


dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

# Calculate loss
loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss:",loss)




# layer1 = Layer_Dense(2,5)
# activation1 = Activation_ReLU()
# layer1.forward(X)
# print(layer1.output)
# activation1.forward(layer1.output)
# print(activation1.output)
# layer2 = Layer_Dense(5,2)

# layer1.forward(inputs)
# print(layer1.output.shape)
# layer2.forward(layer1.output)
# print(layer2.weights.shape)
# print(layer2.output)