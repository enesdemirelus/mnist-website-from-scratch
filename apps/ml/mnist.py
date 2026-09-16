import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('apps/ml/dataset/mnist_train.csv')
data = np.array(data)
m, n = data.shape


train_data = data[0:, :]

X_train = train_data[:, 1:].T
X_train = X_train / 255.0
Y_train = train_data[:, 0]


def initialize_parameters():
    W1 = np.random.rand(10, 784) - 0.5
    B1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    B2 = np.random.rand(10, 1) - 0.5
    return W1, B1, W2, B2

def ReLU(X):
    return np.maximum(X, 0)

def softmax_calc(Z):
    return np.exp(Z) / sum(np.exp(Z))

def forward_prop(W1, B1, W2, B2, X):
    Z1 = W1.dot(X) + B1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = softmax_calc(Z2)
    return Z1, A1, Z2, A2

def one_hot_converter(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def backward_prop(W1, B1, W2, B2, Z1, A1, Z2, A2, X, Y):
    one_hot_Y = one_hot_converter(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    dB2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * (Z1 > 0)
    dW1 = 1 / m * dZ1.dot(X.T)
    dB1 = 1 / m * np.sum(dZ1)
    return dW1, dB1, dW2, dB2

def update_parameters(W1, B1, W2, B2, dW1, dB1, dW2, dB2, learning_rate):
    W1 = W1 - learning_rate * dW1
    B1 = B1 - learning_rate * dB1
    W2 = W2 - learning_rate * dW2
    B2 = B2 - learning_rate * dB2
    return W1, B1, W2, B2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iter_count):
    W1, B1, W2, B2 = initialize_parameters()
    
    for i in range(iter_count):
        Z1, A1, Z2, A2 = forward_prop(W1, B1, W2, B2, X)
        dW1, dB1, dW2, dB2 = backward_prop(W1, B1, W2, B2, Z1, A1, Z2, A2, X, Y)
        W1, B1, W2, B2 = update_parameters(W1, B1, W2, B2,dW1, dB1, dW2, dB2, alpha)
        
        if (i % 20 == 0):
            print(f"Iteration number: {i}")
            print(f"Accuracy = {get_accuracy(get_predictions(A2), Y)}")
    
    return W1, B1, W2, B2
    
W1, B1, W2, B2 = gradient_descent(X_train, Y_train, 0.5, 500)
np.savez('apps/ml/mnist_weights.npz', W1=W1, B1=B1, W2=W2, B2=B2)


# test_data = pd.read_csv('apps/ml/dataset/mnist_test.csv')
# test_data = np.array(test_data)
# m_test, n_test = test_data.shape

# test_data = test_data[0:, :]
# X_test = test_data[:, 1:].T
# X_test = X_test / 255.0
# Y_test = test_data[:, 0]

# predictions = get_predictions(forward_prop(W1, B1, W2, B2, X_test)[3])

# fig, axes = plt.subplots(4, 5, figsize=(12, 10))

# for i, ax in enumerate(axes.flat):
#     ax.imshow(X_test[:, i].reshape(28, 28), cmap='gray')
#     ax.set_title(f"Prediction: {predictions[i]}")
#     ax.axis('off')

# plt.tight_layout()
# plt.show()