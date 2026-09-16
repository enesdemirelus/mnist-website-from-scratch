from fastapi import FastAPI
import numpy as np


app = FastAPI()
weights = np.load('apps/ml/model/mnist_weights.npz')
W1, B1, W2, B2 = weights['W1'], weights['B1'], weights['W2'], weights['B2']

def ReLU(X):
    return np.maximum(X, 0)

def softmax_calc(Z):
    return np.exp(Z) / sum(np.exp(Z))

def get_predictions(A2):
    return np.argmax(A2, 0)

def predict(W1, B1, W2, B2, X):
    Z1 = W1.dot(X) + B1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = softmax_calc(Z2)
    return get_predictions(A2)

@app.post("/predict")
def predict_digit(body: dict = Body(...)):
    pixels = body["pixels"]
    X = np.array(pixels).reshape(784, 1) / 255.0
    prediction = predict(W1, B1, W2, B2, X)
    return {"prediction": int(prediction[0])}