# Multi-Layer Perceptron and Backpropagation
# This code is contributed by Surya Kant Sahu (https://ojus1.github.io)
import numpy as np

# Logic AND operation dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

# Forward Pass
def forward(x, w1, b1, w2, b2):
    h = np.dot(x, w1) + b1
    return np.dot(h, w2) + b2, h

# Hyperparameters
hidden_dim = 4
alpha = 0.1 # learning rate
NUM_EPOCHS = 100

# Loss function
loss_fn = lambda y, y_hat: 0.5 * (y - y_hat) * (y - y_hat)

# Weights and Biases
w1 = np.zeros((2, hidden_dim))
w2 = np.zeros((hidden_dim, 1))
b1 = np.zeros((hidden_dim))
b2 = np.zeros((1))

# Backpropagation Algorithm
for ep in range(NUM_EPOCHS):
    y_hat, h = forward(X, w1, b1, w2, b2)
    loss = np.mean(loss_fn(y, y_hat), axis=0)
    
    #print(y.shape, y_hat.shape, h.shape)
    w2 -= alpha * np.dot(h.T, (y-y_hat)) * (-1)
    b2 -= alpha * np.mean(y-y_hat, axis=0) * (-1)
    w1 -= alpha * np.dot(np.dot(X.T, w2), (y-y_hat).T) * (-1) 
    b1 -= alpha * np.mean(np.dot((y-y_hat), w2.T), axis=0) * (-1)
    

    print(f"Epoch: [{ep+1}/{NUM_EPOCHS}], loss: {loss}")