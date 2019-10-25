# Perceptron and Gradient Descent on pure Python
# This code is contributed by Surya Kant Sahu (https://ojus1.github.io)


# Logic AND operation dataset
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [0, 0, 0, 1]

# Weights of Perceptron model
w = [0, 0]
b = -1

# Hyperparameters
NUM_EPOCHS = 100
LEARNING_RATE = 0.1

# Forward pass
def forward(x, w, b):
    ret = 0
    for xi, wi in zip(x, w):
        ret += xi * wi
    return ret + b

# Loss function
loss_fn = lambda y, y_hat: 0.5 * (y - y_hat) * (y - y_hat)

# Gradient descent algorithm
for ep in range(NUM_EPOCHS):
    loss = 0
    for x, y in zip(X, Y):
        y_hat = forward(x, w, b)
        loss += loss_fn(y, y_hat) / len(Y)

    w[0] -= LEARNING_RATE * (y - y_hat) * (y_hat) * (x[0])
    w[1] -= LEARNING_RATE * (y - y_hat) * (y_hat) * (x[1])
    b -= LEARNING_RATE * (y - y_hat) * (y_hat)

    print(f"Epoch: [{ep+1}/{NUM_EPOCHS}], loss: {loss}, w[0]: {w[0]}, w[1]: {w[1]}, b: {b}")