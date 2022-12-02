import numpy as np

# Data
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
Y_train = np.array([460, 232, 178])

# Cost function
def cost_function(w, b, X, y):
    m = y.shape[0]
    f_wb = np.dot(w.reshape(1,-1), X.T) + b
    cost = (np.sum(f_wb - y))**2
    return cost/(2*m)

# Calculate dj_dw, dj_db
def compute_gradient(w, b, X, y):
    m = y.shape[0]
    f_wb = np.dot(w.reshape(1,-1), X.T) + b
    dj_dw = np.dot(f_wb - y, X)
    dj_db = np.sum(f_wb - y)
    return dj_dw/m , dj_db/m

# Calculate gradient
def gradient_descent(w, b, X, y, alpha, num_iters, gradient_function, cost_function):
    W = w.reshape(1,-1)
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(w, b, X, y)
        W -= alpha * dj_dw
        b -= alpha * dj_db
        print(cost_function(w, b, X, y))
    return W, b

# Set data
w_set = np.array([0., 0., 0., 0.])
b_set = 0.
alpha = 6.0e-7
iterations = 1000

W_final, b_final = gradient_descent(w_set, b_set, X_train, Y_train, alpha,iterations, compute_gradient, cost_function)
print(W_final, b_final)

