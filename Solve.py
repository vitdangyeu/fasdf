import numpy as np
import matplotlib.pyplot as plt
from utils import *

# Data
x_train, y_train = load_data()
# Plot data
plt.scatter(x_train, y_train, marker = 'x', c = 'r')
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000")
plt.show()

# Cost_function
def cost_function(x, y, w, b):
    m = x.shape[0]
    for i in range(m):
        f_wb = w*x + b
        cost = (np.sum(w*x + b - y))**2
    return cost/(2*m)

# Calculate dj_dw, dj_db
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    f_wb = w*x + b
    dj_dw = np.dot((f_wb - y), x)
    dj_db = np.sum(f_wb - y)
    return dj_dw/m, dj_db/m

# Calculate gradient
def gradient_descent(x, y, w, b, num_iters, alpha, gradien_function, cost_function):
    J_hist = []
    for i in range(num_iters):
        dj_dw, dj_db = gradien_function(x, y, w, b)
        w -= alpha*dj_dw
        b -= alpha*dj_db
        J_hist.append(cost_function(x, y, w, b))
    return w, b, J_hist

# Set data
alpha = 5.0e-3
iteration = 5000
w = 0
b = 0
w_final, b_final, J_hist = gradient_descent(x_train, y_train, w, b, iteration, alpha, compute_gradient, cost_function)

# Print
print(f"w = {w_final}\nb = {b_final}")

# Plot cost function
plt.plot(np.arange(iteration), J_hist)
plt.show()
print(J_hist)

# Plot prediction
plt.scatter(x_train, y_train, marker = 'x', c = 'r')
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000")
plt.plot(x_train, w_final * x_train + b_final, c = 'blue')
plt.show()

