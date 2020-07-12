import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def generate_data(n_data, signal_vector=(1.0,0.0), distractor_vector=(1.0,1.0)):

    y = np.expand_dims(np.linspace(-1.0, 1.0, n_data),-1)
    eps = np.random.randn(n_data,1)
    s = np.expand_dims(signal_vector,0) * y
    d = np.expand_dims(distractor_vector,0) * eps
    return s+d, y

def get_signal_coeff(x, y):
    matrix = np.transpose(np.concatenate((x,y), axis=1))
    coeff = np.cov(matrix, ddof=0) / np.var(y)
    return coeff[-1,:2]

distractor_vector = (-2.0, 1.0)
signal_vector = (0.5, -0.5)

X, y = generate_data(50, signal_vector=signal_vector,
                     distractor_vector=distractor_vector)
reg = LinearRegression().fit(X, y)

weights = reg.coef_[0]

sig_coeff = get_signal_coeff(X, y)
sig = sig_coeff * weights

# plot
plt.scatter(X[:,0],X[:,1], label='data')
plt.quiver(0.0, 0.0, signal_vector[0], signal_vector[1], color='r', label='signal', angles='xy', scale_units='xy', scale=1.)
plt.quiver(0.0, 0.0, distractor_vector[0], distractor_vector[1], color='g', label='distractor', angles='xy', scale_units='xy', scale=1.)
plt.quiver(0.0, 0.0, weights[0], weights[1], color='b', label='weights', angles='xy', scale_units='xy', scale=1.)
plt.quiver(0.0, 0.0, sig[0], sig[1], color='y', label='signal reconstruction', angles='xy', scale_units='xy', scale=1.)
plt.legend()
plt.axis('equal')
plt.show()