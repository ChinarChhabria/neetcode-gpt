import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))

        n_samples,n_features = X.shape
        w = np.zeros(n_features)
        b = 0
        for i in range(epochs):
            dj_dw = np.dot(2*(np.dot(X,w) + b - y)/n_samples,X)
            dj_db = 2*np.sum((np.dot(X,w)+b - y),axis = 0)/n_samples
            w -= lr * dj_dw
            b -= lr * dj_db
                  
        return (np.round(w, 5), round(b, 5))

        pass
