import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x, gamma, beta, running_mean, running_var,
               momentum, eps, training):

        x = np.array(x)
        gamma = np.array(gamma)
        beta = np.array(beta)
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)

        if training:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)

            x_hat = (x - batch_mean) / np.sqrt(batch_var + eps)
            y = gamma * x_hat + beta

            running_mean = (1 - momentum) * running_mean + momentum * batch_mean
            running_var = (1 - momentum) * running_var + momentum * batch_var

        else:
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)
            y = gamma * x_hat + beta

        return (
            np.round(y, 4).tolist(),
            np.round(running_mean, 4).tolist(),
            np.round(running_var, 4).tolist()
        )

