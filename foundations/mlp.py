import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        x = x.reshape(-1,1)
        for i in range(len(weights)):
            W = np.array(weights[i])
            b = np.array(biases[i]).reshape(-1,1)
            if i == 0:
                f = np.dot(W.T,x) + b    
            else:
                f = np.dot(W.T,a) + b
            if i<len(weights):
                a = np.where(f>0,f,0)
        
        
        return np.round(f.flatten(),5)        
             

        pass
