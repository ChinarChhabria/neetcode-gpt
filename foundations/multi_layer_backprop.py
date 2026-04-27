import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        x = np.array(x)
        x = x.reshape(-1,1)
        b1 = np.array(b1)
        f1 = np.dot(W1,x) + b1.reshape(-1,1)
        g1 = np.where(f1>0,f1,0)
        
        f2 = np.dot(W2,g1) + b2
        
        loss = (f2 - y_true)**2/x.shape[1]

        df2 = 2*(f2-y_true)/x.shape[1]
        dw2 = df2 * g1
        db2 = df2
        dg1 = W2 * df2
        dg1 = dg1.reshape(-1,1)
        df1 = np.where(f1>0,dg1,0)
        
        dW1 = np.dot(df1,x.T) 
        db1 = df1
       
        return {
                'loss': np.round(loss[0][0],4),
                'dW1': np.round(dW1.astype(float),4).tolist(),
                'db1': np.round(db1.reshape(1,-1)[0],4),
                'dW2': np.round(dw2.reshape(1,-1).astype(float),4).tolist(),
                'db2': np.round(db2[0],4)
                    }



        pass
