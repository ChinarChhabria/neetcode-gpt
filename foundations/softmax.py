import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z -= np.max(z)
        answer = np.zeros(len(z))
        for i in range(len(z)):
            answer[i] = np.exp(z[i])/np.sum(np.exp(z))
        return np.round(answer,4)

        pass
