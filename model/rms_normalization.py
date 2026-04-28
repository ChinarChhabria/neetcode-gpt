import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        x = np.array(x)
        rms = np.sqrt((np.sum(x**2,axis = 0)/len(x))+ eps)
        final = (x/rms) * gamma
        return np.round(final,4).tolist()
        pass

