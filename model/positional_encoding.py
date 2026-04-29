import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.
        PE = np.arange(0,d_model)
        PE = PE * np.ones(shape=(seq_len,d_model))
        for i in range(seq_len):
            for j in range(d_model):
                if j % 2 ==0:
                    PE[i,j] = np.sin(i/(10000**(j/d_model)))
                else:
                    PE[i,j] = np.cos(i/(10000**((j-1)/d_model)))
        
        return np.round(PE,5)
        pass

