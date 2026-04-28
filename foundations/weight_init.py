import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        sigma = (2/(fan_in + fan_out))**0.5
        weights = torch.randn(fan_out,fan_in) * sigma
        return torch.round(weights,decimals=4).tolist()
        pass 

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        sigma = (2/fan_in)**0.5
        weights = torch.randn(fan_out,fan_in) * sigma
        return torch.round(weights,decimals=4).tolist()
        pass

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:

        torch.manual_seed(0)
        
        # 1. Build weight matrices FIRST to consume the RNG state in the correct order
        weights_list = []
        for i in range(num_layers):
            fan_in = input_dim if i == 0 else hidden_dim
            fan_out = hidden_dim
            
            if init_type == 'kaiming':
                sigma = (2 / fan_in) ** 0.5
            elif init_type == 'xavier':
                sigma = (2 / (fan_in + fan_out)) ** 0.5
            else: # 'random'
                sigma = 1.0
                
            weights = torch.randn(fan_out, fan_in) * sigma
            weights_list.append(weights)
            
        # 2. Generate random input SECOND
        x = torch.randn(input_dim)
        
        # 3. Forward the input through the built matrices
        std_list = []
        for weights in weights_list:
            x = torch.relu(weights @ x)
            std_list.append(round(x.std().item(), 2))
            
        return std_list