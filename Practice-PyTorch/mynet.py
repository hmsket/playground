import torch
import torch.nn as nn

class MyNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)
    
    def forward(self, x):
        x = self.linear(x)
        return x
