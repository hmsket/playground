import torch
import torch.nn as nn

class MyNet(nn.Module):

    def __init__(self, ni, nh, no):
        super().__init__()
        self.layer1 = nn.Linear(ni, nh)
        self.act_fn = nn.Tanh()
        self.layer2 = nn.Linear(nh, no)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act_fn(x)
        x = self.layer2(x)
        return x
