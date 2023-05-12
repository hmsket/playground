# 参考：https://take-tech-engineer.com/pytorch-parameters-init/

import torch
import torch.nn as nn

import mynet

model = mynet.MyNet()

for param in model.parameters():
    print(param)

# パラメータを0にする
new_bias = nn.Parameter(torch.tensor([[0.]]))
model.linear.bias = new_bias

for param in model.parameters():
    print(param)
