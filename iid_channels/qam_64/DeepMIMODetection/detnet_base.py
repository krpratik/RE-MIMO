from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class DetNet_base(nn.Module):

    def __init__(self, x_size, v_size, z_size):
        super(DetNet_base, self).__init__()
        self.x_size = x_size
        self.v_size = v_size
        self.z_size = z_size

        self.linear1 = nn.Linear(3*self.x_size+v_size, self.z_size)
        self.linear2 = nn.Linear(self.z_size, self.x_size)
        self.linear3 = nn.Linear(self.z_size, self.v_size)

        nn.init.normal_(self.linear1.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.linear2.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.linear3.weight, mean=0.0, std=0.01)

        self.t = nn.Parameter(torch.tensor(0.1))

    def piecewise_linear_soft_sign(self, x):
        out = -1 + F.relu(x+self.t)/(torch.abs(self.t)+0.00001)-torch.relu(x-self.t)/(torch.abs(self.t)+0.00001)
        return out

    def forward(self, input):
        out = F.relu(self.linear1(input))
        x_out = self.piecewise_linear_soft_sign(self.linear2(out))
        v_out = self.linear3(out)
        return (x_out, v_out)