from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from detnet_base import DetNet_base

class DetNet(nn.Module):

	def __init__(self, num_layers, NT, v_size, z_size, res_alpha=0.9, device='cpu'):
		super(DetNet, self).__init__()
		self.num_layers = num_layers
		self.x_size = 2*NT
		self.v_size = v_size
		self.z_size = z_size
		self.res_alpha = res_alpha
		self.detbases = nn.ModuleList([DetNet_base(self.x_size, self.v_size, self.z_size) for i in range(self.num_layers)])
		self.device = device

	def generate_input(self, HTY, HTH, x, v):
		HHX = torch.einsum('ijk,ik->ij', (HTH,x))
		return torch.cat((HTY, x, HHX, v), dim=1)


	def forward(self, batch_size, HTY, HTH):
		x_prev = torch.zeros(batch_size, self.x_size).to(device=self.device)
		v_prev = torch.zeros(batch_size, self.v_size).to(device=self.device)
		x_list=[x_prev]
		v_list=[v_prev]

		for index,detbase in enumerate(self.detbases):
			xout, vout = detbase.forward(self.generate_input(HTY, HTH, x_list[-1], v_list[-1]))
			x_list.append((1-self.res_alpha)*xout + self.res_alpha*x_list[-1])
			v_list.append((1-self.res_alpha)*vout + self.res_alpha*v_list[-1])
		del v_list
		return (x_list[1:])