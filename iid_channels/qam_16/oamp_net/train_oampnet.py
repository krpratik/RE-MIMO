#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time as tm
import math
import sys
import pickle as pkl

from oampnet import oampnet
from sample_generator import sample_generator


#parameters
NT = 16
NR = 64

snrdb_classical_list = {16:np.arange(8.0, 16.0), 32:np.arange(10.0, 18.0), 44:np.arange(13.0, 21.0), 23:np.arange(9.0, 17.0), 36:np.arange(11.0, 19.0), 40:np.arange(12.0, 20.0)}

num_layers = 10
train_iter = 25000
train_batch_size = 1000
learning_rate = 1e-3

num_snr = 8
mod_n = 16

model_filename = './validtn_results/oampnet_q' + str(mod_n) + '_' + str(NT) + '_' + str(NR) + '.pth'
curr_accr = './validtn_results/curr_accr_' + str(NT) + '.txt'


def sym_detection(x_hat, j_indices, real_QAM_const, imag_QAM_const):
	x_real, x_imag = torch.chunk(x_hat, 2, dim=-1)
	x_real = x_real.unsqueeze(dim=-1).expand(-1,-1, real_QAM_const.numel())
	x_imag = x_imag.unsqueeze(dim=-1).expand(-1, -1, imag_QAM_const.numel())

	x_real = torch.pow(x_real - real_QAM_const, 2)
	x_imag = torch.pow(x_imag - imag_QAM_const, 2)
	x_dist = x_real + x_imag
	x_indices = torch.argmin(x_dist, dim=-1)

	accuracy = (x_indices == j_indices).sum().to(dtype=torch.float32)
	return accuracy.item()/j_indices.numel()


def loss_fn(x, list_batch_x_predicted, j_indices, real_QAM_const, imag_QAM_const, criterion, ser_only=False):
	if (ser_only):
		SER_final = sym_detection(list_batch_x_predicted[-1], j_indices, real_QAM_const, imag_QAM_const)
		return SER_final
	else:
		x_out = torch.cat(list_batch_x_predicted, dim=0)
		x = x.repeat(num_layers, 1)
		loss = criterion(x_out, x)
		SER_final = sym_detection(list_batch_x_predicted[-1], j_indices, real_QAM_const, imag_QAM_const)
		return loss, SER_final


def train(model, optimizer, generator, device='cpu'):
	criterion = nn.MSELoss().to(device=device)
	model.train()
	real_QAM_const = generator.real_QAM_const.to(device=device)
	imag_QAM_const = generator.imag_QAM_const.to(device=device)

	for i in range(train_iter):

		H, y, x, j_indices, noise_sigma = generator.give_batch_data(NT, snr_db_min=snrdb_classical_list[NT][0], snr_db_max=snrdb_classical_list[NT][-1], batch_size=train_batch_size)
		H = H.to(device=device)
		y = y.to(device=device)
		noise_sigma = noise_sigma.to(device=device)

		list_batch_x_predicted = model.forward(H, y, noise_sigma)

		x = x.to(device=device)
		j_indices = j_indices.to(device=device)

		loss, SER = loss_fn(x, list_batch_x_predicted, j_indices, real_QAM_const, imag_QAM_const, criterion)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		del H, y, x, j_indices, noise_sigma, list_batch_x_predicted

		if (i%1000==0):
			model.eval()
			H, y, x, j_indices, noise_sigma = generator.give_batch_data(NT, snr_db_min=snrdb_classical_list[NT][0], snr_db_max=snrdb_classical_list[NT][-1], batch_size=train_batch_size)
			H = H.to(device=device)
			y = y.to(device=device)
			noise_sigma = noise_sigma.to(device=device)
			with torch.no_grad():
				list_batch_x_predicted = model.forward(H, y, noise_sigma)
				x = x.to(device=device)
				j_indices = j_indices.to(device=device)
				loss_last, SER_final = loss_fn(x, list_batch_x_predicted, j_indices, real_QAM_const, imag_QAM_const, criterion)
				results = [loss_last.detach().item(), SER_final]
				print_string = [i]+results
				print(' '.join('%s' % np.round(x,6) for x in print_string))
				with open(curr_accr, 'w') as f:
					print((i, print_string), file=f)
				print('Intermediate validation results saved at : ', curr_accr)
			del H, y, x, j_indices, noise_sigma, list_batch_x_predicted
			model.train()
			torch.save(model.state_dict(), model_filename)
			print('****************Model Saved*************** at directory : ', model_filename)


def main():
	device = 'cuda'
	generator = sample_generator(train_batch_size, mod_n, NR)
	model = oampnet(num_layers, generator.constellation, generator.real_QAM_const, generator.imag_QAM_const, device=device)
	model = model.to(device=device)

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	train(model, optimizer, generator, device)
	print('******************************** Now Testing **********************************************')

if __name__ == '__main__':
	main()
