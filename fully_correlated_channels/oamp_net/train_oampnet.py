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

num_layers = 10

train_iter = 50000
train_batch_size = 1000
mini_validtn_batch_size = 5000
learning_rate = 1e-3

mod_n = 16
corr_flag = True
batch_corr = True
rho_low = 0.55
rho_high = 0.75

save_interim_model = True
save_to_file = True

model_filename = './validtn_results/oampnet_fullcorr_q' + str(mod_n) + '_' + str(NT) + '_' + str(NR) + '.pth'
curr_accr = './validtn_results/curr_accr_' + str(NT) + '.txt'

validtn_NT_list = np.asarray([NT])
snrdb_list = {16:np.arange(11.0, 22.0), 32:np.arange(16.0, 27.0)}


def sym_accuracy(out, j_indices):
	accuracy = (out == j_indices).sum().to(dtype=torch.float32)
	return accuracy.item()/out.numel()

def sym_detection(x_hat, j_indices, real_QAM_const, imag_QAM_const):
	real_QAM_const = real_QAM_const.to(device=x_hat.device)
	imag_QAM_const = imag_QAM_const.to(device=x_hat.device)
	x_real, x_imag = torch.chunk(x_hat, 2, dim=-1)
	x_real = x_real.unsqueeze(dim=-1).expand(-1,-1, real_QAM_const.numel())
	x_imag = x_imag.unsqueeze(dim=-1).expand(-1, -1, imag_QAM_const.numel())

	x_real = torch.pow(x_real - real_QAM_const, 2)
	x_imag = torch.pow(x_imag - imag_QAM_const, 2)
	x_dist = x_real + x_imag
	x_indices = torch.argmin(x_dist, dim=-1)

	return x_indices

def generate_big_validtn_data(generator, batch_size, corr_flag, rho, batch_corr, rho_low, rho_high):
	validtn_data_dict = {int(NT):{} for NT in validtn_NT_list}
	for NT in validtn_NT_list:
		for snr in snrdb_list[NT]:
			big_validtn_H, big_validtn_y, _, big_validtn_j_indices, big_noise_sigma = generator.give_batch_data(int(NT), snr_db_min=snr, snr_db_max=snr, batch_size=batch_size, correlated_flag=corr_flag, rho=rho, batch_corr=batch_corr,rho_low=rho_low, rho_high=rho_high)
			validtn_data_dict[int(NT)][snr] = (big_validtn_H, big_validtn_y , big_validtn_j_indices, big_noise_sigma)
	return validtn_data_dict


def validate_model_given_data(model, validtn_H, validtn_y, validtn_j_indices, big_noise_sigma, real_QAM_const, imag_QAM_const, device):
	
	with torch.no_grad():
		H = validtn_H.to(device=device)
		y = validtn_y.to(device=device)
		noise_sigma = big_noise_sigma.to(device=device)

		list_batch_x_predicted = model.forward(H, y, noise_sigma)
		validtn_out = list_batch_x_predicted[-1].to(device='cpu')
		indices_oampnet = sym_detection(validtn_out, validtn_j_indices, real_QAM_const, imag_QAM_const)
		accr = sym_accuracy(indices_oampnet, validtn_j_indices)

		del H, y, noise_sigma, list_batch_x_predicted

	return accr

def mini_validation(model, mini_validation_dict, i, device, real_QAM_const, imag_QAM_const, save_to_file=True):
	result_dict = {int(NT):{} for NT in validtn_NT_list}
	for index,NT in enumerate(validtn_NT_list):
		for snr in snrdb_list[NT]:
			big_validtn_H, big_validtn_y, big_validtn_j_indices, big_noise_sigma = mini_validation_dict[NT][snr]
			accr = validate_model_given_data(model, big_validtn_H, big_validtn_y, big_validtn_j_indices, big_noise_sigma, real_QAM_const, imag_QAM_const, device)
			result_dict[NT][snr] = accr

	if (save_to_file):
		with open(curr_accr, 'w') as f:
			print((i, result_dict), file=f)
			print('Intermediate validation results stored at : ', curr_accr)

	return result_dict


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

	mini_validation_dict = generate_big_validtn_data(generator, mini_validtn_batch_size, corr_flag, None, batch_corr, rho_low, rho_high)

	criterion = nn.MSELoss().to(device=device)
	model.train()
	real_QAM_const = generator.real_QAM_const.to(device=device)
	imag_QAM_const = generator.imag_QAM_const.to(device=device)

	for i in range(train_iter):
		rho = np.random.uniform(rho_low, rho_high)
		H, y, x, j_indices, noise_sigma = generator.give_batch_data(NT, snr_db_min=snrdb_list[NT][0], snr_db_max=snrdb_list[NT][-1], batch_size=train_batch_size, correlated_flag=corr_flag, rho=rho)
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
			print('iteration number : ', i, 'User : ', NT, 'loss : ', loss.item())
			print('Now validating')

			model.eval()
			mini_validtn_result = mini_validation(model, mini_validation_dict, i, device, real_QAM_const, imag_QAM_const, save_to_file)
			print('Mini validation result : ', mini_validtn_result)

			model.train()
			if (save_interim_model):
				torch.save(model.state_dict(), model_filename)
				print('********Model Saved******* at directory : ', model_filename)


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
