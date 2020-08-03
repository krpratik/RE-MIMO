import torch
import numpy as np
import torch.nn as nn
import pickle
import os
import math
import torch.nn.functional as F

from collections import defaultdict
from sample_generator import sample_generator
from oampnet import oampnet

# Parameters
NT = 16
NR = 64

mod_n = 16
num_layers = 10

# Batch sizes for training and validation sets
validtn_batch_size = 5000
validtn_iter = 1000

M = int(np.sqrt(mod_n))
sigConst = np.linspace(-M+1, M-1, M)
sigConst /= np.sqrt((sigConst ** 2).mean())
sigConst /= np.sqrt(2.) #Each complex transmitted signal will have two parts

validtn_NT_list = np.asarray([NT])
snrdb_classical_list = {16:np.arange(9.0, 15.0), 32:np.arange(11.0, 17.0), 23:np.arange(10.0, 16.0), 36:np.arange(12.0, 18.0), 40:np.arange(13.0, 19.0), 44:np.arange(14.0, 20.0)}

model_filename = './validtn_results/oampnet_q' + str(mod_n) + '_' + str(NT) + '_' + str(NR) + '.pth'
oampnet_validtn_filename = './final_results/oampnet_' + str(NT) + '_validtn_results.pickle'

def bit_indices(indices, mod_n):
	real_indices = (indices//np.sqrt(mod_n)).to(dtype=torch.int32)
	imag_indices = (indices%np.sqrt(mod_n)).to(dtype=torch.int32)
	joint_bit_indices = torch.cat((real_indices, imag_indices), dim=-1)
	return joint_bit_indices

def sym_accuracy(out, j_indices):
	accuracy = (out == j_indices).sum().to(dtype=torch.float32)
	return accuracy.item()/out.numel()

def bit_accuracy(out, j_indices):
	bit_out_indices = bit_indices(out, mod_n)
	bit_j_indices = bit_indices(j_indices, mod_n)
	return sym_accuracy(bit_out_indices, bit_j_indices)

def sym_detection(x_hat, j_indices, real_QAM_const, imag_QAM_const):
	x_real, x_imag = torch.chunk(x_hat, 2, dim=-1)
	x_real = x_real.unsqueeze(dim=-1).expand(-1,-1, real_QAM_const.numel())
	x_imag = x_imag.unsqueeze(dim=-1).expand(-1, -1, imag_QAM_const.numel())

	x_real = torch.pow(x_real - real_QAM_const, 2)
	x_imag = torch.pow(x_imag - imag_QAM_const, 2)
	x_dist = x_real + x_imag
	x_indices = torch.argmin(x_dist, dim=-1)

	return x_indices

def generate_big_validtn_data(generator, batch_size):
	validtn_data_dict = {int(NT):{} for NT in validtn_NT_list}
	for NT in validtn_NT_list:
		for snr in snrdb_classical_list[NT]:
			big_validtn_H, big_validtn_y, _, big_validtn_j_indices, big_noise_sigma = generator.give_batch_data(int(NT), snr_db_min=snr, snr_db_max=snr, batch_size=batch_size)
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


def validate_oampnet(model, generator, device, real_QAM_const, imag_QAM_const, save_result=True):
	result_dict = {int(NT):defaultdict(float) for NT in validtn_NT_list}
	for iter in range(validtn_iter):
		validtn_data_dict = generate_big_validtn_data(generator, validtn_batch_size)
		for NT in validtn_NT_list:
			for snr in snrdb_classical_list[NT]:
				big_validtn_H, big_validtn_y, big_validtn_j_indices, big_noise_sigma = validtn_data_dict[NT][snr]
				accr = validate_model_given_data(model, big_validtn_H, big_validtn_y, big_validtn_j_indices, big_noise_sigma, real_QAM_const, imag_QAM_const, device)
				result_dict[NT][snr] =  result_dict[NT][snr] + (accr-result_dict[NT][snr])/float(iter+1.0)

		if (save_result):
			with open(oampnet_validtn_filename, 'wb') as handle:
				pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
			print('Intermediate Test results saved at : ', oampnet_validtn_filename)
		print('Big Validation resut, Accr for ' + str(NT) + ' : ', result_dict[NT])


def test(model, generator, device):
	model.eval()

	# Testing Trained Network
	validate_oampnet(model, generator, device, generator.real_QAM_const, generator.imag_QAM_const, True)

def main():
	generator = sample_generator(validtn_batch_size, mod_n, NR)
	device = 'cuda'
	model = oampnet(num_layers, generator.constellation, generator.real_QAM_const, generator.imag_QAM_const, device=device)
	model = model.to(device=device)
	model.load_state_dict(torch.load(model_filename))
	print('*******Successfully loaded pre-trained model*********** from directory : ', model_filename)

	test(model, generator, device)
	print('******************************** Now Testing **********************************************')

if __name__ == '__main__':
	main()
