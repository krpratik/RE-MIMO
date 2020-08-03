import torch
import numpy as np
import torch.nn as nn
import pickle
import os
import math
import torch.nn.functional as F
import time

from collections import defaultdict
from sample_generator import sample_generator
from iterative_classifier import iterative_classifier
from oampnet import oampnet

# Parameters
NR = 64
mod_n = 16
d_transmitter_encoding = NR
d_model = 512
n_head = 8
nhid = d_model*4
nlayers = 12
dropout = 0.0

# OAMPNet layers
num_layers = 10

# Batch sizes for training and validation sets
time_batch_size = 200
time_iter = 2000

M = int(np.sqrt(mod_n))
sigConst = np.linspace(-M+1, M-1, M) 
sigConst /= np.sqrt((sigConst ** 2).mean())
sigConst /= np.sqrt(2.) #Each complex transmitted signal will have two parts

time_NT_list = NR * np.linspace(0.125, 0.5, 8)
time_NT_list = time_NT_list.astype(int)

snrdb_classical_list =  {NT:np.array([20.0]) for NT in time_NT_list}

model_network_filename = './validtn_results/model.pth'
model_oampnet_filename = './../oamp_net/validtn_results/oampnet_q16_24_64.pth'
oampnet_timing_filename = './final_results/oampnet_timing_results_64.pickle'
network_timing_filename = './final_results/network_timing_results_64.pickle'


def generate_big_time_data(generator, batch_size):
	time_data_dict = {int(NT):{} for NT in time_NT_list}
	for NT in time_NT_list:
		for snr in snrdb_classical_list[NT]:
			big_time_H, big_time_y, big_time_j_indices, big_noise_sigma = generator.give_batch_data(int(NT), snr_db_min=snr, snr_db_max=snr, batch_size=batch_size)
			time_data_dict[int(NT)][snr] = (big_time_H, big_time_y , big_time_j_indices, big_noise_sigma)
	return time_data_dict

def sym_detection(x_hat, real_QAM_const, imag_QAM_const):
	x_real, x_imag = torch.chunk(x_hat, 2, dim=-1)
	x_real = x_real.unsqueeze(dim=-1).expand(-1,-1, real_QAM_const.numel())
	x_imag = x_imag.unsqueeze(dim=-1).expand(-1, -1, imag_QAM_const.numel())

	x_real = torch.pow(x_real - real_QAM_const, 2)
	x_imag = torch.pow(x_imag - imag_QAM_const, 2)
	x_dist = x_real + x_imag
	x_indices = torch.argmin(x_dist, dim=-1)
	return x_indices

def time_model_given_data(model_network, model_oampnet, time_H, time_y, time_j_indices, time_noise_sigma, real_QAM_const, imag_QAM_const, device):
	with torch.no_grad():
		time_H = time_H.to(device=device)
		time_y = time_y.to(device=device)
		time_noise_sigma = time_noise_sigma.to(device=device)

		# Time network
		start_time = time.time()
		_ = model_network.forward(time_H, time_y, time_noise_sigma)
		end_time = time.time()
		network_runtime = (end_time - start_time)

		# Time OAMPNet
		start_time = time.time()
		list_batch_x_predicted = model_oampnet.forward(time_H, time_y, time_noise_sigma)
		time_out = list_batch_x_predicted[-1].to(device='cpu')
		indices_oampnet = sym_detection(time_out, real_QAM_const, imag_QAM_const)
		end_time = time.time()
		oampnet_runtime = (end_time - start_time)

		del time_H, time_y, time_out, time_noise_sigma, list_batch_x_predicted

	return network_runtime, oampnet_runtime


def time_models(model_network, model_oampnet, generator, device):
	oampnet_time_result_dict = {int(NT):defaultdict(float) for NT in time_NT_list}
	network_time_result_dict = {int(NT):defaultdict(float) for NT in time_NT_list}
	for iter in range(time_iter):
		time_data_dict = generate_big_time_data(generator, time_batch_size)
		for NT in time_NT_list:
			for snr in snrdb_classical_list[NT]:
				big_time_H, big_time_y, big_time_j_indices, big_noise_sigma = time_data_dict[NT][snr]
				network_runtime, oampnet_runtime = time_model_given_data(model_network, model_oampnet, big_time_H, big_time_y, big_time_j_indices, big_noise_sigma, generator.real_QAM_const, generator.imag_QAM_const, device)
				network_time_result_dict[NT][snr] = network_time_result_dict[NT][snr] + (network_runtime - network_time_result_dict[NT][snr])/float(iter+1.0)
				oampnet_time_result_dict[NT][snr] = oampnet_time_result_dict[NT][snr] + (oampnet_runtime - oampnet_time_result_dict[NT][snr])/float(iter+1.0)

		print('model_network_mean : ', network_time_result_dict)
		print('model_oampnet_mean : ', oampnet_time_result_dict)

		with open(network_timing_filename, 'wb') as handle:
			pickle.dump(network_time_result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
		print('Intermediate Timing results for RE-MIMO is saved at the directory : ', network_timing_filename)

		with open(oampnet_timing_filename, 'wb') as handle:
			pickle.dump(oampnet_time_result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
		print('Intermediate Timing results for OAMPNet is saved at the directory : ', oampnet_timing_filename)


def test(model_network, model_oampnet, generator, device):
	model_network.eval()
	model_oampnet.eval()

	# Validating Trained Network
	time_models(model_network, model_oampnet, generator, device)

def main():
	generator = sample_generator(time_batch_size, mod_n, NR)
	device = 'cuda'
	model_network = iterative_classifier(d_model, n_head, nhid, nlayers, mod_n, NR, d_transmitter_encoding, generator.real_QAM_const, generator.imag_QAM_const, generator.constellation, device, dropout)
	model_oampnet = oampnet(num_layers, generator.constellation, generator.real_QAM_const, generator.imag_QAM_const, device=device)

	model_network = model_network.to(device=device)
	model_oampnet = model_oampnet.to(device=device)

	network_checkpoint = torch.load(model_network_filename)

	model_network.load_state_dict(network_checkpoint['model_state_dict'])
	model_oampnet.load_state_dict(torch.load(model_oampnet_filename))
	print('*******Successfully loaded pre-trained model***********')

	test(model_network, model_oampnet, generator, device)
	print('******************************** Now Testing **********************************************')

if __name__ == '__main__':
	main()
