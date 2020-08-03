import torch
import numpy as np
import torch.nn as nn
import pickle
import os
import math
import torch.nn.functional as F

from sample_generator import sample_generator
from iterative_classifier import iterative_classifier

# Parameters
NR = 32
mod_n = 16
d_transmitter_encoding = NR
d_model = 512
n_head = 1
nhid = d_model*4
nlayers = 8
dropout = 0.0

NT_list = np.asarray([12])
snrdb_classical_list = {12:[15.0]}
analysis_size = 10

model_filename = './validtn_results/model.pth'
channel_filename = './validtn_results/channel_matrix.pickle'

save_channel_matrix = True


def save_data(data_dict, file_name):
	with open(file_name, 'wb') as handle:
		pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def analyse_model(model, generator, device, save_result=True):
	data_dict = generate_analysis_data(generator, analysis_size)

	if (save_channel_matrix):
		save_data(data_dict, channel_filename)

	for NT in NT_list:
		for snr in snrdb_classical_list[NT]:
			H, y, noise_sigma = data_dict[NT][snr]

			H = H.to(device=device)
			y = y.to(device=device)
			noise_sigma = noise_sigma.to(device=device)
			out = model.forward(H, y, noise_sigma, True)

def generate_analysis_data(generator, batch_size):
	data_dict = {int(NT):{} for NT in NT_list}
	for NT in NT_list:
		for snr in snrdb_classical_list[NT]:
			H, y, _, noise_sigma = generator.give_batch_data(int(NT), snr_db_min=snr, snr_db_max=snr, batch_size=batch_size)
			noise_sigma = noise_sigma.view(analysis_size)
			data_dict[int(NT)][snr] = (H, y, noise_sigma)
	return data_dict

def test(model, generator, device):
	model.eval()

	# Validating Trained Network
	analyse_model(model, generator, device, True)

def main():
	generator = sample_generator(analysis_size, mod_n, NR, NT_list[0])
	device = 'cuda'
	model = iterative_classifier(d_model, n_head, nhid, nlayers, mod_n, NR, d_transmitter_encoding, generator.real_QAM_const, generator.imag_QAM_const, generator.constellation, device, dropout)
	model = model.to(device=device)

	checkpoint = torch.load(model_filename)
	model.load_state_dict(checkpoint['model_state_dict'])
	print('*******Successfully loaded pre-trained model***********')

	test(model, generator, device)
	print('******************************** Now Testing **********************************************')

if __name__ == '__main__':
	main()
