import torch
import numpy as np
import torch.nn as nn
import pickle
import os
import math
import torch.nn.functional as F

from collections import defaultdict
from sample_generator import sample_generator
from classical_solvers import mmse, sym_detection

# Parameters
NR = 64
mod_n = 16
d_transmitter_encoding = 32
d_model = 512
n_head = 10
nhid = d_model*4 + 4*2*mod_n
nlayers = 10
dropout = 0.0

# Batch sizes for training and validation sets
validtn_batch_size = 5000
validtn_iter = 2000

M = int(np.sqrt(mod_n))
sigConst = np.linspace(-M+1, M-1, M) 
sigConst /= np.sqrt((sigConst ** 2).mean())
sigConst /= np.sqrt(2.) #Each complex transmitted signal will have two parts

validtn_NT_list = np.asarray([16, 32])
snrdb_classical_list = {16:np.arange(9.0, 15.0), 32:np.arange(11.0, 17.0)}

mmse_validtn_filename = './final_results/mmse_validtn_results.pickle'

def accuracy(out, j_indices):
	out = out.permute(1,2,0)
	out = out.argmax(dim=1)
	accuracy = (out == j_indices).sum().to(dtype=torch.float32)
	return accuracy.item()/out.numel()

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

def generate_big_validtn_data(generator, batch_size):
	validtn_data_dict = {int(NT):{} for NT in validtn_NT_list}
	for NT in validtn_NT_list:
		for snr in snrdb_classical_list[NT]:
			big_validtn_H, big_validtn_y, big_validtn_j_indices, big_noise_sigma = generator.give_batch_data(int(NT), snr_db_min=snr, snr_db_max=snr, batch_size=batch_size)
			validtn_data_dict[int(NT)][snr] = (big_validtn_H, big_validtn_y , big_validtn_j_indices, big_noise_sigma)
	return validtn_data_dict

def validate_mmse_given_data(H, y, big_validtn_j_indices, noise_sigma, NT, real_QAM_const, imag_QAM_const):

	results_mmse = mmse(y, H, noise_sigma)
	indices_mmse = sym_detection(results_mmse, real_QAM_const, imag_QAM_const)
	mmse_accr = sym_accuracy(indices_mmse, big_validtn_j_indices)
	return mmse_accr


def validate_classical(generator, real_QAM_const, imag_QAM_const, save_result=True):

	mmse_result_dict = {int(NT):defaultdict(float) for NT in validtn_NT_list}
	for iter in range(validtn_iter):
		validtn_data_dict = generate_big_validtn_data(generator, validtn_batch_size)
		for NT in validtn_NT_list:
			for snr in snrdb_classical_list[NT]:
				print(NT, snr)
				big_validtn_H, big_validtn_y, big_validtn_j_indices, big_noise_sigma = validtn_data_dict[NT][snr]
				accr = validate_mmse_given_data(big_validtn_H, big_validtn_y, big_validtn_j_indices, big_noise_sigma, NT, real_QAM_const, imag_QAM_const)
				mmse_result_dict[NT][snr] = mmse_result_dict[NT][snr] + (accr-mmse_result_dict[NT][snr])/float(iter+1.0)

		if (save_result):
			with open(mmse_validtn_filename, 'wb') as handle:
				pickle.dump(mmse_result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
			print('Intermediate Test results saved at : ', mmse_validtn_filename)

		print('mmse results : ', mmse_result_dict)


def test(generator):
	# Validating Trained Network
	validate_classical(generator, generator.real_QAM_const, generator.imag_QAM_const, True)


def main():
	generator = sample_generator(validtn_batch_size, mod_n, NR)

	test(generator)
	print('******************************** Now Testing **********************************************')

if __name__ == '__main__':
	main()
