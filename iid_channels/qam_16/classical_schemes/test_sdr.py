import torch
import numpy as np
import torch.nn as nn
import pickle
import os
import math
import torch.nn.functional as F


from sample_generator import sample_generator
from classical_solvers import sdrSolver, sym_detection

# Parameters
NR = 64
mod_n = 16

# Batch sizes for training and validation sets
validtn_batch_size = 5000
validtn_iter = 2000

M = int(np.sqrt(mod_n))
sigConst = np.linspace(-M+1, M-1, M) 
sigConst /= np.sqrt((sigConst ** 2).mean())
sigConst /= np.sqrt(2.) #Each complex transmitted signal will have two parts

validtn_NT_list = np.asarray([16, 32])
snrdb_classical_list = {16:np.arange(16.0, 22.0), 32:np.arange(18.0, 24.0)}


sdr_validtn_filename = './final_results/sdr_validtn_results.pickle'

def accuracy(out, j_indices):
	out = out.permute(1,2,0)
	out = out.argmax(dim=1)
	accuracy = (out == j_indices).sum().to(dtype=torch.float32)
	return accuracy.item()/out.numel()


def sym_accuracy(out, j_indices):
	accuracy = (out == j_indices).sum().to(dtype=torch.float32)
	return accuracy.item()/out.numel()

def generate_big_validtn_data(generator, batch_size):
	validtn_data_dict = {int(NT):{} for NT in validtn_NT_list}
	for NT in validtn_NT_list:
		for snr in snrdb_classical_list[NT]:
			big_validtn_H, big_validtn_y, big_validtn_j_indices, big_noise_sigma = generator.give_batch_data(int(NT), snr_db_min=snr, snr_db_max=snr, batch_size=batch_size)
			validtn_data_dict[int(NT)][snr] = (big_validtn_H, big_validtn_y , big_validtn_j_indices, big_noise_sigma)
	return validtn_data_dict

def validate_sdr_given_data(H, y, big_validtn_j_indices, noise_sigma, NT, real_QAM_const, imag_QAM_const):
	# Numpy modifications for classical Symbol detection algorithms
	y = y.unsqueeze(dim=-1).numpy()
	H = H.numpy()

	results_sdr = sdrSolver(H, y, sigConst, NT).squeeze()
	indices_sdr = sym_detection(torch.from_numpy(results_sdr), real_QAM_const, imag_QAM_const)

	sdr_accr = sym_accuracy(indices_sdr, big_validtn_j_indices)

	return sdr_accr

def validate_classical(validtn_data_dict, real_QAM_const, imag_QAM_const, save_result=True):

	sdr_result_dict = {int(NT):defaultdict(float) for NT in validtn_NT_list}
	for iter in range(validtn_iter):
		validtn_data_dict = generate_big_validtn_data(generator, validtn_batch_size)
		for NT in validtn_NT_list:
			for snr in snrdb_classical_list[NT]:
				print(NT, snr)
				big_validtn_H, big_validtn_y, big_validtn_j_indices, big_noise_sigma = validtn_data_dict[NT][snr]
				accr = validate_sdr_given_data(big_validtn_H, big_validtn_y, big_validtn_j_indices, big_noise_sigma, NT, real_QAM_const, imag_QAM_const)
				sdr_result_dict[NT][snr] = sdr_result_dict[NT][snr] + (accr-sdr_result_dict[NT][snr])/float(iter+1.0)

		if (save_result):
			with open(sdr_validtn_filename, 'wb') as handle:
				pickle.dump(sdr_result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
			print('Intermediate Test results saved at : ', sdr_result_dict)

		print('sdr results : ', sdr_result_dict)

def test(generator):
	validate_classical(generator, generator.real_QAM_const, generator.imag_QAM_const, True)


def main():
	generator = sample_generator(validtn_batch_size, mod_n, NR)
	test(generator)
	print('******************************** Now Testing **********************************************')

if __name__ == '__main__':
	main()