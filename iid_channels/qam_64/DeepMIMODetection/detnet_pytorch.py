#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time as tm
import math
import sys
import pickle as pkl
from detnet import DetNet
from sample_generator import sample_generator

#parameters
NT = 16
NR = 64

snrdb_classical_list = {16:np.arange(15.0, 23.0), 32:np.arange(17.0, 25.0)}

L=3*NT
v_size = 2*2*NT
hl_size = 8*2*NT

startingLearningRate = 0.0001
decay_factor = 0.97
decay_step_size = 1000

train_iter = 50000
train_batch_size = 5000

res_alpha=0.9
num_snr = 6


mod_n = 64

model_filename = './validtn_results/model_q' + str(mod_n) + '_' + str(NT) + '_' + str(NR) + '.pth'


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


def loss_fn(batch_X, batch_HY, batch_HH, list_batch_x_predicted, j_indices, real_QAM_const, imag_QAM_const, ber_only=False, last_only=False):
	if (ber_only):
		BER_final = sym_detection(list_batch_x_predicted[-1], j_indices, real_QAM_const, imag_QAM_const)
		return BER_final
	else:
		HtHinv = torch.inverse(batch_HH)
		X_LS = torch.einsum(('ijk,ik->ij'), (HtHinv, batch_HY))

		LSE_error = torch.mean(torch.pow((batch_X - X_LS), 2), dim=1)
		loss_list = []
		BER_final = []
		for index, batch_x_predicted in enumerate(list_batch_x_predicted):
			loss_index = math.log(index+1)*torch.mean(torch.mean(torch.pow((batch_X - batch_x_predicted),2), dim=1)/LSE_error)
			loss_list.append(loss_index)
		BER_final = sym_detection(list_batch_x_predicted[-1], j_indices, real_QAM_const, imag_QAM_const)
		if (last_only):
			return loss_list[-1], BER_final, LSE_error, X_LS
		else:
			return sum(loss_list), BER_final

def loss_ber_ls(X_LS, j_indices, real_QAM_const, imag_QAM_const):
	ber_LS = sym_detection(X_LS, j_indices, real_QAM_const, imag_QAM_const)
	return ber_LS

def pre_process_data(H, y):
	H_t = H.permute(0,2,1)
	HTY = torch.einsum(('ijk,ik->ij'), (H_t, y))
	HTH = torch.matmul(H_t, H)
	return (HTY, HTH)


def train(model, optimizer, lr_scheduler, generator, device='cpu'):
	real_QAM_const = generator.real_QAM_const.to(device=device)
	imag_QAM_const = generator.imag_QAM_const.to(device=device)
	for i in range(train_iter):
		H, y, x, j_indices, _ = generator.give_batch_data(NT, snr_db_min=snrdb_classical_list[NT][0], snr_db_max=snrdb_classical_list[NT][-1], batch_size=train_batch_size)
		HTY, HTH = pre_process_data(H, y)
		HTY = HTY.to(device=device)
		HTH = HTH.to(device=device)
		x = x.to(device=device)
		j_indices = j_indices.to(device=device)
		list_batch_x_predicted = model.forward(train_batch_size, HTY, HTH)
		loss, BER_final = loss_fn(x, HTY, HTH, list_batch_x_predicted, j_indices, real_QAM_const, imag_QAM_const)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		lr_scheduler.step()
		if (i%1000==0):
			H, y, x, j_indices, _ = generator.give_batch_data(NT, snr_db_min=snrdb_classical_list[NT][0], snr_db_max=snrdb_classical_list[NT][-1], batch_size=train_batch_size)
			HTY, HTH = pre_process_data(H, y)
			HTY = HTY.to(device=device)
			HTH = HTH.to(device=device)
			x = x.to(device=device)
			j_indices = j_indices.to(device=device)
			with torch.no_grad():
				list_batch_x_predicted = model.forward(train_batch_size, HTY, HTH)
				loss_last, BER_final, loss_LS, X_LS = loss_fn(x, HTY, HTH, list_batch_x_predicted, j_indices, real_QAM_const, imag_QAM_const, last_only=True)
				ber_LS = loss_ber_ls(X_LS, j_indices, real_QAM_const, imag_QAM_const)
				results = [loss_LS.mean().detach().item(), loss_last.detach().item(),ber_LS, BER_final]
				print_string = [i]+results
				print(' '.join('%s' % np.round(x,6) for x in print_string))
	torch.save(model.state_dict(), model_filename)
	print('************Model Saved************ at directory : ', model_filename)

def main():
	device = 'cuda'
	model = DetNet(L, NT, v_size, hl_size, device=device)
	generator = sample_generator(train_batch_size, mod_n, NR)
	model = model.to(device=device)
	optimizer = torch.optim.Adam(model.parameters(), lr=startingLearningRate)
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, decay_step_size, decay_factor)
	train(model, optimizer, lr_scheduler, generator, device)
	print('******************************** Now Testing **********************************************')

if __name__ == '__main__':
	main()
