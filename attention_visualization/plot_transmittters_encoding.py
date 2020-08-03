import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math

d_transmitter_encoding = 64
d_model = 512
max_transmitter = 64
ntr = r'($N_{tr}$)'

def plot_transmitter_encoding():
	NT = torch.zeros(int(max_transmitter), d_transmitter_encoding)
	num_transmitters = torch.arange(1.0, max_transmitter+1.0, dtype=torch.float).unsqueeze(1)
	div_term = torch.exp(torch.arange(0.0, d_transmitter_encoding, 2).float() * (-math.log(np.float(2*max_transmitter)) / d_transmitter_encoding))
	NT[:, 0::2] = torch.sin(num_transmitters * div_term)
	NT[:, 1::2] = torch.cos(num_transmitters * div_term)
	NT = NT/math.sqrt(d_model)
	return NT

if __name__ == '__main__':
	NT = plot_transmitter_encoding()
	plt.imshow(NT, cmap='plasma', interpolation='nearest')
	plt.colorbar(fraction=0.046, pad=0.04)
	plt.xticks(np.arange(1, d_transmitter_encoding+1, 7))
	plt.xlabel('TE dimensions', fontsize=9)
	plt.ylabel('Number of transmitters '+ ntr, fontsize=9)
	plt.yticks(np.arange(1, max_transmitter+1, 7))
	plt.title('TE Heatmap', fontsize=9)
	plt.savefig('./final_results/' + 'te_heatmap' + '.pdf',  bbox_inches='tight')
	plt.show()