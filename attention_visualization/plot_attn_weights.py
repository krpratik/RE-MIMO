import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

NT = 12
data_point = 2
snrdb_classical_list = {16:[11.0], 32:[13.0], 12:[15.0]}
num_layers = 8
batch_size = 10
data = np.empty((num_layers, batch_size, NT, NT))

channel_filename = './validtn_results/channel_matrix.pickle'

# Load Data
for layer in range(num_layers):
	with open('./validtn_results/attn_weights/' + str(NT)+ '_' + str(layer) +'_attn_weight.pickle', 'rb') as handle:
		data[layer] = pickle.load(handle)

data = data.swapaxes(2,3)

# Load channel Matrix
with open(channel_filename, 'rb') as handle:
	channel_dict = pickle.load(handle)
	H = channel_dict[NT][snrdb_classical_list[NT][0]][0]

# process channel matrix
HT = H.permute(0,2,1)
HTH = torch.einsum(('bij,bjk->bik'),(HT,H))
HTH = torch.chunk(HTH, 2, dim=-1)[0]
HTH_real, HTH_imag = torch.chunk(HTH, 2, dim=1)
corr_mat = HTH_real + HTH_imag
corr_mat = torch.sqrt(torch.pow(HTH_real, 2) + torch.pow(HTH_imag, 2))

NR = 32


for layer in range(num_layers):
	if (True):
		# layer == 1
		plt.figure()
		
		plt.subplot(1,2,1)
		plt.imshow(data[layer][data_point], cmap='plasma', interpolation='nearest')
		plt.xticks(np.arange(1,13,2)-0.95, np.arange(1,13,2))
		plt.xlabel('transmitter index', fontsize=9)
		plt.ylabel('co-transmitter index', fontsize=9)
		plt.yticks(np.arange(0,12,2)+0.05, np.arange(1,13,2))
		plt.colorbar(fraction=0.046, pad=0.04)
		plt.title('attention weight heatmap', fontsize=9)
		plt.tight_layout()
		

		plt.subplot(1,2,2)
		plt.imshow(corr_mat[data_point].abs(), cmap='plasma', interpolation='nearest')
		plt.xticks(np.arange(1,13,2)-0.95, np.arange(1,13,2))
		plt.xlabel('channel index', fontsize=9)
		plt.yticks(np.arange(0,12,2)+0.05, np.arange(1,13,2))
		plt.colorbar(fraction=0.046, pad=0.04)
		plt.title('channel correlation matrix', fontsize=9)
		plt.tight_layout()

		plt.savefig('./final_results/layer_wise' + '/' + 'attention_heatmap_layer_' + str(layer+1) + '.pdf',  bbox_inches='tight')
plt.show()