import pickle
import numpy as np
import matplotlib.pyplot as plt

mod = 'qam_16'
caption = 'QAM-16, '
NT_label =  r'$N_{tr}$' + '='
NR_label = r'$N_{r}$' + '=64'

# Load Data
with open('./final_results/'+ mod +'/mmse_validtn_results.pickle', 'rb') as handle:
	mmse_validtn_results_dict = pickle.load(handle)

with open('./final_results/'+ mod +'/blast_validtn_results.pickle', 'rb') as handle:
	blast_validtn_results_dict = pickle.load(handle)

with open('./final_results/'+ mod +'/amp_validtn_results.pickle', 'rb') as handle:
	amp_validtn_results_dict = pickle.load(handle)

with open('./final_results/'+ mod +'/network_validtn_results.pickle', 'rb') as handle:
	network_validtn_results_dict = pickle.load(handle)

if(mod == 'qam_16'):
	with open('./final_results/'+ mod +'/sdr_validtn_results.pickle', 'rb') as handle:
		sdr_validtn_results_dict = pickle.load(handle)

with open('./final_results/'+ mod +'/detnet_16_validtn_results.pickle', 'rb') as handle:
	detnet_16_validtn_results_dict = pickle.load(handle)

with open('./final_results/'+ mod +'/detnet_32_validtn_results.pickle', 'rb') as handle:
	detnet_32_validtn_results_dict = pickle.load(handle)

with open('./final_results/'+ mod +'/oampnet_16_validtn_results.pickle', 'rb') as handle:
	oampnet_16_validtn_results_dict = pickle.load(handle)

with open('./final_results/'+ mod +'/oampnet_32_validtn_results.pickle', 'rb') as handle:
	oampnet_32_validtn_results_dict = pickle.load(handle)

detnet_validtn_results_dict = {**detnet_16_validtn_results_dict, **detnet_32_validtn_results_dict}
oampnet_validtn_results_dict = {**oampnet_16_validtn_results_dict, **oampnet_32_validtn_results_dict}

def process_data(validtn_result):
	validtn_result_keys = np.asarray(list(validtn_result.keys()))
	validtn_result_values = np.asarray(list(validtn_result.values()))
	valid_indices = [i for i in range(len(validtn_result_values)) if validtn_result_values[i]>=0.96]
	min_valid_index = min(valid_indices)
	valid_indices = np.arange(min_valid_index, min_valid_index+6)
	print(valid_indices)
	result_keys = np.asarray(validtn_result_keys[valid_indices])
	result_values = 1.0 - np.asarray(validtn_result_values[valid_indices])
	return result_keys, result_values

NR = 64
NT_list = list(mmse_validtn_results_dict.keys())

for NT in NT_list:

	plt.figure()

	mmse_validtn_result = mmse_validtn_results_dict[NT]
	mmse_result_keys = list(mmse_validtn_result.keys()) 
	mmse_result_values = 1.0 - np.asarray(list(mmse_validtn_result.values()))
	plt.plot(mmse_result_keys, mmse_result_values, label='MMSE', marker="^")

	amp_validtn_result = amp_validtn_results_dict[NT]
	amp_result_keys = list(amp_validtn_result.keys())
	amp_result_values = 1.0 - np.asarray(list(amp_validtn_result.values()))
	plt.plot(amp_result_keys, amp_result_values, label='AMP', marker="o")

	blast_validtn_result = blast_validtn_results_dict[NT]
	blast_result_keys = list(blast_validtn_result.keys())
	blast_result_values = 1.0 - np.asarray(list(blast_validtn_result.values()))
	plt.plot(blast_result_keys, blast_result_values, label='V-BLAST', marker="s")

	network_validtn_result = network_validtn_results_dict[NT]
	network_result_keys = mmse_result_keys 
	network_result_values = 1.0 - np.asarray([network_validtn_result.get(key) for key in network_result_keys])
	plt.plot(network_result_keys, network_result_values, label='RE-MIMO', marker="*")

	detnet_validtn_result = detnet_validtn_results_dict[NT]
	detnet_result_keys = list(detnet_validtn_result.keys())
	detnet_result_values = 1.0 - np.asarray(list(detnet_validtn_result.values()))
	plt.plot(detnet_result_keys, detnet_result_values, label='DetNet', marker="x")

	oampnet_validtn_result = oampnet_validtn_results_dict[NT]
	oampnet_result_keys = list(oampnet_validtn_result.keys())
	oampnet_result_values = 1.0 - np.asarray(list(oampnet_validtn_result.values()))
	plt.plot(oampnet_result_keys, oampnet_result_values, label='OAMP-NET', marker="D")

	if (mod == 'qam_16'):
		sdr_validtn_result = sdr_validtn_results_dict[NT]
		if (sdr_validtn_result):
			sdr_result_keys = list(sdr_validtn_result.keys())
			sdr_result_values = 1.0 - np.asarray([sdr_validtn_result.get(key) for key in sdr_result_keys])
			plt.plot(sdr_result_keys, sdr_result_values, label='SDR', marker="p")


	plt.yscale('log')
	plt.grid(True, which='both')
	plt.legend(loc='lower left', fontsize=11)
	plt.xlabel('SNR(dB)', fontsize=13)
	plt.ylabel('SER', fontsize=13)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.title(caption+ NT_label+ str(NT) + ', ' + NR_label, fontsize=13)
	plt.tight_layout()
	plt.savefig('./final_results/' + mod + '/' + mod + '_NT' + str(NT) + '_NR64.pdf')
plt.show()