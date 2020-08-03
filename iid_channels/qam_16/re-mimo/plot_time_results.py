import pickle
import numpy as np
import matplotlib.pyplot as plt

oampnet_timing_filename_64 = './final_results/oampnet_timing_results_64.pickle'
network_timing_filename_64 = './final_results/network_timing_results_64.pickle'
mod = 'qam_16'

# Load Data
with open(oampnet_timing_filename_64, 'rb') as handle:
	oampnet_time_result_dict_64 = pickle.load(handle)

with open(network_timing_filename_64, 'rb') as handle:
	network_time_result_dict_64 = pickle.load(handle)


NT_list_64 = list(network_time_result_dict_64.keys())
NT_list_64 = np.asarray(NT_list_64)
ratio_list_64 = NT_list_64/64.0


def extract_values(time_result_dict):
	time_result = [time_result_dict.get(key)[20.0] for key in list(time_result_dict.keys())]
	time_result = np.asarray(time_result)
	return time_result

oampnet_time_result_64 = extract_values(oampnet_time_result_dict_64)
network_time_result_64 = extract_values(network_time_result_dict_64)

time_ratio_result_64 = oampnet_time_result_64/network_time_result_64

mct_unit = r'($ms$)'
oampnet_tau = r'$\tau_{ON}$' + ' : OAMPNet'
remimo_tau = r'$\tau_{RM}$' + ' : RE-MIMO'

plt.figure()
xlabel = r'$\left(\frac{N_{tr}}{N_{r}}\right)$'
title = r'$\frac{\tau_{ON}}{\tau_{RM}}$'
plt.plot(ratio_list_64, time_ratio_result_64, label=title)
plt.xlabel('system size ratio ' + xlabel, fontsize=15)
plt.ylabel('mean computation time ratio', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='upper right', fontsize=20)
plt.savefig('./final_results/' + mod + '/' + 'mct_ratio_'+ 'NR64.pdf', bbox_inches='tight')

plt.figure()
plt.plot(ratio_list_64, oampnet_time_result_64*10.0, label=oampnet_tau)
plt.plot(ratio_list_64, network_time_result_64*10.0, label=remimo_tau)
plt.xlabel('system size ratio ' + xlabel, fontsize=15)
plt.ylabel('mean computation time ' + mct_unit, fontsize=15)
plt.legend(loc='lower right', fontsize=14)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('./final_results/' + mod + '/' + 'mct_'+ 'NR64.pdf', bbox_inches='tight')
plt.show()
