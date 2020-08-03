# Classical Schemes

To replicate the experiment of classical detection schemes on fully correlated channels for QAM-64 modulation, follow the mentioned steps.

1. The classical schemes does not require training of parameters. Hence, they can be tested directly without training
2. Run the command Python test_amp.py to get the results for AMP solver
3. Run the command Python test_blast.py to get the results for V-BLAST solver
4. Run the command Python test_mmse.py to get the results for MMSE solver
5. Run the command Python test_sdr.py to get the results for SDR solver
6. The results of all the above mentioned testing will be stored in the final_results directory
7. Copy those results to the final_results directory of the adjoining re-mimo directory. Because we will plot the results of DetNet, RE-MIMO and classical detection techniques together in the same plot.
