# DetNet

To replicate the experiment of DetNet on fully correlated channels for QAM-64 modulation, follow the mentioned steps.
1. We need two separate DetNet networks, one for NT=16, and the second for NT=32.
2. Before starting the training procedure, go inside the detnet_pytorch.py and put the value of NT to 16
3. Start the training for NT=16 by running the command, Python detnet_pytorch.py
4. Now to start the training for NT=32, go inside the detnet_pytorch.py and put the value of NT to 32
5. Start the training for NT=32 by running the command, Python detnet_pytorch.py
4. Track the intermediate validation results, by running the command cat validtn_results/curr_accr_16.txt and cat validtn_results/curr_accr_32.txt 
5. After the DetNet for NT=16 is trained, to test the trained DetNet, trained for NT=16, go inside the test_detnet.py and put NT=16, and run the command Python test_detnet.py
6. After the DetNet for NT=32 is trained, to test the trained DetNet, trained for NT=32, go inside the test_detnet.py and put NT=32, and run the command Python test_detnet.py
7. The test results are stored in the final_results directory. Copy those results to the final_results directory of the adjoining re-mimo directory. Because we will plot the results of DetNet, RE-MIMO and other detection techniques together in the same plot.
