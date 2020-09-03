# RE-MIMO: Recurrent and Permutation Equivariant Neural MIMO Detection

This repository contains the official implementation of the "RE-MIMO: Recurrent and Permutation Equivariant Neural MIMO Detection" paper (arXiv:2007.00140).


To make it easy to easy to replicate the results, the repository contains separate directories for each of the experiments. For running each of these experiments, please read the separate README.md present inside each of the experiment directory.

The experiments included in the repository are:
1. i.i.d. Gaussian Channel experiment on both QAM-16 and QAM-64
2. Fully correlated (correlation on both the transmitter and receiver side) channels experiments on QAM-16 modulation scheme
3. Partially correlated (correlation only on the receiver side) channels experiment on QAM-16 modualtion scheme
4. Attention Visualization experiment as described in the RE-MIMO paper

Below are the variable naming conventions used throughout the code. Readers are expected to thoroughly go through the RE-MIMO paper before exploring the implementation codes. The variable names and what it coresponds to are as follows:

1. NR : Number of receivers
2. NT_list : Numpy array consisting of array of number of users/trasmitters the network is to be trained on
3. NT : Number of transmitters
4. NT_prob : The probability mass function determining the probability with which a particulat value of NT is chosen for a minibatch during training of RE-MIMO. In our case it is a traingular probability function over the range of NT values as described in the paper.
5. mod_n : QAM modulation scheme, 16 for QAM-16, 64 for QAM-64
6. d_transmitter_encoding : Dimension of the TE (Transmitters Encoding) vector
7. d_model : Dimension of the hidden state vector s
8. n_head : Number of attention heads in the self-attention module
9. nhid : Dimension of the hidden layer of the feed-forward submodule of the encoder module
10. nlayers : Number of EP blocks in the network
11. epoch_size : Number of iterations per epoch
12. train_iter : Number of epochs to train
13. train_batch_size : Batch size used for training
14. mini_validtn_batch_size : Batch size used for intermediate validation (validation in-between training to track training)
15. validtn_NT_list : List of values of number of transmitters to be used during intermediate validation
16. snrdb_list : Dictionary mentioning the SNR values to be used for each value of number of transmitters during intermediate validation
17. model_filename : location to store trained models, model is stored after each training epoch
18. curr_accr : Intermediate validation results are stored in the file at the location mentioned by curr_accr

Some general tips/guidelines that are valid across all the experimental directories:
1. During training, the intermediate validation results are printed on the std-output screen and also stored in the file located by the curr_accr.txt or curr_accr_NT.txt in the validtn_results directory of the concerned directory.
2. The file curr_accr.txt or curr_accr_NT.txt just stores the results for the most current validation as it overwrites the last validation results stored. To keep track of the validation results at all the steps, readers are expected to store/log the std-output (Print output) to a desired file.
3. The Intermediate validation is done with an intnent to get an overview of the training progress and the validation results should never considered as proxy for final test results because intermediate validation is performed on a very small dataset which is not intensive enough to be considered as final result
4. The final test result is obtained only after extensive testing of network, as done in the test_network.py file
5. During training of learning based detection schemes, intermediate trained models are stored after each epoch in the directory validatn_results. After each epoch the last saved model is overwritten by the current model, by the name model.pth
6. It is highly advised to train the learning based schemes on GPU, but if someone wants to train the models on CPU, the device variable inside the training file should be put to 'CPU' instead of 'CUDA'
7. For testing the classical scheme of SDR, the reader must install Gurobi optimization software with Python support. Students can get free student license from the official Gurobi Website

General Requirements:
1. Python3.5 or above version
2. Pytorch CUDA
3. Numpy
4. Matplotlib
5. Gurobi optimizer with Python engine

You may consider citing this project:
@article{pratik2020re,
  title={RE-MIMO: Recurrent and Permutation Equivariant Neural MIMO Detection},
  author={Pratik, Kumar and Rao, Bhaskar D and Welling, Max},
  journal={arXiv preprint arXiv:2007.00140},
  year={2020}
}

In case of any queries, please contact me at: kumar.pratik73@yahoo.com
