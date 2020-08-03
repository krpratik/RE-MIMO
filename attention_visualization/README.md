# RE-MIMO

To replicate the attention visualization experiment follow the mentioned steps below:

1. Start trainig the model by the command, Python train_classifier.py
2. Wait till the training is done. The trained model and intermediate validation results are stored in the validtn_results folder
3. Now, to visualize the attention weight, run the command : Python visualize_network.py
4. The attention weights and the channel matrices used for attnetion visualization are stored in validtn_results folder
5. Now to plot the attention heatmap, use the command Python plot_attn_weights.py
6. The attention heatmap of each layer is stored in the final_results directory