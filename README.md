# sru

An implementation of SRU in Section 2 of "When Attention Meets Fast Recurrence:
Training Language Models with Reduced Compute", with parameter
initializations from original SRU paper, "Simple Recurrent Units for Highly
Parallelizable Recurrence".

## Experiment

A counting model, consisting of SRU (5 layers, with input size of 3, hidden sizes of 9 per hidden layer), and 1 final fully-connected layer is trained on an input sequence of three numbers to output the subsequent number. Utilizing a batch size of 6, a learning rate of 1e-3 (Adam optimizer), a random train-test split of 80-20 is completed and 10 trials are run, with the following results:
wandb: Test Accuracy 99.994
wandb: Test Loss 3e-05
wandb: Train Loss 0.00075

No extensive hyperparameter search was conducted, as results were already good.
A model without the fully-connected layer was also considered, but it fared worse
due to only having a single hidden neuron in each hidden layer.
