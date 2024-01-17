# Counting Sequences with SRUs

This repository contains an implementation of the Simple Recurrent Unit (SRU) as described in "When Attention Meets Fast Recurrence: Training Language Models with Reduced Compute". The implementation follows the parameter initializations from the original SRU paper, "Simple Recurrent Units for Highly Parallelizable Recurrence".

## Experiment Overview

The experiment involves training a counting model using SRU. The model architecture includes:

- **5 layers of SRU**: Each with an input size of 3 and 9 hidden neurons per hidden layer.
- **A final fully-connected layer**: To predict the output based on the SRU layers' representations.

The model is trained to predict the subsequent number in a sequence given three input numbers.

### Training Details

- **Batch Size**: 6
- **Learning Rate**: 1e-3 (Using the Adam optimizer)
- **Dataset Split**: Random train-test split with 80% training data and 20% testing data.

### Trials

A total of 10 trials were conducted, yielding the following results:

```
Test Accuracy: 99.994\%
Test Loss: 3e-05
Train Loss: 0.00075

```

### Observations

- The model demonstrates excellent performance without the need for an extensive hyperparameter search.
- An alternative model architecture without the fully-connected layer was evaluated but performed poorly. This performance drop is attributed to the limited capacity with only a single hidden neuron in each hidden layer.
