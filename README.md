# Counting Sequences with SRUs

This repository contains an implementation of the Simple Recurrent Unit (SRU) as described in "When Attention Meets Fast Recurrence: Training Language Models with Reduced Compute". The implementation follows the parameter initializations from the original SRU paper, "Simple Recurrent Units for Highly Parallelizable Recurrence".

## Experiment Overview

The experiment involves training a counting model using SRU. The model architecture includes:

- **5 layers of SRU**: Input size of 3 for first layer and hidden size of 9.
- **A final fully-connected layer**: To predict the output based on the SRU layers' representations.

The model is trained to predict the subsequent number in a sequence given three input numbers.
To run, navigate to `main.py`. We forego a word embedding layer as we're operating on input sequences and are not using encoding (see Discussion section for reasoning.)

### Training Details

- **Batch Size**: 6
- **Learning Rate**: 1e-3 (Using the Adam optimizer)
- **Dataset Split**: Random train-test split with 80% training data and 20% testing data.
- **Loss**: MSE Loss

### Trials

A total of 10 trials were conducted, yielding the following results:

```
Test Accuracy: 99.92\%
Test Loss: 3e-05
Train Loss: 0.00075

```

One trial's test results:

```
input tensor([[[79., 80., 81.]]])
target 82.0
output 81.97855377197266
input tensor([[[12., 13., 14.]]])
target 15.0
output 15.014595985412598
input tensor([[[80., 81., 82.]]])
target 83.0
output 82.98188781738281
input tensor([[[86., 87., 88.]]])
target 89.0
output 89.0189437866211
input tensor([[[60., 61., 62.]]])
target 63.0
output 63.055606842041016
input tensor([[[57., 58., 59.]]])
target 60.0
output 60.07681655883789
input tensor([[[20., 21., 22.]]])
target 23.0
output 23.014514923095703
input tensor([[[47., 48., 49.]]])
target 50.0
output 50.0598030090332
input tensor([[[11., 12., 13.]]])
target 14.0
output 14.022920608520508
input tensor([[[23., 24., 25.]]])
target 26.0
output 26.048019409179688
input tensor([[[71., 72., 73.]]])
target 74.0
output 73.98202514648438
input tensor([[[27., 28., 29.]]])
target 30.0
output 30.077543258666992
input tensor([[[90., 91., 92.]]])
target 93.0
output 93.0582046508789
input tensor([[[94., 95., 96.]]])
target 97.0
output 97.10755157470703
input tensor([[[13., 14., 15.]]])
target 16.0
output 16.005922317504883
input tensor([[[43., 44., 45.]]])
target 46.0
output 45.9821662902832
input tensor([[[19., 20., 21.]]])
target 22.0
output 22.00603485107422
input tensor([[[67., 68., 69.]]])
target 70.0
output 70.00263214111328
input tensor([[[76., 77., 78.]]])
target 79.0
output 78.9735107421875
input tensor([[[2., 3., 4.]]])
target 5.0
output 5.037868499755859
```

### Discussion

- The model demonstrates excellent performance without the need for an extensive hyperparameter search.
- An alternative model architecture without the fully-connected layer was evaluated but performed worse. This performance drop is attributed to the limited capacity with only a single hidden neuron in each hidden layer. With 16 hidden layers, it performs comparatively.
- Another possibility is to use the SRU as a language model and use one-hot-encoding of the sequence to train. We originally implemented it like this, but decided to forego this method, as this method makes more sense for training on text corpuses, in which there is a possibility that the subsequent word takes on more than one value. Here, we have a fixed outcome in counting sequences, so the training set and the test set would contain overlaps (i.e. we would have [1, 2, 3] -> [2, 3, 4] and take the output as 4. But there is no variation in this since the SRU has memory, so it is 100% accurate. In a text, we could have "I have a dog" or "I ate a sandwich," so predicting the next word has some non-unitary probability associated with the task). If we were to use this one-hot-encoding method of training, we'd just change the training and testing dataset construction, so the test dataset is a subset of the training dataset. Then, use 'CountingDatasetEmbeddings' and 'SRU' as the model, as well as CE Loss (Cross Entropy Loss) rather than MSE Loss.
