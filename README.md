# Counting Sequences with SRUs

This repository contains an implementation of the Simple Recurrent Unit (SRU) as described in "When Attention Meets Fast Recurrence: Training Language Models with Reduced Compute". The implementation follows the parameter initializations from the original SRU paper, "Simple Recurrent Units for Highly Parallelizable Recurrence".

## Experiment Overview

The experiment involves training a counting model using SRU. Two model architectures were considered.
The first uses a sequence-to-sequence RNN as language model approach:

- **5 layers of SRU**: Input and output sizes equal to the length of the vocab size (one hot encoding); sequence length = 3.

The second model architecture uses the RNN in conjunction with a fully-connected layer:

- **5 layers of SRU**: Input size of 3 for first layer and hidden size of 9 (treat the entire 3-number sequence as 3 features of a single time-step sequence, which does not truly make use of main benefits of RNN, but works very well, likely due to the fc layer at the end).
- **A final fully-connected layer**: To predict the output based on the SRU layers' representations.

Both models are trained to predict the subsequent number in a sequence given three input numbers. To run, navigate to `main.py`.

### Training Details

- **Number of Epochs**: 100 (second), 200 (first)
- **Batch Size**: 6
- **Learning Rate**: 1e-3 (second), 1e-4 (first) (Using the Adam optimizer)
- **Dataset Split**: Random train-test split with 80% training data and 20% testing data.
- **Loss**: MSE Loss (first model), CE Loss (second model))

### Trials

Model 1:

```
input tensor([[33., 34., 35.]])
target tensor([[34., 35., 36.]])
output tensor([[34, 35, 36]])
loss 0.00016859466268215328
input tensor([[76., 77., 78.]])
target tensor([[77., 78., 79.]])
output tensor([[77, 78, 79]])
loss 0.00042785381083376706
input tensor([[31., 32., 33.]])
target tensor([[32., 33., 34.]])
output tensor([[32, 33, 34]])
loss 0.00010597872460493818
input tensor([[40., 41., 42.]])
target tensor([[41., 42., 43.]])
output tensor([[41, 42, 43]])
loss 0.00012463287566788495
input tensor([[1., 2., 3.]])
target tensor([[2., 3., 4.]])
output tensor([[2, 3, 4]])
loss 0.00016336393309757113
input tensor([[2., 3., 4.]])
target tensor([[3., 4., 5.]])
output tensor([[ 3,  4, 49]])
loss 0.0009431123617105186
input tensor([[79., 80., 81.]])
target tensor([[80., 81., 82.]])
output tensor([[80, 81, 82]])
loss 7.131805614335462e-05
input tensor([[42., 43., 44.]])
target tensor([[43., 44., 45.]])
output tensor([[43, 44, 45]])
loss 0.00011074219219153747
input tensor([[45., 46., 47.]])
target tensor([[46., 47., 48.]])
output tensor([[46, 47, 48]])
loss 2.612705429783091e-05
input tensor([[58., 59., 60.]])
target tensor([[59., 60., 61.]])
output tensor([[59, 60, 61]])
loss 0.00010467324318597093
input tensor([[95., 96., 97.]])
target tensor([[96., 97., 98.]])
output tensor([[96, 97, 16]])
loss 0.01509164460003376
input tensor([[51., 52., 53.]])
target tensor([[52., 53., 54.]])
output tensor([[52, 53, 54]])
loss 0.00012654939200729132
input tensor([[7., 8., 9.]])
target tensor([[ 8.,  9., 10.]])
output tensor([[ 8,  9, 10]])
loss 0.00010569680307526141
input tensor([[96., 97., 98.]])
target tensor([[97., 98., 99.]])
output tensor([[97, 16, 86]])
loss 0.030178433284163475
input tensor([[66., 67., 68.]])
target tensor([[67., 68., 69.]])
output tensor([[67, 68, 69]])
loss 6.203942029969767e-05
input tensor([[61., 62., 63.]])
target tensor([[62., 63., 64.]])
output tensor([[62, 63, 64]])
loss 0.0002770077553577721
input tensor([[20., 21., 22.]])
target tensor([[21., 22., 23.]])
output tensor([[21, 22, 23]])
loss 3.838343036477454e-05
input tensor([[12., 13., 14.]])
target tensor([[13., 14., 15.]])
output tensor([[13, 14, 15]])
loss 5.434556442196481e-05
input tensor([[91., 92., 93.]])
target tensor([[92., 93., 94.]])
output tensor([[92, 93, 94]])
loss 7.240963168442249e-05
input tensor([[53., 54., 55.]])
target tensor([[54., 55., 56.]])
output tensor([[54, 55, 56]])
```

Model 2:

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

- The models demonstrate decent performance pre-hyperparameter search.
- The first model uses SRU as a language model (sequence-to-sequence) and trains on the one-hot-encoding of the 3-number sequence. The second uses SRU + a fully-connected layer and trains on the raw number sequence. The former model, without the fully-connected layer, makes more errors than the one with if trained with 100 epochs and a learning rate of 1e-3. Fine-tuning of the model (200 epochs, 1e-4 learning rates) results in 100% accuracy values. This demonstrates SRU's ability to "learn how to count."
