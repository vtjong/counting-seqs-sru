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

- **Batch Size**: 6
- **Learning Rate**: 1e-3 (Using the Adam optimizer)
- **Dataset Split**: Random train-test split with 80% training data and 20% testing data.
- **Loss**: MSE Loss (first model), CE Loss (second model))

### Trials

Model 1:

```
input tensor([[81., 82., 83.]])
target tensor([[82., 83., 84.]])
output tensor([[82, 83, 84]])
loss 2.281102388224099e-05
input tensor([[7., 8., 9.]])
target tensor([[ 8.,  9., 10.]])
output tensor([[ 8,  9, 10]])
loss 2.49348086072132e-05
input tensor([[31., 32., 33.]])
target tensor([[32., 33., 34.]])
output tensor([[32, 33, 34]])
loss 2.6588484615786e-05
input tensor([[87., 88., 89.]])
target tensor([[88., 89., 90.]])
output tensor([[88, 89, 90]])
loss 4.603876732289791e-05
input tensor([[96., 97., 98.]])
target tensor([[97., 98., 99.]])
output tensor([[97, 98, 63]])
loss 0.01914927549660206
input tensor([[3., 4., 5.]])
target tensor([[4., 5., 6.]])
output tensor([[4, 5, 6]])
loss 1.5907555280136876e-05
input tensor([[34., 35., 36.]])
target tensor([[35., 36., 37.]])
output tensor([[35, 36, 37]])
loss 2.6028270440292545e-05
input tensor([[73., 74., 75.]])
target tensor([[74., 75., 76.]])
output tensor([[74, 75, 76]])
loss 3.282587567809969e-05
input tensor([[28., 29., 30.]])
target tensor([[29., 30., 31.]])
output tensor([[29, 30, 31]])
loss 2.0435010810615495e-05
input tensor([[52., 53., 54.]])
target tensor([[53., 54., 55.]])
output tensor([[53, 54, 55]])
loss 3.404177914489992e-05
input tensor([[68., 69., 70.]])
target tensor([[69., 70., 71.]])
output tensor([[69, 70, 71]])
loss 2.4823764761094935e-05
input tensor([[75., 76., 77.]])
target tensor([[76., 77., 78.]])
output tensor([[76, 77, 78]])
loss 4.977876233169809e-05
input tensor([[65., 66., 67.]])
target tensor([[66., 67., 68.]])
output tensor([[66, 67, 68]])
loss 1.2659546882787254e-05
input tensor([[55., 56., 57.]])
target tensor([[56., 57., 58.]])
output tensor([[56, 57, 58]])
loss 2.1304467736626975e-05
input tensor([[71., 72., 73.]])
target tensor([[72., 73., 74.]])
output tensor([[72, 73, 74]])
loss 3.13459531753324e-05
input tensor([[17., 18., 19.]])
target tensor([[18., 19., 20.]])
output tensor([[18, 19, 20]])
loss 1.9536642867024057e-05
input tensor([[48., 49., 50.]])
target tensor([[49., 50., 51.]])
output tensor([[49, 50, 51]])
loss 1.5370242181234062e-05
input tensor([[0., 1., 2.]])
target tensor([[1., 2., 3.]])
output tensor([[75,  2,  3]])
loss 0.009817713871598244
input tensor([[60., 61., 62.]])
target tensor([[61., 62., 63.]])
output tensor([[61, 62, 63]])
loss 1.8827344320015982e-05
input tensor([[84., 85., 86.]])
target tensor([[85., 86., 87.]])
output tensor([[85, 86, 87]])
loss 1.8344977434026077e-05
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
- The first model uses SRU as a language model (sequence-to-sequence) and trains on the one-hot-encoding of the 3-number sequence. The second uses SRU + a fully-connected layer and trains on the raw number sequence. The former model, without the fully-connected layer, makes more errors than the one with. Fine-tuning of the model would certainly result in higher accuracy values. However, considering number of mistakes is small across runs, this demonstrates the SRU's ability to "learn how to count."
