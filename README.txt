# SOC-2021-CNN-Report

Basic Deep Learning

Although not used directly
Initializations 
Zero Initialization
Random Initialization
Xavier & He Initializations

Activators->sigmoid tanh,Relu,Leaky Relu

Problems with sigmoid activation function ->Vanishing Gradient Problem
Exploding gradients can still occur in very deep Multilayer Perceptron networks
Solution is to use Relu or leaky Relu 

Regularization -> L1(Lasso),L2(Ridge),dropout,Early stopping,Data Augmentation

Batch Normalisation - > for faster performance and more stable Neural Networks

Optimization Algorithms - > Gradient Descent,SGD,Mini batch gradient descent,
(Momentum+SGD),Adagrad,Rmsprop,Adam

Cost functions(why not to use MSE) maximum likelihood approach  Cross-Entropy Loss
-> Specifically, neural networks for classification that use a sigmoid or softmax activation function in the output layer learn faster and more robustly using a cross-entropy loss function.

General terms Epoch learning rate ,dataloader

PCA(although a ML term)

Overfitting and Underfitting detection +  funny case when loss and accuracy both increases meaning

How to use gpu in colab
Pytorch some very useful functions
Lr_finder,easy to create cnn model class, easy syntax, various torch.optim, lr_scheduler, 


Main Steps -:
Data loading
Data transform
Data loader
Model class
For loop for final training and calculating validation loss and updating



CNN
Why to use cnn rather than FC layers
Feature mapping 
Primary uses of initial and final layers
Padding,stride,kernels,Pooling what are they and their uses
Uses of bigger kernels and smaller kernels

How to write custom CNN models-:

Imagenet dataset->15million and 22,000 categories

Alexnet-:
ReLU Nonlinearity
Multiple GPUs
Overlapping Pooling
Data Augmentation(which increased the training set by a factor of 2048.)->(hey also performed Principle Component Analysis)
Dropout(50%)
Local Region Normalization


VGG-:(16,19) weight layers 
More depth		
Consistent filter size filter 3*3 	stride = 1 padding is also same(input size ==output size)
Pooling 2*2 stride 2
softmax classifier



Resnet-:Problem it solved-> going deeper model acc should increase but in tern it is getting decreased Solved using Residual blocks
 3*3 filters mostly
Downsampling with stride 2
Even made 1024 layered network


GoogleNet-:Problem before -which filter size should we use -> Solution use them all😂
Now another problem arises -> computational cost solution shrink the input to very small size and then apply all the filters (although we would surely loss some information)
2) Apply softmax to the inner layers as well!!

Transfer Learning
Using learned information from one large dataset to smaller available dataset
