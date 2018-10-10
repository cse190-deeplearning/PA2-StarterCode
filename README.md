# Implementing Backpropagation

In this assignment, we will be implementing a configurable neural network in numpy. The starter code ```neuralnet_starter.py``` contains the abstractions of various componenets of a neural network including layers and activation functions. You will be implementing the the forward and backward passes of these components and combining them to create a complete neural network. You will also be implmenting the training procedure to train the neural network on MNIST classification and evaluating it. Feel free to create multiple copies of this starter code for different experiments. Just ensure that you also complete the file ```neuralnet_starter.py``` which will be autograded. We are providing a ```checker.py``` so that you may check the correctness of the functions you implement. You may run ```python checker.py``` at any point generate an evaluation report about the correctness of your implementation.

## Datasets
We are providing the pickle files for train, val and test split in the ```data/``` directory. The data is in the form of ```n * 785``` numpy array in which the first 784 columns contain the flattend 28 * 28 MNIST image and the last column gives the class of image from 0 to 9. You need to implment the function ```load_data``` to return 2 arrays X, Y given a pickle file. X should be the input features and Y should be the one-hot encoded labels of each input image i.e ```shape(X) = n * 784``` and ```shape(Y) = n * 10```

## Activation Functions
You will be 3 activation functions (sigmoid, ReLU and tanh) and their gradients. Gradient of the output of an activation unit with respect to the input will be multiplied by the upstream gradient during the backward pass to be passed on to the previous layer. 

## Layers

Similar to the activation functions, you will be implementing the linear layers of the neural network. The forrward pass of a layer can be implemented as matrix multiplication of the weights with inputs and addition of biases. In the backward pass, given the gradient of the loss with respect to the output of the layer (delta), we need to compute the gradient of the loss with respect to the inputs of the layer and with respect to the weights and biases. The gradient with respect to the inputs will be passed on to the previous layers during backpropogation.

## Neural Network

Having implemented the lower level abstractions of layers and activation functions, the next step is to implment the forward and backward pass of the neural network by iteratively going through the layers and activations of the neural network. Remember that we will be caching the inputs to each layer and activation function during the forward pass (in self.x) and using it during the backward pass to compute the gradients. 

## Training
We will be writing our training procedure by implementing the trainer function. We will go over ```config['epochs']``` epochs over the dataset using mini-batches of ```size config['batch_size']```. During each iteration of the mini-batch we will be calling the forward and backward pass of the neural network and using the gradients ```layer.dw, layer.db``` to update the weights and biases ```layer.w, layer.b``` of each layer in the neural network according to our update rule. Note that activation layers don't have any associated parameters. 



