# Implementing Backpropagation
In this assignment, we will be implementing a configurable, Multi-Layer Perceptron neural network using NumPy. The starter code ```neuralnet_starter.py``` contains the abstractions of various componenets of a neural network, including layers and activation functions. You will be implementing the forward and backward propagation passes and combining the components to create a complete neural network. You will also be implementing the training and evaluation procedures to classify the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). Complete the code in the  ```neuralnet_starter.py```, which we will be autograding. Feel free to create multiple copies of this starter code to run different experiments, however. We are providing a ```checker.py``` so that you may check the correctness of the functions you implement, <i>though we strongly encourage that you also write your own test cases</i>. You may run ```python checker.py``` at any point to generate an evaluation report about the correctness of your implementation.


## Dataset
Unzip ```data.zip``` to get the pickle files for train, validation and test splits of MNIST dataset. The data is in the form of ```n * 785``` NumPy array (in which the first 784 columns contain the flattend 28 * 28 MNIST image and the last column gives the class of image from 0 to 9. All of the splits have been shuffled so you may skip the shuffling step. You need to implment the function ```load_data``` to return 2 arrays X, Y given a pickle file. X should be the input features and Y should be the one-hot encoded labels of each input image i.e ```shape(X) = n,784``` and ```shape(Y) = n,10```


## Activation Functions
There are 3 activation functions (sigmoid, ReLU and tanh) and their gradients which you will implement and experiment with. The gradient of the output of an activation unit with respect to the input will be multiplied by the upstream gradient during the backward pass to be passed on to the previous layer. 


## Layers
Similar to the activation functions, you will be implementing the linear layers of the neural network. The forward pass of a layer can be implemented as matrix multiplication of the weights with inputs and addition of biases. In the backward pass, given the gradient of the loss with respect to the output of the layer (delta), we need to compute the gradient of the loss with respect to the inputs of the layer and with respect to the weights and biases. The gradient with respect to the inputs will be passed on to the previous layers during backpropagation.


## Neural Network
Having implemented the lower level abstractions of layers and activation functions, the next step is to implment the forward and backward pass of the neural network by iteratively going through the layers and activations of the network. Remember that we will be caching the inputs to each layer and activation function during the forward pass (in self.x) and using it during the backward pass to compute the gradients. 

## Training
You will implement the training procedure in the trainer function. The network will be trained for ```config['epochs']``` epochs over the dataset in mini-batches of ```size config['batch_size']```. During each iteration (e.g. each mini-batch) you will call the forward and backward pass of the neural network and use the gradients ```layer.d_w, layer.d_b``` to update the weights and biases ```layer.w, layer.b``` of each layer in the network according to the update rule. Note that activation layers don't have any associated parameters.
