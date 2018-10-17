import neuralnet
import numpy as np
import pickle

def main():
    # make_pickle()
    benchmark_data = pickle.load(open('validate_data.pkl', 'rb'), encoding='latin1')

    config = {}
    config['layer_specs'] = [784, 100, 100, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
    config['activation'] = 'sigmoid' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
    config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
    config['epochs'] = 50  # Number of epochs to train the model
    config['early_stop'] = True  # Implement early stopping or not
    config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
    config['L2_penalty'] = 0  # Regularization constant
    config['momentum'] = False  # Denotes if momentum is to be applied or not
    config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression

    np.random.seed(42)
    x = np.random.randn(1, 100)
    act_sigmoid = neuralnet.Activation('sigmoid')
    act_tanh = neuralnet.Activation('tanh')
    act_ReLU = neuralnet.Activation('ReLU')
    
    
    out_sigmoid = act_sigmoid.forward_pass(x)
    err_sigmoid = np.sum(np.abs(benchmark_data['out_sigmoid'] - out_sigmoid))
    check_error(err_sigmoid, "Sigmoid Forward Pass")

    out_tanh = act_tanh.forward_pass(x)
    err_tanh = np.sum(np.abs(benchmark_data['out_tanh'] - out_tanh))
    check_error(err_tanh, "Tanh Forward Pass")

    out_ReLU = act_ReLU.forward_pass(x)
    err_ReLU = np.sum(np.abs(benchmark_data['out_ReLU'] - out_ReLU))
    check_error(err_ReLU, "ReLU Forward Pass")

    print("**************")

    grad_sigmoid = act_sigmoid.backward_pass(1.0)
    err_sigmoid_grad = np.sum(np.abs(benchmark_data['grad_sigmoid'] - grad_sigmoid))
    check_error(err_sigmoid_grad, "Sigmoid Gradient")

    grad_tanh = act_tanh.backward_pass(1.0)
    err_tanh_grad = np.sum(np.abs(benchmark_data['grad_tanh'] - grad_tanh))
    check_error(err_tanh_grad, "Tanh Gradient")

    grad_ReLU = act_ReLU.backward_pass(1.0)
    err_ReLU_grad = np.sum(np.abs(benchmark_data['grad_ReLU'] - grad_ReLU))
    check_error(err_ReLU_grad, "ReLU Gradient")
    
    np.random.seed(42)
    x_image = np.random.randn(1, 784)

    nnet = neuralnet.Neuralnetwork(config)
    nnet.forward_pass(x_image, targets = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    nnet.backward_pass()

    layer_no = 0
    for layer_idx, layer in enumerate(nnet.layers):
        if isinstance(layer, neuralnet.Layer):
            layer_no += 1
            error_x = np.sum(np.abs(benchmark_data['nnet'].layers[layer_idx].x - layer.x))
            error_w = np.sum(np.abs(benchmark_data['nnet'].layers[layer_idx].w - layer.w))
            error_b = np.sum(np.abs(benchmark_data['nnet'].layers[layer_idx].b - layer.b))
            error_d_w = np.sum(np.abs(benchmark_data['nnet'].layers[layer_idx].d_w - layer.d_w))
            error_d_b = np.sum(np.abs(benchmark_data['nnet'].layers[layer_idx].d_b - layer.d_b))

            check_error(error_x, "Layer{} Input".format(layer_no))
            check_error(error_w, "Layer{} Weights".format(layer_no))
            check_error(error_b, "Layer{} Biases".format(layer_no))
            check_error(error_d_w, "Layer{} Weight Gradient".format(layer_no))
            check_error(error_d_b, "Layer{} Bias Gradient".format(layer_no))


    
    # print(err_sigmoid)
    # print(err_tanh)
    # print(err_ReLU)
  
  
def check_error(error, msg):
    if error < 1e-6:
        print("{} is CORRECT".format(msg))
    else:
        print("{} is WRONG".format(msg))
  
  
if __name__ == '__main__':
  main()