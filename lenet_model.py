import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters
from theano.tensor.signal.downsample import max_pool_2d

def _build_conv_pool(P, n_layer, input_layer, n_feats_out, n_feats_in, conv_size, pool_size):
    P["W_%d"%n_layer] = U.initial_weights(n_feats_out, n_feats_in, conv_size, conv_size)
    P["b_%d"%n_layer] = np.zeros((n_feats_out, ))
    W = P["W_%d"%n_layer]
    b = P["b_%d"%n_layer]
    out_conv = T.nnet.conv2d(input_layer, W)
    out_pool = max_pool_2d(out_conv, (pool_size, pool_size))
    output = T.nnet.sigmoid(out_pool + b.dimshuffle('x', 0, 'x', 'x'))
    return output

def build(P, n_input, n_hidden, n_output):
    P.W_hidden_output = U.initial_weights(n_hidden, n_output)
    P.b_output        = np.zeros(n_output)
    # n_hidden = 50 * 4 * 4  = 800 (n_feats of layer2 * pixels in image)

    # TODO: fix these magic numbers (especially the 800)
    def f(X):
        layer0 = X.reshape((X.shape[0], 1, 28, 28))
        layer1 = _build_conv_pool(P, 1, layer0, 20,  1, 5, 2)
        layer2_= _build_conv_pool(P, 2, layer1, 50, 20, 5, 2)
        layer2 = layer2_.flatten(2)
        output = T.nnet.softmax(T.dot(layer2, P.W_hidden_output) + P.b_output)
        return output

    return f

def cost(P, Y_hat, Y, l2 = 0):
    return (T.mean(T.nnet.categorical_crossentropy(Y_hat, Y)) +
           l2 * sum(T.mean(p**2) for p in P.values()))

if __name__ == "__main__":
    import datasets
    x,y = datasets.mnist()
    x,y = x[0:1000],y[0:1000]

    P = Parameters()
    X = T.matrix('X')
    Y = T.ivector('Y')
    net = build(P, 784, 800, 10)
    Y_hat = net(X)
    
    f = theano.function(inputs = [X], outputs = Y_hat)
    J = cost(P, Y_hat, Y)
    grad = T.grad(J, wrt=P.values())
