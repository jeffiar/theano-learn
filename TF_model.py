import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters
from theano.tensor.signal.downsample import max_pool_2d
from utils import relu

def build(P):
    image_row = 35 # num of base pairs
    image_col = 4  # num of nucleotides
    n_input   = image_row * image_col

    n_feats = [1, 16] # num of "motifs" We'll learn 16 PWM's

    conv_row = 8 # 8-long PWM
    conv_col = 4 # 4 nucleotides
    pool_row = 28 # ??
    pool_col = 1

    n_pool_out = (n_feats[1] 
                  * ((image_row - conv_row + 1) / pool_row)
                  * ((image_col - conv_col + 1) / pool_col))
    n_hidden   = 32
    n_output   = 1

    P.W_input_conv      = U.initial_weights(n_feats[1], n_feats[0], conv_row, conv_col)
    P.b_pool_out        = np.zeros(n_pool_out)
    P.W_pool_out_hidden = U.initial_weights(n_pool_out, n_hidden)
    P.b_hidden          = np.zeros(n_hidden)
    P.W_hidden_output   = U.initial_weights(n_hidden, n_output)
    P.b_output          = np.zeros(n_output)

    def f(X):
        n_samples = X.shape[0]

        input    = X.reshape((n_samples, n_feats[0], image_row, image_col))
        conv_out = T.nnet.conv2d(input, P.W_input_conv)
        pool_out_= max_pool_2d(conv_out, (pool_row, pool_col))
        pool_out = pool_out_.flatten(2) + P.b_pool_out
        hidden   = relu(T.dot(pool_out, P.W_pool_out_hidden) + P.b_hidden)
        output   = T.dot(hidden, P.W_hidden_output) + P.b_output
        return output.astype(theano.config.floatX)

    return f

def cost(P, Y_hat, Y, l2 = 0):
    return (T.mean((Y - Y_hat)**2) +
           l2 * sum(T.mean(p**2) for p in P.values()))

if __name__ == "__main__":
    import datasets
    x,y = datasets.transcription_factor()

    P = Parameters()
    X = T.fmatrix('X')
    Y = T.fmatrix('Y')
    net = build(P)
    Y_hat = net(X)
    
    f = theano.function(inputs = [X], outputs = Y_hat, allow_input_downcast=True)
    J = cost(P, Y_hat, Y)
    grad = T.grad(J, wrt=P.values())

    y_hat = f(x)
