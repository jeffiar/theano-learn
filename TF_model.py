import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters

def build(P, n_input, n_hidden, n_output):
    P.W_i_h = U.initial_weights(n_input, n_hidden)
    P.W_h_o = U.initial_weights(n_hidden, n_output)
    P.b_h   = U.initial_weights(n_hidden)
    P.b_o   = U.initial_weights(n_output)

    def f(X):
        hidden = T.nnet.sigmoid(T.dot(X,      P.W_i_h) + P.b_h)
        output = T.dot(hidden, P.W_h_o) + P.b_o
        return output.astype(theano.config.floatX)

    return f

def cost(P, Y_hat, Y, l2 = 0):
    return (T.mean((Y - Y_hat)**2) +
           l2 * sum(T.mean(p**2) for p in P.values()))

if __name__ == "__main__":
    import datasets
    x,y = datasets.transcription_factor()

    P = Parameters()
    X = T.matrix('X')
    Y = T.ivector('Y')
    net = build(P, 140, 100, 1)
    Y_hat = net(X)
    
    f = theano.function(inputs = [X], outputs = Y_hat)
    J = cost(P, Y_hat, Y)
    grad = T.grad(J, wrt=P.values())
