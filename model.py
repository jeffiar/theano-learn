import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters

def build(P, n_input, n_hidden, n_output):
    P.W_input_hidden  = U.initial_weights(n_input, n_hidden)
    P.W_hidden_output = U.initial_weights(n_hidden, n_output)
    P.b_hidden        = U.initial_weights(n_hidden)
    P.b_output        = U.initial_weights(n_output)

    def f(X):
        hidden = T.nnet.sigmoid(T.dot(X,      P.W_input_hidden  + P.b_hidden))
        output = T.nnet.softmax(T.dot(hidden, P.W_hidden_output + P.b_output))
        return output

    return f

if __name__ == "__main__":
    P = Parameters()
    X = T.matrix('X')
    net = build(P, 2, 10, 1)
    y = net(X)
    
    f = theano.function(inputs = [X], outputs = y)
