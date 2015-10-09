import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters

import model
import datasets

def get_cost(Y_hat, Y, P = None, l2 = 0):
    cost = T.mean(T.nnet.categorical_crossentropy(Y_hat, Y))
    if P is not None:
        for p in P.values():
            cost += l2 * T.mean(T.sqr(p))
    return cost

if __name__ == "__main__":
    print "Loading dataset..."
    x_train, y_train = datasets.mnist(selection = slice(0, 1000))
    m, n = x_train.shape
    k = len(np.unique(y_train))

    print "Compiling theano..."
    P = Parameters()
    X = T.matrix('X')
    Y = T.ivector('Y')
    net = model.build(P, n, 100, k)
    Y_hat = net(X)

    cost = get_cost(Y_hat, Y, P, l2 = 0.01)
    params = P.values()
    grad = T.grad(cost, wrt = params)
    train = theano.function(
                inputs  = [X,Y],
                outputs = cost,
                updates = updates.adadelta(params, grad),
                allow_input_downcast = True
            )
    predict = theano.function(
                inputs = [X],
                outputs = T.argmax(Y_hat, axis = 1),
                allow_input_downcast = True
            )

    def train_loop(nsteps = 50, checkpt = 5):
        for i in range(nsteps):
            cost = train(x_train, y_train)
            if(i % checkpt == 0): 
                accuracy = 100 * (1 - (1.0*len((y_train - predict(x_train)).nonzero()[0]) / m))
                print "%d'th iteration: cost = %.5f, accuracy = %.2f%%" % (i, cost, accuracy)

    print "Starting training..."
    train_loop()
