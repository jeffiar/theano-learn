import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters
from itertools import islice

import model
import datasets

def make_batches(x, y, batch_size=1000):
    """Generates tuples (x,y) of training mini-batches"""
    n_samples = x.shape[0]
    while True:
        shuffle = np.random.permutation(n_samples)
        x,y = x[shuffle],y[shuffle]
        for i in xrange(n_samples/batch_size):
            idxs = slice(i*batch_size, (i+1) * batch_size)
            yield x[idxs],y[idxs]

def make_train_loop(_batches, train, predict):
    def train_loop(nsteps=10, checkpt=1):
        """Trains for `nsteps` steps, and prints info every `checkpt` steps"""
        batches = islice(_batches, nsteps)
        for i, batch in enumerate(batches):
            x,y = batch
            cost = train(x, y)
            acc = 100.0 * sum(y == predict(x)) / len(y)
            if(i % checkpt == 0): 
                print "%d'th iteration: cost = %.5f, accuracy = %.2f%%" % (i, cost, acc)
    return train_loop

if __name__ == "__main__":
    print "Loading dataset..."
    x_train,y_train = datasets.mnist()
    batches = make_batches(x_train, y_train, 1000)
    _,n = x_train.shape
    k = len(np.unique(y_train))

    print "Compiling theano..."
    X = T.matrix('X')
    Y = T.ivector('Y')
    P = Parameters()

    net     = model.build(P, n, 200, k)
    Y_hat   = net(X)
    Y_pred  = T.argmax(Y_hat, axis=1)
    predict = theano.function([X], Y_pred)
    cost   = model.cost(P, Y_hat, Y)

    params = P.values()
    grad   = T.grad(cost, wrt = params)
    train  = theano.function([X,Y], cost,
                updates=updates.rmsprop(params, grad))
    train_loop = make_train_loop(batches, train, predict)

    print "Starting training..."
    train_loop()
