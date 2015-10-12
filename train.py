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
    "Generates tuples (x,y) of training mini-batches"
    n_samples = x.shape[0]

    while True:
        shuffle = np.random.permutation(n_samples)
        x,y = x[shuffle],y[shuffle]

        for i in xrange(n_samples/batch_size):
            idxs = slice(i*batch_size, (i+1) * batch_size)
            yield x[idxs],y[idxs]


def make_train_loop(_batches, train, predict):
    def train_loop(nsteps=10, checkpt=1):
        "Trains for `nsteps` steps, and prints info every `checkpt` steps"
        try:
            batches = islice(_batches, nsteps)
            for i, batch in enumerate(batches):
                x,y  = batch
                cost = train(x, y)

                if(i % checkpt == 0):
                    acc = 100.0 * sum(y == predict(x)) / len(y)
                    print "%d'th iteration: cost = %.6f, accuracy = %.2f%%" % (i, cost, acc)

        except KeyboardInterrupt:
            print "training interrupted."

    return train_loop


if __name__ == "__main__":
    print "Loading dataset..."
    x,y = datasets.mnist()
    batches = make_batches(x, y, 1000)

    print "Compiling theano..."
    X = T.matrix('X')
    Y = T.ivector('Y')
    P = Parameters()

    # TODO: find better place to put magic numbers
    net     = model.build(P, 784, 800, 10)
    Y_hat   = net(X)
    Y_pred  = T.argmax(Y_hat, axis=1)
    predict = theano.function([X], Y_pred)
    cost    = model.cost(P, Y_hat, Y)

    print "Calculating gradient..."
    params = P.values()
    grad   = T.grad(cost, wrt = params)
    train  = theano.function([X,Y], cost,
                updates=updates.gradient_descent(params, grad, learning_rate=0.1))
    train_loop = make_train_loop(batches, train, predict)

    # train_loop()
