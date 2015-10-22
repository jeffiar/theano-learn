import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters

import TF_model as model
import datasets
from trainer import Trainer

def make_batches(x, y, batch_size=1000):
    "Generates tuples (x,y) of training mini-batches"
    n_samples = x.shape[0]

    while True:
        shuffle = np.random.permutation(n_samples)
        x,y = x[shuffle],y[shuffle]

        for i in xrange(n_samples/batch_size):
            idxs = slice(i*batch_size, (i+1) * batch_size)
            yield x[idxs],y[idxs]


if __name__ == "__main__":
    print "Loading dataset..."
    x,y = datasets.transcription_factor()
    batches = make_batches(x, y, 1000)

    print "Compiling theano..."
    X = T.imatrix('X')
    Y = T.fmatrix('Y')
    P = Parameters()

    # TODO: find better place to put magic numbers
    net     = model.build(P, 140, 100, 1)
    Y_hat   = net(X)
    predict = theano.function([X], Y_hat)
    cost    = model.cost(P, Y_hat, Y)

    print "Calculating gradient..."
    params = P.values()
    grad   = T.grad(cost, wrt = params)
    grad   = [g.astype(theano.config.floatX) for g in grad] #idek...
    train  = theano.function([X,Y], cost,
                updates=updates.gradient_descent(params, grad))
    trainer = Trainer(batches, train, predict)

    # trainer()
