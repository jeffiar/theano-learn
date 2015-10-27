import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters

import TF_model as model
import datasets
from trainer import Trainer

def make_batches(x, y, batch_size=64):
    "Generates tuples (x,y) of training mini-batches"
    n_samples = x.shape[0]

    while True:
        shuffle = np.random.permutation(n_samples)
        x,y = x[shuffle],y[shuffle]

        for i in xrange(n_samples/batch_size):
            idxs = slice(i*batch_size, (i+1) * batch_size)
            yield x[idxs],y[idxs]

def preprocess(x,y):
    x = np.append(x, np.fliplr(x))
    y = np.append(y,y)
    return x,y

if __name__ == "__main__":
    print "Loading dataset..."
    x,y = datasets.transcription_factor()
    x,y   = preprocess(x,y)
    batches = make_batches(x, y)

    print "Compiling theano..."
    X = T.fmatrix('X')
    Y = T.fmatrix('Y')
    P = Parameters()

    net     = model.build(P)
    Y_hat   = net(X)
    predict = theano.function([X], Y_hat, allow_input_downcast=True)
    cost    = model.cost(P, Y_hat, Y)

    print "Calculating gradient..."
    params = P.values()
    grad   = T.grad(cost, wrt = params)
    grad   = [g.astype(theano.config.floatX) for g in grad] #idek...
    train  = theano.function([X,Y], cost,
                updates=updates.momentum(params, grad),
                allow_input_downcast=True)
    trainer = Trainer(batches, train, predict)

    # trainer()
