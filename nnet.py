from theano import *
import theano.tensor as T
import numpy as np
import numpy.random as rng

import datasets
from things import * # helpful functions

# hyperparameters
h     = 100  # number of units in hidden layer
lam   = 0.01 # L2 regularization
alpha = 0.1  # learning rate

# load mnist
x_train, y_train = datasets.mnist()
m, n = x_train.shape
k = len(np.unique(y_train)) # num of classes
print "Dataset loaded..."

### THEANO SYMBOLIC VARIABLES---------------
# model parameters
x = T.dmatrix('x')
y = T.ivector('y')

W1 = shared(rng.randn(n, h), name = 'W1')
b1 = shared(rng.randn(h),    name = 'b1')
W2 = shared(rng.randn(h, k), name = 'W2')
b2 = shared(rng.randn(k),    name = 'b2')
params = [W1, b1, W2, b2]

# forward propogation
H      = T.nnet.sigmoid(T.dot(x, W1) + b1)
y_hat  = T.nnet.softmax(T.dot(H, W2) + b2)
y_pred = T.argmax(y_hat, axis = 1)

# loss function
loss = T.mean(T.nnet.categorical_crossentropy(y_hat, y)) \
        +  lam * (T.mean(T.sqr(W1)) + T.mean(T.sqr(W2)))
grad = T.grad(loss, params)

# train method
train = function(
    inputs = [x,y],
    outputs = loss,
    updates = [(p, p - alpha*dp) for (p, dp) in zip(params, grad)],
    name = "train",
    allow_input_downcast = True
)
predict = function([x], y_pred)
print "Theano compiled..."

### DO GRADIENT DESCENT --------------
def train_nnet(nsteps = 50, checkpt = 5):
    print "Starting training..."
    costs = []
    for i in range(nsteps):
        cost = train(x_train, y_train)
        costs.append(cost.tolist())
        if(i % checkpt == 0): 
            accuracy = 100 * (1 - (1.0*len((y_train - predict(x_train)).nonzero()[0]) / m))
            print "%d'th iteration: cost = %.5f, accuracy = %.2f%%" % (i, cost, accuracy)
    return costs
