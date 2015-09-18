from theano import *
import theano.tensor as T
import numpy as np
rng = numpy.random

# hyperparameters
lam = 0.01 # L2 penalty
alpha = 0.1 # learning rate
h = 200 # size of hidden layer
training_steps = 500

# training data
m = 4000
n = 784
k = 10 # num of classes
train_X = rng.randn(m, n)
train_y = rng.randint(size = m, low = 0, high = 9)

### THEANO SYMBOLIC VARIABLES---------------
# model parameters
X = T.dmatrix('X')
y = T.ivector('y')
# X.tag.test_value = train_X
# y.tag.test_value = train_y
W1 = theano.shared(rng.random((n, h)), name = 'W1')
b1  = theano.shared(np.zeros(h), name = 'b1')
W2 = theano.shared(rng.random((h, k)), name = 'W2')
b2  = theano.shared(np.zeros(k), name = 'b2')
params = [W1, b1, W2, b2]

# forward propogation
H    = T.nnet.sigmoid(T.dot(X, W1) + b1)
# H.tag.test_value = np.zeros(h)
yhat = T.nnet.softmax(T.dot(H, W2) + b2)
# yhat.tag.test_value = np.zeros(k)

# loss function
loss = T.nnet.categorical_crossentropy(yhat, y).sum()
loss += lam * ((W1**2).sum() + (W2**2).sum())
grad = T.grad(loss, params)

# train method
train = theano.function(
    inputs = [X,y],
    outputs = loss,
    updates = [(p, p - alpha*dp) for (p, dp) in zip(params, grad)],
    name = "train",
    allow_input_downcast = True
)
predict = theano.function([X], yhat.argmax(axis = 1))

stops = 50
checkpoints = [x*(training_steps/stops) for x in range(stops)]
for i in range(training_steps):
    train(train_X, train_y)
    if(i in checkpoints): 
        print "%d'th ieration. error rate %.3f%%" % (i, 100.0*len((train_y - predict(train_X)).nonzero()[0]) / m)

print "done"
