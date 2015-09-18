from theano import *
import theano.tensor as T
import numpy as np
from mnist import *
rng = numpy.random

# hyperparameters
h      = 200  # number of units in hidden layer
lam    = 0.00 # L2 regularization
alpha  = 0.1  # learning rate
nsteps = 100  # number of gradient descent iterations

# load mnist
MNIST_PATH = "/home/jeffrey/.data/mnist/"
train_X, train_y = load_mnist(path=MNIST_PATH, selection = slice(0, 5000, 1))
# m, n = 5000, 784
# train_X = rng.randn(m, n)
# train_y = rng.randint(size = m, low = 0, high = 9)

# size of data
m, n = train_X.shape
k = len(np.unique(train_y)) # num of classes

print "Dataset loaded..."

### THEANO SYMBOLIC VARIABLES---------------
# model parameters
X = T.dmatrix('X')
y = T.ivector('y')
X.tag.test_value = train_X
y.tag.test_value = train_y

W1 = shared(rng.randn(n, h), name = 'W1')
b1 = shared(np.zeros(h), name = 'b1')
W2 = shared(rng.randn(h, k), name = 'W2')
b2 = shared(np.zeros(k), name = 'b2')
params = [W1, b1, W2, b2]

# forward propogation
H    = T.nnet.sigmoid(T.dot(X, W1) + b1)
H.tag.test_value = np.zeros((h,1))
yhat = T.nnet.softmax(T.dot(H, W2) + b2)
yhat.tag.test_value = np.zeros(k)

# loss function
loss = T.nnet.categorical_crossentropy(yhat, y).sum() \
       +  lam * ((W1**2).sum() + (W2**2).sum())
grad = T.grad(loss, params)

# train method
train = function(
    inputs = [X,y],
    outputs = loss,
    updates = [(p, p - alpha*dp) for (p, dp) in zip(params, grad)],
    name = "train",
    allow_input_downcast = True
)
predict = function([X], yhat.argmax(axis = 1))

print "Theano compiled, starting training..."

### DO GRADIENT DESCENT --------------
check = 1
for i in range(nsteps):
    cost = train(train_X, train_y)
    if(i % check == 0): 
        accuracy = 100 * (1 - (1.0*len((train_y - predict(train_X)).nonzero()[0]) / m))
        print "%d'th iteration: cost = %.f, accuracy = %.2f%%" % (i, cost, accuracy)

print "done"
