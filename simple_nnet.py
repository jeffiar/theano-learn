from theano import *
import theano.tensor as T
import numpy as np
import numpy.random as rng

# hyperparameters
# TODO: put this in a better place
h     = 100  # number of units in hidden layer
lam   = 0.01 # L2 regularization
alpha = 0.1  # learning rate

def get_acc(y, y_hat):
    return 1 - (1.0*len((y - y_hat).nonzero()[0]) / len(y))

# Simple feedforward neural network class
class model:
    def __init__(self, x_train, y_train):
        # TODO: allow different architectures
        # TODO: make function neater
        m, n = x_train.shape
        k = len(np.unique(y_train)) # num of classes

        self.x_train = x_train
        self.y_train = y_train

        # Theano symbolic variables for net
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
        # TODO: see if this works:
        #       lam * T.sum([T.mean(T.sqr(W)) for W in params])
        grad = T.grad(loss, params)

        self.__train__ = function(
            inputs = [x,y],
            outputs = loss,
            updates = [(p, p - alpha*dp) for (p, dp) in zip(params, grad)],
            name = "train",
            allow_input_downcast = True
        )
        self.predict = function([x], y_pred)

        self.nsteps = 0
        self.costs, self.accs = [], []

    def train(self, nsteps = 10, checkpt = None):
        """ Do nsteps iterations of gradient descent, 
        printing status every checkpt steps"""
        if checkpt is None: checkpt = nsteps / 10
        for i in range(nsteps):
            self.train_step()
            if i % checkpt == 0:
                self.print_info()
        self.print_info()

    def train_step(self):
        """ Do a single iteration of gradient descent """
        cost = self.__train__(self.x_train, self.y_train).tolist()
        acc = 100 * get_acc(self.y_train, self.predict(self.x_train))

        self.accs.append(acc)
        self.costs.append(cost)
        self.nsteps = self.nsteps + 1

    def print_info(self):
        print "%d'th iteration: cost = %.4f, accuracy = %.2f%%" \
                % (self.nsteps, self.costs[-1], self.accs[-1])

    def save_params(filename):
        # TODO: do this
        pass

    def load_params(filename):
        pass
