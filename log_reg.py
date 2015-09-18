from theano import *
import theano.tensor as T
rng = numpy.random

# hyperparameters
lam = 0.01 # L2 penalty
alpha = 0.1 # learning rate
training_steps = 10000

# training data
m = 400
n = 784
train_X = rng.randn(m, n)
train_y = rng.randint(size = m, low = 0, high = 2)

# Declare Theano symbolic variables
X = T.matrix('X') # input matrix
y = T.vector('y') # output vector
W = theano.shared(rng.randn(n), name = "W") # weights
b = theano.shared(0.0, name = "b")          # biases

WXpb      = T.dot(X, W) + b        # W*X + b
yhat      = 1 / (1 + T.exp(-WXpb)) # P(y_i = 1 | x_i, theta)
pred      = yhat > 0.5
x_entropy = (-y * T.log(yhat) - (1-y) * T.log(1-yhat)).mean()
L2        = T.dot(W,W)               #L2
loss      = x_entropy + lam * L2     # L2-regularized loss fucntion
d_W, d_b  = T.grad(loss, [W, b])     # gradients wrt W and b

# Compile
train = theano.function(
    inputs  = [X,y],
    outputs = loss,
    updates = [(W, W - alpha * d_W),
               (b, b - alpha * d_b)]
)
predict = theano.function([X], pred)

# # train
for i in range(training_steps):
    if(i % 1000 == 0): print(i)
    train(train_X, train_y)

print "predicted y values:"
print predict(train_X)
print "target y values:"
print train_y
