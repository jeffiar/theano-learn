import datasets
import simple_nnet

x_train, y_train = datasets.mnist(selection = slice(0, 5000, 1))
print "dataset loaded"
net = simple_nnet.model(x_train, y_train)
print "theano compiled"
net.train()
