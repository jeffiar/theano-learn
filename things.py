import pylab

# define some useful functions down below
def plot_cost(net):
    x = range(net.nsteps)
    y = net.costs
    pylab.plot(x, y)
    pylab.xlabel("number of training steps")
    pylab.ylabel("cost")
    pylab.title("Neural Network cost throughout training")
    pylab.show()

def plot_acc(net):
    x = range(net.nsteps)
    y = net.accs
    pylab.plot(x, y)
    pylab.xlabel("number of training steps")
    pylab.ylabel("model accuracy")
    pylab.title("Neural Network accuracy throughout training")
    pylab.show()
