from itertools import islice

class Trainer(object):
    "Simple wrapper to train a model and keep track of history"
    def __init__(self, batches, train, predict):
        self.batches = batches
        self.train   = train
        self.predict = predict
        
        self.costs = []
        self.accs  = []

    def __call__(self, nsteps=10, checkpt=1):
        "Trains for `nsteps` steps, and prints info every `checkpt` steps"
        try:
            batches = islice(self.batches, nsteps)
            for i,batch in enumerate(batches):
                self.train_step(batch)
                if (i % checkpt == 0):
                    self.calc_acc(batch)
                    self.print_info()
        except KeyboardInterrupt:
            print "training interrupted."

    def train_step(self, batch):
        "Performs single training iteration on given batch"
        x,y  = batch
        cost = self.train(x, y)
        self.costs.append(cost.item())
        self.accs.append(None)

    def calc_acc(self, batch):
        "Calculates model accuracy on given batch"
        x,y = batch
        acc = 100.0 * sum(y == self.predict(x)) / len(y)
        self.accs[-1] = acc

    def print_info(self):
        "Prints current model cost and accuracy, if calculated"
        info = "iteration %d: cost = %f" % (len(self.costs),self.costs[-1])
        if self.accs[-1]:
            info += ", accuracy = %.2f%%" % self.accs[-1]
        print info
