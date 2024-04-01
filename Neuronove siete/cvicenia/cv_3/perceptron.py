# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Juraj Holas, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2020-2023

import numpy as np

from util import *

# Vypracoval: Marian Kravec

class Perceptron:
    def __init__(self, dim_in, dim_out):
        # Initialize perceptron and data.
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.W = np.random.rand(dim_out, dim_in+1)  # FIXME

    def compute_accuracy(self, inputs, targets):
        # Computes classification accuracy - percentage of correctly categorized inputs
        return np.mean([d.argmax() == self.compute_output(add_bias(x)).argmax()
                        for (x, d) in zip(inputs.T, targets.T)])

    def compute_output(self, x):
        # Computes output (vector y) of the neural network for given input vector x (including bias).
        # Allow for X to be matrix of all inputs (with biases) if you want efficient batch solution.
        y = self.softmax(self.W @ x)     # FIXME
        return y

    def softmax(self, y):
        return np.exp(y)/np.sum(np.exp(y), axis=0)

    def train_seq(self, inputs, targets, alpha=0.1, eps=100, compute_accuracy=True):
        # Trains the neural network in sequential (stochastic) mode.
        # After each epoch, per-epoch classification accuracy is appended into history, that
        # is return for further plotting.
        count = inputs.shape[1]  # number of input-target pairs
        accuracy_history = [self.compute_accuracy(inputs, targets)]
        for ep in range(eps):

            for idx in np.random.permutation(count):
                x = add_bias(inputs[:, idx])         # FIXME
                d = targets[:, idx]       # FIXME
                y = self.compute_output(x)       # FIXME
                self.W += alpha*(np.outer((d-y),x.T)) # FIXME

            if compute_accuracy:
                acc = self.compute_accuracy(inputs, targets)
                accuracy_history.append(acc)
                if (ep+1) % 5 == 0:
                    print('Epoch {:3d}, accuracy = {:4.1%}'.format(ep+1, acc))

        return accuracy_history

    def train_batch(self, inputs, targets, alpha=0.1, eps=100, compute_accuracy=True):
        # Trains the neural network in batch mode.
        # After each epoch, per-epoch classification accuracy is appended into history, that
        # is return for further plotting.
        count = inputs.shape[1]  # number of input-target pairs
        accuracy_history = [self.compute_accuracy(inputs, targets)]

        for ep in range(eps):

            perm = np.random.permutation(count)

            input_ep = add_bias(inputs[:, perm])
            targets_ep = targets[:, perm]

            y_ep = self.compute_output(input_ep)       # FIXME
            self.W += alpha*((targets_ep-y_ep)@input_ep.T)  # FIXME

            if compute_accuracy:
                acc = self.compute_accuracy(inputs, targets)
                accuracy_history.append(acc)
                if (ep+1) % 5 == 0:
                    print('Epoch {:3d}, accuracy = {:4.1%}'.format(ep+1, acc))

        return accuracy_history
