# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

import numpy as np

from util import *


class ElmanNetwork:
    """
    Simple Recurrent Network with Elman's original simple BP training
    """

    def __init__(self, dim_in, dim_hid, dim_out):
        """
        Initialize model, set initial weights
        """
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.dim_out = dim_out

        self.W_in = np.random.randn(dim_hid, dim_in+1)  # FIXME
        self.W_rec = np.random.randn(dim_hid, dim_hid+1)  # FIXME
        self.W_out = np.random.randn(dim_out, dim_hid+1)  # FIXME

        self.W_in *= 0.1  # Small initial weights help prevent instabilities
        self.W_rec *= 0.1
        self.W_out *= 0.1

        self.reset()

    def reset(self):
        """
        Reset - initialize context values to empty activations:
        c(0) = f_hid(net of zero input/context)
        Alternatively: don't use zeros as "neutral" activation, and try some statistically sound random input/sequence
        of inputs.
        """
        self.context = np.zeros(self.dim_hid)  # FIXME

    ## Activation functions & derivations
    def error(self, targets, outputs):
        return np.sum((targets - outputs)**2, axis=0)

    def f_hid(self, x):
        return 1 / (1 + np.exp(-x))

    def df_hid(self, x):
        return self.f_hid(x) * (1 - self.f_hid(x))

    def f_out(self, x):
        return x

    def df_out(self, x):
        return np.ones(x.shape)

    def forward(self, x):
        """
        Forward pass - single time-step
        x: single input vector (without bias, size=dim_in)
        """
        c = self.context
        a = self.W_in@add_bias(x) + self.W_rec@add_bias(c) # FIXME
        h = self.f_hid(a)  # FIXME
        b = self.W_out@add_bias(h)  # FIXME
        y = self.f_out(b)  # FIXME

        # Update context for next time-step
        self.context = h

        return c, a, h, b, y


    def backward(self, x, c, a, h, b, y, d):
        """
        Backprop pass - compute dW for given input and activations
        x: single input vector (without bias, size=dim_in)
        c: value of context - vector (without bias, size=dim_hid)
        a: net vector on hidden layer (size=dim_hid)
        h: activation of hidden layer (without bias, size=dim_hid)
        b: net vector on output layer (size=dim_out)
        y: output vector of network (size=dim_out)
        d: single target vector (size=dim_out)
        """
        g_out = (d - y) * self.df_out(b)
        g_hid = (self.W_out.T[:-1] @ g_out) * self.df_hid(a)

        dW_in  = np.outer(g_hid, add_bias(x))  # FIXME
        dW_rec = np.outer(g_hid, add_bias(c))  # FIXME
        dW_out = np.outer(g_out, add_bias(h))  # FIXME

        return dW_in, dW_rec, dW_out

    def train(self, inputs, targets, alpha=0.1, eps=100):
        """
        Training of the network
        inputs: matrix of input vectors (each column is one input vector)
        targets: matrix of target vectors (each column is one target vector)
        alpha: learning rate
        eps: number of episodes
        """
        (_, count) = inputs.shape

        errors = []

        for ep in range(eps):
            E = 0

            # Start new sequence with empty state!
            self.reset()

            for idx in range(count):
                x = inputs[:, idx]
                d = targets[:, idx]

                c, a, h, b, y = self.forward(x)
                dW_in, dW_rec, dW_out = self.backward(x, c, a, h, b, y, d)

                E += self.error(d,y)

                self.W_in  += alpha*dW_in  # FIXME
                self.W_rec += alpha*dW_rec  # FIXME
                self.W_out += alpha*dW_out  # FIXME

            E /= count
            errors.append(E)
            print('Ep {:3d}/{}: E = {:.3f}'.format(ep+1, eps, E))

        return errors

    # # Testing functions
    def one_step_predict_seq(self, inputs):
        """
        Let the network predict next value of sequence.
        Essentially, this is just forward-pass for every value in sequence.
        """
        outputs = []

        self.reset()

        for x in inputs.T:
            *_, y = self.forward(x)
            outputs.append(y)

        return np.array(outputs)
