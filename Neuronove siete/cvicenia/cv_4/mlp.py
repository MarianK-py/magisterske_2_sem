# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

from util import *

# Vypracoval: Marian Kravec

class MLP:
    """
    Multi-Layer Perceptron (abstract base class)
    """

    def __init__(self, dim_in, dim_hid, dim_out):
        """
        Initialize model, set initial weights
        """

        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.dim_out = dim_out

        self.W_hid = np.random.randn(dim_hid, dim_in+1)  # FIXME
        self.W_out = np.random.randn(dim_out, dim_hid+1)  # FIXME

    # Activation functions & derivations
    # (not implemented, to be overridden in derived classes)
    def f_hid(self, x):
        raise NotImplementedError

    def df_hid(self, x):
        raise NotImplementedError

    def f_out(self, x):
        raise NotImplementedError

    def df_out(self, x):
        raise NotImplementedError

    # Back-propagation
    def forward(self, x):
        """
        Forward pass - compute output of network
        x: single input vector (without bias, size=dim_in)
        """
        xB = add_bias(x)

        a = self.W_hid @ xB  # FIXME net vector on hidden layer (size=dim_hid)
        h = self.f_hid(a)  # FIXME activation of hidden layer (without bias, size=dim_hid)

        hB = add_bias(h)
        b = self.W_out @ hB  # FIXME net vector on output layer (size=dim_out)
        y = self.f_out(b)  # FIXME output vector of network (size=dim_out)

        return a, h, b, y

    def backward(self, x, a, h, b, y, d):
        """
        Backprop pass - compute dW for given input and activations
        x: single input vector (without bias, size=dim_in)
        a: net vector on hidden layer (size=dim_hid)
        h: activation of hidden layer (without bias, size=dim_hid)
        b: net vector on output layer (size=dim_out)
        y: output vector of network (size=dim_out)
        d: single target vector (size=dim_out)
        """

        g_out = (d - y) * self.df_out(b)  # FIXME
        g_hid = (self.W_out.T @ g_out)[:-1] * self.df_hid(a)  # FIXME

        hB = add_bias(h)
        xB = add_bias(x)

        dW_out = np.outer(g_out, hB)   # FIXME
        dW_hid = np.outer(g_hid, xB)  # FIXME

        return dW_hid, dW_out
