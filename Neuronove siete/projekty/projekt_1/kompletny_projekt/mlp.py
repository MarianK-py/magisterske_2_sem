# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024
import numpy as np

from util import *

# Vypracoval: Marian Kravec

class MLP:
    """
    Multi-Layer Perceptron (abstract base class)
    """

    def __init__(self, dim_in, dim_hid1, dim_hid2, dim_out, lab_vals):
        """
        Initialize model, set initial weights
        """
        self.lab_vals = lab_vals

        self.dim_in = dim_in
        self.dim_hid1 = dim_hid1
        self.dim_hid2 = dim_hid2
        self.dim_out = dim_out

        self.W_hid1 = np.random.randn(dim_hid1, dim_in+1)
        self.W_hid2 = np.random.randn(dim_hid2, dim_hid1+1)
        self.W_out = np.random.randn(dim_out, dim_hid2+1)

        self.v_hid1 = np.zeros([dim_hid1, dim_in+1])
        self.v_hid2 = np.zeros([dim_hid2, dim_hid1+1])
        self.v_out = np.zeros([dim_out, dim_hid2+1])

        self.s_hid1 = 0
        self.s_hid2 = 0
        self.s_out = 0


        # Activation functions & derivations
    # (not implemented, to be overridden in derived classes)
    def f_hid1(self, x):
        raise NotImplementedError

    def df_hid1(self, x):
        raise NotImplementedError

    def f_hid2(self, x):
        raise NotImplementedError

    def df_hid2(self, x):
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
        a = self.W_hid1 @ xB
        h = self.f_hid1(a)

        hB = add_bias(h)
        b = self.W_hid2 @ hB
        i = self.f_hid2(b)

        iB = add_bias(i)
        c = self.W_out @ iB
        y = self.f_out(c)

        return a, h, b, i, c, y

    def backward(self, alpha, beta_1, beta_2, epoch, x, a, h, b, i, c, y, d, batch):
        """
        Backprop pass - compute dW for given input and activations
        x: single input vector (without bias, size=dim_in)
        a: net vector on hidden layer (size=dim_hid)
        h: activation of hidden layer (without bias, size=dim_hid)
        b: net vector on output layer (size=dim_out)
        y: output vector of network (size=dim_out)
        d: single target vector (size=dim_out)
        """

        g_out = (d - y) * self.df_out(c)
        g_hid2 = (self.W_out.T @ g_out)[:-1] * self.df_hid2(b)
        g_hid1 = (self.W_hid2.T @ g_hid2)[:-1] * self.df_hid2(a)

        iB = add_bias(i)
        hB = add_bias(h)
        xB = add_bias(x)

        if batch:
            G_out = g_out @ iB.T
            G_hid2 = g_hid2 @ hB.T
            G_hid1 = g_hid1 @ xB.T
        else:
            G_out = np.outer(g_out, iB)
            G_hid2 = np.outer(g_hid2, hB)
            G_hid1 = np.outer(g_hid1, xB)

        # Learning rate using Adam
        self.v_out = beta_1*self.v_out + (1-beta_1)*G_out
        self.s_out = beta_2*self.s_out + (1-beta_2)*(np.sum(G_out**2))

        self.v_hid2 = beta_1*self.v_hid2 + (1-beta_1)*G_hid2
        self.s_hid2 = beta_2*self.s_hid2 + (1-beta_2)*(np.sum(G_hid2**2))

        self.v_hid1 = beta_1*self.v_hid1 + (1-beta_1)*G_hid1
        self.s_hid1 = beta_2*self.s_hid1 + (1-beta_2)*(np.sum(G_hid1**2))

        v_out_ = self.v_out/(1-beta_1**(epoch+1))
        s_out_ = self.s_out/(1-beta_2**(epoch+1))

        v_hid2_ = self.v_hid2/(1-beta_1**(epoch+1))
        s_hid2_ = self.s_hid2/(1-beta_2**(epoch+1))

        v_hid1_ = self.v_hid1/(1-beta_1**(epoch+1))
        s_hid1_ = self.s_hid1/(1-beta_2**(epoch+1))

        dW_out = (alpha/np.sqrt(s_out_))*v_out_
        dW_hid2 = (alpha/np.sqrt(s_hid2_))*v_hid2_
        dW_hid1 = (alpha/np.sqrt(s_hid1_))*v_hid1_

        #dW_out = alpha*G_out
        #dW_hid2 = alpha*G_hid2
        #dW_hid1 = alpha*G_hid1

        return dW_hid1, dW_hid2, dW_out
