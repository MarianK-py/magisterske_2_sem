# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024
import numpy as np

from util import *

#Marian Kravec

class ESN:
    def __init__(self, dim_in, dim_res, dim_out, spectral_radius, sparsity=0.7, weights_scale=0.01):
        self.dim_in = dim_in
        self.dim_res = dim_res
        self.dim_out = dim_out

        # rand ~ Uni(0,1) -> 2*rand-1 ~ Uni(-1,1) -> (2*rand-1)*weight ~ Uni(-weight,weight)
        self.W_in = (2*np.random.rand(dim_res, dim_in)-1) * weights_scale

        # randn ~ N(0,1)
        self.W_res = np.random.randn(dim_res, dim_res)
        self.W_out = np.random.randn(dim_out, dim_res)

        # Sparsity: delete each weight with probability P(W_res[i,j]==0) = sparsity
        self.W_res = (np.random.rand(dim_res, dim_res)>sparsity)*self.W_res

        # Spectral normalization: set greatest eigenvalue to spectral_radius
        eigenvalues, eigenvectors = np.linalg.eig(self.W_res)
        self.W_res *= spectral_radius/np.max(np.abs(eigenvalues))

        self.reset()

    # # Activation functions
    def f_res(self, X):
        return np.tanh(X) # hyperbolic tangens
        #return X # linear
        #return 1/(1+np.exp(-X))

    def f_out(self, X):
        return X

    def reset(self):
        """
        Reset - initialize reservoir values to empty activations: r(0) = f_res(net of zero input/context)
        Alternatively: don't use zeros as "neutral" activation, and try some statistically sound random input/sequence of inputs.
        """
        self.reservoir = self.f_res(np.zeros(self.dim_res))

    def forward(self, x):
        """
        Forward pass - single time-step
        x: single input vector
        """
        x = np.atleast_1d(x)  # To ensure that x is vector, not scalar
        r = self.f_res((self.W_in@x)+(self.W_res@self.reservoir))
        y = self.f_out(self.W_out@r)

        # Update reservoir for next time-step
        self.reservoir = r

        return y, r

    def palate_cleaning(self, inputs):
        """
        Initial cleaning of palate of model
        """
        inputs = np.atleast_2d(inputs)
        (_, count) = inputs.shape

        for t in range(count):
            x = inputs[:, t]
            self.forward(x)


    def train(self, inputs, targets):
        """
        Training of the network
        inputs: matrix of input vectors (each column is one input vector)
        targets: matrix of target vectors (each column is one target vector)
        """
        inputs = np.atleast_2d(inputs)
        (_, count) = inputs.shape

        # Start new sequence with empty state!
        self.reset()

        # Collect reservoir states (they serve as input data for the output layer)
        R = np.zeros((self.dim_res, count))
        for t in range(count):
            x = inputs[:, t]
            y, r = self.forward(x)
            R[:, t] = r

        # Analytical training via pseudoinverse
        self.W_out = targets@np.linalg.pinv(R)

    # # Testing functions
    def one_step_predict_seq(self, inputs):
        """
        Let the network predict next value of sequence.
        Essentially, this is just forward-pass for every value in sequence.
        """
        inputs = np.atleast_2d(inputs)
        (_, count) = inputs.shape

        R = np.zeros((self.dim_res, count))
        outputs = np.zeros((self.dim_out, count))

        self.reset()

        for t in range(count):
            x = inputs[:, t]
            y, r = self.forward(x)
            R[:, t] = r
            outputs[:, t] = y

        return outputs, R

    def generate_seq(self, inputs, count):
        """
        Initialize network's state (context) using input sequence. Then, let the network generate
        next value of sequence, using its own previous prediction as an input.
        count: length of generated sequence after original inputs run out
        """
        inputs = np.atleast_2d(inputs)
        (_, init_count) = inputs.shape

        R = np.zeros((self.dim_res, init_count+count))
        outputs = np.zeros((self.dim_out, init_count+count))

        self.reset()

        for t in range(init_count):
            x = inputs[:, t]
            y, r = self.forward(x)
            R[:, t] = r
            outputs[:, t] = y

        for t in range(init_count, init_count+count):
            y, r = self.forward(y)
            R[:, t] = r
            outputs[:, t] = y

        return outputs, R
