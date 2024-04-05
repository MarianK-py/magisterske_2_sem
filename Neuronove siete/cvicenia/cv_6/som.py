# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

import numpy as np

from util import *


class SOM:
    def __init__(self, dim_in, n_rows, n_cols, inputs=None):
        self.dim_in = dim_in
        self.n_rows = n_rows
        self.n_cols = n_cols

        self.weights = np.random.rand(n_rows, n_cols, dim_in)-0.5  # FIXME

        if inputs is not None:  # FIXME
            # "Fill" the input space with neurons - scale and shift neurons to inputs' distribution.
            # Note: SOM will train even without it, but it helps.
            mu = np.mean(inputs, axis=1)
            std = np.std(inputs, axis=1)
            print(mu, std)
            self.weights += mu
            self.weights *= np.sqrt(std)

            #pass

    def winner(self, x):
        '''
        Find winner neuron and return its coordinates in grid (i.e. its "index").
        Iterate over all neurons and find the neuron with the lowest distance to input x (np.linalg.norm).
        '''

        dist = np.linalg.norm(self.weights-x, axis=2)
        #print((self.weights-x).shape)
        #print(dist.shape)
        win = np.argmin(dist)
        win_r, win_c = win//self.n_cols, win%self.n_cols

        return win_r, win_c

    def neighbourhoodCont(self, i_1, i_2, j_1, j_2, lambda_t):
        return np.exp(-(self.LMaxdist(i_1, i_2, j_1, j_2)**2)/(lambda_t**2))

    def neighbourhoodDisc(self, i_1, i_2, j_1, j_2, lambda_t):
        return int(self.LMaxdist(i_1, i_2, j_1, j_2) < lambda_t)

    def L2dist(self, i_1, i_2, j_1, j_2):
        return np.sqrt(np.array([i_1, i_2])@np.array([j_1, j_2]))

    def LMaxdist(self, i_1, i_2, j_1, j_2):
        return max(abs(i_1-j_1), abs(i_2-j_2))

    def train(self,
              inputs,   # Matrix of inputs - each column is one input vector
              eps=100,  # Number of epochs
              alpha_s=0.01, alpha_f=0.001, lambda_s=None, lambda_f=1,  # Start & end values for alpha & lambda
              discrete_neighborhood=True,  # Use discrete or continuous (gaussian) neighborhood function
              grid_metric=(lambda u, v: 0),  # Grid distance metric
              live_plot=False, live_plot_interval=10  # Draw plots during training process
              ):

        (_, count) = inputs.shape
        plot_in3d = self.dim_in > 2

        if live_plot:
            interactive_on()
            (plot_grid_3d if plot_in3d else plot_grid_2d)(inputs, self.weights, block=False)
            redraw()

        for ep in range(eps):
            alpha_t = alpha_s*((alpha_f/alpha_s)**(ep/(eps-1)))  # FIXME
            lambda_t = lambda_s*((lambda_f/lambda_s)**(ep/(eps-1)))  # FIXME

            for idx in np.random.permutation(count):
                x = inputs[:, idx]

                win_r, win_c = self.winner(x)

                # Use "d = grid_metric(vector_a, vector_b)" for grid distance
                # Use discrete neighborhood

                for r in range(self.n_rows):
                    for c in range(self.n_cols):
                        # ...
                        self.weights[r, c] += alpha_t*(x-self.weights[r, c])*self.neighbourhoodDisc(r, c, win_r, win_c, lambda_t)  # FIXME

            print('Ep {:3d}/{:3d}:  alpha_t = {:.3f}, lambda_t = {:.3f}'
                  .format(ep+1, eps, alpha_t, lambda_t))

            if live_plot and ((ep+1) % live_plot_interval == 0):
                (plot_grid_3d if plot_in3d else plot_grid_2d)(inputs, self.weights, block=False)
                redraw()

        if live_plot:
            interactive_off()
        else:
            (plot_grid_3d if plot_in3d else plot_grid_2d)(inputs, self.weights, block=True)
