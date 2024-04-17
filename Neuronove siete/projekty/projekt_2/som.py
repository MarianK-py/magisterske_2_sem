# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

import numpy as np

from util import *


class SOM:
    def __init__(self, dim_in, n_rows, n_cols, inputs=None):
        self.dim_in = dim_in
        self.n_rows = n_rows
        self.n_cols = n_cols

        self.weights = np.random.rand(n_rows, n_cols, dim_in)  # FIXME

        if inputs is not None:  # FIXME
            # "Fill" the input space with neurons - scale and shift neurons to inputs' distribution.
            # Note: SOM will train even without it, but it helps.
            mini = np.min(inputs, axis=1)
            size = np.max(inputs, axis=1)-mini
            self.weights *= size
            self.weights += mini

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

    def neighbourhoodCont(self, i_1, i_2, j_1, j_2, lambda_t, grid_metric):
        return np.exp(-(grid_metric(i_1, i_2, j_1, j_2)**2)/(lambda_t**2))

    def neighbourhoodDisc(self, i_1, i_2, j_1, j_2, lambda_t, grid_metric):
        return int(grid_metric(i_1, i_2, j_1, j_2) < lambda_t)

    def neighbourhoodContVect(self, dist_m, lambda_t):
        return np.exp(-(dist_m)**2)/(lambda_t**2)
    def neighbourhoodDiscVect(self, dist_m, lambda_t):
        return dist_m < lambda_t

    def L1dist(self, i_1, i_2, j_1, j_2):
        return abs(i_1-j_1) + abs(i_2-j_2)
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

        if grid_metric == "L_1":
            metr = self.L1dist
        elif grid_metric == "L_2":
            metr = self.L2dist
        elif grid_metric == "L_max":
            metr = self.LMaxdist
        else:
            print("Unknown grid metric")

        if live_plot:
            interactive_on()
            (plot_grid_3d if plot_in3d else plot_grid_2d)(inputs, self.weights, block=False)
            redraw()

        for ep in range(eps):
            alpha_t = alpha_s*((alpha_f/alpha_s)**(ep/(eps-1)))
            lambda_t = lambda_s*((lambda_f/lambda_s)**(ep/(eps-1)))
            quant_err = 0
            for idx in np.random.permutation(count):
                x = inputs[:, idx]

                win_r, win_c = self.winner(x)

                quant_err += np.linalg.norm(x - self.weights[win_r, win_c])

                for r in range(self.n_rows):
                    for c in range(self.n_cols):
                        if discrete_neighborhood:
                            self.weights[r, c] += alpha_t*(x-self.weights[r, c])*self.neighbourhoodDisc(r, c, win_r, win_c, lambda_t, metr)
                        else:
                            self.weights[r, c] += alpha_t*(x-self.weights[r, c])*self.neighbourhoodCont(r, c, win_r, win_c, lambda_t, metr)
            quant_err /= count

            print('Ep {:3d}/{:3d}:  alpha_t = {:.3f}, lambda_t = {:.3f}, quantization_error = {:.3f}'
                  .format(ep+1, eps, alpha_t, lambda_t, quant_err))

            if live_plot and ((ep+1) % live_plot_interval == 0):
                (plot_grid_3d if plot_in3d else plot_grid_2d)(inputs, self.weights, block=False)
                redraw()

        if live_plot:
            interactive_off()
        else:
            (plot_grid_3d if plot_in3d else plot_grid_2d)(inputs, self.weights, block=True)

    def trainVectorized(self,
              inputs,   # Matrix of inputs - each column is one input vector
              eps=100,  # Number of epochs
              alpha_s=0.01, alpha_f=0.001, lambda_s=None, lambda_f=1,  # Start & end values for alpha & lambda
              discrete_neighborhood=True,  # Use discrete or continuous (gaussian) neighborhood function
              grid_metric=(lambda u, v: 0),  # Grid distance metric
              live_plot=False, live_plot_interval=10  # Draw plots during training process
              ):

        (_, count) = inputs.shape
        plot_in3d = self.dim_in > 2

        dummy_grid = cartesian_product(np.array(list(range(self.n_rows))),
                                       np.array(list(range(self.n_cols))))

        quant_errs = []
        adj_of_neurs = []

        if live_plot:
            interactive_on()
            (plot_grid_3d if plot_in3d else plot_grid_2d)(inputs, self.weights, block=False)
            redraw()

        for ep in range(eps):
            alpha_t = alpha_s*((alpha_f/alpha_s)**(ep/(eps-1)))
            lambda_t = lambda_s*((lambda_f/lambda_s)**(ep/(eps-1)))
            quant_err = 0
            for idx in np.random.permutation(count):
                x = inputs[:, idx]

                win_r, win_c = self.winner(x)

                win_pos = np.array([win_r, win_c])

                if grid_metric == "L_1":
                    dist_matrix = np.sum(np.abs(dummy_grid-win_pos), axis=1)
                elif grid_metric == "L_2":
                    dist_matrix = np.sqrt(dummy_grid@win_pos)
                elif grid_metric == "L_max":
                    dist_matrix = np.max(np.abs(dummy_grid-win_pos), axis=1)

                dist_matrix = dist_matrix.reshape((self.n_rows, self.n_cols))

                dist_matrix = np.repeat(dist_matrix[:, :, np.newaxis], self.dim_in, axis=2)

                quant_err += np.linalg.norm(x - self.weights[win_r, win_c])

                if discrete_neighborhood:
                    dweights = alpha_t*(x-self.weights)*self.neighbourhoodDiscVect(dist_matrix, lambda_t)
                else:
                    dweights = alpha_t*(x-self.weights)*self.neighbourhoodContVect(dist_matrix, lambda_t)

                adj_of_neur = (1/(self.n_cols*self.n_cols))*np.linalg.norm(dweights)

                self.weights += dweights
            quant_err /= count
            
            if ((ep+1) % live_plot_interval == 0):
                print('Ep {:3d}/{:3d}:  alpha_t = {:.3f}, lambda_t = {:.3f}, quantization_error = {:.3f}, avg_adj_of_neurons = {:.3f}'
                     .format(ep+1, eps, alpha_t, lambda_t, quant_err, adj_of_neur))

            quant_errs.append(quant_err)
            adj_of_neurs.append(adj_of_neur)

            if live_plot and ((ep+1) % live_plot_interval == 0):
                (plot_grid_3d if plot_in3d else plot_grid_2d)(inputs, self.weights, block=False)
                redraw()

        if live_plot:
            interactive_off()
        else:
            pass
            #(plot_grid_3d if plot_in3d else plot_grid_2d)(inputs, self.weights, block=True)

        return np.array(quant_errs), np.array(adj_of_neurs)