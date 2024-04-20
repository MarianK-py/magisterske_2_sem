# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

import itertools
import numpy as np

from util import *

eps_hard_limit = 100



class Hopfield():

    def __init__(self, dim):
        self.dim  = dim
        self.beta = None  # if beta is None: deterministic run, else stochastic run


    def train(self, patterns):
        '''
        Compute weight matrix analytically
        '''
        W_temp = (patterns.T @ patterns)*(1/patterns.shape[1])# FIXME compute weights - "store" patterns in weight matrix
        D = np.diag([1]*self.dim)
        self.W = W_temp - W_temp*D

    def energy(self, s):
        '''
        Compute energy for given state
        '''
        return -((self.W@s)@s)/2  # FIXME


    # asynchronous dynamics
    def forward_one_neuron(self, s, neuron_index):
        '''
        Perform forward pass in asynchronous dynamics - compute output of selected neuron
        '''
        net = self.W @ s  # FIXME

        if self.beta is None:
            # Deterministic transition
            return np.sign(net[neuron_index])  # FIXME
        else:
            # Stochastic transition (Output for each neuron is either -1 or 1!)
            r = np.random.rand(1)
            prob = 1/(1 + np.exp(-self.beta*net[neuron_index]))      # FIXME
            out = (r < prob)*2 -1
            return out  # FIXME


    # synchronous dynamics (not implemented, not part of the exercise)
    def forward_all_neurons(self, s):
        '''
        Perform forward pass in synchronous dynamics - compute output of all neurons
        '''
        net = self.W @ s

        if self.beta is None:
            # Deterministic transition
            return np.sign(net)
        else:
            # Stochastic transition (Output for each neuron is either -1 or 1!)
            r = np.random.rand(self.dim)
            prob = 1/(1 + np.exp(-self.beta*net))      # FIXME
            out = (r < prob)*2 -1
            return out


    # not implemented properly, modify for correct functioning (not part of the exercise)
    def run_sync(self, x):
        '''
        Run model in synchronous dynamics. One input vector x will produce
        series of outputs (states) s_t.
        '''
        s = x.copy()
        e = self.energy(s)
        S = [s]
        E = [e]

        for t in range(eps_hard_limit): # "enless" loop
            ## Compute new state for all neurons
            s = self.forward_all_neurons(s)
            e = self.energy(s)

            S.append(s)
            E.append(e)

            ## Detect termination criterion
            # if [fixed point is reached] or [cycle is reached]:
            #     return S, E

        return S, E # if eps run out


    def run_async(self, x, eps=None, beta_s=None, beta_f=None, row=1, rows=1, trace=False):
        '''
        Run model in asynchronous dynamics. One input vector x will produce
        series of outputs (states) s_t.
        '''
        s = x.copy()
        e = self.energy(s)
        E = [e]

        title = 'Running: asynchronous {}'.format('stochastic' if beta_s is not None else 'deterministic')

        for ep in range(eps):
            ## Set beta for this episode
            if beta_s is None:
                # Deterministic -> no beta
                self.beta = None
                print('Ep {:2d}/{:2d}:  deterministic'.format(ep+1, eps))
            else:
                # Stochastic -> schedule for beta (or temperature)
                self.beta = beta_s * ( (beta_f/beta_s) ** (ep/(eps-1)))
                print('Ep {:2d}/{:2d}:  stochastic, beta = 1/T = {:7.4f}'.format(ep+1, eps, self.beta))

            ## Compute new state for each neuron individually
            for i in np.random.permutation(self.dim):
                s[i] = self.forward_one_neuron(s, neuron_index=i) # update state of selected neuron
                e = self.energy(s)
                E.append(e)

                # Plot
                if trace:
                    plot_state(s, energys=E, index=i, max_eps=eps*self.dim, row=row, rows=rows, title=title, block=False)
                    redraw()

            # Terminate deterministically when stuck in a local/global minimum (loops generally don't occur)
            if self.beta is None:
                if np.all(self.forward_all_neurons(s) == s):
                    print('Reached local/global minimum after {} episode{}, terminating.'.format(ep+1, 's' if ep > 0 else ''))
                    break

        # Plot
        if not trace:
            plot_state(s, energys=E, index=None, max_eps=eps*self.dim, row=row, rows=rows, title=title, block=True)

        print('Final state energy = {:.2f}'.format(self.energy(s)))
