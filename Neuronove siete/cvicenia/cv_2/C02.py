#!/usr/bin/python3

# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

import numpy as np

from perceptron import *
from util import *

# Vypracoval: Marian Kravec

# Prepare data
data = np.loadtxt('linsep.dat')  # 'linsep.dat' / 'and.dat' / 'or.dat' / 'xor.dat'
#data = np.loadtxt('and.dat')
#data = np.loadtxt('or.dat')
#data = np.loadtxt('xor.dat')
inputs = data[:, :-1]
targets = data[:, -1].astype(int)
(count, dim) = inputs.shape

# # Plot input data
# plot_dots(inputs, targets)

model = Perceptron(dim)
errors = model.train(inputs, targets, alpha=0.1, eps=20, live_plot=True)

# # Plot decision border (use if live_plot was False)
# plot_decision(model.weights, inputs, targets)

# Plot error during training
plot_errors(errors)
