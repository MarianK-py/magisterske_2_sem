#!/usr/bin/python3

# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

import numpy as np
import matplotlib.pyplot as plt
from util import *

# Vypracoval: Marian Kravec

## Generate data
# number of points
N = 100
sigma_q = 30
sigma_noise = 6

# generate x values
x = np.array(list(range(0, N)))   # FIXME: vector [0, 1, 2, ..., N-1]

# generate random slope (k) and shift (q)
k = np.random.randn(1)   # FIXME: random number from normal distribution
q = np.random.rand(1)*sigma_q  # FIXME: random number from uniform distribution

# calculate y values
y = x*k + q   # FIXME: according to slides

# add noise
y += np.random.randn(N)*sigma_noise  # FIXME: add vector of random values from normal distribution

# plot all points
plt.scatter(x, y)


## Find k and q using linear regression
# first append ones to x
X = np.c_[x, np.ones(N)]  # FIXME: X should be matrix of two columns: first column is vector x, second column are ones

# then find params
k_pred, q_pred = np.linalg.inv(X.T @ X) @ X.T @ y   # FIXME: according to slides

# predict ys and plot as a line
y_pred =  x*k_pred + q_pred  # FIXME: according to slides

# plot predicted
plt.plot(x, y_pred, 'r')
use_keypress(plt.gcf())   # Use ' ' or 'enter' to close figure; 'q' or 'escape' to quit program
plt.show()
