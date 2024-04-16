#!/usr/bin/python3

# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

import numpy as np
import random

from som import *
from util import *

# Grid distance metric
# A metric takes two coordinates (vectors) and computes the distance


def L_max(i, j, axis=0):
    return np.max(np.abs(i - j), axis=axis)

# Other possibilities of a metric:
# def L_1(i, j):
#     return 0

# def L_2(i, j):
#     return 0


# # Choose data - try all of them!

# 1) Skewed square
inputs = np.random.rand(2, 250)
inputs[1, :] += 0.5 * inputs[0, :]

# # 2) Circle
# inputs = 2*np.random.rand(2, 250) - 1
# inputs = inputs[:,np.abs(np.linalg.norm(inputs, axis=0)) < 1]

# # 3) Truncated ellipse
# inputs = np.loadtxt('ellipse.dat').T[:2]

# # 4a) first two features of iris
# inputs = np.loadtxt('iris.dat').T[:2]

# # 4b) first three features of iris
# inputs = np.loadtxt('iris.dat').T[:3]

# # 4c) all features of iris
# inputs = np.loadtxt('iris.dat').T

(dim_in, count) = inputs.shape



# # Train model

# Choose size of grid
rows = 13
cols = 17

# Choose grid distance metric - L_1 / L_2 / L_max
grid_metric = L_max

# Some heuristics for choosing initial lambda
top_left = np.array((0, 0))
bottom_right = np.array((rows-1, cols-1))
lambda_s = grid_metric(top_left, bottom_right) * 0.5

# Build and train model
model = SOM(dim_in, rows, cols, inputs)
model.train(inputs,
            eps=100,
            alpha_s=0.2, alpha_f=0.005, lambda_s=lambda_s, lambda_f=1,
            discrete_neighborhood=True,  # Use discrete or continuous (gaussian) neighborhood function
            grid_metric=grid_metric,
            live_plot=True, live_plot_interval=2  # Increase live_plot_interval for faster training
            )
