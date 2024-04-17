#!/usr/bin/python3
import matplotlib.pyplot as plt
# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

import numpy as np
import random

from som import *
from util import *

# Grid distance metric
# A metric takes two coordinates (vectors) and computes the distance


# Other possibilities of a metric:
# def L_1(i, j):
#     return 0

# def L_2(i, j):
#     return 0


# # Choose data - try all of them!

# 1) Skewed square
#inputs = np.random.rand(2, 250)
#inputs[1, :] += 0.5 * inputs[0, :]

# # 2) Circle
#inputs = 2*np.random.rand(2, 250) - 1
#inputs = inputs[:,np.abs(np.linalg.norm(inputs, axis=0)) < 1]

inputs = np.loadtxt('seeds.txt').T[:7]


(dim_in, count) = inputs.shape



# # Train model

# Choose size of grid
rows = 10
cols = 10

row_ind = np.array(list(range(rows)))
col_ind = np.array(list(range(cols)))

# Choose grid distance metric - L_1 / L_2 / L_max
#grid_metric = "L_1"
#grid_metric = "L_2"
grid_metric = "L_max"


# Some heuristics for choosing initial lambda
lambda_s = (rows-1 + cols-1) * 0.5

# Build and train model
model = SOM(dim_in, rows, cols, inputs)
quant_errs, adj_of_neurs = model.trainVectorized(inputs,
            eps=300,
            alpha_s=1, alpha_f=0.05, lambda_s=lambda_s-3, lambda_f=1,
            discrete_neighborhood=True,  # Use discrete or continuous (gaussian) neighborhood function
            grid_metric=grid_metric,
            live_plot=True, live_plot_interval=2  # Increase live_plot_interval for faster training
            )

plot_heatmap(model.weights, row_ind, col_ind, "Row", "Col", "parameter", save=False, name="heatmap_errors")

fig, ax = plt.subplots(2,1)

ax[0].plot(quant_errs)
ax[0].set_title("Quantization error during training")

ax[1].plot(adj_of_neurs)
ax[1].set_title("Average adjustment of neuron positions changes during training")

plt.show()