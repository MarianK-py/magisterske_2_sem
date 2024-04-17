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

data = np.loadtxt('seeds.txt')
inputs = data.T[:7]
labels = data.T[7]

count = inputs.shape[1]
indices = np.array(list(range(0, count)))
np.random.shuffle(indices)
train_indices = indices[:150]
test_indices = indices[150:]

train_inputs = inputs[:, train_indices]
train_labels = labels[train_indices]

test_inputs = inputs[:, test_indices]
test_labels = labels[test_indices]



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
#grid_metric = "L_max"

grid_matrics = ["L_1", "L_2", "L_max"]
is_discrete = [True, False]
epochs = 200

best_alpha_s = 0
best_err = float("inf")
best_grid_metric = ""
best_is_disc = None

for alpha_s in [0.5, 0.7, 1, 2, 5, 10]:
    for grid_metric in grid_matrics:
        for is_disc in is_discrete:
            row_ind = np.array(list(range(rows)))
            col_ind = np.array(list(range(cols)))

            # Some heuristics for choosing initial lambda
            lambda_s = (rows + cols) * 0.5

            # Build and train model
            model = SOM(dim_in, rows, cols, inputs)
            quant_errs, adj_of_neurs = model.trainVectorized(train_inputs,
                            eps=epochs,
                            alpha_s=alpha_s, alpha_f=0.01, lambda_s=lambda_s, lambda_f=1,
                            discrete_neighborhood=is_disc,  # Use discrete or continuous (gaussian) neighborhood function
                            grid_metric=grid_metric,
                            live_plot=False, live_plot_interval=20  # Increase live_plot_interval for faster training
                            )
            if quant_errs[-1] < best_err:
                best_alpha_s = alpha_s
                best_err = quant_errs[-1]
                best_grid_metric = grid_metric
                best_is_disc = is_disc

lambda_s = (rows + cols) * 0.5


model = SOM(dim_in, rows, cols, inputs)
quant_errs, adj_of_neurs = model.trainVectorized(train_inputs,
                                eps=1000,
                                alpha_s=best_alpha_s, alpha_f=0.01, lambda_s=lambda_s, lambda_f=1,
                                discrete_neighborhood=best_is_disc,  # Use discrete or continuous (gaussian) neighborhood function
                                grid_metric=best_grid_metric,
                                live_plot=True, live_plot_interval=20  # Increase live_plot_interval for faster training
                                )

print("Best alpha s:", best_alpha_s)
print("Best grid_metric:", best_grid_metric)
print("Best is discrete:", best_is_disc)

plot_heatmap(model.weights, row_ind, col_ind, "Row", "Col", "parameter", save=True, name="param_heatmaps")

fig, ax = plt.subplots(2,1,figsize=(8,12))

ax[0].plot(quant_errs)
ax[0].set_title("Quantization error during training")

ax[1].plot(adj_of_neurs)
ax[1].set_title("Average adjustment of neuron positions changes during training")

plt.savefig("errors")
plt.show()

plt.show()

mapa = np.empty([rows, cols], dtype=object)
for row in range(rows):
    for col in range(cols):
        mapa[row, col] = []

for t in range(train_inputs.shape[1]):
    winner_r, winner_c = model.winner(train_inputs[:,t])
    mapa[winner_r, winner_c].append(train_labels[t])

sizes = np.zeros([rows, cols])
win_lab = np.zeros([rows, cols])

for row in range(rows):
    for col in range(cols):
        sizes[row,col] = len(mapa[row,col])

        label_list = mapa[row][col]
        if len(label_list)==0:
            label = 0
        else:
            label = max(label_list, key=label_list.count)
        win_lab[row][col] = label

plot_heatmap(win_lab, row_ind, col_ind, "Row", "Col", "Classes of neurons", save=True, name="class_neur")
#plot_heatmap(sizes, row_ind, col_ind, "Row", "Col", "parameter", save=False, name="heatmap_errors")

pred_labs = np.zeros(test_labels.shape[0])

for t in range(test_inputs.shape[1]):
    winner_r, winner_c = model.winner(test_inputs[:,t])
    pred_labs[t] = win_lab[winner_r, winner_c]

print("Clustering accuracy:",sum(test_labels==pred_labs)/test_labels.shape[0])
