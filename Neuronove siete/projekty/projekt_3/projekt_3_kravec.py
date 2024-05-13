#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np

# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

# Modifikoval: Marian Kravec

from esn import ESN
from util import *


# # Generate sequence data
length = 1100

# rand ~ Uni(0,1) -> 2*rand-1 ~ Uni(-1,1)
data = 2*np.random.rand(length)-1

# # Prepare inputs+targets
L = 40

palate_cleaner = data[:100]

train_inputs = data[100:600]
train_indexer = np.arange(L)[None, :] + np.arange(500)[:, None]
train_targets = data[100-L:600][train_indexer].T

test_inputs = data[600:]
test_indexer = np.arange(L)[None, :] + np.arange(500)[:, None]
test_targets = data[600-L:][test_indexer].T


# # Train model
n = 80
repeats = 10
step = 0.02
start = 0.01
all_TMC_mat = np.zeros((2, repeats, n))
avg_TMC_mat = np.zeros((2, n))

best_spec_rad = 0
best_TMC = 0
repeats_best = 20

### Spectral radius optimization:
for j in range(n):
    spec_rad = start + j*step
    avg_TMC = 0
    for k in range(repeats):
        model = ESN(dim_in=1, dim_res=100, dim_out=L, spectral_radius=spec_rad, sparsity=0.7)

        # Palate cleansing
        model.palate_cleaning(palate_cleaner)

        # Model training
        model.train(train_inputs, train_targets)

        # Model predicting
        outputs, R = model.one_step_predict_seq(test_inputs)

        # MC and TMC computations
        out_vars = np.var(outputs, axis=1)
        targ_vars = np.var(test_targets, axis=1)
        covars = np.cov(outputs, test_targets)
        TMCs = (covars[:L,L:2*L].diagonal()**2)/(out_vars*targ_vars)
        TMC = np.sum(TMCs)

        avg_TMC += TMC
        all_TMC_mat[:, k, j] = np.array([spec_rad, TMC])
    avg_TMC /= repeats
    avg_TMC_mat[:, j] = np.array([spec_rad, avg_TMC])

    if avg_TMC > best_TMC:
        best_TMC = avg_TMC
        best_spec_rad = spec_rad

avg_TMC_best = 0
all_TMC_best = np.zeros((repeats_best, L))

### Average MC for delay:
for i in range(repeats_best):
    model = ESN(dim_in=1, dim_res=100, dim_out=L, spectral_radius=best_spec_rad, sparsity=0.7)

    # Palate cleansing
    model.palate_cleaning(palate_cleaner)

    # Model training
    model.train(train_inputs, train_targets)

    # Model predicting
    outputs, R = model.one_step_predict_seq(test_inputs)

    # MC and TMC computations
    out_vars = np.var(outputs, axis=1)
    targ_vars = np.var(test_targets, axis=1)
    covars = np.cov(outputs, test_targets)
    TMCs = (covars[:L,L:2*L].diagonal()**2)/(out_vars*targ_vars)
    TMC = np.sum(TMCs)

    avg_TMC_best += TMC
    all_TMC_best[i, :] = np.flip(TMCs)

avg_TMC_best /= repeats_best

n_spars = 100
repeats_spars = 10
step_spars = 0.01
start_spars = 0
all_TMC_mat_spars = np.zeros((2, repeats_spars, n_spars))
avg_TMC_mat_spars = np.zeros((2, n_spars))

best_spars = 0
best_TMC_spars = 0

### Sparcity check:
for j in range(n_spars):
    sparsity = start_spars + j*step_spars
    avg_TMC = 0
    for k in range(repeats_spars):
        model = ESN(dim_in=1, dim_res=100, dim_out=L, spectral_radius=best_spec_rad, sparsity=sparsity)

        # Palate cleansing
        model.palate_cleaning(palate_cleaner)

        try:
            # Model training
            model.train(train_inputs, train_targets)
            # Model predicting
            outputs, R = model.one_step_predict_seq(test_inputs)
        except:
            print("Error because of sparsity")

        # MC and TMC computations
        out_vars = np.var(outputs, axis=1)
        targ_vars = np.var(test_targets, axis=1)
        covars = np.cov(outputs, test_targets)
        TMCs = (covars[:L,L:2*L].diagonal()**2)/(out_vars*targ_vars)
        TMC = np.sum(TMCs)

        avg_TMC += TMC
        all_TMC_mat_spars[:, k, j] = np.array([sparsity, TMC])
    avg_TMC /= repeats_spars
    avg_TMC_mat_spars[:, j] = np.array([sparsity, avg_TMC])

    if avg_TMC > best_TMC_spars:
        best_TMC_spars = avg_TMC
        best_spars = sparsity


fig1, ax1 = plt.subplots(1,1)
ax1.plot(avg_TMC_mat[0, :], avg_TMC_mat[1, :])
ax1.scatter(np.ndarray.flatten(all_TMC_mat[0, :, :]), np.ndarray.flatten(all_TMC_mat[1, :, :]), alpha=0.3, c="aqua")

#plt.savefig("L_"+str(L))


fig2, ax2 = plt.subplots(1,1, figsize=(20,3))
ax2.boxplot(all_TMC_best)

#plt.savefig("box_L_"+str(L))

fig3, ax3 = plt.subplots(1,1)
ax3.plot(avg_TMC_mat_spars[0, :], avg_TMC_mat_spars[1, :])
ax3.scatter(np.ndarray.flatten(all_TMC_mat_spars[0, :, :]), np.ndarray.flatten(all_TMC_mat_spars[1, :, :]), alpha=0.3, c="aqua")

#plt.savefig("spers_L_"+str(L))

plt.show()
