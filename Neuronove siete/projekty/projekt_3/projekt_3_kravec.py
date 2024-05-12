#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np

# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

from esn import ESN
from util import *


# # Generate sequence data
length = 1100

# rand ~ Uni(0,1) -> 2*rand-1 ~ Uni(-1,1)
X = 2*np.random.rand(length)-1



data = np.sin(X*3)


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

for j in range(n):
    spec_rad = start + j*step
    avg_TMC = 0
    for k in range(repeats):
        model = ESN(dim_in=1, dim_res=100, dim_out=L, spectral_radius=spec_rad, sparsity=0.7)

        # Palate cleansing
        model.palate_cleaning(palate_cleaner)

        # Model training
        model.train(train_inputs, train_targets)

        # # Test model
        # a) one-step prediction of next input
        outputs, R = model.one_step_predict_seq(test_inputs)

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

for i in range(repeats_best):
    model = ESN(dim_in=1, dim_res=100, dim_out=L, spectral_radius=best_spec_rad, sparsity=0.7)

    # Palate cleansing
    model.palate_cleaning(palate_cleaner)

    # Model training
    model.train(train_inputs, train_targets)

    outputs, R = model.one_step_predict_seq(test_inputs)

    out_vars = np.var(outputs, axis=1)
    targ_vars = np.var(test_targets, axis=1)
    covars = np.cov(outputs, test_targets)
    TMCs = (covars[:L,L:2*L].diagonal()**2)/(out_vars*targ_vars)
    TMC = np.sum(TMCs)
    avg_TMC_best += TMC
    all_TMC_best[i, :] = np.flip(TMCs)

avg_TMC_best /= repeats_best

fig1, ax1 = plt.subplots(1,1)
ax1.plot(avg_TMC_mat[0, :], avg_TMC_mat[1, :])
ax1.scatter(np.ndarray.flatten(all_TMC_mat[0, :, :]), np.ndarray.flatten(all_TMC_mat[1, :, :]), alpha=0.3, c="aqua")

#plt.savefig("L_"+str(L))


fig2, ax2 = plt.subplots(1,1)
ax2.boxplot(all_TMC_best)
plt.show()

# b) repeated one-step generation
#outputs, R = model.generate_seq(inputs=train_inputs, count=500)
#plot_cells(R, split=500)
#plot_sequence(full_targets, outputs=outputs.flatten(), split=500, title='Sequence generation')
