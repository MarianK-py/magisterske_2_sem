#!/usr/bin/python3

# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

from esn import ESN
from util import *


# # Generate frequency data
inputs, targets = frequency_data(freqs=[2, 3, 1, 4], slice_length=50, sampling_rate=2)
plot_sequence(targets, inputs=inputs, block=True)


# # Train model
# FIXME: Tune parameters to make the network perform better. Do not increase number of reservoir neurons.
model = ESN(dim_in=1, dim_res=100, dim_out=1, spectral_radius=0.615, sparsity=0.65)
model.train(inputs, targets)


# # Test model
outputs, R = model.one_step_predict_seq(inputs)
plot_cells(R, rows=10, cols=10)
plot_sequence(targets, outputs=outputs.flatten(), inputs=inputs)
