#!/usr/bin/python3

# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

import numpy as np

from elman import *
from util import *


# # Generate sequence data

length = 200
data = mackey_glass(length)

train_test_ratio = 0.75
split = int(train_test_ratio*len(data))
train_data = data[:split]

# plot_sequence(data, split=split)


# # Prepare inputs+targets
step_size = 1

train_inputs = train_data[:-step_size]
train_targets = train_data[step_size:]

full_inputs = data[:-step_size]
full_targets = data[step_size:]



# # Train model

model = ElmanNetwork(dim_in=1, dim_hid=200, dim_out=1)
model.train(inputs=np.atleast_2d(train_inputs), targets=np.atleast_2d(train_targets), alpha=0.1, eps=100)


# # Test model

# a) one-step prediction of next input
outputs = model.one_step_predict_seq(inputs=np.atleast_2d(full_inputs))
plot_sequence(full_targets, outputs, split=split, title='One-step prediction')
