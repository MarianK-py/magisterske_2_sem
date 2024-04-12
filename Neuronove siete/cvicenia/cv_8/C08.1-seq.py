#!/usr/bin/python3

# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

from esn import ESN
from util import *


# # Generate sequence data
length = 800
train_test_ratio = 0.25
X = np.linspace(0, 2*np.pi, num=length) / train_test_ratio

# # Choose from following datasets:
# data = 0*X
# data = 0*X + 4
# data = 0.5*X
# data = 50.0*X
# data = X**2
# data = (2*X-4)**2

data = np.sin(X*3)
# data = np.sin(X*3) + 5
# data = np.sin(X) * np.sin(0.5*X)
# data = np.sin(X*8) + 0.4 * np.cos(X*2)
# data = np.sin(3*X)**3
# data = np.sin(X) * np.sin(2*X) # last that should definitely work
# data = np.sin(X) * np.sin(3*X)
# data = np.sin(X*3) * X
# data = np.sin(X**2*2)
# data = np.sin(X**2*9) + np.cos(X*3)
# data = np.tan(X*3)


# # Prepare inputs+targets
step = 1

split = int(train_test_ratio * length)
train_data = data[:split]

train_inputs = train_data[:-step]
train_targets = train_data[step:]

full_inputs = data[:-step]
full_targets = data[step:]

# plot_sequence(full_inputs, full_targets, split=split)


# # Train model
# FIXME: Tune parameters to make the network perform better. Do not increase number of reservoir neurons.
model = ESN(dim_in=1, dim_res=20, dim_out=1, spectral_radius=0.635, sparsity=0.8)
model.train(train_inputs, train_targets)


# # Test model
# a) one-step prediction of next input
outputs, R = model.one_step_predict_seq(full_inputs)
plot_cells(R, split=split)
plot_sequence(full_targets, outputs=outputs.flatten(), split=split)

# b) repeated one-step generation
outputs, R = model.generate_seq(inputs=train_inputs, count=length-split)
plot_cells(R, split=split)
plot_sequence(full_targets, outputs=outputs.flatten(), split=split, title='Sequence generation')
