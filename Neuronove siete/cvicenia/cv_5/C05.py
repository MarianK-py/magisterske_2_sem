#!/usr/bin/python3

# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Stefan Pócoš, Iveta Bečková 2017-2024

import random

from classifier import *
from util import *

# Vypracoval: Marian Kravec

# # Load data
data = np.loadtxt('iris.dat').T

inputs = data[:-1]
labels = data[-1].astype(int) - 1  # last column is class, starts from 0

(dim, count) = inputs.shape


# # Split training & test set
# Randomly select 80% of data points (with their labels) that will be used for training,
# remaining 20% will be used for testing

# following is a mere suggestion of how you can achieve it, feel free to implement your own way:
indices = np.array(list(range(0, count)))        # FIXME
random.shuffle(indices)
split = int(0.8*count)         # FIXME
train_indices = indices[:split]  # FIXME
test_indices = indices[split:]   # FIXME
# /suggestion

train_inputs = inputs[:, train_indices]  # FIXME
train_labels = labels[train_indices]  # FIXME

test_inputs = inputs[:, test_indices]   # FIXME
test_labels = labels[test_indices]   # FIXME

# Plot data before training
plot_dots(train_inputs, train_labels, None, test_inputs, test_labels, None)


# # Train & test model
# Build model
model = MLPClassifier(dim_in=dim, dim_hid=20, n_classes=np.max(labels)+1)
# Train model on training data
trainCEs, trainREs = model.train(train_inputs, train_labels, alpha=0.1, eps=500, live_plot=False, live_plot_interval=25)

# Test model on testing data
testCE, testRE = model.test(test_inputs, test_labels)
print('Final testing error: CE = {:6.2%}, RE = {:.5f}'.format(testCE, testRE))

# # Plot results
_, train_predicted = model.predict(train_inputs)
_, test_predicted = model.predict(test_inputs)
# Choose which graphs you want to see
plot_dots(train_inputs, train_labels, train_predicted, test_inputs, test_labels, test_predicted, block=False)
plot_dots(None, None, None, test_inputs, test_labels, test_predicted, title='Test data only', block=False)
plot_both_errors(trainCEs, trainREs, testCE, testRE, block=False)
