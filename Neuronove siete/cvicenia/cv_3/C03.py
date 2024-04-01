#!/usr/bin/python3

# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Juraj Holas, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2020-2024

from perceptron import *
from util import *

# Vypracoval: Marian Kravec

# # Load data and visualize
file_path = 'odd_even.in' # Distinguish between odd and even numbers
# file_path = 'numbers.in'  # Recognize digits

inputs, targets = prepare_data(file_path, noise=0.2)
dim_in = inputs.shape[0]
dim_out = targets.shape[0]


# # 1) Train sequentially
# Build model
model_s = Perceptron(dim_in, dim_out)

# Train model
print('Training sequentially:')
training_accuracy = model_s.train_seq(inputs, targets, eps=100, alpha=0.05)

# Print results
plot_accuracy(training_accuracy, title='Accuracy - Sequential training')
plot_original_inputs(model=model_s)
plot_noisy_inputs(model=model_s, count=18)

# # 2) Train batch
# Build model
model_b = Perceptron(dim_in, dim_out)

# Train model
print('\nTraining batch:')
training_accuracy = model_b.train_batch(inputs, targets, eps=100, alpha=0.05)

# Print results
plot_accuracy(training_accuracy, title='Accuracy - Batch training')


# # 3) Compare speeds
# Build models
model_s = Perceptron(dim_in, dim_out)
model_b = Perceptron(dim_in, dim_out)

# Train models and compare speeds
print('\nCompare speeds')
timeit(model_s.train_seq)(inputs, targets, eps=500, alpha=0.005, compute_accuracy=False)
timeit(model_b.train_batch)(inputs, targets, eps=500, alpha=0.005, compute_accuracy=False)
