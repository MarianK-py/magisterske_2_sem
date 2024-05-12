#!/usr/bin/python3

# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017 - 2024

import numpy as np

from rbm import RBM
from util import *


## Prepare MNIST dataset
# Data parameters
classes      = [1, 2, 3]  # only use some classes (0..9)
n_examples   = 10     # select _ examples from each class
util_setup(w=28, h=28) # set up image dimensions for plotting

# Data preparation
raw_inputs, raw_labels = load_mnist()
inputs, labels = balanced_subset(raw_inputs, raw_labels, n_examples, classes=classes)
inputs = append_classes_to_data(inputs, labels)

img_dim = raw_inputs.shape[1]
dim_in = inputs.shape[1]

plot_images(inputs, rows=len(classes), fig=2, block=True)


## Train the model
print('\nTraining...')

dim_hid = 20

model = RBM(dim_vis=dim_in, dim_hid=dim_hid, inputs=inputs.T)
model.train(inputs.T,
            eps=1000,
            alpha=0.02,
            gibbs_rounds=5, # produce CD's v_neg by _ rounds of Gibbs sampling
            use_probs=True  # use probabilities, not samples in CD's v_neg estimation
            )


## Generate artificial data
print('\nGenerating new data...')

test_rows  = 5      # generate _ rows of images
test_cols  = 8      # generate _ cols of images
test_gibbs = 20     # generate images by _ rounds of gibbs sampling (from a random hidden state)

generated = np.array([model.generate(gibbs_rounds=test_gibbs) for _ in range(test_rows * test_cols)])
images = generated[:][:img_dim]
labels = [label_to_title(generated[i][img_dim:]) for i in range(test_rows * test_cols)]

plot_images(images, labels, rows=test_rows)
