#!/usr/bin/python3
import numpy as np

# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

from classifier import *
from util import *
from time import perf_counter

# Vypracoval: Marian Kravec

# # Load data
with open('2d.trn.dat') as f:
    train_header = f.readline().strip().split()
    train_inputs = []
    train_labels = []
    for line in f:
        line = line.strip().split()
        train_inputs.append([float(line[0]), float(line[1])])
        train_labels.append(line[2])

with open('2d.tst.dat') as f:
    test_header = f.readline().strip().split()
    test_inputs = []
    test_labels = []
    for line in f:
        line = line.strip().split()
        test_inputs.append([float(line[0]), float(line[1])])
        test_labels.append(line[2])

train_inputs = np.array(train_inputs).T
train_labels = np.array(train_labels).T

test_inputs = np.array(test_inputs).T
test_labels = np.array(test_labels).T

lab_vals = np.array(list(set(test_labels)))

# Estimation, validation split
count = test_inputs.shape[1]
indices = np.array(list(range(0, count)))
np.random.shuffle(indices)
split = int(0.8*count)
estim_indices = indices[:split]
valid_indices = indices[split:]
# /suggestion

estim_inputs = train_inputs[:, estim_indices]
estim_labels = train_labels[estim_indices]

valid_inputs = train_inputs[:, valid_indices]
valid_labels = train_labels[valid_indices]

normalization_mean = np.mean(estim_inputs, axis=1, keepdims=True)
normalization_std = np.std(estim_inputs, axis=1, keepdims=True)


# # Normalize inputs
train_inputs -= normalization_mean
train_inputs /= normalization_std
estim_inputs -= normalization_mean
estim_inputs /= normalization_std
valid_inputs -= normalization_mean
valid_inputs /= normalization_std
test_inputs -= normalization_mean
test_inputs /= normalization_std

# # Train & visualize
# Build model
best_model = None
best_valid_error = float("inf")
best_training_errors = None
best_h1 = 0
best_h2 = 0

h1_vals = np.array(list(range(10, 26, 2)))
h2_vals = np.array(list(range(10, 26, 2)))
h1_n = h1_vals.shape[0]
h2_n = h2_vals.shape[0]
valid_errors = np.zeros([h2_n, h1_n])
train_errors = np.zeros([h2_n, h1_n])
times = np.zeros([h2_n, h1_n])
for i in range(h1_n):
    for j in range(h2_n):
        model = MLPClassifier(dim_in=estim_inputs.shape[0], dim_hid1=h1_vals[i], dim_hid2=h2_vals[j], dim_out=len(lab_vals), lab_vals=lab_vals)
        start = perf_counter()
        trainREs = model.trainBatch(estim_inputs, estim_labels,
                                    alpha=0.1, beta_1=0.9, beta_2=0.99, epsy=0.9995, eps=500,
                                    verbose=False, comp_test=False)
        stop = perf_counter()
        outputs_val, outputs_lab = model.predict(valid_inputs)
        valid_error = sum(model.error(onehot_encode(valid_labels, lab_vals), outputs_val))/valid_labels.shape[0]
        valid_errors[j, i] = valid_error
        train_errors[j, i] = trainREs["trainREs"][-1]
        times[j, i] = stop-start
        if valid_error < best_valid_error:
            best_model = model
            best_valid_error = valid_error
            best_training_errors = trainREs
            best_h1 = h1_vals[i]
            best_h2 = h2_vals[j]

print_best_layer_sizes(best_h1, best_h2, create_tex=True, tex_name="best_layer_sizes")
# Train model
final_model = MLPClassifier(dim_in=train_inputs.shape[0], dim_hid1=best_h1, dim_hid2=best_h2, dim_out=len(lab_vals), lab_vals=lab_vals)
trainREs = final_model.trainBatch(train_inputs, train_labels,
                                  alpha=0.1, beta_1=0.9, beta_2=0.99, eps=500,
                                  verbose=False, comp_test=True, test_inputs=test_inputs, test_labels=test_labels)

# "Test" model
outputs_val, outputs_lab = final_model.predict(test_inputs)

# Visualize
#plot_reg_density('Density', inputs, targets, outputs, block=False, plot_3D=I_WANT_3D_PLOTS)
#plot_errors('Model loss', best_training_errors[0], block=False)

plot_errors('Model train loss', [trainREs["trainREs"], trainREs["testREs"]], labels=["Training errors", "Testing errors"], save=True, name="errors")


plot_heatmap(valid_errors, h1_vals, h2_vals, "First hidden layer size", "Second hidden layer size", "Heatmap of errors", save=True, name="heatmap_errors")

plot_heatmap(times, h1_vals, h2_vals, "First hidden layer size", "Second hidden layer size", "Heatmap of times", save=True, name="heatmap_time")

plot_decision(final_model, test_inputs, test_labels, lab_vals=lab_vals, save=True, name="decision")

print(sum(final_model.error(onehot_encode(test_labels, lab_vals), outputs_val))/test_labels.shape[0])

confusion_table(final_model, test_inputs, test_labels, lab_vals, create_tex=True, tex_name="confusion_mat")

print_errors(train_errors, h1_vals, h2_vals, create_tex=True, tex_name="train_err")

print_errors(valid_errors, h1_vals, h2_vals, create_tex=True, tex_name="valid_err")

print_errors(times, h1_vals, h2_vals)

#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111, projection='3d')
#res_dots(ax2, model, inputs)

