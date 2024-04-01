# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Juraj Holas, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2020-2024

import matplotlib
matplotlib.use('TkAgg')  # fixme if plotting doesn`t work (try 'Qt5Agg' or 'Qt4Agg')
import matplotlib.pyplot as plt

import numpy as np
import atexit
import os
import time
import functools


def vector(array, row_vector=False):
    """
    Constructs a column vector (i.e. matrix of shape (n,1)) from given array/numpy.ndarray, or a row vector
    (shape (1,n)) if row_vector = True.
    """
    v = np.array(array)
    if np.squeeze(v).ndim > 1:
        raise ValueError('Cannot construct vector from array of shape {}!'.format(v.shape))
    return v.reshape((1, -1) if row_vector else (-1, 1))


def add_bias(inp):
    """
    Add bias term to the vector v, or to every (column) vector in a matrix.
    """
    if inp.ndim == 1:
        return np.concatenate((inp, [1]))
    else:
        pad = np.ones((1, inp.shape[1]))
        return np.concatenate((inp, pad), axis=0)


def timeit(func):
    """
    Profiling function to measure time it takes to finish function.
    Args:
        func(*function): Function to measure
    Returns:
        (*function) New wrapped function with measurement
    """
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        start_time = time.time()
        out = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print('Function [{}] finished in {:.3f} s'.format(func.__name__, elapsed_time))
        return out
    return newfunc


# # Interactive drawing
def clear():
    plt.clf()


def interactive_on():
    plt.ion()
    plt.show(block=False)
    time.sleep(0.1)


def interactive_off():
    plt.ioff()
    plt.close()


def redraw():
    # plt.gcf().canvas.draw()   # fixme: uncomment if interactive drawing does not work
    plt.waitforbuttonpress(timeout=0.001)
    time.sleep(0.001)


def keypress(e):
    if e.key in {'q', 'escape'}:
        os._exit(0)  # unclean exit, but exit() or sys.exit() won't work
    if e.key in {' ', 'enter'}:
        plt.close()  # skip blocking figures


def use_keypress(fig=None):
    if fig is None:
        fig = plt.gcf()
    fig.canvas.mpl_connect('key_press_event', keypress)


# Non-blocking figures still block at end
def finish():
    plt.show(block=True)  # block until all figures are closed


atexit.register(finish)


# # Data preparation
clean_data = {'inputs': None, 'targets': None}
noisy_data = {'inputs': None, 'targets': None}
img_shape = (0, 0)


def prepare_data(file_path, num_copy=10, noise=0.02):
    global clean_data, noisy_data, img_shape
    inp = []
    trg = []
    with open(file_path, 'r') as f:
        r, c, o = [int(x) for x in f.readline().split()]  # rows, cols, output_dim
        img_shape = (r, c)
        # Load inputs and targets
        lines = f.read().splitlines()
        for i in range(0, len(lines), r+1):
            inp.append(np.array([float(a) for line in lines[i:(i+r)] for a in line]))
            trg.append(np.array([float(a) for a in lines[i+r]]))
    # Make numpy arrays
    inp = np.array(inp).T
    trg = np.array(trg).T
    # Generate noisy inputs
    n_trg = np.tile(trg, (1, num_copy))
    n_inp = np.tile(inp, (1, num_copy))
    mask = np.random.rand(*n_inp.shape) < noise
    n_inp[mask] = 1 - n_inp[mask]
    # Save (for plotting)
    clean_data = {'inputs': inp, 'targets': trg}
    noisy_data = {'inputs': n_inp, 'targets': n_trg}
    return n_inp, n_trg


# # Plotting
def plot_images(inputs, targets, model=None, title=None, block=False):
    # Plot all inputs in data, each in subplot
    count = inputs.shape[1]
    nrows = int(np.round(np.sqrt(count/2)))
    ncols = int((count-1) // nrows + 1)
    fig, _ = plt.subplots(nrows, ncols, num=title)
    fig.canvas.mpl_connect('key_press_event', keypress)
    for i in range(count):
        x = inputs[:, i]
        d = targets[:, i]
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(x.reshape(img_shape), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.axis('off')
        dd = d.argmax()
        if model:
            y = model.compute_output(add_bias(x)).argmax()
            plt.title('Target: {}\nNN output: {}'.format(dd, y), fontsize=11)
        else:
            plt.title('Target: {}'.format(dd), fontsize=11)
    i = count
    while i < nrows * ncols:
        plt.subplot(nrows, ncols, i+1)
        plt.axis('off')
        i += 1
    plt.tight_layout()
    plt.show(block=block)


def plot_original_inputs(**kwargs):
    # Plot the 10 original inputs, without noise
    plot_images(**clean_data, title='Original inputs', **kwargs)


def plot_noisy_inputs(count=10, **kwargs):
    # Plot few noisy inputs
    n_inputs = noisy_data['inputs'].shape[1]
    idx = np.random.choice(n_inputs, count)
    plot_images(noisy_data['inputs'][:, idx], noisy_data['targets'][:, idx], title='Noisy inputs', **kwargs)


def plot_accuracy(accuracy_hisotry, title='Training accuracy', block=False):
    # Plot history of accuracy
    plt.figure(num=title)
    use_keypress()
    plt.title('Classification accuracy per epoch [%]')
    plt.plot(np.array(accuracy_hisotry)*100, '-b')
    plt.grid(True)
    plt.xlim(left=-1)
    plt.ylim(-3, 103)
    plt.tight_layout()
    plt.show(block=block)
