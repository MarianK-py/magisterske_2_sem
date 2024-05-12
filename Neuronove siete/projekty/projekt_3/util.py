# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

import matplotlib
matplotlib.use('TkAgg')  # fixme if plotting doesn`t work (try 'Qt5Agg' or 'Qt4Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import atexit
import os
import time
import functools


# # Utilities
def mackey_glass(n, a=0.2, b=0.8, c=0.9, d=23, e=10, initial=0.1):
    """
    Mackey-Glass sequence (discrete equation)
    """
    x = np.zeros(n)
    x[0] = initial
    d = int(d)
    for k in range(0, n-1):
        x[k+1] = c*x[k] + ((a*x[k-d]) / (b + (x[k-d]**e)))
    return x


def frequency_data(freqs, slice_length=100, sampling_rate=1.0):
    """
    Convert list of frequencies to signal of sine waves of given frequencies.
    slice_length: number of samples for each frequency
    sampling_rate: period between samples
    """
    freqs = np.asarray(freqs, dtype='float64')
    inputs = np.repeat(freqs, slice_length)

    T = np.arange(len(inputs)) * sampling_rate
    targets = np.zeros(len(inputs))
    alpha = 0
    for i, t in enumerate(T):
        alpha += inputs[i] * sampling_rate / (2*np.pi)
        targets[i] = np.sin(alpha)

    inputs -= np.mean(inputs)
    return inputs, targets


def vector(array, row_vector=False):
    """
    Constructs a column vector (i.e. matrix of shape (n,1)) from given array/numpy.ndarray, or row
    vector (shape (1,n)) if row_vector = True.
    """
    v = np.atleast_1d(np.array(array))
    if np.squeeze(v).ndim > 1:
        raise ValueError('Cannot construct vector from array of shape {}!'.format(v.shape))
    return v.reshape((1, -1) if row_vector else (-1, 1))


def add_bias(X):
    """
    Add bias term to vector, or to every (column) vector in a matrix.
    """
    if X.ndim == 1:
        return np.concatenate((X, [1]))
    else:
        pad = np.ones((1, X.shape[1]))
        return np.concatenate((X, pad), axis=0)


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


# # Non-blocking figures still block at end
def finish():
    plt.show(block=True)  # block until all figures are closed


atexit.register(finish)


# # Plotting
palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']


def limits(values, gap=0.05, old_limits=None):
    if old_limits is None:
        old_limits = np.array((0, 0))
    x0 = np.minimum(np.min(values), old_limits[0])
    x1 = np.maximum(np.max(values), old_limits[1])
    xg = (x1 - x0) * gap
    return np.array((x0-xg, x1+xg))


def plot_sequence(targets, outputs=None, inputs=None, split=None, title=None, block=True):
    plt.figure()
    use_keypress()

    lim = np.array((0, 0))
    if inputs is not None:
        plt.plot(inputs, lw=5, alpha=0.3, label='Inputs')
        lim = limits(inputs, old_limits=lim)

    if inputs is None and outputs is not None:
        plt.plot(targets, lw=5, alpha=0.3, label='Targets')
    else:
        plt.plot(targets, label='Targets')
    lim = limits(targets, old_limits=lim)

    if outputs is not None:
        plt.plot(outputs, '-r', label='Outputs')
        lim = limits(outputs.flat, old_limits=lim)

    if split is not None:
        plt.vlines([split], ymin=lim[0], ymax=lim[1], color=palette[-1], lw=1)

    plt.ylim(lim)
    plt.legend()
    plt.tight_layout()
    plt.gcf().canvas.manager.set_window_title(title or ('Prediction' if outputs is not None else 'Sequence'))
    plt.show(block=block)


def plot_cells(cells, split=None, rows=4, cols=5, block=False):
    plt.figure()
    use_keypress()

    lim = limits(cells, gap=0)
    ylim = limits(cells)
    xlim = limits((0, cells.shape[1]))

    i = 0
    for r in range(rows):
        for c in range(cols):
            if i < cells.shape[0]:
                plt.subplot(rows, cols, i+1)

                plt.plot(cells[i])

                if split is not None:
                    plt.vlines([split], ymin=ylim[0], ymax=ylim[1], color=palette[-1], lw=1)

                plt.hlines([0], xmin=xlim[0], xmax=xlim[1], color=palette[-1], lw=1)

                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.xticks([])
                plt.yticks([])
                i += 1

    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.05, hspace=0.05)
    # plt.tight_layout()
    plt.gcf().canvas.manager.set_window_title('Reservoir states [{:.3f} ≤ y ≤{:.3f}]'.format(*lim))
    plt.show(block=block)
