# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

import matplotlib
matplotlib.use('TkAgg') # fixme if plotting doesn`t work (try 'Qt5Agg' or 'Qt4Agg')
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


# Interactive drawing

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


# Plotting

palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']


def plot_errors(errors, show=True):
    plt.figure(num='Error')
    use_keypress()
    plt.plot(errors)
    if show:
        plt.show()


def plot_dots(inputs, targets=None, s=100, show=True):
    plt.figure(num='Data')
    use_keypress()
    if targets is None:
        plt.scatter(inputs[:, 0], inputs[:, 1], s=s, c=palette[-1])
    else:
        for i, c in enumerate(set(targets)):
            plt.scatter(inputs[targets == c, 0], inputs[targets == c, 1], s=s, c=palette[i])
    if show:
        plt.show()


def limits(values, gap=0.05):
    x0 = np.min(values)
    x1 = np.max(values)
    xg = (x1 - x0) * gap
    return np.array((x0-xg, x1+xg))


def plot_decision(w, inputs, targets=None, s=100, show=True):
    plot_dots(inputs, targets, s=s, show=False)

    x = limits(inputs[:, 0])
    y = - (w[0] * x + w[2]) / w[1]
    plt.plot(x, y, color='black')

    plt.xlim(limits(inputs[:, 0]))
    plt.ylim(limits(inputs[:, 1]))

    if show:
        plt.show()
