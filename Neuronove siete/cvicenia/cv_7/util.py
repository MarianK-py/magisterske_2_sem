# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

import matplotlib
matplotlib.use('TkAgg')  # fixme if plotting doesn`t work (try 'Qt5Agg' or 'Qt4Agg')
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

import numpy as np
import atexit
import os
import time
import functools


## Utilities

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


def vector(array, row_vector=False):
    """
    Constructs a column vector (i.e. matrix of shape (n,1)) from given array/numpy.ndarray, or row
    vector (shape (1,n)) if row_vector = True.
    """
    v = np.array(array)
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
    plt.gcf().canvas.draw()   # fixme: uncomment if interactive drawing does not work
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


def limits(values, gap=0.05):
    x0 = np.min(values)
    x1 = np.max(values)
    xg = (x1 - x0) * gap
    return np.array((x0-xg, x1+xg))


def plot_errors(title, errors, test_error=None, block=True):
    plt.figure(1)
    use_keypress()
    plt.clf()

    plt.plot(errors)

    if test_error:
        plt.plot([test_error]*len(errors))

    plt.tight_layout()
    plt.gcf().canvas.set_window_title(title)
    plt.show(block=block)


def plot_sequence(targets, outputs=None, split=None, title=None, block=True):
    plt.figure(2)
    use_keypress()

    if outputs is None:
        plt.plot(targets, label='Targets')
        lim = limits(targets)

    else:
        plt.plot(targets, lw=5, alpha=0.3, label='Targets')
        plt.plot(outputs, label='Outputs')
        lim = limits(np.concatenate((outputs.flat, targets)))

    if split is not None:
        plt.vlines([split], ymin=lim[0], ymax=lim[1], color=palette[-1], lw=1)

    plt.ylim(lim)
    plt.legend()
    plt.tight_layout()
    #plt.gcf().canvas.set_window_title(title or ('Prediction' if outputs is not None else 'Sequence'))
    plt.show(block=block)
