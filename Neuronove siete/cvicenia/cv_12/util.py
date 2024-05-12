# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017 - 2024

import matplotlib
matplotlib.use('TkAgg') # fixme if plotting doesn`t work (try 'Qt5Agg' or 'Qt4Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import atexit
import os
import time
import functools

import pickle
import gzip
import urllib.request



## Globals

width  = None
height = None

def util_setup(w, h):
    '''
    Parameters for image plotting
    '''
    global width, height
    width  = w
    height = h



## Utilities

def vector(array, row_vector=False):
    '''
    Constructs a column vector (i.e. matrix of shape (n,1)) from given array/numpy.ndarray, or row
    vector (shape (1,n)) if row_vector = True.
    '''
    v = np.atleast_1d(np.array(array))
    if np.squeeze(v).ndim > 1:
        raise ValueError('Cannot construct vector from array of shape {}!'.format(v.shape))
    return v.reshape((1, -1) if row_vector else (-1, 1))


def add_bias(X):
    '''
    Add bias term to vector, or to every (column) vector in a matrix.
    '''
    if X.ndim == 1:
        return np.concatenate((X, [1]))
    else:
        pad = np.ones((1, X.shape[1]))
        return np.concatenate((X, pad), axis=0)


def timeit(func):
    '''
    Profiling function to measure time it takes to finish function.
    Args:
        func(*function): Function to meassure
    Returns:
        (*function) New wrapped function with meassurment
    '''
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        start_time = time.time()
        out = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        ftime = elapsed_time
        msg = "Function [{}] finished in {:.3f} s"
        print(msg.format(func.__name__, ftime))
        return out
    return newfunc


def onehot_decode(X):
    return np.argmax(X, axis=0)


def onehot_encode(L, c):
    if isinstance(L, int):
        L = [L]
    n = len(L)
    out = np.zeros((c, n))
    out[L, range(n)] = 1
    return np.squeeze(out)



## Data

def load_mnist(filename='mnist.pkl.gz', source='http://deeplearning.net/data/mnist/mnist.pkl.gz'):
    '''
    Download or load previously downloaded MNIST dataset
    '''
    if not os.path.isfile(filename):
        print('Downloading MNIST...')
        filename, _ = urllib.request.urlretrieve(source, filename)

    print('Loading ' + filename + '...')

    # estimation, validation, testing (1D flattened 784xfloat inputs, int labels)
    ((est_inputs, est_labels),
     (val_inputs, val_lables),
     (tst_inputs, tst_labels)) = pickle.load(gzip.open(filename, 'rb'), encoding='latin1')

    dim = est_inputs.shape[1]
    n_classes = 1+np.max(est_labels)

    # binarize input images

    est_inputs = (est_inputs > 0.5).astype(float)
    val_inputs = (val_inputs > 0.5).astype(float)
    tst_inputs = (tst_inputs > 0.5).astype(float)

    return est_inputs, est_labels


def balanced_subset(inputs, labels, count, classes=None):
    '''
    Choose subset of data with balanced classes (same number of examples from each class)
    '''
    classes = np.unique(labels) if classes is None else classes
    inps = []
    labs = []
    for c in classes:
        ind = np.nonzero(labels == c)[0]
        ind = np.random.choice(ind, size=count, replace=False)
        inps.append(inputs[ind])
        labs.append(labels[ind])
    return np.concatenate(inps), np.concatenate(labs)


def append_classes_to_data(inputs, labels):
    '''
    Concatenate one-hot-encoded label to input data
    '''
    n_classes = 1 + np.max(labels)
    targets = onehot_encode(labels, n_classes).T
    inputs = np.concatenate((inputs, targets), axis=1)
    return inputs


## Interactive drawing

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
        os._exit(0) # unclean exit, but exit() or sys.exit() won't work
    if e.key in {' ', 'enter'}:
        plt.close() # skip blocking figures

def use_keypress(fig=None):
    if fig is None:
        fig = plt.gcf()
    fig.canvas.mpl_connect('key_press_event', keypress)


## non-blocking figures still block at end

def finish():
    plt.show(block=True) # block until all figures are closed

atexit.register(finish)



## Plotting

def plot_images(S, L=None, rows=1, title=None, fig=1, block=True):
    fig = plt.figure(fig, figsize=(9,6))
    use_keypress(fig)
    plt.clf()

    S = np.array(S)[:, :height*width]
    cols = S.shape[0] // rows

    i = 0
    for r in range(rows):
        for c in range(cols):
            plt.subplot(rows, cols, i+1)

            plt.imshow(S[i].reshape((height, width)), cmap='gray_r', interpolation='nearest', vmin=0, vmax=+1)
            if L is not None:
                plt.title(L[i])

            plt.xticks([])
            plt.yticks([])
            i += 1

    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99 if L is None else 0.95, wspace=0.05, hspace=0.05 if L is None else 0.3)
    plt.gcf().canvas.manager.set_window_title(title or 'States')
    plt.show(block=block)
    if not block:
        redraw()


def label_to_title(X):
    arg = np.argwhere(X)
    return str(np.concatenate(arg) if len(arg) > 0 else np.array([]))
