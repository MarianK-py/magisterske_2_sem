# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

import matplotlib
matplotlib.use('TkAgg')  # fixme if plotting doesn`t work (try 'Qt5Agg' or 'Qt4Agg')
import matplotlib.pyplot as plt
# for 3D visualization
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
import codecs

import numpy as np
import atexit
import os
import time
import functools

import seaborn as sns


def onehot_decode(inp, lab_vals):
    return lab_vals[np.argmax(inp, axis=0)]


def onehot_encode(lab, lab_vals):
    if isinstance(lab, str):
        lab = [lab]
    n = len(lab)
    num_c = len(lab_vals)
    idx = np.array([np.where(lab_vals == l)[0] for l in lab]).T
    out = np.zeros((num_c, n))
    out[idx, range(n)] = 1
    return np.squeeze(out)

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

# Non-blocking figures still block at end
def finish():
    plt.show(block=True)  # block until all figures are closed


atexit.register(finish)


# # Plotting
def plot_errors(title, errors, labels, save=False, name=""):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_ylim(bottom=0)

    for i, err in enumerate(errors):
        ax.plot(err, label=labels[i])

    #plt.tight_layout()
    ax.set_title(title)
    plt.legend()
    if save:
        plt.savefig(name)
    plt.show()

palette1 = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
palette2 = ['#f8282a', '#6090d0', '#60ee60', '#b36ac3', '#ff9f20', '#ffff55', '#c68638', '#f7a1df', '#aaaaaa']

def plot_dots(ax, inputs, targets, s=10, lab_vals=None, palette=palette1, typ="", save=False, name=""):
    for i, c in enumerate(lab_vals):
        ax.scatter(inputs[0, targets == c], inputs[1, targets == c], s=s, c=palette[i], label=c+" "+typ)

    if save:
        plt.savefig(name+"_"+typ)

def limits(values, gap=0.05):
    x0 = np.min(values)
    x1 = np.max(values)
    xg = (x1 - x0) * gap
    return np.array((x0-xg, x1+xg))

def mesh2D(inputs, step=0.05):
    xLim = limits(inputs[0, :])
    yLim = limits(inputs[1, :])
    xMesh = np.arange(xLim[0], xLim[1], step)
    yMesh = np.arange(yLim[0], yLim[1], step)
    x,y = np.meshgrid(xMesh, yMesh)
    x = np.concatenate(np.concatenate(x))
    y = np.concatenate(np.concatenate(y))
    mesh = np.array([x, y]).T
    return mesh

def plot_decision(model, inputs, targets, s1=20, s2=2, lab_vals=None, save=False, name=""):
    fig = plt.figure(figsize=(12,9))
    #ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot()

    outputs_val, outputs_lab = model.predict(inputs)

    plot_dots(ax, inputs, targets, s=s1, lab_vals=lab_vals, palette=palette1, typ="real", save=save, name=name)
    plot_dots(ax, inputs, outputs_lab, s=s2, lab_vals=lab_vals, palette=palette1, typ="model", save=save, name=name)

    #boundry = model.getBoundry(mesh2D(inputs, 0.1), mid=0).T

    #ax.plot_surface(boundry[0, :], boundry[1, :], boundry[2, :], cmap=cm.hot, ec='k')
    #ax.scatter(boundry[0, :], boundry[1, :], boundry[2, :], c='#ffff33', alpha=0.4)

    #boundry = model.getBoundry(mesh3D(inputs, 0.15), mid=0.5)

    #ax.scatter(boundry[0, :], boundry[1, :], boundry[2, :], c='#ffff33')

    ax.set_xlim(limits(inputs[0, :]))
    ax.set_ylim(limits(inputs[1, :]))

    ax.legend()

    plt.show()


def plot_heatmap(errors, x_ticks, y_ticks, x_lab, y_lab, title, save=False, name=""):
    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot()
    sns.heatmap(errors, ax=ax, vmin=0, xticklabels=x_ticks, yticklabels=y_ticks, cmap='plasma')
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    ax.set_title(title)
    #plt.tight_layout()
    if save:
        plt.savefig(name)
    plt.show()

def confusion_table(model, inputs, targets, lab_vals, create_tex=False,tex_name=""):
    outputs_val, outputs_lab = model.predict(inputs)
    n = targets.shape[0]
    print("\\begin{table}[!h]")
    print("\\begin{tabular}{|p{0.08\\textwidth}|"+len(lab_vals)*"p{0.08\\textwidth}|"+"}")
    print("\\hline")
    print(" ",end=" ")
    for l in lab_vals:
        print("& ",l,end=" ")
    print("\\\\ \\hline")
    for l1 in lab_vals:
        print(l1, end=" ")
        for l2 in lab_vals:
            print("& ", round(100*np.sum((targets == l1) & (outputs_lab == l2))/np.sum((targets == l1)), 2), end="\\% ")
            #print(l1,l2,np.sum((targets == l1) & (outputs_lab == l2)))
        print("\\\\ \\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    if create_tex:
        with codecs.open("report/"+tex_name+".tex", "w", "utf-8") as f:
            f.write("\\begin{table}[!h]\n")
            f.write("\\begin{tabular}{|p{0.08\\textwidth}|"+len(lab_vals)*"p{0.08\\textwidth}|"+"}\n")
            f.write("\\hline\n")
            for l in lab_vals:
                f.write("& "+str(l))
            f.write("\\\\ \\hline\n")
            for l1 in lab_vals:
                f.write(str(l1)+" ")
                for l2 in lab_vals:
                    f.write("& "+str(round(100*np.sum((targets == l1) & (outputs_lab == l2))/np.sum((targets == l1)), 2))+"\\% ")
                    #print(l1,l2,np.sum((targets == l1) & (outputs_lab == l2)))
                f.write("\\\\ \\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

def print_errors(errors, x_vals, y_vals, create_tex=False,tex_name=""):
    print("\\begin{table}[!h]")
    print("\\begin{tabular}{|p{0.08\\textwidth}|"+len(x_vals)*"p{0.08\\textwidth}|"+"}")
    print("\\hline")
    print(" ",end=" ")
    for l in x_vals:
        print("& ",l,end=" ")
    print("\\\\ \\hline")
    for i1 in range(len(y_vals)):
        print(y_vals[i1], end=" ")
        for i2 in range(len(x_vals)):
            print("& ", round(errors[i1, i2], 3), end=" ")
            #print(l1,l2,np.sum((targets == l1) & (outputs_lab == l2)))
        print("\\\\ \\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    if create_tex:
        with codecs.open("report/"+tex_name+".tex", "w", "utf-8") as f:
            f.write("\\begin{table}[!h]\n")
            f.write("\\begin{tabular}{|p{0.08\\textwidth}|"+len(x_vals)*"p{0.08\\textwidth}|"+"}\n")
            f.write("\\hline\n")
            for l in x_vals:
                f.write("& "+str(l))
            f.write("\\\\ \\hline")
            for i1 in range(len(y_vals)):
                f.write(str(y_vals[i1])+" ")
                for i2 in range(len(x_vals)):
                    f.write("& "+str(round(errors[i1, i2], 3)))
                    #print(l1,l2,np.sum((targets == l1) & (outputs_lab == l2)))
                f.write("\\\\ \\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

def print_best_layer_sizes(h1, h2, create_tex=False,tex_name=""):
    print(f"Vidíme, že najlepšiu validačnú chybu máme pre prvú skrytú vrstvu veľkosti {h1}", end="")
    print(f"a druhú skrytú vrstvu veľkosti {h2}.", end="")
    print("Preto náš finálny model budeme trénovať s vrstvami práve tejto veľkosti. ", end="")
    print("Tento model budeme trénovať na celom trénovacom datasete (estimačný aj validačný) ", end="")
    print("a na určenie kvality modelu použijeme testovacie dáta.", end="")

    if create_tex:
        with codecs.open("report/"+tex_name+".tex", "w", "utf-8") as f:
            f.write(f"Vidíme, že najlepšiu validačnú chybu máme pre prvú skrytú vrstvu veľkosti {h1} ")
            f.write(f"a druhú skrytú vrstvu veľkosti {h2}. ")
            f.write("Preto náš finálny model budeme trénovať s vrstvami práve tejto veľkosti. ")
            f.write("Tento model budeme trénovať na celom trénovacom datasete (estimačný aj validačný) ")
            f.write("a na určenie kvality modelu použijeme testovacie dáta.")