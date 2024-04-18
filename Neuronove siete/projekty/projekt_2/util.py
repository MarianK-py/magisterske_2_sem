# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

import matplotlib
matplotlib.use('TkAgg')  # fixme if plotting doesn`t work (try 'Qt5Agg' or 'Qt4Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
import atexit
import os
import time
import seaborn as sns
import codecs


def clear():
    plt.clf()


def interactive_on():
    plt.ion()
    plt.show(block=False)
    time.sleep(0.1)


def interactive_off():
    plt.ioff()
    # plt.close()


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


# # non-blocking figures still block at end

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


def plot_grid_2d(inputs, weights, i_x=0, i_y=1, s=60, block=True):
    fig = plt.figure(1)
    use_keypress(fig)
    plt.gcf().canvas.manager.set_window_title('SOM neurons and inputs (2D)')

    plt.clf()

    plt.scatter(inputs[i_x, :], inputs[i_y, :], s=s, c=palette[-1], edgecolors=[0.4]*3, alpha=0.5)

    n_rows, n_cols, _ = weights.shape

    for r in range(n_rows):
        plt.plot(weights[r, :, i_x], weights[r, :, i_y], c=palette[0])

    for c in range(n_cols):
        plt.plot(weights[:, c, i_x], weights[:, c, i_y], c=palette[0])

    plt.xlim(limits(inputs[i_x, :]))
    plt.ylim(limits(inputs[i_y, :]))
    plt.tight_layout()
    plt.show(block=block)


def plot_grid_3d(inputs, weights, i_x=0, i_y=1, i_z=2, s=60, block=True):
    fig = plt.figure(2)
    use_keypress(fig)
    plt.gcf().canvas.manager.set_window_title('SOM neurons and inputs (3D)')

    if plot_grid_3d.ax is None:
        plot_grid_3d.ax = fig.add_subplot(111, projection='3d')

    ax = plot_grid_3d.ax
    ax.cla()

    ax.scatter(inputs[i_x, :], inputs[i_y, :], inputs[i_z, :], s=s, c=palette[-1], edgecolors=[0.4]*3, alpha=0.5)

    n_rows, n_cols, _ = weights.shape

    for r in range(n_rows):
        ax.plot(weights[r, :, i_x], weights[r, :, i_y], weights[r, :, i_z], c=palette[0])

    for c in range(n_cols):
        ax.plot(weights[:, c, i_x], weights[:, c, i_y], weights[:, c, i_z], c=palette[0])

    ax.set_xlim(limits(inputs[i_x, :]))
    ax.set_ylim(limits(inputs[i_y, :]))
    ax.set_zlim(limits(inputs[i_z, :]))
    plt.show(block=block)


plot_grid_3d.ax = None

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def plot_heatmap(errors, x_ticks, y_ticks, x_lab, y_lab, title, save=False, name=""):
    if len(errors.shape) > 2:
        n = errors.shape[2]
    else:
        n = 1
    if n>2:
        cols = round(n/2+0.1)
        fig, ax = plt.subplots(cols,2, figsize=(10,5*cols))
        for i in range(n):
            sns.heatmap(errors[:,:,i], ax=ax[i//2, i%2], xticklabels=x_ticks, yticklabels=y_ticks, cmap="turbo")
            ax[i//2, i%2].set_xlabel(x_lab)
            ax[i//2, i%2].set_ylabel(y_lab)
            ax[i//2, i%2].set_title(str(i+1)+"th "+title)
    elif n == 2:
        fig, ax = plt.subplots(1, 2)
        for i in range(n):
            sns.heatmap(errors[:,:,i], ax=ax[0, i], xticklabels=x_ticks, yticklabels=y_ticks, cmap="turbo")
            ax[0, i].set_xlabel(x_lab)
            ax[0, i].set_ylabel(y_lab)
            ax[0, i].set_title(str(i+1)+"th "+title)
    else:
        fig, ax = plt.subplots()
        sns.heatmap(errors, ax=ax, xticklabels=x_ticks, yticklabels=y_ticks, cmap="turbo")
        ax.set_xlabel(x_lab)
        ax.set_ylabel(y_lab)
        ax.set_title(title)
    #plt.tight_layout()
    if save:
        plt.savefig(name)
    plt.show()

def confusion_table(predicts, targets, lab_vals, create_tex=False,tex_name=""):
    print("\\begin{table}[!h]")
    print("\\begin{tabular}{|p{0.2\\textwidth}|"+(len(lab_vals)+1)*"p{0.08\\textwidth}|"+"}")
    print("\\hline")
    print("očakávanie/realita ",end=" ")
    for l in lab_vals+[0]:
        print("& ",int(l),end=" ")
    print("\\\\ \\hline")
    for l1 in lab_vals:
        print(int(l1), end=" ")
        for l2 in lab_vals+[0]:
            print("& ", round(100*np.sum((targets == l1) & (predicts == l2))/np.sum((targets == l1)), 2), end="\\% ")
            #print(l1,l2,np.sum((targets == l1) & (outputs_lab == l2)))
        print("\\\\ \\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    if create_tex:
        with codecs.open(tex_name+".tex", "w", "utf-8") as f:
            f.write("\\begin{table}[!h]\n")
            f.write("\\begin{tabular}{|p{0.2\\textwidth}|"+(len(lab_vals)+1)*"p{0.08\\textwidth}|"+"}\n")
            f.write("\\hline\n")
            f.write("očakávanie/realita")
            for l in lab_vals+[0]:
                f.write("& "+str(int(l)))
            f.write("\\\\ \\hline\n")
            for l1 in lab_vals:
                f.write(str(int(l1))+" ")
                for l2 in lab_vals+[0]:
                    f.write("& "+str(round(100*np.sum((targets == l1) & (predicts == l2))/np.sum((targets == l1)), 2))+"\\% ")
                    #print(l1,l2,np.sum((targets == l1) & (outputs_lab == l2)))
                f.write("\\\\ \\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
