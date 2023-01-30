"""
make videos of curvature flows
"""

# load packages
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
import matplotlib.colors as clr


def makevid_analytic(
    width,
    frms,
    epochs,
    l,
    y,
    coloring,
    predictions,
    weightlist,
    biaslist,
    fig,
    ax,
    plot_line=True,
    cmap="viridis",
    label='log10 volume element'
):
    fig.set_tight_layout(True)
    L, M = np.meshgrid(
        l.detach().cpu().numpy(), y.detach().cpu().numpy(), indexing="ij"
    )

    div = make_axes_locatable(ax)
    cax = div.append_axes("right", "5%", "2%")

    xs = []
    ys = []

    # plot decision boundaries
    if plot_line:
        # read
        model_weights = weightlist[0].data.detach().cpu().numpy()
        model_bias = biaslist[0].data.detach().cpu().numpy()

        # plot
        lines = []
        for i in range(width):
            xs.append(np.arange(-1.5, 1.5, 0.1))
            ys.append(
                ((xs[i] * model_weights[i, 0]) + model_bias[i % 2])
                / (-model_weights[i, 1])
            )
            lines.append(ax.plot([], color="tab:red")[0])

    colors = coloring[0].detach().cpu().numpy()
    sct = ax.scatter(L, M, c=colors, cmap=cmap)

    cb = fig.colorbar(sct, cax=cax)
    cb.ax.set_ylabel(label, rotation=270)

    tx = ax.set_title("Epoch 0")

    ax.set_xlim(L.min() - 0.05, L.max() + 0.05)
    ax.set_ylim(M.min() - 0.05, M.max() + 0.05)
    
    ax.set_aspect('equal', adjustable='box')

    def animate(frame_num, data, sct):

        # plot curvature dots
        colors = data[frame_num].detach().cpu().numpy()
        vmax = np.max(colors)
        vmin = np.min(colors)

        if vmin > 0:
            sct = ax.scatter(L, M, norm=clr.LogNorm(), c=colors, cmap=cmap)
            cb = fig.colorbar(sct, cax=cax)
        else:
            sct = ax.scatter(L, M, c=colors, cmap=cmap)
            cb = fig.colorbar(sct, cax=cax)

        cb.ax.set_ylabel(label, rotation=270)
        tx.set_text("Epoch {}".format(int(epochs / frms) * frame_num))

        returns = [sct]

        # plot decision boundary
        if plot_line:
            model_weights = weightlist[frame_num].data.detach().cpu().numpy()
            model_bias = biaslist[frame_num].data.detach().cpu().numpy()

            newys = []

            for i in range(width):
                newys.append(
                    ((xs[i] * model_weights[i, 0]) + model_bias[i % 2])
                    / (-model_weights[i, 1])
                )
                newline = lines[i]
                newline.set_data((xs[i], newys[i]))
                returns.append(newline)
        return tuple(returns)

    anim1 = FuncAnimation(
        fig, animate, frames=frms + 1, interval=300, fargs=(coloring, sct)
    )

    # plt.legend(["neuron_1", "neuron_2"], loc=8)

    return anim1
