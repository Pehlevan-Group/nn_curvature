"""
visualize empirical decision boundary
"""

# load packages
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
import matplotlib.colors as clr


def makevid_empirical(
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
    is_binary=False,
    label='volume element',
):
    fig.set_tight_layout(True)
    spacing = (y[1] - y[0]).item()
    half_spacing = spacing / 2

    L, M = np.meshgrid(
        l.detach().cpu().numpy(), y.detach().cpu().numpy(), indexing="ij"
    )

    div = make_axes_locatable(ax)
    cax = div.append_axes("right", "5%", "2%")

    colors = coloring[0].detach().cpu().numpy()
    if is_binary:  # binary classifications
        cur_cmap = getattr(plt.cm, cmap)
        norm = clr.BoundaryNorm(np.arange(-0.5, 2, 1), cur_cmap.N)
        sct = ax.scatter(L, M, c=colors, cmap=cur_cmap, norm=norm, edgecolor="none")
        cb = fig.colorbar(sct, cax=cax, ticks=[0, 1])
    else:
        sct = ax.scatter(L, M, c=colors, cmap=cmap, vmin=-100, vmax=100)
        cb = fig.colorbar(sct, cax=cax)

    cb.ax.set_ylabel(label, rotation=270, labelpad=8)

    tx = ax.set_title("Epoch 0")

    ax.set_xlim(L.min() - 0.05, L.max() + 0.05)
    ax.set_xlabel('x1')
    ax.set_ylim(M.min() - 0.05, M.max() + 0.05)
    ax.set_ylabel('x2')

    ax.set_aspect('equal', adjustable='box')

    # the colorchange line
    if plot_line:
        line = ax.plot([], c="tab:red")

    def animate(frame_num, data, predictions, sct):

        # plot curvature dots
        colors = data[frame_num].detach().cpu().numpy()
        vmax = np.max(colors)
        vmin = np.min(colors)

        if vmin > 0:
            sct = ax.scatter(L, M, norm=clr.LogNorm(), c=colors, cmap=cmap)
            cb = fig.colorbar(sct, cax=cax)
        else:
            if is_binary:  # binary classifications
                sct = ax.scatter(
                    L, M, c=colors, cmap=cur_cmap, norm=norm, edgecolor="none"
                )
                cb = fig.colorbar(sct, cax=cax, ticks=[0, 1])
            else:
                sct = ax.scatter(L, M, c=colors, cmap=cmap,
                                 vmin=-100, vmax=100)
                cb = fig.colorbar(sct, cax=cax)
        
        cb.ax.set_ylabel(label, rotation=270, labelpad=8)

        tx.set_text("Epoch {}".format(int(epochs / frms) * frame_num))

        if plot_line:
            # plot colorchange boundary
            pred = predictions[frame_num].detach().cpu().numpy().reshape(L.shape)
            transitions = np.where(np.diff(pred, axis=1) != 0)  # horizontal boundary
            boundary_x = L[transitions]
            boundary_y = M[transitions] + half_spacing
            # refresh line
            line[0].set_data((boundary_x, boundary_y))

            return sct, line

        else:
            return (sct,)

    anim1 = FuncAnimation(
        fig, animate, frames=frms + 1, interval=300, fargs=(coloring, predictions, sct)
    )

    # plt.legend(["neuron_1", "neuron_2"], loc=8)

    return anim1


def makevid_empirical_multi(
    width,
    frms,
    epochs,
    l,
    y,
    coloring_list,
    predictions,
    weightlist,
    biaslist,
    fig,
    ax,
    plot_line=True,
    cmap="viridis",
    label='volume element'
):
    fig.set_tight_layout(True)
    spacing = (y[1] - y[0]).item()
    half_spacing = spacing / 2

    L, M = np.meshgrid(
        l.detach().cpu().numpy(), y.detach().cpu().numpy(), indexing="ij"
    )

    div = make_axes_locatable(ax[-1])
    cax = div.append_axes("right", "5%", "2%")

    # initial colorings
    colors_list = [data[0].detach().cpu().numpy() for data in coloring_list]
    vmax = np.max([np.max(colors) for colors in colors_list])
    vmin = np.min([np.min(colors) for colors in colors_list])
    for i, coloring in enumerate(coloring_list):
        colors = coloring[0].detach().cpu().numpy()
        sct = ax[i].scatter(L, M, c=colors, vmax=vmax, vmin=vmin, cmap=cmap)
        ax[i].set_xlim(L.min() - 0.05, L.max() + 0.05)
        ax[i].set_xlabel('x1')
        ax[i].set_ylim(M.min() - 0.05, M.max() + 0.05)
        ax[i].set_ylabel('x2')
        # ax[i].set_aspect('equal', adjustable='box')

    cb = fig.colorbar(sct, cax=cax)
    cb.ax.set_ylabel(label, rotation=270, labelpad=8)

    tx = plt.suptitle("Epoch 0")
    plt.tight_layout()

    # the colorchange line
    if plot_line:
        lines = []
        for i in range(len(coloring_list)):
            line = ax[i].plot([], c="tab:red")
            lines.append(line)

    def animate(frame_num, data_list, predictions, sct):
        # find out min and max across different vis
        colors_list = [data[frame_num].detach().cpu().numpy() for data in data_list]
        vmax = np.max([np.max(colors) for colors in colors_list])
        vmin = np.min([np.min(colors) for colors in colors_list])

        for i, colors in enumerate(colors_list):
            if vmin > 0:
                sct = ax[i].scatter(
                    L, M, norm="log", c=colors, vmin=vmin, vmax=vmax, cmap=cmap
                )
                # fig.colorbar(sct)
            else:
                sct = ax[i].scatter(L, M, c=colors, vmin=vmin, vmax=vmax, cmap=cmap)
                # fig.colorbar(sct)

        cb = fig.colorbar(sct, cax=cax)
        cb.ax.set_ylabel(label, rotation=270, labelpad=8)

        tx.set_text("Epoch {}".format(int(epochs / frms) * frame_num))

        if plot_line:
            # plot colorchange boundary
            pred = predictions[frame_num].detach().cpu().numpy().reshape(L.shape)
            transitions = np.where(np.diff(pred, axis=1) != 0)  # horizontal boundary
            boundary_x = L[transitions]
            boundary_y = M[transitions] + half_spacing
            # refresh line
            for line in lines:
                line[0].set_data((boundary_x, boundary_y))

            return sct, line

        else:
            return (sct,)

    anim1 = FuncAnimation(
        fig,
        animate,
        frames=frms + 1,
        interval=300,
        fargs=(coloring_list, predictions, sct),
    )

    # plt.legend(["neuron_1", "neuron_2"], loc=8)

    return anim1
