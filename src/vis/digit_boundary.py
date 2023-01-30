"""
visualize effective radius
"""

# load packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
import matplotlib.colors as clr

CMAP = "tab10"
plt.style.use("ggplot")

# =================== pictures =====================
def makepic_digits(
    labels, epoch, eff_vol, prediction, img_left, img_right, img_mid, fig, ax
):
    """for cifar10"""
    fig.set_tight_layout(True)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", "5%", "2%")

    colors = prediction.detach().cpu().numpy()
    points = eff_vol.detach().cpu().numpy()
    sct = ax.scatter(
        list(range(len(points))), points, c=colors, cmap=CMAP, vmin=0, vmax=10
    )
    ax.set_ylabel('log10 volume element')

    # colorbar
    cb = fig.colorbar(sct, cax=cax)
    cb.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(labels):
        cb.ax.text(
            0.5,
            (2 * j + 1) / 2,
            lab,
            ha="center",
            va="center",
            rotation=270,
            fontsize="small",
        )
    cb.ax.get_yaxis().labelpad = 15
    cb.ax.set_ylabel("class prediction", rotation=270)

    ax.set_title(f"Epoch {epoch}")

    # visualize sub image
    left_img_ax = ax.inset_axes([0, 0, 0.1, 0.1])
    left_img_ax.imshow(img_left.permute(1, 2, 0).detach().cpu().numpy())
    left_img_ax.axis("off")
    mid_img_ax = ax.inset_axes([0.45, 0, 0.1, 0.1])
    mid_img_ax.imshow(img_mid.permute(1, 2, 0).detach().cpu().numpy())
    mid_img_ax.axis("off")
    right_img_ax = ax.inset_axes([0.9, 0, 0.1, 0.1])
    right_img_ax.imshow(img_right.permute(1, 2, 0).detach().cpu().numpy())
    right_img_ax.axis("off")

    # ax.set_xlim(L.min() - 0.05, L.max() + 0.05)
    # ax.set_ylim(-100, 0)

    # determine range
    # final_epoch_points = eff_vol[-1].detach().cpu().numpy()
    # y_range = (
    #     final_epoch_points.max() - final_epoch_points.min()
    # )  # the range is fixed across epochs
    # cur_y_mid = (points.max() + points.min()) / 2
    # y_lim_max = cur_y_mid + y_range / 2
    # y_lim_min = cur_y_mid - y_range / 2
    # ax.set_ylim(y_lim_min, y_lim_max)


def makepic_digits_plane(
    labels,
    epoch,
    l,
    y,
    eff_vol,
    entropy,
    prediction,
    img_one,
    img_two,
    img_three,
    origin,
    fig,
    ax,
    ternary=False,
    cmap="viridis",
):
    """for cifar10"""
    div1 = make_axes_locatable(ax[0])
    cax1 = div1.append_axes("right", "5%", "2%")
    div2 = make_axes_locatable(ax[1])
    cax2 = div2.append_axes("right", "5%", "2%")
    div3 = make_axes_locatable(ax[2])
    cax3 = div3.append_axes("right", "5%", "2%")

    L, M = np.meshgrid(
        l.detach().cpu().numpy(), y.detach().cpu().numpy(), indexing="ij"
    )
    colors = prediction.detach().cpu().numpy()
    points = eff_vol.detach().cpu().numpy()

    if ternary:  # only visualize the convex hull of the anchor points
        mask = (
            (M < 1 / (2 * 3**0.5))
            & (M > 3**0.5 * L - 1 / 3**0.5)
            & (M > -(3**0.5) * L - 1 / 3**0.5)
        )
        L_masked, M_masked = np.ma.array(L, mask=~mask), np.ma.array(M, mask=~mask)
        # L_masked[~mask] = np.nan
        # M_masked[~mask] = np.nan
        points_masked = np.ma.array(points.reshape(*L_masked.shape), mask=~mask)

        sct1 = ax[0].scatter(L_masked, M_masked, c=colors, cmap=CMAP, vmin=0, vmax=10)
        sct2 = ax[1].scatter(
            L_masked, M_masked, c=entropy, cmap="cividis"
        )  # ? colormap ?
        sct3 = ax[2].contourf(L_masked, M_masked, points_masked, levels=12)

        # ===================== axes 1 ======================
        # manually draw the ternary axis
        ax[0].plot([-0.5, 0.5], [1 / 2 / (3**0.5), 1 / 2 / (3**0.5)], color="black")
        ax[0].plot([-0.5, 0], [1 / 2 / (3**0.5), -1 / (3**0.5)], color="black")
        ax[0].plot([0, 0.5], [-1 / (3**0.5), 1 / 2 / (3**0.5)], color="black")

        tick_width = 0.02
        text_offset = 0.02

        # top line ticks
        # ax[0].plot([-0.5, -0.5], [1/2/(3 ** 0.5), 1/2/(3 ** 0.5) + tick_width], color='black')
        # ax[0].text(-0.5 - text_offset, 1/2/(3 ** 0.5) + tick_width, '0.0')
        ax[0].plot(
            [-0.3, -0.3],
            [1 / 2 / (3**0.5), 1 / 2 / (3**0.5) + tick_width],
            color="black",
        )
        ax[0].text(-0.3 - text_offset, 1 / 2 / (3**0.5) + tick_width, "0.2")
        ax[0].plot(
            [-0.1, -0.1],
            [1 / 2 / (3**0.5), 1 / 2 / (3**0.5) + tick_width],
            color="black",
        )
        ax[0].text(-0.1 - text_offset, 1 / 2 / (3**0.5) + tick_width, "0.4")
        ax[0].plot(
            [0.1, 0.1],
            [1 / 2 / (3**0.5), 1 / 2 / (3**0.5) + tick_width],
            color="black",
        )
        ax[0].text(0.1 - text_offset, 1 / 2 / (3**0.5) + tick_width, "0.6")
        ax[0].plot(
            [0.3, 0.3],
            [1 / 2 / (3**0.5), 1 / 2 / (3**0.5) + tick_width],
            color="black",
        )
        ax[0].text(0.3 - text_offset, 1 / 2 / (3**0.5) + tick_width, "0.8")
        # ax[0].plot([0.5, 0.5], [1/2/(3 ** 0.5), 1/2/(3 ** 0.5) + tick_width], color='black')
        # ax[0].text(0.5 - text_offset, 1/2/(3 ** 0.5) + tick_width, '1.0')

        # left line
        y_descent = 3**0.5 / 10
        # ax[0].plot([-0.5 - tick_width, -0.5], [1/2/(3 ** 0.5), 1/2/(3 ** 0.5)], color='black')
        # ax[0].text(-0.5 - tick_width - 3 * text_offset, 1/2/(3 ** 0.5), '0.0')
        ax[0].plot(
            [-0.4 - tick_width, -0.4],
            [1 / 2 / (3**0.5) - y_descent, 1 / 2 / (3**0.5) - y_descent],
            color="black",
        )
        ax[0].text(
            -0.4 - tick_width - 3 * text_offset, 1 / 2 / (3**0.5) - y_descent, "0.2"
        )
        ax[0].plot(
            [-0.3 - tick_width, -0.3],
            [1 / 2 / (3**0.5) - 2 * y_descent, 1 / 2 / (3**0.5) - 2 * y_descent],
            color="black",
        )
        ax[0].text(
            -0.3 - tick_width - 3 * text_offset,
            1 / 2 / (3**0.5) - 2 * y_descent,
            "0.4",
        )
        ax[0].plot(
            [-0.2 - tick_width, -0.2],
            [1 / 2 / (3**0.5) - 3 * y_descent, 1 / 2 / (3**0.5) - 3 * y_descent],
            color="black",
        )
        ax[0].text(
            -0.2 - tick_width - 3 * text_offset,
            1 / 2 / (3**0.5) - 3 * y_descent,
            "0.6",
        )
        ax[0].plot(
            [-0.1 - tick_width, -0.1],
            [1 / 2 / (3**0.5) - 4 * y_descent, 1 / 2 / (3**0.5) - 4 * y_descent],
            color="black",
        )
        ax[0].text(
            -0.1 - tick_width - 3 * text_offset,
            1 / 2 / (3**0.5) - 4 * y_descent,
            "0.8",
        )
        # ax[0].plot([-0. - tick_width, -0.], [1/2/(3 ** 0.5) - 5  * y_descent, 1/2/(3 ** 0.5) - 5  * y_descent], color='black')
        # ax[0].text(-0. - tick_width - 3 * text_offset, 1/2/(3 ** 0.5) - 5  * y_descent, '1.0')

        # right line
        y_descent = 3**0.5 / 10
        # ax[0].plot([0.5, tick_width + 0.5], [1/2/(3 ** 0.5), 1/2/(3 ** 0.5)], color='black')
        # ax[0].text(0.5 + 0.75 * text_offset + tick_width, 1/2/(3 ** 0.5), '0.0')
        ax[0].plot(
            [0.4, tick_width + 0.4],
            [1 / 2 / (3**0.5) - y_descent, 1 / 2 / (3**0.5) - y_descent],
            color="black",
        )
        ax[0].text(
            0.4 + 0.75 * text_offset + tick_width, 1 / 2 / (3**0.5) - y_descent, "0.2"
        )
        ax[0].plot(
            [0.3, tick_width + 0.3],
            [1 / 2 / (3**0.5) - 2 * y_descent, 1 / 2 / (3**0.5) - 2 * y_descent],
            color="black",
        )
        ax[0].text(
            0.3 + 0.75 * text_offset + tick_width,
            1 / 2 / (3**0.5) - 2 * y_descent,
            "0.4",
        )
        ax[0].plot(
            [0.2, tick_width + 0.2],
            [1 / 2 / (3**0.5) - 3 * y_descent, 1 / 2 / (3**0.5) - 3 * y_descent],
            color="black",
        )
        ax[0].text(
            0.2 + 0.75 * text_offset + tick_width,
            1 / 2 / (3**0.5) - 3 * y_descent,
            "0.6",
        )
        ax[0].plot(
            [0.1, tick_width + 0.1],
            [1 / 2 / (3**0.5) - 4 * y_descent, 1 / 2 / (3**0.5) - 4 * y_descent],
            color="black",
        )
        ax[0].text(
            0.1 + 0.75 * text_offset + tick_width,
            1 / 2 / (3**0.5) - 4 * y_descent,
            "0.8",
        )
        # ax[0].plot([0., tick_width + 0.], [1/2/(3 ** 0.5) - 5  * y_descent, 1/2/(3 ** 0.5) - 5  * y_descent], color='black')
        # ax[0].text(0. + 0.75 * text_offset + tick_width, 1/2/(3 ** 0.5) - 5  * y_descent, '1.0')

        ax[0].set_xlim([-0.6, 0.6])
        ax[0].set_ylim([-0.7, 0.5])

        # ===================== axes 2 ======================
        # manually draw the ternary axis
        ax[1].plot([-0.5, 0.5], [1 / 2 / (3**0.5), 1 / 2 / (3**0.5)], color="black")
        ax[1].plot([-0.5, 0], [1 / 2 / (3**0.5), -1 / (3**0.5)], color="black")
        ax[1].plot([0, 0.5], [-1 / (3**0.5), 1 / 2 / (3**0.5)], color="black")

        tick_width = 0.02
        text_offset = 0.02

        # top line ticks
        # ax[1].plot([-0.5, -0.5], [1/2/(3 ** 0.5), 1/2/(3 ** 0.5) + tick_width], color='black')
        # ax[1].text(-0.5 - text_offset, 1/2/(3 ** 0.5) + tick_width, '0.0')
        ax[1].plot(
            [-0.3, -0.3],
            [1 / 2 / (3**0.5), 1 / 2 / (3**0.5) + tick_width],
            color="black",
        )
        ax[1].text(-0.3 - text_offset, 1 / 2 / (3**0.5) + tick_width, "0.2")
        ax[1].plot(
            [-0.1, -0.1],
            [1 / 2 / (3**0.5), 1 / 2 / (3**0.5) + tick_width],
            color="black",
        )
        ax[1].text(-0.1 - text_offset, 1 / 2 / (3**0.5) + tick_width, "0.4")
        ax[1].plot(
            [0.1, 0.1],
            [1 / 2 / (3**0.5), 1 / 2 / (3**0.5) + tick_width],
            color="black",
        )
        ax[1].text(0.1 - text_offset, 1 / 2 / (3**0.5) + tick_width, "0.6")
        ax[1].plot(
            [0.3, 0.3],
            [1 / 2 / (3**0.5), 1 / 2 / (3**0.5) + tick_width],
            color="black",
        )
        ax[1].text(0.3 - text_offset, 1 / 2 / (3**0.5) + tick_width, "0.8")
        # ax[1].plot([0.5, 0.5], [1/2/(3 ** 0.5), 1/2/(3 ** 0.5) + tick_width], color='black')
        # ax[1].text(0.5 - text_offset, 1/2/(3 ** 0.5) + tick_width, '1.0')

        # left line
        y_descent = 3**0.5 / 10
        # ax[1].plot([-0.5 - tick_width, -0.5], [1/2/(3 ** 0.5), 1/2/(3 ** 0.5)], color='black')
        # ax[1].text(-0.5 - tick_width - 3 * text_offset, 1/2/(3 ** 0.5), '0.0')
        ax[1].plot(
            [-0.4 - tick_width, -0.4],
            [1 / 2 / (3**0.5) - y_descent, 1 / 2 / (3**0.5) - y_descent],
            color="black",
        )
        ax[1].text(
            -0.4 - tick_width - 3 * text_offset, 1 / 2 / (3**0.5) - y_descent, "0.2"
        )
        ax[1].plot(
            [-0.3 - tick_width, -0.3],
            [1 / 2 / (3**0.5) - 2 * y_descent, 1 / 2 / (3**0.5) - 2 * y_descent],
            color="black",
        )
        ax[1].text(
            -0.3 - tick_width - 3 * text_offset,
            1 / 2 / (3**0.5) - 2 * y_descent,
            "0.4",
        )
        ax[1].plot(
            [-0.2 - tick_width, -0.2],
            [1 / 2 / (3**0.5) - 3 * y_descent, 1 / 2 / (3**0.5) - 3 * y_descent],
            color="black",
        )
        ax[1].text(
            -0.2 - tick_width - 3 * text_offset,
            1 / 2 / (3**0.5) - 3 * y_descent,
            "0.6",
        )
        ax[1].plot(
            [-0.1 - tick_width, -0.1],
            [1 / 2 / (3**0.5) - 4 * y_descent, 1 / 2 / (3**0.5) - 4 * y_descent],
            color="black",
        )
        ax[1].text(
            -0.1 - tick_width - 3 * text_offset,
            1 / 2 / (3**0.5) - 4 * y_descent,
            "0.8",
        )
        # ax[1].plot([-0. - tick_width, -0.], [1/2/(3 ** 0.5) - 5  * y_descent, 1/2/(3 ** 0.5) - 5  * y_descent], color='black')
        # ax[1].text(-0. - tick_width - 3 * text_offset, 1/2/(3 ** 0.5) - 5  * y_descent, '1.0')

        # right line
        y_descent = 3**0.5 / 10
        # ax[1].plot([0.5, tick_width + 0.5], [1/2/(3 ** 0.5), 1/2/(3 ** 0.5)], color='black')
        # ax[1].text(0.5 + 0.75 * text_offset + tick_width, 1/2/(3 ** 0.5), '0.0')
        ax[1].plot(
            [0.4, tick_width + 0.4],
            [1 / 2 / (3**0.5) - y_descent, 1 / 2 / (3**0.5) - y_descent],
            color="black",
        )
        ax[1].text(
            0.4 + 0.75 * text_offset + tick_width, 1 / 2 / (3**0.5) - y_descent, "0.2"
        )
        ax[1].plot(
            [0.3, tick_width + 0.3],
            [1 / 2 / (3**0.5) - 2 * y_descent, 1 / 2 / (3**0.5) - 2 * y_descent],
            color="black",
        )
        ax[1].text(
            0.3 + 0.75 * text_offset + tick_width,
            1 / 2 / (3**0.5) - 2 * y_descent,
            "0.4",
        )
        ax[1].plot(
            [0.2, tick_width + 0.2],
            [1 / 2 / (3**0.5) - 3 * y_descent, 1 / 2 / (3**0.5) - 3 * y_descent],
            color="black",
        )
        ax[1].text(
            0.2 + 0.75 * text_offset + tick_width,
            1 / 2 / (3**0.5) - 3 * y_descent,
            "0.6",
        )
        ax[1].plot(
            [0.1, tick_width + 0.1],
            [1 / 2 / (3**0.5) - 4 * y_descent, 1 / 2 / (3**0.5) - 4 * y_descent],
            color="black",
        )
        ax[1].text(
            0.1 + 0.75 * text_offset + tick_width,
            1 / 2 / (3**0.5) - 4 * y_descent,
            "0.8",
        )
        # ax[1].plot([0., tick_width + 0.], [1/2/(3 ** 0.5) - 5  * y_descent, 1/2/(3 ** 0.5) - 5  * y_descent], color='black')
        # ax[1].text(0. + 0.75 * text_offset + tick_width, 1/2/(3 ** 0.5) - 5  * y_descent, '1.0')

        ax[1].set_xlim([-0.6, 0.6])
        ax[1].set_ylim([-0.7, 0.5])

        # ===================== axes 3 ======================
        # manually draw the ternary axis
        ax[2].plot([-0.5, 0.5], [1 / 2 / (3**0.5), 1 / 2 / (3**0.5)], color="black")
        ax[2].plot([-0.5, 0], [1 / 2 / (3**0.5), -1 / (3**0.5)], color="black")
        ax[2].plot([0, 0.5], [-1 / (3**0.5), 1 / 2 / (3**0.5)], color="black")

        tick_width = 0.02
        text_offset = 0.02

        # top line ticks
        # ax[2].plot([-0.5, -0.5], [1/2/(3 ** 0.5), 1/2/(3 ** 0.5) + tick_width], color='black')
        # ax[2].text(-0.5 - text_offset, 1/2/(3 ** 0.5) + tick_width, '0.0')
        ax[2].plot(
            [-0.3, -0.3],
            [1 / 2 / (3**0.5), 1 / 2 / (3**0.5) + tick_width],
            color="black",
        )
        ax[2].text(-0.3 - text_offset, 1 / 2 / (3**0.5) + tick_width, "0.2")
        ax[2].plot(
            [-0.1, -0.1],
            [1 / 2 / (3**0.5), 1 / 2 / (3**0.5) + tick_width],
            color="black",
        )
        ax[2].text(-0.1 - text_offset, 1 / 2 / (3**0.5) + tick_width, "0.4")
        ax[2].plot(
            [0.1, 0.1],
            [1 / 2 / (3**0.5), 1 / 2 / (3**0.5) + tick_width],
            color="black",
        )
        ax[2].text(0.1 - text_offset, 1 / 2 / (3**0.5) + tick_width, "0.6")
        ax[2].plot(
            [0.3, 0.3],
            [1 / 2 / (3**0.5), 1 / 2 / (3**0.5) + tick_width],
            color="black",
        )
        ax[2].text(0.3 - text_offset, 1 / 2 / (3**0.5) + tick_width, "0.8")
        # ax[2].plot([0.5, 0.5], [1/2/(3 ** 0.5), 1/2/(3 ** 0.5) + tick_width], color='black')
        # ax[2].text(0.5 - text_offset, 1/2/(3 ** 0.5) + tick_width, '1.0')

        # left line
        y_descent = 3**0.5 / 10
        # ax[2].plot([-0.5 - tick_width, -0.5], [1/2/(3 ** 0.5), 1/2/(3 ** 0.5)], color='black')
        # ax[2].text(-0.5 - tick_width - 3 * text_offset, 1/2/(3 ** 0.5), '0.0')
        ax[2].plot(
            [-0.4 - tick_width, -0.4],
            [1 / 2 / (3**0.5) - y_descent, 1 / 2 / (3**0.5) - y_descent],
            color="black",
        )
        ax[2].text(
            -0.4 - tick_width - 3 * text_offset, 1 / 2 / (3**0.5) - y_descent, "0.2"
        )
        ax[2].plot(
            [-0.3 - tick_width, -0.3],
            [1 / 2 / (3**0.5) - 2 * y_descent, 1 / 2 / (3**0.5) - 2 * y_descent],
            color="black",
        )
        ax[2].text(
            -0.3 - tick_width - 3 * text_offset,
            1 / 2 / (3**0.5) - 2 * y_descent,
            "0.4",
        )
        ax[2].plot(
            [-0.2 - tick_width, -0.2],
            [1 / 2 / (3**0.5) - 3 * y_descent, 1 / 2 / (3**0.5) - 3 * y_descent],
            color="black",
        )
        ax[2].text(
            -0.2 - tick_width - 3 * text_offset,
            1 / 2 / (3**0.5) - 3 * y_descent,
            "0.6",
        )
        ax[2].plot(
            [-0.1 - tick_width, -0.1],
            [1 / 2 / (3**0.5) - 4 * y_descent, 1 / 2 / (3**0.5) - 4 * y_descent],
            color="black",
        )
        ax[2].text(
            -0.1 - tick_width - 3 * text_offset,
            1 / 2 / (3**0.5) - 4 * y_descent,
            "0.8",
        )
        # ax[2].plot([-0. - tick_width, -0.], [1/2/(3 ** 0.5) - 5  * y_descent, 1/2/(3 ** 0.5) - 5  * y_descent], color='black')
        # ax[2].text(-0. - tick_width - 3 * text_offset, 1/2/(3 ** 0.5) - 5  * y_descent, '1.0')

        # right line
        y_descent = 3**0.5 / 10
        # ax[2].plot([0.5, tick_width + 0.5], [1/2/(3 ** 0.5), 1/2/(3 ** 0.5)], color='black')
        # ax[2].text(0.5 + 0.75 * text_offset + tick_width, 1/2/(3 ** 0.5), '0.0')
        ax[2].plot(
            [0.4, tick_width + 0.4],
            [1 / 2 / (3**0.5) - y_descent, 1 / 2 / (3**0.5) - y_descent],
            color="black",
        )
        ax[2].text(
            0.4 + 0.75 * text_offset + tick_width, 1 / 2 / (3**0.5) - y_descent, "0.2"
        )
        ax[2].plot(
            [0.3, tick_width + 0.3],
            [1 / 2 / (3**0.5) - 2 * y_descent, 1 / 2 / (3**0.5) - 2 * y_descent],
            color="black",
        )
        ax[2].text(
            0.3 + 0.75 * text_offset + tick_width,
            1 / 2 / (3**0.5) - 2 * y_descent,
            "0.4",
        )
        ax[2].plot(
            [0.2, tick_width + 0.2],
            [1 / 2 / (3**0.5) - 3 * y_descent, 1 / 2 / (3**0.5) - 3 * y_descent],
            color="black",
        )
        ax[2].text(
            0.2 + 0.75 * text_offset + tick_width,
            1 / 2 / (3**0.5) - 3 * y_descent,
            "0.6",
        )
        ax[2].plot(
            [0.1, tick_width + 0.1],
            [1 / 2 / (3**0.5) - 4 * y_descent, 1 / 2 / (3**0.5) - 4 * y_descent],
            color="black",
        )
        ax[2].text(
            0.1 + 0.75 * text_offset + tick_width,
            1 / 2 / (3**0.5) - 4 * y_descent,
            "0.8",
        )
        # ax[2].plot([0., tick_width + 0.], [1/2/(3 ** 0.5) - 5  * y_descent, 1/2/(3 ** 0.5) - 5  * y_descent], color='black')
        # ax[2].text(0. + 0.75 * text_offset + tick_width, 1/2/(3 ** 0.5) - 5  * y_descent, '1.0')

        ax[2].set_xlim([-0.6, 0.6])
        ax[2].set_ylim([-0.7, 0.5])

    else:
        sct1 = ax[0].scatter(L, M, c=colors, cmap=CMAP, vmin=0, vmax=10)
        sct2 = ax[1].scatter(L, M, c=entropy, cmap="cividis")  # ? colormap ?
        sct3 = ax[2].scatter(L, M, c=points, cmap=cmap)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].set_xticks([])
    ax[2].set_yticks([])

    # colorbars
    cb1 = fig.colorbar(sct1, cax=cax1)
    cb1.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(labels):
        cb1.ax.text(
            0.5,
            (2 * j + 1) / 2,
            lab,
            ha="center",
            va="center",
            rotation=270,
            fontsize="small",
        )
    cb1.ax.get_yaxis().labelpad = 15
    cb1.ax.set_ylabel("class prediction", rotation=270)
    cb2 = fig.colorbar(sct2, cax=cax2)
    cb2.ax.set_ylabel("entropy (log10)", rotation=270)
    cb3 = fig.colorbar(sct3, cax=cax3)
    cb3.ax.set_ylabel("log10 volume element", rotation=270)

    tx = plt.suptitle(f"Epoch {epoch}")
    plt.tight_layout()

    # visualize sub image in the flow plot
    if ternary:
        one_img_ax = ax[0].inset_axes([0.05, 0.8, 0.1, 0.1])
        one_img_ax.imshow(img_one.permute(1, 2, 0).detach().cpu().numpy())
        one_img_ax.axis("off")
        two_img_ax = ax[0].inset_axes([0.85, 0.8, 0.1, 0.1])
        two_img_ax.imshow(img_two.permute(1, 2, 0).detach().cpu().numpy())
        two_img_ax.axis("off")
        three_img_ax = ax[0].inset_axes([0.45, 0.05, 0.1, 0.1])
        three_img_ax.imshow(img_three.permute(1, 2, 0).detach().cpu().numpy())
        three_img_ax.axis("off")

        origin_img_ax = ax[0].inset_axes([0.45, 0.55, 0.1, 0.1])
        origin_img_ax.imshow(origin.permute(1, 2, 0).detach().cpu().numpy())
        origin_img_ax.axis("off")

        one_img_ax = ax[1].inset_axes([0.05, 0.8, 0.1, 0.1])
        one_img_ax.imshow(img_one.permute(1, 2, 0).detach().cpu().numpy())
        one_img_ax.axis("off")
        two_img_ax = ax[1].inset_axes([0.85, 0.8, 0.1, 0.1])
        two_img_ax.imshow(img_two.permute(1, 2, 0).detach().cpu().numpy())
        two_img_ax.axis("off")
        three_img_ax = ax[1].inset_axes([0.45, 0.05, 0.1, 0.1])
        three_img_ax.imshow(img_three.permute(1, 2, 0).detach().cpu().numpy())
        three_img_ax.axis("off")

        origin_img_ax = ax[1].inset_axes([0.45, 0.55, 0.1, 0.1])
        origin_img_ax.imshow(origin.permute(1, 2, 0).detach().cpu().numpy())
        origin_img_ax.axis("off")

        one_img_ax = ax[2].inset_axes([0.05, 0.8, 0.1, 0.1])
        one_img_ax.imshow(img_one.permute(1, 2, 0).detach().cpu().numpy())
        one_img_ax.axis("off")
        two_img_ax = ax[2].inset_axes([0.85, 0.8, 0.1, 0.1])
        two_img_ax.imshow(img_two.permute(1, 2, 0).detach().cpu().numpy())
        two_img_ax.axis("off")
        three_img_ax = ax[2].inset_axes([0.45, 0.05, 0.1, 0.1])
        three_img_ax.imshow(img_three.permute(1, 2, 0).detach().cpu().numpy())
        three_img_ax.axis("off")

        origin_img_ax = ax[2].inset_axes([0.45, 0.55, 0.1, 0.1])
        origin_img_ax.imshow(origin.permute(1, 2, 0).detach().cpu().numpy())
        origin_img_ax.axis("off")
    else:
        one_img_ax = ax[0].inset_axes([0.2, 0.64, 0.1, 0.1])
        one_img_ax.imshow(img_one.permute(1, 2, 0).detach().cpu().numpy())
        one_img_ax.axis("off")
        two_img_ax = ax[0].inset_axes([0.7, 0.64, 0.1, 0.1])
        two_img_ax.imshow(img_two.permute(1, 2, 0).detach().cpu().numpy())
        two_img_ax.axis("off")
        three_img_ax = ax[0].inset_axes([0.45, 0.21, 0.1, 0.1])
        three_img_ax.imshow(img_three.permute(1, 2, 0).detach().cpu().numpy())
        three_img_ax.axis("off")

        origin_img_ax = ax[0].inset_axes([0.45, 0.45, 0.1, 0.1])
        origin_img_ax.imshow(origin.permute(1, 2, 0).detach().cpu().numpy())
        origin_img_ax.axis("off")

        one_img_ax = ax[1].inset_axes([0.2, 0.64, 0.1, 0.1])
        one_img_ax.imshow(img_one.permute(1, 2, 0).detach().cpu().numpy())
        one_img_ax.axis("off")
        two_img_ax = ax[1].inset_axes([0.7, 0.64, 0.1, 0.1])
        two_img_ax.imshow(img_two.permute(1, 2, 0).detach().cpu().numpy())
        two_img_ax.axis("off")
        three_img_ax = ax[1].inset_axes([0.45, 0.21, 0.1, 0.1])
        three_img_ax.imshow(img_three.permute(1, 2, 0).detach().cpu().numpy())
        three_img_ax.axis("off")

        origin_img_ax = ax[1].inset_axes([0.45, 0.45, 0.1, 0.1])
        origin_img_ax.imshow(origin.permute(1, 2, 0).detach().cpu().numpy())
        origin_img_ax.axis("off")

        one_img_ax = ax[2].inset_axes([0.2, 0.64, 0.1, 0.1])
        one_img_ax.imshow(img_one.permute(1, 2, 0).detach().cpu().numpy())
        one_img_ax.axis("off")
        two_img_ax = ax[2].inset_axes([0.7, 0.64, 0.1, 0.1])
        two_img_ax.imshow(img_two.permute(1, 2, 0).detach().cpu().numpy())
        two_img_ax.axis("off")
        three_img_ax = ax[2].inset_axes([0.45, 0.21, 0.1, 0.1])
        three_img_ax.imshow(img_three.permute(1, 2, 0).detach().cpu().numpy())
        three_img_ax.axis("off")

        origin_img_ax = ax[2].inset_axes([0.45, 0.45, 0.1, 0.1])
        origin_img_ax.imshow(origin.permute(1, 2, 0).detach().cpu().numpy())
        origin_img_ax.axis("off")

    # ax.set_xlim(L.min() - 0.05, L.max() + 0.05)
    # ax.set_ylim(-100, 0)


# ===================== videos ======================
def makevid_digits(
    width,
    frms,
    epochs,
    eff_vols,
    predictions,
    img_left,
    img_right,
    img_mid,
    fig,
    ax,
):
    """for mnist"""
    fig.set_tight_layout(True) 
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", "5%", "2%")

    colors = predictions[0].detach().cpu().numpy()
    points = eff_vols[0].detach().cpu().numpy()
    sct = ax.scatter(
        list(range(len(points))), points, c=colors, cmap=CMAP, vmin=0, vmax=10
    )

    # colorbar
    cb = fig.colorbar(sct, cax=cax)
    cb.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(range(10)):
        cb.ax.text(0.5, (2 * j + 1) / 2, lab, ha="center", va="center")
    cb.ax.get_yaxis().labelpad = 15
    cb.ax.set_ylabel("digit prediction", rotation=270)

    ax.set_title("Epoch 0")

    # visualize sub image
    img_size = int(len(img_left) ** (1 / 2))
    left_img_ax = ax.inset_axes([0, 0, 0.1, 0.1])
    left_img_ax.imshow(img_left.reshape(img_size, img_size).detach().cpu().numpy())
    left_img_ax.axis("off")
    mid_img_ax = ax.inset_axes([0.45, 0, 0.1, 0.1])
    mid_img_ax.imshow(img_mid.reshape(img_size, img_size).detach().cpu().numpy())
    mid_img_ax.axis("off")
    right_img_ax = ax.inset_axes([0.9, 0, 0.1, 0.1])
    right_img_ax.imshow(img_right.reshape(img_size, img_size).detach().cpu().numpy())
    right_img_ax.axis("off")

    # ax.set_xlim(L.min() - 0.05, L.max() + 0.05)
    # ax.set_ylim(-100, 0)

    # determine range
    final_epoch_points = eff_vols[-1].detach().cpu().numpy()
    y_range = (
        final_epoch_points.max() - final_epoch_points.min()
    )  # the range is fixed across epochs
    cur_y_mid = (points.max() + points.min()) / 2
    y_lim_max = cur_y_mid + y_range / 2
    y_lim_min = cur_y_mid - y_range / 2
    ax.set_ylim(y_lim_min, y_lim_max)

    def animate(frame_num, data, sct):

        # log effective volume element
        points = data[frame_num].detach().cpu().numpy()
        # vmax = np.max(colors)
        # vmin = np.min(colors)

        # if vmin > 0:
        #     sct = ax.scatter(L, M, norm=clr.LogNorm(), c=colors)
        #     fig.colorbar(sct, cax=cax)
        # else:
        #     sct = ax.scatter(L, M, c=colors)
        #     fig.colorbar(sct, cax=cax)
        ax.clear()
        colors = predictions[frame_num]  # new predictions
        sct = ax.scatter(
            list(range(len(points))), points, c=colors, cmap=CMAP, vmin=0, vmax=10
        )

        ax.set_title("Epoch {}".format(int(epochs / frms) * frame_num))
        ax.set_ylabel("log10 vol element")
        ax.set_xlabel(r"$t$")

        # set ylimit
        cur_y_mid = (points.max() + points.min()) / 2
        y_lim_max = cur_y_mid + y_range / 2
        y_lim_min = cur_y_mid - y_range / 2
        ax.set_ylim(y_lim_min, y_lim_max)

        # visualize sub image
        left_img_ax = ax.inset_axes([0, 0, 0.1, 0.1])
        left_img_ax.imshow(img_left.reshape(img_size, img_size).detach().cpu().numpy())
        left_img_ax.axis("off")
        mid_img_ax = ax.inset_axes([0.45, 0, 0.1, 0.1])
        mid_img_ax.imshow(img_mid.reshape(img_size, img_size).detach().cpu().numpy())
        mid_img_ax.axis("off")
        right_img_ax = ax.inset_axes([0.9, 0, 0.1, 0.1])
        right_img_ax.imshow(
            img_right.reshape(img_size, img_size).detach().cpu().numpy()
        )
        right_img_ax.axis("off")

        returns = [sct]

        return tuple(returns)

    anim1 = FuncAnimation(
        fig, animate, frames=frms + 1, interval=500, fargs=(eff_vols, sct)
    )

    return anim1


def makevid_digits_plane(
    width,
    frms,
    epochs,
    l,
    y,
    eff_vols,
    predictions,
    img_one,
    img_two,
    img_three,
    origin,
    fig,
    ax,
    ternary=False,
    cmap="viridis",
):
    """for mnist"""
    div1 = make_axes_locatable(ax[0])
    cax1 = div1.append_axes("right", "5%", "2%")
    div2 = make_axes_locatable(ax[1])
    cax2 = div2.append_axes("right", "5%", "2%")

    L, M = np.meshgrid(
        l.detach().cpu().numpy(), y.detach().cpu().numpy(), indexing="ij"
    )
    colors = predictions[0].detach().cpu().numpy()
    points = eff_vols[0].detach().cpu().numpy()

    if ternary:  # only visualize the convex hull of the anchor points
        mask = (
            (M < 1 / (2 * 3**0.5))
            & (M > 3**0.5 * L - 1 / 3**0.5)
            & (M > -(3**0.5) * L - 1 / 3**0.5)
        )
        L_masked, M_masked = np.ma.array(L, mask=~mask), np.ma.array(M, mask=~mask)
        # L_masked[~mask] = np.nan
        # M_masked[~mask] = np.nan
        points_masked = np.ma.array(points.reshape(*L_masked.shape), mask=~mask)

        sct1 = ax[0].scatter(L_masked, M_masked, c=colors, cmap=CMAP, vmin=0, vmax=10)
        sct2 = ax[1].contourf(L_masked, M_masked, points_masked, levels=7)

        # ===================== axes 1 ======================
        # manually draw the ternary axis
        ax[0].plot([-0.5, 0.5], [1 / 2 / (3**0.5), 1 / 2 / (3**0.5)], color="black")
        ax[0].plot([-0.5, 0], [1 / 2 / (3**0.5), -1 / (3**0.5)], color="black")
        ax[0].plot([0, 0.5], [-1 / (3**0.5), 1 / 2 / (3**0.5)], color="black")

        tick_width = 0.02
        text_offset = 0.02

        # top line ticks
        # ax[0].plot([-0.5, -0.5], [1/2/(3 ** 0.5), 1/2/(3 ** 0.5) + tick_width], color='black')
        # ax[0].text(-0.5 - text_offset, 1/2/(3 ** 0.5) + tick_width, '0.0')
        ax[0].plot(
            [-0.3, -0.3],
            [1 / 2 / (3**0.5), 1 / 2 / (3**0.5) + tick_width],
            color="black",
        )
        ax[0].text(-0.3 - text_offset, 1 / 2 / (3**0.5) + tick_width, "0.2")
        ax[0].plot(
            [-0.1, -0.1],
            [1 / 2 / (3**0.5), 1 / 2 / (3**0.5) + tick_width],
            color="black",
        )
        ax[0].text(-0.1 - text_offset, 1 / 2 / (3**0.5) + tick_width, "0.4")
        ax[0].plot(
            [0.1, 0.1],
            [1 / 2 / (3**0.5), 1 / 2 / (3**0.5) + tick_width],
            color="black",
        )
        ax[0].text(0.1 - text_offset, 1 / 2 / (3**0.5) + tick_width, "0.6")
        ax[0].plot(
            [0.3, 0.3],
            [1 / 2 / (3**0.5), 1 / 2 / (3**0.5) + tick_width],
            color="black",
        )
        ax[0].text(0.3 - text_offset, 1 / 2 / (3**0.5) + tick_width, "0.8")
        # ax[0].plot([0.5, 0.5], [1/2/(3 ** 0.5), 1/2/(3 ** 0.5) + tick_width], color='black')
        # ax[0].text(0.5 - text_offset, 1/2/(3 ** 0.5) + tick_width, '1.0')

        # left line
        y_descent = 3**0.5 / 10
        # ax[0].plot([-0.5 - tick_width, -0.5], [1/2/(3 ** 0.5), 1/2/(3 ** 0.5)], color='black')
        # ax[0].text(-0.5 - tick_width - 3 * text_offset, 1/2/(3 ** 0.5), '0.0')
        ax[0].plot(
            [-0.4 - tick_width, -0.4],
            [1 / 2 / (3**0.5) - y_descent, 1 / 2 / (3**0.5) - y_descent],
            color="black",
        )
        ax[0].text(
            -0.4 - tick_width - 3 * text_offset, 1 / 2 / (3**0.5) - y_descent, "0.2"
        )
        ax[0].plot(
            [-0.3 - tick_width, -0.3],
            [1 / 2 / (3**0.5) - 2 * y_descent, 1 / 2 / (3**0.5) - 2 * y_descent],
            color="black",
        )
        ax[0].text(
            -0.3 - tick_width - 3 * text_offset,
            1 / 2 / (3**0.5) - 2 * y_descent,
            "0.4",
        )
        ax[0].plot(
            [-0.2 - tick_width, -0.2],
            [1 / 2 / (3**0.5) - 3 * y_descent, 1 / 2 / (3**0.5) - 3 * y_descent],
            color="black",
        )
        ax[0].text(
            -0.2 - tick_width - 3 * text_offset,
            1 / 2 / (3**0.5) - 3 * y_descent,
            "0.6",
        )
        ax[0].plot(
            [-0.1 - tick_width, -0.1],
            [1 / 2 / (3**0.5) - 4 * y_descent, 1 / 2 / (3**0.5) - 4 * y_descent],
            color="black",
        )
        ax[0].text(
            -0.1 - tick_width - 3 * text_offset,
            1 / 2 / (3**0.5) - 4 * y_descent,
            "0.8",
        )
        # ax[0].plot([-0. - tick_width, -0.], [1/2/(3 ** 0.5) - 5  * y_descent, 1/2/(3 ** 0.5) - 5  * y_descent], color='black')
        # ax[0].text(-0. - tick_width - 3 * text_offset, 1/2/(3 ** 0.5) - 5  * y_descent, '1.0')

        # right line
        y_descent = 3**0.5 / 10
        # ax[0].plot([0.5, tick_width + 0.5], [1/2/(3 ** 0.5), 1/2/(3 ** 0.5)], color='black')
        # ax[0].text(0.5 + 0.75 * text_offset + tick_width, 1/2/(3 ** 0.5), '0.0')
        ax[0].plot(
            [0.4, tick_width + 0.4],
            [1 / 2 / (3**0.5) - y_descent, 1 / 2 / (3**0.5) - y_descent],
            color="black",
        )
        ax[0].text(
            0.4 + 0.75 * text_offset + tick_width, 1 / 2 / (3**0.5) - y_descent, "0.2"
        )
        ax[0].plot(
            [0.3, tick_width + 0.3],
            [1 / 2 / (3**0.5) - 2 * y_descent, 1 / 2 / (3**0.5) - 2 * y_descent],
            color="black",
        )
        ax[0].text(
            0.3 + 0.75 * text_offset + tick_width,
            1 / 2 / (3**0.5) - 2 * y_descent,
            "0.4",
        )
        ax[0].plot(
            [0.2, tick_width + 0.2],
            [1 / 2 / (3**0.5) - 3 * y_descent, 1 / 2 / (3**0.5) - 3 * y_descent],
            color="black",
        )
        ax[0].text(
            0.2 + 0.75 * text_offset + tick_width,
            1 / 2 / (3**0.5) - 3 * y_descent,
            "0.6",
        )
        ax[0].plot(
            [0.1, tick_width + 0.1],
            [1 / 2 / (3**0.5) - 4 * y_descent, 1 / 2 / (3**0.5) - 4 * y_descent],
            color="black",
        )
        ax[0].text(
            0.1 + 0.75 * text_offset + tick_width,
            1 / 2 / (3**0.5) - 4 * y_descent,
            "0.8",
        )
        # ax[0].plot([0., tick_width + 0.], [1/2/(3 ** 0.5) - 5  * y_descent, 1/2/(3 ** 0.5) - 5  * y_descent], color='black')
        # ax[0].text(0. + 0.75 * text_offset + tick_width, 1/2/(3 ** 0.5) - 5  * y_descent, '1.0')

        ax[0].set_xlim([-0.6, 0.6])
        ax[0].set_ylim([-0.7, 0.5])

        # ===================== axes 2 ======================
        # manually draw the ternary axis
        ax[1].plot([-0.5, 0.5], [1 / 2 / (3**0.5), 1 / 2 / (3**0.5)], color="black")
        ax[1].plot([-0.5, 0], [1 / 2 / (3**0.5), -1 / (3**0.5)], color="black")
        ax[1].plot([0, 0.5], [-1 / (3**0.5), 1 / 2 / (3**0.5)], color="black")

        tick_width = 0.02
        text_offset = 0.02

        # top line ticks
        # ax[1].plot([-0.5, -0.5], [1/2/(3 ** 0.5), 1/2/(3 ** 0.5) + tick_width], color='black')
        # ax[1].text(-0.5 - text_offset, 1/2/(3 ** 0.5) + tick_width, '0.0')
        ax[1].plot(
            [-0.3, -0.3],
            [1 / 2 / (3**0.5), 1 / 2 / (3**0.5) + tick_width],
            color="black",
        )
        ax[1].text(-0.3 - text_offset, 1 / 2 / (3**0.5) + tick_width, "0.2")
        ax[1].plot(
            [-0.1, -0.1],
            [1 / 2 / (3**0.5), 1 / 2 / (3**0.5) + tick_width],
            color="black",
        )
        ax[1].text(-0.1 - text_offset, 1 / 2 / (3**0.5) + tick_width, "0.4")
        ax[1].plot(
            [0.1, 0.1],
            [1 / 2 / (3**0.5), 1 / 2 / (3**0.5) + tick_width],
            color="black",
        )
        ax[1].text(0.1 - text_offset, 1 / 2 / (3**0.5) + tick_width, "0.6")
        ax[1].plot(
            [0.3, 0.3],
            [1 / 2 / (3**0.5), 1 / 2 / (3**0.5) + tick_width],
            color="black",
        )
        ax[1].text(0.3 - text_offset, 1 / 2 / (3**0.5) + tick_width, "0.8")
        # ax[1].plot([0.5, 0.5], [1/2/(3 ** 0.5), 1/2/(3 ** 0.5) + tick_width], color='black')
        # ax[1].text(0.5 - text_offset, 1/2/(3 ** 0.5) + tick_width, '1.0')

        # left line
        y_descent = 3**0.5 / 10
        # ax[1].plot([-0.5 - tick_width, -0.5], [1/2/(3 ** 0.5), 1/2/(3 ** 0.5)], color='black')
        # ax[1].text(-0.5 - tick_width - 3 * text_offset, 1/2/(3 ** 0.5), '0.0')
        ax[1].plot(
            [-0.4 - tick_width, -0.4],
            [1 / 2 / (3**0.5) - y_descent, 1 / 2 / (3**0.5) - y_descent],
            color="black",
        )
        ax[1].text(
            -0.4 - tick_width - 3 * text_offset, 1 / 2 / (3**0.5) - y_descent, "0.2"
        )
        ax[1].plot(
            [-0.3 - tick_width, -0.3],
            [1 / 2 / (3**0.5) - 2 * y_descent, 1 / 2 / (3**0.5) - 2 * y_descent],
            color="black",
        )
        ax[1].text(
            -0.3 - tick_width - 3 * text_offset,
            1 / 2 / (3**0.5) - 2 * y_descent,
            "0.4",
        )
        ax[1].plot(
            [-0.2 - tick_width, -0.2],
            [1 / 2 / (3**0.5) - 3 * y_descent, 1 / 2 / (3**0.5) - 3 * y_descent],
            color="black",
        )
        ax[1].text(
            -0.2 - tick_width - 3 * text_offset,
            1 / 2 / (3**0.5) - 3 * y_descent,
            "0.6",
        )
        ax[1].plot(
            [-0.1 - tick_width, -0.1],
            [1 / 2 / (3**0.5) - 4 * y_descent, 1 / 2 / (3**0.5) - 4 * y_descent],
            color="black",
        )
        ax[1].text(
            -0.1 - tick_width - 3 * text_offset,
            1 / 2 / (3**0.5) - 4 * y_descent,
            "0.8",
        )
        # ax[1].plot([-0. - tick_width, -0.], [1/2/(3 ** 0.5) - 5  * y_descent, 1/2/(3 ** 0.5) - 5  * y_descent], color='black')
        # ax[1].text(-0. - tick_width - 3 * text_offset, 1/2/(3 ** 0.5) - 5  * y_descent, '1.0')

        # right line
        y_descent = 3**0.5 / 10
        # ax[1].plot([0.5, tick_width + 0.5], [1/2/(3 ** 0.5), 1/2/(3 ** 0.5)], color='black')
        # ax[1].text(0.5 + 0.75 * text_offset + tick_width, 1/2/(3 ** 0.5), '0.0')
        ax[1].plot(
            [0.4, tick_width + 0.4],
            [1 / 2 / (3**0.5) - y_descent, 1 / 2 / (3**0.5) - y_descent],
            color="black",
        )
        ax[1].text(
            0.4 + 0.75 * text_offset + tick_width, 1 / 2 / (3**0.5) - y_descent, "0.2"
        )
        ax[1].plot(
            [0.3, tick_width + 0.3],
            [1 / 2 / (3**0.5) - 2 * y_descent, 1 / 2 / (3**0.5) - 2 * y_descent],
            color="black",
        )
        ax[1].text(
            0.3 + 0.75 * text_offset + tick_width,
            1 / 2 / (3**0.5) - 2 * y_descent,
            "0.4",
        )
        ax[1].plot(
            [0.2, tick_width + 0.2],
            [1 / 2 / (3**0.5) - 3 * y_descent, 1 / 2 / (3**0.5) - 3 * y_descent],
            color="black",
        )
        ax[1].text(
            0.2 + 0.75 * text_offset + tick_width,
            1 / 2 / (3**0.5) - 3 * y_descent,
            "0.6",
        )
        ax[1].plot(
            [0.1, tick_width + 0.1],
            [1 / 2 / (3**0.5) - 4 * y_descent, 1 / 2 / (3**0.5) - 4 * y_descent],
            color="black",
        )
        ax[1].text(
            0.1 + 0.75 * text_offset + tick_width,
            1 / 2 / (3**0.5) - 4 * y_descent,
            "0.8",
        )
        # ax[1].plot([0., tick_width + 0.], [1/2/(3 ** 0.5) - 5  * y_descent, 1/2/(3 ** 0.5) - 5  * y_descent], color='black')
        # ax[1].text(0. + 0.75 * text_offset + tick_width, 1/2/(3 ** 0.5) - 5  * y_descent, '1.0')

        ax[1].set_xlim([-0.6, 0.6])
        ax[1].set_ylim([-0.7, 0.5])
    else:
        sct1 = ax[0].scatter(L, M, c=colors, cmap=CMAP, vmin=0, vmax=10)
        sct2 = ax[1].scatter(L, M, c=points, cmap=cmap)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    # colorbars
    cb1 = fig.colorbar(sct1, cax=cax1)
    cb1.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(range(10)):
        cb1.ax.text(0.5, (2 * j + 1) / 2, lab, ha="center", va="center")
    cb1.ax.get_yaxis().labelpad = 15
    cb1.ax.set_ylabel("digit prediction", rotation=270)
    cb2 = fig.colorbar(sct2, cax=cax2)
    cb2.ax.set_ylabel("log volume element", rotation=270)

    tx = plt.suptitle("Epoch 0")
    plt.tight_layout()

    # visualize sub image
    img_size = int(len(img_one) ** (1 / 2))

    # visualize sub image in the flow plot
    if ternary:
        one_img_ax = ax[0].inset_axes([0.05, 0.8, 0.1, 0.1])
        one_img_ax.imshow(img_one.reshape(img_size, img_size).detach().cpu().numpy())
        one_img_ax.axis("off")
        two_img_ax = ax[0].inset_axes([0.85, 0.8, 0.1, 0.1])
        two_img_ax.imshow(img_two.reshape(img_size, img_size).detach().cpu().numpy())
        two_img_ax.axis("off")
        three_img_ax = ax[0].inset_axes([0.45, 0.05, 0.1, 0.1])
        three_img_ax.imshow(
            img_three.reshape(img_size, img_size).detach().cpu().numpy()
        )
        three_img_ax.axis("off")

        origin_img_ax = ax[0].inset_axes([0.45, 0.55, 0.1, 0.1])
        origin_img_ax.imshow(origin.reshape(img_size, img_size).detach().cpu().numpy())
        origin_img_ax.axis("off")

        one_img_ax = ax[1].inset_axes([0.05, 0.8, 0.1, 0.1])
        one_img_ax.imshow(img_one.reshape(img_size, img_size).detach().cpu().numpy())
        one_img_ax.axis("off")
        two_img_ax = ax[1].inset_axes([0.85, 0.8, 0.1, 0.1])
        two_img_ax.imshow(img_two.reshape(img_size, img_size).detach().cpu().numpy())
        two_img_ax.axis("off")
        three_img_ax = ax[1].inset_axes([0.45, 0.05, 0.1, 0.1])
        three_img_ax.imshow(
            img_three.reshape(img_size, img_size).detach().cpu().numpy()
        )
        three_img_ax.axis("off")

        origin_img_ax = ax[1].inset_axes([0.45, 0.55, 0.1, 0.1])
        origin_img_ax.imshow(origin.reshape(img_size, img_size).detach().cpu().numpy())
        origin_img_ax.axis("off")
    else:
        one_img_ax = ax[0].inset_axes([0.2, 0.64, 0.1, 0.1])
        one_img_ax.imshow(img_one.reshape(img_size, img_size).detach().cpu().numpy())
        one_img_ax.axis("off")
        two_img_ax = ax[0].inset_axes([0.7, 0.64, 0.1, 0.1])
        two_img_ax.imshow(img_two.reshape(img_size, img_size).detach().cpu().numpy())
        two_img_ax.axis("off")
        three_img_ax = ax[0].inset_axes([0.45, 0.21, 0.1, 0.1])
        three_img_ax.imshow(
            img_three.reshape(img_size, img_size).detach().cpu().numpy()
        )
        three_img_ax.axis("off")

        origin_img_ax = ax[0].inset_axes([0.45, 0.45, 0.1, 0.1])
        origin_img_ax.imshow(origin.reshape(img_size, img_size).detach().cpu().numpy())
        origin_img_ax.axis("off")

        one_img_ax = ax[1].inset_axes([0.2, 0.64, 0.1, 0.1])
        one_img_ax.imshow(img_one.reshape(img_size, img_size).detach().cpu().numpy())
        one_img_ax.axis("off")
        two_img_ax = ax[1].inset_axes([0.7, 0.64, 0.1, 0.1])
        two_img_ax.imshow(img_two.reshape(img_size, img_size).detach().cpu().numpy())
        two_img_ax.axis("off")
        three_img_ax = ax[1].inset_axes([0.45, 0.21, 0.1, 0.1])
        three_img_ax.imshow(
            img_three.reshape(img_size, img_size).detach().cpu().numpy()
        )
        three_img_ax.axis("off")

        origin_img_ax = ax[1].inset_axes([0.45, 0.45, 0.1, 0.1])
        origin_img_ax.imshow(origin.reshape(img_size, img_size).detach().cpu().numpy())
        origin_img_ax.axis("off")

    # ax.set_xlim(L.min() - 0.05, L.max() + 0.05)
    # ax.set_ylim(-100, 0)

    def animate(frame_num, sct1, sct2):

        # log effective volume element
        colors = predictions[frame_num].detach().cpu().numpy()
        points = eff_vols[frame_num].detach().cpu().numpy()
        # vmax = np.max(colors)
        # vmin = np.min(colors)

        # if vmin > 0:
        #     sct = ax.scatter(L, M, norm=clr.LogNorm(), c=colors)
        #     fig.colorbar(sct, cax=cax)
        # else:
        #     sct = ax.scatter(L, M, c=colors)
        #     fig.colorbar(sct, cax=cax)
        # ax.clear()

        if ternary:
            points_masked = np.ma.array(points.reshape(*L_masked.shape), mask=~mask)

            sct1 = ax[0].scatter(
                L_masked, M_masked, c=colors, cmap=CMAP, vmin=0, vmax=10
            )
            sct2 = ax[1].contourf(L_masked, M_masked, points_masked, levels=7)

        else:
            sct1 = ax[0].scatter(L, M, c=colors, cmap=CMAP, vmin=0, vmax=10)
            sct2 = ax[1].scatter(L, M, c=points, cmap=cmap)

        tx.set_text("Epoch {}".format(int(epochs / frms) * frame_num))

        # colorbars
        cb1 = fig.colorbar(sct1, cax=cax1)
        cb1.ax.get_yaxis().set_ticks([])
        for j, lab in enumerate(range(10)):
            cb1.ax.text(0.5, (2 * j + 1) / 2, lab, ha="center", va="center")
        cb1.ax.get_yaxis().labelpad = 15
        cb1.ax.set_ylabel("digit prediction", rotation=270)
        cb2 = fig.colorbar(sct2, cax=cax2)
        cb2.ax.set_ylabel("log10 volume element", rotation=270)

        returns = [sct1, sct2]

        return tuple(returns)

    anim1 = FuncAnimation(
        fig,
        animate,
        frames=frms + 1,
        interval=500,
        fargs=(sct1, sct2),
    )

    return anim1
