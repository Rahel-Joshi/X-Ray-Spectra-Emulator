"""plotting.py

Collection of plotting scripts for result plots
in the ALMaQUST-TNG project

----
The script uses fastkde package for plotting
contours, which can be found at:
https://pypi.org/project/fastkde/

O’Brien, T. A., Kashinath, K., Cavanaugh, N. R., Collins, W. D.
& O’Brien, J. P. A fast and objective multidimensional kernel density
estimation method: fastKDE. Comput. Stat. Data Anal. 101, 148–160 (2016).
<http://dx.doi.org/10.1016/j.csda.2016.02.014>

O’Brien, T. A., Collins, W. D., Rauscher, S. A. & Ringler, T. D.
Reducing the computational cost of the ECF using a nuFFT: A fast and
objective probability density estimation method. Comput. Stat.
Data Anal. 79, 222–234 (2014).
<http://dx.doi.org/10.1016/j.csda.2014.06.002>
---
"""

__version__ = 0.0
__author__ = "Joanna Piotrowska"

# ==================================================================
# imports
# ==================================================================
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


# import support_tools as st
# ==================================================================
# python function definitions
# ==================================================================
def add_colorbar(mappable, ax=None, *args):
    """
    Add colorbar to an existing mappable in the figure
    """
    if ax is None:
        ax = mappable.axes

    axis_divider = make_axes_locatable(ax)
    fig = ax.figure

    if len(args) == 1:
        size = args[0]
        cax = axis_divider.append_axes("right", size=size, pad="2%")
        cbar = fig.colorbar(mappable, cax=cax)

    elif len(args) == 2:
        size = args[0]
        pad = args[1]
        cax = axis_divider.append_axes("right", size=size, pad=pad)
        cbar = fig.colorbar(mappable, cax=cax)

    elif len(args) == 3:
        size = args[0]
        pad = args[1]
        form = args[2]
        cax = axis_divider.append_axes("right", size=size, pad=pad)
        cbar = fig.colorbar(mappable, cax=cax, format=form)

    elif len(args) == 4:
        size = args[0]
        pad = args[1]
        form = args[2]
        ticks = args[3]
        cax = axis_divider.append_axes("right", size=size, pad=pad)
        cbar = fig.colorbar(mappable, cax=cax, format=form, ticks=ticks)

    else:
        cax = axis_divider.append_axes("right", size="3%", pad="2%")
        cbar = fig.colorbar(mappable, cax=cax)

    return cbar


def configure_plots(scheme="white"):
    """
    Setting global Matplotlib settings for plotting lineplots
    """
    # line settings
    rcParams["lines.linewidth"] = 1
    rcParams["lines.markersize"] = 3
    rcParams["errorbar.capsize"] = 0

    # axes linewidth
    rcParams["axes.linewidth"] = 1

    # tick settings
    rcParams["xtick.top"] = True
    rcParams["ytick.right"] = True
    rcParams["xtick.major.size"] = 7
    rcParams["xtick.major.width"] = 1
    rcParams["xtick.minor.width"] = 0.75
    rcParams["xtick.minor.size"] = 4
    rcParams["xtick.direction"] = "in"
    rcParams["ytick.major.size"] = 7
    rcParams["ytick.minor.size"] = 4
    rcParams["ytick.major.width"] = 1
    rcParams["ytick.minor.width"] = 0.75
    rcParams["ytick.direction"] = "in"

    # text settings
    rcParams["mathtext.rm"] = "serif"
    rcParams["font.family"] = "serif"
    rcParams["font.size"] = 12
    rcParams["text.usetex"] = True
    rcParams["axes.titlesize"] = 13
    rcParams["axes.labelsize"] = 12
    rcParams["axes.ymargin"] = 0.5

    # legend
    rcParams["legend.fontsize"] = 12
    rcParams["legend.frameon"] = False

    # grid in plots
    rcParams["grid.linestyle"] = ":"

    # figure settings
    rcParams["figure.figsize"] = 5, 4
    rcParams["figure.dpi"] = 150
    rcParams["savefig.format"] = "pdf"

    # colour config
    if scheme == "white":
        rcParams["figure.facecolor"] = "None"
        rcParams["xtick.color"] = "k"
        rcParams["ytick.color"] = "k"
        rcParams["axes.edgecolor"] = "k"
        rcParams["axes.facecolor"] = "None"
        rcParams["axes.labelcolor"] = "k"
        rcParams["text.color"] = "k"

    elif scheme == "black":
        rcParams["figure.facecolor"] = "None"
        rcParams["xtick.color"] = "w"
        rcParams["ytick.color"] = "w"
        rcParams["axes.edgecolor"] = "w"
        rcParams["axes.facecolor"] = "None"
        rcParams["axes.labelcolor"] = "w"
        rcParams["text.color"] = "w"

    elif scheme == "dark":
        rcParams["figure.facecolor"] = "None"
        rcParams["xtick.color"] = "lightyellow"
        rcParams["ytick.color"] = "lightyellow"
        rcParams["axes.edgecolor"] = "lightyellow"
        rcParams["axes.facecolor"] = "None"
        rcParams["axes.labelcolor"] = "lightyellow"
        rcParams["text.color"] = "lightyellow"

    else:
        print("%s clour scheme not available. Please define manually!" % scheme)


def set_ticks(
    ax,
    xmaj,
    xmin,
    ymaj,
    ymin,
    side=None,
    color="k",
    labelsize=None,
    major_length=7,
    minor_length=4,
):
    """
    Applies my favourite formatting to plot ticks

    Parameters
    -----------
    --- ax: axis instance
    --- xmaj: float, xaxis major locator argument
    --- xmin: float, xaxis minor locator argument
    --- ymaj: float, yaxis major locator argument
    --- ymin: float, yaxis minor locator argument
    """
    if side is not None:
        if side == "right":
            ax.tick_params(
                which="major",
                direction="in",
                length=major_length,
                right=True,
                bottom=False,
                left=False,
                top=False,
                labelleft=False,
                labelright=True,
                labelbottom=False,
                color=color,
                labelsize=labelsize,
            )
            ax.tick_params(
                which="minor",
                direction="in",
                length=minor_length,
                right=True,
                bottom=False,
                left=False,
                top=False,
                labelleft=False,
                labelright=True,
                labelbottom=False,
                color=color,
                labelsize=labelsize,
            )
        elif side == "left":
            ax.tick_params(
                which="major",
                direction="in",
                length=major_length,
                right=False,
                top=True,
                left=True,
                labelleft=True,
                color=color,
                labelsize=labelsize,
            )
            ax.tick_params(
                which="minor",
                direction="in",
                length=minor_length,
                right=False,
                top=True,
                left=True,
                labelleft=True,
                color=color,
                labelsize=labelsize,
            )
    else:
        ax.tick_params(
            which="major",
            direction="in",
            length=major_length,
            right=True,
            top=True,
            color=color,
            labelsize=labelsize,
        )
        ax.tick_params(
            which="minor",
            direction="in",
            length=minor_length,
            right=True,
            top=True,
            color=color,
            labelsize=labelsize,
        )

    ax.xaxis.set_major_locator(MultipleLocator(xmaj))
    ax.xaxis.set_minor_locator(MultipleLocator(xmin))
    ax.yaxis.set_major_locator(MultipleLocator(ymaj))
    ax.yaxis.set_minor_locator(MultipleLocator(ymin))
