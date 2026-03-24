from matplotlib.patches import Circle
import numpy as np
import matplotlib.pylab as pl


def color_list(n, cmap="jet"):
    #Get cmap from plt.cm
    cmap_ = getattr(pl.cm, cmap)
    #Get colors from cmap
    colors = cmap_(np.linspace(0, 1, n))
    # colors = pl.cm.jet(np.linspace(0, 1, n))
    return colors

def circular_grid(
    ax,
    radius: list | np.ndarray,
    rlabels: bool = True,
    label_angle: float = 45,
    relative_offset: tuple[float, float] = (0.25, 0.25),
    rlabel_fmt: str = "%.1f",
    color: str = "black",
    linewidth: float = 0.5,
    linestyle: str = "--",
):
    if isinstance(radius, list):
        radius = np.array(radius)
    elif not isinstance(radius, np.ndarray):
        raise TypeError("radius must be a list or numpy array.")

    #Leave only positive values and should be unique
    radius = np.unique(radius[radius > 0])

    # Add circular grid lines
    for r in radius:  # Specify the radii for the circles
        circle = Circle((0, 0), r, color=color, linewidth=linewidth, linestyle=linestyle, fill=False)
        ax.add_artist(circle)
    
    if rlabels:
        for r in radius:
            label_x = r * np.cos(np.deg2rad(label_angle)) + relative_offset[0]
            label_y = r * np.sin(np.deg2rad(label_angle)) + relative_offset[1]
            ax.text(
                    label_x, label_y,
                    rlabel_fmt % r,
                    verticalalignment="center",
                    horizontalalignment="center",
                )