import datetime as dt
import numpy as np
import pandas as pd
from functools import wraps
from matplotlib import pyplot as plt
from typing import Union

from matplotlib.transforms import Bbox


def cut_dates(df: Union[pd.DataFrame, pd.Series], dates: str) -> pd.DataFrame:
    """
    Trim the dataframe according to dates.

    It works in the same way as python slices:
    * "date_start:" means all dates after `date_start`
    * ":date_end" means all dates before `date_end`
    * "date_start:date_end" - a slice of dates from `date_start` till `date_end`

    :param df: a DataFrame to cut
    :param dates: a string with `:` as a separator, e.g. "2020-01-15:2020-10-20"
    :return: the DataFrame trimmed according to the dates passed
    """
    try:
        date_start, date_end = dates.split(':')
    except ValueError:
        raise ValueError(f"`dates` is not a valid time range, try something "
                         f"like dates='{df.index[0].date()}:{df.index[-1].date()}'")

    if date_start == '':
        date_start = str(df.index[0])
    if date_end == '':
        date_end = str(dt.date.today())

    return df[date_start:date_end]


def linear_mapping(_from, _to, x):
    """
    Lets use a simple linear function that maps the ends of one scale onto ends
    of another. Obviously, we need that
          f(left_min) -> right_min and
          f(left_max) -> right_max,
    and points in between are mapped proportionally.
    """
    return _to[0] + (x - _from[0]) / (_from[1] - _from[0]) * (_to[1] - _to[0])
    
    
def align_twinx_ticks(ax_left: plt.Axes, ax_right: plt.Axes) -> np.ndarray:
    """
    Returns an array of ticks for the right axis which match ones on the left.

    There's no easy way of aligning ticks nor a good general solution.
    """
    left = ax_left.get_ylim()
    right = ax_right.get_ylim()
    return linear_mapping(left, right, ax_left.get_yticks())


def has_twin(ax: plt.Axes) -> bool:
    """ Returns True if the `ax` axis has a twinned axis """
    for other_ax in ax.figure.axes:
        if other_ax is ax:
            continue
        if other_ax.bbox.bounds == ax.bbox.bounds:
            return True
    return False


def limit_to_smoothed(func, max_ratio=1.5):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ax = func(*args, **kwargs)
        lines, means, points, maxpoint = [], [], [], None
        if has_twin(ax):
            graphs = ax.get_shared_x_axes().get_siblings(ax)
            for graph in graphs:
                for line in graph.lines:
                    lines += [line]
        else:
            # no twin axes, everything is great!
            lines = ax.lines
        for line in lines:
            if "rolling mean" in line._label:
                means += [line]
            else:
                points += [line]
        if means:
            maxpoint = max([np.nanmax(line._y) for line in means])

            _, ylim = ax.get_ylim()
            ax.set_ylim(top=min(ylim, maxpoint * max_ratio), bottom=0)
        return ax
    return wrapper


def uncertain_tail(ax: plt.Axes, data: pd.Series, days: int = 3,
                   color: str = 'gray', linewidth: float = 4, alpha: float = 0.7) -> None:
    """ Take a curve and plt the last `days` with dashed lines which supposed to mean 'uncertain region'
    """
    tail = data.rolling(2 * days, center=True, win_type='gaussian', min_periods=3).mean(std=4)
    ax.plot(tail.index[-days - 1:], tail.values[-days - 1:],
            color=color, linestyle='dashed', linewidth=linewidth, alpha=alpha)


def full_extent(ax, fig, num_subplots=6):
    """Get the full extent of an axes, including axes labels, tick labels, and titles.

    Return the size of bbox to save (in inches)
    """
    # For text objects, we need to draw the figure first, otherwise the extents are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
    items += [ax.xaxis.label, ax.yaxis.label]

    items += [ax]

    bbox = Bbox.union([item.get_window_extent() for item in items])
    if has_twin(ax):
        bbox.x1 += 75   # if there is a second y-axis, extend graph area to the right

    # limit the maximum height of all subplots except from the first one
    max_height = fig.get_size_inches()[1] * fig.dpi / num_subplots
    if ax is not ax.figure.axes[0] and bbox.height > max_height * 1.01:
        bbox.y1 = bbox.y0 + max_height * 1.01

    return bbox.expanded(1., 1.).transformed(fig.dpi_scale_trans.inverted())
