from io import StringIO
import pandas as pd
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np


def csv_precip16():
    """
    # Converted from MeteoSwiss NCL library

    # number of colors in table
    ncolors = 17

    r,g,b
    255, 255, 255
    214, 226, 255
    181, 201, 255
    142, 178, 255
    127, 150, 255
    99, 112, 247
    0,  99, 255
    0, 150, 150
    0, 198,  51
    99, 255,   0
    150, 255,   0
    198, 255,  51
    255, 255,   0
    255, 198,   0
    255, 160,   0
    255, 124,   0
    255,  25,   0
    """


def get_colors():
    f = StringIO(csv_precip16.__doc__)
    return pd.read_csv(f, header=3, names=("r", "g", "b"))


def build_cmap():
    c = get_colors()
    colors = np.array([c.r, c.g, c.b], dtype="f4").T/255
    cmap = ListedColormap(colors)
    return cmap


def split(color):
    n = len(color)
    x = np.linspace(0, 1, n)
    y = color/255
    # return [(x[i], y[i], 1-(1-y[i])/2) for i in range(n)]
    return [(x[i], y[i], y[i]) for i in range(n)]


def build_linear_cmap():
    c = get_colors()
    r = split(c.r)
    return {
        "red": split(c.r),
        "green": split(c.g),
        "blue": split(c.b)
    }


precip16_discrete = build_cmap()
precip16 = LinearSegmentedColormap("precip16", build_linear_cmap())
