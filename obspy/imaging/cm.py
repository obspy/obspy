# -*- coding: utf-8 -*-
"""
Module for ObsPy's default colormaps.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import inspect
import os

import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap


def _get_cmap(name):
    """
    Load a :class:`~matplotlib.colors.LinearSegmentedColormap` from
    `segmentdata` dictionary saved as numpy compressed binary data.

    :type name: str
    :param name: Name of colormap to load, same as filename in
        `obspy/imaging/data` without `.npz` file suffix.
    :rtype: :class:`~matplotlib.colors.LinearSegmentedColormap`
    """
    directory = os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))
    directory = os.path.join(directory, "data")
    if name.endswith(".npz"):
        name = name.rsplit(".npz", 1)[0]
    filename = os.path.join(directory, name + ".npz")
    cmap = LinearSegmentedColormap(name=name,
                                   segmentdata=dict(np.load(filename)))
    return cmap


obspy_BuGnYl = _get_cmap("obspy_BuGnYl")
obspy_sequential = obspy_BuGnYl
obspy_divergent = get_cmap("RdYlBu_r")


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
