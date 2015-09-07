# -*- coding: utf-8 -*-
"""
Module for ObsPy's default colormaps.

"Viridis" is matplotlib's new default colormap from version 2.0 onwards and is
based on a design by Eric Firing (@efiring, see
http://thread.gmane.org/gmane.comp.python.matplotlib.devel/13522/focus=13542).

Colormap of PQLX for PPSD is available as :const:`obspy.imaging.cm.pqlx`.

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


def _get_cmap(name, lut=None, reverse=False):
    """
    Load a :class:`~matplotlib.colors.LinearSegmentedColormap` from
    `segmentdata` dictionary saved as numpy compressed binary data.

    :type name: str
    :param name: Name of colormap to load, same as filename in
        `obspy/imaging/data` without `.npz` file suffix.
    :type lut: int
    :param lut: Specifies the number of discrete color values in the colormap.
        `None` to use matplotlib default value (continuous colormap).
    :type reverse: bool
    :param reverse: Whether to return the specified colormap reverted.
    :rtype: :class:`~matplotlib.colors.LinearSegmentedColormap`
    """
    directory = os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))
    directory = os.path.join(directory, "data")
    if name.endswith(".npz"):
        name = name.rsplit(".npz", 1)[0]
    filename = os.path.join(directory, name + ".npz")
    data = dict(np.load(filename))
    if reverse:
        data_r = {}
        for key, val in data.items():
            # copied from matplotlib source, cm.py@f7a578656abc2b2c13 line 47
            data_r[key] = [(1.0 - x, y1, y0) for x, y0, y1 in reversed(val)]
        data = data_r
    kwargs = lut and {"N": lut} or {}
    cmap = LinearSegmentedColormap(name=name, segmentdata=data, **kwargs)
    return cmap

viridis = _get_cmap("viridis")
viridis_r = _get_cmap("viridis", reverse=True)
obspy_sequential = viridis
obspy_sequential_r = viridis_r
obspy_divergent = get_cmap("RdBu_r")
obspy_divergent_r = get_cmap("RdBu")
pqlx = _get_cmap("pqlx")


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
