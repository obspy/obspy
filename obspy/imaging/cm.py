# -*- coding: utf-8 -*-
"""
Module for ObsPy's default colormaps.

"Viridis" is matplotlib's new default colormap from version 2.0 onwards and is
based on a design by Eric Firing (@efiring, see
http://thread.gmane.org/gmane.comp.python.matplotlib.devel/13522/focus=13542).

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


viridis = _get_cmap("viridis")
obspy_sequential = viridis
obspy_divergent = get_cmap("RdBu_r")
obspy_divergent_r = get_cmap("RdBu")


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
