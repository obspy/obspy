# -*- coding: utf-8 -*-
# -----------------------------------------------
# Filename: mopad_wrapper.py
#  Purpose: Wrapper for mopad
#   Author: Tobias Megies, Moritz Beyreuther
#    Email: megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2012 ObsPy Development Team
# -----------------------------------------------
"""
ObsPy wrapper to the *Moment tensor Plotting and Decomposition tool* (MoPaD)
written by Lars Krieger and Sebastian Heimann.

.. seealso:: [Krieger2012]_

.. warning:: The MoPaD wrapper does not yet provide the full functionality of
    MoPaD. Please consider using the command line script ``obspy-mopad`` for
    now if you need the full power of MoPaD.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import numpy as np
from matplotlib import patches, transforms
import matplotlib.collections as mpl_collections
from obspy.imaging.scripts.mopad import BeachBall as mopad_BeachBall
from obspy.imaging.scripts.mopad import MomentTensor as mopad_MomentTensor
from obspy.imaging.scripts.mopad import epsilon
from obspy.imaging.beachball import xy2patch


# seems the base system we (gmt) are using is called "USE" in mopad
KWARG_MAP = {
    'size': ['plot_size', 'plot_aux_plot_size'],
    'linewidth': ['plot_nodalline_width', 'plot_outerline_width'],
    'facecolor': ['plot_tension_colour'],
    'edgecolor': ['plot_outerline_colour'],
    'bgcolor': [],
    'alpha': ['plot_total_alpha'],
    'width': [],
    'outfile': ['plot_outfile'],
    'format': ['plot_outfile_format'],
    'nofill': ['plot_only_lines']
}


def Beach(fm, linewidth=2, facecolor='b', bgcolor='w', edgecolor='k',
          alpha=1.0, xy=(0, 0), width=200, size=100, nofill=False,
          zorder=100, mopad_basis='USE', axes=None):
    """
    Return a beach ball as a collection which can be connected to an
    current matplotlib axes instance (ax.add_collection). Based on MoPaD.

    S1, D1, and R1, the strike, dip and rake of one of the focal planes, can
    be vectors of multiple focal mechanisms.

    :param fm: Focal mechanism that is either number of mechanisms (NM) by 3
        (strike, dip, and rake) or NM x 6 (M11, M22, M33, M12, M13, M23 - the
        six independent components of the moment tensor, where the coordinate
        system is 1,2,3 = Up,South,East which equals r,theta,phi). The strike
        is of the first plane, clockwise relative to north.
        The dip is of the first plane, defined clockwise and perpendicular to
        strike, relative to horizontal such that 0 is horizontal and 90 is
        vertical. The rake is of the first focal plane solution. 90 moves the
        hanging wall up-dip (thrust), 0 moves it in the strike direction
        (left-lateral), -90 moves it down-dip (normal), and 180 moves it
        opposite to strike (right-lateral).
    :param facecolor: Color to use for quadrants of tension; can be a string,
        e.g. ``'r'``, ``'b'`` or three component color vector, [R G B].
        Defaults to ``'b'`` (blue).
    :param bgcolor: The background color. Defaults to ``'w'`` (white).
    :param edgecolor: Color of the edges. Defaults to ``'k'`` (black).
    :param alpha: The alpha level of the beach ball. Defaults to ``1.0``
        (opaque).
    :param xy: Origin position of the beach ball as tuple. Defaults to
        ``(0, 0)``.
    :type width: int
    :param width: Symbol size of beach ball. Defaults to ``200``.
    :param size: Controls the number of interpolation points for the
        curves. Minimum is automatically set to ``100``.
    :param nofill: Do not fill the beach ball, but only plot the planes.
    :param zorder: Set zorder. Artists with lower zorder values are drawn
        first.
    :param mopad_basis: The basis system. Defaults to ``'USE'``. See the
        `Supported Basis Systems`_ section below for a full list of supported
        systems.
    :type axes: :class:`matplotlib.axes.Axes`
    :param axes: Used to make beach balls circular on non-scaled axes. Also
        maintains the aspect ratio when resizing the figure. Will not add
        the returned collection to the axes instance.

    .. rubric:: _`Supported Basis Systems`

    ========= =================== =============================================
    Short     Basis vectors       Usage
    ========= =================== =============================================
    ``'NED'`` North, East, Down   Jost and Herrmann 1989
    ``'USE'`` Up, South, East     Global CMT Catalog, Larson et al. 2010
    ``'XYZ'`` East, North, Up     General formulation, Jost and Herrmann 1989
    ``'RT'``  Radial, Transverse, psmeca (GMT), Wessel and Smith 1999
              Tangential
    ``'NWU'`` North, West, Up     Stein and Wysession 2003
    ========= =================== =============================================
    """
    # initialize beachball
    mt = mopad_MomentTensor(fm, system=mopad_basis)
    bb = mopad_BeachBall(mt, npoints=size)
    bb._setup_BB(unit_circle=False)

    # extract the coordinates and colors of the lines
    radius = width / 2.0
    neg_nodalline = bb._nodalline_negative_final_US
    pos_nodalline = bb._nodalline_positive_final_US
    tension_colour = facecolor
    pressure_colour = bgcolor

    if nofill:
        tension_colour = 'none'
        pressure_colour = 'none'

    # based on mopads _setup_plot_US() function
    # collect patches for the selection
    coll = [None, None, None]
    coll[0] = patches.Circle(xy, radius=radius)
    coll[1] = xy2patch(neg_nodalline[0, :], neg_nodalline[1, :], radius, xy)
    coll[2] = xy2patch(pos_nodalline[0, :], pos_nodalline[1, :], radius, xy)

    # set the color of the three parts
    fc = [None, None, None]
    if bb._plot_clr_order > 0:
        fc[0] = pressure_colour
        fc[1] = tension_colour
        fc[2] = tension_colour
        if bb._plot_curve_in_curve != 0:
            fc[0] = tension_colour
            if bb._plot_curve_in_curve < 1:
                fc[1] = pressure_colour
                fc[2] = tension_colour
            else:
                coll = [coll[i] for i in (0, 2, 1)]
                fc[1] = pressure_colour
                fc[2] = tension_colour
    else:
        fc[0] = tension_colour
        fc[1] = pressure_colour
        fc[2] = pressure_colour
        if bb._plot_curve_in_curve != 0:
            fc[0] = pressure_colour
            if bb._plot_curve_in_curve < 1:
                fc[1] = tension_colour
                fc[2] = pressure_colour
            else:
                coll = [coll[i] for i in (0, 2, 1)]
                fc[1] = tension_colour
                fc[2] = pressure_colour

    if bb._pure_isotropic:
        if abs(np.trace(bb._M)) > epsilon:
            # use the circle as the most upper layer
            coll = [coll[0]]
            if bb._plot_clr_order < 0:
                fc = [tension_colour]
            else:
                fc = [pressure_colour]

    # transform the patches to a path collection and set
    # the appropriate attributes
    collection = mpl_collections.PatchCollection(coll, match_original=False)
    collection.set_facecolors(fc)
    # Use the given axes to maintain the aspect ratio of beachballs on figure
    # resize.
    if axes is not None:
        # This is what holds the aspect ratio (but breaks the positioning)
        collection.set_transform(transforms.IdentityTransform())
        # Next is a dirty hack to fix the positioning:
        # 1. Need to bring the all patches to the origin (0, 0).
        for p in collection._paths:
            p.vertices -= xy
        # 2. Then use the offset property of the collection to position the
        # patches
        collection.set_offsets(xy)
        collection._transOffset = axes.transData
    collection.set_edgecolors(edgecolor)
    collection.set_alpha(alpha)
    collection.set_linewidth(linewidth)
    collection.set_zorder(zorder)
    return collection


def Beachball(fm, linewidth=2, facecolor='b', bgcolor='w', edgecolor='k',
              alpha=1.0, xy=(0, 0), width=200, size=100, nofill=False,
              zorder=100, mopad_basis='USE', outfile=None, format=None,
              fig=None):
    """
    Draws a beach ball diagram of an earthquake focal mechanism. Based on
    MoPaD.

    S1, D1, and R1, the strike, dip and rake of one of the focal planes, can
    be vectors of multiple focal mechanisms.

    :param fm: Focal mechanism that is either number of mechanisms (NM) by 3
        (strike, dip, and rake) or NM x 6 (M11, M22, M33, M12, M13, M23 - the
        six independent components of the moment tensor, where the coordinate
        system is 1,2,3 = Up,South,East which equals r,theta,phi). The strike
        is of the first plane, clockwise relative to north.
        The dip is of the first plane, defined clockwise and perpendicular to
        strike, relative to horizontal such that 0 is horizontal and 90 is
        vertical. The rake is of the first focal plane solution. 90 moves the
        hanging wall up-dip (thrust), 0 moves it in the strike direction
        (left-lateral), -90 moves it down-dip (normal), and 180 moves it
        opposite to strike (right-lateral).
    :param facecolor: Color to use for quadrants of tension; can be a string,
        e.g. ``'r'``, ``'b'`` or three component color vector, [R G B].
        Defaults to ``'b'`` (blue).
    :param bgcolor: The background color. Defaults to ``'w'`` (white).
    :param edgecolor: Color of the edges. Defaults to ``'k'`` (black).
    :param alpha: The alpha level of the beach ball. Defaults to ``1.0``
        (opaque).
    :param xy: Origin position of the beach ball as tuple. Defaults to
        ``(0, 0)``.
    :type width: int
    :param width: Symbol size of beach ball. Defaults to ``200``.
    :param size: Controls the number of interpolation points for the
        curves. Minimum is automatically set to ``100``.
    :param nofill: Do not fill the beach ball, but only plot the planes.
    :param zorder: Set zorder. Artists with lower zorder values are drawn
        first.
    :param mopad_basis: The basis system. Defaults to ``'USE'``. See the
        `Supported Basis Systems`_ section below for a full list of supported
        systems.
    :param outfile: Output file string. Also used to automatically
        determine the output format. Supported file formats depend on your
        matplotlib backend. Most backends support png, pdf, ps, eps and
        svg. Defaults to ``None``.
    :param format: Format of the graph picture. If no format is given the
        outfile parameter will be used to try to automatically determine
        the output format. If no format is found it defaults to png output.
        If no outfile is specified but a format is, than a binary
        imagestring will be returned.
        Defaults to ``None``.
    :param fig: Give an existing figure instance to plot into. New Figure if
        set to ``None``.

    .. rubric:: _`Supported Basis Systems`

    ========= =================== =============================================
    Short     Basis vectors       Usage
    ========= =================== =============================================
    ``'NED'`` North, East, Down   Jost and Herrmann 1989
    ``'USE'`` Up, South, East     Global CMT Catalog, Larson et al. 2010
    ``'XYZ'`` East, North, Up     General formulation, Jost and Herrmann 1989
    ``'RT'``  Radial, Transverse, psmeca (GMT), Wessel and Smith 1999
              Tangential
    ``'NWU'`` North, West, Up     Stein and Wysession 2003
    ========= =================== =============================================

    .. rubric:: Examples

    (1) Using basis system ``'NED'``.

        >>> from obspy.imaging.mopad_wrapper import Beachball
        >>> mt = [1, 2, 3, -4, -5, -10]
        >>> Beachball(mt, mopad_basis='NED') #doctest: +SKIP

        .. plot::

            from obspy.imaging.mopad_wrapper import Beachball
            mt = [1, 2, 3, -4, -5, -10]
            Beachball(mt, mopad_basis='NED')
    """
    mopad_kwargs = {}
    loc = locals()
    # map to kwargs used in mopad
    for key in KWARG_MAP:
        value = loc[key]
        for mopad_key in KWARG_MAP[key]:
            mopad_kwargs[mopad_key] = value
    # convert from points to size in cm
    for key in ['plot_aux_plot_size', 'plot_size']:
        # 100.0 is matplotlib's default DPI for savefig
        mopad_kwargs[key] = mopad_kwargs[key] / 100.0 * 2.54
    # use nofill kwarg

    mt = mopad_MomentTensor(fm, system=mopad_basis)
    bb = mopad_BeachBall(mt, npoints=size)

    # show plot in a window
    if outfile is None:
        bb.ploBB(mopad_kwargs)
    # save plot to file
    else:
        # no format specified, parse it from outfile name
        if mopad_kwargs['plot_outfile_format'] is None:
            mopad_kwargs['plot_outfile_format'] = \
                mopad_kwargs['plot_outfile'].split(".")[-1]
        else:
            # append file format if not already at end of outfile
            if not mopad_kwargs['plot_outfile'].endswith(
               mopad_kwargs['plot_outfile_format']):
                mopad_kwargs['plot_outfile'] += "." + \
                    mopad_kwargs['plot_outfile_format']
        bb.save_BB(mopad_kwargs)
