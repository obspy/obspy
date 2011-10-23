# -*- coding: utf-8 -*-
#-----------------------------------------------
# Filename: mopad_wrapper.py
#  Purpose: Wrapper for mopad
#   Author: Tobias Megies, Moritz Beyreuther
#    Email: megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2011 ObsPy Development Team
#-----------------------------------------------
"""
ObsPy wrapper to the *Moment tensor Plotting and Decomposition tool* (mopad)
written by Lars Krieger and Sebastian Heimann.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU General Public License (GPL)
    (http://www.gnu.org/licenses/gpl.txt)
"""

import warnings
import numpy as np
from matplotlib import patches, collections
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
          alpha=1.0, xy=(0, 0), width=200, size=80, nofill=False,
          zorder=100, mopad_basis='USE'):
    """
    Return a beach ball as a collection which can be connected to an
    current matplotlib axes instance (ax.add_collection). Based on mopad.

    S1, D1, and R1, the strike, dip and rake of one of the focal planes, can
    be vectors of multiple focal mechanisms.

    :param fm: Focal mechanism that is either number of mechanisms (NM) by 3
        (strike, dip, and rake) or NM x 6 (Mxx, Myy, Mzz, Mxy, Mxz, Myz - the
        six independent components of the moment tensor). The strike is of the
        first plane, clockwise relative to north.
        The dip is of the first plane, defined clockwise and perpendicular to
        strike, relative to horizontal such that 0 is horizontal and 90 is
        vertical. The rake is of the first focal plane solution. 90 moves the
        hanging wall up-dip (thrust), 0 moves it in the strike direction
        (left-lateral), -90 moves it down-dip (normal), and 180 moves it
        opposite to strike (right-lateral).
    :param size: Controls the number of interpolation points for the
        curves. Defaults to 80, note that this and especially smaller
        values might produce artifacts, however it makes the plotting much
        faster. Use 360 for full resolution and no artifacts.
    :param facecolor: Color to use for quadrants of tension; can be a string,
        e.g. 'r', 'b' or three component color vector, [R G B].
    :param edgecolor: Color of the edges.
    :param bgcolor: The background color, usually white.
    :param alpha: The alpha level of the beach ball.
    :param xy: Origin position of the beach ball as tuple.
    :param width: Symbol size of beach ball.
    :param nofill: Do not fill the beach ball, but only plot the planes.
    :param zorder: Set zorder. Artists with lower zorder values are drawn
        first.
    :param mopad_basis: The system which may be chosen as 'NED' (North, East
        Down), 'USE' (Up, South, East), 'NWU' (North, West, Up) or 'XYZ'. 'USE'
        mimics the ObsPy Beachball behaviour.
    """
    # initialize beachball
    mt = mopad_MomentTensor(fm, mopad_basis)
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
    collection = collections.PatchCollection(coll, match_original=False)
    collection.set_facecolors(fc)
    collection.set_alpha(alpha)
    collection.set_linewidth(linewidth)
    collection.set_zorder(zorder)
    return collection


def Beachball(fm, size=200, linewidth=2, facecolor='b', edgecolor='k',
              bgcolor='w', alpha=1.0, xy=(0, 0), width=200, outfile=None,
              format=None, nofill=False, fig=None, mopad_basis='USE'):
    """
    Draws a beach ball diagram of an earthquake focal mechanism. Based on
    mopad.

    S1, D1, and R1, the strike, dip and rake of one of the focal planes, can
    be vectors of multiple focal mechanisms.

    :param size: Draw with this diameter.
    :param fig: Give an existing figure instance to plot into. New Figure if
        set to None.
    :param format: If specified the format in which the plot should be saved,
        e.g. pdf, png, jpg, or eps.

    For info on the remaining parameters see the
    :func:`~obspy.imaging.beachball.Beach` function of this module.
    """
    msg = "mopad wrapping is still beta stage!"
    warnings.warn(msg)

    mopad_kwargs = {}
    loc = locals()
    # map to kwargs used in mopad
    for key in KWARG_MAP:
        value = loc[key]
        for mopad_key in KWARG_MAP[key]:
            mopad_kwargs[mopad_key] = value
    # convert from points to size in cm
    for key in ['plot_aux_plot_size', 'plot_size']:
        # 100.0 is matplotlibs default dpi for savefig
        mopad_kwargs[key] = mopad_kwargs[key] / 100.0 * 2.54
    # use nofill kwarg

    mt = mopad_MomentTensor(fm, mopad_basis)
    bb = mopad_BeachBall(mt)

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
            if not mopad_kwargs['plot_outfile'].endswith(\
               mopad_kwargs['plot_outfile_format']):
                mopad_kwargs['plot_outfile'] += "." + \
                    mopad_kwargs['plot_outfile_format']
        bb.save_BB(mopad_kwargs)
