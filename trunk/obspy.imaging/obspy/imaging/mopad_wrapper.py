# -*- coding: utf-8 -*-
#-----------------------------------------------
# Filename: core.py
#  Purpose: Wrapper for mopad
#   Author: Tobias Megies, Moritz Beyreuther
#    Email: megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2011 ObsPy Development Team
#-----------------------------------------------
"""
ObsPy bindings to mopad
"""

import warnings
import numpy as np
from matplotlib import pyplot as plt, patches
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
from mopad import BeachBall as mopad_BeachBall
from mopad import MomentTensor as mopad_MomentTensor
from mopad import epsilon


# seems the base system we (gmt) are using is called "USE" in mopad
MOPAD_BASIS = "USE"
KWARG_MAP = {'size': ['plot_size', 'plot_aux_plot_size'],
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
#  state of mopads kwargs dict before plotting after following command:
#python mopad.py p 10,10,10 -f bla.png -s 0.5 -a 0.5 -l 3 r 0.4 -n 2 b 0.8 -w b -r g -q 72
# ipdb> kwargs_dict
# Out[0]: 
#{'plot_aux_plot_size': 1.9685039370078741,
# 'plot_full_sphere': False,
# 'plot_nodalline': True,
# 'plot_nodalline_alpha': 0.80000000000000004,
# 'plot_nodalline_colour': 'b',
# 'plot_nodalline_width': 2.0,
# 'plot_outerline': True,
# 'plot_outerline_alpha': 0.40000000000000002,
# 'plot_outerline_colour': 'r',
# 'plot_outerline_width': 3.0,
# 'plot_outfile': '/home/megies/obspy/branches/obspy.imaging/bla.png',
# 'plot_outfile_format': 'png',
# 'plot_pa_plot': False,
# 'plot_pressure_colour': 'b',
# 'plot_save_plot': True,
# 'plot_size': 1.9685039370078741,
# 'plot_tension_colour': 'g',
# 'plot_total_alpha': 0.5}


def Beach(fm, linewidth=2, facecolor='b', bgcolor='w', edgecolor='k',
          alpha=1.0, xy=(0, 0), width=200, size=80, nofill=False,
          zorder=100):
    """
    Return a beach ball as a collection which can be connected to an
    current matplotlib axes instance (ax.add_collection).
    
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
        curves. Minimum is automatically set to 100.
    :param facecolor: Color to use for quadrants of tension; can be a string, e.g. 
        'r', 'b' or three component color vector, [R G B].
    :param edgecolor: Color of the edges.
    :param bgcolor: The background color, usually white.
    :param alpha: The alpha level of the beach ball.
    :param xy: Origin position of the beach ball as tuple.
    :param width: Symbol size of beach ball.
    :param nofill: Do not fill the beach ball, but only plot the planes.
    :param zorder: Set zorder. Artists with lower zorder values are drawn
                   first.
    """
    # initialize beachball
    mt = mopad_MomentTensor(fm, MOPAD_BASIS)
    bb = mopad_BeachBall(mt, npoints=size)
    bb._setup_BB(unit_circle=False)
    
    # extract the coordinates and colors of the lines
    res = width/2.0
    neg_nodalline = bb._nodalline_negative_final_US
    pos_nodalline = bb._nodalline_positive_final_US
    tension_colour = bb._plot_tension_colour
    pressure_colour = bb._plot_pressure_colour

    # based on mopads _setup_plot_US() function
    # collect patches for the selection
    coll = [None, None, None]
    coll[0] = patches.Circle(xy, radius=res)
    coll[1] = xy2patch(neg_nodalline[0,:], neg_nodalline[1,:], res, xy)
    coll[2] = xy2patch(pos_nodalline[0,:], pos_nodalline[1,:], res, xy)

    # set the color of the three parts
    fc = [None, None, None]
    if bb._plot_clr_order > 0 :
        fc[0] = pressure_colour
        fc[1] = tension_colour
        fc[2] = tension_colour
        if bb._plot_curve_in_curve != 0:
            fc[0] = tension_colour
            if bb._plot_curve_in_curve < 1 :
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
            if bb._plot_curve_in_curve < 1 :
                fc[1] = tension_colour
                fc[2] = pressure_colour
            else:
                coll = [coll[i] for i in (0, 2, 1)]
                fc[1] = tension_colour
                fc[2] = pressure_colour

    if bb._pure_isotropic:
        if abs( np.trace( bb._M )) > epsilon:
            # use the circle as the upperst layer
            coll = [coll[0]]
            if bb._plot_clr_order < 0:
                fc = [tension_colour]
            else:
                fc = [pressure_colour]

    # transfrom the patches to a path collection and set
    # the appropriate attributes
    collection = PatchCollection(coll, match_original=False)
    collection.set_facecolors(fc)
    collection.set_alpha(alpha)
    collection.set_linewidth(linewidth)
    collection.set_zorder(zorder)
    return collection


def Beachball(fm, size=200, linewidth=2, facecolor='b', edgecolor='k',
              bgcolor='w', alpha=1.0, xy=(0, 0), width=200, outfile=None,
              format=None, nofill=False, fig=None):
    """
    Draws a beach ball diagram of an earthquake focal mechanism. 
    
    S1, D1, and R1, the strike, dip and rake of one of the focal planes, can 
    be vectors of multiple focal mechanisms.

    :param size: Draw with this diameter.
    :param fig: Give an existing figure instance to plot into. New Figure if
                set to None.
    :param format: If specified the format in which the plot should be
                   saved. E.g. (pdf, png, jpg, eps)

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


    mt = mopad_MomentTensor(fm, MOPAD_BASIS)
    bb = mopad_BeachBall(mt)

    # show plot in a window
    if outfile is None:
        bb.ploBB(mopad_kwargs)
    # save plot to file
    else:
        # no format specified, parse it from outfile name
        if mopad_kwargs['plot_outfile_format'] is None:
            mopad_kwargs['plot_outfile_format'] = mopad_kwargs['plot_outfile'].split(".")[-1]
        else:
            # append file format if not already at end of outfile
            if not mopad_kwargs['plot_outfile'].endswith(mopad_kwargs['plot_outfile_format']):
                mopad_kwargs['plot_outfile'] += "." + mopad_kwargs['plot_outfile_format']
        bb.save_BB(mopad_kwargs)

 
def xy2patch(x, y, res, xy): 
    # transform into the Path coordinate system  
    x = x * res + xy[0] 
    y = y * res + xy[1] 
    verts = zip(x.tolist(), y.tolist()) 
    codes = [Path.MOVETO] 
    codes.extend([Path.LINETO]*(len(x)-2)) 
    codes.append(Path.CLOSEPOLY) 
    path = Path(verts, codes)  
    return patches.PathPatch(path) 


if __name__ == '__main__':
    """
    """
    mt = [[0.91, -0.89, -0.02, 1.78, -1.55, 0.47],
          [274, 13, 55],
          [130, 79, 98],
          [264.98, 45.00, -159.99],
          [160.55, 76.00, -46.78],
          [1.45, -6.60, 5.14, -2.67, -3.16, 1.36],
          [235, 80, 35],
          [138, 56, 168],
          [1, 1, 1, 0, 0, 0],
          [-1, -1, -1, 0, 0, 0],
          [1, -2, 1, 0, 0, 0],
          [1, -1, 0, 0, 0, 0],
          [1, -1, 0, 0, 0, -1],
          [179, 55, -78],
          [10, 42.5, 90],
          [10, 42.5, 92],
          [150, 87, 1],
          [0.99, -2.00, 1.01, 0.92, 0.48, 0.15],
          [5.24, -6.77, 1.53, 0.81, 1.49, -0.05],
          [16.578, -7.987, -8.592, -5.515, -29.732, 7.517],
          [-2.39, 1.04, 1.35, 0.57, -2.94, -0.94],
          [150, 87, 1]]


    # Initialize figure
    fig = plt.figure(1, figsize=(3, 3), dpi=100)
    ax = fig.add_subplot(111, aspect='equal')

    # Plot the stations or borders
    ax.plot([-100, -100, 100, 100], [-100, 100, -100, 100], 'rv')

    x = -100
    y = -100
    for i, t in enumerate(mt):
        # add the beachball (a collection of two patches) to the axis
        ax.add_collection(Beach(t, size=100, width=30, xy=(x, y),
                                linewidth=.6))
        x += 50
        if (i + 1) % 5 == 0:
            x = -100
            y += 50

    # Set the x and y limits and save the output
    ax.axis([-120, 120, -120, 120])
    plt.show()
