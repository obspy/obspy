# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: beachball.py
#  Purpose: Draws a beach ball diagram of an earthquake focal mechanism.
#   Author: Robert Barsch
#    Email: barsch@egu.eu
#
# Copyright (C) 2008-2012 Robert Barsch
# ---------------------------------------------------------------------

"""
Draws a beachball diagram of an earthquake focal mechanism

Most source code provided here are adopted from

1. MatLab script `bb.m`_ written by Andy Michael, Chen Ji and Oliver Boyd.
2. ps_meca program from the `Generic Mapping Tools (GMT)`_.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

.. _`Generic Mapping Tools (GMT)`: https://www.generic-mapping-tools.org
.. _`bb.m`: http://www.ceri.memphis.edu/people/olboyd/Software/Software.html
"""
import io
import warnings

import numpy as np
from decorator import decorator


D2R = np.pi / 180
R2D = 180 / np.pi
EPSILON = 0.00001


@decorator
def mopad_fallback(func, *args, **kwargs):
    try:
        result = func(*args, **kwargs)
    except IndexError:
        msg = "Encountered an exception while plotting the beachball. " \
              "Falling back to the mopad wrapper which is slower but more " \
              "stable."
        warnings.warn(msg)

        # Could be done with the inspect module but this wrapper is only a
        # single purpose wrapper and thus KISS.
        arguments = ["fm", "linewidth", "facecolor", "bgcolor", "edgecolor",
                     "alpha", "xy", "width", "size", "nofill", "zorder",
                     "axes"]

        final_kwargs = {}
        for _i, arg in enumerate(args):
            final_kwargs[arguments[_i]] = arg

        final_kwargs.update(kwargs)

        from .mopad_wrapper import beach as _mopad_beach

        result = _mopad_beach(**final_kwargs)

    return result


@mopad_fallback
def beach(fm, linewidth=2, facecolor='b', bgcolor='w', edgecolor='k',
          alpha=1.0, xy=(0, 0), width=200, size=100, nofill=False,
          zorder=100, axes=None):
    """
    Return a beach ball as a collection which can be connected to an
    current matplotlib axes instance (ax.add_collection).

    S1, D1, and R1, the strike, dip and rake of one of the focal planes, can
    be vectors of multiple focal mechanisms.

    :param fm: Focal mechanism that is either number of mechanisms (NM) by 3
        (strike, dip, and rake) or NM x 6 (M11, M22, M33, M12, M13, M23 - the
        six independent components of the moment tensor, where the coordinate
        system is 1,2,3 = Up,South,East which equals r,theta,phi -
        Harvard/Global CMT convention). The relation to Aki and Richards
        x,y,z equals North,East,Down convention is as follows: Mrr=Mzz,
        Mtt=Mxx, Mpp=Myy, Mrt=Mxz, Mrp=-Myz, Mtp=-Mxy.
        The strike is of the first plane, clockwise relative to north.
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
    :type width: int or tuple
    :param width: Symbol size of beach ball, or tuple for elliptically
        shaped patches. Defaults to size ``200``.
    :param size: Controls the number of interpolation points for the
        curves. Minimum is automatically set to ``100``.
    :param nofill: Do not fill the beach ball, but only plot the planes.
    :param zorder: Set zorder. Artists with lower zorder values are drawn
        first.
    :type axes: :class:`matplotlib.axes.Axes`
    :param axes: Used to make beach balls circular on non-scaled axes. Also
        maintains the aspect ratio when resizing the figure. Will not add
        the returned collection to the axes instance.
    """
    # check if one or two widths are specified (Circle or Ellipse)
    from matplotlib import collections, transforms
    try:
        assert len(width) == 2
    except TypeError:
        width = (width, width)
    mt = None
    np1 = None
    if isinstance(fm, MomentTensor):
        mt = fm
        np1 = mt2plane(mt)
    elif isinstance(fm, NodalPlane):
        np1 = fm
    elif len(fm) == 6:
        mt = MomentTensor(fm[0], fm[1], fm[2], fm[3], fm[4], fm[5], 0)
        np1 = mt2plane(mt)
    elif len(fm) == 3:
        np1 = NodalPlane(fm[0], fm[1], fm[2])
    else:
        raise TypeError("Wrong input value for 'fm'.")

    # Only at least size 100, i.e. 100 points in the matrix are allowed
    if size < 100:
        size = 100

    # Return as collection
    plot_dc_used = True
    if mt:
        (t, n, p) = mt2axes(mt.normalized)
        if np.fabs(n.val) < EPSILON and np.fabs(t.val + p.val) < EPSILON:
            colors, p = plot_dc(np1, size, xy=xy, width=width)
        else:
            colors, p = plot_mt(t, n, p, size,
                                plot_zerotrace=True, xy=xy, width=width)
            plot_dc_used = False
    else:
        colors, p = plot_dc(np1, size=size, xy=xy, width=width)

    col = collections.PatchCollection(p, match_original=False)
    if nofill:
        col.set_facecolor('none')
    else:
        # Replace color dummies 'b' and 'w' by face and bgcolor
        fc = [facecolor if c == 'b' else bgcolor for c in colors]
        col.set_facecolors(fc)

    # Use the given axes to maintain the aspect ratio of beachballs on figure
    # resize.
    if axes is not None:
        # This is what holds the aspect ratio (but breaks the positioning)
        col.set_transform(transforms.IdentityTransform())
        # Next is a dirty hack to fix the positioning:
        # 1. Need to bring the all patches to the origin (0, 0).
        for p in col._paths:
            p.vertices -= xy
        # 2. Then use the offset property of the collection to position the
        #    patches
        col.set_offsets(xy)
        col._transOffset = axes.transData

    col.set_edgecolor(edgecolor)
    col.set_alpha(alpha)
    col.set_linewidth(linewidth)
    col.set_zorder(zorder)

    # warn about color blending bug, see #1464
    if alpha != 1 and not nofill and not plot_dc_used:
        msg = ("There is a known bug when plotting semi-transparent patches "
               "for non-DC sources, which leads to blending of pressure and "
               "tension color, see issue #1464.")
        warnings.warn(msg)

    return col


def beachball(fm, linewidth=2, facecolor='b', bgcolor='w', edgecolor='k',
              alpha=1.0, xy=(0, 0), width=200, size=100, nofill=False,
              zorder=100, outfile=None, format=None, fig=None):
    """
    Draws a beach ball diagram of an earthquake focal mechanism.

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
    """
    import matplotlib.pyplot as plt
    plot_width = width * 0.95

    # plot the figure
    if not fig:
        fig = plt.figure(figsize=(3, 3), dpi=100)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        fig.set_figheight(width // 100)
        fig.set_figwidth(width // 100)
    ax = fig.add_subplot(111, aspect='equal')

    # hide axes + ticks
    ax.axison = False

    # plot the collection
    collection = beach(fm, linewidth=linewidth, facecolor=facecolor,
                       edgecolor=edgecolor, bgcolor=bgcolor,
                       alpha=alpha, nofill=nofill, xy=xy,
                       width=plot_width, size=size, zorder=zorder)
    ax.add_collection(collection)

    ax.autoscale_view(tight=False, scalex=True, scaley=True)
    # export
    if outfile:
        if format:
            fig.savefig(outfile, dpi=100, transparent=True, format=format)
        else:
            fig.savefig(outfile, dpi=100, transparent=True)
    elif format and not outfile:
        imgdata = io.BytesIO()
        fig.savefig(imgdata, format=format, dpi=100, transparent=True)
        imgdata.seek(0)
        return imgdata.read()
    else:
        plt.show()
        return fig


def plot_mt(T, N, P, size=200, plot_zerotrace=True,  # noqa
            x0=0, y0=0, xy=(0, 0), width=200):
    """
    Uses a principal axis T, N and P to draw a beach ball plot.

    :param ax: axis object of a matplotlib figure
    :param T: :class:`~PrincipalAxis`
    :param N: :class:`~PrincipalAxis`
    :param P: :class:`~PrincipalAxis`

    Adapted from ps_tensor / utilmeca.c / `Generic Mapping Tools (GMT)`_.

    .. _`Generic Mapping Tools (GMT)`: https://www.generic-mapping-tools.org
    """
    # check if one or two widths are specified (Circle or Ellipse)
    from matplotlib import patches
    try:
        assert len(width) == 2
    except TypeError:
        width = (width, width)
    collect = []
    colors = []
    res = [value / float(size) for value in width]
    b = 1
    big_iso = 0
    j = 1
    j2 = 0
    j3 = 0
    n = 0
    azi = np.zeros((3, 2))
    x = np.zeros(400)
    y = np.zeros(400)
    x2 = np.zeros(400)
    y2 = np.zeros(400)
    x3 = np.zeros(400)
    y3 = np.zeros(400)
    xp1 = np.zeros(800)
    yp1 = np.zeros(800)
    xp2 = np.zeros(400)
    yp2 = np.zeros(400)

    a = np.zeros(3)
    p = np.zeros(3)
    v = np.zeros(3)
    a[0] = T.strike
    a[1] = N.strike
    a[2] = P.strike
    p[0] = T.dip
    p[1] = N.dip
    p[2] = P.dip
    v[0] = T.val
    v[1] = N.val
    v[2] = P.val

    vi = (v[0] + v[1] + v[2]) / 3.
    for i in range(0, 3):
        v[i] = v[i] - vi

    radius_size = size * 0.5

    if np.fabs(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) < EPSILON:
        # pure implosion-explosion
        if vi > 0.:
            cir = patches.Ellipse(xy, width=width[0], height=width[1])
            collect.append(cir)
            colors.append('b')
        if vi < 0.:
            cir = patches.Ellipse(xy, width=width[0], height=width[1])
            collect.append(cir)
            colors.append('w')
        return colors, collect

    if np.fabs(v[0]) >= np.fabs(v[2]):
        d = 0
        m = 2
    else:
        d = 2
        m = 0

    if (plot_zerotrace):
        vi = 0.

    f = -v[1] / float(v[d])
    iso = vi / float(v[d])

    # Cliff Frohlich, Seismological Research letters,
    # Vol 7, Number 1, January-February, 1996
    # Unless the isotropic parameter lies in the range
    # between -1 and 1 - f there will be no nodes whatsoever

    if iso < -1:
        cir = patches.Ellipse(xy, width=width[0], height=width[1])
        collect.append(cir)
        colors.append('w')
        return colors, collect
    elif iso > 1 - f:
        cir = patches.Ellipse(xy, width=width[0], height=width[1])
        collect.append(cir)
        colors.append('b')
        return colors, collect

    spd = np.sin(p[d] * D2R)
    cpd = np.cos(p[d] * D2R)
    spb = np.sin(p[b] * D2R)
    cpb = np.cos(p[b] * D2R)
    spm = np.sin(p[m] * D2R)
    cpm = np.cos(p[m] * D2R)
    sad = np.sin(a[d] * D2R)
    cad = np.cos(a[d] * D2R)
    sab = np.sin(a[b] * D2R)
    cab = np.cos(a[b] * D2R)
    sam = np.sin(a[m] * D2R)
    cam = np.cos(a[m] * D2R)

    for i in range(0, 360):
        fir = i * D2R
        s2alphan = (2. + 2. * iso) / \
            float(3. + (1. - 2. * f) * np.cos(2. * fir))
        if s2alphan > 1.:
            big_iso += 1
        else:
            alphan = np.arcsin(np.sqrt(s2alphan))
            sfi = np.sin(fir)
            cfi = np.cos(fir)
            san = np.sin(alphan)
            can = np.cos(alphan)

            xz = can * spd + san * sfi * spb + san * cfi * spm
            xn = can * cpd * cad + san * sfi * cpb * cab + \
                san * cfi * cpm * cam
            xe = can * cpd * sad + san * sfi * cpb * sab + \
                san * cfi * cpm * sam

            if np.fabs(xn) < EPSILON and np.fabs(xe) < EPSILON:
                takeoff = 0.
                az = 0.
            else:
                az = np.arctan2(xe, xn)
                if az < 0.:
                    az += np.pi * 2.
                takeoff = np.arccos(xz / float(np.sqrt(xz * xz + xn * xn +
                                                       xe * xe)))
            if takeoff > np.pi / 2.:
                takeoff = np.pi - takeoff
                az += np.pi
                if az > np.pi * 2.:
                    az -= np.pi * 2.
            r = np.sqrt(2) * np.sin(takeoff / 2.)
            si = np.sin(az)
            co = np.cos(az)
            if i == 0:
                azi[i][0] = az
                x[i] = x0 + radius_size * r * si
                y[i] = y0 + radius_size * r * co
                azp = az
            else:
                if np.fabs(np.fabs(az - azp) - np.pi) < D2R * 10.:
                    azi[n][1] = azp
                    n += 1
                    azi[n][0] = az
                if np.fabs(np.fabs(az - azp) - np.pi * 2.) < D2R * 2.:
                    if azp < az:
                        azi[n][0] += np.pi * 2.
                    else:
                        azi[n][0] -= np.pi * 2.
                if n == 0:
                    x[j] = x0 + radius_size * r * si
                    y[j] = y0 + radius_size * r * co
                    j += 1
                elif n == 1:
                    x2[j2] = x0 + radius_size * r * si
                    y2[j2] = y0 + radius_size * r * co
                    j2 += 1
                elif n == 2:
                    x3[j3] = x0 + radius_size * r * si
                    y3[j3] = y0 + radius_size * r * co
                    j3 += 1
                azp = az
    azi[n][1] = az

    if v[1] < 0.:
        rgb1 = 'b'
        rgb2 = 'w'
    else:
        rgb1 = 'w'
        rgb2 = 'b'

    cir = patches.Ellipse(xy, width=width[0], height=width[1])
    collect.append(cir)
    colors.append(rgb2)
    if n == 0:
        collect.append(xy2patch(x[0:360], y[0:360], res, xy))
        colors.append(rgb1)
        return colors, collect
    elif n == 1:
        for i in range(0, j):
            xp1[i] = x[i]
            yp1[i] = y[i]
        if azi[0][0] - azi[0][1] > np.pi:
            azi[0][0] -= np.pi * 2.
        elif azi[0][1] - azi[0][0] > np.pi:
            azi[0][0] += np.pi * 2.
        if azi[0][0] < azi[0][1]:
            az = azi[0][1] - D2R
            while az > azi[0][0]:
                si = np.sin(az)
                co = np.cos(az)
                xp1[i] = x0 + radius_size * si
                yp1[i] = y0 + radius_size * co
                i += 1
                az -= D2R
        else:
            az = azi[0][1] + D2R
            while az < azi[0][0]:
                si = np.sin(az)
                co = np.cos(az)
                xp1[i] = x0 + radius_size * si
                yp1[i] = y0 + radius_size * co
                i += 1
                az += D2R
        collect.append(xy2patch(xp1[0:i], yp1[0:i], res, xy))
        colors.append(rgb1)
        for i in range(0, j2):
            xp2[i] = x2[i]
            yp2[i] = y2[i]
        if azi[1][0] - azi[1][1] > np.pi:
            azi[1][0] -= np.pi * 2.
        elif azi[1][1] - azi[1][0] > np.pi:
            azi[1][0] += np.pi * 2.
        if azi[1][0] < azi[1][1]:
            az = azi[1][1] - D2R
            while az > azi[1][0]:
                si = np.sin(az)
                co = np.cos(az)
                xp2[i] = x0 + radius_size * si
                i += 1
                yp2[i] = y0 + radius_size * co
                az -= D2R
        else:
            az = azi[1][1] + D2R
            while az < azi[1][0]:
                si = np.sin(az)
                co = np.cos(az)
                xp2[i] = x0 + radius_size * si
                i += 1
                yp2[i] = y0 + radius_size * co
                az += D2R
        collect.append(xy2patch(xp2[0:i], yp2[0:i], res, xy))
        colors.append(rgb1)
        return colors, collect
    elif n == 2:
        for i in range(0, j3):
            xp1[i] = x3[i]
            yp1[i] = y3[i]
        for ii in range(0, j):
            xp1[i] = x[ii]
            i += 1
            yp1[i] = y[ii]
        if big_iso:
            ii = j2 - 1
            while ii >= 0:
                xp1[i] = x2[ii]
                i += 1
                yp1[i] = y2[ii]
                ii -= 1
            collect.append(xy2patch(xp1[0:i], yp1[0:i], res, xy))
            colors.append(rgb1)
            return colors, collect

        if azi[2][0] - azi[0][1] > np.pi:
            azi[2][0] -= np.pi * 2.
        elif azi[0][1] - azi[2][0] > np.pi:
            azi[2][0] += np.pi * 2.
        if azi[2][0] < azi[0][1]:
            az = azi[0][1] - D2R
            while az > azi[2][0]:
                si = np.sin(az)
                co = np.cos(az)
                xp1[i] = x0 + radius_size * si
                i += 1
                yp1[i] = y0 + radius_size * co
                az -= D2R
        else:
            az = azi[0][1] + D2R
            while az < azi[2][0]:
                si = np.sin(az)
                co = np.cos(az)
                xp1[i] = x0 + radius_size * si
                i += 1
                yp1[i] = y0 + radius_size * co
                az += D2R
        collect.append(xy2patch(xp1[0:i], yp1[0:i], res, xy))
        colors.append(rgb1)

        for i in range(0, j2):
            xp2[i] = x2[i]
            yp2[i] = y2[i]
        if azi[1][0] - azi[1][1] > np.pi:
            azi[1][0] -= np.pi * 2.
        elif azi[1][1] - azi[1][0] > np.pi:
            azi[1][0] += np.pi * 2.
        if azi[1][0] < azi[1][1]:
            az = azi[1][1] - D2R
            while az > azi[1][0]:
                si = np.sin(az)
                co = np.cos(az)
                xp2[i] = x0 + radius_size * si
                i += 1
                yp2[i] = y0 + radius_size * co
                az -= D2R
        else:
            az = azi[1][1] + D2R
            while az < azi[1][0]:
                si = np.sin(az)
                co = np.cos(az)
                xp2[i] = x0 + radius_size * si
                i += 1
                yp2[i] = y0 + radius_size * co
                az += D2R
        collect.append(xy2patch(xp2[0:i], yp2[0:i], res, xy))
        colors.append(rgb1)
        return colors, collect


def plot_dc(np1, size=200, xy=(0, 0), width=200):
    """
    Uses one nodal plane of a double couple to draw a beach ball plot.

    :param ax: axis object of a matplotlib figure
    :param np1: :class:`~NodalPlane`

    Adapted from MATLAB script
    `bb.m <http://www.ceri.memphis.edu/people/olboyd/Software/Software.html>`_
    written by Andy Michael, Chen Ji and Oliver Boyd.
    """
    # check if one or two widths are specified (Circle or Ellipse)
    try:
        assert len(width) == 2
    except TypeError:
        width = (width, width)
    s_1 = np1.strike
    d_1 = np1.dip
    r_1 = np1.rake

    m = 0
    if r_1 > 180:
        r_1 -= 180
        m = 1
    if r_1 < 0:
        r_1 += 180
        m = 1

    # Get azimuth and dip of second plane
    (s_2, d_2, _r_2) = aux_plane(s_1, d_1, r_1)

    d = size / 2

    if d_1 >= 90:
        d_1 = 89.9999
    if d_2 >= 90:
        d_2 = 89.9999

    # arange checked for numerical stability, np.pi is not multiple of 0.1
    phi = np.arange(0, np.pi, .01)
    l1 = np.sqrt(
        np.power(90 - d_1, 2) / (
            np.power(np.sin(phi), 2) +
            np.power(np.cos(phi), 2) *
            np.power(90 - d_1, 2) / np.power(90, 2)))
    l2 = np.sqrt(
        np.power(90 - d_2, 2) / (
            np.power(np.sin(phi), 2) + np.power(np.cos(phi), 2) *
            np.power(90 - d_2, 2) / np.power(90, 2)))

    collect = []
    # plot paths, once for tension areas and once for pressure areas
    for m_ in ((m + 1) % 2, m):
        inc = 1
        (x_1, y_1) = pol2cart(phi + s_1 * D2R, l1)

        if m_ == 1:
            lo = s_1 - 180
            hi = s_2
            if lo > hi:
                inc = -1
            th1 = np.arange(s_1 - 180, s_2, inc)
            (xs_1, ys_1) = pol2cart(th1 * D2R, 90 * np.ones((1, len(th1))))
            (x_2, y_2) = pol2cart(phi + s_2 * D2R, l2)
            th2 = np.arange(s_2 + 180, s_1, -inc)
        else:
            hi = s_1 - 180
            lo = s_2 - 180
            if lo > hi:
                inc = -1
            th1 = np.arange(hi, lo, -inc)
            (xs_1, ys_1) = pol2cart(th1 * D2R, 90 * np.ones((1, len(th1))))
            (x_2, y_2) = pol2cart(phi + s_2 * D2R, l2)
            x_2 = x_2[::-1]
            y_2 = y_2[::-1]
            th2 = np.arange(s_2, s_1, inc)
        (xs_2, ys_2) = pol2cart(th2 * D2R, 90 * np.ones((1, len(th2))))
        x = np.concatenate((x_1, xs_1[0], x_2, xs_2[0]))
        y = np.concatenate((y_1, ys_1[0], y_2, ys_2[0]))

        x = x * d / 90
        y = y * d / 90

        # calculate resolution
        res = [value / float(size) for value in width]

        # construct the patch
        collect.append(xy2patch(y, x, res, xy))
    return ['b', 'w'], collect


def xy2patch(x, y, res, xy):
    # check if one or two resolutions are specified (Circle or Ellipse)
    from matplotlib import path as mplpath
    from matplotlib import patches
    try:
        assert len(res) == 2
    except TypeError:
        res = (res, res)
    # transform into the Path coordinate system
    x = x * res[0] + xy[0]
    y = y * res[1] + xy[1]
    verts = list(zip(x.tolist(), y.tolist()))
    codes = [mplpath.Path.MOVETO]
    codes.extend([mplpath.Path.LINETO] * (len(x) - 2))
    codes.append(mplpath.Path.CLOSEPOLY)
    path = mplpath.Path(verts, codes)
    return patches.PathPatch(path)


def pol2cart(th, r):
    """
    """
    x = r * np.cos(th)
    y = r * np.sin(th)
    return (x, y)


def strike_dip(n, e, u):
    """
    Finds strike and dip of plane given normal vector having components n, e,
    and u.

    Adapted from MATLAB script
    `bb.m <http://www.ceri.memphis.edu/people/olboyd/Software/Software.html>`_
    written by Andy Michael, Chen Ji and Oliver Boyd.
    """
    r2d = 180 / np.pi
    if u < 0:
        n = -n
        e = -e
        u = -u

    strike = np.arctan2(e, n) * r2d
    strike = strike - 90
    while strike >= 360:
        strike = strike - 360
    while strike < 0:
        strike = strike + 360
    x = np.sqrt(np.power(n, 2) + np.power(e, 2))
    dip = np.arctan2(x, u) * r2d
    return (strike, dip)


def aux_plane(s1, d1, r1):
    """
    Get Strike and dip of second plane.

    Adapted from MATLAB script
    `bb.m <http://www.ceri.memphis.edu/people/olboyd/Software/Software.html>`_
    written by Andy Michael, Chen Ji and Oliver Boyd.
    """
    r2d = 180 / np.pi

    z = (s1 + 90) / r2d
    z2 = d1 / r2d
    z3 = r1 / r2d
    # slick vector in plane 1
    sl1 = -np.cos(z3) * np.cos(z) - np.sin(z3) * np.sin(z) * np.cos(z2)
    sl2 = np.cos(z3) * np.sin(z) - np.sin(z3) * np.cos(z) * np.cos(z2)
    sl3 = np.sin(z3) * np.sin(z2)
    (strike, dip) = strike_dip(sl2, sl1, sl3)

    n1 = np.sin(z) * np.sin(z2)  # normal vector to plane 1
    n2 = np.cos(z) * np.sin(z2)
    h1 = -sl2  # strike vector of plane 2
    h2 = sl1
    # note h3=0 always so we leave it out
    # n3 = np.cos(z2)

    z = h1 * n1 + h2 * n2
    z = z / np.sqrt(h1 * h1 + h2 * h2)
    # we might get above 1.0 only due to floating point
    # precision. Clip for those cases.
    float64epsilon = 2.2204460492503131e-16
    if 1.0 < abs(z) < 1.0 + 100 * float64epsilon:
        z = np.copysign(1.0, z)
    z = np.arccos(z)
    rake = 0
    if sl3 > 0:
        rake = z * r2d
    if sl3 <= 0:
        rake = -z * r2d
    return (strike, dip, rake)


def mt2plane(mt):
    """
    Calculates a nodal plane of a given moment tensor.

    :param mt: :class:`~MomentTensor`
    :return: :class:`~NodalPlane`

    Adapted from MATLAB script
    `bb.m <http://www.ceri.memphis.edu/people/olboyd/Software/Software.html>`_
    written by Andy Michael, Chen Ji and Oliver Boyd.
    """
    (d, v) = np.linalg.eig(mt.mt)
    d = np.array([d[1], d[0], d[2]])
    v = np.array([[v[1, 1], -v[1, 0], -v[1, 2]],
                 [v[2, 1], -v[2, 0], -v[2, 2]],
                 [-v[0, 1], v[0, 0], v[0, 2]]])
    imax = d.argmax()
    imin = d.argmin()
    ae = (v[:, imax] + v[:, imin]) / np.sqrt(2.0)
    an = (v[:, imax] - v[:, imin]) / np.sqrt(2.0)
    aer = np.sqrt(np.power(ae[0], 2) + np.power(ae[1], 2) + np.power(ae[2], 2))
    anr = np.sqrt(np.power(an[0], 2) + np.power(an[1], 2) + np.power(an[2], 2))
    ae = ae / aer
    if not anr:
        an = np.array([np.nan, np.nan, np.nan])
    else:
        an = an / anr
    if an[2] <= 0.:
        an1 = an
        ae1 = ae
    else:
        an1 = -an
        ae1 = -ae
    (ft, fd, fl) = tdl(an1, ae1)
    return NodalPlane(360 - ft, fd, 180 - fl)


def tdl(an, bn):
    """
    Helper function for mt2plane.

    Adapted from MATLAB script
    `bb.m <http://www.ceri.memphis.edu/people/olboyd/Software/Software.html>`_
    written by Andy Michael, Chen Ji and Oliver Boyd.
    """
    xn = an[0]
    yn = an[1]
    zn = an[2]
    xe = bn[0]
    ye = bn[1]
    ze = bn[2]
    aaa = 1.0 / (1000000)
    con = 57.2957795
    if np.fabs(zn) < aaa:
        fd = 90.
        axn = np.fabs(xn)
        if axn > 1.0:
            axn = 1.0
        ft = np.arcsin(axn) * con
        st = -xn
        ct = yn
        if st >= 0. and ct < 0:
            ft = 180. - ft
        if st < 0. and ct <= 0:
            ft = 180. + ft
        if st < 0. and ct > 0:
            ft = 360. - ft
        fl = np.arcsin(abs(ze)) * con
        sl = -ze
        if np.fabs(xn) < aaa:
            cl = xe / yn
        else:
            cl = -ye / xn
        if sl >= 0. and cl < 0:
            fl = 180. - fl
        if sl < 0. and cl <= 0:
            fl = fl - 180.
        if sl < 0. and cl > 0:
            fl = -fl
    else:
        if -zn > 1.0:
            zn = -1.0
        fdh = np.arccos(-zn)
        fd = fdh * con
        sd = np.sin(fdh)
        if sd == 0:
            return
        st = -xn / sd
        ct = yn / sd
        sx = np.fabs(st)
        if sx > 1.0:
            sx = 1.0
        ft = np.arcsin(sx) * con
        if st >= 0. and ct < 0:
            ft = 180. - ft
        if st < 0. and ct <= 0:
            ft = 180. + ft
        if st < 0. and ct > 0:
            ft = 360. - ft
        sl = -ze / sd
        sx = np.fabs(sl)
        if sx > 1.0:
            sx = 1.0
        fl = np.arcsin(sx) * con
        if st == 0:
            cl = xe / ct
        else:
            xxx = yn * zn * ze / sd / sd + ye
            cl = -sd * xxx / xn
            if ct == 0:
                cl = ye / st
        if sl >= 0. and cl < 0:
            fl = 180. - fl
        if sl < 0. and cl <= 0:
            fl = fl - 180.
        if sl < 0. and cl > 0:
            fl = -fl
    return (ft, fd, fl)


def mt2axes(mt):
    """
    Calculates the principal axes of a given moment tensor.

    :param mt: :class:`~MomentTensor`
    :return: tuple of :class:`~PrincipalAxis` T, N and P

    Adapted from ps_tensor / utilmeca.c /
    `Generic Mapping Tools (GMT) <https://www.generic-mapping-tools.org>`_.
    """
    (d, v) = np.linalg.eigh(mt.mt)
    pl = np.arcsin(-v[0])
    az = np.arctan2(v[2], -v[1])
    for i in range(0, 3):
        if pl[i] <= 0:
            pl[i] = -pl[i]
            az[i] += np.pi
        if az[i] < 0:
            az[i] += 2 * np.pi
        if az[i] > 2 * np.pi:
            az[i] -= 2 * np.pi
    pl *= R2D
    az *= R2D

    t = PrincipalAxis(d[2], az[2], pl[2])
    n = PrincipalAxis(d[1], az[1], pl[1])
    p = PrincipalAxis(d[0], az[0], pl[0])
    return (t, n, p)


class PrincipalAxis(object):
    """
    A principal axis.

    Strike and dip values are in degrees.

    >>> a = PrincipalAxis(1.3, 20, 50)
    >>> a.dip
    50
    >>> a.strike
    20
    >>> a.val
    1.3
    """
    def __init__(self, val=0, strike=0, dip=0):
        self.val = val
        self.strike = strike
        self.dip = dip


class NodalPlane(object):
    """
    A nodal plane.

    All values are in degrees.

    >>> a = NodalPlane(13, 20, 50)
    >>> a.strike
    13
    >>> a.dip
    20
    >>> a.rake
    50
    """
    def __init__(self, strike=0, dip=0, rake=0):
        self.strike = strike
        self.dip = dip
        self.rake = rake


class MomentTensor(object):
    """
    A moment tensor.

    >>> a = MomentTensor(1, 1, 0, 0, 0, -1, 26)
    >>> b = MomentTensor(np.array([1, 1, 0, 0, 0, -1]), 26)
    >>> c = MomentTensor(np.array([[1, 0, 0], [0, 1, -1], [0, -1, 0]]), 26)
    >>> a.mt
    array([[ 1,  0,  0],
           [ 0,  1, -1],
           [ 0, -1,  0]])
    >>> b.yz
    -1
    >>> a.expo
    26
    """
    def __init__(self, *args):
        if len(args) == 2:
            a = args[0]
            self.expo = args[1]
            if len(a) == 6:
                # six independent components
                self.mt = np.array([[a[0], a[3], a[4]],
                                    [a[3], a[1], a[5]],
                                    [a[4], a[5], a[2]]])
            elif isinstance(a, np.ndarray) and a.shape == (3, 3):
                # full matrix
                self.mt = a
            else:
                raise TypeError("Wrong size of input parameter.")
        elif len(args) == 7:
            # six independent components
            self.mt = np.array([[args[0], args[3], args[4]],
                                [args[3], args[1], args[5]],
                                [args[4], args[5], args[2]]])
            self.expo = args[6]
        else:
            raise TypeError("Wrong size of input parameter.")

    @property
    def normalized(self):
        return MomentTensor(self.mt_normalized, self.expo)

    @property
    def mt_normalized(self):
        return self.mt / np.linalg.norm(self.mt)

    @property
    def xx(self):
        return self.mt[0][0]

    @property
    def xy(self):
        return self.mt[0][1]

    @property
    def xz(self):
        return self.mt[0][2]

    @property
    def yz(self):
        return self.mt[1][2]

    @property
    def yy(self):
        return self.mt[1][1]

    @property
    def zz(self):
        return self.mt[2][2]


if __name__ == '__main__':
    import doctest
    doctest.testmod()
