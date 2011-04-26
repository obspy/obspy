# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
# Filename: beachball.py
#  Purpose: Draws a beach ball diagram of an earthquake focal mechanism.
#   Author: Robert Barsch
#    Email: barsch@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2011 Robert Barsch
#---------------------------------------------------------------------

"""
Draws a beachball diagram of an earthquake focal mechanism

Most source code provided here are adopted from

1. MatLab script written by Oliver Boyd 
   see: http://www.ceri.memphis.edu/people/olboyd/Software/Software.html
2. ps_meca program from the Generic Mapping Tools (GMT)
   see: http://gmt.soest.hawaii.edu


GNU General Public License (GPL)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
USA.
"""

#Needs to be done before importing pyplot and the like.
from matplotlib import pyplot as plt, patches
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
from numpy import array, linalg, zeros, sqrt, fabs, arcsin, arccos, pi, cos, \
    power, abs, arange, sin, ones, arctan2, ndarray, concatenate
from pylab import show
import StringIO
import doctest


D2R = pi / 180
R2D = 180 / pi
EPSILON = 0.00001



def Beach(fm, linewidth=2, facecolor='b', bgcolor='w', edgecolor='k',
          alpha=1.0, xy=(0, 0), width=200, size=100, nofill=False,
          zorder=100):
    """
    Return a beach ball as a collection which can be connected to an
    current matplotlib axes instance (ax.add_collection).
    
    S1, D1, and R1, the strike, dip and rake of one of the focal planes, can 
    be vectors of multiple focal mechanisms.
    
    :param fm: Focal mechanism that is either number of mechanisms (NM) by 3 
        (strike, dip, and rake) or NM x 6 (Mxx, Myy, Mzz, Mxy, Mxz, Myz - the 
        six independent components of the moment tensor, where the coordinate
        system is x,y,z = Up,South,East). The strike is of the first plane,
        clockwise relative to north. 
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
    mt = None
    np1 = None
    if isinstance(fm, MomentTensor):
        mt = fm
        np1 = MT2Plane(mt)
    elif isinstance(fm, NodalPlane):
        np1 = fm
    elif len(fm) == 6:
        mt = MomentTensor(fm[0], fm[1], fm[2], fm[3], fm[4], fm[5], 0)
        np1 = MT2Plane(mt)
    elif len(fm) == 3:
        np1 = NodalPlane(fm[0], fm[1], fm[2])
    else:
        raise TypeError("Wrong input value for 'fm'.")

    # Only at least size 100, i.e. 100 points in the matrix are allowed
    if size < 100:
        size = 100

    # Return as collection
    if mt:
        (T, N, P) = MT2Axes(mt)
        if fabs(N.val) < EPSILON and fabs(T.val + P.val) < EPSILON:
            colors, p = plotDC(np1, size, xy=xy, width=width)
        else:
            colors, p = plotMT(T, N, P, size, outline=True,
                               plot_zerotrace=True, xy=xy, width=width)
    else:
        colors, p = plotDC(np1, size=size, xy=xy, width=width)


    if nofill:
        #XXX not tested with plotMT
        collection = PatchCollection([p[1]], match_original=False)
        collection.set_facecolor('none')
    else:
        collection = PatchCollection(p, match_original=False)
        # Replace color dummies 'b' and 'w' by face and bgcolor
        fc = [facecolor if c == 'b' else bgcolor for c in colors]
        collection.set_facecolors(fc)

    collection.set_edgecolor(edgecolor)
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
    plot_size = size * 0.95

    # plot the figure
    if not fig:
        fig = plt.figure(figsize=(3, 3), dpi=100)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        fig.set_figheight(size // 100)
        fig.set_figwidth(size // 100)
    ax = fig.add_subplot(111, aspect='equal')

    # hide axes + ticks
    ax.axison = False

    # plot the collection
    collection = Beach(fm, linewidth=linewidth, facecolor=facecolor,
                       edgecolor=edgecolor, bgcolor=bgcolor,
                       alpha=alpha, nofill=nofill, xy=(0, 0),
                       width=plot_size, size=plot_size)
    ax.add_collection(collection)

    ax.autoscale_view(tight=False, scalex=True, scaley=True)
    # export
    if outfile:
        if format:
            fig.savefig(outfile, dpi=100, transparent=True, format=format)
        else:
            fig.savefig(outfile, dpi=100, transparent=True)
    elif format and not outfile:
        imgdata = StringIO.StringIO()
        fig.savefig(imgdata, format=format, dpi=100, transparent=True)
        imgdata.seek(0)
        return imgdata.read()
    else:
        show()
        return fig


def plotMT(T, N, P, size=200, outline=True, plot_zerotrace=True,
           x0=0, y0=0, xy=(0, 0), width=200):
    """
    Uses a principal axis T, N and P to draw a beach ball plot.
    
    :param ax: axis object of a matplotlib figure
    :param T: L{PrincipalAxis}
    :param N: L{PrincipalAxis}
    :param P: L{PrincipalAxis}
    
    Adapted from ps_tensor / utilmeca.c / Generic Mapping Tools (GMT).
    @see: http://gmt.soest.hawaii.edu
    """
    collect = []
    colors = []
    res = width / float(size)
    b = 1
    big_iso = 0
    j = 1
    j2 = 0
    j3 = 0
    n = 0
    azi = zeros((3, 2))
    x = zeros(400)
    y = zeros(400)
    x2 = zeros(400)
    y2 = zeros(400)
    x3 = zeros(400)
    y3 = zeros(400)
    xp1 = zeros(800)
    yp1 = zeros(800)
    xp2 = zeros(400)
    yp2 = zeros(400)

    a = zeros(3)
    p = zeros(3)
    v = zeros(3)
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

    if fabs(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) < EPSILON:
        # pure implosion-explosion
        if vi > 0.:
            cir = patches.Circle(xy, radius=width / 2.0)
            collect.append(cir)
            colors.append('b')
        if vi < 0.:
            cir = patches.Circle(xy, radius=width / 2.0)
            collect.append(cir)
            colors.append('w')
        return colors, collect

    if fabs(v[0]) >= fabs(v[2]):
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
        cir = patches.Circle(xy, radius=width / 2.0)
        collect.append(cir)
        colors.append('w')
        return colors, collect
    elif iso > 1 - f:
        cir = patches.Circle(xy, radius=width / 2.0)
        collect.append(cir)
        colors.append('b')
        return colors, collect

    spd = sin(p[d] * D2R)
    cpd = cos(p[d] * D2R)
    spb = sin(p[b] * D2R)
    cpb = cos(p[b] * D2R)
    spm = sin(p[m] * D2R)
    cpm = cos(p[m] * D2R)
    sad = sin(a[d] * D2R)
    cad = cos(a[d] * D2R)
    sab = sin(a[b] * D2R)
    cab = cos(a[b] * D2R)
    sam = sin(a[m] * D2R)
    cam = cos(a[m] * D2R)

    for i in range(0, 360):
        fir = i * D2R
        s2alphan = (2. + 2. * iso) / float(3. + (1. - 2. * f) * cos(2. * fir))
        if s2alphan > 1.:
            big_iso += 1
        else:
            alphan = arcsin(sqrt(s2alphan))
            sfi = sin(fir)
            cfi = cos(fir)
            san = sin(alphan)
            can = cos(alphan)

            xz = can * spd + san * sfi * spb + san * cfi * spm
            xn = can * cpd * cad + san * sfi * cpb * cab + \
                 san * cfi * cpm * cam
            xe = can * cpd * sad + san * sfi * cpb * sab + \
                 san * cfi * cpm * sam

            if fabs(xn) < EPSILON and fabs(xe) < EPSILON:
                takeoff = 0.
                az = 0.
            else:
                az = arctan2(xe, xn)
                if az < 0.:
                    az += pi * 2.
                takeoff = arccos(xz / float(sqrt(xz * xz + xn * xn + xe * xe)))
            if takeoff > pi / 2.:
                takeoff = pi - takeoff
                az += pi
                if az > pi * 2.:
                    az -= pi * 2.
            r = sqrt(2) * sin(takeoff / 2.)
            si = sin(az)
            co = cos(az)
            if i == 0:
                azi[i][0] = az
                x[i] = x0 + radius_size * r * si
                y[i] = y0 + radius_size * r * co
                azp = az
            else:
                if fabs(fabs(az - azp) - pi) < D2R * 10.:
                        azi[n][1] = azp
                        n += 1
                        azi[n][0] = az
                if fabs(fabs(az - azp) - pi * 2.) < D2R * 2.:
                        if azp < az:
                            azi[n][0] += pi * 2.
                        else:
                            azi[n][0] -= pi * 2.
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

    cir = patches.Circle(xy, radius=width / 2.0)
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
        if azi[0][0] - azi[0][1] > pi:
            azi[0][0] -= pi * 2.;
        elif azi[0][1] - azi[0][0] > pi:
            azi[0][0] += pi * 2.
        if azi[0][0] < azi[0][1]:
            az = azi[0][1] - D2R
            while az > azi[0][0]:
                si = sin(az)
                co = cos(az)
                xp1[i] = x0 + radius_size * si
                yp1[i] = y0 + radius_size * co
                i += 1
                az -= D2R
        else:
            az = azi[0][1] + D2R
            while az < azi[0][0]:
                si = sin(az)
                co = cos(az)
                xp1[i] = x0 + radius_size * si
                yp1[i] = y0 + radius_size * co
                i += 1
                az += D2R
        collect.append(xy2patch(xp1[0:i], yp1[0:i], res, xy))
        colors.append(rgb1)
        for i in range(0, j2):
            xp2[i] = x2[i]
            yp2[i] = y2[i]
        if azi[1][0] - azi[1][1] > pi:
            azi[1][0] -= pi * 2.
        elif azi[1][1] - azi[1][0] > pi:
            azi[1][0] += pi * 2.
        if azi[1][0] < azi[1][1]:
            az = azi[1][1] - D2R
            while az > azi[1][0]:
                si = sin(az)
                co = cos(az)
                xp2[i] = x0 + radius_size * si;
                i += 1
                yp2[i] = y0 + radius_size * co;
                az -= D2R
        else:
            az = azi[1][1] + D2R
            while az < azi[1][0]:
                si = sin(az)
                co = cos(az)
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

        if azi[2][0] - azi[0][1] > pi:
            azi[2][0] -= pi * 2.
        elif azi[0][1] - azi[2][0] > pi:
            azi[2][0] += pi * 2.
        if azi[2][0] < azi[0][1]:
            az = azi[0][1] - D2R
            while az > azi[2][0]:
                si = sin(az)
                co = cos(az)
                xp1[i] = x0 + radius_size * si
                i += 1
                yp1[i] = y0 + radius_size * co
                az -= D2R
        else:
            az = azi[0][1] + D2R
            while az < azi[2][0]:
                si = sin(az)
                co = cos(az)
                xp1[i] = x0 + radius_size * si
                i += 1
                yp1[i] = y0 + radius_size * co
                az += D2R
        collect.append(xy2patch(xp1[0:i], yp1[0:i], res, xy))
        colors.append(rgb1)

        for i in range(0, j2):
            xp2[i] = x2[i]
            yp2[i] = y2[i]
        if azi[1][0] - azi[1][1] > pi:
            azi[1][0] -= pi * 2.
        elif azi[1][1] - azi[1][0] > pi:
            azi[1][0] += pi * 2.
        if azi[1][0] < azi[1][1]:
            az = azi[1][1] - D2R
            while az > azi[1][0]:
                si = sin(az)
                co = cos(az)
                xp2[i] = x0 + radius_size * si
                i += 1
                yp2[i] = y0 + radius_size * co
                az -= D2R
        else:
            az = azi[1][1] + D2R
            while az < azi[1][0]:
                si = sin(az)
                co = cos(az)
                xp2[i] = x0 + radius_size * si
                i += 1
                yp2[i] = y0 + radius_size * co
                az += D2R
        collect.append(xy2patch(xp2[0:i], yp2[0:i], res, xy))
        colors.append(rgb1)
        return colors, collect


def plotDC(np1, size=200, xy=(0, 0), width=200):
    """
    Uses one nodal plane of a double couple to draw a beach ball plot.
    
    :param ax: axis object of a matplotlib figure
    :param np1: L{NodalPlane}
    
    Adapted from bb.m written by Oliver S. Boyd.
    @see: http://www.ceri.memphis.edu/people/olboyd/Software/Software.html
    """
    S1 = np1.strike
    D1 = np1.dip
    R1 = np1.rake

    M = 0
    if R1 > 180:
        R1 -= 180
        M = 1
    if R1 < 0:
        R1 += 180
        M = 1

    # Get azimuth and dip of second plane
    (S2, D2, _R2) = AuxPlane(S1, D1, R1)

    D = size / 2

    if D1 >= 90:
        D1 = 89.9999
    if D2 >= 90:
        D2 = 89.9999

    phi = arange(0, pi, .01)
    l1 = sqrt(power(90 - D1, 2) / (power(sin(phi), 2) + power(cos(phi), 2) * \
                                 power(90 - D1, 2) / power(90, 2)))
    l2 = sqrt(power(90 - D2, 2) / (power(sin(phi), 2) + power(cos(phi), 2) * \
                                 power(90 - D2, 2) / power(90, 2)))

    inc = 1
    (X1, Y1) = Pol2Cart(phi + S1 * D2R, l1)

    if M == 1:
        lo = S1 - 180
        hi = S2
        if lo > hi:
            inc = -inc
        th1 = arange(S1 - 180, S2, inc)
        (Xs1, Ys1) = Pol2Cart(th1 * D2R, 90 * ones((1, len(th1))))
        (X2, Y2) = Pol2Cart(phi + S2 * D2R, l2)
        th2 = arange(S2 + 180, S1, -inc)
    else:
        hi = S1 - 180
        lo = S2 - 180
        if lo > hi:
            inc = -inc
        th1 = arange(hi, lo, -inc)
        (Xs1, Ys1) = Pol2Cart(th1 * D2R, 90 * ones((1, len(th1))))
        (X2, Y2) = Pol2Cart(phi + S2 * D2R, l2)
        X2 = X2[::-1]
        Y2 = Y2[::-1]
        th2 = arange(S2, S1, inc)
    (Xs2, Ys2) = Pol2Cart(th2 * D2R, 90 * ones((1, len(th2))))
    X = concatenate((X1, Xs1[0], X2, Xs2[0]), 1)
    Y = concatenate((Y1, Ys1[0], Y2, Ys2[0]), 1)

    X = X * D / 90
    Y = Y * D / 90

    # calculate resolution
    res = width / float(size)

    # construct the patches
    collect = [patches.Circle(xy, radius=width / 2.0)]
    collect.append(xy2patch(Y, X, res, xy))
    return ['b', 'w'], collect

def xy2patch(x, y, res, xy):
    # transform into the Path coordinate system 
    x = x * res + xy[0]
    y = y * res + xy[1]
    verts = zip(x.tolist(), y.tolist())
    codes = [Path.MOVETO]
    codes.extend([Path.LINETO] * (len(x) - 2))
    codes.append(Path.CLOSEPOLY)
    path = Path(verts, codes)
    return patches.PathPatch(path)

def Pol2Cart(th, r):
    """
    """
    x = r * cos(th)
    y = r * sin(th)
    return (x, y)


def StrikeDip(n, e, u):
    """
    Finds strike and dip of plane given normal vector having components n, e, 
    and u.
   
    Adapted from bb.m written by Andy Michaels and Oliver Boyd.
    @see: http://www.ceri.memphis.edu/people/olboyd/Software/Software.html
    """
    r2d = 180 / pi
    if u < 0:
        n = -n
        e = -e
        u = -u

    strike = arctan2(e, n) * r2d
    strike = strike - 90
    while strike >= 360:
            strike = strike - 360
    while strike < 0:
            strike = strike + 360
    x = sqrt(power(n, 2) + power(e, 2))
    dip = arctan2(x, u) * r2d
    return (strike, dip)


def AuxPlane(s1, d1, r1):
    """
    Get Strike and dip of second plane.
    
    Adapted from bb.m written by Andy Michael and Oliver Boyd.
    @see: http://www.ceri.memphis.edu/people/olboyd/Software/Software.html
    """
    r2d = 180 / pi

    z = (s1 + 90) / r2d
    z2 = d1 / r2d
    z3 = r1 / r2d
    # slick vector in plane 1
    sl1 = -cos(z3) * cos(z) - sin(z3) * sin(z) * cos(z2)
    sl2 = cos(z3) * sin(z) - sin(z3) * cos(z) * cos(z2)
    sl3 = sin(z3) * sin(z2)
    (strike, dip) = StrikeDip(sl2, sl1, sl3)

    n1 = sin(z) * sin(z2) # normal vector to plane 1
    n2 = cos(z) * sin(z2)
    h1 = -sl2 # strike vector of plane 2
    h2 = sl1
    # note h3=0 always so we leave it out
    # n3 = cos(z2)

    z = h1 * n1 + h2 * n2
    z = z / sqrt(h1 * h1 + h2 * h2)
    z = arccos(z)
    rake = 0
    if sl3 > 0:
        rake = z * r2d
    if sl3 <= 0:
        rake = -z * r2d
    return (strike, dip, rake)


def MT2Plane(mt):
    """
    Calculates a nodal plane of a given moment tensor.
     
    :param mt: L{MomentTensor}
    :return: L{NodalPlane}
    
    Adapted from bb.m written by Oliver S. Boyd.
    @see: http://www.ceri.memphis.edu/people/olboyd/Software/Software.html
    """
    (d, v) = linalg.eig(mt.mt)
    D = array([d[1], d[0], d[2]])
    V = array([[v[1, 1], -v[1, 0], -v[1, 2]],
               [v[2, 1], -v[2, 0], -v[2, 2]],
               [-v[0, 1], v[0, 0], v[0, 2]]])
    IMAX = D.argmax()
    IMIN = D.argmin()
    AE = (V[:, IMAX] + V[:, IMIN]) / sqrt(2.0)
    AN = (V[:, IMAX] - V[:, IMIN]) / sqrt(2.0)
    AER = sqrt(power(AE[0], 2) + power(AE[1], 2) + power(AE[2], 2))
    ANR = sqrt(power(AN[0], 2) + power(AN[1], 2) + power(AN[2], 2))
    AE = AE / AER
    AN = AN / ANR
    if AN[2] <= 0.:
        AN1 = AN
        AE1 = AE
    else:
        AN1 = -AN
        AE1 = -AE
    (ft, fd, fl) = TDL(AN1, AE1)
    return NodalPlane(360 - ft, fd, 180 - fl)


def TDL(AN, BN):
    """
    Helper function for MT2Plane.
    
    Adapted from bb.m written by Oliver S. Boyd.
    @see: http://www.ceri.memphis.edu/people/olboyd/Software/Software.html
    """
    XN = AN[0]
    YN = AN[1]
    ZN = AN[2]
    XE = BN[0]
    YE = BN[1]
    ZE = BN[2]
    AAA = 1.0 / (1000000)
    CON = 57.2957795
    if fabs(ZN) < AAA:
        FD = 90.
        AXN = fabs(XN)
        if AXN > 1.0:
            AXN = 1.0
        FT = arcsin(AXN) * CON
        ST = -XN
        CT = YN
        if ST >= 0. and CT < 0:
            FT = 180. - FT
        if ST < 0. and CT <= 0:
            FT = 180. + FT
        if ST < 0. and CT > 0:
            FT = 360. - FT
        FL = arcsin(abs(ZE)) * CON
        SL = -ZE
        if fabs(XN) < AAA:
            CL = XE / YN
        else:
            CL = -YE / XN
        if SL >= 0. and CL < 0:
            FL = 180. - FL
        if SL < 0. and CL <= 0:
            FL = FL - 180.
        if SL < 0. and CL > 0:
            FL = -FL
    else:
        if - ZN > 1.0:
            ZN = -1.0
        FDH = arccos(-ZN)
        FD = FDH * CON
        SD = sin(FDH)
        if SD == 0:
            return
        ST = -XN / SD
        CT = YN / SD
        SX = fabs(ST)
        if SX > 1.0:
            SX = 1.0
        FT = arcsin(SX) * CON
        if ST >= 0. and CT < 0:
            FT = 180. - FT
        if ST < 0. and CT <= 0:
            FT = 180. + FT
        if ST < 0. and CT > 0:
            FT = 360. - FT
        SL = -ZE / SD
        SX = fabs(SL)
        if SX > 1.0:
            SX = 1.0
        FL = arcsin(SX) * CON
        if ST == 0:
            CL = XE / CT
        else:
            XXX = YN * ZN * ZE / SD / SD + YE
            CL = -SD * XXX / XN
            if CT == 0:
                CL = YE / ST
        if SL >= 0. and CL < 0:
            FL = 180. - FL
        if SL < 0. and CL <= 0:
            FL = FL - 180.
        if SL < 0. and CL > 0:
            FL = -FL
    return (FT, FD, FL)


def MT2Axes(mt):
    """
    Calculates the principal axes of a given moment tensor.
     
    :param mt: L{MomentTensor}
    :return: tuple of L{PrincipalAxis} T, N and P
    
    Adapted from GMT_momten2axe / utilmeca.c / Generic Mapping Tools (GMT).
    @see: http://gmt.soest.hawaii.edu
    """
    (D, V) = linalg.eigh(mt.mt)
    pl = arcsin(-V[0])
    az = arctan2(V[2], -V[1])
    for i in range(0, 3):
        if pl[i] <= 0:
            pl[i] = -pl[i]
            az[i] += pi
        if az[i] < 0:
            az[i] += 2 * pi
        if az[i] > 2 * pi:
            az[i] -= 2 * pi
    pl *= R2D
    az *= R2D

    T = PrincipalAxis(D[2], az[2], pl[2])
    N = PrincipalAxis(D[1], az[1], pl[1])
    P = PrincipalAxis(D[0], az[0], pl[0])
    return (T, N, P)


class PrincipalAxis(object):
    """
    A principal axis.
    
    Strike and dip values are in degrees.
    
    Usage:
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
    
    Usage:
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
    
    Usage:
      >>> a = MomentTensor(1, 1, 0, 0, 0, -1, 26)
      >>> b = MomentTensor(array([1, 1, 0, 0, 0, -1]), 26)
      >>> c = MomentTensor(array([[1, 0, 0], [0, 1, -1], [0, -1, 0]]), 26)
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
            A = args[0]
            self.expo = args[1]
            if len(A) == 6:
                # six independent components
                self.mt = array([[A[0], A[3], A[4]],
                                 [A[3], A[1], A[5]],
                                 [A[4], A[5], A[2]]])
            elif isinstance(A, ndarray) and A.shape == (3, 3):
                # full matrix
                self.mt = A
            else:
                raise TypeError("Wrong size of input parameter.")
        elif len(args) == 7:
            # six independent components
            self.mt = array([[args[0], args[3], args[4]],
                             [args[3], args[1], args[5]],
                             [args[4], args[5], args[2]]])
            self.expo = args[6]
        else:
            raise TypeError("Wrong size of input parameter.")

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
    doctest.testmod()
