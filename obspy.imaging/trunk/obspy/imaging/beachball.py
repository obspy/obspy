# -*- coding: utf-8 -*-
"""
obspy.imaging.beachball 

This is an adopted matlab script written from Oliver Boyd. You can find the 
latest Version of his Script at his homepage.

@see: http://www.ceri.memphis.edu/people/olboyd/Software/Software.html
"""

from matplotlib import pyplot as plt, patches, lines
from numpy import array, linalg, zeros, mean, sqrt, fabs, arcsin, arccos, \
    concatenate, pi, cos, power, abs, sum, fliplr, isnan, arange, sin, ones, \
    arctan2, arctan, tan
from pylab import figure, getp, setp, gca, show
import StringIO


d2r = pi/180


def Beachball(fm, diam=200, linewidth=2, color='b', alpha=1.0, file=None, 
              format=None):
    """
    Draws beachball diagram of earthquake double-couple focal mechanism(s). S1, D1, and
        R1, the strike, dip and rake of one of the focal planes, can be vectors of
        multiple focal mechanisms.
    fm - focal mechanism that is either number of mechanisms (NM) by 3 (strike, dip, and rake)
        or NM x 6 (mxx, myy, mzz, mxy, mxz, myz - the six independent components of
        the moment tensor). The strike is of the first plane, clockwise relative to north.
        The dip is of the first plane, defined clockwise and perpedicular to strike, 
        relative to horizontal such that 0 is horizontal and 90 is vertical. The rake is 
        of the first focal plane solution. 90 moves the hanging wall up-dip (thrust),
        0 moves it in the strike direction (left-lateral), -90 moves it down-dip
        (normal), and 180 moves it opposite to strike (right-lateral).
    centerX - place beachball(s) at position centerX
    centerY - place beachball(s) at position centerY
    diam - draw with this diameter
    color - color to use for quadrants of tension; can be a string, e.g. 'r'
        'b' or three component color vector, [R G B].
    """
    n = len(fm)
    special = False
    if n == 6:
        (S1, D1, R1) = Mij2SDR(fm[0], fm[1], fm[2], fm[3], fm[4], fm[5])
        # catch explosion
        if (fm[0]+fm[1]+fm[2])/3. > 0:
            special = True
    elif n == 3:
        S1 = fm[0]
        D1 = fm[1]
        R1 = fm[2]
    else:
        raise TypeError("Wrong input value for 'fm'.")
    
    M = 0
    if R1 > 180:
        R1 -= 180
        M = 1
    if R1 < 0:
        R1 += 180
        M = 1
    
    # Get azimuth and dip of second plane
    (S2, D2, _R2) = AuxPlane(S1, D1, R1)
    
    # Diam must be at least 100
    if diam<100:
        diam=100
    # actual painting diam is only 95% due to an axis display glitch
    D = diam*0.95
    
    if D1 >= 90:
        D1 = 89.9999
    if D2 >= 90:
        D2 = 89.9999
    
    phi = arange(0, pi, .01)
    l1 = sqrt(power(90 - D1, 2)/(power(sin(phi), 2) + power(cos(phi), 2) * \
                                 power(90 - D1, 2) / power(90, 2)))
    l2 = sqrt(power(90 - D2, 2)/(power(sin(phi), 2) + power(cos(phi), 2) * \
                                 power(90 - D2, 2) / power(90, 2)))
    
    inc = 1
    (X1, Y1) = Pol2Cart(phi+S1*d2r, l1)
    
    if M == 1:
        lo = S1 - 180
        hi = S2
        if lo > hi:
            inc = -inc
        th1 = arange(S1-180, S2, inc)
        (Xs1, Ys1) = Pol2Cart(th1*d2r, 90*ones((1, len(th1))))
        (X2, Y2) = Pol2Cart(phi+S2*d2r, l2)
        th2 = arange(S2+180, S1, -inc)
    else:
        hi = S1 - 180
        lo = S2 - 180
        if lo > hi:
            inc = -inc
        th1 = arange(hi, lo, -inc)
        (Xs1, Ys1) = Pol2Cart(th1*d2r, 90*ones((1, len(th1))))
        (X2, Y2) = Pol2Cart(phi+S2*d2r, l2)
        X2 = X2[::-1]
        Y2 = Y2[::-1]
        th2 = arange(S2, S1, inc)
    (Xs2, Ys2) = Pol2Cart(th2*d2r, 90*ones((1, len(th2))))
    X = concatenate((X1,Xs1[0],X2,Xs2[0]),1)
    Y = concatenate((Y1,Ys1[0],Y2,Ys2[0]),1)
    
    X = X * D/90
    Y = Y * D/90
    phid = arange(0,2*pi,.01)
    (x,y) = Pol2Cart(phid, 90)
    xx = x*D/90
    yy = y*D/90
    
    # plot the figure
    fig = plt.figure(figsize=(3,3), dpi=100)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    fig.set_figheight(diam/100)
    fig.set_figwidth(diam/100)
    ax = fig.add_subplot(111, aspect='equal')
    ax.fill(xx, yy, 'w', alpha=alpha, linewidth=linewidth)
    if special:
        # explosion = fill all
        ax.fill(xx, yy, color, alpha=alpha, linewidth=linewidth)
    else:
        ax.fill(Y, X, color, alpha=alpha, linewidth=linewidth)
    lines.Line2D(xx, yy, color='k', 
                 linewidth=linewidth, zorder=10, alpha=alpha)
    # hide axes + ticks
    ax.axison = False
    # export
    if file:
        if format:
            fig.savefig(file, dpi=100, transparent=True, format=format)
        else:
            fig.savefig(file, dpi=100, transparent=True)
    elif format and not file:
        imgdata = StringIO.StringIO()
        fig.savefig(imgdata, format=format, dpi=100, transparent=True)
        imgdata.seek(0)
        return imgdata.read()
    else:
        show()


def Pol2Cart(th, r):
    x = r*cos(th)
    y = r*sin(th)
    return (x, y)


def StrikeDip(n, e, u):
    """
    Finds strike and dip of plane given normal vector having components n, e, 
    and u.
   
    Adapted from Andy Michaels and Oliver Boyd.
    """
    r2d = 180/pi
    if u < 0:
        n = -n
        e = -e
        u = -u
    
    strike = arctan2(e,n)*r2d
    strike = strike - 90
    while strike >= 360:
            strike = strike - 360
    while strike < 0:
            strike = strike + 360
    x = sqrt(power(n, 2) + power(e, 2))
    dip = arctan2(x, u)*r2d
    return (strike, dip)


def AuxPlane(s1, d1, r1):
    """
    Get Strike and dip of second plane.
    
    Adapted from Andy Michael and Oliver Boyd.
    """
    r2d = 180/pi
    
    z = (s1+90)/r2d
    z2 = d1/r2d
    z3 = r1/r2d
    # slick vector in plane 1
    sl1 = -cos(z3)*cos(z)-sin(z3)*sin(z)*cos(z2)
    sl2 = cos(z3)*sin(z)-sin(z3)*cos(z)*cos(z2)
    sl3 = sin(z3)*sin(z2)
    (strike, dip) = StrikeDip(sl2, sl1, sl3)
    
    n1 = sin(z)*sin(z2) # normal vector to plane 1
    n2 = cos(z)*sin(z2)
    h1 = -sl2 # strike vector of plane 2
    h2 = sl1
    # note h3=0 always so we leave it out
    # n3 = cos(z2)
    
    z = h1*n1 + h2*n2
    z = z/sqrt(h1*h1 + h2*h2)
    z = arccos(z)
    rake = 0
    if sl3 > 0:
        rake = z*r2d
    if sl3 <= 0:
        rake = -z*r2d
    return (strike, dip, rake)


def Mij2SDR(mxx, myy, mzz, mxy, mxz, myz):
    """
    @param mij: - six independent components of the moment tensor
    @return (strike, dip, rake): 
        strike - strike of first focal plane (degrees)
        dip - dip of first focal plane (degrees)
        rake - rake of first focal plane (degrees)
    
    Adapted from code from Chen Ji, Gaven Hayes and Oliver Boyd.
    """
    a = array([[mxx, mxy, mxz], 
               [mxy, myy, myz], 
               [mxz, myz, mzz]])
    (d, v) = linalg.eig(a)
    D = array([d[1], d[0], d[2]])
    V = array([[v[1, 1], -v[1, 0], -v[1, 2]],
               [v[2, 1], -v[2, 0], -v[2, 2]],
               [-v[0, 1], v[0, 0], v[0, 2]]])
    IMAX = D.argmax()
    IMIN = D.argmin()
    AE = (V[:,IMAX]+V[:,IMIN])/sqrt(2.0)
    AN = (V[:,IMAX]-V[:,IMIN])/sqrt(2.0)
    AER = sqrt(power(AE[0], 2) + power(AE[1], 2) + power(AE[2], 2))
    ANR = sqrt(power(AN[0], 2) + power(AN[1], 2) + power(AN[2], 2))
    AE = AE/AER
    AN = AN/ANR
    if AN[2] <= 0.:
        AN1 = AN
        AE1 = AE
    else:
        AN1 = -AN
        AE1 = -AE
    (ft, fd, fl) = TDL(AN1, AE1)
    strike = 360 - ft
    dip = fd
    rake = 180 - fl
    return (strike, dip, rake)


def TDL(AN, BN):
    XN=AN[0]
    YN=AN[1]
    ZN=AN[2]
    XE=BN[0]
    YE=BN[1]
    ZE=BN[2]
    AAA=1.0E-06
    CON=57.2957795
    if fabs(ZN) < AAA:
        FD=90.
        AXN=fabs(XN)
        if AXN > 1.0:
            AXN=1.0
        FT=arcsin(AXN)*CON
        ST=-XN
        CT=YN
        if ST >= 0. and CT < 0:
            FT=180.-FT
        if ST < 0. and CT <= 0:
            FT=180.+FT
        if ST < 0. and CT > 0:
            FT=360.-FT
        FL=arcsin(abs(ZE))*CON
        SL=-ZE
        if fabs(XN) < AAA:
            CL=XE/YN
        else:
            CL=-YE/XN
        if SL >= 0. and CL < 0: 
            FL=180.-FL
        if SL < 0. and CL <= 0: 
            FL=FL-180.
        if SL < 0. and CL > 0: 
            FL=-FL
    else:
        if -ZN > 1.0: 
            ZN=-1.0
        FDH=arccos(-ZN)
        FD=FDH*CON
        SD=sin(FDH)
        if SD == 0:
            return
        ST=-XN/SD
        CT=YN/SD
        SX=fabs(ST)
        if SX > 1.0: 
            SX=1.0
        FT=arcsin(SX)*CON
        if ST >= 0. and CT < 0: 
            FT=180.-FT
        if ST < 0. and CT <= 0: 
            FT=180.+FT
        if ST < 0. and CT > 0: 
            FT=360.-FT
        SL=-ZE/SD
        SX=fabs(SL)
        if SX > 1.0: 
            SX=1.0
        FL=arcsin(SX)*CON
        if ST == 0:
            CL=XE/CT
        else:
            XXX=YN*ZN*ZE/SD/SD+YE
            CL=-SD*XXX/XN
            if CT == 0:
                CL=YE/ST
        if SL >= 0. and CL < 0: 
            FL=180.-FL
        if SL < 0. and CL <= 0: 
            FL=FL-180.
        if SL < 0. and CL > 0: 
            FL=-FL
    return (FT, FD, FL)
