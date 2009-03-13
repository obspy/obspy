# -*- coding: utf-8 -*-
"""
obspy.imaging.beachball 

This is an ported MatLab script written by Oliver Boyd. You can find the 
latest version of his Script at his home page:
@see: http://www.ceri.memphis.edu/people/olboyd/Software/Software.html
"""

from matplotlib import pyplot as plt, patches, lines
from numpy import array, linalg, zeros, mean, sqrt, fabs, arcsin, arccos, \
    concatenate, pi, cos, power, abs, sum, fliplr, isnan, arange, sin, ones, \
    arctan2, arctan, tan
from pylab import figure, getp, setp, gca, show
import StringIO


D2R = pi/180
R2D = 180/pi
EPSILON = 0.00001


def Beachball(fm, diam=200, linewidth=2, color='b', alpha=1.0, file=None, 
              format=None):
    """
    Draws beachball diagram of earthquake double-couple focal mechanism(s). 
    
    S1, D1, and R1, the strike, dip and rake of one of the focal planes, can 
    be vectors of multiple focal mechanisms.
    
    @param fm: Focal mechanism that is either number of mechanisms (NM) by 3 
        (strike, dip, and rake) or NM x 6 (Mxx, Myy, Mzz, Mxy, Mxz, Myz - the 
        six independent components of the moment tensor). The strike is of the 
        first plane, clockwise relative to north. 
        The dip is of the first plane, defined clockwise and perpendicular to 
        strike, relative to horizontal such that 0 is horizontal and 90 is 
        vertical. The rake is of the first focal plane solution. 90 moves the 
        hanging wall up-dip (thrust), 0 moves it in the strike direction 
        (left-lateral), -90 moves it down-dip (normal), and 180 moves it 
        opposite to strike (right-lateral). 
    @param diam: Draw with this diameter.
    @param color: Color to use for quadrants of tension; can be a string, e.g. 
        'r', 'b' or three component color vector, [R G B].
    """
    n = len(fm)
    special = False
    if n == 6:
        (S1, D1, R1) = Mij2SDR(fm[0], fm[1], fm[2], fm[3], fm[4], fm[5])
        # catch explosion
        if (fm[0]+fm[1]+fm[2])/3. > EPSILON:
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
    (X1, Y1) = Pol2Cart(phi+S1*D2R, l1)
    
    if M == 1:
        lo = S1 - 180
        hi = S2
        if lo > hi:
            inc = -inc
        th1 = arange(S1-180, S2, inc)
        (Xs1, Ys1) = Pol2Cart(th1*D2R, 90*ones((1, len(th1))))
        (X2, Y2) = Pol2Cart(phi+S2*D2R, l2)
        th2 = arange(S2+180, S1, -inc)
    else:
        hi = S1 - 180
        lo = S2 - 180
        if lo > hi:
            inc = -inc
        th1 = arange(hi, lo, -inc)
        (Xs1, Ys1) = Pol2Cart(th1*D2R, 90*ones((1, len(th1))))
        (X2, Y2) = Pol2Cart(phi+S2*D2R, l2)
        X2 = X2[::-1]
        Y2 = Y2[::-1]
        th2 = arange(S2, S1, inc)
    (Xs2, Ys2) = Pol2Cart(th2*D2R, 90*ones((1, len(th2))))
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
    A = array([[mxx, mxy, mxz], 
               [mxy, myy, myz], 
               [mxz, myz, mzz]])
    (d, v) = linalg.eig(A)
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


class Axis(object):
    """
    A principal axis object.
    """
    def __init__(self, val=0, strike=0, dip=0):
        self.val = val
        self.strike = strike
        self.dip = dip


def Mij2Axes(mxx, myy, mzz, mxy, mxz, myz):
    """
    Returns the principal axes (T, N, P) of a the given six independent 
    components of a moment tensor (Mxx, Myy, Mzz, Mxy, Mxz, Myz).
    """
    A = array([[mxx, mxy, mxz], 
               [mxy, myy, myz], 
               [mxz, myz, mzz]])
    (D, V) = linalg.eigh(A)
    pl = arcsin(-V[0])
    az = arctan2(V[2], -V[1])
    for i in range(0, 3):
        if pl[i]<=0:
            pl[i] = -pl[i]
            az[i] += pi
        if az[i] < 0:
            az[i] += 2*pi
        if az[i] > 2*pi:
            az[i] -= 2*pi
    pl *= R2D
    az *= R2D
    
    T = Axis(D[2], az[2], pl[2])
    N = Axis(D[1], az[1], pl[1])
    P = Axis(D[0], az[0], pl[0])
    return (T, N, P)


#def Beachball2(T, N, P, size, x0=0, y0=0, c_rgb='r', e_rgb='b', outline=False, 
#               plot_zerotrace=False):
#    pass
#{
#    int d, b = 1, m;
#    int i, ii, n = 0, j = 1, j2 = 0, j3 = 0;
#    GMT_LONG npoints;
#    int lineout = 1;
#    int rgb1[3], rgb2[3];
#    int big_iso = 0;
#
#    double a[3], p[3], v[3];
#    double vi, iso, f;
#    double fir, s2alphan, alphan;
#    double cfi, sfi, can, san;
#    double cpd, spd, cpb, spb, cpm, spm;
#    double cad, sad, cab, sab, cam, sam;
#    double xz, xn, xe;
#    double az = 0., azp = 0., takeoff, r;
#    double azi[3][2];
#    double x[400], y[400], x2[400], y2[400], x3[400], y3[400];
#    double xp1[800], yp1[800], xp2[400], yp2[400];
#    double radius_size;
#    double si, co;
#
#    a[0] = T.str; a[1] = N.str; a[2] = P.str;
#    p[0] = T.dip; p[1] = N.dip; p[2] = P.dip;
#    v[0] = T.val; v[1] = N.val; v[2] = P.val;
#
#    vi = (v[0] + v[1] + v[2]) / 3.;
#    for (i=0; i<=2; i++) v[i] = v[i] - vi;
#
#    radius_size = size * 0.5;
#
#    if (fabs(squared(v[0]) + squared(v[1]) + squared(v[2])) < EPSIL) {
#        /* pure implosion-explosion */
#        if (vi > 0.) {
#            ps_circle(x0, y0, radius_size*2., c_rgb, lineout);
#        }
#        if (vi < 0.) {
#            ps_circle(x0, y0, radius_size*2., e_rgb, lineout);
#        }
#        return(radius_size*2.);
#    }
#
#    if (fabs(v[0]) >= fabs(v[2])) {
#        d = 0;
#        m = 2;
#    }
#    else {
#        d = 2;
#        m = 0;
#    }
#
#    if (plot_zerotrace) vi = 0.;
#
#    f = - v[1] / v[d];
#    iso = vi / v[d];
#
#    # Cliff Frohlich, Seismological Research letters,
#    # Vol 7, Number 1, January-February, 1996
#    # Unless the isotropic parameter lies in the range
#    # between -1 and 1 - f there will be no nodes whatsoever
#
#    if (iso < -1) {
#        ps_circle(x0, y0, radius_size*2., e_rgb, lineout);
#        return(radius_size*2.);
#    }
#    else if (iso > 1-f) {
#        ps_circle(x0, y0, radius_size*2., c_rgb, lineout);
#        return(radius_size*2.);
#    }
#
#    sincosd (p[d], &spd, &cpd);
#    sincosd (p[b], &spb, &cpb);
#    sincosd (p[m], &spm, &cpm);
#    sincosd (a[d], &sad, &cad);
#    sincosd (a[b], &sab, &cab);
#    sincosd (a[m], &sam, &cam);
#
#    for (i=0; i<360; i++) {
#        fir = (double) i * D2R;
#        s2alphan = (2. + 2. * iso) / (3. + (1. - 2. * f) * cos(2. * fir));
#        if (s2alphan > 1.) big_iso++;
#        else {
#            alphan = asin(sqrt(s2alphan));
#            sincos (fir, &sfi, &cfi);
#            sincos (alphan, &san, &can);
#            
#            xz = can * spd + san * sfi * spb + san * cfi * spm;
#            xn = can * cpd * cad + san * sfi * cpb * cab + san * cfi * cpm * cam;
#            xe = can * cpd * sad + san * sfi * cpb * sab + san * cfi * cpm * sam;
#            
#            if (fabs(xn) < EPSIL && fabs(xe) < EPSIL) {
#                takeoff = 0.;
#                az = 0.;
#            }
#            else {
#                az = atan2(xe, xn);
#                if (az < 0.) az += M_PI * 2.;
#                takeoff = acos(xz / sqrt(xz * xz + xn * xn + xe * xe));
#            }
#            if (takeoff > M_PI_2) {
#                takeoff = M_PI - takeoff;
#                az += M_PI;
#                if (az > M_PI * 2.) az -= M_PI * 2.;
#            }
#            r = M_SQRT2 * sin(takeoff / 2.);
#            sincos (az, &si, &co);
#            if (i == 0) {
#                azi[i][0] = az;
#                x[i] = x0 + radius_size * r * si;
#                y[i] = y0 + radius_size * r * co;
#                azp = az;
#            }
#            else {
#                if (fabs(fabs(az - azp) - M_PI) < D2R * 10.) {
#                        azi[n][1] = azp;
#                        azi[++n][0] = az;
#                }
#                if (fabs(fabs(az -azp) - M_PI * 2.) < D2R * 2.) {
#                        if (azp < az) azi[n][0] += M_PI * 2.;
#                        else azi[n][0] -= M_PI * 2.;
#                }
#                switch (n) {
#                        case 0 :
#                                x[j] = x0 + radius_size * r * si;
#                                y[j] = y0 + radius_size * r * co;
#                                j++;
#                                break;
#                        case 1 :
#                                x2[j2] = x0 + radius_size * r * si;
#                                y2[j2] = y0 + radius_size * r * co;
#                                j2++;
#                                break;
#                        case 2 :
#                                x3[j3] = x0 + radius_size * r * si;
#                                y3[j3] = y0 + radius_size * r * co;
#                                j3++;
#                                break;
#                }
#                azp = az;
#            }
#        }
#    }
#    azi[n][1] = az;
#    
#    if (v[1] < 0.) for (i=0;i<=2;i++) {rgb1[i] = c_rgb[i]; rgb2[i] = e_rgb[i];}
#    else for (i=0;i<=2;i++) {rgb1[i] = e_rgb[i]; rgb2[i] = c_rgb[i];}
#    
#    ps_circle(x0, y0, radius_size*2., rgb2, lineout);
#    switch(n) {
#        case 0 :
#            for (i=0; i<360; i++) {
#                xp1[i] = x[i]; yp1[i] = y[i];
#            }
#            npoints = i;
#            ps_polygon(xp1, yp1, npoints, rgb1, outline);
#            break;
#        case 1 :
#            for (i=0; i<j; i++) {
#                xp1[i] = x[i]; yp1[i] = y[i];
#            }
#            if (azi[0][0] - azi[0][1] > M_PI) azi[0][0] -= M_PI * 2.;
#            else if (azi[0][1] - azi[0][0] > M_PI) azi[0][0] += M_PI * 2.;
#            if (azi[0][0] < azi[0][1])
#                for (az = azi[0][1] - D2R; az > azi[0][0]; az -= D2R) {
#                    sincos (az, &si, &co);
#                    xp1[i] = x0 + radius_size * si;
#                    yp1[i++] = y0 + radius_size * co;
#                }
#            else
#                for (az = azi[0][1] + D2R; az < azi[0][0]; az += D2R) {
#                    sincos (az, &si, &co);
#                    xp1[i] = x0 + radius_size * si;
#                    yp1[i++] = y0 + radius_size * co;
#                }
#            npoints = i;
#            ps_polygon(xp1, yp1, npoints, rgb1, outline);
#            for (i=0; i<j2; i++) {
#                xp2[i] = x2[i]; yp2[i] = y2[i];
#            }
#            if (azi[1][0] - azi[1][1] > M_PI) azi[1][0] -= M_PI * 2.;
#            else if (azi[1][1] - azi[1][0] > M_PI) azi[1][0] += M_PI * 2.;
#            if (azi[1][0] < azi[1][1])
#                for (az = azi[1][1] - D2R; az > azi[1][0]; az -= D2R) {
#                    sincos (az, &si, &co);
#                    xp2[i] = x0 + radius_size * si;
#                    yp2[i++] = y0 + radius_size * co;
#                }
#            else
#                for (az = azi[1][1] + D2R; az < azi[1][0]; az += D2R) {
#                    sincos (az, &si, &co);
#                    xp2[i] = x0 + radius_size * si;
#                    yp2[i++] = y0 + radius_size * co;
#                }
#            npoints = i;
#            ps_polygon(xp2, yp2, npoints, rgb1, outline);
#            break;
#        case 2 :
#            for (i=0; i<j3; i++) {
#                xp1[i] = x3[i]; yp1[i] = y3[i];
#            }
#            for (ii=0; ii<j; ii++) {
#                xp1[i] = x[ii]; yp1[i++] = y[ii];
#            }
#            
#            if (big_iso) {
#                for (ii=j2-1; ii>=0; ii--) {
#                    xp1[i] = x2[ii]; yp1[i++] = y2[ii];
#                }
#                npoints = i;
#                ps_polygon(xp1, yp1, npoints, rgb1, outline);
#                break;
#            }
#
#            if (azi[2][0] - azi[0][1] > M_PI) azi[2][0] -= M_PI * 2.;
#            else if (azi[0][1] - azi[2][0] > M_PI) azi[2][0] += M_PI * 2.;
#            if (azi[2][0] < azi[0][1])
#                for (az = azi[0][1] - D2R; az > azi[2][0]; az -= D2R) {
#                    sincos (az, &si, &co);
#                    xp1[i] = x0+ radius_size * si;
#                    yp1[i++] = y0+ radius_size * co;
#                }
#            else
#                for (az = azi[0][1] + D2R; az < azi[2][0]; az += D2R) {
#                    sincos (az, &si, &co);
#                    xp1[i] = x0+ radius_size * si;
#                    yp1[i++] = y0+ radius_size * co;
#                }
#            npoints = i;
#            ps_polygon(xp1, yp1, npoints, rgb1, outline);
#            for (i=0; i<j2; i++) {
#                xp2[i] = x2[i]; yp2[i] = y2[i];
#            }
#            if (azi[1][0] - azi[1][1] > M_PI) azi[1][0] -= M_PI * 2.;
#            else if (azi[1][1] - azi[1][0] > M_PI) azi[1][0] += M_PI * 2.;
#            if (azi[1][0] < azi[1][1])
#                for (az = azi[1][1] - D2R; az > azi[1][0]; az -= D2R) {
#                    sincos (az, &si, &co);
#                    xp2[i] = x0+ radius_size * si;
#                    yp2[i++] = y0+ radius_size * co;
#                }
#            else
#                for (az = azi[1][1] + D2R; az < azi[1][0]; az += D2R) {
#                    sincos (az, &si, &co);
#                    xp2[i] = x0+ radius_size * si;
#                    yp2[i++] = y0+ radius_size * co;
#                }
#            npoints = i;
#            ps_polygon(xp2, yp2, npoints, rgb1, outline);
#            break;
#    }
#    return(radius_size*2.);
#}

#Beachball2([1,-1,0,0,0,-1])