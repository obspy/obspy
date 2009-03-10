# do this before importing pylab or pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from numpy import array, linalg, zeros, mean, sqrt, fabs, arcsin, arccos, sin, \
    pi, cos, arctan2, power, abs, sum 
from pylab import plot, axis, getp, setp, gca


def Beachball(fm, centerX=0, centerY=0, diam=0, ta=0, color='b'):
    """
    Draws beachball diagram of earthquake double-couple focal mechanism(s). S1, D1, and
        R1, the strike, dip and rake of one of the focal planes, can be vectors of
        multiple focal mechanisms.
    fm - focal mechanism that is either number of mechnisms (NM) by 3 (strike, dip, and rake)
        or NM x 6 (mxx, myy, mzz, mxy, mxz, myz - the six independent components of
        the moment tensor). The strike is of the first plane, clockwise relative to north.
        The dip is of the first plane, defined clockwise and perpedicular to strike, 
        relative to horizontal such that 0 is horizontal and 90 is vertical. The rake is 
        of the first focal plane solution. 90 moves the hanging wall up-dip (thrust),
        0 moves it in the strike direction (left-lateral), -90 moves it down-dip
        (normal), and 180 moves it opposite to strike (right-lateral).
    centerX - place beachball(s) at position centerX
    centerY - place beachball(s) at position centerY
    diam - draw with this diameter.  If diam is zero, beachball
        is drawn on stereonet.
    ta - type of axis. If 0, this is a normal axis, if 1 it is a map axis. In case
        of the latter, centerX and centerY are Lons and Lats, respectively.
    color - color to use for quadrants of tension; can be a string, e.g. 'r'
        'b' or three component color vector, [R G B].
    """
    n = len(fm)
    
    if n == 6:
        (s1, d1, r1) = Mij2SDR(fm[0], fm[1], fm[2], fm[3], fm[4], fm[5])
    elif n == 3:
        s1 = fm[0]
        d1 = fm[1]
        r1 = fm[2]
    else:
        raise TypeError("Wrong input value for 'fm'.")
    
    r2d = 180/pi
    d2r = pi/180
    ampy = cos(mean(centerY)*d2r)
    
    mech = 0
    if r1 > 180:
        r1 = r1 - 180
        mech = 1
    if r1 < 0:
        r1 = r1 + 180
        mech = 1
    
    # Get azimuth and dip of second plane
    (s2, d2, r2) = AuxPlane(s1, d1, r1)
    
    S1 = s1
    D1 = d1
    S2 = s2
    D2 = d2
    P = r1
    CX = centerX
    CY = centerY
    D = diam
    M = mech
    
    if M > 0:
        P = 2
    else:
        P = 1
    
    if D1 >= 90:
       D1 = 89.9999
    if D2 >= 90:
       D2 = 89.9999
       
    pi_315 = pi/315.
    
    phi = array([pi_315*i for i in xrange(316)])
    d = 90 - D1
    m = 90
    
    l1 = sqrt(power(d,2)/(power(sin(phi),2) + power(cos(phi),2) * power(d,2)/power(m,2)))
    
    d = 90 - D2
    m = 90
    l2 = sqrt(power(d,2)/(power(sin(phi),2) + power(cos(phi),2) * power(d,2)/power(m,2)))
    
    if D == 0:
       Stereo(phi+S1*d2r, l1, 'k')
#       hold on
       Stereo(phi+S2*d2r, l2, 'k')
#    
#    inc = 1;
#    [X1,Y1] = pol2cart(phi+S1*d2r,l1);
#    if P == 1
#       lo = S1 - 180;
#       hi = S2;
#       if lo > hi
#          inc = -inc;
#       end
#       th1 = S1-180:inc:S2;
#       [Xs1,Ys1] = pol2cart(th1*d2r,90*ones(1,length(th1)));
#       [X2,Y2] = pol2cart(phi+S2*d2r,l2);
#       th2 = S2+180:-inc:S1;
#    else
#       hi = S1 - 180;
#       lo = S2 - 180;
#       if lo > hi
#          inc = -inc;
#       end
#       th1 = hi:-inc:lo;
#       [Xs1,Ys1] = pol2cart(th1*d2r,90*ones(1,length(th1)));
#       [X2,Y2] = pol2cart(phi+S2*d2r,l2);
#       X2 = fliplr(X2);
#       Y2 = fliplr(Y2);
#       th2 = S2:inc:S1;
#    end
#    [Xs2,Ys2] = pol2cart(th2*d2r,90*ones(1,length(th2)));
#    
#    X = cat(2,X1,Xs1,X2,Xs2);
#    Y = cat(2,Y1,Ys1,Y2,Ys2);
#    
#    if ta == 0
#       fill(X,Y,color)
#    else
#       fillm(Y,X,color)
#    end
#    view(90,-90)


def Stereo(theta, rho, line_style):
    """
    """
    fig = plt.figure()
    cax = fig.add_subplot(111, aspect='equal')
    # make a radial grid
    maxrho = max(abs(rho))
    hhh=cax.plot([-maxrho,-maxrho,maxrho,maxrho],[-maxrho,maxrho,maxrho,-maxrho])
    setp(gca(),'xlim',[-90, 90])
    setp(gca(),'ylim',[-90, 90])
    v = [-90,90,-90,90]
    #ticks = sum([v for v in getp(cax, 'ytick') if v>=0])
    del(hhh)
    
    # check radial limits and ticks
    rmin = 0
    rmax = v[3]
    #rticks = max(ticks-1, 2)
#    %    if rticks > 5   % see if we can reduce the number
#    %        if rem(rticks,2) == 0
#    %            rticks = rticks/2;
#    %        elseif rem(rticks,3) == 0
#    %            rticks = rticks/3;
#    %        end
#    %    end
    rticks = 5
#    
    # define a circle
    step = pi/50.0
    th = array([i*step for i in xrange(50)])
    xunit = cos(th)
    yunit = sin(th)
    # now really force points on x/y axes to lie on them exactly
    step = (len(th)-1)/4
    inds = array([i*step for i in xrange(len(th))])
    xunit[inds[1:1:3]] = zeros((2,1))
    yunit[inds[0:1:4]] = zeros((3,1))
    # plot background if necessary
    cax.Patch('xdata',xunit*rmax,'ydata',yunit*rmax, 
          edgecolor=[0,0,0],facecolor='b');
    # draw radial circles
    c82 = cos(82*pi/180)
    s82 = sin(82*pi/180)
    rinc = (rmax-rmin)/rticks
#        for i=(rmin+rinc):rinc:rmax
#            hhh = plot(xunit*i,yunit*i,ls,'color',tc,'linewidth',1,...
#                       'handlevisibility','off');
#        end
#        set(hhh,'linestyle','-') % Make outer circle solid
#    
#    % plot spokes
#        th = (1:6)*2*pi/12;
#        cst = cos(th); snt = sin(th);
#    
    # set view to 2-D
    #view(2)
    # set axis limits
    #axis(rmax*[-1, 1, -1.15, 1.15])
#    end
#    
#    % Reset defaults.
#    set(cax, 'DefaultTextFontAngle', fAngle , ...
#        'DefaultTextFontName',   fName , ...
#        'DefaultTextFontSize',   fSize, ...
#        'DefaultTextFontWeight', fWeight, ...
#        'DefaultTextUnits',fUnits );
#    
#    % transform data to Cartesian coordinates.
#    xx = rho.*cos(theta);
#    yy = rho.*sin(theta);
#    
#    % plot data on top of grid
#    if strcmp(line_style,'auto')
#        q = plot(xx,yy);
#    else
#        q = plot(xx,yy,line_style);
#    end
#    if nargout > 0
#        hpol = q;
#    end
#    if ~hold_state
#        set(gca,'dataaspectratio',[1 1 1]), axis off; set(cax,'NextPlot',next);
#    end
#    set(get(gca,'xlabel'),'visible','on')
#    set(get(gca,'ylabel'),'visible','on')
    fig.savefig('test.png')


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
    @param mij: - siz independent components of the moment tensor
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
        FT=asin(AXN)*CON
        ST=-XN
        CT=YN
        if ST >= 0. and CT < 0:
            FT=180.-FT
        if ST < 0. and CT <= 0:
            FT=180.+FT
        if ST < 0. and CT > 0:
            FT=360.-FT
        FL=asin(abs(ZE))*CON
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