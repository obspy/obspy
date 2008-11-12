# obspy_class module
#---------------------------------------------------
# Filename: obspy_class.py
#  Purpose: Python module for gse conversion...
#  Version: n.a.
#   Author: Moritz Beyreuther
#    Email: moritz.beyreuther@geophysik.uni-muenchen.de
# Revision: 2008/03/23 Moritz
#           2008/08/15 Moritz starting OOP Design
#---------------------------------------------------

import os, sys
from copy import deepcopy

#
# Importing array and signal processing support
from numpy import array, arange, zeros, ones, concatenate
from numpy import pi, ndarray, float64
from scipy.signal import iirfilter,lfilter

#
# Importing the plotting routines
import pylab
from matplotlib.ticker import FuncFormatter

#
# Importing wraped C libaries/functions
import ext_gse
from ext_recstalta import rec_stalta
from ext_pk_mbaer import baerPick
from ext_arpicker import arPick

class Seismogram:

  def readgse(self,gsefile):
    """NOTE: documentation is assigned AFTER definition by:
    read.__doc__ = gse_ext.read.__doc__"""
    try:
      os.path.exists(gsefile)
      (self.header,self.data) = ext_gse.read(gsefile)
    except IOError:
      print "No such file to write: " + gsefile
      sys.exit(2)
    #
    # define the header entries as attributes
    self.auxid = self.header['auxid']
    self.calib = self.header['calib']
    self.calper = self.header['calper']
    self.channel = self.header['channel']
    self.d_day = self.header['d_day']
    self.d_mon = self.header['d_mon']
    self.d_year = self.header['d_year']
    self.datatype = self.header['datatype']
    self.hang = self.header['hang']
    self.instype = self.header['instype']
    self.n_samps = self.header['n_samps']
    self.samp_rate = self.header['samp_rate']
    self.station = self.header['station']
    self.t_hour = self.header['t_hour']
    self.t_min = self.header['t_min']
    self.t_sec = self.header['t_sec']
    self.vang = self.header['vang']
    return 0
  readgse.__doc__ = ext_gse.read.__doc__

  def isattr(self,attr,typ,default,length=False,assertation=False):
    """function for verifying that Seismogram has attribute of cetain type
    and, if given, certain length. If not, the instance is set to the given
    default """
    if not attr in self.__dict__.keys():
      if assertation:
        assert False,"%s attribute of Seismogram required" % attr
      print "WARNING: %s attribute of Seismogram missing",
      print "forcing",attr,"=",default
      setattr(self,attr,default)
    if not isinstance(getattr(self,attr),typ):
      print "WARNING: %s attribute of Seismogram not of type %s" % (attr,typ),
      print "forcing",attr,"=",default
      setattr(self,attr,default)
    if (length):
      if (len(getattr(self,attr)) > length):
        print "%s attribute of Seismogram is > %i" % (attribute,length)
        print "forcing",attribute,"=",default
        attribute=default
    return True

  def plot(self):
    self.isattr('data',ndarray,None,assertation=True)
    self.isattr('n_samps',int,len(self.data))
    self.isattr('samp_rate',float,1.0)
    self.isattr('d_year',int,-1)
    self.isattr('d_mon',int,-1)
    self.isattr('d_day',int,-1)
    self.isattr('t_hour',int,-1)
    self.isattr('t_min',int,-1)
    self.isattr('t_sec',float,-1.0)
    self.isattr('station',str,'XXXXX',length=6)
    #
    if not 'time' in self.__dict__.keys():
      self.time = arange(self.n_samps) / self.samp_rate
    else: 
      print """NOTE:    Using existing time attribute of Seismogram, if
this is not wished first delete attribute (delattr) time"""
    #
    def format(x,pos):
      # x is of type numpy.float64, the string representation of that float
      # strips of all tailing zeros pos returns the position of x on the
      # axis while zooming, None otherwise
      min = float64(x/60)
      sec = float64((min - int(min)) * 60)
      if ( min-int(min) > 0.1 ):
        return "%02d:%05.2f" % (int(min),sec)
      else:
        return str(int(min))
    #
    pylab.figure()
    formatter = FuncFormatter(format)
    ax = pylab.axes()
    ax.xaxis.set_major_formatter(formatter)
    pylab.plot(self.time,self.data)
    pylab.xlabel('Time elapsed since Starttime [min [:sec]]')
    pylab.title('%s Starttime: %04d/%02d/%02d %02d:%02d:%05.2f' % (
        self.station,
        self.d_year,self.d_mon,self.d_day,
        self.t_hour,self.t_min,self.t_sec)
    )
    pylab.show()

  def bandpass(self,freqmin,freqmax,corners=4):
    self.isattr('data',ndarray,None,assertation=True)
    self.isattr('samp_rate',float,1.)
    """Butterworth-Bandpass: filter self.data from freqmin to freqmax using
    corners corners"""
    nyf=.5*self.samp_rate # Nyquist Frequencey
    if freqmax > nyf:
      print 'Warning freqmax greater than Nyquist frequency. Forcing\
          freqmax to Nyquist frequency'
      freqmax = nyf
    [b,a]=iirfilter(corners, [freqmin/nyf, freqmax/nyf], btype='band',ftype='butter',output='ba')
    return lfilter(b, a, self.data)



  def writegse(self,gsefile):
    """write seismogram to gsefile, the necessary header entries are
    given in documentation of the extern C function appended after this
    documentation. Defaults are set automatically
  
    write(header,data,gsefile)
    h            : tuple containing the header variables
    data         : LONG array containing the data to write
    gsefile      : target file to write
    ----------------------------------------------------------------------
    """
    # 
    # function for testing correctness of header entries
    #
    # check if header has the necessary tuples and if those are of
    # correct type
    self.isattr('data',ndarray,None,assertation=True)
    self.isattr('d_year',int,2007)
    self.isattr('d_mon',int,05)
    self.isattr('d_day',int,27)
    self.isattr('t_hour',int,23)
    self.isattr('t_min',int,59)
    self.isattr('t_sec',float,24.123)
    self.isattr('station',str,'STAU ',length=6)
    self.isattr('channel',str,'SHZ',length=4)
    self.isattr('auxid',str,'VEL ',length=5)
    self.isattr('datatype',str,'CM6 ',length=4)
    self.isattr('n_samps',int,len(self.data))
    self.isattr('samp_rate',float,200.)
    self.isattr('calib',float,1./(2*pi)) #calper not correct in gse_driver!
    self.isattr('calper',float,1.)
    self.isattr('instype',str,'LE-3D ',length=7)
    self.isattr('hang',float,-1.0)
    self.isattr('vang',float,0.)
  
    # I have errors with the data pointer, only solution seems to explicitly copy it
    data2 = self.data.copy()
    err = ext_gse.write((self.d_year, self.d_mon, self.d_day, self.t_hour,
      self.t_min, self.t_sec, self.station, self.channel, self.auxid,
      self.datatype, self.n_samps, self.samp_rate, self.calib, self.calper,
      self.instype, self.hang, self.vang), data2, gsefile)
    del data2
    return err
  writegse.__doc__ = writegse.__doc__ + ext_gse.write.__doc__

  def classicStaLta(self,Nsta,Nlta):
    """Computes the standard STA/LTA from a given imput array a. The length of
    the STA is given by Nsta in samples, respectively is the length of the
    LTA given by Nlta in samples.
    """
    self.isattr('data',ndarray,None,assertation=True)
    m=len(self.data)
    stalta=zeros(m,dtype=float)
    start = 0
    stop = 0
    #
    # compute the short time average (STA)
    sta=zeros(m,dtype=float)
    pad_sta=zeros(Nsta)
    for i in range(Nsta): # window size to smooth over
      sta=sta+concatenate((pad_sta,self.data[i:m-Nsta+i]**2))
    sta=sta/Nsta
    #
    # compute the long time average (LTA)
    lta=zeros(m,dtype=float)
    pad_lta=ones(Nlta) # avoid for 0 division 0/1=0
    for i in range(Nlta): # window size to smooth over
      lta=lta+concatenate((pad_lta,self.data[i:m-Nlta+i]**2))
    lta=lta/Nlta
    #
    # pad zeros of length Nlta to avoid overfit and
    # return STA/LTA ratio
    sta[0:Nlta]=0
    return sta/lta

  def delayedStaLta(self,Nsta,Nlta):
    """Delayed STA/LTA, (see Withers et al. 1998 p. 97)
    This functions returns the characteristic function of the delayes STA/LTA
    trigger. Nsta/Nlta is the length of the STA/LTA window in points
    respectively"""
    self.isattr('data',ndarray,None,assertation=True)
    m=len(self.data)
    stalta=zeros(m,dtype=float)
    on = 0;
    start = 0;
    stop = 0;
    #
    # compute the short time average (STA) and long time average (LTA)
    # don't start for STA at Nsta because it's muted later anyway
    sta=zeros(m,dtype=float)
    lta=zeros(m,dtype=float)
    for i in range(Nlta+Nsta+1,m):
      sta[i]=(self.data[i]**2 + self.data[i-Nsta]**2)/Nsta + sta[i-1]
      lta[i]=(self.data[i-Nsta-1]**2 + self.data[i-Nsta-Nlta-1]**2)/Nlta + lta[i-1]
      sta[0:Nlta+Nsta+50]=0
    return sta/lta

  def recursiveStaLta(self,Nsta,Nlta):
    """Recursive STA/LTA (see Withers et al. 1998 p. 98)
    NOTE: There exists a version of this trigger wrapped in C called
    rec_stalta in this module!"""
    self.isattr('data',ndarray,None,assertation=True)
    m=len(self.data)
    #
    # compute the short time average (STA) and long time average (LTA)
    # given by Evans and Allen
    sta=zeros(m,dtype=float)
    lta=zeros(m,dtype=float)
    #Csta = 1-exp(-S/Nsta); Clta = 1-exp(-S/Nlta)
    Csta = 1./Nsta; Clta = 1./Nlta
    for i in range(1,m):
      # THERE IS A SQUARED MISSING IN THE FORMULA, I ADDED IT
      sta[i]=Csta*self.data[i]**2 + (1-Csta)*sta[i-1]
      lta[i]=Clta*self.data[i]**2 + (1-Clta)*lta[i-1]
    sta[0:Nlta]=0
    return sta/lta

  def zdetect(self,Nsta,number_dummy):
    """Z-detector, (see Withers et al. 1998 p. 99)
    This functions returns the characteristic function of the Z-detector.
    Nsta gives the number of points for the sta window"""
    self.isattr('data',ndarray,None,assertation=True)
    m=len(self.data)
    #
    # Z-detector given by Swindell and Snell (1977)
    Z=zeros(m,dtype=float)
    sta=zeros(m,dtype=float)
    # Standard Sta
    pad_sta=zeros(Nsta)
    for i in range(Nsta): # window size to smooth over
      sta=sta+concatenate((pad_sta,self.data[i:m-Nsta+i]**2))
    a_mean=sta.mean()
    a_std=sta.std()
    return (sta-a_mean)/a_std
  
  def copy(self):
    return deepcopy(self)
