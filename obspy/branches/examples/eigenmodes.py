# -*- coding: utf-8 -*-
# 2009-11-27 Moritz

from obspy.core import read
import numpy as np
from obspy.signal import lowpass
import matplotlib.pyplot as plt
from matplotlib.mlab import detrend

def stringToBool(s):
    """
    Special function for parsing S and T modes of eiglst
    """
    if s == 'S':
        return True
    return False

# load data
tr = read("china.mseed")[0]
df, npts = (tr.stats.sampling_rate, tr.stats.npts)
tr.data = tr.data.astype('float64') #convert to double

# lowpass at 30s and downsample to 10s
f0 = 1.0/50
tr.data = lowpass(tr.data, f0, df=df, corners=2)
tr.data = tr.data[0::10] #resample at 10Hz
df, npts = (.1, len(tr.data)) #redefine df and npts

# do the fourier transformation
#data = np.loadtxt("china8b.asc",usecols=[0], dtype='float64')
#tr.data -= tr.data.mean()
tr.data = detrend(tr.data, 'linear')
tr.data *= np.hanning(npts)
df = 0.1
fdat = np.fft.rfft(tr.data, n=4*npts) #smooty by pading with zeros
fdat /= abs(fdat).max() #normalize to 1

# get the eigenmodes
eigen = np.loadtxt("eiglst", usecols=[0,1,2,3], converters={1:stringToBool})
# only the S part
ind1 = eigen[:,1].astype(bool)
ind2 = eigen[:,0]
ind = ((ind2 == 0) & ind1) #bitwise comparing for bool arrays
modes = eigen[ind,3]/1000  #normalize, freq given in mHz

# plot the first N points only
N = 4000
freq = np.linspace(0,df/2,len(fdat))
freq = freq[1:N+1] #zero frequency is offset
fdat = fdat[1:N+1]
plt.clf()
plt.plot(freq,abs(fdat))
plt.vlines(modes[0:len(modes)/2],0,1)
plt.show()
