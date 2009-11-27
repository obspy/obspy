# -*- coding: utf-8 -*-
# 2009-11-27 Moritz

import numpy as np
import matplotlib
matplotlib.rc('figure.subplot', hspace=.35, wspace=.35) #adjust subplot layout
matplotlib.rc('font', size=8) # adjust font size of plot
import matplotlib.pyplot as plt
from obspy.signal import lowpass, highpass, bandpass
from scipy.io import loadmat

#
# f-domain representation of Butterworth filter
#
f = np.arange(0,100,dtype='float64')
f0 = 20 #cut off frequency f0
fs = 15

plt.figure(1)
for i in xrange(1,5):
    plt.subplot(2,2,i)
    # n=1
    n=i**2
    Fb = 1/( 1+(f/f0)**(2*n))
    plt.plot(f,Fb)
    plt.title('Butterworth n=%i, f0=%i Hz'% (n,f0))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Filter amplitude')


#
# effect on a spike
#
N = 1000
dt = .005;
t = np.linspace(dt,N*dt,N)
s = np.zeros(N)
s[99] = 1

plt.figure(2)
plt.subplot(2,2,1)
plt.plot(t,s)
plt.title('Original function')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')


for i in xrange(2,5):
    n=(i-1)**2
    plt.subplot(2,2,i )
    Fs=lowpass(s,f0,1./dt,n);
    plt.plot(t,Fs)
    plt.title('Filtered with n=%i, f0=%i Hz'% (n,f0))
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.axis([0,1,min(Fs),max(Fs)])


#
# effect on a seismogram
#

# load Matlab mat file into Python
mat = loadmat("germany.mat")

# be sure it is a 1 DIM array => ravel()
t=mat['translation_time'].ravel()
dt=.05
f0=1
s=mat['translation_Z'].ravel()

plt.figure(3)
plt.subplot(2,2,1)
plt.plot(t,s)
plt.title('Original function')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
t1=40
t2=65
rmax= float(max(abs(s[np.arange(int(t1/dt),int(t2/dt))])))
plt.axis([40, 65, -rmax, rmax])

for i in xrange(2,5):

    n=(i)**2
    plt.subplot(2,2,i )
    Fs=lowpass(s,f0,1./dt,n);
    plt.plot(t, Fs)
    plt.title('Filtered with n=%i, f0=%i Hz'% (n,f0))
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    t1=40
    t2=65
    rmax= float(max(abs(s[np.arange(int(t1/dt),int(t2/dt))])))
    plt.axis([ 40, 65, -rmax, rmax ])

plt.figure(4)

plt.subplot(2,2,1)
plt.plot(t,s)
plt.title('Original function')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
t1=40
t2=65
rmax= float(max(abs(s[np.arange(int(t1/dt),int(t2/dt))])))
plt.axis([ 40,65, -rmax,rmax ])

for i in xrange(2,5):
    n=4
    f0=2.0/i
    plt.subplot(2,2,i)
    Fs=lowpass(s,f0,1./dt,n);
    plt.plot(t,Fs)
    plt.title('Filtered with n=%i, f0=%g Hz'% (n,f0))
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    t1=40
    t2=65
    rmax= float(max( abs(Fs[np.arange(int(t1/dt),int(t2/dt))])))
    plt.axis([ 40,65, -rmax,rmax ])

plt.show()
