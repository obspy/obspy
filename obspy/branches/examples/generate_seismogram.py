# -*- coding: utf-8 -*-
# 2009-11-27 Moritz

import numpy as np
import matplotlib
matplotlib.rc('figure.subplot', hspace=0.8) # set default
import matplotlib.pyplot as plt


def ricker(f, dt):
    """
    Generate Ricker Wavelet with central frequency f

    written by M.D.Sacchi, last modified December 10,  1998
    Copyright (C) 1998 Seismic Processing and Imaging Group
                       Department of Physics
                       The University of Alberta

    @param f: central frequency
    @param dt: sampling interval in [s]
    @return: Ricker wavelet as numpy.ndarray dtype float64
    """
    nw=6./f/dt
    nw=2*np.floor(nw/2)+1
    nc=np.floor(nw/2)
    i=np.arange(1,nw,dtype='float64')
    alpha=(nc-i+1)*f*dt*np.pi
    beta=alpha**2
    return (1.0-beta*2.0)*np.exp(-beta)


# Generate ricker wavelet
# Play with frequency (here 2)
wavelet = ricker(2, 0.05)

# Generate greensfunction
# Play with different coeficients
N = 2e2
greenfct = np.zeros(N,dtype='float64')
greenfct[np.array([48,50,54,60,65,76,110,140])] = 1
greenfct[np.array([49,52,57,61,63,70,90,120,170])] = -1

# Convolve Seismogram with ricker wavelet
seismogram = np.convolve(wavelet,greenfct)



#
# Plot all the results
#
plt.clf()
plt.subplot(311)
plt.plot(wavelet)
plt.title("Ricker Wavelet")
plt.subplot(312)
plt.stem(np.arange(N),greenfct,markerfmt='.',basefmt='b-')
plt.title("Greens Function")
plt.ylim(-1.5,1.5)
plt.subplot(313)
plt.plot(seismogram)
plt.title("Convolution of Ricker Wavelet with Greens Function")
plt.show()
