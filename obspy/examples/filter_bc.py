# -*- coding: utf-8 -*-
# 2009-11-27 Moritz

import numpy as np

def filter_bc(x, dt, f0):
    """
    The simple-most low-pass/boxcar filter

    @param x: Time series
    @param dt: Sampling intervall
    @param f0: Frequency of lowpass
    @return: (freq, H, x_filt) with x_f frequency range, 
             H consturcted filter, x_filt filtered time series
    """
    npts = len(x)
    
    # Initialize
    nyq = 1.0/(2*dt) # Nyquist frequency
    freq = np.linspace(0, nyq, npts//2+1) #// == integer devision

    # Construct the Boxcar
    # generate array with 0 everywhere but 1 where freq < f0
    H = np.where(freq<f0, 1.0, 0.0)

    # Filter in the Fourier Domain
    x_f = np.fft.rfft(x)
    return freq, H, np.real(np.fft.irfft(x_f*H))
