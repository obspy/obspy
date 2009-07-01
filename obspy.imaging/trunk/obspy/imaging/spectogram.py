# -*- coding: utf-8 -*-

import obspy, time, glob, os
import pylab as pl
import numpy as N
import math as M
from matplotlib import mlab

def nearestPow2(x):
    """
    Find power of two nearest to x

    >>> nearestPow2(3)
    2
    >>> nearestPow2(15)
    4

    @type x: Float
    @param x: Number
    @rtype: Int
    @return: Nearest power of 2 to x
    """
    a = M.pow(2,M.ceil(N.log2(x)))
    b = M.pow(2,M.floor(N.log2(x)))
    if abs(a-x) < abs(b-x):
        return a
    else: 
        return b

def spectoGram(data,samp_rate=100.0,log=False,outfile=None,format=None):
    """
    Computes and plots logarithmic spectogram of the input trace.
    
    @type data: Numpy ndarray
    @param data: Input data
    @type sample_rate: Float
    @param sample_rate: Samplerate in Hz
    @type log: Bool
    @param log: True logarithmic frequency axis, False linear frequency axis
    @type outfile: String
    @param outfile: String for the filename of output file, if None
        interactive plotting is activated.
    @type format: String
    @param format: Format of image to save
    """
    pl.ioff()

    samp_int = 1/float(samp_rate)
    npts = len(data)
    nfft = nearestPow2(npts/10.0)
    if nfft > 4096:
        nfft = 4096
    nlap = int(nfft*.8)

    data = data-data.mean()

    # here we call not pl.specgram as this already produces a plot
    # matplotlib.mlab.specgram should be faster as it computes only the
    # arrays
    # XXX mlab.specgram uses fft, would be better and faster use rfft
    spectogram, freq, time = mlab.specgram(data,Fs=samp_rate,
                                           NFFT=nfft,noverlap=nlap)
    # db scale and remove offset
    spectogram = 10*N.log10(spectogram[1:,:])
    freq = freq[1:]

    X,Y = N.meshgrid(time,freq)
    pl.pcolor(X,Y,spectogram)
    if log:
        pl.semilogy()
    pl.ylim((freq[0],freq[-1]))
    pl.grid(False)
    pl.xlabel('Time [s]')
    pl.ylabel('Frequency [Hz]')

    if outfile:
        if format:
            pl.savefig(outfile,format=format)
        else:
            pl.savefig(outfile)
    else:
        pl.show()

