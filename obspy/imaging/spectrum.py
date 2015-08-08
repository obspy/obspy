# -*- coding: utf-8 -*-

"""
Plotting the spectrum of seismograms
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft

def plot_spectrum(data, samp_rate, outfile=None, fmt=None,
             axes=None, title=None, show=True):

    """
     Compute and plot the spectrum of the input data
     """

    # enforce float for samp_rate
    samp_rate = float(samp_rate)

    npts = len(data)

    # Compute the FFT
    npts = len(data)
    k = np.arange(npts)
    T = npts/samp_rate
    frq = k/T  # two sides frequency range
    frq = frq[range(npts/2)]  # one side frequency range

    X = fft(data)/npts  # fft computing and normalization
    X = X[range(npts/2)]

    if not axes:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = axes

    ax.loglog(frq, abs(X), color = 'k') # plotting the spectrum
    ax.set_xlabel('Freq (Hz)')
    ax.set_ylabel('|Y(freq)|')

    # set correct way of axis, whitespace before and after with window
    # length
    ax.axis('tight')
    ax.set_xlim(0, frq[-1])
    ax.grid(False)

    if axes:
        return ax

    if title:
        ax.set_title(title)

    if outfile:
        if fmt:
            fig.savefig(outfile, format=fmt)
        else:
            fig.savefig(outfile)
    elif show:
        plt.show()
    else:
        return fig
