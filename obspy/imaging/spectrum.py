# -*- coding: utf-8 -*-

"""
Plotting the spectrum of seismograms
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq


def plot_spectrum(data, samp_rate, outfile=None, fmt=None,
                  axes=None, title=None, show=True):

    """
     Compute and plot the spectrum of the input data
     """

    # enforce float for samp_rate
    samp_rate = float(samp_rate)

    npts = len(data)

    # Compute the FFT
    frq = rfftfreq(data.size, d=1./samp_rate)
    X = rfft(data)/npts  # fft computing and normalization

    if not axes:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = axes

    print(np.shape(frq), np.shape(X))
    ax.loglog(frq, abs(X), color='k')
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
