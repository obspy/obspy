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
                  fig=None, ax=None, title=None, show=True, retspec=False):

    """
     Compute and plot the spectrum of the input data.

     :param data: Input data
     :type samp_rate: float
     :param samp_rate: Samplerate in Hz
     :type outfile: str
     :param outfile: String for the filename of output file, if None
        interactive plotting is activated.
     :type fmt: str
     :param fmt: Format of image to save
     :type fig: :class:`matplotlib.figure.Figure`
     :param ax: Plot into given figure
     :type ax: :class:`matplotlib.axes.Axes`
     :param ax: Plot into given axes
     :type title: str
     :param title: Set the plot title
     :type show: bool
     :param show: Do not call `plt.show()` at end of routine. That way, further
        modifications can be done to the figure before showing it.
     """

    # enforce float for samp_rate
    samp_rate = float(samp_rate)

    # Compute the FFT
    frq = rfftfreq(data.size, d=1./samp_rate)
    X = rfft(data)/samp_rate  # fft computing and normalization
    X = rfft(data)/(2*data.size)

    if fig is None:
        fig = plt.figure()

    if ax is None:
        ax = plt.subplot(111)

    print(np.shape(frq), np.shape(X))
    ax.loglog(frq, abs(X), color='k')
    ax.set_xlabel('Freq (Hz)')
    ax.set_ylabel('|Y(freq)|')

    ax.axis('tight')
    ax.set_xlim(0, frq[-1])
    ax.grid(False)

    if title:
        ax.set_title(title)

    if outfile:
        if fmt:
            fig.savefig(outfile, format=fmt)
        else:
            fig.savefig(outfile)

    if show:
        plt.show()

    if retspec:
        return fig, ax, frq, abs(X)

    else:
        return fig, ax
