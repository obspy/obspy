# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
# Filename: spectrogram.py
#  Purpose: Plotting spectrogram of Seismograms.
#   Author: Christian Sippl, Moritz Beyreuther
#    Email: sippl@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2010 Christian Sippl
#---------------------------------------------------------------------
"""
Plotting Spectrogram of Seismograms.


GNU General Public License (GPL)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
    
You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
02110-1301, USA.
"""


from matplotlib import mlab
import matplotlib.pyplot as plt
import math as M
import numpy as np


def nearestPow2(x):
    """
    Find power of two nearest to x

    >>> nearestPow2(3)
    2
    >>> nearestPow2(15)
    4

    :type x: Float
    :param x: Number
    :rtype: Int
    :return: Nearest power of 2 to x
    """
    a = M.pow(2, M.ceil(np.log2(x)))
    b = M.pow(2, M.floor(np.log2(x)))
    if abs(a - x) < abs(b - x):
        return a
    else:
        return b


def spectrogram(data, samp_rate=100.0, per_lap = .8, nwin = 10, log=False, 
                outfile=None, format=None, ax=None):
    """
    Computes and plots logarithmic spectrogram of the input trace.
    
    :param data: Input data
    :param sample_rate: Samplerate in Hz
    :param log: True logarithmic frequency axis, False linear frequency axis
    :param per_lap: Percent of overlap
    :param nwin: Approximate number of windows.
    :param outfile: String for the filename of output file, if None
        interactive plotting is activated.
    :param format: Format of image to save
    :param ax: Plot into given axis, this deactivates the format and
        outfile option
    """
    # enforce float for samp_rate
    samp_rate = float(samp_rate)

    npts = len(data)
    # nfft needs to be an integer, otherwise a deprecation will be raised
    nfft = int(nearestPow2(npts / float(nwin)))
    if nfft > 4096:
        nfft = 4096
    nlap = int(nfft * float(per_lap))

    data = data - data.mean()
    end = npts/samp_rate

    # here we call not plt.specgram as this already produces a plot
    # matplotlib.mlab.specgram should be faster as it computes only the
    # arrays
    # XXX mlab.specgram uses fft, would be better and faster use rfft
    spectrogram, freq, time = mlab.specgram(data, Fs=samp_rate,
                                            NFFT=nfft, noverlap=nlap)
    print time[0], time[-1]
    # db scale and remove zero/offset for amplitude
    spectrogram = 10 * np.log10(spectrogram[1:, :])
    freq = freq[1:]
    
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # calculate half bin width
    halfbin_time = (time[1] - time[0])/2.0
    halfbin_freq = (freq[1] - freq[0])/2.0

    if log:
        # pcolor expects one bin more at the right end
        freq = np.concatenate((freq, [freq[-1] + 2 * halfbin_freq]))
        time = np.concatenate((time, [time[-1] + 2 * halfbin_time]))
        # center bin
        time -= halfbin_time
        freq -= halfbin_freq
        X, Y = np.meshgrid(time, freq)
        ax.pcolor(X, Y, spectrogram)
        ax.semilogy()
        ax.set_ylim((freq[0], freq[-1]))
        ax.set_xlim(0, end)
    else:
        # this method is much much faster!
        spectrogram = np.flipud(spectrogram)
        # center bin
        extent = (time[0] - halfbin_time, time[-1] + halfbin_time, \
                  freq[0] - halfbin_freq, freq[-1] + halfbin_freq)
        ax.imshow(spectrogram, interpolation="nearest", extent=extent)
        ax.set_xlim(0, end)

    ax.grid(False)

    if not ax:
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [Hz]')
        fig.canvas.draw()

        if outfile:
            if format:
                fig.savefig(outfile, format=format)
            else:
                fig.savefig(outfile)
        else:
            plt.show()
